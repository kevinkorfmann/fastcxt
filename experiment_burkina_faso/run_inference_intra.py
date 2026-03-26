#!/usr/bin/env python3
"""Run pairwise TMRCA inference on Ag1000G data — 2La-heterozygous individuals only.

For each population, filters to 2La-heterozygous individuals (meta["2La"]==1),
then runs all pairwise haplotype combinations within that subset.
Loops over all populations, creating:
    out_dir/<population>/<arm>/   — inference results per chromosome

Usage:
    python run_inference.py --out-dir /sietch_colab/kkor/fastcxt/results/het_allpop
    python run_inference.py --out-dir ./results/test --populations "Burkina Faso" --arms 2L
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tskit

from fastcxt.config import FastCxtConfig, TrainingConfig
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_genotype_matrix


torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

ARMS = ["2L", "2R", "3L", "3R", "X"]
MUTATION_RATE = 3.5e-9
BLOCK_SIZE = 100_000  # 100 kb blocks (window_size=200 × n_windows=500)

DEFAULT_TREES_DIR = "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/tsinfer_data_v2"
DEFAULT_METADATA = "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/gamb.meta.tsinfer.csv"
DEFAULT_CHECKPOINT = "/sietch_colab/kkor/fastcxt/experiment_mosquito/experiment/outputs/pairwise/csv_metrics/version_1/checkpoints/epoch=29-step=20880.ckpt"
TREE_PATTERN = "gamb.{arm}.gff.dated.ne.trees"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_model(checkpoint_path: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt["state_dict"] = {
        k.replace("._orig_mod", ""): v for k, v in ckpt["state_dict"].items()
    }
    lit = LitFastCxt(**ckpt["hyper_parameters"])
    lit.load_state_dict(ckpt["state_dict"])
    model = lit.model.eval()
    cfg = ckpt["hyper_parameters"]["model_config"]
    log.info("Model: window_size=%d, n_windows=%d", cfg.window_size, cfg.n_windows)
    return model, cfg


def get_het_population_pairs(metadata_path, ts, population, max_pairs=None):
    """Get all pairwise haplotype pairs among 2La-heterozygous individuals.

    Returns original (full-ts) node IDs for haploid_pairs, plus a sorted list
    of all haploid nodes belonging to the filtered individuals (for simplify).
    """
    meta = pd.read_csv(metadata_path)
    pop = meta[(meta["country"] == population) & (meta["2La"].astype(int) == 1)].copy()
    if len(pop) == 0:
        raise ValueError(
            f"No 2La-het samples for '{population}'. "
            f"Available countries: {meta['country'].unique().tolist()}"
        )

    log.info("Population '%s': %d 2La-het individuals (%d haplotypes)",
             population, len(pop), len(pop) * 2)

    # Map sample IDs to tree-sequence individuals
    ts_individuals = {}
    for ind in ts.individuals():
        ind_meta = json.loads(ind.metadata.decode("ascii"))
        sid = ind_meta.get("sample_name", ind_meta.get("sample_id", ""))
        ts_individuals[sid] = ind

    # Collect het individuals present in tree sequence
    het_ids = []
    het_nodes = {}  # sample_id -> (node0, node1)
    for sid in pop["sample_id"].values:
        if sid in ts_individuals:
            nodes = ts_individuals[sid].nodes
            het_ids.append(sid)
            het_nodes[sid] = (int(nodes[0]), int(nodes[1]))

    # All pairwise combinations of individuals
    individual_pairs = list(combinations(het_ids, 2))

    if max_pairs is not None and len(individual_pairs) > max_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(individual_pairs), max_pairs, replace=False)
        individual_pairs = [individual_pairs[i] for i in sorted(idx)]
        log.info("Subsampled to %d pairs", max_pairs)

    # Haploid pairs using first haplotype of each individual (original node IDs)
    haploid_pairs = [
        (het_nodes[id_a][0], het_nodes[id_b][0])
        for id_a, id_b in individual_pairs
    ]

    # All haploid nodes for simplify (both haplotypes per individual)
    all_pop_nodes = sorted(n for sid in het_ids for n in het_nodes[sid])

    log.info("Total pairs: %d", len(haploid_pairs))
    return pop, haploid_pairs, individual_pairs, all_pop_nodes


def run_arm(arm, model, cfg, trees_dir, haploid_pairs, all_pop_nodes,
            out_dir, batch_size=64, device="cuda:0"):
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    if (arm_dir / "means.npz").exists():
        log.info("  %s: already computed, skipping", arm)
        return

    tree_path = f"{trees_dir}/{TREE_PATTERN.format(arm=arm)}"
    if not Path(tree_path).exists():
        log.warning("  %s: tree not found, skipping", arm)
        return

    t0 = time.time()
    log.info("  %s: loading tree sequence...", arm)
    ts_full = tskit.load(tree_path)
    arm_length = int(ts_full.sequence_length)

    # Simplify to just the population's haplotypes
    n_ind = len(all_pop_nodes) // 2
    log.info("  %s: simplifying to %d individuals (%d haplotypes)...",
             arm, n_ind, len(all_pop_nodes))
    ts = ts_full.simplify(samples=all_pop_nodes)
    del ts_full

    gm = ts.genotype_matrix().T
    positions = ts.tables.sites.position
    del ts
    log.info("  %s: %d sites, %d haplotypes, %.1f Mb",
             arm, len(positions), gm.shape[0], arm_length / 1e6)

    # Remap haploid pairs from original node IDs to simplified 0-indexed
    old_to_new = {old: new for new, old in enumerate(all_pop_nodes)}
    haploid_pairs = [(old_to_new[a], old_to_new[b]) for a, b in haploid_pairs]

    # Build uniform-size blocks; drop the last one if it's shorter than full
    block_tuples = []
    for start in range(0, arm_length, BLOCK_SIZE):
        end = start + BLOCK_SIZE
        if end > arm_length and (arm_length - start) < BLOCK_SIZE:
            break  # drop short trailing block
        block_tuples.append((start, end))
    n_blocks = len(block_tuples)
    log.info("  %s: %d blocks of %d bp", arm, n_blocks, BLOCK_SIZE)

    pair_batch_size = 500
    n_pairs = len(haploid_pairs)

    all_means = []
    all_vars = []
    all_index_maps = []

    for batch_start in range(0, n_pairs, pair_batch_size):
        batch_end = min(batch_start + pair_batch_size, n_pairs)
        batch_pairs = haploid_pairs[batch_start:batch_end]
        batch_n = len(batch_pairs)

        t_batch = time.time()
        means, variances, index_map = translate_from_genotype_matrix(
            gm=gm,
            positions=positions,
            model=model,
            blocks=block_tuples,
            pivot_pairs=batch_pairs,
            mutation_rate=MUTATION_RATE,
            device=device,
            batch_size=batch_size,
            build_workers=64,
            progress=False,
        )

        all_means.append(means)
        all_vars.append(variances)
        adjusted_map = index_map.copy()
        adjusted_map[:, 1] += batch_start
        all_index_maps.append(adjusted_map)

        elapsed_batch = time.time() - t_batch
        pairs_per_sec = batch_n / elapsed_batch
        remaining = (n_pairs - batch_end) / pairs_per_sec if pairs_per_sec > 0 else 0
        log.info("  %s: pairs %d-%d/%d (%.0f pairs/s, ~%.0f min remaining)",
                 arm, batch_start, batch_end, n_pairs, pairs_per_sec, remaining / 60)

    means = np.concatenate(all_means, axis=0)
    variances = np.concatenate(all_vars, axis=0)
    index_map = np.concatenate(all_index_maps, axis=0)

    np.savez_compressed(arm_dir / "means.npz", means=means)
    np.savez_compressed(arm_dir / "variances.npz", variances=variances)
    np.save(arm_dir / "index_map.npy", index_map)

    block_meta = [{"idx": i, "start": s, "end": e}
                  for i, (s, e) in enumerate(block_tuples)]
    with open(arm_dir / "blocks.json", "w") as f:
        json.dump(block_meta, f, indent=2)

    elapsed = time.time() - t0
    size_mb = sum(f.stat().st_size for f in arm_dir.glob("*")) / 1e6
    log.info("  %s: done in %.1f min, %.1f MB", arm, elapsed / 60, size_mb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--populations", nargs="+", default=None,
                        help="Populations to process (default: all)")
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--trees-dir", default=DEFAULT_TREES_DIR)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.checkpoint)

    # Discover populations
    full_meta = pd.read_csv(args.metadata)
    if args.populations:
        populations = args.populations
    else:
        populations = sorted(full_meta["country"].unique().tolist())
    log.info("Populations to process: %s", populations)

    # Load first tree sequence for sample mapping
    first_tree = f"{args.trees_dir}/{TREE_PATTERN.format(arm=args.arms[0])}"
    ts_first = tskit.load(first_tree)

    total_t0 = time.time()

    for population in populations:
        log.info("=" * 60)
        log.info("Population: %s", population)
        log.info("=" * 60)

        # Check for 2La-het individuals
        pop_het = full_meta[
            (full_meta["country"] == population) & (full_meta["2La"].astype(int) == 1)
        ]
        if len(pop_het) == 0:
            log.warning("  Skipping '%s': no 2La-heterozygous individuals", population)
            continue

        try:
            pop_meta, haploid_pairs, individual_pairs, all_pop_nodes = \
                get_het_population_pairs(args.metadata, ts_first, population, args.max_pairs)
        except ValueError as e:
            log.warning("  Skipping '%s': %s", population, e)
            continue

        # Population output directory
        pop_dir = args.out_dir / population.replace(" ", "_")
        pop_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        pop_meta.to_csv(pop_dir / "population_metadata.csv", index=False)
        pair_df = pd.DataFrame(individual_pairs, columns=["sample_a", "sample_b"])
        pair_df.index.name = "pair_idx"
        pair_df.to_csv(pop_dir / "pair_index.csv")
        np.save(pop_dir / "haploid_pairs.npy", np.array(haploid_pairs, dtype=np.int32))

        run_config = {
            "population": population,
            "filter": "2La_heterozygous",
            "n_individuals": len(pop_meta),
            "n_pairs": len(haploid_pairs),
            "arms": args.arms,
            "checkpoint": args.checkpoint,
            "mutation_rate": MUTATION_RATE,
            "block_size": BLOCK_SIZE,
        }
        with open(pop_dir / "config.json", "w") as f:
            json.dump(run_config, f, indent=2)

        log.info("Starting %s: %d pairs, device=%s",
                 population, len(haploid_pairs), args.device)

        for arm in args.arms:
            log.info("Processing %s / %s...", population, arm)
            run_arm(
                arm=arm, model=model, cfg=cfg,
                trees_dir=args.trees_dir,
                haploid_pairs=haploid_pairs,
                all_pop_nodes=all_pop_nodes,
                out_dir=pop_dir,
                batch_size=args.batch_size,
                device=args.device,
            )

        log.info("Population '%s' done", population)

    del ts_first
    log.info("All populations done in %.1f min", (time.time() - total_t0) / 60)


if __name__ == "__main__":
    main()
