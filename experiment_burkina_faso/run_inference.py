#!/usr/bin/env python3
"""Run comprehensive TMRCA inference on Ag1000G data.

For each population and chromosome arm, groups individuals by karyotype
(for arms with inversions: hom_standard, heterozygous, hom_inverted;
for others: all individuals), then runs both intra-individual and
inter-individual (capped at 100) pairs.

Output structure:
    out_dir/<population>/<arm>/<karyotype>_intra/
    out_dir/<population>/<arm>/<karyotype>_inter/

Usage:
    python run_inference.py --out-dir /sietch_colab/kkor/fastcxt/results/comprehensive
    python run_inference.py --out-dir ./test --populations "Burkina Faso" --arms 2L
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
BLOCK_SIZE = 100_000
MAX_INTER_PAIRS = 100
MAX_INDIVIDUALS = 50

# Inversion karyotype column per arm; arms not listed use all individuals
ARM_INVERSION = {
    "2L": "2La",
    "2R": "2Rb",
}

# Karyotype groups for arms with inversions
KARYO_GROUPS = [
    (0, "hom_standard"),
    (1, "heterozygous"),
    (2, "hom_inverted"),
]

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


def get_ts_individual_map(ts):
    """Map sample IDs to tree-sequence individual objects."""
    ts_individuals = {}
    for ind in ts.individuals():
        ind_meta = json.loads(ind.metadata.decode("ascii"))
        sid = ind_meta.get("sample_name", ind_meta.get("sample_id", ""))
        ts_individuals[sid] = ind
    return ts_individuals


def get_karyotype_groups(meta, population, arm):
    """Return list of (group_name, filtered_dataframe) for a population/arm."""
    pop = meta[meta["country"] == population]
    inv_col = ARM_INVERSION.get(arm)

    if inv_col is not None:
        groups = []
        for karyo_val, label in KARYO_GROUPS:
            sub = pop[pop[inv_col].astype(int) == karyo_val].copy()
            if len(sub) > 0:
                groups.append((f"{inv_col}_{label}", sub))
        return groups
    else:
        return [("all", pop.copy())]


def build_pairs(sample_ids, ts_individuals, pair_type,
                max_inter=MAX_INTER_PAIRS, max_individuals=MAX_INDIVIDUALS):
    """Build haploid pairs and metadata for a group of individuals.

    pair_type: "intra" or "inter"
    Returns (haploid_pairs, individual_pairs, all_nodes) or (None, None, None) if empty.
    """
    # Collect diploid individuals present in tree sequence
    # (skip haploid individuals, e.g. males on X chromosome)
    valid_ids = []
    nodes = {}
    for sid in sample_ids:
        if sid in ts_individuals:
            n = ts_individuals[sid].nodes
            if len(n) < 2:
                continue  # haploid, skip
            valid_ids.append(sid)
            nodes[sid] = (int(n[0]), int(n[1]))

    if len(valid_ids) == 0:
        return None, None, None

    # Cap individuals to keep genotype matrices manageable
    if len(valid_ids) > max_individuals:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(valid_ids), max_individuals, replace=False)
        valid_ids = [valid_ids[i] for i in sorted(idx)]

    if pair_type == "intra":
        individual_pairs = [(sid, sid) for sid in valid_ids]
        haploid_pairs = [(nodes[sid][0], nodes[sid][1]) for sid in valid_ids]
    else:  # inter
        individual_pairs = list(combinations(valid_ids, 2))
        if len(individual_pairs) > max_inter:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(individual_pairs), max_inter, replace=False)
            individual_pairs = [individual_pairs[i] for i in sorted(idx)]
        haploid_pairs = [(nodes[a][0], nodes[b][0]) for a, b in individual_pairs]

    all_nodes = sorted(n for sid in valid_ids for n in nodes[sid])
    return haploid_pairs, individual_pairs, all_nodes


def run_group(
    arm, group_name, pair_type, model, cfg, trees_dir,
    haploid_pairs, all_pop_nodes, out_dir,
    batch_size=64, device="cuda:0",
):
    """Run inference for one karyotype group + pair type."""
    result_dir = out_dir / f"{group_name}_{pair_type}"
    result_dir.mkdir(parents=True, exist_ok=True)

    if (result_dir / "means.npz").exists():
        log.info("    %s_%s: already computed, skipping", group_name, pair_type)
        return

    tree_path = f"{trees_dir}/{TREE_PATTERN.format(arm=arm)}"
    if not Path(tree_path).exists():
        log.warning("    %s: tree not found, skipping", arm)
        return

    t0 = time.time()
    log.info("    %s_%s: loading tree sequence...", group_name, pair_type)
    ts_full = tskit.load(tree_path)
    arm_length = int(ts_full.sequence_length)

    n_ind = len(all_pop_nodes) // 2
    log.info("    simplifying to %d individuals (%d haplotypes)...", n_ind, len(all_pop_nodes))
    ts = ts_full.simplify(samples=all_pop_nodes)
    del ts_full

    gm = ts.genotype_matrix().T
    positions = ts.tables.sites.position
    del ts
    log.info("    %d sites, %d haplotypes, %.1f Mb",
             len(positions), gm.shape[0], arm_length / 1e6)

    old_to_new = {old: new for new, old in enumerate(all_pop_nodes)}
    remapped = [(old_to_new[a], old_to_new[b]) for a, b in haploid_pairs]

    block_tuples = []
    for start in range(0, arm_length, BLOCK_SIZE):
        end = start + BLOCK_SIZE
        if end > arm_length and (arm_length - start) < BLOCK_SIZE:
            break
        block_tuples.append((start, end))
    log.info("    %d blocks of %d bp, %d pairs", len(block_tuples), BLOCK_SIZE, len(remapped))

    pair_batch_size = 500
    n_pairs = len(remapped)
    all_means, all_vars, all_index_maps = [], [], []

    for batch_start in range(0, n_pairs, pair_batch_size):
        batch_end = min(batch_start + pair_batch_size, n_pairs)
        batch_pairs = remapped[batch_start:batch_end]

        t_batch = time.time()
        means, variances, index_map = translate_from_genotype_matrix(
            gm=gm, positions=positions, model=model,
            blocks=block_tuples, pivot_pairs=batch_pairs,
            mutation_rate=MUTATION_RATE, device=device,
            batch_size=batch_size, build_workers=64, progress=False,
        )

        all_means.append(means)
        all_vars.append(variances)
        adj = index_map.copy()
        adj[:, 1] += batch_start
        all_index_maps.append(adj)

        elapsed = time.time() - t_batch
        pps = len(batch_pairs) / elapsed if elapsed > 0 else 0
        rem = (n_pairs - batch_end) / pps / 60 if pps > 0 else 0
        log.info("    pairs %d-%d/%d (%.0f pairs/s, ~%.0f min left)",
                 batch_start, batch_end, n_pairs, pps, rem)

    means = np.concatenate(all_means)
    variances = np.concatenate(all_vars)
    index_map = np.concatenate(all_index_maps)

    np.savez_compressed(result_dir / "means.npz", means=means)
    np.savez_compressed(result_dir / "variances.npz", variances=variances)
    np.save(result_dir / "index_map.npy", index_map)

    block_meta = [{"idx": i, "start": s, "end": e} for i, (s, e) in enumerate(block_tuples)]
    with open(result_dir / "blocks.json", "w") as f:
        json.dump(block_meta, f, indent=2)

    elapsed = time.time() - t0
    size_mb = sum(f.stat().st_size for f in result_dir.glob("*")) / 1e6
    log.info("    %s_%s: done in %.1f min, %.1f MB", group_name, pair_type, elapsed / 60, size_mb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--populations", nargs="+", default=None)
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--trees-dir", default=DEFAULT_TREES_DIR)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--max-inter", type=int, default=MAX_INTER_PAIRS)
    parser.add_argument("--max-individuals", type=int, default=MAX_INDIVIDUALS)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    model, cfg = load_model(args.checkpoint)

    full_meta = pd.read_csv(args.metadata)
    if args.populations:
        populations = args.populations
    else:
        populations = sorted(full_meta["country"].unique().tolist())
    log.info("Populations: %s", populations)

    total_t0 = time.time()

    for population in populations:
        log.info("=" * 70)
        log.info("Population: %s", population)
        log.info("=" * 70)

        pop_dir = args.out_dir / population.replace(" ", "_")

        for arm in args.arms:
            log.info("  Arm: %s", arm)

            tree_path = f"{args.trees_dir}/{TREE_PATTERN.format(arm=arm)}"
            if not Path(tree_path).exists():
                log.warning("    tree not found, skipping")
                continue

            ts = tskit.load(tree_path)
            ts_ind_map = get_ts_individual_map(ts)
            del ts

            groups = get_karyotype_groups(full_meta, population, arm)
            arm_dir = pop_dir / arm

            for group_name, group_meta in groups:
                sample_ids = group_meta["sample_id"].values

                for pair_type in ("intra", "inter"):
                    haploid_pairs, individual_pairs, all_nodes = build_pairs(
                        sample_ids, ts_ind_map, pair_type,
                        max_inter=args.max_inter,
                        max_individuals=args.max_individuals,
                    )
                    if haploid_pairs is None or len(haploid_pairs) == 0:
                        log.info("    %s_%s: no pairs, skipping", group_name, pair_type)
                        continue

                    if pair_type == "inter" and len(sample_ids) < 2:
                        continue

                    result_dir = arm_dir / f"{group_name}_{pair_type}"
                    result_dir.mkdir(parents=True, exist_ok=True)

                    # Save metadata
                    config = {
                        "population": population,
                        "arm": arm,
                        "group": group_name,
                        "pair_type": pair_type,
                        "n_individuals": len(set(a for a, _ in individual_pairs) | set(b for _, b in individual_pairs)),
                        "n_pairs": len(haploid_pairs),
                        "max_inter": args.max_inter if pair_type == "inter" else None,
                        "mutation_rate": MUTATION_RATE,
                        "block_size": BLOCK_SIZE,
                    }
                    with open(result_dir / "config.json", "w") as f:
                        json.dump(config, f, indent=2)

                    pair_df = pd.DataFrame(individual_pairs, columns=["sample_a", "sample_b"])
                    pair_df.index.name = "pair_idx"
                    pair_df.to_csv(result_dir / "pair_index.csv")
                    np.save(result_dir / "haploid_pairs.npy",
                            np.array(haploid_pairs, dtype=np.int32))

                    log.info("    %s_%s: %d pairs", group_name, pair_type, len(haploid_pairs))

                    run_group(
                        arm=arm, group_name=group_name, pair_type=pair_type,
                        model=model, cfg=cfg, trees_dir=args.trees_dir,
                        haploid_pairs=haploid_pairs, all_pop_nodes=all_nodes,
                        out_dir=arm_dir,
                        batch_size=args.batch_size, device=args.device,
                    )

        log.info("Population '%s' done", population)

    log.info("All done in %.1f min", (time.time() - total_t0) / 60)


if __name__ == "__main__":
    main()
