#!/usr/bin/env python3
"""Run pairwise TMRCA inference on real Ag1000G data for a population.

Loads a trained fastcxt model and runs inference on all pairs of individuals
from a specified population across all chromosome arms. Results are saved
as compressed numpy arrays with full metadata for downstream analysis.

Usage:
    # Burkina Faso (default)
    python run_inference.py --out-dir /sietch_colab/kkor/fastcxt/results/burkina_faso

    # Different population
    python run_inference.py --out-dir ./results/uganda --population Uganda

    # Subset of pairs (first 100)
    python run_inference.py --out-dir ./results/bf_test --max-pairs 100

    # Single arm
    python run_inference.py --out-dir ./results/bf_2L --arms 2L
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
from fastcxt.mosquito import (
    ANOGAM_CHROMOSOME_ARMS,
    AccessibilityMask,
    generate_blocks,
)

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

# ---------------------------------------------------------------------------
# Defaults (sietch paths)
# ---------------------------------------------------------------------------

ARMS = ["2L", "2R", "3L", "3R", "X"]
MUTATION_RATE = 3.5e-9
BLOCK_SIZE = 1_000_000  # 1 Mb blocks

DEFAULT_TREES_DIR = "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/tsinfer_data_v2"
DEFAULT_METADATA = "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/gamb.meta.tsinfer.csv"
DEFAULT_ACCESSIBILITY = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/agp3.is_accessible.txt.npz"
DEFAULT_CHECKPOINT = "/sietch_colab/kkor/fastcxt/experiment_mosquito/experiment/outputs/pairwise/csv_metrics/version_1/checkpoints/epoch=29-step=20880.ckpt"

# Tree file naming pattern
TREE_PATTERN = "gamb.{arm}.gff.dated.ne.trees"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_path: str):
    """Load pairwise FastCxtModel from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt["state_dict"] = {
        k.replace("._orig_mod", ""): v for k, v in ckpt["state_dict"].items()
    }
    lit = LitFastCxt(**ckpt["hyper_parameters"])
    lit.load_state_dict(ckpt["state_dict"])
    model = lit.model.eval()
    cfg = ckpt["hyper_parameters"]["model_config"]
    log.info("Loaded model: %s, window_size=%d, n_windows=%d",
             checkpoint_path, cfg.window_size, cfg.n_windows)
    return model, cfg


# ---------------------------------------------------------------------------
# Population setup
# ---------------------------------------------------------------------------

def get_population_pairs(
    metadata_path: str,
    ts: tskit.TreeSequence,
    population: str,
    max_pairs: int | None = None,
) -> tuple[pd.DataFrame, list[tuple[int, int]], list[tuple[str, str]]]:
    """Get all haploid pairs for a population.

    Returns
    -------
    pop_meta : DataFrame of individuals in the population
    haploid_pairs : list of (hap_idx_a, hap_idx_b) for model input
    individual_pairs : list of (sample_id_a, sample_id_b) for metadata
    """
    meta = pd.read_csv(metadata_path)
    pop = meta[meta["country"] == population].copy()
    if len(pop) == 0:
        raise ValueError(f"No samples found for population '{population}'. "
                         f"Available: {meta['country'].unique().tolist()}")

    log.info("Population '%s': %d individuals (%d haplotypes)",
             population, len(pop), len(pop) * 2)

    # Map sample IDs to tree sequence node indices
    # Tree sequence has 2 haploid nodes per individual
    ts_individuals = {}
    for ind in ts.individuals():
        ind_meta = json.loads(ind.metadata.decode("ascii"))
        sample_id = ind_meta.get("sample_name", ind_meta.get("sample_id", ""))
        ts_individuals[sample_id] = ind

    # Build pairs of individuals (all combinations)
    pop_ids = pop["sample_id"].values
    individual_pairs = list(combinations(pop_ids, 2))

    if max_pairs is not None and len(individual_pairs) > max_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(individual_pairs), max_pairs, replace=False)
        individual_pairs = [individual_pairs[i] for i in sorted(idx)]
        log.info("Subsampled to %d pairs", max_pairs)

    # Convert to haploid pairs (use first haplotype of each individual)
    haploid_pairs = []
    skipped = 0
    valid_individual_pairs = []
    for id_a, id_b in individual_pairs:
        if id_a not in ts_individuals or id_b not in ts_individuals:
            skipped += 1
            continue
        node_a = ts_individuals[id_a].nodes[0]  # first haplotype
        node_b = ts_individuals[id_b].nodes[0]
        haploid_pairs.append((int(node_a), int(node_b)))
        valid_individual_pairs.append((id_a, id_b))

    if skipped:
        log.warning("Skipped %d pairs (samples not found in tree sequence)", skipped)

    log.info("Total pairs: %d", len(haploid_pairs))
    return pop, haploid_pairs, valid_individual_pairs


# ---------------------------------------------------------------------------
# Per-arm inference
# ---------------------------------------------------------------------------

def run_arm(
    arm: str,
    model,
    cfg: FastCxtConfig,
    trees_dir: str,
    haploid_pairs: list[tuple[int, int]],
    accessibility_path: str,
    out_dir: Path,
    batch_size: int = 64,
    device: str = "cuda:0",
):
    """Run inference for one chromosome arm and save results."""
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)

    # Check if already computed
    if (arm_dir / "means.npz").exists():
        log.info("  %s: already computed, skipping", arm)
        return

    tree_path = f"{trees_dir}/{TREE_PATTERN.format(arm=arm)}"
    if not Path(tree_path).exists():
        log.warning("  %s: tree file not found at %s, skipping", arm, tree_path)
        return

    t0 = time.time()
    log.info("  %s: loading tree sequence...", arm)
    ts = tskit.load(tree_path)
    arm_length = int(ts.sequence_length)
    gm = ts.genotype_matrix().T  # (n_haploids, n_sites)
    positions = ts.tables.sites.position
    log.info("  %s: %d sites, %d haplotypes, %.1f Mb",
             arm, len(positions), gm.shape[0], arm_length / 1e6)

    # Generate 1Mb blocks
    blocks = generate_blocks(arm, arm_length, BLOCK_SIZE)
    block_tuples = [(b.start, b.end) for b in blocks]
    n_blocks = len(block_tuples)
    log.info("  %s: %d blocks of %d bp", arm, n_blocks, BLOCK_SIZE)

    # Run inference in pair batches to manage memory
    pair_batch_size = 500  # pairs per batch
    n_pairs = len(haploid_pairs)
    n_windows_per_block = cfg.n_windows

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
            progress=False,
        )

        all_means.append(means)
        all_vars.append(variances)
        # Adjust index_map pair indices for global offset
        adjusted_map = index_map.copy()
        adjusted_map[:, 1] += batch_start
        all_index_maps.append(adjusted_map)

        elapsed_batch = time.time() - t_batch
        pairs_per_sec = batch_n / elapsed_batch
        remaining = (n_pairs - batch_end) / pairs_per_sec if pairs_per_sec > 0 else 0
        log.info("  %s: pairs %d-%d/%d (%.0f pairs/s, ~%.0f min remaining)",
                 arm, batch_start, batch_end, n_pairs, pairs_per_sec, remaining / 60)

    # Concatenate results
    means = np.concatenate(all_means, axis=0)
    variances = np.concatenate(all_vars, axis=0)
    index_map = np.concatenate(all_index_maps, axis=0)

    elapsed = time.time() - t0

    # Save
    np.savez_compressed(
        arm_dir / "means.npz",
        means=means,  # (n_pairs * n_blocks, n_windows)
    )
    np.savez_compressed(
        arm_dir / "variances.npz",
        variances=variances,  # (n_pairs * n_blocks, n_windows)
    )
    np.save(arm_dir / "index_map.npy", index_map)  # (n_rows, 2) [block_idx, pair_idx]

    # Save block metadata
    block_meta = [{"idx": i, "start": int(b.start), "end": int(b.end)} for i, b in enumerate(blocks)]
    with open(arm_dir / "blocks.json", "w") as f:
        json.dump(block_meta, f, indent=2)

    size_mb = sum(f.stat().st_size for f in arm_dir.glob("*")) / 1e6
    log.info("  %s: done in %.1f min, saved %.1f MB", arm, elapsed / 60, size_mb)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run pairwise TMRCA inference on a population"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--population", default="Burkina Faso",
                        help="Country name to filter samples")
    parser.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--trees-dir", default=DEFAULT_TREES_DIR)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--accessibility", default=DEFAULT_ACCESSIBILITY)
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Max number of pairs (for testing)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, cfg = load_model(args.checkpoint)

    # Load first tree sequence to get sample mapping
    first_arm = args.arms[0]
    tree_path = f"{args.trees_dir}/{TREE_PATTERN.format(arm=first_arm)}"
    ts_first = tskit.load(tree_path)

    # Get population pairs
    pop_meta, haploid_pairs, individual_pairs = get_population_pairs(
        args.metadata, ts_first, args.population, args.max_pairs,
    )

    # Save metadata
    pop_meta.to_csv(args.out_dir / "population_metadata.csv", index=False)

    # Save pair index — maps pair_idx to (sample_id_a, sample_id_b)
    pair_df = pd.DataFrame(individual_pairs, columns=["sample_a", "sample_b"])
    pair_df.index.name = "pair_idx"
    pair_df.to_csv(args.out_dir / "pair_index.csv")

    # Save haploid pair mapping
    np.save(args.out_dir / "haploid_pairs.npy", np.array(haploid_pairs, dtype=np.int32))

    # Save run config
    run_config = {
        "population": args.population,
        "n_individuals": len(pop_meta),
        "n_pairs": len(haploid_pairs),
        "arms": args.arms,
        "checkpoint": args.checkpoint,
        "trees_dir": args.trees_dir,
        "mutation_rate": MUTATION_RATE,
        "block_size": BLOCK_SIZE,
        "model_window_size": cfg.window_size,
        "model_n_windows": cfg.n_windows,
        "model_max_samples": cfg.max_samples,
    }
    with open(args.out_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    log.info("Config saved. Starting inference on %d pairs across %s",
             len(haploid_pairs), args.arms)

    # Run per arm
    total_t0 = time.time()
    for arm in args.arms:
        log.info("Processing %s...", arm)
        run_arm(
            arm=arm,
            model=model,
            cfg=cfg,
            trees_dir=args.trees_dir,
            haploid_pairs=haploid_pairs,
            accessibility_path=args.accessibility,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            device=args.device,
        )

    total_elapsed = time.time() - total_t0
    log.info("All done in %.1f min", total_elapsed / 60)
    log.info("Results: %s", args.out_dir)
    log.info("Output structure:")
    log.info("  config.json           — run parameters")
    log.info("  population_metadata.csv — sample info")
    log.info("  pair_index.csv        — pair_idx -> (sample_a, sample_b)")
    log.info("  haploid_pairs.npy     — (n_pairs, 2) haploid node indices")
    log.info("  {arm}/means.npz       — (n_rows, n_windows) predicted log(TMRCA)")
    log.info("  {arm}/variances.npz   — (n_rows, n_windows) predicted variance")
    log.info("  {arm}/index_map.npy   — (n_rows, 2) [block_idx, pair_idx]")
    log.info("  {arm}/blocks.json     — block start/end coordinates")


if __name__ == "__main__":
    main()
