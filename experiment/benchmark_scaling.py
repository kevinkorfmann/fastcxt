#!/usr/bin/env python3
"""Benchmark pair-based vs node-time scaling up to 19,900 pairs (200 samples).

Saves results to a CSV file that plot_scaling.py reads for figure generation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python benchmark_scaling.py
    CUDA_VISIBLE_DEVICES=0 python benchmark_scaling.py \
        --base-checkpoint lightning_logs/version_0/checkpoints/last.ckpt \
        --node-weights node_time_model.pt
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import msprime
import tskit

from fastcxt.tree_utils import (
    FEATS_PER_NODE,
    extract_topology_features,
    extract_node_times,
    lca_lookup_batch,
)
from fastcxt.node_time_model import NodeTimeModel
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_ts

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEVICE = "cuda:0"
WINDOW_SIZE = 2000
SEQ_LEN = 1_000_000
N_WINDOWS = SEQ_LEN // WINDOW_SIZE
MUTATION_RATE = 1e-8


def simulate_ts(n_samples: int, seed: int = 42) -> "tskit.TreeSequence":
    ts = msprime.sim_ancestry(
        samples=n_samples,
        sequence_length=SEQ_LEN,
        recombination_rate=1e-8,
        population_size=10_000,
        random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=MUTATION_RATE, random_seed=seed)
    return ts


def bench_pair_based(model, ts, pairs, blocks, repeats=3):
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        translate_from_ts(
            ts, model, blocks=blocks, pivot_pairs=list(pairs),
            mutation_rate=MUTATION_RATE, device=DEVICE, progress=False,
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


def bench_node_time(model, ts, pairs, tree_feats_t, log_mu_t, n_windows, window_size, repeats=3):
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            node_times_pred = model(tree_feats_t, log_mu_t)
        node_times_np = node_times_pred[0].cpu().numpy()
        _ = lca_lookup_batch(
            ts, node_times_np, list(pairs),
            n_windows=n_windows, window_size=window_size,
        )
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


def run_benchmark(n_samples: int, base_model, node_weights_path: Path | None,
                  max_samples: int, pair_counts: list[int]):
    print(f"\n{'='*80}")
    print(f"  {n_samples} samples  ({n_samples*(n_samples-1)//2} total pairs)")
    print(f"{'='*80}")

    print(f"  Simulating ts with {n_samples} samples ...")
    ts = simulate_ts(n_samples)
    seq_len = int(ts.sequence_length)
    blocks = [(0, seq_len)]

    max_internal = max_samples - 1
    tree_feat_dim = max_internal * FEATS_PER_NODE

    print(f"  Extracting tree features ...")
    tree_feats_np = extract_topology_features(
        ts, n_windows=N_WINDOWS, window_size=WINDOW_SIZE, max_internal=max_internal,
    )
    tree_feats_t = torch.as_tensor(
        tree_feats_np[np.newaxis], dtype=torch.float32,
    ).to(DEVICE)

    node_model = NodeTimeModel(
        tree_feat_dim=tree_feat_dim,
        n_internal=max_internal,
        d_model=256,
        n_layers=4,
        n_windows=N_WINDOWS,
    ).to(DEVICE)

    if node_weights_path and node_weights_path.exists():
        node_model.load_state_dict(torch.load(node_weights_path, map_location=DEVICE, weights_only=True))
        print(f"  Loaded node-time weights from {node_weights_path}")
    else:
        print(f"  Using random node-time weights (timing only)")
    node_model.eval()

    log_mu_t = torch.tensor([[np.log(MUTATION_RATE)]], dtype=torch.float32).to(DEVICE)

    # Warmup
    with torch.inference_mode():
        _ = node_model(tree_feats_t, log_mu_t)
    translate_from_ts(
        ts, base_model, blocks=blocks, pivot_pairs=[(0, 1)],
        mutation_rate=MUTATION_RATE, device=DEVICE, progress=False,
    )

    all_pairs = list(combinations(range(n_samples), 2))
    pair_counts = sorted(set(p for p in pair_counts if p <= len(all_pairs)))
    if len(all_pairs) not in pair_counts:
        pair_counts.append(len(all_pairs))

    results = []
    for n_pairs in pair_counts:
        pivot = all_pairs[:n_pairs]
        repeats = 3 if n_pairs <= 500 else 1

        mb, sb = bench_pair_based(base_model, ts, pivot, blocks, repeats)
        mn, sn = bench_node_time(node_model, ts, pivot, tree_feats_t, log_mu_t,
                                 N_WINDOWS, WINDOW_SIZE, repeats)
        speedup = mb / mn if mn > 0 else float("inf")
        results.append((n_pairs, mb, sb, mn, sn, speedup))
        print(f"  {n_pairs:6d} pairs: pair={mb:8.0f}ms  node={mn:6.0f}ms  {speedup:5.1f}x")

    return results


def main():
    ap = argparse.ArgumentParser(description="Benchmark pairwise vs node-time scaling")
    ap.add_argument("--base-checkpoint", type=str, default=None,
                    help="Pairwise model checkpoint (auto-discovers latest if omitted)")
    ap.add_argument("--node-weights", type=str, default=None,
                    help="NodeTimeModel .pt weights (random if omitted)")
    ap.add_argument("--max-samples", type=int, default=200,
                    help="Max sample size for tree feature padding")
    ap.add_argument("--out-csv", type=str,
                    default=str(EXPERIMENT_DIR / "benchmark_results.csv"),
                    help="Output CSV path")
    args = ap.parse_args()

    ckpt = args.base_checkpoint
    if ckpt is None:
        ckpts = sorted(EXPERIMENT_DIR.rglob("lightning_logs/*/checkpoints/*.ckpt"))
        if not ckpts:
            print("ERROR: No pairwise checkpoint found. Train the base model first.")
            return
        ckpt = str(ckpts[-1])
    print(f"Loading pair-based model from {ckpt} ...")
    lit = LitFastCxt.load_from_checkpoint(ckpt, map_location="cpu")
    base_model = lit.model
    base_model.to(DEVICE)
    base_model.eval()

    node_weights = Path(args.node_weights) if args.node_weights else EXPERIMENT_DIR / "node_time_model.pt"

    all_results = {}
    for n_samples in [50, 100, 200]:
        max_pairs = n_samples * (n_samples - 1) // 2
        counts = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, max_pairs]
        counts = [c for c in counts if c <= max_pairs]
        all_results[n_samples] = run_benchmark(
            n_samples, base_model, node_weights, args.max_samples, counts,
        )

    # Summary table
    print("\n" + "=" * 90)
    print("FULL SCALING COMPARISON")
    print("=" * 90)
    print(f"{'n_samples':>9} {'pairs':>7} | {'pair-based (ms)':>16} | {'node-time (ms)':>16} | {'speedup':>8}")
    print("-" * 90)
    for n_samples, results in all_results.items():
        for n_pairs, mb, sb, mn, sn, speedup in results:
            print(f"{n_samples:9d} {n_pairs:7d} | {mb:8.0f} ± {sb:5.0f} | {mn:8.0f} ± {sn:5.0f} | {speedup:6.1f}x")
        print("-" * 90)
    print("=" * 90)

    # Save to CSV
    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_samples", "n_pairs", "pair_ms", "pair_std", "node_ms", "node_std", "speedup"])
        for n_samples, results in all_results.items():
            for n_pairs, mb, sb, mn, sn, speedup in results:
                w.writerow([n_samples, n_pairs, f"{mb:.1f}", f"{sb:.1f}", f"{mn:.1f}", f"{sn:.1f}", f"{speedup:.1f}"])
    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
