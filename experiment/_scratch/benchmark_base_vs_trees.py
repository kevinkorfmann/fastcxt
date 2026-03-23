#!/usr/bin/env python3
"""Benchmark base vs base_trees inference.

Run on physical GPU 2: CUDA_VISIBLE_DEVICES=2 python benchmark_base_vs_trees.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import tskit

from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_ts

EXPERIMENT_DIR = Path(__file__).resolve().parent
CHECKPOINT_BASE = EXPERIMENT_DIR / "lightning_logs/version_9/checkpoints/epoch=19-step=4680.ckpt"
CHECKPOINT_TREES = EXPERIMENT_DIR / "lightning_logs/version_11/checkpoints/epoch=19-step=4680.ckpt"
TS_FILE = EXPERIMENT_DIR / "sims/constant/ts_00000000.trees"
MUTATION_RATE = 1e-8
# Use cuda:0 with CUDA_VISIBLE_DEVICES=2 to target physical GPU 2
DEVICE = "cuda:0"
WARMUP = 2
REPEATS = 3


def main():
    print(f"Device: {DEVICE}")
    print("Loading tree sequence ...")
    ts = tskit.load(str(TS_FILE))
    seq_len = int(ts.sequence_length)
    blocks = [(0, seq_len)]

    print("Loading models ...")
    lit_base = LitFastCxt.load_from_checkpoint(str(CHECKPOINT_BASE), map_location="cpu")
    lit_trees = LitFastCxt.load_from_checkpoint(str(CHECKPOINT_TREES), map_location="cpu")

    # Vary number of pairs to probe scaling
    n_samples = ts.num_samples
    pair_counts = [1, 5, 10, 25, 50, 100, 150, 200]
    pair_counts = [p for p in pair_counts if p <= n_samples * (n_samples - 1) // 2]

    # Generate disjoint pairs: (0,1), (2,3), (4,5), ...
    def make_pairs(n):
        out = []
        for i in range(0, min(2 * n, n_samples) - 1, 2):
            out.append((i, i + 1))
        return out[:n]

    results = []
    for n_pairs in pair_counts:
        pivot_pairs = make_pairs(n_pairs)
        if len(pivot_pairs) < n_pairs:
            continue

        times_base = []
        times_trees = []

        for _ in range(WARMUP + REPEATS):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            translate_from_ts(
                ts, lit_base.model,
                blocks=blocks, pivot_pairs=pivot_pairs,
                mutation_rate=MUTATION_RATE,
                device=DEVICE, progress=False,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if _ >= WARMUP:
                times_base.append(t1 - t0)

        for _ in range(WARMUP + REPEATS):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            translate_from_ts(
                ts, lit_trees.model,
                blocks=blocks, pivot_pairs=pivot_pairs,
                mutation_rate=MUTATION_RATE,
                device=DEVICE, progress=False,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            if _ >= WARMUP:
                times_trees.append(t1 - t0)

        mean_base = np.mean(times_base) * 1000
        mean_trees = np.mean(times_trees) * 1000
        std_base = np.std(times_base) * 1000
        std_trees = np.std(times_trees) * 1000
        results.append((n_pairs, mean_base, std_base, mean_trees, std_trees))
        print(f"  {n_pairs:3d} pairs: base {mean_base:.0f} ± {std_base:.0f} ms | base_trees {mean_trees:.0f} ± {std_trees:.0f} ms")

    print("\n" + "=" * 70)
    print("Benchmark results (inference time in ms)")
    print("=" * 70)
    print(f"{'pairs':>6} | {'base (ms)':>12} | {'base_trees (ms)':>16} | {'ratio':>6}")
    print("-" * 70)
    for n_pairs, mb, sb, mt, st in results:
        ratio = mt / mb if mb > 0 else float("nan")
        print(f"{n_pairs:6d} | {mb:8.0f} ± {sb:.0f} | {mt:12.0f} ± {st:.0f} | {ratio:.2f}x")
    print("=" * 70)


if __name__ == "__main__":
    main()
