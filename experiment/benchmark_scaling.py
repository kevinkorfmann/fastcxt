#!/usr/bin/env python3
"""Benchmark pairwise O(n²) vs node-time O(1) inference scaling.

Compares three approaches:
  1. Pairwise FastCxtModel (one forward pass per pair)
  2. NodeTimeModel with true tree topology + LCA lookup
  3. NodeTimeModel with tsinfer-inferred topology + LCA lookup

Usage:
    python benchmark_scaling.py \
        --pairwise-ckpt outputs/pairwise/.../best.ckpt \
        --node-weights-true outputs/node_time_true.pt \
        --node-weights-tsinfer outputs/node_time_tsinfer.pt
"""

from __future__ import annotations

import argparse
import csv
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.serialization
import msprime
import tskit

from fastcxt.config import FastCxtConfig, TrainingConfig
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_ts
from fastcxt.tree_utils import (
    FEATS_PER_NODE,
    extract_topology_features,
    lca_lookup_batch,
)
from fastcxt.node_time_model import NodeTimeModel
from config import (
    OUTPUTS_DIR, FIGURES_DIR, MAX_SAMPLES, N_WINDOWS, WINDOW_SIZE,
    SEQ_LEN, MUTATION_RATE, NE, NODE_D_MODEL, NODE_N_LAYERS,
)

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

DEVICE = "cuda:0"
REPEATS = 3
SAMPLE_SIZES = [50, 100, 200]
PAIR_COUNTS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]


def simulate_ts(n_samples: int, seed: int = 0) -> tskit.TreeSequence:
    ts = msprime.sim_ancestry(
        samples=n_samples, population_size=NE,
        recombination_rate=1e-8, sequence_length=SEQ_LEN,
        random_seed=seed,
    )
    return msprime.sim_mutations(ts, rate=MUTATION_RATE, random_seed=seed + 1)


def load_node_model(weights_path: Path) -> NodeTimeModel:
    max_internal = MAX_SAMPLES - 1
    model = NodeTimeModel(
        tree_feat_dim=max_internal * FEATS_PER_NODE,
        n_internal=max_internal,
        d_model=NODE_D_MODEL, n_layers=NODE_N_LAYERS,
        n_windows=N_WINDOWS,
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model.to(DEVICE).eval()


def bench_pairwise(model, ts, pairs, repeats=REPEATS):
    blocks = [(0, int(ts.sequence_length))]
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        translate_from_ts(ts, model, blocks=blocks, pivot_pairs=list(pairs),
                          mutation_rate=MUTATION_RATE, device=DEVICE, progress=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


def bench_node_time(node_model, ts, pairs, tree_feats_t, log_mu_t, repeats=REPEATS):
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            pred = node_model(tree_feats_t, log_mu_t)
        pred_np = pred[0].cpu().numpy()
        _ = lca_lookup_batch(ts, pred_np, list(pairs),
                             n_windows=N_WINDOWS, window_size=WINDOW_SIZE)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return np.mean(times) * 1000, np.std(times) * 1000


def extract_feats_tensor(ts, use_tsinfer=False):
    max_internal = MAX_SAMPLES - 1
    source_ts = ts
    if use_tsinfer:
        import tsinfer
        sd = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        source_ts = tsinfer.infer(sd)
    feats = extract_topology_features(
        source_ts, n_windows=N_WINDOWS, window_size=WINDOW_SIZE,
        max_internal=max_internal,
    )
    return torch.as_tensor(feats[np.newaxis], dtype=torch.float32).to(DEVICE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairwise-ckpt", type=str, required=True)
    ap.add_argument("--node-weights-true", type=str, default=str(OUTPUTS_DIR / "node_time_true.pt"))
    ap.add_argument("--node-weights-tsinfer", type=str, default=None)
    ap.add_argument("--out-csv", type=str, default=str(OUTPUTS_DIR / "benchmark_results.csv"))
    args = ap.parse_args()

    lit = LitFastCxt.load_from_checkpoint(args.pairwise_ckpt, map_location="cpu")
    pair_model = lit.model.to(DEVICE).eval()
    node_true = load_node_model(Path(args.node_weights_true))
    node_tsinfer = load_node_model(Path(args.node_weights_tsinfer)) if args.node_weights_tsinfer else None

    log_mu_t = torch.tensor([[np.log(MUTATION_RATE)]], dtype=torch.float32).to(DEVICE)

    results = []
    for n_samples in SAMPLE_SIZES:
        print(f"\n{'='*60}")
        print(f"  n_samples = {n_samples}")
        print(f"{'='*60}")
        ts = simulate_ts(n_samples, seed=n_samples)
        all_pairs = list(combinations(range(ts.num_samples), 2))

        feats_true = extract_feats_tensor(ts, use_tsinfer=False)
        feats_tsinfer = extract_feats_tensor(ts, use_tsinfer=True) if node_tsinfer else None

        # Warmup
        translate_from_ts(ts, pair_model, blocks=[(0, SEQ_LEN)], pivot_pairs=[(0, 1)],
                          mutation_rate=MUTATION_RATE, device=DEVICE, progress=False)
        with torch.inference_mode():
            node_true(feats_true, log_mu_t)

        counts = sorted(set(p for p in PAIR_COUNTS if p <= len(all_pairs)) | {len(all_pairs)})
        for n_pairs in counts:
            pairs = all_pairs[:n_pairs]
            pm, ps = bench_pairwise(pair_model, ts, pairs)
            nm, ns = bench_node_time(node_true, ts, pairs, feats_true, log_mu_t)
            row = {"n_samples": n_samples, "n_pairs": n_pairs,
                   "pair_ms": pm, "pair_std": ps,
                   "node_true_ms": nm, "node_true_std": ns,
                   "speedup_true": pm / nm if nm > 0 else 0}
            if node_tsinfer and feats_tsinfer is not None:
                tm, ts_ = bench_node_time(node_tsinfer, ts, pairs, feats_tsinfer, log_mu_t)
                row.update({"node_tsinfer_ms": tm, "node_tsinfer_std": ts_,
                            "speedup_tsinfer": pm / tm if tm > 0 else 0})
            results.append(row)
            su = row["speedup_true"]
            print(f"  {n_pairs:5d} pairs: pair={pm:7.0f}ms  node_true={nm:6.0f}ms  {su:5.1f}x")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {out_csv}")


if __name__ == "__main__":
    main()
