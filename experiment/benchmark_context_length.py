#!/usr/bin/env python3
"""Benchmark AnoGam tsinfer-augmented models: 500 vs 2000 windows.

Measures inference time broken down into three components:
  1. SFS build (CPU preprocessing)
  2. tsinfer + tree feature extraction
  3. Forward pass (GPU model inference)

Requires GPU + mamba-ssm.  Run on betty:
    CUDA_VISIBLE_DEVICES=0 python experiment/benchmark_context_length.py \
        --ckpt-500  <path> --ckpt-2000 <path>

Outputs:
    experiment/paper_figures/table_context_benchmark.csv
    experiment/paper_figures/fig6_context_scaling.png
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.serialization

from fastcxt.config import FastCxtConfig, TrainingConfig
from fastcxt.train import LitFastCxt
from fastcxt.translate import _build_sources, _extract_tree_feats_for_blocks, predict
from fastcxt.mosquito import simulate_anogam

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

MUTATION_RATE = 3.5e-9  # AnoGam
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
N_WARMUP = 2
N_REPEATS = 5


def load_model(ckpt_path: str):
    # Load checkpoint, strip _orig_mod prefix from torch.compile if needed
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", {})
    if any("_orig_mod" in k for k in sd):
        ckpt["state_dict"] = {k.replace("._orig_mod", ""): v for k, v in sd.items()}
    hp = ckpt["hyper_parameters"]
    lit = LitFastCxt(hp["model_config"], hp.get("training_config"))
    lit.load_state_dict(ckpt["state_dict"], strict=True)
    m = lit.model.to(DEVICE)
    m.eval()
    return m


def _sync():
    if "cuda" in DEVICE:
        torch.cuda.synchronize()


def bench_components(model, ts, n_pairs: int, window_size: int,
                     seq_len: int, warmup: int = N_WARMUP,
                     repeats: int = N_REPEATS) -> dict:
    """Time each pipeline component separately."""
    pairs = [(2 * i, 2 * i + 1) for i in range(n_pairs)
             if 2 * i + 1 < ts.num_samples]
    blocks = [(0, seq_len)]
    n_windows = seq_len // window_size

    gm = ts.genotype_matrix().T
    positions = ts.tables.sites.position
    tree_feat_dim = getattr(model.config, "tree_feat_dim", None)

    # --- Warmup all components ---
    for _ in range(warmup):
        X, idx = _build_sources(gm, positions, blocks, pairs,
                                window_size=window_size, workers=4, progress=False)
        tree_feats = _extract_tree_feats_for_blocks(
            ts, blocks, pairs, window_size=window_size,
            use_tsinfer=True, tree_feat_dim=tree_feat_dim)
        param_dtype = next(model.parameters()).dtype
        X_t = torch.as_tensor(X, dtype=param_dtype)
        tree_t = torch.as_tensor(tree_feats, dtype=param_dtype)
        _sync()
        predict(model, X_t, mutation_rate=MUTATION_RATE, device=DEVICE,
                progress=False, tree_feats=tree_t)
        _sync()

    # --- Time SFS build ---
    sfs_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        X, idx = _build_sources(gm, positions, blocks, pairs,
                                window_size=window_size, workers=4, progress=False)
        sfs_times.append(time.perf_counter() - t0)

    # --- Time tsinfer + tree feature extraction ---
    tsinfer_times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        tree_feats = _extract_tree_feats_for_blocks(
            ts, blocks, pairs, window_size=window_size,
            use_tsinfer=True, tree_feat_dim=tree_feat_dim)
        tsinfer_times.append(time.perf_counter() - t0)

    # --- Time forward pass (GPU) ---
    param_dtype = next(model.parameters()).dtype
    X_t = torch.as_tensor(X, dtype=param_dtype)
    tree_t = torch.as_tensor(tree_feats, dtype=param_dtype)
    if torch.cuda.is_available():
        X_t = X_t.pin_memory()
        tree_t = tree_t.pin_memory()

    fwd_times = []
    for _ in range(repeats):
        _sync()
        t0 = time.perf_counter()
        predict(model, X_t, mutation_rate=MUTATION_RATE, device=DEVICE,
                progress=False, tree_feats=tree_t)
        _sync()
        fwd_times.append(time.perf_counter() - t0)

    sfs_arr = np.array(sfs_times) * 1000
    tsinfer_arr = np.array(tsinfer_times) * 1000
    fwd_arr = np.array(fwd_times) * 1000
    total_arr = sfs_arr + tsinfer_arr + fwd_arr

    return {
        "n_pairs": len(pairs),
        "n_windows": n_windows,
        "seq_len": seq_len,
        "window_size": window_size,
        "sfs_ms": float(sfs_arr.mean()),
        "sfs_std": float(sfs_arr.std()),
        "tsinfer_ms": float(tsinfer_arr.mean()),
        "tsinfer_std": float(tsinfer_arr.std()),
        "fwd_ms": float(fwd_arr.mean()),
        "fwd_std": float(fwd_arr.std()),
        "total_ms": float(total_arr.mean()),
        "total_std": float(total_arr.std()),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark AnoGam tsinfer models: 500 vs 2000 windows")
    ap.add_argument("--ckpt-500", type=str, required=True,
                    help="Checkpoint for 100kb / 500-window model")
    ap.add_argument("--ckpt-2000", type=str, required=True,
                    help="Checkpoint for 400kb / 2000-window model")
    ap.add_argument("--n-samples", type=int, default=50,
                    help="Number of diploid samples for simulation")
    ap.add_argument("--n-pairs", type=int, default=3,
                    help="Number of focal pairs to predict")
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--warmup", type=int, default=N_WARMUP)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-dir", type=str,
                    default="experiment/paper_figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = {
        "500win_100kb": {
            "ckpt": args.ckpt_500,
            "seq_len": 100_000,
            "window_size": 200,
            "label": "100 kb (500 win)",
        },
        "2000win_400kb": {
            "ckpt": args.ckpt_2000,
            "seq_len": 400_000,
            "window_size": 200,
            "label": "400 kb (2k win)",
        },
    }

    results = []
    for key, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"  {cfg['label']}")
        print(f"{'='*60}")

        print(f"  Simulating AnoGam ({cfg['seq_len']//1000} kb, "
              f"n={args.n_samples}) ...")
        ts = simulate_anogam(seed=args.seed, n_samples=args.n_samples,
                             segment_length=cfg["seq_len"])
        print(f"    {ts.num_trees} trees, {ts.num_mutations} mutations, "
              f"{ts.num_samples} haplotypes")

        print(f"  Loading model from {cfg['ckpt']} ...")
        model = load_model(cfg["ckpt"])
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    {n_params:,} parameters")

        print(f"  Benchmarking ({args.warmup} warmup + {args.repeats} timed) ...")
        r = bench_components(model, ts, args.n_pairs,
                             cfg["window_size"], cfg["seq_len"],
                             warmup=args.warmup, repeats=args.repeats)
        r["key"] = key
        r["label"] = cfg["label"]
        r["n_params"] = n_params
        results.append(r)

        print(f"    SFS build:     {r['sfs_ms']:8.1f} ± {r['sfs_std']:.1f} ms")
        print(f"    tsinfer+trees: {r['tsinfer_ms']:8.1f} ± {r['tsinfer_std']:.1f} ms")
        print(f"    Forward pass:  {r['fwd_ms']:8.1f} ± {r['fwd_std']:.1f} ms")
        print(f"    Total:         {r['total_ms']:8.1f} ± {r['total_std']:.1f} ms")

        del model
        torch.cuda.empty_cache()

    # ---- Summary table ----
    print(f"\n{'='*80}")
    print("CONTEXT-LENGTH SCALING BENCHMARK (component breakdown)")
    print(f"{'='*80}")
    print(f"{'Model':<22} {'W':>5} {'SFS (ms)':>12} {'tsinfer (ms)':>14} "
          f"{'Fwd (ms)':>12} {'Total (ms)':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['label']:<22} {r['n_windows']:>5} "
              f"{r['sfs_ms']:>8.1f}±{r['sfs_std']:<4.1f}"
              f"{r['tsinfer_ms']:>10.1f}±{r['tsinfer_std']:<4.1f}"
              f"{r['fwd_ms']:>8.1f}±{r['fwd_std']:<4.1f}"
              f"{r['total_ms']:>8.1f}±{r['total_std']:<4.1f}")

    # Ratios
    r0, r1 = results[0], results[1]
    print(f"\n{'Ratio (2k/500):':<22}       "
          f"{'':>4}{r1['sfs_ms']/r0['sfs_ms']:.1f}x    "
          f"{'':>6}{r1['tsinfer_ms']/r0['tsinfer_ms']:.1f}x    "
          f"{'':>4}{r1['fwd_ms']/r0['fwd_ms']:.1f}x    "
          f"{'':>4}{r1['total_ms']/r0['total_ms']:.1f}x")
    print(f"{'='*80}")

    # ---- Save CSV ----
    csv_path = out_dir / "table_context_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_windows", "seq_len_bp", "window_size_bp",
                     "n_pairs", "n_params",
                     "sfs_ms", "sfs_std", "tsinfer_ms", "tsinfer_std",
                     "fwd_ms", "fwd_std", "total_ms", "total_std"])
        for r in results:
            w.writerow([
                r["label"], r["n_windows"], r["seq_len"], r["window_size"],
                r["n_pairs"], r["n_params"],
                f"{r['sfs_ms']:.1f}", f"{r['sfs_std']:.1f}",
                f"{r['tsinfer_ms']:.1f}", f"{r['tsinfer_std']:.1f}",
                f"{r['fwd_ms']:.1f}", f"{r['fwd_std']:.1f}",
                f"{r['total_ms']:.1f}", f"{r['total_std']:.1f}",
            ])
    print(f"\nResults saved to {csv_path}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        BLACK = "#252525"
        plt.rcParams.update({
            "figure.facecolor": "white", "axes.facecolor": "white",
            "axes.edgecolor": BLACK, "axes.linewidth": 0.6, "axes.grid": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
            "xtick.labelsize": 7, "ytick.labelsize": 7,
            "legend.fontsize": 7, "legend.frameon": False,
            "xtick.major.width": 0.6, "ytick.major.width": 0.6,
            "xtick.direction": "out", "ytick.direction": "out",
            "figure.dpi": 300, "savefig.dpi": 300,
            "pdf.fonttype": 42, "ps.fonttype": 42,
        })

        labels = [r["label"] for r in results]
        sfs = [r["sfs_ms"] for r in results]
        tsinfer = [r["tsinfer_ms"] for r in results]
        fwd = [r["fwd_ms"] for r in results]

        c_sfs = "#636363"
        c_tsinfer = "#2166ac"
        c_fwd = "#e08214"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

        # Panel a: stacked bar (component breakdown)
        x = np.arange(len(labels))
        w = 0.45
        ax1.bar(x, sfs, w, label="SFS build", color=c_sfs)
        ax1.bar(x, tsinfer, w, bottom=sfs, label="tsinfer + trees", color=c_tsinfer)
        bottoms = [s + t for s, t in zip(sfs, tsinfer)]
        ax1.bar(x, fwd, w, bottom=bottoms, label="Forward pass", color=c_fwd)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylabel("Time (ms)")
        ax1.set_title("a  Component breakdown", fontweight="bold",
                       loc="left", fontsize=8, pad=4)
        ax1.legend(loc="upper left", fontsize=6)
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)

        # Panel b: forward pass only (the Mamba scaling story)
        fwd_stds = [r["fwd_std"] for r in results]
        bars = ax2.bar(labels, fwd, yerr=fwd_stds, color=c_fwd,
                       width=0.45, capsize=4, edgecolor="none")
        ratio_fwd = fwd[1] / fwd[0]
        ax2.annotate(f"{ratio_fwd:.1f}×", xy=(1, fwd[1]),
                     xytext=(1, fwd[1] * 1.18),
                     ha="center", fontsize=7, fontweight="bold")
        ax2.set_ylabel("Forward pass time (ms)")
        ax2.set_title("b  Model forward pass (GPU)", fontweight="bold",
                       loc="left", fontsize=8, pad=4)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        fig_path = out_dir / "fig6_context_scaling.png"
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Figure saved to {fig_path}")
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()
