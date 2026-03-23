#!/usr/bin/env python3
"""Generate paper figures: tsinfer evaluation + scaling benchmark (Figure 5 + Table 2).

Usage:
    CUDA_VISIBLE_DEVICES=0 python generate_paper_figures.py \
        --base-ckpt ../experiment/lightning_logs/version_1/checkpoints/epoch=29-step=20880.ckpt \
        --trees-ckpt ../experiment/tree_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt \
        --tsinfer-ckpt ../experiment/tsinfer_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt \
        --sims-dir ../experiment/sims
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torch.serialization
import msprime
import tskit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

from fastcxt.config import FastCxtConfig, TrainingConfig, PRESETS
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_ts
from fastcxt.preprocess import windowed_tmrca
from fastcxt.tree_utils import (
    FEATS_PER_NODE,
    extract_topology_features,
    lca_lookup_batch,
)
from fastcxt.node_time_model import NodeTimeModel

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 2000
SEQ_LEN = 1_000_000
N_WINDOWS = SEQ_LEN // WINDOW_SIZE
MUTATION_RATE = 1e-8

BLUE = "#2166ac"
GREEN = "#1b7837"
RED = "#b2182b"
ORANGE = "#e08214"
GREY = "#636363"
BLACK = "#252525"


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": BLACK,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_model(ckpt_path, device=DEVICE):
    lit = LitFastCxt.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model = lit.model.to(device)
    model.eval()
    return model


def find_test_ts(sims_dir):
    """Find a test tree sequence file."""
    ts_files = sorted(Path(sims_dir).rglob("*.trees"))
    if not ts_files:
        raise FileNotFoundError(f"No .trees files under {sims_dir}")
    for f in ts_files:
        if "n50" in str(f):
            return f
    return ts_files[0]


# ── Figure 2: Scatter plots (3 panels) ───────────────────────────────────

def figure2_scatter(models, ts, out_dir, pivot_pairs=[(0,1),(2,3),(4,5)]):
    """Paper Figure 2: True vs Predicted scatter for base, tree-aug, tsinfer."""
    print("Generating Figure 2 (scatter plots)...")
    blocks = [(0, int(ts.sequence_length))]
    names = ["Base (SFS-only)", "Tree-aug (true)", "Tree-aug (tsinfer)"]
    colors = [BLUE, GREEN, ORANGE]

    true_log = []
    for sa, sb in pivot_pairs:
        raw = windowed_tmrca(ts, sa, sb, WINDOW_SIZE, int(ts.sequence_length))
        true_log.append(np.log(np.clip(raw, 1e-10, None)))
    true_log = np.array(true_log)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))

    for idx, (name, model, color) in enumerate(zip(names, models, colors)):
        if model is None:
            axes[idx].text(0.5, 0.5, f"{name}\n(no checkpoint)", transform=axes[idx].transAxes,
                          ha="center", va="center", color=GREY)
            axes[idx].set_title(f"{'abc'[idx]}", fontweight="bold", loc="left")
            continue

        kw = dict(blocks=blocks, pivot_pairs=pivot_pairs,
                  mutation_rate=MUTATION_RATE, device=DEVICE, progress=False)
        if "tsinfer" in name.lower():
            kw["use_tsinfer"] = True

        pred_means, pred_vars, _ = translate_from_ts(ts, model, **kw)

        all_true = true_log.ravel()
        all_pred = pred_means.ravel()

        r = np.corrcoef(all_true, all_pred)[0, 1]
        rmse = np.sqrt(np.mean((all_pred - all_true) ** 2))

        ax = axes[idx]
        ax.scatter(all_true, all_pred, s=3, alpha=0.25, c=color,
                   edgecolors="none", rasterized=True)
        lo = min(all_true.min(), all_pred.min()) - 0.3
        hi = max(all_true.max(), all_pred.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.6, ls="--", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("True log(TMRCA)")
        if idx == 0:
            ax.set_ylabel("Predicted log(TMRCA)")

        stats = f"r = {r:.3f}\nRMSE = {rmse:.3f}"
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, va="top",
                fontsize=6.5, family="monospace")
        ax.set_title(f"{'abc'[idx]}  {name}", fontweight="bold", loc="left",
                     fontsize=8, pad=4)

    plt.tight_layout(w_pad=0.8)
    fig.savefig(out_dir / "fig2_scatter.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2_scatter.png")


# ── Figure 3: TMRCA along genome ─────────────────────────────────────────

def figure3_genome(models, ts, out_dir, pivot_pairs=[(0,1),(2,3),(4,5)]):
    """Paper Figure 3: TMRCA traces for 3 pairs, overlaying all models."""
    print("Generating Figure 3 (TMRCA along genome)...")
    blocks = [(0, int(ts.sequence_length))]
    model_names = ["Base (SFS-only)", "Tree-aug (true)", "Tree-aug (tsinfer)"]
    model_colors = [BLUE, GREEN, ORANGE]

    true_log = []
    for sa, sb in pivot_pairs:
        raw = windowed_tmrca(ts, sa, sb, WINDOW_SIZE, int(ts.sequence_length))
        true_log.append(np.log(np.clip(raw, 1e-10, None)))
    true_log = np.array(true_log)

    n_windows = true_log.shape[1]
    pos_mb = (np.arange(n_windows) + 0.5) * WINDOW_SIZE / 1e6

    all_preds = []
    all_vars = []
    for i, (name, model) in enumerate(zip(model_names, models)):
        if model is None:
            all_preds.append(None)
            all_vars.append(None)
            continue
        kw = dict(blocks=blocks, pivot_pairs=pivot_pairs,
                  mutation_rate=MUTATION_RATE, device=DEVICE, progress=False)
        if "tsinfer" in name.lower():
            kw["use_tsinfer"] = True
        pm, pv, _ = translate_from_ts(ts, model, **kw)
        all_preds.append(pm)
        all_vars.append(pv)

    fig, axes = plt.subplots(len(pivot_pairs), 1,
                             figsize=(7.2, 1.8 * len(pivot_pairs)),
                             sharex=True)
    if len(pivot_pairs) == 1:
        axes = [axes]

    for i, (sa, sb) in enumerate(pivot_pairs):
        ax = axes[i]
        ax.step(pos_mb, true_log[i], where="mid", color=RED, lw=0.7, alpha=0.85,
                label="True", zorder=10)

        for j, (name, color) in enumerate(zip(model_names, model_colors)):
            if all_preds[j] is None:
                continue
            pred_std = np.sqrt(all_vars[j][i])
            ax.fill_between(pos_mb,
                            all_preds[j][i] - pred_std,
                            all_preds[j][i] + pred_std,
                            alpha=0.10, color=color, lw=0)
            ax.plot(pos_mb, all_preds[j][i], color=color, lw=0.8, alpha=0.8,
                    label=name)

        ax.set_ylabel("log(TMRCA)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        label = "abc"[i]
        ax.set_title(f"{label}  Pair ({sa}, {sb})", fontweight="bold", loc="left",
                     fontsize=8, pad=3)
        if i == 0:
            ax.legend(loc="upper right", ncol=4, fontsize=6.5,
                      handlelength=1.5, columnspacing=1.0)

    axes[-1].set_xlabel("Genomic position (Mb)")
    axes[-1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    plt.tight_layout(h_pad=0.6)
    fig.savefig(out_dir / "fig3_tmrca_genome.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_tmrca_genome.png")


# ── Figure 5: Scaling benchmark ──────────────────────────────────────────

def simulate_ts(n_samples, seed=42):
    ts = msprime.sim_ancestry(
        samples=n_samples, sequence_length=SEQ_LEN,
        recombination_rate=1e-8, population_size=20_000, random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=MUTATION_RATE, random_seed=seed)
    return ts


def bench_pairwise(model, ts, n_pairs, repeats=3):
    blocks = [(0, int(ts.sequence_length))]
    all_pairs = list(combinations(range(ts.num_samples), 2))[:n_pairs]
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        translate_from_ts(ts, model, blocks=blocks, pivot_pairs=list(all_pairs),
                          mutation_rate=MUTATION_RATE, device=DEVICE, progress=False)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench_node_time_only(node_model, ts, tree_feats_t, log_mu_t, repeats=3):
    all_pairs = list(combinations(range(ts.num_samples), 2))
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            node_times_pred = node_model(tree_feats_t, log_mu_t)
        node_times_np = node_times_pred[0].cpu().numpy()
        _ = lca_lookup_batch(ts, node_times_np, all_pairs,
                             n_windows=N_WINDOWS, window_size=WINDOW_SIZE)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def bench_tsinfer_pipeline(ts, repeats=2):
    """Time tsinfer.infer + feature extraction (CPU only)."""
    import tsinfer
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        samples = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        _ = tsinfer.infer(samples)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


def figure5_scaling(base_model, node_weights_path, max_samples, out_dir):
    """Paper Figure 5: scaling comparison and Table 2."""
    print("Generating Figure 5 (scaling benchmark)...")

    sample_sizes = [30, 50, 75, 100]
    max_internal = max_samples - 1
    tree_feat_dim = max_internal * FEATS_PER_NODE

    node_model = NodeTimeModel(
        tree_feat_dim=tree_feat_dim, n_internal=max_internal,
        d_model=256, n_layers=4, n_windows=N_WINDOWS,
    ).to(DEVICE)
    if node_weights_path and Path(node_weights_path).exists():
        node_model.load_state_dict(
            torch.load(node_weights_path, map_location=DEVICE, weights_only=True))
        print(f"  Loaded node-time weights from {node_weights_path}")
    else:
        print(f"  Using random node-time weights (timing only)")
    node_model.eval()

    log_mu_t = torch.tensor([[np.log(MUTATION_RATE)]], dtype=torch.float32).to(DEVICE)

    results = []
    for n in sample_sizes:
        n_pairs = n * (n - 1) // 2
        print(f"  n={n} ({n_pairs} pairs)...")

        ts = simulate_ts(n)
        tree_feats_np = extract_topology_features(
            ts, n_windows=N_WINDOWS, window_size=WINDOW_SIZE, max_internal=max_internal)
        tree_feats_t = torch.as_tensor(
            tree_feats_np[np.newaxis], dtype=torch.float32).to(DEVICE)

        # Warmup
        with torch.inference_mode():
            _ = node_model(tree_feats_t, log_mu_t)
        translate_from_ts(ts, base_model, blocks=[(0, SEQ_LEN)], pivot_pairs=[(0,1)],
                          mutation_rate=MUTATION_RATE, device=DEVICE, progress=False)

        t_pair = bench_pairwise(base_model, ts, n_pairs, repeats=2 if n >= 75 else 3)
        t_node = bench_node_time_only(node_model, ts, tree_feats_t, log_mu_t, repeats=3)

        try:
            t_tsinfer = bench_tsinfer_pipeline(ts, repeats=2)
        except Exception as e:
            print(f"    tsinfer failed: {e}")
            t_tsinfer = float("nan")

        t_full = t_tsinfer + t_node if not np.isnan(t_tsinfer) else float("nan")
        speedup = t_pair / t_node if t_node > 0 else float("inf")

        results.append({
            "n": n, "pairs": n_pairs,
            "pairwise": t_pair, "node_time": t_node,
            "tsinfer": t_tsinfer, "full_pipeline": t_full,
            "speedup": speedup,
        })
        print(f"    pairwise={t_pair:.2f}s  node={t_node:.3f}s  "
              f"tsinfer={t_tsinfer:.2f}s  speedup={speedup:.0f}x")

    # ── Plot Figure 5a: inference time vs n ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))

    ns = [r["n"] for r in results]
    t_pw = [r["pairwise"] for r in results]
    t_nt = [r["node_time"] for r in results]
    t_ts = [r["tsinfer"] for r in results]
    t_fp = [r["full_pipeline"] for r in results]
    speedups = [r["speedup"] for r in results]

    ax1.plot(ns, t_pw, "o-", color=BLUE, lw=1.2, markersize=4, label="Pairwise", zorder=10)
    ax1.plot(ns, t_nt, "s-", color=GREEN, lw=1.2, markersize=4, label="Node-time", zorder=10)
    ax1.plot(ns, t_ts, "^-", color=GREY, lw=0.9, markersize=3.5, label="tsinfer", alpha=0.8)
    valid_fp = [(n, t) for n, t in zip(ns, t_fp) if not np.isnan(t)]
    if valid_fp:
        ax1.plot([x[0] for x in valid_fp], [x[1] for x in valid_fp],
                 "D--", color=ORANGE, lw=0.9, markersize=3.5, label="Full pipeline", alpha=0.9)

    ax1.set_xlabel("Number of samples ($n$)")
    ax1.set_ylabel("Inference time (s)")
    ax1.set_title("a", fontweight="bold", loc="left", fontsize=9, pad=3)
    ax1.legend(loc="upper left", fontsize=6.5, handlelength=1.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bars = ax2.bar(range(len(ns)), speedups, color=GREEN, alpha=0.85, width=0.55,
                   edgecolor="white", linewidth=0.3)
    ax2.set_xticks(range(len(ns)))
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.set_xlabel("Number of samples ($n$)")
    ax2.set_ylabel("Speedup (node-time vs pairwise)")
    ax2.set_title("b", fontweight="bold", loc="left", fontsize=9, pad=3)

    for bar, s in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{s:.0f}$\\times$", ha="center", va="bottom", fontsize=7,
                 fontweight="bold")

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=1.2)
    fig.savefig(out_dir / "fig5_scaling.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig5_scaling.png")

    # ── Table 2: print + save CSV ──
    print("\n" + "=" * 90)
    print("Table 2: Benchmark results")
    print("=" * 90)
    print(f"{'n':>5} {'Pairs':>7} {'Pairwise (s)':>14} {'Node-time (s)':>14} "
          f"{'tsinfer (s)':>12} {'Full pipe (s)':>14} {'Speedup':>8}")
    print("-" * 90)
    for r in results:
        print(f"{r['n']:5d} {r['pairs']:7d} {r['pairwise']:14.2f} {r['node_time']:14.3f} "
              f"{r['tsinfer']:12.2f} {r['full_pipeline']:14.2f} {r['speedup']:7.0f}×")

    csv_path = out_dir / "table2_benchmark.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n", "pairs", "pairwise", "node_time",
                                          "tsinfer", "full_pipeline", "speedup"])
        w.writeheader()
        for r in results:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in r.items()})
    print(f"  Saved {csv_path}")


# ── Training curves (supplementary Fig S1) ────────────────────────────────

def figure_s1_training(metrics_paths, out_dir):
    """Supplementary Figure S1: training curves for all 3 models."""
    print("Generating Figure S1 (training curves)...")
    names = ["Base (SFS-only)", "Tree-aug (true)", "Tree-aug (tsinfer)"]
    colors = [BLUE, GREEN, ORANGE]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))

    for name, mpath, color in zip(names, metrics_paths, colors):
        if mpath is None or not Path(mpath).exists():
            continue
        epochs, val_loss, val_rmse = [], [], []
        with open(mpath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["val_rmse"]:
                    epochs.append(int(row["epoch"]))
                    val_loss.append(float(row["val_loss"]))
                    val_rmse.append(float(row["val_rmse"]))
        if val_rmse:
            ax1.plot(epochs, val_loss, color=color, lw=1.0,
                     label=name, alpha=0.85)
            ax2.plot(epochs, val_rmse, color=color, lw=1.0,
                     label=name, alpha=0.85)
            ax2.annotate(f"{val_rmse[-1]:.3f}", xy=(epochs[-1], val_rmse[-1]),
                         textcoords="offset points", xytext=(4, 3), fontsize=6, color=color)

    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation loss (Beta-NLL)")
    ax1.set_title("a", fontweight="bold", loc="left", fontsize=9, pad=3)
    ax1.legend(loc="upper right", fontsize=6.5)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation RMSE")
    ax2.set_title("b", fontweight="bold", loc="left", fontsize=9, pad=3)
    ax2.legend(loc="upper right", fontsize=6.5)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(out_dir / "figS1_training_curves.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figS1_training_curves.png")


def figure4_ood(model, out_dir, n_samples=50, n_sims=5,
                pivot_pairs=[(0,1),(2,3),(4,5)]):
    """Paper Figure 4: OOD robustness heatmaps (bias + MSE)."""
    print("Generating Figure 4 (OOD robustness)...")

    ne_base = 20_000
    mu_base = MUTATION_RATE
    grid = np.logspace(-1, 1, 11, base=2)

    bias_grid = np.full((len(grid), len(grid)), np.nan)
    mse_grid = np.full((len(grid), len(grid)), np.nan)

    for i, mu_scale in enumerate(grid):
        for j, ne_scale in enumerate(grid):
            ne = int(ne_base * ne_scale)
            mu = mu_base * mu_scale
            biases, mses = [], []
            for seed in range(n_sims):
                ts = msprime.sim_ancestry(
                    samples=n_samples, sequence_length=SEQ_LEN,
                    recombination_rate=1e-8, population_size=ne,
                    random_seed=seed + 1000 * i + j,
                )
                ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed + 2000 * i + j)
                if ts.num_sites < 10:
                    continue
                blocks = [(0, int(ts.sequence_length))]
                try:
                    pred_means, _, _ = translate_from_ts(
                        ts, model, blocks=blocks, pivot_pairs=pivot_pairs,
                        mutation_rate=mu, device=DEVICE, progress=False)
                except Exception:
                    continue
                true_log = []
                for sa, sb in pivot_pairs:
                    raw = windowed_tmrca(ts, sa, sb, WINDOW_SIZE, int(ts.sequence_length))
                    true_log.append(np.log(np.clip(raw, 1e-10, None)))
                true_log = np.array(true_log)
                diff = pred_means.ravel() - true_log.ravel()
                biases.append(np.mean(diff))
                mses.append(np.mean(diff ** 2))
            if biases:
                bias_grid[i, j] = np.mean(biases)
                mse_grid[i, j] = np.mean(mses)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    extent = [np.log2(grid[0]), np.log2(grid[-1]),
              np.log2(grid[0]), np.log2(grid[-1])]

    vmax_bias = max(abs(np.nanmin(bias_grid)), abs(np.nanmax(bias_grid)))
    im1 = ax1.imshow(bias_grid, origin='lower', extent=extent, aspect='auto',
                     cmap='RdBu_r', vmin=-vmax_bias, vmax=vmax_bias,
                     interpolation='nearest')
    ax1.set_xlabel("$N_e$ scaling (log$_2$)")
    ax1.set_ylabel("$\\mu$ scaling (log$_2$)")
    ax1.set_title("a  Bias", fontweight="bold", loc="left", fontsize=9, pad=3)
    center_i = len(grid) // 2
    center_x = np.log2(grid[center_i])
    ax1.plot(center_x, center_x, 'o', color=GREEN, markersize=5, zorder=10,
             markeredgecolor="white", markeredgewidth=0.5)
    cb1 = plt.colorbar(im1, ax=ax1, shrink=0.85, pad=0.02)
    cb1.ax.tick_params(labelsize=6)

    im2 = ax2.imshow(mse_grid, origin='lower', extent=extent, aspect='auto',
                     cmap='YlOrRd', vmin=0, interpolation='nearest')
    ax2.set_xlabel("$N_e$ scaling (log$_2$)")
    ax2.set_ylabel("$\\mu$ scaling (log$_2$)")
    ax2.set_title("b  MSE", fontweight="bold", loc="left", fontsize=9, pad=3)
    ax2.plot(center_x, center_x, 'o', color=GREEN, markersize=5, zorder=10,
             markeredgecolor="white", markeredgewidth=0.5)
    cb2 = plt.colorbar(im2, ax=ax2, shrink=0.85, pad=0.02)
    cb2.ax.tick_params(labelsize=6)

    plt.tight_layout(w_pad=0.8)
    fig.savefig(out_dir / "fig4_ood.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4_ood.png")


def figure_s2_residuals(models, ts, out_dir, pivot_pairs=[(0,1),(2,3),(4,5)]):
    """Supplementary Figure S2: Residuals for all 3 models."""
    print("Generating Figure S2 (residuals)...")
    blocks = [(0, int(ts.sequence_length))]
    model_names = ["Base (SFS-only)", "Tree-aug (true)", "Tree-aug (tsinfer)"]
    model_colors = [BLUE, GREEN, ORANGE]

    true_log = []
    for sa, sb in pivot_pairs:
        raw = windowed_tmrca(ts, sa, sb, WINDOW_SIZE, int(ts.sequence_length))
        true_log.append(np.log(np.clip(raw, 1e-10, None)))
    true_log = np.array(true_log)

    n_windows = true_log.shape[1]
    pos_mb = (np.arange(n_windows) + 0.5) * WINDOW_SIZE / 1e6
    n_pairs = len(pivot_pairs)

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.4),
                             gridspec_kw={"width_ratios": [3, 1], "wspace": 0.03})

    for idx, (name, model, color) in enumerate(zip(model_names, models, model_colors)):
        ax_res = axes[idx, 0]
        ax_hist = axes[idx, 1]

        if model is None:
            ax_res.text(0.5, 0.5, f"{name}\n(no checkpoint)",
                       transform=ax_res.transAxes, ha="center", va="center")
            continue

        kw = dict(blocks=blocks, pivot_pairs=pivot_pairs,
                  mutation_rate=MUTATION_RATE, device=DEVICE, progress=False)
        if "tsinfer" in name.lower():
            kw["use_tsinfer"] = True
        pred_means, _, _ = translate_from_ts(ts, model, **kw)

        resid = pred_means.ravel() - true_log.ravel()
        ax_res.scatter(
            np.tile(pos_mb, n_pairs), resid,
            s=3, alpha=0.25, c=color, edgecolors="none", rasterized=True,
        )
        ax_res.axhline(0, color=GREY, ls="--", lw=0.5)
        ax_res.set_ylabel("Residual")
        ax_res.spines["top"].set_visible(False)
        ax_res.spines["right"].set_visible(False)
        label = "ace"[idx]
        ax_res.set_title(f"{label}  {name}", fontweight="bold", loc="left",
                         fontsize=8, pad=3)
        if idx == 2:
            ax_res.set_xlabel("Genomic position (Mb)")

        bins_r = np.linspace(-3, 3, 50)
        ax_hist.hist(resid, bins=bins_r, orientation="horizontal",
                     color=color, alpha=0.5, edgecolor="none")
        ax_hist.axhline(0, color=GREY, ls="--", lw=0.5)
        ax_hist.tick_params(labelleft=False)
        ax_hist.set_ylim(ax_res.get_ylim())
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        label_h = "bdf"[idx]
        stats = f"$\\mu$={resid.mean():.3f}\n$\\sigma$={resid.std():.3f}"
        ax_hist.text(0.92, 0.95, stats, transform=ax_hist.transAxes,
                     va="top", ha="right", fontsize=6, family="monospace")
        ax_hist.set_title(f"{label_h}", fontweight="bold", loc="left",
                          fontsize=8, pad=3)
        if idx == 2:
            ax_hist.set_xlabel("Count")

    plt.tight_layout(h_pad=0.6)
    fig.savefig(out_dir / "figS2_residuals.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figS2_residuals.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", type=str, default=None)
    ap.add_argument("--trees-ckpt", type=str, default=None)
    ap.add_argument("--tsinfer-ckpt", type=str, default=None)
    ap.add_argument("--node-weights", type=str, default=None)
    ap.add_argument("--sims-dir", type=str, default=None)
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--out-dir", type=str, default="paper_figures")
    ap.add_argument("--skip-scatter", action="store_true")
    ap.add_argument("--skip-genome", action="store_true")
    ap.add_argument("--skip-scaling", action="store_true")
    ap.add_argument("--skip-training-curves", action="store_true")
    ap.add_argument("--skip-ood", action="store_true")
    ap.add_argument("--skip-residuals", action="store_true")
    args = ap.parse_args()

    _setup_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model = load_model(args.base_ckpt) if args.base_ckpt else None
    trees_model = load_model(args.trees_ckpt) if args.trees_ckpt else None
    tsinfer_model = load_model(args.tsinfer_ckpt) if args.tsinfer_ckpt else None
    models = [base_model, trees_model, tsinfer_model]

    if args.sims_dir:
        ts_file = find_test_ts(args.sims_dir)
        print(f"Using tree sequence: {ts_file}")
        ts = tskit.load(str(ts_file))
    else:
        print("No sims-dir, simulating fresh test data...")
        ts = simulate_ts(50, seed=123)

    if not args.skip_scatter:
        figure2_scatter(models, ts, out_dir)

    if not args.skip_genome:
        figure3_genome(models, ts, out_dir)

    if not args.skip_scaling and base_model is not None:
        figure5_scaling(base_model, args.node_weights, args.max_samples, out_dir)

    if not args.skip_training_curves:
        def _metrics_path(ckpt):
            if ckpt is None:
                return None
            p = Path(ckpt)
            if "checkpoints" in str(p):
                return str(p.parent.parent / "metrics.csv")
            return None
        figure_s1_training(
            [_metrics_path(args.base_ckpt),
             _metrics_path(args.trees_ckpt),
             _metrics_path(args.tsinfer_ckpt)],
            out_dir,
        )

    if not args.skip_ood and base_model is not None:
        figure4_ood(base_model, out_dir)

    if not args.skip_residuals:
        figure_s2_residuals(models, ts, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
