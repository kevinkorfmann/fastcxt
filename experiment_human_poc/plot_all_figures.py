#!/usr/bin/env python3
"""Generate all figures for the human PoC experiment.

Produces:
  figures/training_curves.png       — pairwise model loss + RMSE over epochs
  figures/pairwise/scatter.png      — true vs predicted scatter (already from evaluate)
  figures/pairwise/genome_profile.png
  figures/pairwise/residuals.png
  figures/node_time_true/scatter.png       — node-time (true trees) evaluation
  figures/node_time_true/genome_profile.png
  figures/node_time_tsinfer/scatter.png    — node-time (tsinfer trees) evaluation
  figures/node_time_tsinfer/genome_profile.png
  figures/benchmark_scaling.png     — pairwise vs node-time speed comparison
  figures/model_comparison.png      — side-by-side R² comparison across models

Usage:
    python plot_all_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import tskit

from fastcxt.config import FastCxtConfig, TrainingConfig
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_ts
from fastcxt.preprocess import windowed_tmrca
from fastcxt.node_time_model import NodeTimeModel
from fastcxt.tree_utils import FEATS_PER_NODE

from config import (
    SIMS_DIR, FIGURES_DIR, OUTPUTS_DIR, WINDOW_SIZE, MUTATION_RATE,
    SEQ_LEN, MAX_SAMPLES, N_WINDOWS, NODE_D_MODEL, NODE_N_LAYERS,
    BLUE, RED, GREEN, ORANGE, GREY, BLACK,
)

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

PIVOT_PAIRS = [(0, 1), (2, 3), (4, 5)]
N_LCA_PAIRS = 50  # pairs for LCA evaluation (reduces discretization)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": BLACK, "axes.linewidth": 0.6, "axes.grid": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "legend.frameon": False,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "figure.dpi": 300, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def load_pairwise_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["state_dict"] = {k.replace("._orig_mod", ""): v for k, v in ckpt["state_dict"].items()}
    lit = LitFastCxt(**ckpt["hyper_parameters"])
    lit.load_state_dict(ckpt["state_dict"])
    return lit.model


def plot_training_curves(metrics_csv, out_dir):
    """Plot training loss and RMSE over epochs."""
    import pandas as pd
    df = pd.read_csv(metrics_csv)

    # Separate train and val rows
    train = df[df["train_loss"].notna()].copy()
    val = df[df["val_loss"].notna()].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    # Loss
    ax1.plot(train["step"], train["train_loss"], color=BLUE, lw=0.5, alpha=0.3, label="Train (per step)")
    # Epoch averages
    train_epoch = train.groupby("epoch").agg({"train_loss": "mean", "step": "max"}).reset_index()
    ax1.plot(train_epoch["step"], train_epoch["train_loss"], color=BLUE, lw=1.5, label="Train (epoch avg)")
    ax1.scatter(val["step"], val["val_loss"], color=RED, s=20, zorder=5, label="Validation")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.set_yscale("symlog", linthresh=0.1)

    # RMSE
    ax2.plot(train["step"], train["train_rmse"], color=BLUE, lw=0.5, alpha=0.3)
    train_epoch_rmse = train.groupby("epoch").agg({"train_rmse": "mean", "step": "max"}).reset_index()
    ax2.plot(train_epoch_rmse["step"], train_epoch_rmse["train_rmse"], color=BLUE, lw=1.5, label="Train (epoch avg)")
    ax2.scatter(val["step"], val["val_rmse"], color=RED, s=20, zorder=5, label="Validation")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("RMSE")
    ax2.set_title("RMSE")
    ax2.legend()

    out_path = out_dir / "training_curves.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")

    # Also plot R² / coverage
    if "val_coverage_95" in val.columns and val["val_coverage_95"].notna().any():
        fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
        ax.scatter(val["step"], val["val_coverage_95"], color=GREEN, s=20)
        ax.set_xlabel("Step")
        ax.set_ylabel("95% Coverage")
        ax.set_title("Validation Coverage (95% CI)")
        ax.axhline(0.95, color=GREY, ls="--", lw=0.8, label="Target")
        ax.legend()
        out_path = out_dir / "val_coverage.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_path}")


def evaluate_and_plot(model, ts, name, out_dir, color=BLUE):
    """Run inference and create scatter + genome profile for a model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_len = int(ts.sequence_length)

    pred_means, pred_vars, _ = translate_from_ts(
        ts, model, blocks=[(0, seq_len)], pivot_pairs=PIVOT_PAIRS,
        mutation_rate=MUTATION_RATE, device=DEVICE,
    )

    true_log = np.array([
        np.log(np.clip(windowed_tmrca(ts, a, b, WINDOW_SIZE, seq_len), 1e-10, None))
        for a, b in PIVOT_PAIRS
    ])

    n_windows = pred_means.shape[1]
    pos_mb = (np.arange(n_windows) + 0.5) * WINDOW_SIZE / 1e6
    all_true, all_pred = true_log.ravel(), pred_means.ravel()
    rmse = np.sqrt(np.mean((all_pred - all_true) ** 2))
    mae = np.mean(np.abs(all_pred - all_true))
    corr = np.corrcoef(all_true, all_pred)[0, 1]

    # Scatter
    fig = plt.figure(figsize=(5.5, 5.5))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_hx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_hy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax.scatter(all_true, all_pred, s=10, alpha=0.35, c=color, edgecolors="none", rasterized=True)
    lo = min(all_true.min(), all_pred.min()) - 0.3
    hi = max(all_true.max(), all_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color=GREY, lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True log(TMRCA)"); ax.set_ylabel("Predicted log(TMRCA)")
    ax.set_title(name)
    ax.text(0.04, 0.96, f"r = {corr:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace",
            bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.4"))
    bins = np.linspace(lo, hi, 40)
    ax_hx.hist(all_true, bins=bins, color=RED, alpha=0.5, edgecolor="none")
    ax_hx.hist(all_pred, bins=bins, color=color, alpha=0.5, edgecolor="none")
    ax_hx.tick_params(labelbottom=False)
    ax_hy.hist(all_pred, bins=bins, orientation="horizontal", color=color, alpha=0.5, edgecolor="none")
    ax_hy.tick_params(labelleft=False)
    fig.savefig(out_dir / "scatter.png", bbox_inches="tight")
    plt.close(fig)

    # Genome profile
    n_pairs = len(PIVOT_PAIRS)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 2.8 * n_pairs), sharex=True, layout="constrained")
    if n_pairs == 1:
        axes = [axes]
    for i, (sa, sb) in enumerate(PIVOT_PAIRS):
        ax = axes[i]
        std = np.sqrt(pred_vars[i])
        ax.fill_between(pos_mb, pred_means[i] - std, pred_means[i] + std, alpha=0.2, color=color)
        ax.plot(pos_mb, pred_means[i], color=color, lw=1.5, label="Predicted")
        ax.step(pos_mb, true_log[i], where="mid", color=RED, lw=0.9, alpha=0.85, label="True")
        ax.set_ylabel("log(TMRCA)")
        ax.set_title(f"Pair ({sa}, {sb})")
        if i == 0:
            ax.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Genomic position (Mb)")
    fig.suptitle(name, fontsize=11)
    fig.savefig(out_dir / "genome_profile.png", bbox_inches="tight")
    plt.close(fig)

    print(f"  {name}: r={corr:.3f}, RMSE={rmse:.3f}, MAE={mae:.3f}")
    return {"name": name, "r": corr, "rmse": rmse, "mae": mae, "lca_r": corr, "lca_rmse": rmse}


def evaluate_node_time(weights_path, features_dir, ts, name, out_dir, color=GREEN,
                       use_tsinfer=False):
    """Evaluate a NodeTimeModel and create scatter + genome plots."""
    from fastcxt.tree_utils import lca_lookup, extract_topology_features
    from fastcxt.preprocess import estimate_mutation_rate

    out_dir.mkdir(parents=True, exist_ok=True)
    seq_len = int(ts.sequence_length)
    max_internal = MAX_SAMPLES - 1

    # Load model
    model = NodeTimeModel(
        tree_feat_dim=max_internal * FEATS_PER_NODE,
        n_internal=max_internal,
        d_model=NODE_D_MODEL,
        n_layers=NODE_N_LAYERS,
        n_windows=N_WINDOWS,
    ).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # Extract features from the SAME tree sequence used for LCA lookup
    if use_tsinfer:
        import tsinfer
        sd = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
        inferred_ts = tsinfer.infer(sd)
        feats = extract_topology_features(
            inferred_ts, n_windows=N_WINDOWS,
            window_size=WINDOW_SIZE, max_internal=max_internal,
        )
    else:
        feats = extract_topology_features(
            ts, n_windows=N_WINDOWS,
            window_size=WINDOW_SIZE, max_internal=max_internal,
        )

    mu = np.log(estimate_mutation_rate(ts))

    feats_t = torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    mu_t = torch.as_tensor([[mu]], dtype=torch.float32).to(DEVICE)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred_times = model(feats_t, mu_t).float().cpu().numpy()[0]  # (n_windows, n_internal)

    pred_mean_times = pred_times  # already (n_windows, n_internal)
    n_windows = pred_mean_times.shape[0]
    pos_mb = (np.arange(n_windows) + 0.5) * WINDOW_SIZE / 1e6

    # --- Direct node-time comparison (all nodes, no LCA) ---
    from fastcxt.tree_utils import extract_node_times
    true_node_times = extract_node_times(
        ts, n_windows=N_WINDOWS, window_size=WINDOW_SIZE, max_internal=max_internal,
    )
    mask_nodes = true_node_times != 0
    direct_true = true_node_times[mask_nodes]
    direct_pred = pred_mean_times[mask_nodes]
    direct_corr = np.corrcoef(direct_true, direct_pred)[0, 1]
    direct_rmse = np.sqrt(np.mean((direct_pred - direct_true) ** 2))

    # Direct scatter
    fig = plt.figure(figsize=(5.5, 5.5))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_hx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_hy = fig.add_subplot(gs[1, 1], sharey=ax)
    # Subsample for plotting if too many points
    n_pts = len(direct_true)
    if n_pts > 10000:
        idx = np.random.default_rng(42).choice(n_pts, 10000, replace=False)
        plot_true, plot_pred = direct_true[idx], direct_pred[idx]
    else:
        plot_true, plot_pred = direct_true, direct_pred
    ax.scatter(plot_true, plot_pred, s=4, alpha=0.2, c=color, edgecolors="none", rasterized=True)
    lo = min(direct_true.min(), direct_pred.min()) - 0.3
    hi = max(direct_true.max(), direct_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color=GREY, lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True log(node time)"); ax.set_ylabel("Predicted log(node time)")
    ax.set_title(f"{name} — Direct Node Times")
    ax.text(0.04, 0.96, f"r = {direct_corr:.3f}\nRMSE = {direct_rmse:.3f}\nn = {n_pts:,}",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace",
            bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.4"))
    bins = np.linspace(lo, hi, 50)
    ax_hx.hist(direct_true, bins=bins, color=RED, alpha=0.5, edgecolor="none")
    ax_hx.hist(direct_pred, bins=bins, color=color, alpha=0.5, edgecolor="none")
    ax_hx.tick_params(labelbottom=False)
    ax_hy.hist(direct_pred, bins=bins, orientation="horizontal", color=color, alpha=0.5, edgecolor="none")
    ax_hy.tick_params(labelleft=False)
    fig.savefig(out_dir / "scatter.png", bbox_inches="tight")
    plt.close(fig)

    # --- LCA-based TMRCA (using N_LCA_PAIRS to reduce discretization) ---
    n_lca = max(N_LCA_PAIRS, len(PIVOT_PAIRS))
    lca_pairs = [(i, i + 1) for i in range(0, min(ts.num_samples, n_lca * 2), 2)]
    all_true = []
    all_pred = []
    for sa, sb in lca_pairs:
        true_tmrca = np.log(np.clip(windowed_tmrca(ts, sa, sb, WINDOW_SIZE, seq_len), 1e-10, None))
        pred_tmrca = lca_lookup(ts, pred_mean_times, sa, sb, n_windows=N_WINDOWS, window_size=WINDOW_SIZE)
        all_true.append(true_tmrca)
        all_pred.append(pred_tmrca)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    flat_true = all_true.ravel()
    flat_pred = all_pred.ravel()

    valid = np.isfinite(flat_true) & np.isfinite(flat_pred) & (flat_pred != 0)
    flat_true_v = flat_true[valid]
    flat_pred_v = flat_pred[valid]

    corr = np.corrcoef(flat_true_v, flat_pred_v)[0, 1] if len(flat_true_v) > 1 else 0
    rmse = np.sqrt(np.mean((flat_pred_v - flat_true_v) ** 2))
    mae = np.mean(np.abs(flat_pred_v - flat_true_v))

    # LCA scatter
    fig = plt.figure(figsize=(5.5, 5.5))
    gs_lca = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs_lca[1, 0])
    ax_hx = fig.add_subplot(gs_lca[0, 0], sharex=ax)
    ax_hy = fig.add_subplot(gs_lca[1, 1], sharey=ax)
    ax.scatter(flat_true_v, flat_pred_v, s=10, alpha=0.35, c=color, edgecolors="none", rasterized=True)
    lo = min(flat_true_v.min(), flat_pred_v.min()) - 0.3
    hi = max(flat_true_v.max(), flat_pred_v.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color=GREY, lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True log(TMRCA)"); ax.set_ylabel("Predicted log(TMRCA)")
    ax.set_title(f"{name} — TMRCA via LCA ({len(lca_pairs)} pairs)")
    ax.text(0.04, 0.96, f"r = {corr:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace",
            bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.4"))
    bins = np.linspace(lo, hi, 40)
    ax_hx.hist(flat_true_v, bins=bins, color=RED, alpha=0.5, edgecolor="none")
    ax_hx.hist(flat_pred_v, bins=bins, color=color, alpha=0.5, edgecolor="none")
    ax_hx.tick_params(labelbottom=False)
    ax_hy.hist(flat_pred_v, bins=bins, orientation="horizontal", color=color, alpha=0.5, edgecolor="none")
    ax_hy.tick_params(labelleft=False)
    fig.savefig(out_dir / "scatter_tmrca.png", bbox_inches="tight")
    plt.close(fig)

    # Genome profile (first 3 pairs)
    plot_pairs = lca_pairs[:3]
    fig, axes = plt.subplots(len(plot_pairs), 1, figsize=(10, 2.8 * len(plot_pairs)), sharex=True, layout="constrained")
    if len(plot_pairs) == 1:
        axes = [axes]
    for i, (sa, sb) in enumerate(plot_pairs):
        ax = axes[i]
        ax.plot(pos_mb, all_pred[i], color=color, lw=1.5, label="Predicted")
        ax.step(pos_mb, all_true[i], where="mid", color=RED, lw=0.9, alpha=0.85, label="True")
        ax.set_ylabel("log(TMRCA)")
        ax.set_title(f"Pair ({sa}, {sb})")
        if i == 0:
            ax.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Genomic position (Mb)")
    fig.suptitle(name, fontsize=11)
    fig.savefig(out_dir / "genome_profile.png", bbox_inches="tight")
    plt.close(fig)

    print(f"  {name}: LCA r={corr:.3f} RMSE={rmse:.3f} | Direct r={direct_corr:.3f} RMSE={direct_rmse:.3f}")
    return {"name": name, "r": direct_corr, "rmse": direct_rmse, "mae": float(np.mean(np.abs(direct_pred - direct_true))),
            "lca_r": corr, "lca_rmse": rmse}


def plot_benchmark(csv_path, out_dir):
    """Plot benchmark scaling results."""
    import pandas as pd
    if not csv_path.exists():
        print(f"  Benchmark CSV not found at {csv_path}, skipping")
        return

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")

    for n_samples in sorted(df["n_samples"].unique()):
        sub = df[df["n_samples"] == n_samples]
        ax.plot(sub["n_pairs"], sub["pair_ms"], "o-", label=f"Pairwise (n={n_samples})", alpha=0.7)
        ax.plot(sub["n_pairs"], sub["node_true_ms"], "s--", label=f"Node-time (n={n_samples})", alpha=0.7)

    ax.set_xlabel("Number of pairs")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Inference Speed: Pairwise vs Node-time")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=6, ncol=2)
    out_path = out_dir / "benchmark_scaling.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_model_comparison(results, out_dir):
    """Bar chart comparing TMRCA prediction across all models."""
    names = [r["name"] for r in results]
    rs = [r["lca_r"] for r in results]
    rmses = [r["lca_rmse"] for r in results]
    colors_bar = [BLUE, GREEN, ORANGE][:len(results)]

    x = np.arange(len(names))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")

    ax1.bar(x, rs, color=colors_bar, edgecolor=BLACK, linewidth=0.5)
    ax1.set_ylabel("Correlation (r)")
    ax1.set_title("TMRCA Prediction: Correlation")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=7)
    for i, v in enumerate(rs):
        ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)

    ax2.bar(x, rmses, color=colors_bar, edgecolor=BLACK, linewidth=0.5)
    ax2.set_ylabel("RMSE")
    ax2.set_title("TMRCA Prediction: RMSE")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=7)
    for i, v in enumerate(rmses):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)

    fig.savefig(out_dir / "model_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved model_comparison.png")


def main():
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Pick a test tree sequence (use n100 for better evaluation)
    ts_files = sorted(SIMS_DIR.glob("n100/*.trees"))
    if not ts_files:
        ts_files = sorted(SIMS_DIR.rglob("*.trees"))
    ts_file = ts_files[0]
    ts = tskit.load(str(ts_file))
    print(f"Using tree sequence: {ts_file} (n={ts.num_samples})")

    results = []

    # 1. Training curves
    print("\n=== Training Curves ===")
    metrics_csv = OUTPUTS_DIR / "pairwise" / "csv_metrics" / "version_0" / "metrics.csv"
    if metrics_csv.exists():
        plot_training_curves(metrics_csv, FIGURES_DIR)
    else:
        # Try version_1
        for v in sorted((OUTPUTS_DIR / "pairwise" / "csv_metrics").glob("version_*")):
            mc = v / "metrics.csv"
            if mc.exists():
                plot_training_curves(mc, FIGURES_DIR)
                break

    # 2. Pairwise model evaluation
    print("\n=== Pairwise Model ===")
    ckpt = sorted(
        (OUTPUTS_DIR / "pairwise").rglob("*.ckpt"),
        key=lambda p: int(p.stem.split("=")[1].split("-")[0])
    )[-1]
    print(f"  Checkpoint: {ckpt.name}")
    pair_model = load_pairwise_model(ckpt)
    r = evaluate_and_plot(pair_model, ts, "Pairwise (SFS)", FIGURES_DIR / "pairwise", BLUE)
    results.append(r)

    # 3. Node-time (true trees)
    print("\n=== Node-time (True Trees) ===")
    nt_true_path = OUTPUTS_DIR / "node_time_true.pt"
    if nt_true_path.exists():
        r = evaluate_node_time(
            nt_true_path, OUTPUTS_DIR / "node_features",
            ts, "Node-time (True Trees)", FIGURES_DIR / "node_time_true", GREEN,
            use_tsinfer=False,
        )
        results.append(r)

    # 4. Node-time (tsinfer trees)
    print("\n=== Node-time (Tsinfer Trees) ===")
    nt_tsinfer_path = OUTPUTS_DIR / "node_time_tsinfer.pt"
    if nt_tsinfer_path.exists():
        r = evaluate_node_time(
            nt_tsinfer_path, OUTPUTS_DIR / "node_features_tsinfer",
            ts, "Node-time (Tsinfer Trees)", FIGURES_DIR / "node_time_tsinfer", ORANGE,
            use_tsinfer=True,
        )
        results.append(r)

    # 5. Model comparison
    print("\n=== Model Comparison ===")
    if len(results) > 1:
        plot_model_comparison(results, FIGURES_DIR)

    # 6. Benchmark scaling
    print("\n=== Benchmark Scaling ===")
    plot_benchmark(OUTPUTS_DIR / "benchmark_results.csv", FIGURES_DIR)

    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
