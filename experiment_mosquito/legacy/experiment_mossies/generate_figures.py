#!/usr/bin/env python3
"""Generate evaluation figures for the AnoGam (mosquito) experiment.

Single model: tree-augmented (tsinfer).
Key differences from the main experiment:
    SEQ_LEN     = 100_000 bp  (100 kb)
    WINDOW_SIZE = 200 bp
    N_WINDOWS   = 500

Usage:
    python generate_figures.py \
        --checkpoint tsinfer_logs/lightning_logs/version_0/checkpoints/epoch=29-step=*.ckpt \
        --sims-dir sims \
        --out-dir paper_figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.serialization

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from fastcxt.config import FastCxtConfig, TrainingConfig
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_ts
from fastcxt.preprocess import windowed_tmrca

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
WINDOW_SIZE = 200
SEQ_LEN = 100_000
N_WINDOWS = SEQ_LEN // WINDOW_SIZE  # 500
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
    """Find a test tree sequence, preferring n50."""
    import tskit
    ts_files = sorted(Path(sims_dir).rglob("*.trees"))
    if not ts_files:
        raise FileNotFoundError(f"No .trees files under {sims_dir}")
    for f in ts_files:
        if "n50" in str(f):
            return tskit.load(str(f))
    return tskit.load(str(ts_files[0]))


def _run_inference(model, ts, pivot_pairs):
    """Run tsinfer-augmented inference and return (pred_means, pred_vars)."""
    blocks = [(0, int(ts.sequence_length))]
    pred_means, pred_vars, _ = translate_from_ts(
        ts, model,
        blocks=blocks,
        pivot_pairs=pivot_pairs,
        mutation_rate=MUTATION_RATE,
        device=DEVICE,
        progress=False,
        use_tsinfer=True,
    )
    return pred_means, pred_vars


def _true_log_tmrca(ts, pivot_pairs):
    true_log = []
    for sa, sb in pivot_pairs:
        raw = windowed_tmrca(ts, sa, sb, WINDOW_SIZE, int(ts.sequence_length))
        true_log.append(np.log(np.clip(raw, 1e-10, None)))
    return np.array(true_log)


# ── Figure: Scatter (true vs predicted) ──────────────────────────────────

def figure_scatter(model, ts, out_dir, pivot_pairs=[(0, 1), (2, 3), (4, 5)]):
    """Scatter plot of true vs predicted log(TMRCA)."""
    print("Generating scatter plot...")
    true_log = _true_log_tmrca(ts, pivot_pairs)
    pred_means, _ = _run_inference(model, ts, pivot_pairs)

    all_true = true_log.ravel()
    all_pred = pred_means.ravel()
    r = np.corrcoef(all_true, all_pred)[0, 1]
    rmse = np.sqrt(np.mean((all_pred - all_true) ** 2))

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.scatter(all_true, all_pred, s=3, alpha=0.25, c=ORANGE,
               edgecolors="none", rasterized=True)
    lo = min(all_true.min(), all_pred.min()) - 0.3
    hi = max(all_true.max(), all_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.6, ls="--", zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("True log(TMRCA)")
    ax.set_ylabel("Predicted log(TMRCA)")

    stats = f"r = {r:.3f}\nRMSE = {rmse:.3f}"
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, va="top",
            fontsize=6.5, family="monospace")
    ax.set_title("AnoGam — Tree-aug (tsinfer)", fontweight="bold",
                 loc="left", fontsize=8, pad=4)

    plt.tight_layout()
    fig.savefig(out_dir / "scatter_anogam.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter_anogam.png  (r={r:.3f}, RMSE={rmse:.3f})")
    return r, rmse


# ── Figure: TMRCA along genome ───────────────────────────────────────────

def figure_genome(model, ts, out_dir, pivot_pairs=[(0, 1), (2, 3), (4, 5)]):
    """TMRCA traces along the genome for several pairs."""
    print("Generating genome TMRCA plot...")
    true_log = _true_log_tmrca(ts, pivot_pairs)
    pred_means, pred_vars = _run_inference(model, ts, pivot_pairs)

    n_windows = true_log.shape[1]
    pos_kb = (np.arange(n_windows) + 0.5) * WINDOW_SIZE / 1e3

    fig, axes = plt.subplots(len(pivot_pairs), 1,
                             figsize=(7.2, 1.8 * len(pivot_pairs)),
                             sharex=True)
    if len(pivot_pairs) == 1:
        axes = [axes]

    for i, (sa, sb) in enumerate(pivot_pairs):
        ax = axes[i]
        ax.step(pos_kb, true_log[i], where="mid", color=RED, lw=0.7,
                alpha=0.85, label="True", zorder=10)

        pred_std = np.sqrt(pred_vars[i])
        ax.fill_between(pos_kb,
                        pred_means[i] - pred_std,
                        pred_means[i] + pred_std,
                        alpha=0.10, color=ORANGE, lw=0)
        ax.plot(pos_kb, pred_means[i], color=ORANGE, lw=0.8, alpha=0.8,
                label="Tree-aug (tsinfer)")

        ax.set_ylabel("log(TMRCA)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(f"{'abc'[i]}  Pair ({sa}, {sb})", fontweight="bold",
                     loc="left", fontsize=8, pad=3)
        if i == 0:
            ax.legend(loc="upper right", ncol=2, fontsize=6.5,
                      handlelength=1.5, columnspacing=1.0)

    axes[-1].set_xlabel("Genomic position (kb)")
    plt.tight_layout(h_pad=0.6)
    fig.savefig(out_dir / "genome_anogam.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved genome_anogam.png")


# ── Figure: Residuals ────────────────────────────────────────────────────

def figure_residuals(model, ts, out_dir, pivot_pairs=[(0, 1), (2, 3), (4, 5)]):
    """Residual plot + histogram."""
    print("Generating residuals plot...")
    true_log = _true_log_tmrca(ts, pivot_pairs)
    pred_means, _ = _run_inference(model, ts, pivot_pairs)

    n_windows = true_log.shape[1]
    pos_kb = (np.arange(n_windows) + 0.5) * WINDOW_SIZE / 1e3
    n_pairs = len(pivot_pairs)

    resid = pred_means.ravel() - true_log.ravel()

    fig, (ax_res, ax_hist) = plt.subplots(
        1, 2, figsize=(7.2, 2.4),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.03},
    )

    ax_res.scatter(
        np.tile(pos_kb, n_pairs), resid,
        s=14, alpha=0.45, c=ORANGE, edgecolors="none", rasterized=True,
    )
    ax_res.axhline(0, color=GREY, ls="--", lw=0.5)
    ax_res.set_xlabel("Genomic position (kb)")
    ax_res.set_ylabel("Residual")
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["right"].set_visible(False)
    ax_res.set_title("a  AnoGam — Tree-aug (tsinfer)", fontweight="bold",
                     loc="left", fontsize=8, pad=3)

    bins_r = np.linspace(-3, 3, 50)
    ax_hist.hist(resid, bins=bins_r, orientation="horizontal",
                 color=ORANGE, alpha=0.5, edgecolor="none")
    ax_hist.axhline(0, color=GREY, ls="--", lw=0.5)
    ax_hist.tick_params(labelleft=False)
    ax_hist.set_ylim(ax_res.get_ylim())
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    stats = f"$\\mu$={resid.mean():.3f}\n$\\sigma$={resid.std():.3f}"
    ax_hist.text(0.92, 0.95, stats, transform=ax_hist.transAxes,
                 va="top", ha="right", fontsize=6, family="monospace")
    ax_hist.set_title("b", fontweight="bold", loc="left", fontsize=8, pad=3)
    ax_hist.set_xlabel("Count")

    plt.tight_layout()
    fig.savefig(out_dir / "residuals_anogam.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved residuals_anogam.png  (mu={resid.mean():.3f}, sigma={resid.std():.3f})")


# ── Training curve ───────────────────────────────────────────────────────

def figure_training(metrics_path, out_dir):
    """Training curve for the tsinfer model."""
    if metrics_path is None or not Path(metrics_path).exists():
        print("  Skipping training curve (no metrics.csv)")
        return
    print("Generating training curve...")

    epochs, val_loss, val_rmse = [], [], []
    with open(metrics_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("val_rmse"):
                epochs.append(int(row["epoch"]))
                val_loss.append(float(row["val_loss"]))
                val_rmse.append(float(row["val_rmse"]))

    if not val_rmse:
        print("  No validation data in metrics.csv")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6))

    ax1.plot(epochs, val_loss, color=ORANGE, lw=1.0, alpha=0.85)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Validation loss (Beta-NLL)")
    ax1.set_title("a", fontweight="bold", loc="left", fontsize=9, pad=3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2.plot(epochs, val_rmse, color=ORANGE, lw=1.0, alpha=0.85)
    ax2.annotate(f"{val_rmse[-1]:.3f}", xy=(epochs[-1], val_rmse[-1]),
                 textcoords="offset points", xytext=(4, 3), fontsize=6,
                 color=ORANGE)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation RMSE")
    ax2.set_title("b", fontweight="bold", loc="left", fontsize=9, pad=3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(w_pad=1.0)
    fig.savefig(out_dir / "training_anogam.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training_anogam.png")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate AnoGam experiment evaluation figures",
    )
    ap.add_argument("--checkpoint", type=str, required=True,
                    help="Path to tsinfer-augmented model checkpoint")
    ap.add_argument("--sims-dir", type=str, default=None,
                    help="Directory with .trees files (prefers n50)")
    ap.add_argument("--out-dir", type=str, default="paper_figures")
    ap.add_argument("--skip-scatter", action="store_true")
    ap.add_argument("--skip-genome", action="store_true")
    ap.add_argument("--skip-residuals", action="store_true")
    ap.add_argument("--skip-training", action="store_true")
    args = ap.parse_args()

    _setup_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint)

    if args.sims_dir:
        print(f"Loading test tree sequence from {args.sims_dir}")
        ts = find_test_ts(args.sims_dir)
    else:
        print("No sims-dir provided; simulating AnoGam test data...")
        from fastcxt.mosquito import simulate_anogam
        ts = simulate_anogam(seed=123, n_samples=50, segment_length=SEQ_LEN)

    seq_len = int(ts.sequence_length)
    print(f"  Tree sequence: {ts.num_samples} samples, {ts.num_trees} trees, "
          f"{seq_len} bp, {ts.num_sites} sites")

    pivot_pairs = [(0, 1), (2, 3), (4, 5)]
    pivot_pairs = [(a, b) for a, b in pivot_pairs
                   if a < ts.num_samples and b < ts.num_samples]

    r, rmse = None, None
    if not args.skip_scatter:
        r, rmse = figure_scatter(model, ts, out_dir, pivot_pairs)

    if not args.skip_genome:
        figure_genome(model, ts, out_dir, pivot_pairs)

    if not args.skip_residuals:
        figure_residuals(model, ts, out_dir, pivot_pairs)

    if not args.skip_training:
        ckpt_p = Path(args.checkpoint)
        metrics_path = None
        if "checkpoints" in str(ckpt_p):
            metrics_path = str(ckpt_p.parent.parent / "metrics.csv")
        figure_training(metrics_path, out_dir)

    # Write a summary CSV for easy paper integration
    if r is not None:
        summary = out_dir / "summary_anogam.csv"
        with open(summary, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "value"])
            w.writerow(["r", f"{r:.4f}"])
            w.writerow(["rmse", f"{rmse:.4f}"])
            w.writerow(["species", "AnoGam"])
            w.writerow(["seq_len", SEQ_LEN])
            w.writerow(["window_size", WINDOW_SIZE])
            w.writerow(["n_windows", N_WINDOWS])
        print(f"  Saved summary_anogam.csv")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
