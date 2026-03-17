#!/usr/bin/env python3
"""Generate experiment figures for the quickstart 6-epoch model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import tskit

import torch.serialization
from fastcxt.config import FastCxtConfig, TrainingConfig
from fastcxt.preprocess import windowed_tmrca
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_ts

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

EXPERIMENT_DIR = Path(__file__).resolve().parent

WINDOW_SIZE = 2000
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


def main():
    ap = argparse.ArgumentParser(description="Generate experiment figures")
    ap.add_argument("--checkpoint", type=str,
                    default=str(EXPERIMENT_DIR / "lightning_logs/version_9/checkpoints/epoch=19-step=4680.ckpt"),
                    help="Path to Lightning checkpoint")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory (default: figures/ or figures_trees/ from checkpoint)")
    ap.add_argument("--title", type=str, default=None,
                    help="Subtitle for fig 2 (e.g. 'base (20 epochs, Beta-NLL)')")
    ap.add_argument("--ts-file", type=str, default=None,
                    help="Path to .trees file for evaluation (auto-discovers if omitted)")
    ap.add_argument("--use-tsinfer", action="store_true",
                    help="Run tsinfer to get inferred topology for tree features")
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    if args.out_dir:
        fig_dir = Path(args.out_dir)
    else:
        fig_dir = EXPERIMENT_DIR / "figures"
    if args.title:
        fig_title = args.title
    else:
        version = checkpoint.parent.parent.name if "checkpoints" in str(checkpoint) else "model"
        fig_title = f"{version} (Beta-NLL)"

    _setup_style()
    fig_dir.mkdir(parents=True, exist_ok=True)
    for f in fig_dir.glob("*.png"):
        f.unlink()

    if args.ts_file:
        ts_file = Path(args.ts_file)
    else:
        sims_dir = EXPERIMENT_DIR / "sims"
        ts_files = sorted(sims_dir.rglob("*.trees"))
        if not ts_files:
            print(f"ERROR: No .trees files found under {sims_dir}")
            return
        ts_file = ts_files[0]

    print(f"Loading model and tree sequence ({ts_file.name}) ...")
    lit = LitFastCxt.load_from_checkpoint(str(checkpoint), map_location="cpu")
    model = lit.model
    ts = tskit.load(str(ts_file))
    seq_len = int(ts.sequence_length)

    pivot_pairs = [(0, 1), (2, 3), (4, 5)]
    blocks = [(0, seq_len)]

    device = "cuda:0" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Running inference on {device} ...")
    translate_kwargs = dict(
        blocks=blocks, pivot_pairs=pivot_pairs,
        mutation_rate=MUTATION_RATE, device=device,
    )
    if args.use_tsinfer:
        translate_kwargs["use_tsinfer"] = True
    pred_means, pred_vars, idx_map = translate_from_ts(
        ts, model, **translate_kwargs,
    )

    true_log = []
    for sa, sb in pivot_pairs:
        raw = windowed_tmrca(ts, sa, sb, WINDOW_SIZE, seq_len)
        true_log.append(np.log(np.clip(raw, 1e-10, None)))
    true_log = np.array(true_log)

    n_windows = pred_means.shape[1]
    pos_mb = (np.arange(n_windows) + 0.5) * WINDOW_SIZE / 1e6

    all_true = true_log.ravel()
    all_pred = pred_means.ravel()
    rmse = np.sqrt(np.mean((all_pred - all_true) ** 2))
    mae = np.mean(np.abs(all_pred - all_true))
    corr = np.corrcoef(all_true, all_pred)[0, 1]

    # ── Figure 1 : True vs Predicted scatter with marginals ──────────
    fig = plt.figure(figsize=(5.5, 5.5))
    gs = GridSpec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4],
        hspace=0.04, wspace=0.04,
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)

    ax_main.scatter(all_true, all_pred, s=10, alpha=0.35, c=BLUE,
                    edgecolors="none", rasterized=True)
    lo = min(all_true.min(), all_pred.min()) - 0.3
    hi = max(all_true.max(), all_pred.max()) + 0.3
    ax_main.plot([lo, hi], [lo, hi], "--", color=GREY, lw=1, zorder=0)
    ax_main.set_xlim(lo, hi)
    ax_main.set_ylim(lo, hi)
    ax_main.set_xlabel("True log(TMRCA)")
    ax_main.set_ylabel("Predicted log(TMRCA)")

    stats_text = f"r = {corr:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}"
    ax_main.text(
        0.04, 0.96, stats_text, transform=ax_main.transAxes,
        va="top", fontsize=9, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.4"),
    )

    bins = np.linspace(lo, hi, 40)
    ax_histx.hist(all_true, bins=bins, color=RED, alpha=0.5, edgecolor="none")
    ax_histx.hist(all_pred, bins=bins, color=BLUE, alpha=0.5, edgecolor="none")
    ax_histx.set_ylabel("Count")
    ax_histx.tick_params(labelbottom=False)

    ax_histy.hist(all_pred, bins=bins, orientation="horizontal",
                  color=BLUE, alpha=0.5, edgecolor="none")
    ax_histy.hist(all_true, bins=bins, orientation="horizontal",
                  color=RED, alpha=0.5, edgecolor="none")
    ax_histy.set_xlabel("Count")
    ax_histy.tick_params(labelleft=False)

    fig.suptitle("True vs Predicted log(TMRCA)", fontsize=12, y=0.95)
    fig.savefig(fig_dir / "01_true_vs_predicted.png", bbox_inches="tight")
    plt.close(fig)
    print("  01_true_vs_predicted.png")

    # ── Figure 2 : TMRCA along the genome ────────────────────────────
    n_pairs = len(pivot_pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 2.8 * n_pairs),
                             sharex=True, layout="constrained")
    if n_pairs == 1:
        axes = [axes]

    for i, (sa, sb) in enumerate(pivot_pairs):
        ax = axes[i]
        pred_std = np.sqrt(pred_vars[i])
        ax.fill_between(
            pos_mb,
            pred_means[i] - pred_std,
            pred_means[i] + pred_std,
            alpha=0.20, color=BLUE, label="Pred ±1 s.d.",
        )
        ax.plot(pos_mb, pred_means[i], color=BLUE, lw=1.5, label="Predicted")
        ax.step(pos_mb, true_log[i], where="mid", color=RED, lw=0.9,
                alpha=0.85, label="True")
        ax.set_ylabel("log(TMRCA)")
        ax.set_title(f"Pair ({sa}, {sb})", fontsize=10)
        if i == 0:
            ax.legend(loc="upper right", framealpha=0.9, ncol=3)

    axes[-1].set_xlabel("Genomic position (Mb)")
    axes[-1].xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    fig.suptitle(f"TMRCA along the genome  —  {fig_title}", fontsize=12)
    fig.savefig(fig_dir / "02_tmrca_along_genome.png", bbox_inches="tight")
    plt.close(fig)
    print("  02_tmrca_along_genome.png")

    # ── Figure 3 : Residuals + histogram ─────────────────────────────
    fig = plt.figure(figsize=(10, 4.5), layout="constrained")
    gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.02, figure=fig)
    ax_res = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_res)

    resid = all_pred - all_true
    ax_res.scatter(
        np.tile(pos_mb, n_pairs), resid,
        s=14, alpha=0.45, c=BLUE, edgecolors="none", rasterized=True,
    )
    ax_res.axhline(0, color=GREY, ls="--", lw=0.8)
    ax_res.set_xlabel("Genomic position (Mb)")
    ax_res.set_ylabel("Residual  (predicted − true)")
    ax_res.set_title("Prediction residuals", fontsize=11)

    bins_r = np.linspace(resid.min() - 0.2, resid.max() + 0.2, 40)
    ax_hist.hist(resid, bins=bins_r, orientation="horizontal",
                 color=BLUE, alpha=0.5, edgecolor="none")
    ax_hist.axhline(0, color=GREY, ls="--", lw=0.8)
    ax_hist.tick_params(labelleft=False)
    ax_hist.set_xlabel("Count")

    rmse_txt = f"mean = {resid.mean():.2f}\nstd  = {resid.std():.2f}"
    ax_hist.text(
        0.95, 0.95, rmse_txt, transform=ax_hist.transAxes,
        va="top", ha="right", fontsize=9, family="monospace",
        bbox=dict(facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.3"),
    )
    fig.savefig(fig_dir / "03_residuals.png", bbox_inches="tight")
    plt.close(fig)
    print("  03_residuals.png")

    # ── Figure 4 : Per-pair correlation bar chart + training curve ───
    metrics_csv = checkpoint.parent.parent / "metrics.csv"
    fig, (ax_bar, ax_curve) = plt.subplots(
        1, 2, figsize=(9, 3.5), layout="constrained",
    )

    pair_corrs = []
    pair_labels = []
    for i, (sa, sb) in enumerate(pivot_pairs):
        r = np.corrcoef(true_log[i], pred_means[i])[0, 1]
        pair_corrs.append(r)
        pair_labels.append(f"({sa},{sb})")
    bars = ax_bar.bar(pair_labels, pair_corrs, color=BLUE, width=0.5, alpha=0.8)
    ax_bar.set_ylabel("Pearson r")
    ax_bar.set_title("Per-pair correlation")
    ax_bar.set_ylim(0, max(1.0, max(pair_corrs) + 0.1))
    for b, v in zip(bars, pair_corrs):
        ax_bar.text(b.get_x() + b.get_width() / 2, v + 0.02,
                    f"{v:.3f}", ha="center", fontsize=9)

    if metrics_csv.exists():
        import csv
        epochs, val_rmses = [], []
        with open(metrics_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["val_rmse"]:
                    epochs.append(int(row["epoch"]))
                    val_rmses.append(float(row["val_rmse"]))
        if val_rmses:
            ax_curve.plot(epochs, val_rmses, "o-", color=RED, markersize=5, lw=1.5)
            ax_curve.set_xlabel("Epoch")
            ax_curve.set_ylabel("Validation RMSE")
            ax_curve.set_title("Training progress")
            ax_curve.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.savefig(fig_dir / "04_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("  04_summary.png")

    print(f"\nDone — {fig_dir}")
    print(f"  Overall: RMSE={rmse:.3f}  MAE={mae:.3f}  r={corr:.3f}")


if __name__ == "__main__":
    main()
