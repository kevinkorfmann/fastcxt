#!/usr/bin/env python3
"""Evaluate a pairwise FastCxtModel: scatter, genome profile, residuals.

Usage:
    python evaluate_pairwise.py --checkpoint outputs/pairwise/checkpoints/best.ckpt
"""

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
from config import (
    SIMS_DIR, FIGURES_DIR, WINDOW_SIZE, MUTATION_RATE,
    BLUE, RED, GREY, BLACK,
)

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

PIVOT_PAIRS = [(0, 1), (2, 3), (4, 5)]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--ts-file", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=str(FIGURES_DIR / "pairwise"))
    args = ap.parse_args()

    setup_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tree sequence
    if args.ts_file:
        ts_file = Path(args.ts_file)
    else:
        ts_file = sorted(SIMS_DIR.rglob("*.trees"))[0]

    # Load checkpoint, stripping torch.compile _orig_mod. prefix if present
    import torch
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    ckpt["state_dict"] = {k.replace("._orig_mod", ""): v for k, v in ckpt["state_dict"].items()}
    lit = LitFastCxt(**ckpt["hyper_parameters"])
    lit.load_state_dict(ckpt["state_dict"])
    model = lit.model
    ts = tskit.load(str(ts_file))
    seq_len = int(ts.sequence_length)

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    pred_means, pred_vars, _ = translate_from_ts(
        ts, model, blocks=[(0, seq_len)], pivot_pairs=PIVOT_PAIRS,
        mutation_rate=MUTATION_RATE, device=device,
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

    # 1. Scatter with marginals
    fig = plt.figure(figsize=(5.5, 5.5))
    gs = GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.04, wspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    ax_hx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_hy = fig.add_subplot(gs[1, 1], sharey=ax)

    ax.scatter(all_true, all_pred, s=10, alpha=0.35, c=BLUE, edgecolors="none", rasterized=True)
    lo = min(all_true.min(), all_pred.min()) - 0.3
    hi = max(all_true.max(), all_pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], "--", color=GREY, lw=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("True log(TMRCA)"); ax.set_ylabel("Predicted log(TMRCA)")
    ax.text(0.04, 0.96, f"r = {corr:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
            transform=ax.transAxes, va="top", fontsize=9, family="monospace",
            bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=0.4"))
    bins = np.linspace(lo, hi, 40)
    ax_hx.hist(all_true, bins=bins, color=RED, alpha=0.5, edgecolor="none")
    ax_hx.hist(all_pred, bins=bins, color=BLUE, alpha=0.5, edgecolor="none")
    ax_hx.tick_params(labelbottom=False)
    ax_hy.hist(all_pred, bins=bins, orientation="horizontal", color=BLUE, alpha=0.5, edgecolor="none")
    ax_hy.tick_params(labelleft=False)
    fig.savefig(out_dir / "scatter.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Genome profile
    n_pairs = len(PIVOT_PAIRS)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 2.8 * n_pairs), sharex=True, layout="constrained")
    if n_pairs == 1:
        axes = [axes]
    for i, (sa, sb) in enumerate(PIVOT_PAIRS):
        ax = axes[i]
        std = np.sqrt(pred_vars[i])
        ax.fill_between(pos_mb, pred_means[i] - std, pred_means[i] + std, alpha=0.2, color=BLUE)
        ax.plot(pos_mb, pred_means[i], color=BLUE, lw=1.5, label="Predicted")
        ax.step(pos_mb, true_log[i], where="mid", color=RED, lw=0.9, alpha=0.85, label="True")
        ax.set_ylabel("log(TMRCA)")
        ax.set_title(f"Pair ({sa}, {sb})")
        if i == 0:
            ax.legend(loc="upper right", ncol=2)
    axes[-1].set_xlabel("Genomic position (Mb)")
    fig.savefig(out_dir / "genome_profile.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Residuals
    fig = plt.figure(figsize=(10, 4.5), layout="constrained")
    gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.02, figure=fig)
    ax_res = fig.add_subplot(gs[0, 0])
    ax_hist = fig.add_subplot(gs[0, 1], sharey=ax_res)
    resid = all_pred - all_true
    ax_res.scatter(np.tile(pos_mb, n_pairs), resid, s=14, alpha=0.45, c=BLUE,
                   edgecolors="none", rasterized=True)
    ax_res.axhline(0, color=GREY, ls="--", lw=0.8)
    ax_res.set_xlabel("Genomic position (Mb)"); ax_res.set_ylabel("Residual")
    bins_r = np.linspace(resid.min() - 0.2, resid.max() + 0.2, 40)
    ax_hist.hist(resid, bins=bins_r, orientation="horizontal", color=BLUE, alpha=0.5, edgecolor="none")
    ax_hist.axhline(0, color=GREY, ls="--", lw=0.8)
    ax_hist.tick_params(labelleft=False)
    ax_hist.text(0.95, 0.95, f"mean={resid.mean():.2f}\nstd={resid.std():.2f}",
                 transform=ax_hist.transAxes, va="top", ha="right", fontsize=9, family="monospace")
    fig.savefig(out_dir / "residuals.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Figures saved to {out_dir}/")
    print(f"  RMSE={rmse:.3f}  MAE={mae:.3f}  r={corr:.3f}")


if __name__ == "__main__":
    main()
