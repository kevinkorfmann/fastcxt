#!/usr/bin/env python3
"""Regenerate paper figures locally without torch/msprime/CUDA.

Re-uses the existing fig2/fig3/figS2 panels for human + AnoGam 100kb
by reading pixel data from the old PNGs, and adds new AnoGam 400kb
panels from the saved npz files.

Requires only: numpy, matplotlib.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
BLUE = "#2166ac"
GREEN = "#1b7837"
RED = "#b2182b"
ORANGE = "#e08214"
GREY = "#636363"
BLACK = "#252525"

def setup_style():
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


# ---------------------------------------------------------------------------
# Load 400kb data
# ---------------------------------------------------------------------------
LARGE_CTX_DIR = Path("experiment-large-context/tmrca_predictions")

def load_large_context_npz():
    all_true, all_pred, all_var = [], [], []
    for fname in sorted(LARGE_CTX_DIR.glob("pair_*.npz")):
        d = np.load(fname)
        all_true.append(d["y_true"])
        all_pred.append(d["pred_mu"])
        all_var.append(d["pred_std"] ** 2)
    pos_kb = np.load(LARGE_CTX_DIR / "genomic_pos_kb.npy")
    return {
        "true": np.stack(all_true),
        "pred": np.stack(all_pred),
        "var": np.stack(all_var),
        "pos_kb": pos_kb,
    }


def scatter_panel(ax, true, pred, color, title, fontsize_stats=6.5,
                  fontsize_title=8, s=6):
    """Draw a scatter panel with diagonal, r, RMSE."""
    r = np.corrcoef(true, pred)[0, 1]
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    ax.scatter(true, pred, s=s, alpha=0.3, c=color,
               edgecolors="none", rasterized=True)
    lo = min(true.min(), pred.min()) - 0.3
    hi = max(true.max(), pred.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.6, ls="--", zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("True log(TMRCA)")
    ax.set_ylabel("Predicted log(TMRCA)")
    ax.text(0.05, 0.95, f"r = {r:.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, va="top", fontsize=fontsize_stats,
            family="monospace")
    ax.set_title(title, fontweight="bold", loc="left",
                 fontsize=fontsize_title, pad=4)


# ---------------------------------------------------------------------------
# Figure 2: composite old 2x2 top-left 3 panels + new bottom-right split
# ---------------------------------------------------------------------------
def make_fig2(anogam_400kb, out_dir):
    print("  fig2_combined.png ...")
    old_img = plt.imread(out_dir / "fig2_combined.png")

    # The old image is a 2x2 grid. We'll embed the top-left 3 panels as an
    # image and replace the bottom-right with two new scatter panels.
    h, w = old_img.shape[:2]
    # Crop: top row = [:h//2], bottom-left = [h//2:, :w//2]
    top_row = old_img[:h//2, :, :]
    bot_left = old_img[h//2:, :w//2, :]

    fig = plt.figure(figsize=(6.5, 6.5))
    outer = GridSpec(2, 2, figure=fig, hspace=0.02, wspace=0.02)

    # Top-left: panel a (old image crop)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_a.imshow(old_img[:h//2, :w//2, :])
    ax_a.axis("off")

    # Top-right: panel b (old image crop)
    ax_b = fig.add_subplot(outer[0, 1])
    ax_b.imshow(old_img[:h//2, w//2:, :])
    ax_b.axis("off")

    # Bottom-left: panel c (old image crop)
    ax_c = fig.add_subplot(outer[1, 0])
    ax_c.imshow(old_img[h//2:, :w//2, :])
    ax_c.axis("off")

    # Bottom-right: split into d (100kb) and e (400kb)
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 1], hspace=0.5)

    # Panel d: AnoGam 100kb from old image bottom-right crop
    ax_d = fig.add_subplot(inner[0])
    ax_d.imshow(old_img[h//2:, w//2:, :])
    ax_d.axis("off")

    # Panel e: AnoGam 400kb scatter (new, from npz)
    ax_e = fig.add_subplot(inner[1])
    true_400 = anogam_400kb["true"].ravel()
    pred_400 = anogam_400kb["pred"].ravel()
    scatter_panel(ax_e, true_400, pred_400, ORANGE,
                  "e  AnoGam 0.4 Mb", fontsize_stats=5.5,
                  fontsize_title=7, s=4)
    ax_e.set_xlabel("True log(TMRCA)", fontsize=6.5)
    ax_e.set_ylabel("Predicted log(TMRCA)", fontsize=6.5)
    ax_e.tick_params(labelsize=6)

    fig.savefig(out_dir / "fig2_combined.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: composite old 2-row + new 3rd row
# ---------------------------------------------------------------------------
def make_fig3(anogam_400kb, out_dir):
    print("  fig3_combined.png ...")
    old_img = plt.imread(out_dir / "fig3_combined.png")
    h, w = old_img.shape[:2]

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4),
                              gridspec_kw={"height_ratios": [1, 1, 1]})

    # (a) top row from old image
    axes[0].imshow(old_img[:h//2, :, :])
    axes[0].axis("off")

    # (b) bottom row from old image
    axes[1].imshow(old_img[h//2:, :, :])
    axes[1].axis("off")

    # (c) AnoGam 400kb genome profile (new)
    ax = axes[2]
    pos_kb = anogam_400kb["pos_kb"]
    # pair index 1 = pair_50.npz (best RMSE, pair 14,170)
    ax.step(pos_kb, anogam_400kb["true"][1], where="mid", color=RED,
            lw=0.7, alpha=0.85, label="True", zorder=10)
    pred_std = np.sqrt(anogam_400kb["var"][1])
    ax.fill_between(pos_kb, anogam_400kb["pred"][1] - pred_std,
                    anogam_400kb["pred"][1] + pred_std,
                    alpha=0.10, color=ORANGE, lw=0)
    ax.plot(pos_kb, anogam_400kb["pred"][1], color=ORANGE, lw=0.8,
            alpha=0.8, label="Tree-aug (tsinfer)")
    ax.set_ylabel("log(TMRCA)")
    ax.set_xlabel("Genomic position (kb)")
    ax.legend(loc="upper right", ncol=2, fontsize=5.5, handlelength=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("c  AnoGam 400 kb — Pair (14, 170)", fontweight="bold",
                 loc="left", fontsize=8, pad=3)

    plt.tight_layout(h_pad=0.5)
    fig.savefig(out_dir / "fig3_combined.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FigS2: composite old 4-row + new 5th row
# ---------------------------------------------------------------------------
def make_figS2(anogam_400kb, out_dir):
    print("  figS2_combined.png ...")
    old_img = plt.imread(out_dir / "figS2_combined.png")
    h, w = old_img.shape[:2]

    # Old image has 4 rows; add a 5th
    fig = plt.figure(figsize=(7.2, 2.0 * 5))
    gs = GridSpec(5, 1, figure=fig, height_ratios=[1, 1, 1, 1, 1])

    # Rows a-d from old image (split into 4 equal strips)
    row_h = h // 4
    for i in range(4):
        ax = fig.add_subplot(gs[i])
        ax.imshow(old_img[i*row_h:(i+1)*row_h, :, :])
        ax.axis("off")

    # Row e: AnoGam 400kb residuals (new)
    # Use a nested 1x2 grid for residual + histogram
    gs_e = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[4],
                                   width_ratios=[3, 1], wspace=0.03)
    ax_res = fig.add_subplot(gs_e[0])
    ax_hist = fig.add_subplot(gs_e[1])

    true_400, pred_400 = anogam_400kb["true"], anogam_400kb["pred"]
    pos_kb = anogam_400kb["pos_kb"]
    n_pairs = true_400.shape[0]
    resid = (pred_400 - true_400).ravel()

    ax_res.scatter(np.tile(pos_kb, n_pairs), resid, s=6, alpha=0.3,
                   c=ORANGE, edgecolors="none", rasterized=True)
    ax_res.axhline(0, color=GREY, ls="--", lw=0.5)
    ax_res.set_xlabel("Genomic position (kb)")
    ax_res.set_ylabel("Residual")
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["right"].set_visible(False)
    ax_res.set_title("e  AnoGam 0.4 Mb — Tree-aug (tsinfer)", fontweight="bold",
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
    ax_hist.set_xlabel("Count")

    plt.tight_layout()
    fig.savefig(out_dir / "figS2_combined.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FigS1: add 2k training curves panel
# ---------------------------------------------------------------------------
def make_figS1(out_dir):
    print("  figS1_combined.png ...")
    old_img = plt.imread(out_dir / "figS1_combined.png")
    new_panel = plt.imread("experiment-large-context/paper_figures/training_2k.png")

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0))

    # Old figS1 has 2 panels side-by-side; split at midpoint
    h, w = old_img.shape[:2]
    axes[0].imshow(old_img[:, :w//2, :])
    axes[0].axis("off")
    axes[1].imshow(old_img[:, w//2:, :])
    axes[1].axis("off")

    # Panel c: 2k training curves
    axes[2].imshow(new_panel)
    axes[2].axis("off")
    axes[2].set_title("c  AnoGam 400 kb (2k windows)", fontweight="bold",
                      loc="left", fontsize=8, pad=3)

    plt.tight_layout()
    fig.savefig(out_dir / "figS1_combined.png", bbox_inches="tight")
    plt.close(fig)


def main():
    setup_style()
    out_dir = Path("experiment/paper_figures")

    print("Loading AnoGam 400kb predictions ...")
    anogam_400kb = load_large_context_npz()

    print("Generating figures (local composite mode) ...")
    make_fig2(anogam_400kb, out_dir)
    make_fig3(anogam_400kb, out_dir)
    make_figS2(anogam_400kb, out_dir)
    make_figS1(out_dir)
    print(f"Done — all figures in {out_dir}/")


if __name__ == "__main__":
    main()
