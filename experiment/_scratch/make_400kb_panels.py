#!/usr/bin/env python3
"""Generate standalone 400kb AnoGam figure panels from saved npz files.

No torch/msprime/CUDA required. Produces:
  - fig2e_scatter_400kb.png   (scatter panel for Fig 2e)
  - fig3c_genome_400kb.png    (genome profile for Fig 3c)
  - figS2e_residual_400kb.png (residual row for FigS2e)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Style (matches main figures exactly)
# ---------------------------------------------------------------------------
RED = "#b2182b"
ORANGE = "#e08214"
GREY = "#636363"
BLACK = "#252525"

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
# Load data
# ---------------------------------------------------------------------------
DATA_DIR = Path("experiment-large-context/tmrca_predictions")
OUT_DIR = Path("experiment/paper_figures")

all_true, all_pred, all_std = [], [], []
for f in sorted(DATA_DIR.glob("pair_*.npz")):
    d = np.load(f)
    all_true.append(d["y_true"])
    all_pred.append(d["pred_mu"])
    all_std.append(d["pred_std"])
pos_kb = np.load(DATA_DIR / "genomic_pos_kb.npy")

true = np.stack(all_true)   # (3, 2000)
pred = np.stack(all_pred)   # (3, 2000)
std = np.stack(all_std)     # (3, 2000)

# ---------------------------------------------------------------------------
# Panel: scatter (matches fig2 panel style)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(3.0, 3.0))
t, p = true.ravel(), pred.ravel()
r = np.corrcoef(t, p)[0, 1]
rmse = np.sqrt(np.mean((p - t) ** 2))
ax.scatter(t, p, s=6, alpha=0.3, c=ORANGE, edgecolors="none", rasterized=True)
lo = min(t.min(), p.min()) - 0.3
hi = max(t.max(), p.max()) + 0.3
ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.6, ls="--", zorder=0)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.set_xlabel("True log(TMRCA)")
ax.set_ylabel("Predicted log(TMRCA)")
ax.text(0.05, 0.95, f"r = {r:.3f}\nRMSE = {rmse:.3f}",
        transform=ax.transAxes, va="top", fontsize=6.5, family="monospace")
ax.set_title("e  AnoGam 0.4 Mb — Tree-aug (tsinfer)", fontweight="bold",
             loc="left", fontsize=8, pad=4)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig2e_scatter_400kb.png", bbox_inches="tight")
plt.close(fig)
print("  fig2e_scatter_400kb.png")

# ---------------------------------------------------------------------------
# Panel: genome profile (matches fig3 panel style)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 1.6))
# pair index 1 = pair_50.npz, pair (14,170), best RMSE
ax.step(pos_kb, true[1], where="mid", color=RED,
        lw=0.7, alpha=0.85, label="True", zorder=10)
ax.fill_between(pos_kb, pred[1] - std[1], pred[1] + std[1],
                alpha=0.10, color=ORANGE, lw=0)
ax.plot(pos_kb, pred[1], color=ORANGE, lw=0.8, alpha=0.8,
        label="Tree-aug (tsinfer)")
ax.set_ylabel("log(TMRCA)")
ax.set_xlabel("Genomic position (kb)")
ax.legend(loc="upper right", ncol=2, fontsize=5.5, handlelength=1.2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_title("c  AnoGam 400 kb — Pair (14, 170)", fontweight="bold",
             loc="left", fontsize=8, pad=3)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig3c_genome_400kb.png", bbox_inches="tight")
plt.close(fig)
print("  fig3c_genome_400kb.png")

# ---------------------------------------------------------------------------
# Panel: residuals + histogram (matches figS2 row style)
# ---------------------------------------------------------------------------
fig, (ax_res, ax_hist) = plt.subplots(1, 2, figsize=(7.2, 1.8),
                                       gridspec_kw={"width_ratios": [3, 1],
                                                     "wspace": 0.03})
n_pairs = true.shape[0]
resid = (pred - true).ravel()
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
fig.savefig(OUT_DIR / "figS2e_residual_400kb.png", bbox_inches="tight")
plt.close(fig)
print("  figS2e_residual_400kb.png")

print("Done.")
