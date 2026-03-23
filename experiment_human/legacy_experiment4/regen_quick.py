#!/usr/bin/env python3
"""Quick figure regeneration: scaling + training (no model inference needed)."""
import csv, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "experiment" / "paper_figures"

BLUE = "#2166ac"
GREEN = "#1b7837"
ORANGE = "#e08214"
GREY = "#636363"
BLACK = "#252525"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": BLACK, "axes.linewidth": 0.8, "axes.grid": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "legend.frameon": False,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 4, "ytick.major.size": 4,
    "xtick.direction": "out", "ytick.direction": "out",
    "figure.dpi": 300, "savefig.dpi": 300,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

def _strip(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def scaling():
    print("=== Figure 5: Scaling ===", flush=True)
    with open(OUT / "table2_benchmark.csv") as f:
        rows = [r for r in csv.DictReader(f) if int(r["n"]) >= 30]
    ns = [int(r["n"]) for r in rows]
    t_pw = [float(r["pairwise"]) for r in rows]
    t_nt = [float(r["node_time"]) for r in rows]
    t_ts = [float(r["tsinfer"]) for r in rows]
    t_fp = [float(r["full_pipeline"]) for r in rows]
    sp_nt = [pw/nt for pw, nt in zip(t_pw, t_nt)]
    sp_fp = [pw/fp for pw, fp in zip(t_pw, t_fp)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ns, t_pw, "o-", color=BLUE, lw=2.2, ms=9, label="Pairwise Mamba", zorder=10)
    ax1.plot(ns, t_nt, "s-", color=GREEN, lw=2.2, ms=9, label="Node-time only", zorder=10)
    ax1.plot(ns, t_ts, "^-", color=GREY, lw=1.6, ms=8, label="tsinfer", alpha=0.8)
    ax1.plot(ns, t_fp, "D--", color=ORANGE, lw=1.6, ms=8, label="Full pipeline", alpha=0.9)
    ax1.set_xlabel("Number of haplotypes ($n$)")
    ax1.set_ylabel("Inference time (s)")
    ax1.set_title("a", fontweight="bold", loc="left", fontsize=14, pad=8)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.95, edgecolor="none")
    _strip(ax1)

    x = np.arange(len(ns))
    w = 0.32
    b1 = ax2.bar(x - w/2, sp_nt, w, color=GREEN, alpha=0.85,
                 edgecolor="white", lw=0.5, label="Node-time only")
    b2 = ax2.bar(x + w/2, sp_fp, w, color=ORANGE, alpha=0.85,
                 edgecolor="white", lw=0.5, label="Full pipeline")
    for bar, s in zip(b1, sp_nt):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{s:.0f}x", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color="#1a5e2a")
    for bar, s in zip(b2, sp_fp):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{s:.0f}x", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color="#c05000")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.set_xlabel("Number of haplotypes ($n$)")
    ax2.set_ylabel("Speedup over pairwise Mamba")
    ax2.set_title("b", fontweight="bold", loc="left", fontsize=14, pad=8)
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.95, edgecolor="none")
    ax2.set_ylim(0, max(sp_nt)*1.2)
    _strip(ax2)

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.92, wspace=0.28)
    fig.savefig(OUT / "fig5_scaling.png", dpi=300)
    plt.close(fig)
    print("  Saved fig5_scaling.png", flush=True)


def training():
    print("=== Figure S1: Training ===", flush=True)
    csvs = {
        "Base (SFS-only)": (REPO/"experiment/lightning_logs/version_1/metrics.csv", BLUE),
        "Tree-aug (true)": (REPO/"experiment/tree_logs/lightning_logs/version_0/metrics.csv", GREEN),
        "Tree-aug (tsinfer)": (REPO/"experiment/tsinfer_logs/lightning_logs/version_0/metrics.csv", ORANGE),
    }
    anogam = REPO/"experiment_mossies/tsinfer_logs/lightning_logs/version_0/metrics.csv"

    def _read(p, col="val_loss"):
        ep, vl = [], []
        with open(p) as f:
            for r in csv.DictReader(f):
                v = r.get(col, "")
                if v:
                    try:
                        ep.append(int(float(r.get("epoch", 0))))
                        vl.append(float(v))
                    except ValueError:
                        pass
        return np.array(ep), np.array(vl)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
    for lab, (p, c) in csvs.items():
        if p.exists():
            e, v = _read(p)
            if len(e):
                ax1.plot(e, v, color=c, lw=2.0, label=lab)
    ax1.set_ylabel("Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_title("a   Human-like models", fontweight="bold", loc="left", fontsize=13, pad=8)
    ax1.legend(loc="upper right", fontsize=10)
    _strip(ax1)

    if anogam.exists():
        e, v = _read(anogam)
        if len(e):
            ax2.plot(e, v, color=ORANGE, lw=2.0, label="Tree-aug (tsinfer)")
    ax2.set_ylabel("Validation loss")
    ax2.set_xlabel("Epoch")
    ax2.set_title("b   AnoGam", fontweight="bold", loc="left", fontsize=13, pad=8)
    ax2.legend(loc="upper right", fontsize=10)
    _strip(ax2)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.94, hspace=0.38)
    fig.savefig(OUT / "figS1_combined.png", dpi=300)
    plt.close(fig)
    print("  Saved figS1_combined.png", flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    scaling()
    training()
    print("Done (quick figs).", flush=True)
