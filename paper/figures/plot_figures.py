#!/usr/bin/env python3
"""Generate all paper figures from cached data.

Requires only numpy + matplotlib.  No torch, msprime, or GPU.

Run cache_predictions.py first (on a GPU machine) to populate
paper/figures/cached_data/.  Then:

    python paper/figures/plot_figures.py

Outputs all figures to paper/figures/out/
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "paper" / "figures" / "cached_data"
OUT = ROOT / "paper" / "figures" / "out"

# Fallback data sources (if cache not yet generated on cluster)
LARGE_CTX_DIR = ROOT / "experiment-large-context" / "tmrca_predictions"
BENCHMARK_CSV = ROOT / "experiment" / "paper_figures" / "table2_benchmark.csv"
TRAINING_2K_CSV = ROOT / "experiment-large-context" / "tsinfer_2k_logs" / "csv_metrics" / "version_2" / "metrics.csv"

# Pre-generated figures we can't regenerate locally
AG1000G_FIG = ROOT / "experiment_mossies" / "data_application" / "results" / "fig_ag1000g_2L.png"

# ═══════════════════════════════════════════════════════════════════════════
# Colors & Style
# ═══════════════════════════════════════════════════════════════════════════
BLUE   = "#2166ac"
GREEN  = "#1b7837"
RED    = "#b2182b"
ORANGE = "#e08214"
GREY   = "#636363"
BLACK  = "#252525"

MODEL_COLORS = {"base": BLUE, "tree_true": GREEN, "tree_tsinfer": ORANGE}
MODEL_LABELS = {
    "base": "Base (SFS-only)",
    "tree_true": "Tree-aug (true)",
    "tree_tsinfer": "Tree-aug (tsinfer)",
}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    BLACK,
        "axes.linewidth":    0.6,
        "axes.grid":         False,
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         8,
        "axes.titlesize":    9,
        "axes.labelsize":    8,
        "xtick.labelsize":   7,
        "ytick.labelsize":   7,
        "legend.fontsize":   7,
        "legend.frameon":    False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  3,
        "ytick.major.size":  3,
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "pdf.fonttype":      42,
        "ps.fonttype":       42,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════
def load_human():
    """Load cached human model predictions (base, tree_true, tree_tsinfer)."""
    out = {}
    for key in ["base", "tree_true", "tree_tsinfer"]:
        d = np.load(CACHE / f"human_{key}.npz")
        out[key] = {"true": d["true"], "pred": d["pred"], "var": d["var"],
                     "window_size": int(d["window_size"]),
                     "seq_len": int(d["seq_len"])}
    return out


def load_anogam_100kb():
    d = np.load(CACHE / "anogam_100kb.npz")
    return {"true": d["true"], "pred": d["pred"], "var": d["var"],
            "window_size": int(d["window_size"]),
            "seq_len": int(d["seq_len"])}


def load_anogam_400kb():
    # Try unified cache first, fall back to raw experiment files
    cached = CACHE / "anogam_400kb.npz"
    if cached.exists():
        d = np.load(cached)
        return {"true": d["true"], "pred": d["pred"], "var": d["var"],
                "pos_kb": d["pos_kb"]}

    all_true, all_pred, all_var = [], [], []
    for f in sorted(LARGE_CTX_DIR.glob("pair_*.npz")):
        d = np.load(f)
        all_true.append(d["y_true"])
        all_pred.append(d["pred_mu"])
        all_var.append(d["pred_std"] ** 2)
    pos_kb = np.load(LARGE_CTX_DIR / "genomic_pos_kb.npy")
    return {"true": np.stack(all_true), "pred": np.stack(all_pred),
            "var": np.stack(all_var), "pos_kb": pos_kb}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════
def _scatter(ax, true, pred, color, *, s=6, stats_fs=6.5):
    """Scatter with diagonal, r, RMSE annotation."""
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
            transform=ax.transAxes, va="top", fontsize=stats_fs,
            family="monospace")


def _genome_profile(ax, pos, true, pred, var, color, *, label="Tree-aug (tsinfer)"):
    """Stepped true + predicted line with uncertainty band."""
    ax.step(pos, true, where="mid", color=RED,
            lw=0.7, alpha=0.85, label="True", zorder=10)
    std = np.sqrt(var)
    ax.fill_between(pos, pred - std, pred + std,
                    alpha=0.10, color=color, lw=0)
    ax.plot(pos, pred, color=color, lw=0.8, alpha=0.8, label=label)
    ax.set_ylabel("log(TMRCA)")
    ax.legend(loc="upper right", ncol=2, fontsize=5.5, handlelength=1.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _residual_row(ax_res, ax_hist, pos, true, pred, color, *, xlabel=""):
    """Residual scatter + histogram panel pair."""
    n_pairs = true.shape[0]
    resid = (pred - true).ravel()
    ax_res.scatter(np.tile(pos, n_pairs), resid, s=6, alpha=0.3,
                   c=color, edgecolors="none", rasterized=True)
    ax_res.axhline(0, color=GREY, ls="--", lw=0.5)
    ax_res.set_ylabel("Residual")
    ax_res.set_xlabel(xlabel)
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["right"].set_visible(False)

    bins = np.linspace(-3, 3, 50)
    ax_hist.hist(resid, bins=bins, orientation="horizontal",
                 color=color, alpha=0.5, edgecolor="none")
    ax_hist.axhline(0, color=GREY, ls="--", lw=0.5)
    ax_hist.tick_params(labelleft=False)
    ax_hist.set_ylim(ax_res.get_ylim())
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    stats = f"$\\mu$={resid.mean():.3f}\n$\\sigma$={resid.std():.3f}"
    ax_hist.text(0.92, 0.95, stats, transform=ax_hist.transAxes,
                 va="top", ha="right", fontsize=6, family="monospace")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Scatter (2×2 + split bottom-right)
# ═══════════════════════════════════════════════════════════════════════════
def make_fig2(human, anogam_100, anogam_400):
    print("  fig2_combined ...")
    fig = plt.figure(figsize=(6.5, 7.0))
    outer = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.32)

    # a–c: human models (full-size cells)
    for idx, key in enumerate(["base", "tree_true", "tree_tsinfer"]):
        ax = fig.add_subplot(outer[idx // 2, idx % 2])
        _scatter(ax, human[key]["true"].ravel(),
                 human[key]["pred"].ravel(), MODEL_COLORS[key])
        ax.set_title(f"{'abc'[idx]}  {MODEL_LABELS[key]}",
                     fontweight="bold", loc="left", fontsize=8, pad=4)

    # Bottom-right: split into d (100kb) and e (400kb)
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 1],
                                    hspace=0.55)

    ax_d = fig.add_subplot(inner[0])
    _scatter(ax_d, anogam_100["true"].ravel(),
             anogam_100["pred"].ravel(), ORANGE, s=4, stats_fs=5.5)
    ax_d.set_title("d  AnoGam 0.1 Mb", fontweight="bold",
                   loc="left", fontsize=7, pad=3)
    ax_d.set_xlabel("True log(TMRCA)", fontsize=6.5)
    ax_d.set_ylabel("Predicted log(TMRCA)", fontsize=6.5)
    ax_d.tick_params(labelsize=6)

    ax_e = fig.add_subplot(inner[1])
    _scatter(ax_e, anogam_400["true"].ravel(),
             anogam_400["pred"].ravel(), ORANGE, s=4, stats_fs=5.5)
    ax_e.set_title("e  AnoGam 0.4 Mb", fontweight="bold",
                   loc="left", fontsize=7, pad=3)
    ax_e.set_xlabel("True log(TMRCA)", fontsize=6.5)
    ax_e.set_ylabel("Predicted log(TMRCA)", fontsize=6.5)
    ax_e.tick_params(labelsize=6)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_combined.{ext}", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Genome TMRCA profiles (3 rows)
# ═══════════════════════════════════════════════════════════════════════════
def make_fig3(human, anogam_100, anogam_400):
    print("  fig3_combined ...")
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.4), sharex=False)

    # (a) Human 1 Mb — all 3 models
    ws = human["base"]["window_size"]
    n_win = human["base"]["true"].shape[1]
    pos_mb = (np.arange(n_win) + 0.5) * ws / 1e6

    axes[0].step(pos_mb, human["base"]["true"][0], where="mid", color=RED,
                 lw=0.7, alpha=0.85, label="True", zorder=10)
    for key in ["base", "tree_true", "tree_tsinfer"]:
        std = np.sqrt(human[key]["var"][0])
        axes[0].fill_between(pos_mb, human[key]["pred"][0] - std,
                              human[key]["pred"][0] + std,
                              alpha=0.08, color=MODEL_COLORS[key], lw=0)
        axes[0].plot(pos_mb, human[key]["pred"][0], color=MODEL_COLORS[key],
                     lw=0.8, alpha=0.8, label=MODEL_LABELS[key])
    axes[0].set_ylabel("log(TMRCA)")
    axes[0].set_xlabel("Genomic position (Mb)")
    axes[0].legend(loc="upper right", ncol=2, fontsize=5.5, handlelength=1.2)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)
    axes[0].set_title("a  Human-like 1 Mb — Pair (0, 1)", fontweight="bold",
                      loc="left", fontsize=8, pad=3)

    # (b) AnoGam 100 kb
    n_win_a = anogam_100["true"].shape[1]
    pos_kb = (np.arange(n_win_a) + 0.5) * anogam_100["window_size"] / 1e3
    _genome_profile(axes[1], pos_kb, anogam_100["true"][0],
                    anogam_100["pred"][0], anogam_100["var"][0], ORANGE)
    axes[1].set_xlabel("Genomic position (kb)")
    axes[1].set_title("b  AnoGam 100 kb — Pair (0, 1)", fontweight="bold",
                      loc="left", fontsize=8, pad=3)

    # (c) AnoGam 400 kb (pair index 1 = pair_50, best RMSE)
    pos_kb_400 = anogam_400["pos_kb"]
    _genome_profile(axes[2], pos_kb_400, anogam_400["true"][1],
                    anogam_400["pred"][1], anogam_400["var"][1], ORANGE)
    axes[2].set_xlabel("Genomic position (kb)")
    axes[2].set_title("c  AnoGam 400 kb — Pair (14, 170)", fontweight="bold",
                      loc="left", fontsize=8, pad=3)

    plt.tight_layout(h_pad=1.0)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig3_combined.{ext}", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Computational scaling (from CSV)
# ═══════════════════════════════════════════════════════════════════════════
def make_fig5():
    print("  fig5_scaling ...")
    df = pd.read_csv(BENCHMARK_CSV)

    fig, (ax_t, ax_s) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    # (a) Inference time
    ax_t.plot(df["n"], df["pairwise"], "o-", color=BLUE, lw=1.2, ms=4,
              label="Pairwise")
    ax_t.plot(df["n"], df["node_time"], "s-", color=GREEN, lw=1.2, ms=4,
              label="Node-time")
    ax_t.plot(df["n"], df["tsinfer"], "^--", color=RED, lw=0.8, ms=4,
              alpha=0.7, label="tsinfer")
    ax_t.plot(df["n"], df["full_pipeline"], "^--", color=RED, lw=1.2, ms=4,
              label="Full pipeline")
    ax_t.set_xlabel("Number of samples (n)")
    ax_t.set_ylabel("Inference time (s)")
    ax_t.legend(fontsize=6, loc="upper left")
    ax_t.spines["top"].set_visible(False)
    ax_t.spines["right"].set_visible(False)
    ax_t.set_title("a", fontweight="bold", loc="left", fontsize=9, pad=3)

    # (b) Speedup
    nt_speedup = df["pairwise"] / df["node_time"]
    full_speedup = df["pairwise"] / df["full_pipeline"]
    x = np.arange(len(df))
    w = 0.35
    bars1 = ax_s.bar(x - w/2, nt_speedup, w, color=GREEN, alpha=0.8,
                      label="Pairwise / Node-time")
    bars2 = ax_s.bar(x + w/2, full_speedup, w, color=RED, alpha=0.7,
                      label="Pairwise / Full pipeline")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels(df["n"].astype(int))
    ax_s.set_xlabel("Samples (n)")
    ax_s.set_ylabel("Speedup")
    ax_s.legend(fontsize=6, loc="upper left")
    ax_s.spines["top"].set_visible(False)
    ax_s.spines["right"].set_visible(False)
    ax_s.set_title("b", fontweight="bold", loc="left", fontsize=9, pad=3)

    for bar_group in (bars1, bars2):
        for bar in bar_group:
            h = bar.get_height()
            ax_s.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                      f"{h:.0f}$\\times$", ha="center", fontsize=5.5)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig5_scaling.{ext}", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure S1 — Training dynamics (3 rows: human, 100kb, 400kb)
# ═══════════════════════════════════════════════════════════════════════════
def make_figS1():
    print("  figS1_combined ...")
    # 2k training metrics from CSV
    df = pd.read_csv(TRAINING_2K_CSV)
    val = df.dropna(subset=["val_rmse"]).copy()

    fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.0))

    # Row 1-2: embed existing figS1 (human + 100kb) as raster
    # These were generated on GPU; we embed the canonical PNG.
    old_s1 = plt.imread(str(ROOT / "experiment" / "paper_figures" /
                             "figS1_combined.png"))
    h, w = old_s1.shape[:2]
    # Top half = human row (2 panels), bottom half = AnoGam 100kb row
    for col in range(2):
        axes[0, col].imshow(old_s1[:h//2, col*(w//2):(col+1)*(w//2), :])
        axes[0, col].axis("off")
        axes[1, col].imshow(old_s1[h//2:, col*(w//2):(col+1)*(w//2), :])
        axes[1, col].axis("off")

    # Row 3: AnoGam 400kb from CSV
    axes[2, 0].plot(val["epoch"], val["val_loss"], color=ORANGE, lw=1.0)
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("Validation loss (Beta-NLL)")
    axes[2, 0].spines["top"].set_visible(False)
    axes[2, 0].spines["right"].set_visible(False)
    axes[2, 0].set_title("e  AnoGam 400 kb", fontweight="bold",
                         loc="left", fontsize=8, pad=3)

    axes[2, 1].plot(val["epoch"], val["val_rmse"], color=ORANGE, lw=1.0)
    final_rmse = val["val_rmse"].iloc[-1]
    axes[2, 1].text(val["epoch"].iloc[-1], final_rmse,
                    f"  {final_rmse:.3f}", color=ORANGE, fontsize=6,
                    va="center")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Validation RMSE")
    axes[2, 1].spines["top"].set_visible(False)
    axes[2, 1].spines["right"].set_visible(False)
    axes[2, 1].set_title("f  AnoGam 400 kb", fontweight="bold",
                         loc="left", fontsize=8, pad=3)

    plt.tight_layout(h_pad=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figS1_combined.{ext}", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure S2 — Residuals (5 rows)
# ═══════════════════════════════════════════════════════════════════════════
def make_figS2(human, anogam_100, anogam_400):
    print("  figS2_combined ...")
    n_rows = 5
    fig, axes = plt.subplots(n_rows, 2, figsize=(7.2, 2.0 * n_rows),
                              gridspec_kw={"width_ratios": [3, 1],
                                           "wspace": 0.03})
    labels = "abcde"

    # Rows a–c: human models
    for i, key in enumerate(["base", "tree_true", "tree_tsinfer"]):
        ws = human[key]["window_size"]
        n_win = human[key]["true"].shape[1]
        pos_mb = (np.arange(n_win) + 0.5) * ws / 1e6
        _residual_row(axes[i, 0], axes[i, 1], pos_mb,
                      human[key]["true"], human[key]["pred"],
                      MODEL_COLORS[key])
        axes[i, 0].set_title(f"{labels[i]}  {MODEL_LABELS[key]}",
                             fontweight="bold", loc="left", fontsize=8, pad=3)

    # Row d: AnoGam 100kb
    n_win_a = anogam_100["true"].shape[1]
    pos_kb = (np.arange(n_win_a) + 0.5) * anogam_100["window_size"] / 1e3
    _residual_row(axes[3, 0], axes[3, 1], pos_kb,
                  anogam_100["true"], anogam_100["pred"], ORANGE)
    axes[3, 0].set_title("d  AnoGam 0.1 Mb — Tree-aug (tsinfer)",
                         fontweight="bold", loc="left", fontsize=8, pad=3)

    # Row e: AnoGam 400kb
    pos_kb_400 = anogam_400["pos_kb"]
    _residual_row(axes[4, 0], axes[4, 1], pos_kb_400,
                  anogam_400["true"], anogam_400["pred"], ORANGE,
                  xlabel="Genomic position (kb)")
    axes[4, 0].set_title("e  AnoGam 0.4 Mb — Tree-aug (tsinfer)",
                         fontweight="bold", loc="left", fontsize=8, pad=3)
    axes[4, 1].set_xlabel("Count")

    # Only bottom row gets x-label for human panels
    axes[2, 0].set_xlabel("Genomic position (Mb)")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figS2_combined.{ext}", bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Figure Ag1000G — copy pre-generated
# ═══════════════════════════════════════════════════════════════════════════
def copy_ag1000g():
    """Copy the Ag1000G figure (generated separately with real data)."""
    import shutil
    if AG1000G_FIG.exists():
        shutil.copy2(AG1000G_FIG, OUT / "fig_ag1000g_2L.png")
        pdf = AG1000G_FIG.with_suffix(".pdf")
        if pdf.exists():
            shutil.copy2(pdf, OUT / "fig_ag1000g_2L.pdf")
        print("  fig_ag1000g_2L (copied)")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)

    # Check what data we have
    have_human = (CACHE / "human_base.npz").exists()
    have_100kb = (CACHE / "anogam_100kb.npz").exists()
    have_400kb = (CACHE / "anogam_400kb.npz").exists() or LARGE_CTX_DIR.exists()

    if have_human and have_100kb:
        human = load_human()
        anogam_100 = load_anogam_100kb()
    else:
        print("WARNING: human/100kb cache missing — run cache_predictions.py")
        print("         Skipping fig2, fig3, figS2 (need cached predictions)")
        human = None
        anogam_100 = None

    anogam_400 = load_anogam_400kb() if have_400kb else None

    print("Generating figures ...")

    if human and anogam_100 and anogam_400:
        make_fig2(human, anogam_100, anogam_400)
        make_fig3(human, anogam_100, anogam_400)
        make_figS2(human, anogam_100, anogam_400)

    if BENCHMARK_CSV.exists():
        make_fig5()

    if TRAINING_2K_CSV.exists():
        make_figS1()

    copy_ag1000g()

    print(f"\nDone — all figures in {OUT}/")
    print("\nTo use in paper.tex, set:")
    print(r"  \graphicspath{{../paper/figures/out/}}")


if __name__ == "__main__":
    main()
