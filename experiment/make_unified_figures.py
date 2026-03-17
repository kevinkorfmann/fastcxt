#!/usr/bin/env python3
"""Generate ALL paper figures in a single unified style.

Produces fig2_combined.png, fig3_combined.png, figS2_combined.png
using the same clean white-box style as the AnoGam/Ag1000G figures.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.serialization
import tskit
import msprime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastcxt.config import FastCxtConfig, TrainingConfig
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_ts
from fastcxt.preprocess import windowed_tmrca

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

# ---------------------------------------------------------------------------
# Unified style (matches AnoGam / Ag1000G figures)
# ---------------------------------------------------------------------------
BLUE = "#2166ac"
GREEN = "#1b7837"
RED = "#b2182b"
ORANGE = "#e08214"
GREY = "#636363"
BLACK = "#252525"

COLORS = {"base": BLUE, "tree_true": GREEN, "tree_tsinfer": ORANGE}
LABELS = {"base": "Base (SFS-only)", "tree_true": "Tree-aug (true)",
          "tree_tsinfer": "Tree-aug (tsinfer)"}

WINDOW_SIZE = 2000
MUTATION_RATE = 1e-8
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


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


def load_model(ckpt, device="cpu"):
    lit = LitFastCxt.load_from_checkpoint(str(ckpt), map_location="cpu")
    m = lit.model.to(device)
    m.eval()
    return m


def run_model(model, ts, use_tsinfer=False, mutation_rate=None, window_size=None):
    seq_len = int(ts.sequence_length)
    if mutation_rate is None:
        mutation_rate = MUTATION_RATE
    if window_size is None:
        window_size = WINDOW_SIZE
    pairs = [(0, 1), (2, 3), (4, 5)]
    pairs = [(a, b) for a, b in pairs if b < ts.num_samples]
    blocks = [(0, seq_len)]
    kw = dict(blocks=blocks, pivot_pairs=pairs, mutation_rate=mutation_rate,
              device=DEVICE, progress=False)
    if use_tsinfer:
        kw["use_tsinfer"] = True
    means, varis, _ = translate_from_ts(ts, model, **kw)
    true_log = []
    for sa, sb in pairs:
        raw = windowed_tmrca(ts, sa, sb, window_size, seq_len)
        true_log.append(np.log(np.clip(raw, 1e-10, None)))
    return np.array(true_log), means, varis, pairs


# ---------------------------------------------------------------------------
# Figure 2: Scatter (2x2)
# ---------------------------------------------------------------------------
def make_fig2(results, anogam_results, out_dir):
    print("  fig2_combined.png ...")
    panel_keys = ["base", "tree_true", "tree_tsinfer"]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.5))

    for idx, key in enumerate(panel_keys):
        ax = axes[idx // 2, idx % 2]
        true, pred = results[key]["true"].ravel(), results[key]["pred"].ravel()
        r = np.corrcoef(true, pred)[0, 1]
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        ax.scatter(true, pred, s=6, alpha=0.3, c=COLORS[key],
                   edgecolors="none", rasterized=True)
        lo = min(true.min(), pred.min()) - 0.3
        hi = max(true.max(), pred.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.6, ls="--", zorder=0)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("True log(TMRCA)")
        ax.set_ylabel("Predicted log(TMRCA)")
        ax.text(0.05, 0.95, f"r = {r:.3f}\nRMSE = {rmse:.3f}",
                transform=ax.transAxes, va="top", fontsize=6.5, family="monospace")
        ax.set_title(f"{'abcd'[idx]}  {LABELS[key]}", fontweight="bold",
                     loc="left", fontsize=8, pad=4)

    # Panel d: AnoGam
    ax = axes[1, 1]
    true_a = anogam_results["true"].ravel()
    pred_a = anogam_results["pred"].ravel()
    r = np.corrcoef(true_a, pred_a)[0, 1]
    rmse = np.sqrt(np.mean((pred_a - true_a) ** 2))
    ax.scatter(true_a, pred_a, s=6, alpha=0.3, c=ORANGE,
               edgecolors="none", rasterized=True)
    lo = min(true_a.min(), pred_a.min()) - 0.3
    hi = max(true_a.max(), pred_a.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.6, ls="--", zorder=0)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("True log(TMRCA)")
    ax.set_ylabel("Predicted log(TMRCA)")
    ax.text(0.05, 0.95, f"r = {r:.3f}\nRMSE = {rmse:.3f}",
            transform=ax.transAxes, va="top", fontsize=6.5, family="monospace")
    ax.set_title("d  AnoGam — Tree-aug (tsinfer)", fontweight="bold",
                 loc="left", fontsize=8, pad=4)

    plt.tight_layout()
    fig.savefig(out_dir / "fig2_combined.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Genome profiles (2 rows)
# ---------------------------------------------------------------------------
def make_fig3(results, anogam_results, out_dir):
    print("  fig3_combined.png ...")
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(7.2, 3.6), sharex=False)

    # Top: human pair (0,1) — all 3 models
    n_win = results["base"]["true"].shape[1]
    pos_mb = (np.arange(n_win) + 0.5) * WINDOW_SIZE / 1e6
    ax_top.step(pos_mb, results["base"]["true"][0], where="mid", color=RED,
                lw=0.7, alpha=0.85, label="True", zorder=10)
    for key in ["base", "tree_true", "tree_tsinfer"]:
        pred_std = np.sqrt(results[key]["var"][0])
        ax_top.fill_between(pos_mb, results[key]["pred"][0] - pred_std,
                            results[key]["pred"][0] + pred_std,
                            alpha=0.08, color=COLORS[key], lw=0)
        ax_top.plot(pos_mb, results[key]["pred"][0], color=COLORS[key],
                    lw=0.8, alpha=0.8, label=LABELS[key])
    ax_top.set_ylabel("log(TMRCA)")
    ax_top.set_xlabel("Genomic position (Mb)")
    ax_top.legend(loc="upper right", ncol=2, fontsize=5.5, handlelength=1.2)
    ax_top.spines["top"].set_visible(False)
    ax_top.spines["right"].set_visible(False)
    ax_top.set_title("a  Human-like 1 Mb — Pair (0, 1)", fontweight="bold",
                     loc="left", fontsize=8, pad=3)

    # Bottom: AnoGam pair (0,1)
    n_win_a = anogam_results["true"].shape[1]
    pos_kb = (np.arange(n_win_a) + 0.5) * 200 / 1e3
    ax_bot.step(pos_kb, anogam_results["true"][0], where="mid", color=RED,
                lw=0.7, alpha=0.85, label="True", zorder=10)
    pred_std = np.sqrt(anogam_results["var"][0])
    ax_bot.fill_between(pos_kb, anogam_results["pred"][0] - pred_std,
                        anogam_results["pred"][0] + pred_std,
                        alpha=0.10, color=ORANGE, lw=0)
    ax_bot.plot(pos_kb, anogam_results["pred"][0], color=ORANGE, lw=0.8,
                alpha=0.8, label="Tree-aug (tsinfer)")
    ax_bot.set_ylabel("log(TMRCA)")
    ax_bot.set_xlabel("Genomic position (kb)")
    ax_bot.legend(loc="upper right", ncol=2, fontsize=5.5, handlelength=1.2)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.set_title("b  AnoGam 100 kb — Pair (0, 1)", fontweight="bold",
                     loc="left", fontsize=8, pad=3)

    plt.tight_layout(h_pad=1.0)
    fig.savefig(out_dir / "fig3_combined.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FigS2: Residuals (4 rows)
# ---------------------------------------------------------------------------
def make_figS2(results, anogam_results, out_dir):
    print("  figS2_combined.png ...")
    keys = ["base", "tree_true", "tree_tsinfer"]
    n_rows = len(keys) + 1

    fig, axes = plt.subplots(n_rows, 2, figsize=(7.2, 2.0 * n_rows),
                             gridspec_kw={"width_ratios": [3, 1], "wspace": 0.03})

    for i, key in enumerate(keys):
        true, pred = results[key]["true"], results[key]["pred"]
        n_win = true.shape[1]
        pos_mb = (np.arange(n_win) + 0.5) * WINDOW_SIZE / 1e6
        n_pairs = true.shape[0]
        resid = (pred - true).ravel()
        ax_res, ax_hist = axes[i, 0], axes[i, 1]

        ax_res.scatter(np.tile(pos_mb, n_pairs), resid, s=6, alpha=0.3,
                       c=COLORS[key], edgecolors="none", rasterized=True)
        ax_res.axhline(0, color=GREY, ls="--", lw=0.5)
        ax_res.set_ylabel("Residual")
        ax_res.spines["top"].set_visible(False)
        ax_res.spines["right"].set_visible(False)
        ax_res.set_title(f"{'abcd'[i]}  {LABELS[key]}", fontweight="bold",
                         loc="left", fontsize=8, pad=3)
        if i < n_rows - 1:
            ax_res.set_xlabel("")
        else:
            ax_res.set_xlabel("Genomic position (Mb)")

        bins_r = np.linspace(-3, 3, 50)
        ax_hist.hist(resid, bins=bins_r, orientation="horizontal",
                     color=COLORS[key], alpha=0.5, edgecolor="none")
        ax_hist.axhline(0, color=GREY, ls="--", lw=0.5)
        ax_hist.tick_params(labelleft=False)
        ax_hist.set_ylim(ax_res.get_ylim())
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)
        stats = f"$\\mu$={resid.mean():.3f}\n$\\sigma$={resid.std():.3f}"
        ax_hist.text(0.92, 0.95, stats, transform=ax_hist.transAxes,
                     va="top", ha="right", fontsize=6, family="monospace")

    # AnoGam row
    true_a, pred_a = anogam_results["true"], anogam_results["pred"]
    n_win_a = true_a.shape[1]
    pos_kb = (np.arange(n_win_a) + 0.5) * 200 / 1e3
    n_pairs_a = true_a.shape[0]
    resid_a = (pred_a - true_a).ravel()
    ax_res, ax_hist = axes[3, 0], axes[3, 1]
    ax_res.scatter(np.tile(pos_kb, n_pairs_a), resid_a, s=6, alpha=0.3,
                   c=ORANGE, edgecolors="none", rasterized=True)
    ax_res.axhline(0, color=GREY, ls="--", lw=0.5)
    ax_res.set_xlabel("Genomic position (kb)")
    ax_res.set_ylabel("Residual")
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["right"].set_visible(False)
    ax_res.set_title("d  AnoGam — Tree-aug (tsinfer)", fontweight="bold",
                     loc="left", fontsize=8, pad=3)

    bins_r = np.linspace(-3, 3, 50)
    ax_hist.hist(resid_a, bins=bins_r, orientation="horizontal",
                 color=ORANGE, alpha=0.5, edgecolor="none")
    ax_hist.axhline(0, color=GREY, ls="--", lw=0.5)
    ax_hist.tick_params(labelleft=False)
    ax_hist.set_ylim(ax_res.get_ylim())
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    stats = f"$\\mu$={resid_a.mean():.3f}\n$\\sigma$={resid_a.std():.3f}"
    ax_hist.text(0.92, 0.95, stats, transform=ax_hist.transAxes,
                 va="top", ha="right", fontsize=6, family="monospace")
    ax_hist.set_xlabel("Count")

    plt.tight_layout()
    fig.savefig(out_dir / "figS2_combined.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_style()
    out_dir = Path("experiment/paper_figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Simulate a human-like test tree
    print("Simulating human-like test data ...")
    ts_human = msprime.sim_ancestry(50, sequence_length=1e6,
                                     recombination_rate=1e-8,
                                     population_size=10000, random_seed=42)
    ts_human = msprime.sim_mutations(ts_human, rate=1e-8, random_seed=42)

    # Simulate AnoGam test data
    print("Simulating AnoGam test data ...")
    from fastcxt.mosquito import simulate_anogam
    ts_anogam = simulate_anogam(seed=123, n_samples=50, segment_length=100_000)

    # Load models and run inference
    ckpts = {
        "base": "experiment/lightning_logs/version_1/checkpoints/epoch=29-step=20880.ckpt",
        "tree_true": "experiment/tree_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt",
        "tree_tsinfer": "experiment/tsinfer_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt",
    }
    anogam_ckpt = "experiment_mossies/tsinfer_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt"

    results = {}
    for key, ckpt in ckpts.items():
        print(f"Running {key} ...")
        model = load_model(ckpt, device=DEVICE)
        use_tsinfer = (key == "tree_tsinfer")
        true, pred, var, pairs = run_model(model, ts_human, use_tsinfer=use_tsinfer)
        results[key] = {"true": true, "pred": pred, "var": var}

    print("Running AnoGam ...")
    model_a = load_model(anogam_ckpt, device=DEVICE)
    true_a, pred_a, var_a, _ = run_model(model_a, ts_anogam, use_tsinfer=True,
                                          mutation_rate=3.5e-9, window_size=200)
    anogam_results = {"true": true_a, "pred": pred_a, "var": var_a}

    # Generate figures
    print("Generating figures ...")
    make_fig2(results, anogam_results, out_dir)
    make_fig3(results, anogam_results, out_dir)
    make_figS2(results, anogam_results, out_dir)
    print(f"Done — all figures in {out_dir}/")


if __name__ == "__main__":
    main()
