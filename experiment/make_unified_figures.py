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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

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

LARGE_CTX_DIR = Path("experiment-large-context/tmrca_predictions")


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


def load_large_context_npz():
    """Load 400kb / 2k-window AnoGam predictions from saved npz files."""
    all_true, all_pred, all_var = [], [], []
    for fname in sorted(LARGE_CTX_DIR.glob("pair_*.npz")):
        d = np.load(fname)
        all_true.append(d["y_true"])
        all_pred.append(d["pred_mu"])
        all_var.append(d["pred_std"] ** 2)
    pos_kb = np.load(LARGE_CTX_DIR / "genomic_pos_kb.npy")
    return {
        "true": np.stack(all_true),    # (3, 2000)
        "pred": np.stack(all_pred),    # (3, 2000)
        "var": np.stack(all_var),      # (3, 2000)
        "pos_kb": pos_kb,              # (2000,)
    }


# ---------------------------------------------------------------------------
# Figure 2: Scatter (2x2, bottom-right split into top/bottom)
# ---------------------------------------------------------------------------
def make_fig2(results, anogam_results, anogam_400kb, out_dir):
    print("  fig2_combined.png ...")
    panel_keys = ["base", "tree_true", "tree_tsinfer"]

    fig = plt.figure(figsize=(6.5, 6.5))
    outer = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35)

    # Panels a-c: full-height cells
    for idx, key in enumerate(panel_keys):
        ax = fig.add_subplot(outer[idx // 2, idx % 2])
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
        ax.set_title(f"{'abcde'[idx]}  {LABELS[key]}", fontweight="bold",
                     loc="left", fontsize=8, pad=4)

    # Bottom-right cell: split into two stacked mini-scatter panels
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1, 1], hspace=0.45)

    # Panel d: AnoGam 100kb (existing)
    ax_d = fig.add_subplot(inner[0])
    true_a = anogam_results["true"].ravel()
    pred_a = anogam_results["pred"].ravel()
    r = np.corrcoef(true_a, pred_a)[0, 1]
    rmse = np.sqrt(np.mean((pred_a - true_a) ** 2))
    ax_d.scatter(true_a, pred_a, s=4, alpha=0.3, c=ORANGE,
                 edgecolors="none", rasterized=True)
    lo = min(true_a.min(), pred_a.min()) - 0.3
    hi = max(true_a.max(), pred_a.max()) + 0.3
    ax_d.plot([lo, hi], [lo, hi], color=GREY, lw=0.6, ls="--", zorder=0)
    ax_d.set_xlim(lo, hi); ax_d.set_ylim(lo, hi)
    ax_d.set_aspect("equal")
    ax_d.set_xlabel("True log(TMRCA)", fontsize=6.5)
    ax_d.set_ylabel("Predicted log(TMRCA)", fontsize=6.5)
    ax_d.tick_params(labelsize=6)
    ax_d.text(0.05, 0.95, f"r = {r:.3f}\nRMSE = {rmse:.3f}",
              transform=ax_d.transAxes, va="top", fontsize=5.5, family="monospace")
    ax_d.set_title("d  AnoGam 0.1 Mb", fontweight="bold",
                   loc="left", fontsize=7, pad=3)

    # Panel e: AnoGam 400kb (new, from npz)
    ax_e = fig.add_subplot(inner[1])
    true_400 = anogam_400kb["true"].ravel()
    pred_400 = anogam_400kb["pred"].ravel()
    r = np.corrcoef(true_400, pred_400)[0, 1]
    rmse = np.sqrt(np.mean((pred_400 - true_400) ** 2))
    ax_e.scatter(true_400, pred_400, s=4, alpha=0.3, c=ORANGE,
                 edgecolors="none", rasterized=True)
    lo = min(true_400.min(), pred_400.min()) - 0.3
    hi = max(true_400.max(), pred_400.max()) + 0.3
    ax_e.plot([lo, hi], [lo, hi], color=GREY, lw=0.6, ls="--", zorder=0)
    ax_e.set_xlim(lo, hi); ax_e.set_ylim(lo, hi)
    ax_e.set_aspect("equal")
    ax_e.set_xlabel("True log(TMRCA)", fontsize=6.5)
    ax_e.set_ylabel("Predicted log(TMRCA)", fontsize=6.5)
    ax_e.tick_params(labelsize=6)
    ax_e.text(0.05, 0.95, f"r = {r:.3f}\nRMSE = {rmse:.3f}",
              transform=ax_e.transAxes, va="top", fontsize=5.5, family="monospace")
    ax_e.set_title("e  AnoGam 0.4 Mb", fontweight="bold",
                   loc="left", fontsize=7, pad=3)

    fig.savefig(out_dir / "fig2_combined.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Genome profiles (3 rows)
# ---------------------------------------------------------------------------
def make_fig3(results, anogam_results, anogam_400kb, out_dir):
    print("  fig3_combined.png ...")
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(3, 1, figsize=(7.2, 5.4),
                                                   sharex=False)

    # (a) Human-like 1 Mb — Pair (0,1), all 3 models
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

    # (b) AnoGam 100 kb — Pair (0,1)
    n_win_a = anogam_results["true"].shape[1]
    pos_kb = (np.arange(n_win_a) + 0.5) * 200 / 1e3
    ax_mid.step(pos_kb, anogam_results["true"][0], where="mid", color=RED,
                lw=0.7, alpha=0.85, label="True", zorder=10)
    pred_std = np.sqrt(anogam_results["var"][0])
    ax_mid.fill_between(pos_kb, anogam_results["pred"][0] - pred_std,
                        anogam_results["pred"][0] + pred_std,
                        alpha=0.10, color=ORANGE, lw=0)
    ax_mid.plot(pos_kb, anogam_results["pred"][0], color=ORANGE, lw=0.8,
                alpha=0.8, label="Tree-aug (tsinfer)")
    ax_mid.set_ylabel("log(TMRCA)")
    ax_mid.set_xlabel("Genomic position (kb)")
    ax_mid.legend(loc="upper right", ncol=2, fontsize=5.5, handlelength=1.2)
    ax_mid.spines["top"].set_visible(False)
    ax_mid.spines["right"].set_visible(False)
    ax_mid.set_title("b  AnoGam 100 kb — Pair (0, 1)", fontweight="bold",
                     loc="left", fontsize=8, pad=3)

    # (c) AnoGam 400 kb — from npz (pair_50 = best RMSE pair)
    # pair_50.npz has pair (14, 170), RMSE=0.696
    pos_kb_400 = anogam_400kb["pos_kb"]
    # Use the second pair (index 1 = pair_50.npz) for the genome profile
    ax_bot.step(pos_kb_400, anogam_400kb["true"][1], where="mid", color=RED,
                lw=0.7, alpha=0.85, label="True", zorder=10)
    pred_std_400 = np.sqrt(anogam_400kb["var"][1])
    ax_bot.fill_between(pos_kb_400, anogam_400kb["pred"][1] - pred_std_400,
                        anogam_400kb["pred"][1] + pred_std_400,
                        alpha=0.10, color=ORANGE, lw=0)
    ax_bot.plot(pos_kb_400, anogam_400kb["pred"][1], color=ORANGE, lw=0.8,
                alpha=0.8, label="Tree-aug (tsinfer)")
    ax_bot.set_ylabel("log(TMRCA)")
    ax_bot.set_xlabel("Genomic position (kb)")
    ax_bot.legend(loc="upper right", ncol=2, fontsize=5.5, handlelength=1.2)
    ax_bot.spines["top"].set_visible(False)
    ax_bot.spines["right"].set_visible(False)
    ax_bot.set_title("c  AnoGam 400 kb — Pair (14, 170)", fontweight="bold",
                     loc="left", fontsize=8, pad=3)

    plt.tight_layout(h_pad=1.0)
    fig.savefig(out_dir / "fig3_combined.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# FigS2: Residuals (5 rows)
# ---------------------------------------------------------------------------
def make_figS2(results, anogam_results, anogam_400kb, out_dir):
    print("  figS2_combined.png ...")
    keys = ["base", "tree_true", "tree_tsinfer"]
    n_rows = len(keys) + 2  # +1 AnoGam 100kb, +1 AnoGam 400kb

    fig, axes = plt.subplots(n_rows, 2, figsize=(7.2, 2.0 * n_rows),
                             gridspec_kw={"width_ratios": [3, 1], "wspace": 0.03})
    panel_labels = "abcde"

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
        ax_res.set_title(f"{panel_labels[i]}  {LABELS[key]}", fontweight="bold",
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

    # Row d: AnoGam 100kb
    true_a, pred_a = anogam_results["true"], anogam_results["pred"]
    n_win_a = true_a.shape[1]
    pos_kb = (np.arange(n_win_a) + 0.5) * 200 / 1e3
    n_pairs_a = true_a.shape[0]
    resid_a = (pred_a - true_a).ravel()
    ax_res, ax_hist = axes[3, 0], axes[3, 1]
    ax_res.scatter(np.tile(pos_kb, n_pairs_a), resid_a, s=6, alpha=0.3,
                   c=ORANGE, edgecolors="none", rasterized=True)
    ax_res.axhline(0, color=GREY, ls="--", lw=0.5)
    ax_res.set_xlabel("")
    ax_res.set_ylabel("Residual")
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["right"].set_visible(False)
    ax_res.set_title("d  AnoGam 0.1 Mb — Tree-aug (tsinfer)", fontweight="bold",
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

    # Row e: AnoGam 400kb (from npz)
    true_400, pred_400 = anogam_400kb["true"], anogam_400kb["pred"]
    pos_kb_400 = anogam_400kb["pos_kb"]
    n_pairs_400 = true_400.shape[0]
    resid_400 = (pred_400 - true_400).ravel()
    ax_res, ax_hist = axes[4, 0], axes[4, 1]
    ax_res.scatter(np.tile(pos_kb_400, n_pairs_400), resid_400, s=6, alpha=0.3,
                   c=ORANGE, edgecolors="none", rasterized=True)
    ax_res.axhline(0, color=GREY, ls="--", lw=0.5)
    ax_res.set_xlabel("Genomic position (kb)")
    ax_res.set_ylabel("Residual")
    ax_res.spines["top"].set_visible(False)
    ax_res.spines["right"].set_visible(False)
    ax_res.set_title("e  AnoGam 0.4 Mb — Tree-aug (tsinfer)", fontweight="bold",
                     loc="left", fontsize=8, pad=3)

    bins_r = np.linspace(-3, 3, 50)
    ax_hist.hist(resid_400, bins=bins_r, orientation="horizontal",
                 color=ORANGE, alpha=0.5, edgecolor="none")
    ax_hist.axhline(0, color=GREY, ls="--", lw=0.5)
    ax_hist.tick_params(labelleft=False)
    ax_hist.set_ylim(ax_res.get_ylim())
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)
    stats = f"$\\mu$={resid_400.mean():.3f}\n$\\sigma$={resid_400.std():.3f}"
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

    # Load 400kb AnoGam predictions from saved npz files
    print("Loading AnoGam 400kb predictions ...")
    anogam_400kb = load_large_context_npz()

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
    make_fig2(results, anogam_results, anogam_400kb, out_dir)
    make_fig3(results, anogam_results, anogam_400kb, out_dir)
    make_figS2(results, anogam_results, anogam_400kb, out_dir)
    print(f"Done — all figures in {out_dir}/")


if __name__ == "__main__":
    main()
