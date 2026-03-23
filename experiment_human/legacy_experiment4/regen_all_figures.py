#!/usr/bin/env python3
"""Regenerate ALL paper figures from checkpoints.

Fixes: large readable fonts, no overlapping labels, matched y-axes, proper padding.

Produces in experiment/paper_figures/:
  fig2_combined.png  -- 2x2 scatter
  fig3_combined.png  -- 2-row genome, shared y-axis [6, 15]
  fig4_ood.png       -- OOD robustness heatmaps
  fig5_scaling.png   -- scaling benchmark
  figS1_combined.png -- training curves stacked
  figS2_combined.png -- residuals stacked

Usage:
  .venv/bin/python experiment4/regen_all_figures.py
"""
from __future__ import annotations
import sys, csv
from pathlib import Path

import numpy as np
import torch
import torch.serialization
import msprime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from fastcxt.config import FastCxtConfig, TrainingConfig
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_ts
from fastcxt.preprocess import windowed_tmrca

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

HUMAN_WINDOW = 2000
HUMAN_SEQ = 1_000_000
HUMAN_MU = 1e-8

ANOGAM_WINDOW = 200
ANOGAM_SEQ = 100_000
ANOGAM_MU = 1e-8

BLUE = "#2166ac"
GREEN = "#1b7837"
RED = "#b2182b"
ORANGE = "#e08214"
GREY = "#636363"
BLACK = "#252525"

OUT = REPO / "experiment" / "paper_figures"
YLIM = (6, 15)

BASE_CKPT = REPO / "experiment/lightning_logs/version_1/checkpoints/epoch=29-step=20880.ckpt"
TREES_CKPT = REPO / "experiment/tree_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt"
TSINFER_CKPT = REPO / "experiment/tsinfer_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt"
ANOGAM_CKPT = REPO / "experiment_mossies/tsinfer_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt"


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": BLACK,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

_apply_style()

_MODEL_CACHE = {}

def load_model(ckpt_path):
    key = str(ckpt_path)
    if key not in _MODEL_CACHE:
        lit = LitFastCxt.load_from_checkpoint(key, map_location="cpu")
        model = lit.model.to(DEVICE)
        model.eval()
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def simulate_human_ts(n=50, seed=42):
    ts = msprime.sim_ancestry(
        samples=n, sequence_length=HUMAN_SEQ,
        recombination_rate=1e-8, population_size=10_000,
        random_seed=seed,
    )
    return msprime.sim_mutations(ts, rate=HUMAN_MU, random_seed=seed)


def load_anogam_ts():
    import tskit
    ts_files = sorted((REPO / "experiment_mossies" / "sims").rglob("*.trees"))
    for f in ts_files:
        if "n50" in str(f):
            return tskit.load(str(f))
    return tskit.load(str(ts_files[0]))


def _strip_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: Scatter (2x2)
# ══════════════════════════════════════════════════════════════════════════

def figure2():
    print("=== Figure 2: Scatter (2x2) ===")
    ts_human = simulate_human_ts()
    ts_anogam = load_anogam_ts()
    pivot = [(0, 1), (2, 3), (4, 5)]

    def _scatter_data(ts, model, use_tsinfer, mu, window_sz, pairs):
        blocks = [(0, int(ts.sequence_length))]
        kw = dict(blocks=blocks, pivot_pairs=pairs, mutation_rate=mu,
                  device=DEVICE, progress=False)
        if use_tsinfer:
            kw["use_tsinfer"] = True
        pm, _, _ = translate_from_ts(ts, model, **kw)
        true_log = []
        for sa, sb in pairs:
            raw = windowed_tmrca(ts, sa, sb, window_sz, int(ts.sequence_length))
            true_log.append(np.log(np.clip(raw, 1e-10, None)))
        return np.array(true_log).ravel(), pm.ravel()

    panels = [
        ("a   Base (SFS-only)",       ts_human,  load_model(BASE_CKPT),    False, HUMAN_MU,  HUMAN_WINDOW,  pivot, BLUE),
        ("b   Tree-aug (true trees)", ts_human,  load_model(TREES_CKPT),   False, HUMAN_MU,  HUMAN_WINDOW,  pivot, GREEN),
        ("c   Tree-aug (tsinfer)",    ts_human,  load_model(TSINFER_CKPT), True,  HUMAN_MU,  HUMAN_WINDOW,  pivot, ORANGE),
        ("d   AnoGam (tsinfer)",      ts_anogam, load_model(ANOGAM_CKPT),  True,  ANOGAM_MU, ANOGAM_WINDOW, pivot, ORANGE),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))

    for idx, (title, ts, model, use_ts, mu, ws, pairs, color) in enumerate(panels):
        ax = axes.flat[idx]
        all_true, all_pred = _scatter_data(ts, model, use_ts, mu, ws, pairs)
        r = np.corrcoef(all_true, all_pred)[0, 1]
        rmse = np.sqrt(np.mean((all_pred - all_true) ** 2))

        ax.scatter(all_true, all_pred, s=5, alpha=0.2, c=color,
                   edgecolors="none", rasterized=True)
        lo = min(all_true.min(), all_pred.min()) - 0.5
        hi = max(all_true.max(), all_pred.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], color=GREY, lw=0.8, ls="--", zorder=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")

        if idx >= 2:
            ax.set_xlabel("True log(TMRCA)")
        if idx % 2 == 0:
            ax.set_ylabel("Predicted log(TMRCA)")

        ax.text(0.05, 0.95, f"r = {r:.3f}\nRMSE = {rmse:.3f}",
                transform=ax.transAxes, va="top", fontsize=11, family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"))
        ax.set_title(title, fontweight="bold", loc="left", fontsize=12, pad=8)
        _strip_spines(ax)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.06, top=0.95, wspace=0.30, hspace=0.28)
    fig.savefig(OUT / "fig2_combined.png", dpi=300)
    plt.close(fig)
    print(f"  Saved fig2_combined.png")


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Genome TMRCA (2 rows, shared y-axis)
# ══════════════════════════════════════════════════════════════════════════

def figure3():
    print("=== Figure 3: Genome TMRCA ===")
    ts_human = simulate_human_ts()
    ts_anogam = load_anogam_ts()
    pair = [(0, 1)]

    # Human
    raw_h = windowed_tmrca(ts_human, 0, 1, HUMAN_WINDOW, HUMAN_SEQ)
    true_h = np.log(np.clip(raw_h, 1e-10, None))
    pos_h = (np.arange(len(true_h)) + 0.5) * HUMAN_WINDOW / 1e6

    kw_h = dict(blocks=[(0, HUMAN_SEQ)], pivot_pairs=pair,
                mutation_rate=HUMAN_MU, device=DEVICE, progress=False)
    pm_base, pv_base, _ = translate_from_ts(ts_human, load_model(BASE_CKPT), **kw_h)
    pm_trees, pv_trees, _ = translate_from_ts(ts_human, load_model(TREES_CKPT), **kw_h)
    pm_tsinfer, pv_tsinfer, _ = translate_from_ts(ts_human, load_model(TSINFER_CKPT),
                                                   **{**kw_h, "use_tsinfer": True})

    # AnoGam
    raw_a = windowed_tmrca(ts_anogam, 0, 1, ANOGAM_WINDOW, ANOGAM_SEQ)
    true_a = np.log(np.clip(raw_a, 1e-10, None))
    pos_a = (np.arange(len(true_a)) + 0.5) * ANOGAM_WINDOW / 1e3

    kw_a = dict(blocks=[(0, ANOGAM_SEQ)], pivot_pairs=pair,
                mutation_rate=ANOGAM_MU, device=DEVICE, progress=False, use_tsinfer=True)
    pm_anogam, pv_anogam, _ = translate_from_ts(ts_anogam, load_model(ANOGAM_CKPT), **kw_a)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    # Top: Human
    ax1.step(pos_h, true_h, where="mid", color=RED, lw=1.0, alpha=0.85, label="True", zorder=10)
    for pm, pv, col, lab in [
        (pm_base[0], pv_base[0], BLUE, "Base"),
        (pm_trees[0], pv_trees[0], GREEN, "Tree-aug (true)"),
        (pm_tsinfer[0], pv_tsinfer[0], ORANGE, "Tree-aug (tsinfer)"),
    ]:
        sd = np.sqrt(pv)
        ax1.fill_between(pos_h, pm - sd, pm + sd, alpha=0.08, color=col, lw=0)
        ax1.plot(pos_h, pm, color=col, lw=1.0, alpha=0.8, label=lab)
    ax1.set_ylim(*YLIM)
    ax1.set_ylabel("log(TMRCA)")
    ax1.set_xlabel("Genomic position (Mb)")
    ax1.set_title("a   Human-like, pair (0, 1), 1 Mb", fontweight="bold", loc="left", fontsize=13, pad=8)
    ax1.legend(loc="upper right", ncol=4, fontsize=10, handlelength=1.8, columnspacing=1.2)
    _strip_spines(ax1)
    ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # Bottom: AnoGam
    ax2.step(pos_a, true_a, where="mid", color=RED, lw=1.0, alpha=0.85, label="True", zorder=10)
    sd_a = np.sqrt(pv_anogam[0])
    ax2.fill_between(pos_a, pm_anogam[0] - sd_a, pm_anogam[0] + sd_a,
                     alpha=0.08, color=ORANGE, lw=0)
    ax2.plot(pos_a, pm_anogam[0], color=ORANGE, lw=1.0, alpha=0.8, label="Tree-aug (tsinfer)")
    ax2.set_ylim(*YLIM)
    ax2.set_ylabel("log(TMRCA)")
    ax2.set_xlabel("Genomic position (kb)")
    ax2.set_title("b   AnoGam, pair (0, 1), 100 kb", fontweight="bold", loc="left", fontsize=13, pad=8)
    ax2.legend(loc="upper right", ncol=2, fontsize=10, handlelength=1.8, columnspacing=1.2)
    _strip_spines(ax2)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.94, hspace=0.38)
    fig.savefig(OUT / "fig3_combined.png", dpi=300)
    plt.close(fig)
    print(f"  Saved fig3_combined.png")


# ══════════════════════════════════════════════════════════════════════════
# Figure 4: OOD robustness heatmaps
# ══════════════════════════════════════════════════════════════════════════

def figure4_ood():
    print("=== Figure 4: OOD Robustness ===")
    model = load_model(BASE_CKPT)
    pivot = [(0, 1), (2, 3), (4, 5)]
    n_samples = 50
    n_sims = 5

    ne_base = 20_000
    mu_base = HUMAN_MU
    grid = np.logspace(-1, 1, 11, base=2)

    bias_grid = np.full((len(grid), len(grid)), np.nan)
    mse_grid = np.full((len(grid), len(grid)), np.nan)

    for i, mu_scale in enumerate(grid):
        for j, ne_scale in enumerate(grid):
            ne = int(ne_base * ne_scale)
            mu = mu_base * mu_scale
            biases, mses = [], []
            for seed in range(n_sims):
                ts = msprime.sim_ancestry(
                    samples=n_samples, sequence_length=HUMAN_SEQ,
                    recombination_rate=1e-8, population_size=ne,
                    random_seed=seed + 1000 * i + j,
                )
                ts = msprime.sim_mutations(ts, rate=mu, random_seed=seed + 2000 * i + j)
                if ts.num_sites < 10:
                    continue
                try:
                    pm, _, _ = translate_from_ts(
                        ts, model, blocks=[(0, HUMAN_SEQ)], pivot_pairs=pivot,
                        mutation_rate=mu, device=DEVICE, progress=False)
                except Exception:
                    continue
                true_log = []
                for sa, sb in pivot:
                    raw = windowed_tmrca(ts, sa, sb, HUMAN_WINDOW, HUMAN_SEQ)
                    true_log.append(np.log(np.clip(raw, 1e-10, None)))
                true_log = np.array(true_log)
                diff = pm.ravel() - true_log.ravel()
                biases.append(np.mean(diff))
                mses.append(np.mean(diff ** 2))
            if biases:
                bias_grid[i, j] = np.mean(biases)
                mse_grid[i, j] = np.mean(mses)
            print(f"  mu_scale={mu_scale:.2f} ne_scale={ne_scale:.2f} done")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    extent = [np.log2(grid[0]), np.log2(grid[-1]),
              np.log2(grid[0]), np.log2(grid[-1])]

    vmax_bias = max(abs(np.nanmin(bias_grid)), abs(np.nanmax(bias_grid)))
    im1 = ax1.imshow(bias_grid, origin="lower", extent=extent, aspect="auto",
                     cmap="RdBu_r", vmin=-vmax_bias, vmax=vmax_bias,
                     interpolation="nearest")
    ax1.set_xlabel("$N_e$ scaling (log$_2$)")
    ax1.set_ylabel("$\\mu$ scaling (log$_2$)")
    ax1.set_title("a   Bias", fontweight="bold", loc="left", fontsize=13, pad=8)
    center_x = np.log2(grid[len(grid) // 2])
    ax1.plot(center_x, center_x, "o", color=GREEN, markersize=8, zorder=10,
             markeredgecolor="white", markeredgewidth=1.0)
    cb1 = plt.colorbar(im1, ax=ax1, shrink=0.9, pad=0.03)
    cb1.ax.tick_params(labelsize=10)

    im2 = ax2.imshow(mse_grid, origin="lower", extent=extent, aspect="auto",
                     cmap="YlOrRd", vmin=0, interpolation="nearest")
    ax2.set_xlabel("$N_e$ scaling (log$_2$)")
    ax2.set_ylabel("$\\mu$ scaling (log$_2$)")
    ax2.set_title("b   MSE", fontweight="bold", loc="left", fontsize=13, pad=8)
    ax2.plot(center_x, center_x, "o", color=GREEN, markersize=8, zorder=10,
             markeredgecolor="white", markeredgewidth=1.0)
    cb2 = plt.colorbar(im2, ax=ax2, shrink=0.9, pad=0.03)
    cb2.ax.tick_params(labelsize=10)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.14, top=0.90, wspace=0.35)
    fig.savefig(OUT / "fig4_ood.png", dpi=300)
    plt.close(fig)
    print(f"  Saved fig4_ood.png")


# ══════════════════════════════════════════════════════════════════════════
# Figure 5: Scaling benchmark (from CSV)
# ══════════════════════════════════════════════════════════════════════════

def figure5_scaling():
    print("=== Figure 5: Scaling ===")
    csv_path = OUT / "table2_benchmark.csv"
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if int(r["n"]) >= 30]

    ns = [int(r["n"]) for r in rows]
    t_pw = [float(r["pairwise"]) for r in rows]
    t_nt = [float(r["node_time"]) for r in rows]
    t_ts = [float(r["tsinfer"]) for r in rows]
    t_fp = [float(r["full_pipeline"]) for r in rows]
    sp_nt = [pw / nt for pw, nt in zip(t_pw, t_nt)]
    sp_fp = [pw / fp for pw, fp in zip(t_pw, t_fp)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(ns, t_pw, "o-", color=BLUE, lw=2.2, markersize=9, label="Pairwise Mamba", zorder=10)
    ax1.plot(ns, t_nt, "s-", color=GREEN, lw=2.2, markersize=9, label="Node-time only", zorder=10)
    ax1.plot(ns, t_ts, "^-", color=GREY, lw=1.6, markersize=8, label="tsinfer", alpha=0.8)
    ax1.plot(ns, t_fp, "D--", color=ORANGE, lw=1.6, markersize=8, label="Full pipeline", alpha=0.9)
    ax1.set_xlabel("Number of haplotypes ($n$)")
    ax1.set_ylabel("Inference time (s)")
    ax1.set_title("a", fontweight="bold", loc="left", fontsize=14, pad=8)
    ax1.legend(loc="upper left", fontsize=10, framealpha=0.95, edgecolor="none")
    _strip_spines(ax1)

    x = np.arange(len(ns))
    w = 0.32
    bars1 = ax2.bar(x - w / 2, sp_nt, w, color=GREEN, alpha=0.85,
                    edgecolor="white", linewidth=0.5, label="Node-time only")
    bars2 = ax2.bar(x + w / 2, sp_fp, w, color=ORANGE, alpha=0.85,
                    edgecolor="white", linewidth=0.5, label="Full pipeline")

    for bar, s in zip(bars1, sp_nt):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{s:.0f}$\\times$", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color="#1a5e2a")
    for bar, s in zip(bars2, sp_fp):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{s:.0f}$\\times$", ha="center", va="bottom", fontsize=10,
                 fontweight="bold", color="#c05000")

    ax2.set_xticks(x)
    ax2.set_xticklabels([str(n) for n in ns])
    ax2.set_xlabel("Number of haplotypes ($n$)")
    ax2.set_ylabel("Speedup over pairwise Mamba")
    ax2.set_title("b", fontweight="bold", loc="left", fontsize=14, pad=8)
    ax2.legend(loc="upper left", fontsize=10, framealpha=0.95, edgecolor="none")
    ax2.set_ylim(0, max(sp_nt) * 1.2)
    _strip_spines(ax2)

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.92, wspace=0.28)
    fig.savefig(OUT / "fig5_scaling.png", dpi=300)
    plt.close(fig)
    print(f"  Saved fig5_scaling.png")


# ══════════════════════════════════════════════════════════════════════════
# Figure S1: Training curves
# ══════════════════════════════════════════════════════════════════════════

def figS1_training():
    print("=== Figure S1: Training Dynamics ===")
    csvs = {
        "Base (SFS-only)":    (REPO / "experiment/lightning_logs/version_1/metrics.csv", BLUE),
        "Tree-aug (true)":    (REPO / "experiment/tree_logs/lightning_logs/version_0/metrics.csv", GREEN),
        "Tree-aug (tsinfer)": (REPO / "experiment/tsinfer_logs/lightning_logs/version_0/metrics.csv", ORANGE),
    }
    anogam_csv = REPO / "experiment_mossies/tsinfer_logs/lightning_logs/version_0/metrics.csv"

    def _read(path, col="val_loss"):
        epochs, vals = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                v = row.get(col, "")
                if v:
                    try:
                        epochs.append(int(float(row.get("epoch", 0))))
                        vals.append(float(v))
                    except (ValueError, KeyError):
                        pass
        return np.array(epochs), np.array(vals)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))

    for lab, (path, col) in csvs.items():
        if path.exists():
            ep, vl = _read(path)
            if len(ep) > 0:
                ax1.plot(ep, vl, color=col, lw=2.0, label=lab)
    ax1.set_ylabel("Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_title("a   Human-like models", fontweight="bold", loc="left", fontsize=13, pad=8)
    ax1.legend(loc="upper right", fontsize=10)
    _strip_spines(ax1)

    if anogam_csv.exists():
        ep, vl = _read(anogam_csv)
        if len(ep) > 0:
            ax2.plot(ep, vl, color=ORANGE, lw=2.0, label="Tree-aug (tsinfer)")
    ax2.set_ylabel("Validation loss")
    ax2.set_xlabel("Epoch")
    ax2.set_title("b   AnoGam", fontweight="bold", loc="left", fontsize=13, pad=8)
    ax2.legend(loc="upper right", fontsize=10)
    _strip_spines(ax2)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.09, top=0.94, hspace=0.38)
    fig.savefig(OUT / "figS1_combined.png", dpi=300)
    plt.close(fig)
    print(f"  Saved figS1_combined.png")


# ══════════════════════════════════════════════════════════════════════════
# Figure S2: Residuals
# ══════════════════════════════════════════════════════════════════════════

def figS2_residuals():
    print("=== Figure S2: Residuals ===")
    ts_human = simulate_human_ts()
    ts_anogam = load_anogam_ts()
    pairs = [(0, 1), (2, 3), (4, 5)]

    def _residuals(ts, model, use_tsinfer, mu, ws, pairs):
        kw = dict(blocks=[(0, int(ts.sequence_length))], pivot_pairs=pairs,
                  mutation_rate=mu, device=DEVICE, progress=False)
        if use_tsinfer:
            kw["use_tsinfer"] = True
        pm, _, _ = translate_from_ts(ts, model, **kw)
        true_log = []
        for sa, sb in pairs:
            raw = windowed_tmrca(ts, sa, sb, ws, int(ts.sequence_length))
            true_log.append(np.log(np.clip(raw, 1e-10, None)))
        true_log = np.array(true_log)
        n_win = true_log.shape[1]
        pos = np.tile((np.arange(n_win) + 0.5) * ws, len(pairs))
        return pm.ravel() - true_log.ravel(), pos

    specs = [
        ("a   Base (SFS-only)",       BASE_CKPT,    False, ts_human,  HUMAN_MU,  HUMAN_WINDOW,  pairs, BLUE),
        ("b   Tree-aug (true)",       TREES_CKPT,   False, ts_human,  HUMAN_MU,  HUMAN_WINDOW,  pairs, GREEN),
        ("c   Tree-aug (tsinfer)",    TSINFER_CKPT, True,  ts_human,  HUMAN_MU,  HUMAN_WINDOW,  pairs, ORANGE),
        ("d   AnoGam (tsinfer)",      ANOGAM_CKPT,  True,  ts_anogam, ANOGAM_MU, ANOGAM_WINDOW, pairs, ORANGE),
    ]

    fig, axes = plt.subplots(len(specs), 2, figsize=(12, 3.2 * len(specs)),
                             gridspec_kw={"width_ratios": [3.5, 1]})

    for i, (title, ckpt, use_ts, ts, mu, ws, prs, col) in enumerate(specs):
        model = load_model(ckpt)
        resid, pos = _residuals(ts, model, use_ts, mu, ws, prs)
        mu_r, sig_r = resid.mean(), resid.std()
        x_pos = pos / 1e6 if ws == HUMAN_WINDOW else pos / 1e3

        ax_l, ax_r = axes[i, 0], axes[i, 1]

        ax_l.scatter(x_pos, resid, s=1.5, alpha=0.12, c=col, edgecolors="none", rasterized=True)
        ax_l.axhline(0, color=GREY, lw=0.7, ls="--")
        ax_l.set_ylabel("Residual")
        ax_l.set_xlabel("Genomic position (Mb)" if ws == HUMAN_WINDOW else "Genomic position (kb)")
        ax_l.set_title(f"{title}  ($\\mu$={mu_r:.3f}, $\\sigma$={sig_r:.3f})",
                       fontweight="bold", loc="left", fontsize=12, pad=8)
        _strip_spines(ax_l)

        ax_r.hist(resid, bins=60, color=col, alpha=0.7, edgecolor="white",
                  linewidth=0.3, orientation="horizontal")
        ax_r.axhline(0, color=GREY, lw=0.7, ls="--")
        ax_r.set_xlabel("Count")
        ax_r.set_yticklabels([])
        _strip_spines(ax_r)

        ax_l.set_ylim(resid.min() - 0.3, resid.max() + 0.3)
        ax_r.set_ylim(ax_l.get_ylim())

    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.05, top=0.96, hspace=0.50, wspace=0.08)
    fig.savefig(OUT / "figS2_combined.png", dpi=300)
    plt.close(fig)
    print(f"  Saved figS2_combined.png")


# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure2()
    figure3()
    figure4_ood()
    figure5_scaling()
    figS1_training()
    figS2_residuals()
    print(f"\nAll figures saved to {OUT}/")
