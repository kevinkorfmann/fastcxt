#!/usr/bin/env python3
"""Real-data application: pairwise TMRCA dating for Burkina Faso Ag1000G 2L.

Loads the Ag1000G chromosome 2L tree sequence, isolates Burkina Faso
individuals heterozygous for the 2La inversion (2La=1), builds pairwise
focal pairs (within- and between-individual), and runs the trained AnoGam
model to infer TMRCA across 100 kb blocks tiling the full 49.4 Mb arm.

Produces:
  - Gamma-SMC-style heatmap (pairs sorted by median TMRCA)
  - Smoothed population-mean landscape with uncertainty band
  - RDL-locus zoom (supplementary)

Usage:
    python run_burkina_faso.py                    # defaults (25 ind, all pairs)
    python run_burkina_faso.py --n-pairs 50       # cap pairs
    python run_burkina_faso.py --from-cache       # re-plot from saved .npz
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.serialization
import tskit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from scipy.ndimage import uniform_filter1d

# fastcxt imports
from fastcxt.config import FastCxtConfig, TrainingConfig
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_genotype_matrix
from fastcxt.mosquito import (
    AccessibilityMask,
    generate_blocks,
    ANOGAM_CHROMOSOME_ARMS,
)

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
TS_PATH = REPO.parent / "cxt_paper_archive" / "ag1000g" / "gamb.2L.gff.dated.ne.trees"
MASK_PATH = REPO.parent / "cxt_paper_archive" / "ag1000g" / "agp3.is_accessible.txt.npz"
CKPT_PATH = ROOT.parent / "tsinfer_logs" / "lightning_logs" / "version_0" / "checkpoints" / "epoch=29-step=20880.ckpt"

# Model / data constants (must match training)
BLOCK_SIZE = 100_000       # 100 kb
WINDOW_SIZE = 200          # 200 bp -> 500 windows per block
N_WINDOWS = 500
MUTATION_RATE = 3.5e-9     # Anopheles gambiae (stdpopsim)
# The Ag1000G dated tree sequence was produced with mu=1e-9 (tsdate default),
# so tsdate node times are inflated by a factor of 3.5x.  We rescale when
# overlaying tsdate on the plots.
TSDATE_MU = 1e-9
TSDATE_LOG_RESCALE = np.log(TSDATE_MU / MUTATION_RATE)  # -1.253

ARM = "2L"
ARM_LENGTH_MB = ANOGAM_CHROMOSOME_ARMS[ARM] / 1e6

# 2La inversion breakpoints (Mb) — Sharakhova et al. 2006
INV_2LA_START_MB = 20.524
INV_2LA_END_MB = 42.165

# RDL locus (Rdl, resistance to dieldrin) — approximate position on 2L
RDL_START_MB = 25.2
RDL_END_MB = 25.6

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
ORANGE = "#e08214"
BLUE = "#2166ac"
RED = "#b2182b"
GREEN = "#1b7837"
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


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device="cpu"):
    lit = LitFastCxt.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model = lit.model.to(device)
    model.eval()
    return model


def isolate_burkina_faso(ts, require_2La=True):
    """Return (sample_nodes, individual_ids) for Burkina Faso."""
    bf_nodes = []
    bf_ind_ids = []
    for ind in ts.individuals():
        meta = json.loads(ind.metadata.decode()) if ind.metadata else {}
        if meta.get("country") != "Burkina Faso":
            continue
        if require_2La and int(meta.get("2La", 0)) != 1:
            continue
        bf_ind_ids.append(ind.id)
        bf_nodes.extend(ind.nodes)
    return np.array(bf_nodes, dtype=np.int32), bf_ind_ids


def build_pivot_pairs(n_individuals, n_pairs=None, mode="all"):
    """Build pivot pairs from simplified haploid indices.

    After simplification, individual i has haplotypes (2i, 2i+1).

    mode="within"  : within-individual pairs only (n_individuals pairs)
    mode="between" : between-individual pairs only
    mode="all"     : within + between
    """
    within = [(2 * i, 2 * i + 1) for i in range(n_individuals)]
    between = []
    for i, j in itertools.combinations(range(n_individuals), 2):
        # pair first haplotype of each individual
        between.append((2 * i, 2 * j))

    if mode == "within":
        pairs = within
    elif mode == "between":
        pairs = between
    else:
        pairs = within + between

    if n_pairs is not None and len(pairs) > n_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pairs), n_pairs, replace=False)
        idx.sort()
        pairs = [pairs[i] for i in idx]

    return pairs


def simplify_and_extract(ts, sample_nodes):
    ts_simple = ts.simplify(samples=sample_nodes)
    gm = ts_simple.genotype_matrix().T
    positions = ts_simple.tables.sites.position.astype(np.float64)
    return ts_simple, gm, positions


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(model, gm, positions, pivot_pairs, blocks, mask, device,
                  ts_simplified=None):
    """Run inference block-by-block with tsinfer tree features.

    If ts_simplified is provided and the model has a tree encoder, runs
    tsinfer on each block to extract topology features.  Tree features
    are extracted once per block and tiled across pairs, then discarded
    to keep memory usage O(n_pairs * n_windows * feat_dim) per block.
    """
    pos_int = positions.astype(np.int64)
    accessible = mask.mask[pos_int]
    gm_acc = gm[:, accessible]
    pos_acc = positions[accessible]

    block_tuples = [(b.start, b.end) for b in blocks]
    n_blocks = len(block_tuples)
    n_pairs = len(pivot_pairs)

    use_tsinfer = (ts_simplified is not None
                   and getattr(model, "tree_encoder", None) is not None)

    print(f"  Sites total: {gm.shape[1]}, accessible: {gm_acc.shape[1]}")
    print(f"  Blocks: {n_blocks}, Pairs: {n_pairs}")
    print(f"  Total inference items: {n_blocks * n_pairs}")
    print(f"  tsinfer tree features: {'yes' if use_tsinfer else 'no'}")

    if not use_tsinfer:
        # Simple path: all blocks at once, no tree features
        means, variances, index_map = translate_from_genotype_matrix(
            gm=gm_acc, positions=pos_acc, model=model,
            blocks=block_tuples, pivot_pairs=pivot_pairs,
            mutation_rate=MUTATION_RATE, device=device,
            batch_size=64, build_workers=4, progress=True,
        )
        return means, variances, index_map

    # --- Tree-augmented path ---
    # The Ag1000G tree sequence is already tsinfer-inferred and dated,
    # so we extract topology features directly from the existing trees.
    # We compute features for the whole genome in one pass, then slice
    # per block and tile across pairs.
    from fastcxt.translate import _build_sources, predict
    from fastcxt.tree_utils import extract_topology_features, FEATS_PER_NODE
    import torch, gc

    tree_feat_dim = getattr(model.config, "tree_feat_dim", None)
    max_internal = tree_feat_dim // FEATS_PER_NODE if tree_feat_dim else (ts_simplified.num_samples - 1)
    window_size = int((block_tuples[0][1] - block_tuples[0][0]) / 500)
    n_win = 500  # windows per block

    # One-pass feature extraction over the full simplified tree sequence
    total_windows = int(ts_simplified.sequence_length / window_size)
    print(f"  Extracting topology features ({total_windows} windows, {window_size} bp)...")
    all_feats = extract_topology_features(
        ts_simplified, n_windows=total_windows,
        window_size=window_size, max_internal=max_internal,
    )  # shape: (total_windows, feat_dim)
    print(f"  Tree features extracted: {all_feats.shape}")

    all_means = []
    all_vars = []
    all_idx = []

    from fastcxt.sfs import basic_filtering
    gm_filt, pos_filt = basic_filtering(gm_acc, pos_acc)

    from tqdm.auto import tqdm
    for b_idx, (bstart, bend) in enumerate(tqdm(block_tuples, desc="Blocks")):
        # Slice tree features for this block
        w_start = bstart // window_size
        w_end = w_start + n_win
        if w_end > all_feats.shape[0]:
            w_end = all_feats.shape[0]
        tf_block = all_feats[w_start:w_end]

        # Pad if needed
        if tf_block.shape[0] < n_win:
            pad = np.zeros((n_win - tf_block.shape[0], tf_block.shape[1]), dtype=np.float32)
            tf_block = np.concatenate([tf_block, pad], axis=0)

        # Tile across pairs
        tf_tiled = np.tile(tf_block[np.newaxis], (n_pairs, 1, 1))

        # Build SFS sources for this block
        X, idx_map = _build_sources(
            gm_filt, pos_filt, [(bstart, bend)], pivot_pairs,
            window_size=window_size, workers=4, progress=False,
        )

        param_dtype = next(model.parameters()).dtype
        X_t = torch.as_tensor(X, dtype=param_dtype)
        tf_t = torch.as_tensor(tf_tiled, dtype=param_dtype)

        m, v = predict(
            model, X_t, mutation_rate=MUTATION_RATE,
            batch_size=64, device=device, tree_feats=tf_t, progress=False,
        )

        idx_fixed = idx_map.copy()
        idx_fixed[:, 0] = b_idx

        all_means.append(m)
        all_vars.append(v)
        all_idx.append(idx_fixed)

        del X_t, tf_t, tf_tiled, X
        gc.collect()

    means = np.concatenate(all_means, axis=0)
    variances = np.concatenate(all_vars, axis=0)
    index_map = np.concatenate(all_idx, axis=0)

    del all_feats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return means, variances, index_map


# ---------------------------------------------------------------------------
# Helper: assemble per-pair genome-wide profiles
# ---------------------------------------------------------------------------

def assemble_profiles(means, variances, index_map, blocks):
    """Assemble per-pair, per-block data into (n_pairs, total_windows) arrays."""
    n_blocks = len(blocks)
    n_pairs = index_map[:, 1].max() + 1

    # Build block-sorted pair arrays
    mu_grid = np.full((n_pairs, n_blocks, N_WINDOWS), np.nan, dtype=np.float32)
    var_grid = np.full_like(mu_grid, np.nan)

    for row_idx in range(len(index_map)):
        b_idx, p_idx = index_map[row_idx]
        mu_grid[p_idx, b_idx] = means[row_idx]
        var_grid[p_idx, b_idx] = variances[row_idx]

    # Flatten to (n_pairs, n_blocks * N_WINDOWS)
    mu_flat = mu_grid.reshape(n_pairs, -1)
    var_flat = var_grid.reshape(n_pairs, -1)

    # Genomic positions (Mb)
    pos_mb = np.concatenate([
        (np.arange(N_WINDOWS) + 0.5) * WINDOW_SIZE / 1e6 + b.start / 1e6
        for b in blocks
    ])

    return mu_flat, var_flat, pos_mb


# ---------------------------------------------------------------------------
# Diversity-based bias correction (following cxt: kr-colab/cxt)
# ---------------------------------------------------------------------------

def diversity_bias_correction(
    mu_flat: np.ndarray,
    pivot_pairs: list[tuple],
    ts_simplified,
    mutation_rate: float = MUTATION_RATE,
) -> np.ndarray:
    """Per-pair additive log-space shift so predicted diversity matches observed.

    For each pair, computes:
        obs_div  = tskit pairwise diversity (from the tree sequence)
        pred_div = 2 * mu * mean(exp(log_tmrca))  (from predictions)
        shift    = log(obs_div / pred_div)

    Then: corrected = predictions + shift[pair]

    This is the deterministic correction from cxt (kr-colab/cxt).
    """
    n_pairs = mu_flat.shape[0]
    pairs_arr = np.array(pivot_pairs)

    # Observed diversity per pair from the dated tree sequence
    obs_div = ts_simplified.diversity(sample_sets=pairs_arr)

    # Predicted diversity per pair from current TMRCA estimates
    pred_div = 2 * mutation_rate * np.nanmean(np.exp(mu_flat), axis=1)

    # Handle zeros
    obs_div[obs_div == 0] = obs_div[obs_div > 0].min() if (obs_div > 0).any() else 1e-10

    shift = np.log(obs_div) - np.log(pred_div)
    print(f"  Diversity correction: mean shift = {shift.mean():.3f} "
          f"(range [{shift.min():.3f}, {shift.max():.3f}])")

    corrected = mu_flat + shift[:, np.newaxis]
    return corrected


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_heatmap_gamma_smc(mu_flat, pos_mb, pivot_pairs, out_dir,
                             pos_bins=500, tmrca_bins=200):
    """Gamma-SMC-style density heatmap: 2D histogram of (position, TMRCA).

    X = genomic position, Y = log(TMRCA), color = density of pairs.
    Overlays all pair inferences into a single density field.
    """
    print("Generating gamma-SMC-style density heatmap...")

    # Use within-individual pairs only
    within_idx = [i for i, (a, b) in enumerate(pivot_pairs)
                  if a // 2 == b // 2 and b == a + 1]
    mu_within = mu_flat[within_idx] if within_idx else mu_flat
    n_within = mu_within.shape[0]

    # Bin positions
    pos_edges = np.linspace(pos_mb[0], pos_mb[-1], pos_bins + 1)

    # TMRCA range
    tmrca_lo, tmrca_hi = np.nanpercentile(mu_within, [0.5, 99.5])
    tmrca_edges = np.linspace(tmrca_lo, tmrca_hi, tmrca_bins + 1)

    # Build 2D histogram
    density = np.zeros((tmrca_bins, pos_bins), dtype=np.float64)
    pos_bin_idx = np.clip(np.digitize(pos_mb, pos_edges) - 1, 0, pos_bins - 1)

    for p in range(n_within):
        row = mu_within[p]
        valid = ~np.isnan(row)
        if not valid.any():
            continue
        for w in np.where(valid)[0]:
            pb = pos_bin_idx[w]
            tb = np.searchsorted(tmrca_edges, row[w], side="right") - 1
            tb = min(max(tb, 0), tmrca_bins - 1)
            density[tb, pb] += 1

    # Log-scale density for better contrast
    density_log = np.log1p(density)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))

    extent = [pos_mb[0], pos_mb[-1], tmrca_lo, tmrca_hi]
    im = ax.imshow(
        density_log, aspect="auto", cmap="inferno",
        extent=extent, origin="lower",
        interpolation="bilinear", rasterized=True,
    )

    # Mark 2La inversion boundaries
    for xpos in [INV_2LA_START_MB, INV_2LA_END_MB]:
        ax.axvline(xpos, color="white", lw=0.5, ls="--", alpha=0.5)

    # Mark RDL
    ax.axvline((RDL_START_MB + RDL_END_MB) / 2, color=RED, lw=0.5,
               ls="-", alpha=0.6)
    ax.text(RDL_END_MB + 0.3, tmrca_lo + 0.3, "Rdl",
            fontsize=6, color=RED, va="bottom")

    ax.set_xlabel("Position on 2L (Mb)")
    ax.set_ylabel("log(TMRCA)")
    ax.set_xlim(pos_mb[0], pos_mb[-1])

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("log(count + 1)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)

    ax.set_title(
        f"Burkina Faso 2La heterozygotes --- {n_within} within-individual pairs",
        fontweight="bold", loc="left", fontsize=8, pad=5,
    )

    plt.tight_layout()
    fig.savefig(out_dir / "fig_heatmap_2L.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_heatmap_2L.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_heatmap_2L.png/pdf")


def figure_landscape_smoothed(mu_flat, var_flat, pos_mb, out_dir,
                              smooth_kb=150):
    """Population-mean TMRCA landscape with smoothing and +/- 1 s.d. band."""
    print("Generating smoothed population-mean landscape...")

    # Per-window mean and std across all pairs
    pop_mean = np.nanmean(mu_flat, axis=0)
    pop_std = np.nanstd(mu_flat, axis=0)

    # Smooth with a rolling average
    window_spacing_kb = WINDOW_SIZE / 1e3  # 0.2 kb
    kernel = max(1, int(smooth_kb / window_spacing_kb))
    pop_mean_sm = uniform_filter1d(pop_mean, size=kernel, mode="nearest")
    pop_std_sm = uniform_filter1d(pop_std, size=kernel, mode="nearest")

    fig, ax = plt.subplots(figsize=(7.2, 2.2))

    ax.fill_between(
        pos_mb, pop_mean_sm - pop_std_sm, pop_mean_sm + pop_std_sm,
        alpha=0.18, color=ORANGE, lw=0, label=r"$\pm\,1$ s.d. (across pairs)",
    )
    ax.plot(pos_mb, pop_mean_sm, color=ORANGE, lw=0.9, alpha=0.95,
            label="Population mean")

    # Mark 2La inversion
    ax.axvspan(INV_2LA_START_MB, INV_2LA_END_MB,
               alpha=0.06, color=BLUE, zorder=0)
    inv_mid = (INV_2LA_START_MB + INV_2LA_END_MB) / 2
    ymax = ax.get_ylim()[1]
    ax.text(inv_mid, ax.get_ylim()[0] + 0.15, "2La inversion",
            ha="center", fontsize=6.5, color=BLUE, alpha=0.7, style="italic")

    # Mark RDL
    ax.axvspan(RDL_START_MB, RDL_END_MB, alpha=0.15, color=RED, zorder=0)
    ax.text(RDL_END_MB + 0.3, pop_mean_sm[int(RDL_END_MB / ARM_LENGTH_MB * len(pos_mb))],
            "Rdl", fontsize=6.5, color=RED, va="center")

    ax.set_xlabel("Position on 2L (Mb)")
    ax.set_ylabel("Mean log(TMRCA)")
    ax.legend(loc="upper left", fontsize=6.5, handlelength=1.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, ARM_LENGTH_MB)
    # Tight y-limits based on the mean so the inversion step is visible
    mean_lo, mean_hi = np.nanmin(pop_mean_sm), np.nanmax(pop_mean_sm)
    margin = (mean_hi - mean_lo) * 0.15
    ax.set_ylim(mean_lo - margin, mean_hi + margin)

    plt.tight_layout()
    fig.savefig(out_dir / "fig_landscape_2L.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_landscape_2L.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_landscape_2L.png/pdf")


def figure_rdl_zoom(mu_flat, var_flat, pos_mb, pivot_pairs, out_dir,
                    n_traces=8, zoom_start_mb=23.0, zoom_end_mb=28.0):
    """Zoom into the RDL locus showing individual pair TMRCA traces."""
    print("Generating RDL zoom figure...")

    # Select window indices in the zoom region
    mask = (pos_mb >= zoom_start_mb) & (pos_mb <= zoom_end_mb)
    pos_zoom = pos_mb[mask]
    mu_zoom = mu_flat[:, mask]
    var_zoom = var_flat[:, mask]

    # Pick representative pairs: spread across median TMRCA range
    medians = np.nanmedian(mu_zoom, axis=1)
    valid = ~np.isnan(medians)
    valid_idx = np.where(valid)[0]
    sorted_idx = valid_idx[np.argsort(medians[valid_idx])]

    # Evenly sample across the sorted range
    pick_idx = sorted_idx[np.linspace(0, len(sorted_idx) - 1, n_traces, dtype=int)]

    # Smoothing for individual traces
    kernel = 50  # ~10 kb

    cmap = plt.cm.viridis
    colors = [cmap(i / (n_traces - 1)) for i in range(n_traces)]

    fig, ax = plt.subplots(figsize=(7.2, 3.0))

    for i, pidx in enumerate(pick_idx):
        mu_tr = uniform_filter1d(mu_zoom[pidx], size=kernel, mode="nearest")
        std_tr = np.sqrt(uniform_filter1d(var_zoom[pidx], size=kernel, mode="nearest"))
        ax.fill_between(pos_zoom, mu_tr - std_tr, mu_tr + std_tr,
                        alpha=0.08, color=colors[i], lw=0)
        ax.plot(pos_zoom, mu_tr, color=colors[i], lw=0.7, alpha=0.85)

    # Mark RDL
    ax.axvspan(RDL_START_MB, RDL_END_MB, alpha=0.12, color=RED, zorder=0)
    ymin, ymax = ax.get_ylim()
    ax.annotate(
        "Rdl", xy=((RDL_START_MB + RDL_END_MB) / 2, ymin + 0.08 * (ymax - ymin)),
        fontsize=7, ha="center", color=RED, fontweight="bold",
    )

    ax.set_xlabel("Position on 2L (Mb)")
    ax.set_ylabel("log(TMRCA)")
    ax.set_title(
        f"RDL locus region ({zoom_start_mb:.0f}--{zoom_end_mb:.0f} Mb) "
        f"--- {n_traces} representative pairs",
        fontweight="bold", loc="left", fontsize=8, pad=4,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(zoom_start_mb, zoom_end_mb)

    plt.tight_layout()
    fig.savefig(out_dir / "figS_rdl_zoom_2L.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "figS_rdl_zoom_2L.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figS_rdl_zoom_2L.png/pdf")


def figure_combined_main(mu_flat, var_flat, pos_mb, pivot_pairs, out_dir,
                         smooth_kb=150, pos_bins=500, tmrca_bins=200,
                         n_individuals=25, ts_simplified=None):
    """Combined figure for the paper: (a) density heatmap, (b) smoothed landscape."""
    print("Generating combined main figure...")

    # Use within-individual pairs only for both panels
    within_idx = [i for i, (a, b) in enumerate(pivot_pairs)
                  if a // 2 == b // 2 and b == a + 1]
    mu_within = mu_flat[within_idx] if within_idx else mu_flat
    n_within = mu_within.shape[0]

    # --- (a) Gamma-SMC-style density heatmap (within-individual pairs) ---
    pos_edges = np.linspace(pos_mb[0], pos_mb[-1], pos_bins + 1)
    tmrca_lo, tmrca_hi = np.nanpercentile(mu_within, [0.5, 99.5])
    tmrca_edges = np.linspace(tmrca_lo, tmrca_hi, tmrca_bins + 1)

    density = np.zeros((tmrca_bins, pos_bins), dtype=np.float64)
    pos_bin_idx = np.clip(np.digitize(pos_mb, pos_edges) - 1, 0, pos_bins - 1)

    for p in range(n_within):
        row = mu_within[p]
        valid = ~np.isnan(row)
        if not valid.any():
            continue
        for w in np.where(valid)[0]:
            pb = pos_bin_idx[w]
            tb = np.searchsorted(tmrca_edges, row[w], side="right") - 1
            tb = min(max(tb, 0), tmrca_bins - 1)
            density[tb, pb] += 1

    density_log = np.log1p(density)

    # --- (b) Smoothed landscape ---
    pop_mean = np.nanmean(mu_flat, axis=0)
    pop_std = np.nanstd(mu_flat, axis=0)
    window_spacing_kb = WINDOW_SIZE / 1e3
    kernel = max(1, int(smooth_kb / window_spacing_kb))
    pop_mean_sm = uniform_filter1d(pop_mean, size=kernel, mode="nearest")
    pop_std_sm = uniform_filter1d(pop_std, size=kernel, mode="nearest")

    # --- Layout: use GridSpec so colorbar doesn't compress panel (a) ---
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 0.02],
                  height_ratios=[2.2, 1], hspace=0.30, wspace=0.03)
    ax_heat = fig.add_subplot(gs[0, 0])
    ax_cbar = fig.add_subplot(gs[0, 1])
    ax_land = fig.add_subplot(gs[1, 0], sharex=ax_heat)

    # Panel (a): density heatmap
    heat_extent = [pos_mb[0], pos_mb[-1], tmrca_lo, tmrca_hi]
    im = ax_heat.imshow(
        density_log, aspect="auto", cmap="inferno",
        extent=heat_extent, origin="lower",
        interpolation="bilinear", rasterized=True,
    )
    for xpos in [INV_2LA_START_MB, INV_2LA_END_MB]:
        ax_heat.axvline(xpos, color="white", lw=0.5, ls="--", alpha=0.5)

    ax_heat.axvline((RDL_START_MB + RDL_END_MB) / 2, color=RED,
                    lw=0.5, ls="-", alpha=0.5)
    ax_heat.text(RDL_END_MB + 0.3, tmrca_lo + 0.2, "Rdl",
                 fontsize=5.5, color=RED, va="bottom")

    ax_heat.set_ylabel("log(TMRCA)")
    ax_heat.set_title(
        f"a   Burkina Faso 2La heterozygotes --- {n_within} within-individual pairs",
        fontweight="bold", loc="left", fontsize=8, pad=4,
    )
    plt.setp(ax_heat.get_xticklabels(), visible=False)

    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label("log(density)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)

    # Panel (b): block-level mean TMRCA using WITHIN-individual pairs only
    # Within-individual pairs (2i, 2i+1) compare standard vs inverted
    # haplotypes in 2La heterozygotes, giving the clearest inversion signal.
    within_idx = [i for i, (a, b) in enumerate(pivot_pairs)
                  if a // 2 == b // 2 and b == a + 1]
    mu_within = mu_flat[within_idx] if within_idx else mu_flat

    n_blocks = mu_flat.shape[1] // N_WINDOWS
    block_means = np.full(n_blocks, np.nan)
    for bi in range(n_blocks):
        vals = mu_within[:, bi * N_WINDOWS:(bi + 1) * N_WINDOWS]
        block_means[bi] = np.nanmean(vals)
    block_pos_mb = (np.arange(n_blocks) + 0.5) * BLOCK_SIZE / 1e6

    ax_land.plot(block_pos_mb, block_means, color=ORANGE, lw=0.9, alpha=0.95,
                 label="fastcxt")

    # Overlay tsdate ground truth (from the dated tree sequence)
    # tsdate used mu=1e-9; rescale to mu=3.5e-9 so times are comparable.
    # T_rescaled = T_tsdate * (mu_tsdate / mu_real), in log: + log(mu_tsdate/mu_real)
    if ts_simplified is not None:
        within_pairs_raw = [(2 * i, 2 * i + 1) for i in range(n_individuals)
                            if 2 * i + 1 < ts_simplified.num_samples]
        tsdate_block_means = np.full(n_blocks, np.nan)
        tree_iter = ts_simplified.trees()
        current_tree = next(tree_iter)
        for bi in range(n_blocks):
            block_mid = bi * BLOCK_SIZE + BLOCK_SIZE // 2
            while current_tree.interval.right <= block_mid:
                try:
                    current_tree = next(tree_iter)
                except StopIteration:
                    break
            pair_vals = []
            for sa, sb in within_pairs_raw:
                m = current_tree.mrca(sa, sb)
                if m != tskit.NULL:
                    pair_vals.append(np.log(max(current_tree.time(m), 1e-10)))
            if pair_vals:
                tsdate_block_means[bi] = np.mean(pair_vals)
        # Rescale tsdate times from mu=1e-9 to mu=3.5e-9
        tsdate_block_means += TSDATE_LOG_RESCALE
        ax_land.plot(block_pos_mb, tsdate_block_means, color=BLUE, lw=0.9,
                     alpha=0.85, label="tsdate (rescaled)")

    # Mark 2La inversion
    ax_land.axvspan(INV_2LA_START_MB, INV_2LA_END_MB,
                    alpha=0.06, color=BLUE, zorder=0)
    inv_mid = (INV_2LA_START_MB + INV_2LA_END_MB) / 2

    # Mark RDL
    ax_land.axvspan(RDL_START_MB, RDL_END_MB, alpha=0.15, color=RED, zorder=0)
    rdl_bi = np.argmin(np.abs(block_pos_mb - RDL_END_MB))
    ax_land.text(RDL_END_MB + 0.4, block_means[rdl_bi], "Rdl",
                 fontsize=6, color=RED, va="center")

    ax_land.set_xlabel("Position on 2L (Mb)")
    ax_land.set_ylabel("Mean log(TMRCA)")
    ax_land.legend(loc="upper left", fontsize=6, handlelength=1.2)
    ax_land.spines["top"].set_visible(False)
    ax_land.spines["right"].set_visible(False)
    ax_land.set_xlim(0, ARM_LENGTH_MB)
    # Tight y-limits based on both traces
    all_vals = [block_means]
    if ts_simplified is not None:
        all_vals.append(tsdate_block_means)
    all_concat = np.concatenate(all_vals)
    mean_lo, mean_hi = np.nanmin(all_concat), np.nanmax(all_concat)
    margin = (mean_hi - mean_lo) * 0.12
    ax_land.set_ylim(mean_lo - margin, mean_hi + margin)
    ax_land.text(inv_mid, ax_land.get_ylim()[0] + margin * 0.3, "2La",
                 ha="center", fontsize=6, color=BLUE, alpha=0.6, style="italic")
    ax_land.set_title("b   Population-mean TMRCA (within-individual pairs)",
                      fontweight="bold", loc="left", fontsize=8, pad=4)

    # Hide the empty cell at gs[1,1]
    ax_empty = fig.add_subplot(gs[1, 1])
    ax_empty.axis("off")

    fig.savefig(out_dir / "fig_ag1000g_2L.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_ag1000g_2L.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_ag1000g_2L.png/pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Pairwise TMRCA dating for Burkina Faso Ag1000G 2L samples",
    )
    ap.add_argument("--ts", type=str, default=str(TS_PATH))
    ap.add_argument("--mask", type=str, default=str(MASK_PATH))
    ap.add_argument("--checkpoint", type=str, default=str(CKPT_PATH))
    ap.add_argument("--n-pairs", type=int, default=None,
                    help="Max number of pairs (default: all within+between)")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "results"))
    ap.add_argument("--no-2la-filter", action="store_true")
    ap.add_argument("--from-cache", action="store_true",
                    help="Skip inference, re-plot from saved .npz")
    args = ap.parse_args()

    _setup_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cache_path = out_dir / "tmrca_burkina_faso_2L.npz"

    if args.from_cache and cache_path.exists():
        print(f"Loading cached results from {cache_path}")
        data = np.load(cache_path)
        means, variances, index_map = data["means"], data["variances"], data["index_map"]
        pivot_pairs = [tuple(p) for p in data["pivot_pairs"]]
        block_starts = data["block_starts"]
        block_ends = data["block_ends"]

        # Reconstruct blocks as simple objects
        from types import SimpleNamespace
        blocks = [SimpleNamespace(start=int(s), end=int(e), block_idx=i)
                  for i, (s, e) in enumerate(zip(block_starts, block_ends))]

    else:
        # -- Load model --
        print(f"Loading model from {args.checkpoint}")
        model = load_model(args.checkpoint, device=device)

        # -- Load tree sequence --
        print(f"Loading tree sequence from {args.ts}")
        ts = tskit.load(args.ts)
        print(f"  Full TS: {ts.num_samples} samples, {ts.num_trees} trees, "
              f"{ts.sequence_length/1e6:.1f} Mb, {ts.num_sites} sites")

        # -- Isolate Burkina Faso --
        require_2La = not args.no_2la_filter
        bf_nodes, bf_ind_ids = isolate_burkina_faso(ts, require_2La=require_2La)
        n_ind = len(bf_ind_ids)
        print(f"  Burkina Faso: {n_ind} individuals, {len(bf_nodes)} haploid samples"
              f"{' (2La=1 only)' if require_2La else ''}")

        if n_ind < 2:
            raise RuntimeError("Too few Burkina Faso individuals to form pairs")

        # -- Simplify --
        print("Simplifying tree sequence to Burkina Faso samples...")
        ts_bf, gm, positions = simplify_and_extract(ts, bf_nodes)
        print(f"  Simplified: {ts_bf.num_samples} samples, {ts_bf.num_sites} sites")

        # -- Build pivot pairs --
        pivot_pairs = build_pivot_pairs(n_ind, n_pairs=args.n_pairs, mode="all")
        print(f"  Pivot pairs: {len(pivot_pairs)} (within + between individual)")

        # -- Load accessibility mask --
        print("Loading accessibility mask...")
        mask = AccessibilityMask.from_npz(args.mask, ARM)
        print(f"  Accessible fraction: {mask.accessible_fraction:.3f}")

        # -- Generate blocks --
        blocks = generate_blocks(ARM, block_size=BLOCK_SIZE, min_block_fraction=1.0)
        print(f"  Blocks: {len(blocks)} x {BLOCK_SIZE/1e3:.0f} kb")

        # -- Run inference --
        print("\nRunning pairwise TMRCA inference...")
        means, variances, index_map = run_inference(
            model, gm, positions, pivot_pairs, blocks, mask, device,
            ts_simplified=ts_bf,
        )
        print(f"  Output shape: means={means.shape}, variances={variances.shape}")

        # -- Save --
        np.savez_compressed(
            cache_path,
            means=means,
            variances=variances,
            index_map=index_map,
            pivot_pairs=np.array(pivot_pairs),
            block_starts=np.array([b.start for b in blocks]),
            block_ends=np.array([b.end for b in blocks]),
        )
        print(f"  Saved {cache_path.name}")

    # -- Assemble profiles --
    mu_flat, var_flat, pos_mb = assemble_profiles(means, variances, index_map, blocks)

    # -- Load simplified tree sequence (needed for correction + tsdate overlay) --
    ts_bf = None
    try:
        ts_full = tskit.load(args.ts)
        require_2La = not args.no_2la_filter
        bf_nodes, _ = isolate_burkina_faso(ts_full, require_2La=require_2La)
        ts_bf = ts_full.simplify(samples=bf_nodes)
        print(f"  Loaded simplified TS ({ts_bf.num_samples} samples)")
    except Exception as e:
        print(f"  Warning: could not load TS: {e}")

    # -- Per-pair diversity bias correction (cxt-style) --
    if ts_bf is not None:
        print("Applying diversity-based bias correction...")
        mu_flat = diversity_bias_correction(mu_flat, pivot_pairs, ts_bf)

    # -- Generate figures --
    print(f"\nGenerating figures ({len(pivot_pairs)} pairs, {len(blocks)} blocks)...")

    figure_heatmap_gamma_smc(mu_flat, pos_mb, pivot_pairs, out_dir)
    figure_landscape_smoothed(mu_flat, var_flat, pos_mb, out_dir)
    figure_rdl_zoom(mu_flat, var_flat, pos_mb, pivot_pairs, out_dir)

    figure_combined_main(mu_flat, var_flat, pos_mb, pivot_pairs, out_dir,
                         ts_simplified=ts_bf)

    # -- Summary --
    overall_mean = np.nanmean(mu_flat)
    overall_std = np.nanstd(mu_flat)
    print(f"\n{'='*60}")
    print(f"  Burkina Faso 2L --- {len(pivot_pairs)} pairs, {len(blocks)} blocks")
    print(f"  Overall mean log(TMRCA): {overall_mean:.3f}")
    print(f"  Overall std  log(TMRCA): {overall_std:.3f}")
    print(f"  Results saved to {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
