#!/usr/bin/env python3
"""Real-data application: pairwise TMRCA dating for Burkina Faso Ag1000G 2L.

Large-context version (800 kb blocks, 4000 windows per block).

Runs the trained base_trees_4k model in TWO modes on the same checkpoint:
  1. Pairwise mode: SFS only (no tree features)
  2. Tsinfer mode:  SFS + tree topology features

Produces comparison figures for both strategies.

Usage:
    python run_burkina_faso.py                    # defaults
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
CKPT_PATH = ROOT.parent / "tsinfer_4k_logs" / "lightning_logs" / "version_0" / "checkpoints"

# Model / data constants (must match training)
BLOCK_SIZE = 800_000       # 800 kb
WINDOW_SIZE = 200          # 200 bp -> 4000 windows per block
N_WINDOWS = 4000
MUTATION_RATE = 3.5e-9     # Anopheles gambiae (stdpopsim)
TSDATE_MU = 1e-9
TSDATE_LOG_RESCALE = np.log(TSDATE_MU / MUTATION_RATE)

ARM = "2L"
ARM_LENGTH_MB = ANOGAM_CHROMOSOME_ARMS[ARM] / 1e6

# 2La inversion breakpoints (Mb)
INV_2LA_START_MB = 20.524
INV_2LA_END_MB = 42.165

# RDL locus
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
PURPLE = "#7b3294"


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
# Data preparation (same as experiment_mossies version)
# ---------------------------------------------------------------------------

def load_model(ckpt_path, device="cpu"):
    lit = LitFastCxt.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model = lit.model.to(device)
    model.eval()
    return model


def isolate_burkina_faso(ts, require_2La=True):
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
    within = [(2 * i, 2 * i + 1) for i in range(n_individuals)]
    between = []
    for i, j in itertools.combinations(range(n_individuals), 2):
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

def run_inference_pairwise(model, gm, positions, pivot_pairs, blocks, mask, device):
    """Run inference in pairwise (SFS-only) mode -- no tree features."""
    pos_int = positions.astype(np.int64)
    accessible = mask.mask[pos_int]
    gm_acc = gm[:, accessible]
    pos_acc = positions[accessible]

    block_tuples = [(b.start, b.end) for b in blocks]
    print(f"  [pairwise] Sites: {gm_acc.shape[1]}, Blocks: {len(block_tuples)}, Pairs: {len(pivot_pairs)}")

    means, variances, index_map = translate_from_genotype_matrix(
        gm=gm_acc, positions=pos_acc, model=model,
        blocks=block_tuples, pivot_pairs=pivot_pairs,
        mutation_rate=MUTATION_RATE, device=device,
        batch_size=32, build_workers=4, progress=True,
    )
    return means, variances, index_map


def run_inference_tsinfer(model, gm, positions, pivot_pairs, blocks, mask, device,
                          ts_simplified=None):
    """Run inference with tsinfer tree features extracted from the dated TS."""
    import gc
    from fastcxt.translate import _build_sources, predict
    from fastcxt.tree_utils import extract_topology_features, FEATS_PER_NODE
    from fastcxt.sfs import basic_filtering
    from tqdm.auto import tqdm

    pos_int = positions.astype(np.int64)
    accessible = mask.mask[pos_int]
    gm_acc = gm[:, accessible]
    pos_acc = positions[accessible]

    block_tuples = [(b.start, b.end) for b in blocks]
    n_pairs = len(pivot_pairs)
    n_win = N_WINDOWS

    tree_feat_dim = getattr(model.config, "tree_feat_dim", None)
    max_internal = tree_feat_dim // FEATS_PER_NODE if tree_feat_dim else (ts_simplified.num_samples - 1)

    # One-pass feature extraction over the full simplified tree sequence
    total_windows = int(ts_simplified.sequence_length / WINDOW_SIZE)
    print(f"  [tsinfer] Extracting topology features ({total_windows} windows)...")
    all_feats = extract_topology_features(
        ts_simplified, n_windows=total_windows,
        window_size=WINDOW_SIZE, max_internal=max_internal,
    )
    print(f"  Tree features: {all_feats.shape}")

    gm_filt, pos_filt = basic_filtering(gm_acc, pos_acc)

    all_means, all_vars, all_idx = [], [], []

    for b_idx, (bstart, bend) in enumerate(tqdm(block_tuples, desc="Blocks (tsinfer)")):
        w_start = bstart // WINDOW_SIZE
        w_end = w_start + n_win
        if w_end > all_feats.shape[0]:
            w_end = all_feats.shape[0]
        tf_block = all_feats[w_start:w_end]

        if tf_block.shape[0] < n_win:
            pad = np.zeros((n_win - tf_block.shape[0], tf_block.shape[1]), dtype=np.float32)
            tf_block = np.concatenate([tf_block, pad], axis=0)

        tf_tiled = np.tile(tf_block[np.newaxis], (n_pairs, 1, 1))

        X, idx_map = _build_sources(
            gm_filt, pos_filt, [(bstart, bend)], pivot_pairs,
            window_size=WINDOW_SIZE, workers=4, progress=False,
        )

        param_dtype = next(model.parameters()).dtype
        X_t = torch.as_tensor(X, dtype=param_dtype)
        tf_t = torch.as_tensor(tf_tiled, dtype=param_dtype)

        m, v = predict(
            model, X_t, mutation_rate=MUTATION_RATE,
            batch_size=32, device=device, tree_feats=tf_t, progress=False,
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
# Helpers
# ---------------------------------------------------------------------------

def assemble_profiles(means, variances, index_map, blocks):
    n_blocks = len(blocks)
    n_pairs = index_map[:, 1].max() + 1

    mu_grid = np.full((n_pairs, n_blocks, N_WINDOWS), np.nan, dtype=np.float32)
    var_grid = np.full_like(mu_grid, np.nan)

    for row_idx in range(len(index_map)):
        b_idx, p_idx = index_map[row_idx]
        mu_grid[p_idx, b_idx] = means[row_idx]
        var_grid[p_idx, b_idx] = variances[row_idx]

    mu_flat = mu_grid.reshape(n_pairs, -1)
    var_flat = var_grid.reshape(n_pairs, -1)

    pos_mb = np.concatenate([
        (np.arange(N_WINDOWS) + 0.5) * WINDOW_SIZE / 1e6 + b.start / 1e6
        for b in blocks
    ])

    return mu_flat, var_flat, pos_mb


def diversity_bias_correction(mu_flat, pivot_pairs, ts_simplified,
                              mutation_rate=MUTATION_RATE):
    n_pairs = mu_flat.shape[0]
    pairs_arr = np.array(pivot_pairs)
    obs_div = ts_simplified.diversity(sample_sets=pairs_arr)
    pred_div = 2 * mutation_rate * np.nanmean(np.exp(mu_flat), axis=1)
    obs_div[obs_div == 0] = obs_div[obs_div > 0].min() if (obs_div > 0).any() else 1e-10
    shift = np.log(obs_div) - np.log(pred_div)
    print(f"  Diversity correction: mean shift = {shift.mean():.3f} "
          f"(range [{shift.min():.3f}, {shift.max():.3f}])")
    corrected = mu_flat + shift[:, np.newaxis]
    return corrected


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_comparison_landscape(mu_pw, mu_ts, pos_mb, pivot_pairs, blocks,
                                out_dir, n_individuals=25, ts_simplified=None,
                                smooth_kb=150):
    """Combined figure comparing pairwise-only vs tsinfer-augmented predictions."""
    print("Generating comparison landscape figure...")

    within_idx = [i for i, (a, b) in enumerate(pivot_pairs)
                  if a // 2 == b // 2 and b == a + 1]

    n_blocks = mu_pw.shape[1] // N_WINDOWS

    def block_means(mu, idx):
        mu_w = mu[idx] if idx else mu
        bm = np.full(n_blocks, np.nan)
        for bi in range(n_blocks):
            bm[bi] = np.nanmean(mu_w[:, bi * N_WINDOWS:(bi + 1) * N_WINDOWS])
        return bm

    bm_pw = block_means(mu_pw, within_idx)
    bm_ts = block_means(mu_ts, within_idx)
    block_pos_mb = (np.arange(n_blocks) + 0.5) * BLOCK_SIZE / 1e6

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.0), sharex=True)

    # Panel a: pairwise vs tsinfer overlay
    ax1.plot(block_pos_mb, bm_pw, color=ORANGE, lw=0.9, alpha=0.95, label="Pairwise (SFS only)")
    ax1.plot(block_pos_mb, bm_ts, color=PURPLE, lw=0.9, alpha=0.95, label="Tsinfer-augmented")

    if ts_simplified is not None:
        within_pairs_raw = [(2 * i, 2 * i + 1) for i in range(n_individuals)
                            if 2 * i + 1 < ts_simplified.num_samples]
        tsdate_bm = np.full(n_blocks, np.nan)
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
                tsdate_bm[bi] = np.mean(pair_vals)
        tsdate_bm += TSDATE_LOG_RESCALE
        ax1.plot(block_pos_mb, tsdate_bm, color=BLUE, lw=0.9, alpha=0.85,
                 label="tsdate (rescaled)")

    ax1.axvspan(INV_2LA_START_MB, INV_2LA_END_MB, alpha=0.06, color=BLUE, zorder=0)
    ax1.axvspan(RDL_START_MB, RDL_END_MB, alpha=0.15, color=RED, zorder=0)
    ax1.set_ylabel("Mean log(TMRCA)")
    ax1.legend(loc="upper left", fontsize=6, handlelength=1.2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("a   Pairwise vs tsinfer-augmented (within-individual pairs)",
                  fontweight="bold", loc="left", fontsize=8, pad=4)

    # Panel b: difference (tsinfer - pairwise)
    diff = bm_ts - bm_pw
    ax2.bar(block_pos_mb, diff, width=BLOCK_SIZE / 1e6 * 0.8,
            color=np.where(diff >= 0, PURPLE, ORANGE), alpha=0.6)
    ax2.axhline(0, color=GREY, lw=0.5, ls="--")
    ax2.axvspan(INV_2LA_START_MB, INV_2LA_END_MB, alpha=0.06, color=BLUE, zorder=0)
    ax2.set_xlabel("Position on 2L (Mb)")
    ax2.set_ylabel("Tsinfer - Pairwise")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("b   Tree feature contribution",
                  fontweight="bold", loc="left", fontsize=8, pad=4)
    ax2.set_xlim(0, ARM_LENGTH_MB)

    plt.tight_layout(h_pad=0.8)
    fig.savefig(out_dir / "fig_comparison_4k_2L.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_comparison_4k_2L.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_comparison_4k_2L.png/pdf")


def figure_heatmap(mu_flat, pos_mb, pivot_pairs, out_dir, label="",
                   pos_bins=500, tmrca_bins=200):
    """Gamma-SMC-style density heatmap."""
    print(f"Generating heatmap ({label})...")

    within_idx = [i for i, (a, b) in enumerate(pivot_pairs)
                  if a // 2 == b // 2 and b == a + 1]
    mu_within = mu_flat[within_idx] if within_idx else mu_flat
    n_within = mu_within.shape[0]

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

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    extent = [pos_mb[0], pos_mb[-1], tmrca_lo, tmrca_hi]
    im = ax.imshow(
        density_log, aspect="auto", cmap="inferno",
        extent=extent, origin="lower",
        interpolation="bilinear", rasterized=True,
    )

    for xpos in [INV_2LA_START_MB, INV_2LA_END_MB]:
        ax.axvline(xpos, color="white", lw=0.5, ls="--", alpha=0.5)
    ax.axvline((RDL_START_MB + RDL_END_MB) / 2, color=RED, lw=0.5, ls="-", alpha=0.6)
    ax.text(RDL_END_MB + 0.3, tmrca_lo + 0.3, "Rdl", fontsize=6, color=RED, va="bottom")

    ax.set_xlabel("Position on 2L (Mb)")
    ax.set_ylabel("log(TMRCA)")
    ax.set_xlim(pos_mb[0], pos_mb[-1])

    cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02, aspect=25)
    cbar.set_label("log(count + 1)", fontsize=6.5)
    cbar.ax.tick_params(labelsize=5.5)

    ax.set_title(
        f"Burkina Faso 2La --- {n_within} within-ind pairs --- {label}",
        fontweight="bold", loc="left", fontsize=8, pad=5,
    )

    plt.tight_layout()
    suffix = label.lower().replace(" ", "_").replace("-", "_")
    fig.savefig(out_dir / f"fig_heatmap_4k_{suffix}_2L.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"fig_heatmap_4k_{suffix}_2L.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig_heatmap_4k_{suffix}_2L.png/pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Pairwise TMRCA dating for Burkina Faso Ag1000G 2L (4k context)",
    )
    ap.add_argument("--ts", type=str, default=str(TS_PATH))
    ap.add_argument("--mask", type=str, default=str(MASK_PATH))
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint (auto-detected from tsinfer_4k_logs if not given)")
    ap.add_argument("--n-pairs", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "results"))
    ap.add_argument("--no-2la-filter", action="store_true")
    ap.add_argument("--from-cache", action="store_true")
    args = ap.parse_args()

    _setup_style()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_dir = CKPT_PATH
        if ckpt_dir.is_dir():
            ckpts = sorted(ckpt_dir.glob("*.ckpt"))
            if ckpts:
                ckpt_path = str(ckpts[-1])
        if ckpt_path is None:
            raise FileNotFoundError(
                f"No checkpoint found in {ckpt_dir}. Provide --checkpoint.")

    cache_pw = out_dir / "tmrca_burkina_faso_4k_pairwise.npz"
    cache_ts = out_dir / "tmrca_burkina_faso_4k_tsinfer.npz"

    if args.from_cache and cache_pw.exists() and cache_ts.exists():
        print("Loading cached results...")
        data_pw = np.load(cache_pw)
        data_ts = np.load(cache_ts)
        means_pw, vars_pw, idx_pw = data_pw["means"], data_pw["variances"], data_pw["index_map"]
        means_ts, vars_ts, idx_ts = data_ts["means"], data_ts["variances"], data_ts["index_map"]
        pivot_pairs = [tuple(p) for p in data_pw["pivot_pairs"]]
        from types import SimpleNamespace
        blocks = [SimpleNamespace(start=int(s), end=int(e), block_idx=i)
                  for i, (s, e) in enumerate(zip(data_pw["block_starts"], data_pw["block_ends"]))]
    else:
        # Load model
        print(f"Loading model from {ckpt_path}")
        model = load_model(ckpt_path, device=device)

        # Load tree sequence
        print(f"Loading tree sequence from {args.ts}")
        ts = tskit.load(args.ts)
        print(f"  Full TS: {ts.num_samples} samples, {ts.num_trees} trees, "
              f"{ts.sequence_length/1e6:.1f} Mb, {ts.num_sites} sites")

        # Isolate Burkina Faso
        require_2La = not args.no_2la_filter
        bf_nodes, bf_ind_ids = isolate_burkina_faso(ts, require_2La=require_2La)
        n_ind = len(bf_ind_ids)
        print(f"  Burkina Faso: {n_ind} individuals, {len(bf_nodes)} haploid samples"
              f"{' (2La=1 only)' if require_2La else ''}")

        if n_ind < 2:
            raise RuntimeError("Too few Burkina Faso individuals")

        # Simplify
        print("Simplifying tree sequence...")
        ts_bf, gm, positions = simplify_and_extract(ts, bf_nodes)
        print(f"  Simplified: {ts_bf.num_samples} samples, {ts_bf.num_sites} sites")

        # Build pivot pairs
        pivot_pairs = build_pivot_pairs(n_ind, n_pairs=args.n_pairs, mode="all")
        print(f"  Pivot pairs: {len(pivot_pairs)}")

        # Load accessibility mask
        mask = AccessibilityMask.from_npz(args.mask, ARM)
        print(f"  Accessible fraction: {mask.accessible_fraction:.3f}")

        # Generate blocks (800 kb)
        blocks = generate_blocks(ARM, block_size=BLOCK_SIZE, min_block_fraction=1.0)
        print(f"  Blocks: {len(blocks)} x {BLOCK_SIZE/1e3:.0f} kb")

        # Mode 1: Pairwise (SFS only)
        print("\n--- Mode 1: Pairwise (SFS only) ---")
        means_pw, vars_pw, idx_pw = run_inference_pairwise(
            model, gm, positions, pivot_pairs, blocks, mask, device,
        )
        print(f"  Output: means={means_pw.shape}")

        # Mode 2: Tsinfer-augmented
        print("\n--- Mode 2: Tsinfer-augmented ---")
        means_ts, vars_ts, idx_ts = run_inference_tsinfer(
            model, gm, positions, pivot_pairs, blocks, mask, device,
            ts_simplified=ts_bf,
        )
        print(f"  Output: means={means_ts.shape}")

        # Save caches
        block_starts = np.array([b.start for b in blocks])
        block_ends = np.array([b.end for b in blocks])
        for path, m, v, idx in [(cache_pw, means_pw, vars_pw, idx_pw),
                                 (cache_ts, means_ts, vars_ts, idx_ts)]:
            np.savez_compressed(path, means=m, variances=v, index_map=idx,
                                pivot_pairs=np.array(pivot_pairs),
                                block_starts=block_starts, block_ends=block_ends)
        print(f"  Saved caches")

    # Assemble profiles
    mu_pw, var_pw, pos_mb = assemble_profiles(means_pw, vars_pw, idx_pw, blocks)
    mu_ts, var_ts, _ = assemble_profiles(means_ts, vars_ts, idx_ts, blocks)

    # Load simplified TS for correction + overlay
    ts_bf = None
    n_ind = 25
    try:
        ts_full = tskit.load(args.ts)
        require_2La = not args.no_2la_filter
        bf_nodes, bf_ind_ids = isolate_burkina_faso(ts_full, require_2La=require_2La)
        n_ind = len(bf_ind_ids)
        ts_bf = ts_full.simplify(samples=bf_nodes)
        print(f"  Loaded simplified TS ({ts_bf.num_samples} samples)")
    except Exception as e:
        print(f"  Warning: could not load TS: {e}")

    # Diversity bias correction
    if ts_bf is not None:
        print("Applying diversity-based bias correction...")
        mu_pw = diversity_bias_correction(mu_pw, pivot_pairs, ts_bf)
        mu_ts = diversity_bias_correction(mu_ts, pivot_pairs, ts_bf)

    # Generate figures
    print(f"\nGenerating figures ({len(pivot_pairs)} pairs, {len(blocks)} blocks)...")

    figure_heatmap(mu_pw, pos_mb, pivot_pairs, out_dir, label="pairwise")
    figure_heatmap(mu_ts, pos_mb, pivot_pairs, out_dir, label="tsinfer")

    figure_comparison_landscape(
        mu_pw, mu_ts, pos_mb, pivot_pairs, blocks, out_dir,
        n_individuals=n_ind, ts_simplified=ts_bf,
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"  Burkina Faso 2L (4k context) --- {len(pivot_pairs)} pairs, {len(blocks)} blocks")
    print(f"  Pairwise mean log(TMRCA): {np.nanmean(mu_pw):.3f}")
    print(f"  Tsinfer  mean log(TMRCA): {np.nanmean(mu_ts):.3f}")
    print(f"  Results saved to {out_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
