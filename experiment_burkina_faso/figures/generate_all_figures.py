#!/usr/bin/env python3
"""Generate all figures for the comprehensive Ag1000G inference run.

Reads results from a local mirror and generates:
  overview/       — cross-population summary heatmaps
  per_chromosome/ — distribution plots: rows=populations, cols=groups
  inversion_signals/ — TMRCA profiles along chromosomes highlighting inversions
  karyotype_comparisons/ — intra vs inter, hom vs het comparisons

Plots that don't have data yet show "PENDING".
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("/tmp/het_results")
FIG_DIR = Path(__file__).parent

POPULATIONS = [
    "Burkina_Faso", "Cameroon", "Central_African_Republic",
    "Democratic_Republic_of_the_Congo", "Equatorial_Guinea", "Gabon",
    "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Mali",
    "Mayotte", "Mozambique", "Tanzania", "Uganda",
]
POP_LABELS = [p.replace("_", " ") for p in POPULATIONS]

ARMS = ["2L", "2R", "3L", "3R", "X"]

ARM_GROUPS = {
    "2L": ["2La_hom_standard", "2La_heterozygous", "2La_hom_inverted"],
    "2R": ["2Rb_hom_standard", "2Rb_heterozygous", "2Rb_hom_inverted"],
    "3L": ["all"], "3R": ["all"], "X": ["all"],
}

PAIR_TYPES = ["intra", "inter"]

# 2La inversion breakpoints (Mb)
INV_2LA = (20.524, 42.165)

GROUP_COLORS = {
    "hom_standard": "#2196F3",
    "heterozygous": "#FF9800",
    "hom_inverted": "#E91E63",
    "all": "#607D8B",
}

def group_color(group_name):
    for key, color in GROUP_COLORS.items():
        if key in group_name:
            return color
    return "#607D8B"


# Gene annotations
GENES_FILE = RESULTS_DIR / "genes.json"
_GENES = None

# Well-known genes of interest (for bold labeling)
NOTABLE_GENES = {
    "vgsc", "Vgsc", "AGAP004707",  # voltage-gated sodium channel (kdr)
    "Rdl", "AGAP006028",            # GABA receptor (dieldrin resistance)
    "Ace1", "AGAP001356",           # acetylcholinesterase
    "CYP6P4", "AGAP002869",         # cytochrome P450
    "CYP9K1", "AGAP000818",         # cytochrome P450
    "Gste2", "AGAP009195",          # glutathione S-transferase
    "Tep1", "AGAP010815",           # thioester-containing protein (immunity)
    "CYP6M2", "AGAP008212",
    "CYP6Z1", "AGAP008219",
}

def _load_genes():
    global _GENES
    if _GENES is None and GENES_FILE.exists():
        with open(GENES_FILE) as f:
            _GENES = json.load(f)
    return _GENES


def _find_gene_at(arm, pos_bp):
    """Find the gene overlapping a position (in bp).
    Returns a short display name: symbol if available, else short description, else AGAP ID."""
    genes = _load_genes()
    if genes is None or arm not in genes:
        return None

    def _display_name(g):
        if g.get("symbol"):
            return g["symbol"]
        desc = g.get("description", "")
        if desc and desc != "unspecified product":
            # Shorten long descriptions
            short = desc.split(",")[0].split("(")[0].strip()
            if len(short) > 25:
                short = short[:22] + "..."
            return short
        return g.get("id", g.get("name", ""))

    # Exact overlap
    for g in genes[arm]:
        if g["start"] <= pos_bp <= g["end"]:
            return _display_name(g)
    # Nearest within 50kb
    best_dist = 50_000
    best = None
    for g in genes[arm]:
        mid = (g["start"] + g["end"]) / 2
        dist = abs(mid - pos_bp)
        if dist < best_dist:
            best_dist = dist
            best = g
    return _display_name(best) if best else None


# Accessibility data
ACCESSIBILITY_FILE = RESULTS_DIR / "accessibility_100kb.npz"
_ACC_DATA = None

def _load_accessibility():
    global _ACC_DATA
    if _ACC_DATA is None and ACCESSIBILITY_FILE.exists():
        _ACC_DATA = np.load(ACCESSIBILITY_FILE)
    return _ACC_DATA

def _plot_accessibility_track(ax, arm, xlim=None):
    """Draw a smoothed accessibility track on the given axes.
    Expects ax to have x-axis in Mb."""
    acc = _load_accessibility()
    if acc is None:
        return
    key_frac = f"{arm}_frac"
    key_mids = f"{arm}_mids"
    if key_frac not in acc:
        return
    frac = acc[key_frac]
    mids = acc[key_mids]

    # Light smooth (window=3 blocks = 300kb)
    kernel = 3
    if len(frac) > kernel:
        smooth = np.convolve(frac, np.ones(kernel)/kernel, mode="same")
    else:
        smooth = frac

    # Inaccessible fraction (1 - accessible)
    missing = 1.0 - smooth

    ax.fill_between(mids, 0, missing, color="#E53935", alpha=0.35, linewidth=0)
    ax.plot(mids, missing, color="#E53935", lw=0.6, alpha=0.6)
    ax.set_ylim(0, max(0.8, missing.max() * 1.2))
    ax.set_ylabel("Missing", fontsize=9, color="#E53935")
    ax.tick_params(labelsize=8, colors="#999")
    ax.set_xlim(xlim if xlim else (mids.min(), mids.max()))
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_color("#ddd")


def pretty_group(name):
    return (name
            .replace("2La_", "").replace("2Rb_", "")
            .replace("hom_standard", "Hom Std")
            .replace("heterozygous", "Het")
            .replace("hom_inverted", "Hom Inv")
            .replace("all", "All"))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_group(pop, arm, group, pair_type):
    """Load means and block info for one group. Returns None if missing."""
    d = RESULTS_DIR / pop / arm / f"{group}_{pair_type}"
    if not (d / "means.npz").exists():
        return None
    means = np.load(d / "means.npz")["means"]
    index_map = np.load(d / "index_map.npy")
    blocks = json.load(open(d / "blocks.json"))
    config = json.load(open(d / "config.json"))
    return dict(means=means, index_map=index_map, blocks=blocks, config=config)


def block_means(data):
    """Compute per-block mean TMRCA from raw data."""
    means = data["means"]
    idx = data["index_map"]
    blocks = data["blocks"]
    n_blocks = len(blocks)
    n_pairs = int(idx[:, 1].max()) + 1
    n_win = means.shape[1]

    arr = np.full((n_blocks, n_pairs, n_win), np.nan)
    for i in range(len(idx)):
        arr[idx[i, 0], idx[i, 1], :] = means[i]

    return np.nanmean(arr, axis=(1, 2))  # (n_blocks,)


def block_mids_mb(data):
    """Block midpoints in Mb."""
    blocks = data["blocks"]
    return np.array([(b["start"] + b["end"]) / 2 / 1e6 for b in blocks])


def genome_mean(data):
    """Single scalar: genome-wide mean log-TMRCA."""
    return float(np.nanmean(data["means"]))


def pending_ax(ax, text="PENDING\nAwaiting data"):
    """Fill an axes with a pending message."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, text, ha="center", va="center",
            fontsize=24, color="#bbb", fontstyle="italic", fontweight="bold",
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#ddd")


# ===========================================================================
# Figure 1: Overview heatmap — mean coalescence per population × group
# ===========================================================================

def fig1_overview_heatmap():
    """Heatmap: rows=populations, cols=arm×group×pair_type."""
    cols = []
    col_labels = []
    for arm in ARMS:
        for group in ARM_GROUPS[arm]:
            for pt in PAIR_TYPES:
                cols.append((arm, group, pt))
                col_labels.append(f"{arm}\n{pretty_group(group)}\n{pt}")

    matrix = np.full((len(POPULATIONS), len(cols)), np.nan)
    for i, pop in enumerate(POPULATIONS):
        for j, (arm, group, pt) in enumerate(cols):
            data = load_group(pop, arm, group, pt)
            if data is not None:
                matrix[i, j] = genome_mean(data)

    has_data = ~np.all(np.isnan(matrix), axis=1)

    fig, ax = plt.subplots(figsize=(24, 12))

    if np.any(~np.isnan(matrix)):
        im = ax.imshow(matrix, aspect="auto", cmap="viridis",
                        interpolation="none")
        cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.set_label("Mean log(TMRCA)", fontsize=12)

        # Mark NaN cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if np.isnan(matrix[i, j]):
                    ax.text(j, i, "—", ha="center", va="center",
                            color="#999", fontsize=12)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11, rotation=0, ha="center")
    ax.set_yticks(range(len(POPULATIONS)))
    ax.set_yticklabels(POP_LABELS, fontsize=13)
    ax.set_title("Mean coalescence time — all populations × karyotype groups × pair types",
                 fontsize=14, pad=15)

    # Add arm separators
    pos = 0
    for arm in ARMS:
        n = len(ARM_GROUPS[arm]) * 2
        if pos > 0:
            ax.axvline(pos - 0.5, color="white", lw=2)
        pos += n

    plt.savefig(FIG_DIR / "overview" / "heatmap_all_vs_all.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "overview" / "heatmap_all_vs_all.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved overview/heatmap_all_vs_all")


# ===========================================================================
# Figure 2: Distribution grid — rows=populations, cols=arm groups (intra)
# ===========================================================================

def fig2_distribution_grid():
    """Grid: rows=populations, 6 columns (3 karyotype groups × 2 for 2L/2R, or 1 for 3L/3R/X).
    Shows sorted per-block mean TMRCA distributions for intra pairs."""

    col_specs = []
    for arm in ARMS:
        for group in ARM_GROUPS[arm]:
            col_specs.append((arm, group))
    n_cols = len(col_specs)

    fig, axes = plt.subplots(len(POPULATIONS), n_cols, figsize=(n_cols * 3.5, len(POPULATIONS) * 2.2),
                              squeeze=False)
    fig.suptitle("Sorted block-level coalescence distributions (intra-individual pairs)\n"
                 "Each subplot: sorted per-block mean log(TMRCA)", fontsize=14, y=0.98)

    for i, pop in enumerate(POPULATIONS):
        for j, (arm, group) in enumerate(col_specs):
            ax = axes[i, j]
            data = load_group(pop, arm, group, "intra")

            if i == 0:
                ax.set_title(f"{arm}\n{pretty_group(group)}", fontsize=12, pad=3)
            if j == 0:
                ax.set_ylabel(POP_LABELS[i], fontsize=11, rotation=0, ha="right", va="center")

            if data is None:
                pending_ax(ax)
                continue

            bm = block_means(data)
            sorted_bm = np.sort(bm)
            color = group_color(group)
            ax.fill_between(range(len(sorted_bm)), sorted_bm, alpha=0.4, color=color)
            ax.plot(sorted_bm, lw=0.8, color=color)
            ax.set_xlim(0, len(sorted_bm))
            ax.tick_params(labelsize=9)
            if j > 0:
                ax.set_yticklabels([])

    plt.savefig(FIG_DIR / "per_chromosome" / "distribution_grid_intra.png",
                dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "per_chromosome" / "distribution_grid_intra.pdf",
                bbox_inches="tight")
    plt.close()
    print("  Saved per_chromosome/distribution_grid_intra")


# ===========================================================================
# Figure 2b: Density distribution grid — same layout as fig2 but KDE/hist
# ===========================================================================

def fig2b_density_grid():
    """Grid: rows=populations, cols=arm×group.
    Shows density (histogram + KDE) of per-block mean log(TMRCA) for intra pairs."""

    col_specs = []
    for arm in ARMS:
        for group in ARM_GROUPS[arm]:
            col_specs.append((arm, group))
    n_cols = len(col_specs)

    fig, axes = plt.subplots(len(POPULATIONS), n_cols,
                              figsize=(n_cols * 3.5, len(POPULATIONS) * 2.2),
                              squeeze=False)
    fig.suptitle("Density of block-level coalescence times (intra-individual pairs)\n"
                 "Each subplot: histogram + KDE of per-block mean log(TMRCA)",
                 fontsize=14, y=0.98)

    for i, pop in enumerate(POPULATIONS):
        for j, (arm, group) in enumerate(col_specs):
            ax = axes[i, j]
            data = load_group(pop, arm, group, "intra")

            if i == 0:
                ax.set_title(f"{arm}\n{pretty_group(group)}", fontsize=12, pad=3)
            if j == 0:
                ax.set_ylabel(POP_LABELS[i], fontsize=11, rotation=0, ha="right", va="center")

            if data is None:
                pending_ax(ax)
                continue

            bm = block_means(data)
            color = group_color(group)

            ax.hist(bm, bins=40, density=True, alpha=0.4, color=color, edgecolor="none")

            # Simple KDE
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(bm, bw_method=0.3)
                x = np.linspace(bm.min(), bm.max(), 200)
                ax.plot(x, kde(x), lw=1.5, color=color)
            except Exception:
                pass

            ax.axvline(np.median(bm), color=color, ls="--", lw=1, alpha=0.7)
            ax.tick_params(labelsize=9)
            if j > 0:
                ax.set_yticklabels([])

    plt.savefig(FIG_DIR / "per_chromosome" / "density_grid_intra.png",
                dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "per_chromosome" / "density_grid_intra.pdf",
                bbox_inches="tight")
    plt.close()
    print("  Saved per_chromosome/density_grid_intra")


# ===========================================================================
# Figure 2c: Overlaid density per karyotype group — one plot per population × arm
# ===========================================================================

def fig2c_overlay_densities():
    """For each population: one subplot per arm, overlaying density curves
    for each karyotype group (intra). Shows how karyotype affects TMRCA distribution."""

    for pop in POPULATIONS:
        fig, axes = plt.subplots(1, 5, figsize=(28, 4.5),
                                  gridspec_kw={"wspace": 0.3})
        any_data = False

        for a_idx, arm in enumerate(ARMS):
            ax = axes[a_idx]
            ax.set_title(arm, fontsize=12)

            for group in ARM_GROUPS[arm]:
                data = load_group(pop, arm, group, "intra")
                if data is None:
                    continue
                any_data = True
                bm = block_means(data)
                n = data["config"]["n_pairs"]
                color = group_color(group)
                label = f"{pretty_group(group)} (n={n})"

                ax.hist(bm, bins=50, density=True, alpha=0.25, color=color, edgecolor="none")
                from scipy.stats import gaussian_kde
                try:
                    kde = gaussian_kde(bm, bw_method=0.3)
                    x = np.linspace(bm.min() - 0.5, bm.max() + 0.5, 300)
                    ax.plot(x, kde(x), lw=2, color=color, label=label)
                except Exception:
                    pass

            ax.set_xlabel("log(TMRCA)", fontsize=12)
            if a_idx == 0:
                ax.set_ylabel("Density")
            ax.legend(fontsize=11, loc="upper left")

        if not any_data:
            for ax in axes:
                pending_ax(ax)

        fig.suptitle(f"{pop.replace('_', ' ')} — TMRCA density by karyotype (intra-individual)",
                     fontsize=14, y=1.02)

        plt.savefig(FIG_DIR / "per_population" / f"{pop}_density_overlay.png",
                    dpi=150, bbox_inches="tight")
        plt.savefig(FIG_DIR / "per_population" / f"{pop}_density_overlay.pdf",
                    bbox_inches="tight")
        plt.close()

    print("  Saved per_population/*_density_overlay.png")


# ===========================================================================
# Figure 3: Inversion signal — TMRCA profile along 2L for each karyotype
# ===========================================================================

def fig3_inversion_signals():
    """Per population: TMRCA along 2L for all 3 karyotype groups (intra),
    with accessibility track at the bottom."""
    groups_2L = ARM_GROUPS["2L"]

    for pop in POPULATIONS:
        any_data = False
        for g in groups_2L:
            if load_group(pop, "2L", g, "intra") is not None:
                any_data = True
                break

        fig, axes = plt.subplots(2, 1, figsize=(16, 6.5),
                                  gridspec_kw={"height_ratios": [5, 1], "hspace": 0.08},
                                  sharex=True)
        ax, ax_acc = axes

        if not any_data:
            pending_ax(ax, f"PENDING\n{pop.replace('_', ' ')} — 2L")
            ax_acc.set_visible(False)
        else:
            xlim = None
            for group in groups_2L:
                data = load_group(pop, "2L", group, "intra")
                if data is None:
                    continue
                bm = block_means(data)
                mids = block_mids_mb(data)
                if xlim is None:
                    xlim = (0, mids.max())
                n_pairs = data["config"]["n_pairs"]
                color = group_color(group)
                label = f"{pretty_group(group)} (n={n_pairs})"
                ax.plot(mids, bm, lw=1.5, color=color, alpha=0.8, label=label)

            ax.axvspan(*INV_2LA, alpha=0.08, color="red")
            ax.axvline(INV_2LA[0], color="red", ls="--", lw=1, alpha=0.5)
            ax.axvline(INV_2LA[1], color="red", ls="--", lw=1, alpha=0.5)
            ax.text(sum(INV_2LA) / 2, ax.get_ylim()[1], "In(2L)a",
                    ha="center", va="top", fontsize=11, color="red", fontweight="bold")
            ax.set_ylabel("Mean log(TMRCA)")
            ax.legend(loc="lower right", fontsize=13)
            ax.tick_params(labelbottom=False)

            _plot_accessibility_track(ax_acc, "2L", xlim=xlim)
            ax_acc.set_xlabel("Position on 2L (Mb)", fontsize=12)

        ax.set_title(f"{pop.replace('_', ' ')} — Intra-individual coalescence along 2L by karyotype",
                     fontsize=13)

        plt.savefig(FIG_DIR / "inversion_signals" / f"2L_{pop}.png", dpi=150, bbox_inches="tight")
        plt.savefig(FIG_DIR / "inversion_signals" / f"2L_{pop}.pdf", bbox_inches="tight")
        plt.close()

    print("  Saved inversion_signals/2L_*.png")


# ===========================================================================
# Figure 4: Karyotype comparison — intra vs inter, boxplots
# ===========================================================================

def fig4_karyotype_comparison():
    """For each population with data: boxplot comparing genome-wide TMRCA
    across karyotype groups and pair types."""

    for pop in POPULATIONS:
        entries = []
        for arm in ARMS:
            for group in ARM_GROUPS[arm]:
                for pt in PAIR_TYPES:
                    data = load_group(pop, arm, group, pt)
                    if data is not None:
                        bm = block_means(data)
                        entries.append({
                            "arm": arm, "group": pretty_group(group),
                            "pair_type": pt, "block_means": bm,
                            "color": group_color(group),
                        })

        if not entries:
            fig, ax = plt.subplots(figsize=(12, 5))
            pending_ax(ax, f"PENDING\n{pop.replace('_', ' ')}")
            ax.set_title(f"{pop.replace('_', ' ')} — Karyotype comparison")
            plt.savefig(FIG_DIR / "karyotype_comparisons" / f"{pop}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()
            continue

        fig, ax = plt.subplots(figsize=(max(len(entries) * 1.2, 8), 6))

        positions = range(len(entries))
        labels = []
        for i, e in enumerate(entries):
            bp = ax.boxplot([e["block_means"]], positions=[i], widths=0.6,
                           patch_artist=True,
                           boxprops=dict(facecolor=e["color"], alpha=0.6),
                           medianprops=dict(color="black", lw=2),
                           flierprops=dict(markersize=2))
            symbol = "●" if e["pair_type"] == "intra" else "○"
            labels.append(f"{e['arm']}\n{e['group']}\n{symbol} {e['pair_type']}")

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=11, ha="center")
        ax.set_ylabel("Block mean log(TMRCA)")
        ax.set_title(f"{pop.replace('_', ' ')} — Coalescence by karyotype group and pair type",
                     fontsize=13)

        plt.savefig(FIG_DIR / "karyotype_comparisons" / f"{pop}.png",
                    dpi=150, bbox_inches="tight")
        plt.savefig(FIG_DIR / "karyotype_comparisons" / f"{pop}.pdf",
                    bbox_inches="tight")
        plt.close()

    print("  Saved karyotype_comparisons/*.png")


# ===========================================================================
# Figure 5: Per-population multi-arm profile
# ===========================================================================

def fig5_per_population_profiles():
    """For each population: 5 columns (one per arm), each with TMRCA + accessibility."""

    for pop in POPULATIONS:
        fig, all_axes = plt.subplots(2, 5, figsize=(28, 6),
                                      gridspec_kw={"wspace": 0.3, "hspace": 0.08,
                                                   "height_ratios": [4, 1]})
        any_data = False

        for a_idx, arm in enumerate(ARMS):
            ax = all_axes[0, a_idx]
            ax_acc = all_axes[1, a_idx]
            ax.set_title(arm, fontsize=12)

            xlim = None
            for group in ARM_GROUPS[arm]:
                data = load_group(pop, arm, group, "intra")
                if data is None:
                    continue
                any_data = True
                bm = block_means(data)
                mids = block_mids_mb(data)
                if xlim is None:
                    xlim = (0, mids.max())
                n = data["config"]["n_pairs"]
                color = group_color(group)
                ax.plot(mids, bm, lw=1, color=color, alpha=0.8,
                        label=f"{pretty_group(group)} (n={n})")

            if arm == "2L":
                ax.axvspan(*INV_2LA, alpha=0.06, color="red")
            ax.tick_params(labelbottom=False)
            if a_idx == 0:
                ax.set_ylabel("log(TMRCA)")
            ax.legend(fontsize=6, loc="lower right")

            if xlim:
                _plot_accessibility_track(ax_acc, arm, xlim=xlim)
            else:
                ax_acc.set_visible(False)
            ax_acc.set_xlabel("Mb", fontsize=10)

        if not any_data:
            for ax in all_axes.flat:
                pending_ax(ax)

        fig.suptitle(f"{pop.replace('_', ' ')} — Intra-individual TMRCA across all arms",
                     fontsize=14, y=1.02)

        plt.savefig(FIG_DIR / "per_population" / f"{pop}_all_arms.png",
                    dpi=150, bbox_inches="tight")
        plt.savefig(FIG_DIR / "per_population" / f"{pop}_all_arms.pdf",
                    bbox_inches="tight")
        plt.close()

    print("  Saved per_population/*.png")


# ===========================================================================
# Figure 6: Outlier skyline — candidate regions with extreme coalescence
# ===========================================================================

# Trim from each end of a chromosome to avoid edge artifacts
# 2L has particularly bad quality in the first ~10Mb
EDGE_TRIM_MB = 2.0
ARM_EDGE_TRIM = {
    "2L": (10.0, 2.0),  # (left_trim_Mb, right_trim_Mb)
}

# stdpopsim AnoGam reference demography (GabonAg1000G_1A17)
ANOGAM_DEMOGRAPHY = Path("/tmp/anogam_demography.npz")

def fig6_outlier_skylines():
    """For each population × arm × group (intra): identify blocks with
    extreme coalescence times (top/bottom 1%) after trimming chromosome edges.
    Plot the full skyline with outlier blocks highlighted and annotated."""

    os.makedirs(FIG_DIR / "outlier_skylines", exist_ok=True)

    for pop in POPULATIONS:
        for arm in ARMS:
            for group in ARM_GROUPS[arm]:
                data = load_group(pop, arm, group, "intra")
                if data is None:
                    continue

                bm = block_means(data)
                mids = block_mids_mb(data)
                n_pairs = data["config"]["n_pairs"]

                # Trim edges (per-arm overrides for bad quality regions)
                left_trim, right_trim = ARM_EDGE_TRIM.get(arm, (EDGE_TRIM_MB, EDGE_TRIM_MB))
                edge_mask = (mids > left_trim) & (mids < mids.max() - right_trim)
                bm_trimmed = bm[edge_mask]
                mids_trimmed = mids[edge_mask]

                if len(bm_trimmed) < 20:
                    continue

                # Compute thresholds on trimmed data
                q01 = np.percentile(bm_trimmed, 1)
                q99 = np.percentile(bm_trimmed, 99)
                q05 = np.percentile(bm_trimmed, 5)
                q95 = np.percentile(bm_trimmed, 95)

                # Get per-block accessibility for trimmed blocks
                acc = _load_accessibility()
                acc_frac_trimmed = None
                MIN_ACC = 0.5  # only flag outliers in blocks with >50% accessibility
                if acc is not None and f"{arm}_frac" in acc:
                    acc_frac_full = acc[f"{arm}_frac"]
                    # Match to trimmed blocks by index
                    trimmed_indices = np.where(edge_mask)[0]
                    acc_frac_trimmed = np.array([
                        acc_frac_full[i] if i < len(acc_frac_full) else 0.0
                        for i in trimmed_indices
                    ])

                # Masks: extreme AND accessible (not missingness-driven)
                young_mask = (bm_trimmed <= q05)
                old_mask = (bm_trimmed >= q95)
                if acc_frac_trimmed is not None:
                    accessible = acc_frac_trimmed >= MIN_ACC
                    young_credible = young_mask & accessible
                    old_credible = old_mask & accessible
                    young_suspect = young_mask & ~accessible
                    old_suspect = old_mask & ~accessible
                else:
                    young_credible = young_mask
                    old_credible = old_mask
                    young_suspect = np.zeros_like(young_mask)
                    old_suspect = np.zeros_like(old_mask)

                fig, axes = plt.subplots(3, 1, figsize=(18, 8),
                                          gridspec_kw={"height_ratios": [2.5, 0.8, 0.5], "hspace": 0.12},
                                          sharex=True)

                # --- Top panel: full skyline with outliers ---
                ax = axes[0]
                ax.plot(mids, bm, lw=0.5, color="#ccc", zorder=1)
                ax.plot(mids_trimmed, bm_trimmed, lw=0.8, color="steelblue", zorder=2)

                # Shade edge regions
                ax.axvspan(0, left_trim, alpha=0.08, color="gray")
                ax.axvspan(mids.max() - right_trim, mids.max(), alpha=0.08, color="gray")
                ax.text(left_trim / 2, bm.min(),
                        "edge\ntrimmed", ha="center", va="bottom", fontsize=9, color="#999")
                ax.text(mids.max() - right_trim / 2, bm.min(),
                        "edge\ntrimmed", ha="center", va="bottom", fontsize=9, color="#999")

                # Suspect outliers (low accessibility) — gray, no annotation
                if young_suspect.any():
                    ax.scatter(mids_trimmed[young_suspect], bm_trimmed[young_suspect],
                              color="#bbb", s=25, zorder=4, marker="x", linewidths=1)
                if old_suspect.any():
                    ax.scatter(mids_trimmed[old_suspect], bm_trimmed[old_suspect],
                              color="#bbb", s=25, zorder=4, marker="x", linewidths=1)

                _outlier_texts = []  # collect for adjust_text at the end
                _outlier_xys = []

                def _collect_outlier_labels(ax, indices, mids_arr, vals_arr, text_color):
                    """Add text labels for outliers, to be repositioned by adjust_text."""
                    if len(indices) == 0:
                        return
                    for si in indices:
                        x_mb = mids_arr[si]
                        y_val = vals_arr[si]
                        pos_bp = int(x_mb * 1e6)
                        gene = _find_gene_at(arm, pos_bp)

                        if gene and gene in NOTABLE_GENES:
                            label = f"{gene} ({x_mb:.1f}Mb)"
                            fw, fs = "bold", 9
                        elif gene:
                            label = f"{gene} ({x_mb:.1f}Mb)"
                            fw, fs = "normal", 8
                        else:
                            label = f"{x_mb:.1f}Mb"
                            fw, fs = "normal", 8

                        t = ax.text(x_mb, y_val, label, fontsize=fs, color=text_color,
                                    fontweight=fw, ha="center", va="center", zorder=12)
                        _outlier_texts.append(t)
                        _outlier_xys.append((x_mb, y_val))

                # Credible young outliers (high accessibility)
                n_young_cred = young_credible.sum()
                if young_credible.any():
                    ax.scatter(mids_trimmed[young_credible], bm_trimmed[young_credible],
                              color="#2196F3", s=50, zorder=5,
                              label=f"Young credible ({n_young_cred}, acc>{MIN_ACC:.0%})",
                              edgecolors="navy", linewidths=0.8)
                    cred_idx = np.where(young_credible)[0]
                    top8 = cred_idx[np.argsort(bm_trimmed[cred_idx])[:8]]
                    _collect_outlier_labels(ax, top8, mids_trimmed, bm_trimmed, "#1565C0")

                # Credible old outliers (high accessibility)
                n_old_cred = old_credible.sum()
                if old_credible.any():
                    ax.scatter(mids_trimmed[old_credible], bm_trimmed[old_credible],
                              color="#E91E63", s=50, zorder=5,
                              label=f"Old credible ({n_old_cred}, acc>{MIN_ACC:.0%})",
                              edgecolors="darkred", linewidths=0.8)
                    cred_idx = np.where(old_credible)[0]
                    top8 = cred_idx[np.argsort(bm_trimmed[cred_idx])[-8:]]
                    _collect_outlier_labels(ax, top8, mids_trimmed, bm_trimmed, "#C62828")

                # Legend note for suspect markers
                if young_suspect.any() or old_suspect.any():
                    ax.scatter([], [], color="#bbb", s=25, marker="x",
                              label=f"Suspect (acc<{MIN_ACC:.0%}, not annotated)")

                # Percentile bands
                ax.axhline(q95, color="#E91E63", ls=":", lw=0.8, alpha=0.5)
                ax.axhline(q05, color="#2196F3", ls=":", lw=0.8, alpha=0.5)
                ax.axhline(np.median(bm_trimmed), color="black", ls="--", lw=1, alpha=0.4)

                if arm == "2L":
                    ax.axvspan(*INV_2LA, alpha=0.06, color="red")
                    ax.axvline(INV_2LA[0], color="red", ls="--", lw=0.8, alpha=0.4)
                    ax.axvline(INV_2LA[1], color="red", ls="--", lw=0.8, alpha=0.4)

                # Tighten y-axis to data range + small padding for labels
                data_min = min(bm.min(), bm_trimmed.min())
                data_max = max(bm.max(), bm_trimmed.max())
                pad = (data_max - data_min) * 0.15
                ax.set_ylim(data_min - pad * 0.3, data_max + pad)

                ax.set_ylabel("Block mean log(TMRCA)", fontsize=12)
                ax.legend(fontsize=10, loc="lower right")
                ax.set_title(
                    f"{pop.replace('_', ' ')} — {arm} — {pretty_group(group)} intra (n={n_pairs})\n"
                    f"Outlier blocks (trimmed: {left_trim}Mb left, {right_trim}Mb right)",
                    fontsize=13)

                # --- Bottom panel: z-score skyline ---
                ax2 = axes[1]
                mean_val = np.mean(bm_trimmed)
                std_val = np.std(bm_trimmed)
                if std_val > 0:
                    z = (bm_trimmed - mean_val) / std_val
                else:
                    z = np.zeros_like(bm_trimmed)

                colors = np.where(z > 2, "#E91E63",
                         np.where(z < -2, "#2196F3", "#90A4AE"))
                ax2.bar(mids_trimmed, z, width=(mids_trimmed[1] - mids_trimmed[0]) * 0.9,
                        color=colors, edgecolor="none", alpha=0.7)
                ax2.axhline(0, color="black", lw=0.5)
                ax2.axhline(2, color="#E91E63", ls=":", lw=0.8, alpha=0.5)
                ax2.axhline(-2, color="#2196F3", ls=":", lw=0.8, alpha=0.5)
                ax2.set_xlabel("Position (Mb)", fontsize=12)
                ax2.set_ylabel("Z-score", fontsize=12)
                ax2.set_title("Standardized deviation from mean (|Z|>2 highlighted)", fontsize=11)

                if arm == "2L":
                    ax2.axvspan(*INV_2LA, alpha=0.06, color="red")
                # Adjust text positions to avoid overlap
                if _outlier_texts:
                    from adjustText import adjust_text
                    adjust_text(
                        _outlier_texts, ax=ax,
                        x=[xy[0] for xy in _outlier_xys],
                        y=[xy[1] for xy in _outlier_xys],
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
                        force_text=(2.0, 3.0),
                        force_points=(1.0, 1.0),
                        expand_text=(2.0, 2.5),
                        only_move={"text": "xy"},
                        max_move=30,
                    )

                ax2.tick_params(labelbottom=False)

                # --- Accessibility track ---
                ax3 = axes[2]
                _plot_accessibility_track(ax3, arm, xlim=(mids.min(), mids.max()))
                ax3.set_xlabel("Position (Mb)", fontsize=12)

                fname = f"{pop}_{arm}_{group}"
                plt.savefig(FIG_DIR / "outlier_skylines" / f"{fname}.png",
                            dpi=150, bbox_inches="tight")
                plt.savefig(FIG_DIR / "outlier_skylines" / f"{fname}.pdf",
                            bbox_inches="tight")
                plt.close()

    print("  Saved outlier_skylines/*.png")


# ===========================================================================
# Figure 7: Geographic maps — coalescence projected onto Africa
# ===========================================================================

# Mean lat/long per population from Ag3 metadata
POP_COORDS = {
    "Burkina_Faso":                     (11.269, -4.037),
    "Cameroon":                         (5.627, 13.631),
    "Central_African_Republic":         (4.367, 18.583),
    "Democratic_Republic_of_the_Congo": (4.283, 21.017),
    "Equatorial_Guinea":                (3.700, 8.700),
    "Gabon":                            (0.384, 9.455),
    "Gambia":                           (13.567, -14.917),
    "Ghana":                            (5.940, -0.246),
    "Guinea":                           (8.870, -9.774),
    "Guinea-Bissau":                    (12.272, -14.222),
    "Kenya":                            (-3.511, 39.909),
    "Mali":                             (11.670, -8.042),
    "Mayotte":                          (-12.857, 45.137),
    "Mozambique":                       (-23.716, 35.299),
    "Tanzania":                         (-3.451, 35.285),
    "Uganda":                           (0.072, 32.041),
}

# Simplified Africa coastline (lon, lat pairs) for matplotlib polygon
# Rough outline — enough for a clean background
AFRICA_OUTLINE = [
    (-17.5, 14.7), (-17.3, 21.1), (-13.2, 23.5), (-12.0, 25.9),
    (-8.7, 27.7), (-5.9, 29.5), (-2.2, 35.1), (1.3, 36.3),
    (3.5, 36.9), (8.6, 36.9), (9.6, 37.2), (11.1, 37.1),
    (11.6, 33.2), (12.3, 32.8), (15.2, 32.3), (20.1, 31.9),
    (25.0, 31.5), (28.5, 31.2), (29.9, 31.2), (32.6, 31.3),
    (34.2, 31.4), (34.9, 29.5), (32.9, 28.5), (33.2, 28.0),
    (34.1, 26.0), (35.8, 22.0), (36.9, 22.0), (37.5, 18.0),
    (39.5, 16.0), (41.8, 11.7), (43.3, 11.3), (45.0, 10.4),
    (51.4, 11.8), (51.0, 8.0), (47.0, 4.0), (46.0, 1.5),
    (44.0, -1.0), (42.0, -2.5), (41.5, -4.5), (40.5, -10.5),
    (39.5, -15.0), (35.5, -22.0), (35.0, -25.0), (33.0, -26.5),
    (30.0, -28.5), (28.5, -32.0), (27.0, -33.5), (25.0, -33.9),
    (20.0, -34.8), (18.0, -34.6), (17.9, -32.5), (18.3, -28.5),
    (16.5, -25.0), (14.5, -22.5), (13.0, -20.0), (12.0, -17.5),
    (12.0, -13.5), (12.5, -6.0), (11.5, -2.0), (9.5, 1.0),
    (9.5, 4.0), (8.7, 4.7), (7.0, 4.3), (4.3, 6.3),
    (2.7, 6.2), (1.6, 6.1), (-3.0, 5.0), (-7.5, 4.3),
    (-8.3, 7.6), (-13.2, 8.5), (-15.0, 11.0), (-16.7, 12.4),
    (-17.5, 14.7),
]


def _draw_africa(ax, facecolor="#f0f0f0", edgecolor="#999", lw=0.8):
    """Draw simplified Africa outline on an axes."""
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    poly = Polygon([(lon, lat) for lon, lat in AFRICA_OUTLINE],
                   closed=True, facecolor=facecolor, edgecolor=edgecolor, lw=lw)
    ax.add_patch(poly)
    ax.set_xlim(-20, 52)
    ax.set_ylim(-37, 40)
    ax.set_aspect("equal")


def fig7_geographic_maps():
    """Geographic visualizations of coalescence data on Africa map."""

    os.makedirs(FIG_DIR / "geographic", exist_ok=True)

    # --- 7a: Bubble map — mean TMRCA per population × karyotype (2L intra) ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 10))
    groups_2L = ARM_GROUPS["2L"]

    for g_idx, group in enumerate(groups_2L):
        ax = axes[g_idx]
        _draw_africa(ax)

        for pop in POPULATIONS:
            if pop not in POP_COORDS:
                continue
            lat, lon = POP_COORDS[pop]
            data = load_group(pop, "2L", group, "intra")

            if data is not None:
                val = genome_mean(data)
                n = data["config"]["n_pairs"]
                size = max(n * 8, 60)
                sc = ax.scatter(lon, lat, s=size, c=val, cmap="magma",
                               vmin=11, vmax=14, edgecolors="white", linewidths=1.5,
                               zorder=10)
                ax.annotate(pop.replace("_", "\n"), (lon, lat),
                           textcoords="offset points", xytext=(8, -5),
                           fontsize=8, color="#333", zorder=11)
            else:
                ax.scatter(lon, lat, s=60, c="#ddd", edgecolors="#999",
                          linewidths=1, zorder=10, marker="x")

        ax.set_title(f"2L — {pretty_group(group)} — intra", fontsize=14)
        ax.set_xlabel("Longitude", fontsize=11)
        if g_idx == 0:
            ax.set_ylabel("Latitude", fontsize=11)

    # Add colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap="magma",
                        norm=plt.Normalize(11, 14)),
                        ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("Mean log(TMRCA)", fontsize=12)

    fig.suptitle("Geographic distribution of intra-individual coalescence — chromosome 2L",
                 fontsize=16, y=0.98)
    plt.savefig(FIG_DIR / "geographic" / "map_2L_bubbles.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "geographic" / "map_2L_bubbles.pdf", bbox_inches="tight")
    plt.close()

    # --- 7b: Map with embedded sparklines — 2L het intra profile per population ---
    fig, ax = plt.subplots(figsize=(16, 14))
    _draw_africa(ax)

    for pop in POPULATIONS:
        if pop not in POP_COORDS:
            continue
        lat, lon = POP_COORDS[pop]

        data = load_group(pop, "2L", "2La_heterozygous", "intra")
        if data is not None:
            bm = block_means(data)
            mids = block_mids_mb(data)
            n = data["config"]["n_pairs"]

            # Create inset axes for sparkline at population location
            # Convert data coords to figure coords
            inv = ax.transData
            fig_trans = fig.transFigure.inverted()

            # Position sparkline
            x_fig, y_fig = fig_trans.transform(inv.transform((lon, lat)))
            inset_w, inset_h = 0.08, 0.04
            inset = fig.add_axes([x_fig - inset_w/2, y_fig + 0.01,
                                  inset_w, inset_h])
            inset.plot(mids, bm, lw=0.6, color="#FF9800")
            inset.axvspan(*INV_2LA, alpha=0.15, color="red")
            inset.set_xlim(mids.min(), mids.max())
            inset.set_ylim(bm.min() - 0.5, bm.max() + 0.5)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_linewidth(0.5)
                spine.set_color("#666")
            inset.patch.set_alpha(0.85)
            inset.patch.set_facecolor("white")

            ax.annotate(pop.replace("_", " "), (lon, lat),
                       textcoords="offset points", xytext=(0, -12),
                       fontsize=8, ha="center", color="#333")
            ax.plot(lon, lat, "o", color="#FF9800", ms=5, zorder=10)
        else:
            ax.scatter(lon, lat, s=40, c="#ddd", edgecolors="#999",
                      linewidths=1, zorder=10, marker="x")
            ax.annotate(pop.replace("_", " ") + "\n(pending)", (lon, lat),
                       textcoords="offset points", xytext=(0, -15),
                       fontsize=7, ha="center", color="#999")

    ax.set_title("2L heterozygous — intra-individual TMRCA profiles across Africa\n"
                 "(orange sparklines show coalescence along 2L, red = In(2L)a region)",
                 fontsize=14)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    plt.savefig(FIG_DIR / "geographic" / "map_2L_sparklines.png", dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "geographic" / "map_2L_sparklines.pdf", bbox_inches="tight")
    plt.close()

    # --- 7c: Inversion effect map — ratio of TMRCA inside vs outside 2La ---
    fig, ax = plt.subplots(figsize=(14, 12))
    _draw_africa(ax)

    ratios = {}
    for pop in POPULATIONS:
        if pop not in POP_COORDS:
            continue
        data = load_group(pop, "2L", "2La_heterozygous", "intra")
        if data is None:
            continue

        bm = block_means(data)
        mids = block_mids_mb(data)
        inside = (mids >= INV_2LA[0]) & (mids <= INV_2LA[1])
        outside = (mids > EDGE_TRIM_MB) & (mids < mids.max() - EDGE_TRIM_MB) & ~inside

        if outside.any() and inside.any():
            ratio = np.mean(bm[inside]) - np.mean(bm[outside])
            ratios[pop] = ratio

    if ratios:
        vals = list(ratios.values())
        vmin, vmax = min(vals) - 0.1, max(vals) + 0.1

        for pop, ratio in ratios.items():
            lat, lon = POP_COORDS[pop]
            sc = ax.scatter(lon, lat, s=200, c=ratio, cmap="RdYlBu_r",
                           vmin=vmin, vmax=vmax, edgecolors="black",
                           linewidths=1.2, zorder=10)
            ax.annotate(f"{pop.replace('_', ' ')}\n({ratio:+.2f})",
                       (lon, lat), textcoords="offset points", xytext=(10, -5),
                       fontsize=9, color="#333", zorder=11)

        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label("TMRCA difference (inside - outside In(2L)a)", fontsize=12)

    # Mark populations without data
    for pop in POPULATIONS:
        if pop not in ratios and pop in POP_COORDS:
            lat, lon = POP_COORDS[pop]
            ax.scatter(lon, lat, s=60, c="#ddd", edgecolors="#999",
                      linewidths=1, zorder=10, marker="x")

    ax.set_title("2La inversion effect — how much deeper is coalescence inside vs outside\n"
                 "(2La-heterozygous, intra-individual pairs)", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    plt.savefig(FIG_DIR / "geographic" / "map_2La_inversion_effect.png",
                dpi=150, bbox_inches="tight")
    plt.savefig(FIG_DIR / "geographic" / "map_2La_inversion_effect.pdf",
                bbox_inches="tight")
    plt.close()

    print("  Saved geographic/*.png")


# ===========================================================================
# Figure 8: Demographic inference — IICR / Ne(t) from TMRCA distributions
# ===========================================================================

def coalescence_rates(tmrcas, time_windows):
    """Compute piecewise-constant coalescence rates from TMRCA samples.

    Bins TMRCA values into time windows and estimates the coalescence rate
    in each interval as the fraction of lineages that coalesce, normalized
    by interval width.

    Parameters
    ----------
    tmrcas : 1-D array of coalescence times (generations, NOT log)
    time_windows : 1-D array of bin edges (n_bins + 1)

    Returns
    -------
    rates : 1-D array of coalescence rates per window (n_bins)
    """
    counts, _ = np.histogram(tmrcas, bins=time_windows)
    # Fraction of pairs coalescing in each window
    total = len(tmrcas)
    if total == 0:
        return np.zeros(len(time_windows) - 1)
    # Cumulative survival: fraction of pairs not yet coalesced at start of window
    cum_coalesced = np.cumsum(counts)
    surviving = total - np.concatenate([[0], cum_coalesced[:-1]])
    # Width of each window
    widths = np.diff(time_windows)
    # Rate = (count / surviving) / width, with protection against division by zero
    rates = np.where(
        (surviving > 0) & (widths > 0),
        (counts / surviving) / widths,
        0.0,
    )
    return rates


def _load_anogam_ref():
    """Load stdpopsim AnoGam GabonAg1000G_1A17 reference IICR."""
    if not ANOGAM_DEMOGRAPHY.exists():
        try:
            import stdpopsim
            species = stdpopsim.get_species("AnoGam")
            demogr = species.get_demographic_model("GabonAg1000G_1A17")
            pop_name = demogr.model.populations[0].name
            fine_time_grid = np.logspace(0, 7, 1000)
            coalrate, _ = demogr.model.debug().coalescence_rate_trajectory(
                lineages={pop_name: 2}, steps=fine_time_grid,
            )
            iicr = 1.0 / (2.0 * coalrate)
            np.savez(ANOGAM_DEMOGRAPHY, time_grid=fine_time_grid,
                     coalrate=coalrate, iicr=iicr,
                     gen_time=species.generation_time)
        except Exception as e:
            print(f"  Warning: could not generate AnoGam reference: {e}")
            return None, None
    data = np.load(ANOGAM_DEMOGRAPHY)
    return data["time_grid"], data["iicr"]


def _plot_anogam_ref(ax):
    """Add stdpopsim Gabon reference curve to an IICR plot."""
    t, iicr = _load_anogam_ref()
    if t is not None:
        ax.plot(t, iicr, "--", color="black", lw=2, alpha=0.6,
                label="stdpopsim Gabon (GabonAg1000G_1A17)")


# SMC++ pre-computed references (from cxt paper, 2L, 200bp resolution)
SMCPP_DIR = RESULTS_DIR / "smcpp_reference"
SMCPP_POP_MAP = {
    "Burkina_Faso": "burkina_faso",
    "Cameroon": "cameroon",
    "Ghana": "ghana",
    "Mali": "mali",
    "Uganda": "uganda",
}

def _plot_smcpp_ref(ax, pop, time_windows):
    """Overlay SMC++ IICR for a population (if available)."""
    smcpp_key = SMCPP_POP_MAP.get(pop)
    if smcpp_key is None:
        return
    path = SMCPP_DIR / f"smcpp_{smcpp_key}.npz"
    if not path.exists():
        return
    data = np.load(path)
    yhats = data["yhats"]  # (n_reps, n_windows) in log-space

    # Average across replicates, convert to generations
    mean_log = np.mean(yhats, axis=0)
    tmrcas_gen = np.exp(mean_log[mean_log > 0])  # filter out invalid

    if len(tmrcas_gen) == 0:
        return

    rates = coalescence_rates(tmrcas_gen, time_windows)
    with np.errstate(divide="ignore", invalid="ignore"):
        iicr = np.where(rates > 0, 1.0 / (2.0 * rates), np.nan)

    time_mids = np.sqrt(time_windows[:-1] * time_windows[1:])
    time_mids[0] = time_windows[1] / 2

    ax.plot(time_mids, iicr, ":", color="#9C27B0", lw=2.5, alpha=0.7,
            label="SMC++ (cxt paper, 2L)")


DEMOG_MIN_ACC = 0.5  # minimum accessibility fraction to include a block

def _get_masked_tmrcas(data, arm):
    """Get TMRCAs (in generations) from accessible blocks only.

    Filters out blocks with low accessibility and chromosome edges,
    then returns flattened exp(means) for IICR computation.
    """
    means = data["means"]
    idx = data["index_map"]
    blocks = data["blocks"]
    n_blocks = len(blocks)
    n_pairs = int(idx[:, 1].max()) + 1
    n_win = means.shape[1]

    # Reshape to (n_blocks, n_pairs, n_windows)
    arr = np.full((n_blocks, n_pairs, n_win), np.nan)
    for i in range(len(idx)):
        arr[idx[i, 0], idx[i, 1], :] = means[i]

    block_mids = np.array([(b["start"] + b["end"]) / 2 / 1e6 for b in blocks])

    # Edge trimming
    left_trim, right_trim = ARM_EDGE_TRIM.get(arm, (EDGE_TRIM_MB, EDGE_TRIM_MB))
    edge_ok = (block_mids > left_trim) & (block_mids < block_mids.max() - right_trim)

    # Accessibility filtering
    acc = _load_accessibility()
    if acc is not None and f"{arm}_frac" in acc:
        acc_frac = acc[f"{arm}_frac"]
        acc_ok = np.array([
            acc_frac[i] >= DEMOG_MIN_ACC if i < len(acc_frac) else False
            for i in range(n_blocks)
        ])
    else:
        acc_ok = np.ones(n_blocks, dtype=bool)

    keep = edge_ok & acc_ok
    masked = arr[keep]  # (n_kept_blocks, n_pairs, n_windows)
    tmrcas_gen = np.exp(masked[~np.isnan(masked)])
    return tmrcas_gen


def fig8_demography():
    """Estimate and plot Ne(t) via IICR for each population.

    Uses intra-individual pairs from all available arms and karyotype groups.
    IICR = 1 / (2 * coalescence_rate) is a proxy for effective population size.
    Masks out blocks with low accessibility and chromosome edges.
    """

    os.makedirs(FIG_DIR / "demography", exist_ok=True)

    # Time windows (log-spaced in generations)
    n_time_bins = 40
    time_windows = np.logspace(2, 7, n_time_bins + 1)
    time_windows[0] = 0.0
    time_mids = np.sqrt(time_windows[:-1] * time_windows[1:])  # geometric mean
    time_mids[0] = time_windows[1] / 2

    # --- 8a: All populations overlaid (one plot per arm × karyotype, intra) ---
    for arm in ARMS:
        for group in ARM_GROUPS[arm]:
            fig, ax = plt.subplots(figsize=(10, 7))
            any_data = False

            for pop in POPULATIONS:
                data = load_group(pop, arm, group, "intra")
                if data is None:
                    continue
                any_data = True

                # Flatten all TMRCAs to generations
                tmrcas_gen = _get_masked_tmrcas(data, arm)
                rates = coalescence_rates(tmrcas_gen, time_windows)

                # IICR = 1 / (2 * rate), proxy for Ne
                with np.errstate(divide="ignore", invalid="ignore"):
                    iicr = np.where(rates > 0, 1.0 / (2.0 * rates), np.nan)

                label = pop.replace("_", " ")
                ax.plot(time_mids, iicr, lw=1.5, alpha=0.8, label=label)

            if not any_data:
                pending_ax(ax)
            else:
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_ylim(1e3, 5e6)
                ax.set_xlabel("Time (generations)", fontsize=13)
                ax.set_ylabel("IICR  (~Ne)", fontsize=13)
                ax.grid(True, which="both", alpha=0.2)

                ax.legend(fontsize=9, ncol=2, loc="upper left")

            ax.set_title(f"Demographic inference — {arm} {pretty_group(group)} intra\n"
                         f"Intra-individual pairs",
                         fontsize=14)

            fname = f"demography_{arm}_{group}_allpops"
            plt.savefig(FIG_DIR / "demography" / f"{fname}.png",
                        dpi=150, bbox_inches="tight")
            plt.savefig(FIG_DIR / "demography" / f"{fname}.pdf",
                        bbox_inches="tight")
            plt.close()

    # --- 8b: Per population — all karyotype groups overlaid for 2L ---
    for pop in POPULATIONS:
        fig, ax = plt.subplots(figsize=(10, 7))
        any_data = False

        for group in ARM_GROUPS["2L"]:
            data = load_group(pop, "2L", group, "intra")
            if data is None:
                continue
            any_data = True

            tmrcas_gen = _get_masked_tmrcas(data, arm)
            rates = coalescence_rates(tmrcas_gen, time_windows)
            with np.errstate(divide="ignore", invalid="ignore"):
                iicr = np.where(rates > 0, 1.0 / (2.0 * rates), np.nan)

            n = data["config"]["n_pairs"]
            color = group_color(group)
            ax.plot(time_mids, iicr, lw=2, color=color, alpha=0.8,
                    label=f"{pretty_group(group)} (n={n})")

        if not any_data:
            pending_ax(ax)
        else:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e3, 5e6)
            ax.set_xlabel("Time (generations)", fontsize=13)
            ax.set_ylabel("IICR  (~Ne)", fontsize=13)
            ax.grid(True, which="both", alpha=0.2)
            ax.legend(fontsize=11)

        ax.set_title(f"{pop.replace('_', ' ')} — 2L IICR by karyotype\n"
                     f"Intra-individual pairs", fontsize=14)

        plt.savefig(FIG_DIR / "demography" / f"demography_2L_{pop}.png",
                    dpi=150, bbox_inches="tight")
        plt.savefig(FIG_DIR / "demography" / f"demography_2L_{pop}.pdf",
                    bbox_inches="tight")
        plt.close()

    # --- 8c: Per population — all arms overlaid (het intra only) ---
    for pop in POPULATIONS:
        fig, ax = plt.subplots(figsize=(10, 7))
        any_data = False

        for arm in ARMS:
            # Pick the het group if available, else "all"
            groups = ARM_GROUPS[arm]
            het_group = next((g for g in groups if "heterozygous" in g), groups[0])
            data = load_group(pop, arm, het_group, "intra")
            if data is None:
                continue
            any_data = True

            tmrcas_gen = _get_masked_tmrcas(data, arm)
            rates = coalescence_rates(tmrcas_gen, time_windows)
            with np.errstate(divide="ignore", invalid="ignore"):
                iicr = np.where(rates > 0, 1.0 / (2.0 * rates), np.nan)

            n = data["config"]["n_pairs"]
            ax.plot(time_mids, iicr, lw=2, alpha=0.8,
                    label=f"{arm} — {pretty_group(het_group)} (n={n})")

        if not any_data:
            pending_ax(ax)
        else:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e3, 5e6)
            ax.set_xlabel("Time (generations)", fontsize=13)
            ax.set_ylabel("IICR  (~Ne)", fontsize=13)
            ax.grid(True, which="both", alpha=0.2)
            ax.legend(fontsize=11)

        ax.set_title(f"{pop.replace('_', ' ')} — IICR across chromosome arms\n"
                     f"Intra-individual pairs", fontsize=14)

        plt.savefig(FIG_DIR / "demography" / f"demography_allarms_{pop}.png",
                    dpi=150, bbox_inches="tight")
        plt.savefig(FIG_DIR / "demography" / f"demography_allarms_{pop}.pdf",
                    bbox_inches="tight")
        plt.close()

    print("  Saved demography/*.png")


# ===========================================================================
# Figure 9: Cross-population sweep comparison at resistance loci
# ===========================================================================

# Known insecticide-resistance loci (arm, center_bp, name, window_kb)
RESISTANCE_LOCI = [
    ("2L", 2_422_652, "vgsc/kdr", 500),
    ("2L", 28_497_422, "Rdl", 500),
    ("2L", 28_598_038, "CYP6P4", 500),
    ("3R", 28_598_000, "Gste2", 500),
    ("X", 15_241_718, "CYP9K1", 500),
]

POP_COLORS_SWEEP = {
    "Burkina_Faso": "#2563eb", "Cameroon": "#dc2626",
    "Central_African_Republic": "#059669", "Democratic_Republic_of_the_Congo": "#7c3aed",
    "Equatorial_Guinea": "#d97706", "Gabon": "#0891b2",
    "Gambia": "#4f46e5", "Ghana": "#be185d",
    "Guinea": "#15803d", "Guinea-Bissau": "#a21caf",
    "Kenya": "#b91c1c", "Mali": "#0369a1",
    "Mayotte": "#86198f", "Mozambique": "#c2410c",
    "Tanzania": "#4338ca", "Uganda": "#047857",
}


def fig9_sweep_comparison():
    """Cross-population TMRCA profiles at resistance loci (Fig 3c)."""
    outdir = FIG_DIR / "sweep_comparison"
    outdir.mkdir(exist_ok=True)

    for arm, center_bp, locus_name, window_kb in RESISTANCE_LOCI:
        half = int(window_kb * 1e3 / 2)
        start_bp = max(0, center_bp - half)
        end_bp = center_bp + half

        fig, ax = plt.subplots(figsize=(12, 5))
        any_data = False

        for pop in POPULATIONS:
            # Use hom_standard for 2L/2R, all for others
            group = ARM_GROUPS[arm][0]
            data = load_group(pop, arm, group, "intra")
            if data is None:
                continue

            bm = block_means(data)
            mids = block_mids_mb(data)
            start_mb = start_bp / 1e6
            end_mb = end_bp / 1e6
            mask = (mids >= start_mb) & (mids <= end_mb)
            if mask.sum() < 2:
                continue

            color = POP_COLORS_SWEEP.get(pop, "#888")
            label = pop.replace("_", " ")
            ax.plot(mids[mask], bm[mask], color=color, lw=1.2, alpha=0.8, label=label)
            any_data = True

        if any_data:
            ax.axvline(center_bp / 1e6, color="red", ls="--", lw=1, alpha=0.5)
            ax.text(center_bp / 1e6, ax.get_ylim()[1], f"  {locus_name}",
                    color="red", fontsize=10, va="top", fontstyle="italic")
            ax.set_xlabel("Position (Mb)", fontsize=12)
            ax.set_ylabel("Block mean log(TMRCA)", fontsize=12)
            ax.set_title(f"Cross-population TMRCA at {locus_name} ({arm})",
                         fontsize=13, fontweight="bold")
            ax.legend(fontsize=8, ncol=2, loc="best")
            ax.grid(True, alpha=0.2)
        else:
            pending_ax(ax)

        plt.savefig(outdir / f"sweep_{locus_name.replace('/', '_')}_{arm}.png",
                    dpi=150, bbox_inches="tight")
        plt.savefig(outdir / f"sweep_{locus_name.replace('/', '_')}_{arm}.pdf",
                    bbox_inches="tight")
        plt.close()

    print("  Saved sweep_comparison/*.png")


# ===========================================================================
# Figure 10: Population x locus outlier summary heatmap (Fig 3d)
# ===========================================================================

def fig10_pop_locus_heatmap():
    """Heatmap: populations x resistance loci, colored by Z-score departure."""
    outdir = FIG_DIR / "sweep_comparison"
    outdir.mkdir(exist_ok=True)

    locus_names = [name for _, _, name, _ in RESISTANCE_LOCI]
    n_pops = len(POPULATIONS)
    n_loci = len(RESISTANCE_LOCI)
    z_matrix = np.full((n_pops, n_loci), np.nan)

    for j, (arm, center_bp, locus_name, _) in enumerate(RESISTANCE_LOCI):
        for i, pop in enumerate(POPULATIONS):
            group = ARM_GROUPS[arm][0]
            data = load_group(pop, arm, group, "intra")
            if data is None:
                continue

            bm = block_means(data)
            blocks = data["blocks"]
            mids_bp = np.array([(b["start"] + b["end"]) / 2 for b in blocks])

            # Find nearest block to locus center
            dists = np.abs(mids_bp - center_bp)
            nearest = np.argmin(dists)
            if dists[nearest] > 200_000:
                continue

            # Z-score: negative = recent coalescence (sweep), positive = deep
            mean_val = np.nanmean(bm)
            std_val = np.nanstd(bm)
            if std_val > 0:
                z_matrix[i, j] = (bm[nearest] - mean_val) / std_val

    # Only plot populations with at least one non-NaN
    has_data = ~np.all(np.isnan(z_matrix), axis=1)
    if not has_data.any():
        return

    z_sub = z_matrix[has_data]
    pop_labels = [POPULATIONS[i].replace("_", " ") for i in range(n_pops) if has_data[i]]

    fig, ax = plt.subplots(figsize=(max(6, n_loci * 1.5), max(4, len(pop_labels) * 0.5)))

    from matplotlib.colors import TwoSlopeNorm
    vmax = max(3, np.nanmax(np.abs(z_sub)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(z_sub, aspect="auto", cmap="RdBu_r", norm=norm, interpolation="nearest")
    ax.set_xticks(range(n_loci))
    ax.set_xticklabels(locus_names, fontsize=11, fontweight="bold", rotation=30, ha="right")
    ax.set_yticks(range(len(pop_labels)))
    ax.set_yticklabels(pop_labels, fontsize=10)

    # Annotate cells with Z-scores
    for yi in range(z_sub.shape[0]):
        for xi in range(z_sub.shape[1]):
            val = z_sub[yi, xi]
            if np.isfinite(val):
                color = "white" if abs(val) > 1.5 else "black"
                ax.text(xi, yi, f"{val:.1f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight="bold" if abs(val) > 2 else "normal")

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("Z-score (negative = sweep signal)", fontsize=10)

    ax.set_title("Resistance loci — TMRCA departure by population",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    plt.savefig(outdir / "pop_locus_heatmap.png", dpi=150, bbox_inches="tight")
    plt.savefig(outdir / "pop_locus_heatmap.pdf", bbox_inches="tight")
    plt.close()

    print("  Saved sweep_comparison/pop_locus_heatmap.png")


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Generating figures...")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Figures dir: {FIG_DIR}")

    fig1_overview_heatmap()
    fig2_distribution_grid()
    fig2b_density_grid()
    fig2c_overlay_densities()
    fig3_inversion_signals()
    fig4_karyotype_comparison()
    fig5_per_population_profiles()
    fig6_outlier_skylines()
    fig7_geographic_maps()
    fig8_demography()
    fig9_sweep_comparison()
    fig10_pop_locus_heatmap()

    print("\nDone! Re-run after more data arrives to fill in PENDING plots.")
