#!/usr/bin/env python3
"""TimeAtlas showcase visualizations with simulated Ag1000G-style data.

Generates publication-quality figures combining geographic maps of African
Anopheles gambiae collection sites with genome-wide TMRCA inference results.
All data is simulated placeholder data mimicking real Ag1000G structure.

Usage:
    python scripts/plot_atlas_showcase.py [--outdir figures/]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform
from scipy.ndimage import gaussian_filter1d
import seaborn as sns

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from fastcxt.atlas import TimeAtlas
from fastcxt.mosquito import ANOGAM_CHROMOSOME_ARMS


# ──────────────────────────────────────────────────────────────────────
# Typography & global style
# ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.5,
})

COL = {
    "bg":     "#ffffff",
    "panel":  "#fafafa",
    "grid":   "#e5e7eb",
    "border": "#d1d5db",
    "text":   "#1f2937",
    "muted":  "#6b7280",
    "accent": "#2563eb",
    "warm":   "#dc2626",
    "sweep":  "#dc2626",
}

TMRCA_CMAP = LinearSegmentedColormap.from_list("tmrca", [
    "#f0f4ff", "#dbeafe", "#93c5fd", "#3b82f6",
    "#1d4ed8", "#4338ca", "#6d28d9", "#9333ea",
    "#c026d3", "#e11d48",
])

UNCERTAINTY_CMAP = LinearSegmentedColormap.from_list("unc", [
    "#f9fafb", "#e0e7ff", "#a5b4fc", "#818cf8",
    "#6366f1", "#4f46e5", "#4338ca",
])

ARM_COLORS = {
    "2L": "#2563eb",
    "2R": "#7c3aed",
    "3L": "#059669",
    "3R": "#d97706",
    "X":  "#dc2626",
}

POP_COLORS = {
    "BFM": "#2563eb",
    "BFS": "#3b82f6",
    "GNS": "#06b6d4",
    "CMS": "#8b5cf6",
    "GAS": "#a855f7",
    "UGS": "#f59e0b",
    "KES": "#ef4444",
    "TZS": "#ec4899",
    "GWA": "#10b981",
    "AOM": "#f97316",
}

POP_ORDER = ["GWA", "GNS", "BFM", "BFS", "CMS", "GAS", "AOM", "UGS", "KES", "TZS"]


def _clean_ax(ax, title=None, xlabel=None, ylabel=None, grid_y=False, grid_x=False):
    ax.set_facecolor(COL["bg"])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(COL["border"])
    ax.tick_params(colors=COL["text"])
    if grid_y:
        ax.yaxis.grid(True, color=COL["grid"], linewidth=0.4, alpha=0.7)
        ax.set_axisbelow(True)
    if grid_x:
        ax.xaxis.grid(True, color=COL["grid"], linewidth=0.4, alpha=0.7)
        ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=COL["text"], pad=10, loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, color=COL["text"])
    if ylabel:
        ax.set_ylabel(ylabel, color=COL["text"])


def _panel_label(ax, label, x=-0.08, y=1.08):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=14,
            fontweight="bold", color=COL["text"], va="top", ha="left")


# ──────────────────────────────────────────────────────────────────────
# Population metadata (real Ag1000G coordinates)
# ──────────────────────────────────────────────────────────────────────

POPULATIONS = {
    "BFM": {"name": "Burkina Faso (Mopti)",   "lat": 14.49, "lon": -4.20, "n": 81,  "region": "West"},
    "BFS": {"name": "Burkina Faso (Savanna)",  "lat": 11.17, "lon": -1.52, "n": 82,  "region": "West"},
    "GNS": {"name": "Guinea-Bissau",           "lat": 12.10, "lon":-14.95, "n": 12,  "region": "West"},
    "CMS": {"name": "Cameroon",                "lat":  3.85, "lon": 11.50, "n": 79,  "region": "Central"},
    "GAS": {"name": "Gabon",                   "lat": -0.39, "lon":  9.45, "n": 69,  "region": "Central"},
    "UGS": {"name": "Uganda",                  "lat":  0.35, "lon": 32.58, "n": 112, "region": "East"},
    "KES": {"name": "Kenya",                   "lat": -0.09, "lon": 34.77, "n": 48,  "region": "East"},
    "TZS": {"name": "Tanzania",                "lat": -6.17, "lon": 35.74, "n": 29,  "region": "East"},
    "GWA": {"name": "Gambia",                  "lat": 13.45, "lon":-16.58, "n": 73,  "region": "West"},
    "AOM": {"name": "Angola",                  "lat": -8.84, "lon": 13.23, "n": 78,  "region": "Central"},
}

REGION_COLORS = {"West": "#2563eb", "Central": "#8b5cf6", "East": "#ef4444"}
ARM_ORDER = ["2L", "2R", "3L", "3R", "X"]


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _haversine(a, b):
    R = 6371.0
    lat1, lon1 = np.radians(a["lat"]), np.radians(a["lon"])
    lat2, lon2 = np.radians(b["lat"]), np.radians(b["lon"])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(h), np.sqrt(1 - h))


def _pop_mean_tmrca(atlas, pop_sample_map, p1, p2, arm="2L"):
    vals = []
    for sa in pop_sample_map[p1]:
        for sb in pop_sample_map[p2]:
            if sa == sb:
                continue
            result = atlas.query_pair(arm, sa, sb)
            if result is not None:
                vals.append(np.exp(result[0]).mean())
    return np.mean(vals) if vals else np.nan


def _pop_tmrca_matrix(atlas, pop_sample_map, arm="2L"):
    codes = POP_ORDER
    n = len(codes)
    mat = np.full((n, n), np.nan)
    for i, p1 in enumerate(codes):
        for j, p2 in enumerate(codes):
            if i <= j:
                mat[i, j] = mat[j, i] = _pop_mean_tmrca(atlas, pop_sample_map, p1, p2, arm)
    return mat


# ──────────────────────────────────────────────────────────────────────
# Simulated atlas
# ──────────────────────────────────────────────────────────────────────

def _sweep_signal(n_w, center=0.42, sigma=0.02):
    x = np.linspace(0, 1, n_w)
    return -1.8 * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def build_simulated_atlas(rng, n_per_pop=4, window_size=50_000):
    pop_codes = POP_ORDER
    pop_sample_map, sample_pop = {}, {}
    idx = 0
    for code in pop_codes:
        samples = list(range(idx, idx + n_per_pop))
        pop_sample_map[code] = samples
        for s in samples:
            sample_pop[s] = code
        idx += n_per_pop

    n_total = idx
    pairs = [(i, j) for i in range(n_total) for j in range(i + 1, n_total)]
    pairs_arr = np.array(pairs, dtype=np.int32)

    pop_base = {}
    for i, p1 in enumerate(pop_codes):
        for j, p2 in enumerate(pop_codes):
            if i <= j:
                if i == j:
                    base = rng.uniform(8.5, 9.5)
                else:
                    dist = _haversine(POPULATIONS[p1], POPULATIONS[p2])
                    base = 9.0 + 0.6 * np.log1p(dist / 500) + rng.normal(0, 0.12)
                pop_base[(p1, p2)] = pop_base[(p2, p1)] = base

    atlas = TimeAtlas()
    atlas.metadata = {
        "species": "Anopheles gambiae",
        "dataset": "Ag1000G Phase 3 (simulated placeholder)",
        "n_populations": len(pop_codes),
        "n_total_samples": n_total,
    }

    for arm in ARM_ORDER:
        arm_len = ANOGAM_CHROMOSOME_ARMS[arm]
        n_w = arm_len // window_size
        means = np.zeros((len(pairs_arr), n_w), dtype=np.float32)
        variances = np.zeros_like(means)
        x = np.linspace(0, 1, n_w)

        for pi, (sa, sb) in enumerate(pairs_arr):
            pa, pb = sample_pop[sa], sample_pop[sb]
            base = pop_base[(pa, pb)]
            noise = gaussian_filter1d(rng.normal(0, 0.06, n_w), sigma=15)
            landscape = base + noise

            if arm == "2L":
                sweep = _sweep_signal(n_w, center=0.42, sigma=0.02)
                geo_d = _haversine(POPULATIONS[pa], POPULATIONS[pb])
                strength = 0.3 + 0.7 * min(geo_d / 4000, 1.0)
                landscape += sweep * strength

            centro_bump = 0.25 * np.exp(-0.5 * ((x - 0.5) / 0.04) ** 2)
            landscape += centro_bump
            means[pi] = landscape

            var_base = 0.04 + 0.08 * rng.uniform(0.5, 1.5, n_w)
            if arm == "2L":
                var_base += 0.35 * np.exp(-0.5 * ((x - 0.42) / 0.035) ** 2)
            variances[pi] = var_base

        atlas.add_arm(arm, means, variances, pairs_arr,
                      window_size=window_size, mutation_rate=3.5e-9)

    return atlas, pop_sample_map, sample_pop


# ──────────────────────────────────────────────────────────────────────
# Figure 1 — Geographic collection sites
# ──────────────────────────────────────────────────────────────────────

def plot_collection_sites(atlas, psm, sp, outdir):
    fig = plt.figure(figsize=(14, 9), facecolor=COL["bg"])
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.80], projection=ccrs.Mercator())
    ax.set_extent([-22, 42, -16, 22], crs=ccrs.PlateCarree())
    ax.set_facecolor("#f0f4ff")

    ax.add_feature(cfeature.LAND, facecolor="#f5f5f4", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="#eff6ff", zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#d1d5db", zorder=2)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#9ca3af", zorder=2)
    ax.add_feature(cfeature.RIVERS, linewidth=0.3, edgecolor="#bfdbfe", alpha=0.6, zorder=2)
    ax.add_feature(cfeature.LAKES, facecolor="#dbeafe", edgecolor="#93c5fd",
                    linewidth=0.3, zorder=2)

    gl = ax.gridlines(draw_labels=True, linewidth=0.2, color="#e5e7eb",
                      alpha=0.5, linestyle="--", zorder=1)
    gl.top_labels = gl.right_labels = False
    gl.xlocator = mticker.FixedLocator([-20, -10, 0, 10, 20, 30, 40])
    gl.ylocator = mticker.FixedLocator([-15, -10, -5, 0, 5, 10, 15, 20])
    gl.xlabel_style = {"size": 7, "color": COL["muted"]}
    gl.ylabel_style = {"size": 7, "color": COL["muted"]}

    within = {}
    for code, samples in psm.items():
        vals = []
        for i, sa in enumerate(samples):
            for sb in samples[i + 1:]:
                r = atlas.query_pair("2L", sa, sb)
                if r:
                    vals.append(np.exp(r[0]).mean())
        within[code] = np.mean(vals) if vals else 0

    vmin, vmax = min(within.values()), max(within.values())
    norm = Normalize(vmin=vmin, vmax=vmax)
    transform = ccrs.PlateCarree()

    for code in POP_ORDER:
        info = POPULATIONS[code]
        c = TMRCA_CMAP(norm(within[code]))
        sz = 40 + info["n"] * 2.2

        ax.scatter(info["lon"], info["lat"], s=sz + 120, c="white",
                   alpha=0.7, transform=transform, zorder=4, edgecolors="none")
        ax.scatter(info["lon"], info["lat"], s=sz, c=[c],
                   edgecolors="white", linewidth=1.2, transform=transform, zorder=5)

        offset_x, offset_y = 8, 8
        if code == "KES":
            offset_x, offset_y = 8, -12
        if code == "GNS":
            offset_x = -8

        ax.annotate(
            f" {code}", (info["lon"], info["lat"]),
            xytext=(offset_x, offset_y), textcoords="offset points",
            fontsize=8.5, fontweight="bold", color=COL["text"],
            transform=transform, zorder=6,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )
        ax.annotate(
            f" n={info['n']}", (info["lon"], info["lat"]),
            xytext=(offset_x, offset_y - 11), textcoords="offset points",
            fontsize=6.5, color=COL["muted"], transform=transform, zorder=6,
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    sm = plt.cm.ScalarMappable(cmap=TMRCA_CMAP, norm=norm)
    cbar_ax = fig.add_axes([0.20, 0.07, 0.60, 0.018])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Within-population mean TMRCA (generations)", fontsize=8.5,
                   color=COL["text"], labelpad=6)
    cbar.ax.tick_params(labelsize=7, colors=COL["text"])
    cbar.outline.set_edgecolor(COL["border"])
    cbar.outline.set_linewidth(0.5)

    fig.text(0.50, 0.94, "Anopheles gambiae — Ag1000G Collection Sites",
             ha="center", fontsize=15, fontweight="bold", color=COL["text"])
    fig.text(0.50, 0.91,
             "Circle size proportional to sample count · Color encodes within-population TMRCA on chr2L",
             ha="center", fontsize=8.5, color=COL["muted"])

    fig.savefig(outdir / "01_collection_sites.png", facecolor=COL["bg"],
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("  ✓ 01_collection_sites.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 2 — Connectivity arcs
# ──────────────────────────────────────────────────────────────────────

def plot_connectivity_map(atlas, psm, sp, outdir):
    fig = plt.figure(figsize=(14, 9), facecolor=COL["bg"])
    ax = fig.add_axes([0.04, 0.10, 0.92, 0.80], projection=ccrs.Mercator())
    ax.set_extent([-22, 42, -16, 22], crs=ccrs.PlateCarree())
    ax.set_facecolor("#f8fafc")

    ax.add_feature(cfeature.LAND, facecolor="#f5f5f4", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor="#f0f7ff", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#9ca3af", zorder=2)
    ax.add_feature(cfeature.LAKES, facecolor="#e0f2fe", edgecolor="#7dd3fc",
                    linewidth=0.2, zorder=2)

    pop_codes = POP_ORDER
    between = {}
    for i, p1 in enumerate(pop_codes):
        for j, p2 in enumerate(pop_codes):
            if i < j:
                between[(p1, p2)] = _pop_mean_tmrca(atlas, psm, p1, p2)

    all_v = list(between.values())
    norm = Normalize(vmin=min(all_v), vmax=max(all_v))
    transform = ccrs.PlateCarree()

    sorted_pairs = sorted(between.items(), key=lambda kv: -kv[1])
    for (p1, p2), val in sorted_pairs:
        i1, i2 = POPULATIONS[p1], POPULATIONS[p2]
        c = TMRCA_CMAP(norm(val))
        alpha = 0.25 + 0.55 * (1 - norm(val))
        width = 0.4 + 3.0 * (1 - norm(val))

        ax.plot([i1["lon"], i2["lon"]], [i1["lat"], i2["lat"]],
                color=c, alpha=alpha, linewidth=width,
                solid_capstyle="round", transform=ccrs.Geodetic(), zorder=3)

    for code in POP_ORDER:
        info = POPULATIONS[code]
        region_c = REGION_COLORS[info["region"]]
        sz = 50 + info["n"] * 1.5

        ax.scatter(info["lon"], info["lat"], s=sz + 100,
                   c="white", alpha=0.85, transform=transform, zorder=4)
        ax.scatter(info["lon"], info["lat"], s=sz,
                   c=[region_c], edgecolors="white", linewidth=1.0,
                   transform=transform, zorder=5)

        ax.annotate(
            f" {code}  n={info['n']}", (info["lon"], info["lat"]),
            xytext=(9, 7), textcoords="offset points",
            fontsize=7.5, fontweight="bold", color=COL["text"],
            transform=transform, zorder=6,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        )

    sm = plt.cm.ScalarMappable(cmap=TMRCA_CMAP, norm=norm)
    cbar_ax = fig.add_axes([0.20, 0.07, 0.60, 0.018])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Between-population mean TMRCA (generations)", fontsize=8.5,
                   color=COL["text"], labelpad=6)
    cbar.ax.tick_params(labelsize=7, colors=COL["text"])
    cbar.outline.set_edgecolor(COL["border"])
    cbar.outline.set_linewidth(0.5)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=REGION_COLORS[r],
               markersize=8, label=f"{r} Africa", markeredgecolor="white", markeredgewidth=0.5)
        for r in ["West", "Central", "East"]
    ]
    leg = ax.legend(handles=legend_elements, loc="lower left", frameon=True,
                    facecolor="white", edgecolor=COL["border"], framealpha=0.9,
                    fontsize=7.5, borderpad=0.8)
    leg.get_frame().set_linewidth(0.5)

    fig.text(0.50, 0.94, "Population Connectivity — Pairwise TMRCA Across Africa",
             ha="center", fontsize=15, fontweight="bold", color=COL["text"])
    fig.text(0.50, 0.91,
             "Thicker lines = more recent coalescence · Node color = geographic region",
             ha="center", fontsize=8.5, color=COL["muted"])

    fig.savefig(outdir / "02_connectivity_map.png", facecolor=COL["bg"],
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("  ✓ 02_connectivity_map.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 3 — Genome-wide landscape (karyotype layout)
# ──────────────────────────────────────────────────────────────────────

def plot_genome_landscape(atlas, psm, sp, outdir):
    fig = plt.figure(figsize=(18, 9), facecolor=COL["bg"])
    gs = gridspec.GridSpec(2, 5,
                           width_ratios=[ANOGAM_CHROMOSOME_ARMS[a] for a in ARM_ORDER],
                           height_ratios=[1, 0.04], hspace=0.08, wspace=0.06)
    gs.update(left=0.06, right=0.97, top=0.88, bottom=0.08)

    y_global_min, y_global_max = 1e9, -1e9
    axes = []

    for ai, arm in enumerate(ARM_ORDER):
        ax = fig.add_subplot(gs[0, ai])
        axes.append(ax)
        _clean_ax(ax, grid_y=True)
        ad = atlas.arms[arm]
        x_mb = ad.window_starts / 1e6

        for code in POP_ORDER:
            traces = []
            for i, sa in enumerate(psm[code]):
                for sb in psm[code][i + 1:]:
                    r = atlas.query_pair(arm, sa, sb)
                    if r:
                        traces.append(np.exp(r[0]))
            if not traces:
                continue
            traces = np.array(traces)
            med = np.median(traces, axis=0)
            q10, q90 = np.percentile(traces, [10, 90], axis=0)

            ax.fill_between(x_mb, q10, q90, color=POP_COLORS[code], alpha=0.08)
            ax.plot(x_mb, med, color=POP_COLORS[code], linewidth=0.9, alpha=0.85)

            y_global_min = min(y_global_min, q10.min())
            y_global_max = max(y_global_max, q90.max())

        if arm == "2L":
            ax.axvline(20.7, color=COL["sweep"], linewidth=0.7, linestyle="--", alpha=0.5)
            ax.text(20.7, y_global_max * 0.98 if y_global_max < 1e9 else 10000,
                    "  Rdl", fontsize=6.5, color=COL["sweep"], va="top", fontstyle="italic")

        # chromosome ideogram bar
        ideo_ax = fig.add_subplot(gs[1, ai])
        ideo_ax.set_xlim(0, ad.arm_length / 1e6)
        ideo_ax.barh(0, ad.arm_length / 1e6, height=1, color=ARM_COLORS[arm],
                     alpha=0.25, edgecolor=ARM_COLORS[arm], linewidth=0.5)
        ideo_ax.set_ylim(-0.5, 0.5)
        ideo_ax.set_xlabel(f"chr{arm}", fontsize=9, fontweight="bold",
                           color=ARM_COLORS[arm], labelpad=3)
        ideo_ax.set_yticks([])
        ideo_ax.set_xticks([])
        for sp_name in ideo_ax.spines:
            ideo_ax.spines[sp_name].set_visible(False)

        ax.set_xlim(0, ad.arm_length / 1e6)
        if ai == 0:
            ax.set_ylabel("TMRCA (generations)", color=COL["text"])
        else:
            ax.set_yticklabels([])

    for ax in axes:
        ax.set_ylim(y_global_min * 0.93, y_global_max * 1.07)

    # legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=POP_COLORS[c], linewidth=1.5, label=c)
               for c in POP_ORDER]
    leg = axes[0].legend(handles=handles, loc="upper left", ncol=2,
                         frameon=True, facecolor="white", edgecolor=COL["border"],
                         framealpha=0.95, fontsize=6.5, borderpad=0.6,
                         columnspacing=0.8, handlelength=1.5)
    leg.get_frame().set_linewidth(0.4)

    fig.text(0.50, 0.96, "Genome-Wide TMRCA Landscape — Anopheles gambiae",
             ha="center", fontsize=15, fontweight="bold", color=COL["text"])
    fig.text(0.50, 0.93,
             "Within-population median TMRCA (shaded 10th–90th percentile) · "
             "Sweep visible on chr2L ~21 Mb",
             ha="center", fontsize=8.5, color=COL["muted"])

    fig.savefig(outdir / "03_genome_landscape.png", facecolor=COL["bg"],
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("  ✓ 03_genome_landscape.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 4 — Population heatmap with dendrogram + geographic inset
# ──────────────────────────────────────────────────────────────────────

def plot_population_heatmap(atlas, psm, sp, outdir):
    mat = _pop_tmrca_matrix(atlas, psm)
    n = len(POP_ORDER)

    dist_mat = mat.copy()
    np.fill_diagonal(dist_mat, 0)
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method="average")
    order = leaves_list(Z)

    mat_ord = mat[np.ix_(order, order)]
    labels_ord = [POP_ORDER[i] for i in order]

    fig = plt.figure(figsize=(14, 10), facecolor=COL["bg"])
    gs = gridspec.GridSpec(2, 3, width_ratios=[0.15, 1, 0.55],
                           height_ratios=[0.15, 1], hspace=0.02, wspace=0.04)
    gs.update(left=0.06, right=0.95, top=0.88, bottom=0.06)

    # top dendrogram
    ax_dend_top = fig.add_subplot(gs[0, 1])
    dn = dendrogram(Z, ax=ax_dend_top, color_threshold=0,
                    above_threshold_color=COL["accent"], no_labels=True)
    ax_dend_top.axis("off")
    ax_dend_top.set_facecolor(COL["bg"])

    # left dendrogram
    ax_dend_left = fig.add_subplot(gs[1, 0])
    dendrogram(Z, ax=ax_dend_left, orientation="left", color_threshold=0,
               above_threshold_color=COL["accent"], no_labels=True)
    ax_dend_left.axis("off")
    ax_dend_left.set_facecolor(COL["bg"])

    # heatmap
    ax_heat = fig.add_subplot(gs[1, 1])
    ax_heat.set_facecolor(COL["bg"])
    im = ax_heat.imshow(mat_ord, cmap=TMRCA_CMAP, aspect="equal", interpolation="nearest")

    ax_heat.set_xticks(range(n))
    ax_heat.set_yticks(range(n))
    ax_heat.set_xticklabels(labels_ord, rotation=40, ha="right", fontsize=8)
    ax_heat.set_yticklabels(labels_ord, fontsize=8)
    ax_heat.tick_params(length=0, colors=COL["text"])

    for i in range(n):
        for j in range(n):
            v = mat_ord[i, j]
            tc = "white" if v > np.median(mat_ord) else COL["text"]
            ax_heat.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=6, color=tc, fontweight="medium")

    for sp_name in ax_heat.spines.values():
        sp_name.set_edgecolor(COL["border"])
        sp_name.set_linewidth(0.4)

    cbar = fig.colorbar(im, ax=ax_heat, shrink=0.5, pad=0.02, aspect=25)
    cbar.set_label("Mean TMRCA (generations)", fontsize=8, color=COL["text"])
    cbar.ax.tick_params(labelsize=6.5, colors=COL["text"])
    cbar.outline.set_edgecolor(COL["border"])
    cbar.outline.set_linewidth(0.4)

    # geographic inset
    ax_geo = fig.add_subplot(gs[0:2, 2], projection=ccrs.Mercator())
    ax_geo.set_extent([-22, 42, -16, 22], crs=ccrs.PlateCarree())
    ax_geo.set_facecolor("#f8fafc")
    ax_geo.add_feature(cfeature.LAND, facecolor="#f5f5f4", edgecolor="none")
    ax_geo.add_feature(cfeature.OCEAN, facecolor="#f0f7ff")
    ax_geo.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="#d1d5db")

    transform = ccrs.PlateCarree()
    for code in POP_ORDER:
        info = POPULATIONS[code]
        ax_geo.scatter(info["lon"], info["lat"], s=50 + info["n"],
                       c=[REGION_COLORS[info["region"]]], edgecolors="white",
                       linewidth=0.6, transform=transform, zorder=5)
        ax_geo.annotate(code, (info["lon"], info["lat"]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=6, fontweight="bold", color=COL["text"],
                        transform=transform, zorder=6,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    fig.text(0.50, 0.95, "Population TMRCA Matrix — chr2L (Average Linkage Clustering)",
             ha="center", fontsize=14, fontweight="bold", color=COL["text"])
    fig.text(0.50, 0.92,
             "Hierarchical clustering on between-population mean TMRCA · Geographic inset shows collection sites",
             ha="center", fontsize=8.5, color=COL["muted"])

    fig.savefig(outdir / "04_population_heatmap.png", facecolor=COL["bg"],
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("  ✓ 04_population_heatmap.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 5 — Selective sweep deep-dive (5-panel)
# ──────────────────────────────────────────────────────────────────────

def plot_sweep_panel(atlas, psm, sp, outdir):
    fig = plt.figure(figsize=(16, 14), facecolor=COL["bg"])
    gs = gridspec.GridSpec(3, 2, hspace=0.38, wspace=0.32,
                           height_ratios=[1.1, 1, 1])
    gs.update(left=0.07, right=0.96, top=0.92, bottom=0.05)

    ad = atlas.arms["2L"]
    x_mb = ad.window_starts / 1e6
    sweep_start, sweep_end = 18, 24

    # Panel A — full arm
    ax_a = fig.add_subplot(gs[0, :])
    _clean_ax(ax_a, grid_y=True, xlabel="chr2L position (Mb)", ylabel="TMRCA (generations)")
    _panel_label(ax_a, "A")

    all_mean = np.exp(ad.means).mean(axis=0)
    all_std = np.exp(ad.means).std(axis=0)
    ax_a.fill_between(x_mb, all_mean - all_std, all_mean + all_std,
                      color=COL["accent"], alpha=0.12, linewidth=0)
    ax_a.plot(x_mb, all_mean, color=COL["accent"], linewidth=1.2)
    ax_a.axvspan(sweep_start, sweep_end, color=COL["sweep"], alpha=0.06, zorder=0)
    ax_a.axvline(20.7, color=COL["sweep"], linewidth=0.8, linestyle="--", alpha=0.5)
    ax_a.text(20.7 + 0.3, ax_a.get_ylim()[1] * 0.95, "Rdl", fontsize=8,
              color=COL["sweep"], fontstyle="italic", va="top")
    ax_a.set_title("chr2L — Mean TMRCA across all pairs (±1 SD)", loc="left")

    # Panel B — per-population zoom into sweep
    ax_b = fig.add_subplot(gs[1, 0])
    _clean_ax(ax_b, grid_y=True, xlabel="Position (Mb)", ylabel="TMRCA (generations)")
    _panel_label(ax_b, "B")

    mask = (x_mb >= sweep_start) & (x_mb <= sweep_end)
    x_zoom = x_mb[mask]
    for code in POP_ORDER:
        traces = []
        for i, sa in enumerate(psm[code]):
            for sb in psm[code][i + 1:]:
                r = atlas.query_pair("2L", sa, sb)
                if r:
                    traces.append(np.exp(r[0][mask]))
        if traces:
            med = np.median(traces, axis=0)
            ax_b.plot(x_zoom, med, color=POP_COLORS[code], linewidth=1.3, label=code)

    ax_b.axvline(20.7, color=COL["sweep"], linewidth=0.7, linestyle="--", alpha=0.4)
    leg = ax_b.legend(ncol=2, frameon=True, facecolor="white",
                      edgecolor=COL["border"], framealpha=0.95, fontsize=6,
                      borderpad=0.5, columnspacing=0.6)
    leg.get_frame().set_linewidth(0.4)
    ax_b.set_title("Sweep region — per-population TMRCA", loc="left")

    # Panel C — uncertainty
    ax_c = fig.add_subplot(gs[1, 1])
    _clean_ax(ax_c, grid_y=True, xlabel="chr2L position (Mb)", ylabel="Mean log-variance")
    _panel_label(ax_c, "C")

    mean_var = ad.variances.mean(axis=0)
    ax_c.fill_between(x_mb, 0, mean_var, color=COL["accent"], alpha=0.15, linewidth=0)
    ax_c.plot(x_mb, mean_var, color=COL["accent"], linewidth=0.9)
    ax_c.axvspan(sweep_start, sweep_end, color=COL["sweep"], alpha=0.06, zorder=0)
    ax_c.axvline(20.7, color=COL["sweep"], linewidth=0.7, linestyle="--", alpha=0.4)
    ax_c.set_title("Prediction uncertainty (variance spike at sweep)", loc="left")

    # Panel D — sorted pairwise waterfall
    ax_d = fig.add_subplot(gs[2, 0])
    _clean_ax(ax_d, grid_y=True, xlabel="Pairs (sorted by TMRCA)", ylabel="TMRCA (generations)")
    _panel_label(ax_d, "D")

    sw = ad.window_at(20_700_000)
    tmrcas = np.exp(ad.means[:, sw])
    order = np.argsort(tmrcas)
    colors = []
    for idx in order:
        sa, sb = ad.pairs[idx]
        pa, pb = sp[int(sa)], sp[int(sb)]
        colors.append(POP_COLORS[pa] if pa == pb else "#d1d5db")
    ax_d.bar(range(len(order)), tmrcas[order], color=colors, width=1.0,
             edgecolor="none", alpha=0.85)
    ax_d.set_title("Pairwise TMRCA at Rdl locus (sorted)", loc="left")

    # Panel E — isolation-by-distance at sweep
    ax_e = fig.add_subplot(gs[2, 1])
    _clean_ax(ax_e, grid_y=True, grid_x=True,
              xlabel="Great-circle distance (km)", ylabel="TMRCA at Rdl (generations)")
    _panel_label(ax_e, "E")

    dists, tmrca_vals, scatter_colors = [], [], []
    for pi in range(len(ad.pairs)):
        sa, sb = ad.pairs[pi]
        pa, pb = sp[int(sa)], sp[int(sb)]
        if pa != pb:
            dists.append(_haversine(POPULATIONS[pa], POPULATIONS[pb]))
            tmrca_vals.append(np.exp(ad.means[pi, sw]))
            scatter_colors.append(COL["accent"])

    ax_e.scatter(dists, tmrca_vals, s=8, alpha=0.35, c=scatter_colors,
                edgecolors="none", rasterized=True)
    z = np.polyfit(dists, tmrca_vals, 1)
    x_fit = np.linspace(min(dists), max(dists), 100)
    ax_e.plot(x_fit, np.polyval(z, x_fit), color=COL["warm"], linewidth=1.2,
              linestyle="--", alpha=0.7, label=f"slope = {z[0]:.2f}")
    leg = ax_e.legend(frameon=True, facecolor="white", edgecolor=COL["border"],
                      fontsize=7, framealpha=0.95)
    leg.get_frame().set_linewidth(0.4)
    ax_e.set_title("Isolation-by-distance at sweep locus", loc="left")

    fig.text(0.50, 0.97, "Selective Sweep at Rdl — chr2L ~20.7 Mb",
             ha="center", fontsize=16, fontweight="bold", color=COL["text"])

    fig.savefig(outdir / "05_sweep_panel.png", facecolor=COL["bg"],
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("  ✓ 05_sweep_panel.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 6 — Dense TMRCA raster (pairs × windows)
# ──────────────────────────────────────────────────────────────────────

def plot_tmrca_raster(atlas, psm, sp, outdir):
    fig = plt.figure(figsize=(18, 9), facecolor=COL["bg"])
    gs = gridspec.GridSpec(1, 3, width_ratios=[0.03, 1, 0.03], wspace=0.01)
    gs.update(left=0.06, right=0.93, top=0.88, bottom=0.10)

    ad = atlas.arms["2L"]

    within_idx, between_idx, row_pop_labels = [], [], []
    for code in POP_ORDER:
        samps = psm[code]
        for i, sa in enumerate(samps):
            for sb in samps[i + 1:]:
                pidx = ad.pair_index(sa, sb)
                if pidx is not None:
                    within_idx.append(pidx)
                    row_pop_labels.append(code)
    for pi in range(ad.n_pairs):
        if pi not in within_idx:
            between_idx.append(pi)
            row_pop_labels.append("between")
    order = within_idx + between_idx
    sorted_means = np.exp(ad.means[order])

    step = max(1, sorted_means.shape[1] // 900)
    display = sorted_means[:, ::step]

    # population sidebar
    ax_side = fig.add_subplot(gs[0, 0])
    for i, lab in enumerate(row_pop_labels):
        c = POP_COLORS.get(lab, "#e5e7eb")
        ax_side.barh(i, 1, color=c, height=1.0, edgecolor="none")
    ax_side.set_ylim(len(row_pop_labels) - 0.5, -0.5)
    ax_side.set_xlim(0, 1)
    ax_side.set_xticks([])
    ax_side.set_yticks([])
    for s in ax_side.spines.values():
        s.set_visible(False)

    # heatmap
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(display, aspect="auto", cmap=TMRCA_CMAP, interpolation="nearest")
    ax.set_facecolor(COL["bg"])

    n_xticks = 8
    xtick_pos = np.linspace(0, display.shape[1] - 1, n_xticks, dtype=int)
    xtick_labels = [f"{(ad.window_starts[p * step] / 1e6):.0f}" for p in xtick_pos]
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(xtick_labels, fontsize=7)
    ax.set_xlabel("chr2L position (Mb)", fontsize=9, color=COL["text"])
    ax.set_ylabel("Sample pairs (grouped by population)", fontsize=9, color=COL["text"])
    ax.set_yticks([])
    ax.tick_params(colors=COL["text"])
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]:
        ax.spines[s].set_color(COL["border"])
        ax.spines[s].set_linewidth(0.4)

    sweep_col = int(display.shape[1] * 0.42)
    ax.axvline(sweep_col, color=COL["sweep"], linewidth=0.6, linestyle="--", alpha=0.4)

    # population labels on sidebar
    offset = 0
    for code in POP_ORDER:
        n_w = len(psm[code]) * (len(psm[code]) - 1) // 2
        if n_w > 0:
            mid = offset + n_w / 2
            ax_side.text(0.5, mid, code, ha="center", va="center",
                         fontsize=5.5, color=COL["text"], fontweight="bold", rotation=0)
            offset += n_w

    # colorbar
    cbar_ax = fig.add_subplot(gs[0, 2])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("TMRCA (generations)", fontsize=8, color=COL["text"])
    cbar.ax.tick_params(labelsize=6.5, colors=COL["text"])
    cbar.outline.set_edgecolor(COL["border"])
    cbar.outline.set_linewidth(0.4)

    fig.text(0.50, 0.95,
             "Pairwise TMRCA Raster — chr2L",
             ha="center", fontsize=15, fontweight="bold", color=COL["text"])
    fig.text(0.50, 0.92,
             f"{ad.n_pairs:,} pairs × {ad.n_windows:,} windows · "
             "Rows grouped by population · Sweep stripe at ~21 Mb",
             ha="center", fontsize=8.5, color=COL["muted"])

    fig.savefig(outdir / "06_tmrca_raster.png", facecolor=COL["bg"],
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("  ✓ 06_tmrca_raster.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 7 — Composite dashboard
# ──────────────────────────────────────────────────────────────────────

def plot_composite_dashboard(atlas, psm, sp, outdir):
    fig = plt.figure(figsize=(22, 16), facecolor=COL["bg"])
    gs = gridspec.GridSpec(3, 4, hspace=0.40, wspace=0.35,
                           height_ratios=[1.2, 1, 0.9],
                           width_ratios=[1.4, 1, 1, 1])
    gs.update(left=0.05, right=0.97, top=0.91, bottom=0.04)

    # ---- A: geographic map ----
    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.Mercator())
    ax_map.set_extent([-22, 42, -16, 22], crs=ccrs.PlateCarree())
    ax_map.set_facecolor("#f8fafc")
    ax_map.add_feature(cfeature.LAND, facecolor="#f5f5f4", edgecolor="none")
    ax_map.add_feature(cfeature.OCEAN, facecolor="#f0f7ff")
    ax_map.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="#d1d5db")
    transform = ccrs.PlateCarree()
    for code in POP_ORDER:
        info = POPULATIONS[code]
        ax_map.scatter(info["lon"], info["lat"], s=40 + info["n"],
                       c=[REGION_COLORS[info["region"]]], edgecolors="white",
                       linewidth=0.5, transform=transform, zorder=5)
        ax_map.annotate(code, (info["lon"], info["lat"]),
                        xytext=(5, 4), textcoords="offset points",
                        fontsize=5.5, fontweight="bold", color=COL["text"],
                        transform=transform, zorder=6,
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
    _panel_label(ax_map, "A", x=0.02, y=0.98)
    ax_map.set_title("Collection sites", loc="left", fontsize=10, pad=8)

    # ---- B, C: two arm landscapes ----
    for i, arm in enumerate(["2L", "2R"]):
        ax = fig.add_subplot(gs[0, i + 1])
        _clean_ax(ax, grid_y=True, ylabel="TMRCA (gen)" if i == 0 else None)
        _panel_label(ax, chr(66 + i))

        ad = atlas.arms[arm]
        x_mb = ad.window_starts / 1e6
        for code in POP_ORDER[:6]:
            traces = []
            for ii, sa in enumerate(psm[code]):
                for sb in psm[code][ii + 1:]:
                    r = atlas.query_pair(arm, sa, sb)
                    if r:
                        traces.append(np.exp(r[0]))
            if traces:
                ax.plot(x_mb, np.median(traces, axis=0),
                       color=POP_COLORS[code], linewidth=0.8, alpha=0.85)

        if arm == "2L":
            ax.axvline(20.7, color=COL["sweep"], linewidth=0.6, linestyle="--", alpha=0.4)
        ax.set_xlabel(f"chr{arm} (Mb)", fontsize=8)
        ax.set_title(f"chr{arm} — TMRCA landscape", loc="left", fontsize=10)

    # ---- D: arm summary bars ----
    ax_arms = fig.add_subplot(gs[0, 3])
    _clean_ax(ax_arms, grid_y=True, ylabel="Mean log-variance")
    _panel_label(ax_arms, "D")
    arm_vars = [atlas.arms[a].variances.mean() for a in ARM_ORDER]
    bars = ax_arms.bar(ARM_ORDER, arm_vars, color=[ARM_COLORS[a] for a in ARM_ORDER],
                       edgecolor="white", linewidth=0.5, alpha=0.8, width=0.65)
    ax_arms.set_title("Prediction uncertainty by arm", loc="left", fontsize=10)

    # ---- E: population matrix ----
    mat = _pop_tmrca_matrix(atlas, psm)
    ax_mat = fig.add_subplot(gs[1, 0:2])
    _panel_label(ax_mat, "E")
    im = ax_mat.imshow(mat, cmap=TMRCA_CMAP, aspect="equal", interpolation="nearest")
    n = len(POP_ORDER)
    ax_mat.set_xticks(range(n))
    ax_mat.set_yticks(range(n))
    ax_mat.set_xticklabels(POP_ORDER, rotation=40, ha="right", fontsize=7)
    ax_mat.set_yticklabels(POP_ORDER, fontsize=7)
    ax_mat.tick_params(length=0, colors=COL["text"])
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            tc = "white" if v > np.median(mat) else COL["text"]
            ax_mat.text(j, i, f"{v:.0f}", ha="center", va="center",
                       fontsize=5, color=tc)
    for s in ax_mat.spines.values():
        s.set_edgecolor(COL["border"])
        s.set_linewidth(0.4)
    ax_mat.set_title("Population TMRCA matrix", loc="left", fontsize=10)

    # ---- F: violin distributions ----
    ax_viol = fig.add_subplot(gs[1, 2])
    _clean_ax(ax_viol, grid_y=True, ylabel="TMRCA (generations)")
    _panel_label(ax_viol, "F")
    vdata, vlabels, vcols = [], [], []
    for code in POP_ORDER:
        vals = []
        for i, sa in enumerate(psm[code]):
            for sb in psm[code][i + 1:]:
                r = atlas.query_pair("2L", sa, sb)
                if r:
                    vals.append(np.exp(r[0]).mean())
        if vals:
            vdata.append(vals)
            vlabels.append(code)
            vcols.append(POP_COLORS[code])

    parts = ax_viol.violinplot(vdata, positions=range(len(vdata)),
                                showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(vcols[i])
        pc.set_edgecolor(vcols[i])
        pc.set_alpha(0.4)
        pc.set_linewidth(0.5)
    parts["cmedians"].set_color(COL["text"])
    parts["cmedians"].set_linewidth(0.8)
    ax_viol.set_xticks(range(len(vlabels)))
    ax_viol.set_xticklabels(vlabels, rotation=40, ha="right", fontsize=6.5)
    ax_viol.set_title("Within-pop TMRCA distribution", loc="left", fontsize=10)

    # ---- G: isolation-by-distance scatter ----
    ax_ibd = fig.add_subplot(gs[1, 3])
    _clean_ax(ax_ibd, grid_y=True, grid_x=True,
              xlabel="Distance (km)", ylabel="Between-pop TMRCA")
    _panel_label(ax_ibd, "G")
    for i, p1 in enumerate(POP_ORDER):
        for j, p2 in enumerate(POP_ORDER):
            if i < j:
                d = _haversine(POPULATIONS[p1], POPULATIONS[p2])
                v = _pop_mean_tmrca(atlas, psm, p1, p2)
                ax_ibd.scatter(d, v, s=20, c=[COL["accent"]], alpha=0.6,
                              edgecolors="white", linewidth=0.3, zorder=3)
                ax_ibd.annotate(f"{p1}-{p2}", (d, v), fontsize=3.5,
                               color=COL["muted"], xytext=(2, 2),
                               textcoords="offset points")
    ax_ibd.set_title("Isolation-by-distance", loc="left", fontsize=10)

    # ---- Bottom: summary table ----
    ax_tbl = fig.add_subplot(gs[2, :])
    ax_tbl.axis("off")
    ax_tbl.set_facecolor(COL["bg"])

    summary = atlas.summary()
    col_labels = ["Chromosome", "Pairs", "Windows", "Length (Mb)",
                  "Mean log-TMRCA", "Mutation rate"]
    rows = []
    for arm in ARM_ORDER:
        info = summary["per_arm"][arm]
        rows.append([
            f"chr{arm}", f"{info['n_pairs']:,}", f"{info['n_windows']:,}",
            f"{info['arm_length_bp'] / 1e6:.1f}",
            f"{info['mean_log_tmrca']:.3f}", f"{info['mutation_rate']:.1e}",
        ])

    table = ax_tbl.table(cellText=rows, colLabels=col_labels,
                         cellLoc="center", loc="center",
                         colColours=["#f1f5f9"] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    for key, cell in table.get_celld().items():
        cell.set_edgecolor(COL["border"])
        cell.set_linewidth(0.3)
        if key[0] == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(fontweight="bold", color=COL["text"])
        else:
            cell.set_facecolor("white")
            cell.set_text_props(color=COL["text"])

    fig.text(0.50, 0.97,
             "fastcxt TimeAtlas — Anopheles gambiae Ag1000G Dashboard",
             ha="center", fontsize=18, fontweight="bold", color=COL["text"])
    fig.text(0.50, 0.94,
             f"Simulated data · {len(POP_ORDER)} populations · "
             f"5 chromosome arms · {atlas.total_pairs:,} pairs · "
             f"{atlas.total_windows:,} windows",
             ha="center", fontsize=9, color=COL["muted"])

    fig.savefig(outdir / "07_composite_dashboard.png", facecolor=COL["bg"],
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("  ✓ 07_composite_dashboard.png")


# ──────────────────────────────────────────────────────────────────────
# Figure 8 — Scaling comparison (cxt vs fastcxt vs fastcxt+tsinfer)
# ──────────────────────────────────────────────────────────────────────

def plot_scaling_comparison(outdir, benchmark_json=None):
    """Log-log scaling comparison of all three inference modes.

    If benchmark_json is provided, overlays measured runtimes on the
    theoretical curves.  Otherwise uses theoretical scaling only.
    """
    n_samples = np.array([5, 10, 20, 50, 100, 200, 500, 1000])
    n_haploids = 2 * n_samples
    n_pairs = n_haploids * (n_haploids - 1) / 2
    n_nodes = n_haploids - 1

    t_cxt = n_pairs * 0.05
    t_fastcxt_pw = n_pairs * 0.0004 + 0.5
    t_fastcxt_tree = n_nodes * 0.0004 + n_haploids * np.log2(n_haploids) * 0.00005 + 0.5

    measured = {"fastcxt_notree": {}, "fastcxt_tree": {}}
    if benchmark_json and Path(benchmark_json).exists():
        import json as _json
        with open(benchmark_json) as f:
            for r in _json.load(f):
                ns = r.get("n_samples", r.get("n_haploids", 0) // 2)
                measured[r["mode"]][ns] = r["total_s"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor=COL["bg"])

    # ---- Panel A: absolute runtime ----
    ax = axes[0]
    _clean_ax(ax, grid_y=True, grid_x=True,
              xlabel="Diploid samples", ylabel="Total runtime (seconds)")
    _panel_label(ax, "A")

    ax.loglog(n_samples, t_cxt, "o-", color="#9ca3af", linewidth=2.0,
              markersize=5, label="cxt (transformer)", markeredgecolor="white",
              markeredgewidth=0.5, zorder=4)
    ax.loglog(n_samples, t_fastcxt_pw, "s-", color=COL["accent"], linewidth=2.0,
              markersize=5, label="fastcxt (pairwise)", markeredgecolor="white",
              markeredgewidth=0.5, zorder=5)
    ax.loglog(n_samples, t_fastcxt_tree, "D-", color="#059669", linewidth=2.0,
              markersize=5, label="fastcxt (tsinfer)", markeredgecolor="white",
              markeredgewidth=0.5, zorder=5)

    if measured["fastcxt_notree"]:
        mx = sorted(measured["fastcxt_notree"])
        ax.loglog(mx, [measured["fastcxt_notree"][k] for k in mx],
                  "s", color=COL["accent"], markersize=9, markeredgecolor="white",
                  markeredgewidth=1.5, zorder=10, label="measured (pairwise)")
    if measured["fastcxt_tree"]:
        mx = sorted(measured["fastcxt_tree"])
        ax.loglog(mx, [measured["fastcxt_tree"][k] for k in mx],
                  "D", color="#059669", markersize=9, markeredgecolor="white",
                  markeredgewidth=1.5, zorder=10, label="measured (tsinfer)")

    ax.axhline(60, color=COL["border"], linewidth=0.5, linestyle=":")
    ax.axhline(3600, color=COL["border"], linewidth=0.5, linestyle=":")
    ax.text(n_samples[0] * 0.85, 60 * 1.3, "1 min", fontsize=6.5, color=COL["muted"])
    ax.text(n_samples[0] * 0.85, 3600 * 1.3, "1 hour", fontsize=6.5, color=COL["muted"])

    leg = ax.legend(frameon=True, facecolor="white", edgecolor=COL["border"],
                    framealpha=0.95, fontsize=8, borderpad=0.6, loc="upper left")
    leg.get_frame().set_linewidth(0.4)
    ax.set_title("Runtime vs sample size", loc="left")

    # ---- Panel B: speedup over cxt ----
    ax = axes[1]
    _clean_ax(ax, grid_y=True, grid_x=True,
              xlabel="Diploid samples", ylabel="Speedup over cxt (×)")
    _panel_label(ax, "B")

    speedup_pw = t_cxt / t_fastcxt_pw
    speedup_tree = t_cxt / t_fastcxt_tree

    ax.semilogx(n_samples, speedup_pw, "s-", color=COL["accent"], linewidth=2.0,
                markersize=5, label="fastcxt pairwise", markeredgecolor="white",
                markeredgewidth=0.5)
    ax.semilogx(n_samples, speedup_tree, "D-", color="#059669", linewidth=2.0,
                markersize=5, label="fastcxt tsinfer", markeredgecolor="white",
                markeredgewidth=0.5)

    ax.axhline(1, color=COL["border"], linewidth=0.5, linestyle=":")
    ax.text(n_samples[-1] * 0.7, 1 + max(speedup_tree) * 0.01, "cxt baseline",
            fontsize=6.5, color=COL["muted"], ha="right")

    for i, ns in enumerate(n_samples):
        if ns in [10, 50, 200, 1000]:
            ax.annotate(f"{speedup_tree[i]:.0f}×",
                       (ns, speedup_tree[i]),
                       xytext=(0, 10), textcoords="offset points",
                       fontsize=7, color="#059669", ha="center", fontweight="bold")

    leg = ax.legend(frameon=True, facecolor="white", edgecolor=COL["border"],
                    framealpha=0.95, fontsize=8, borderpad=0.6, loc="upper left")
    leg.get_frame().set_linewidth(0.4)
    ax.set_title("Speedup vs cxt baseline", loc="left")

    # ---- Panel C: scaling exponent (local slope) ----
    ax = axes[2]
    _clean_ax(ax, grid_y=True, grid_x=True,
              xlabel="Diploid samples", ylabel="Local scaling exponent")
    _panel_label(ax, "C")

    log_n = np.log10(n_samples)
    for label, t, color, marker in [
        ("cxt", t_cxt, "#9ca3af", "o"),
        ("fastcxt pairwise", t_fastcxt_pw, COL["accent"], "s"),
        ("fastcxt tsinfer", t_fastcxt_tree, "#059669", "D"),
    ]:
        log_t = np.log10(t)
        slope = np.diff(log_t) / np.diff(log_n)
        mid_n = 10 ** ((log_n[:-1] + log_n[1:]) / 2)
        ax.semilogx(mid_n, slope, f"{marker}-", color=color, linewidth=1.8,
                    markersize=4, label=label, markeredgecolor="white",
                    markeredgewidth=0.5)

    ax.axhline(2.0, color="#9ca3af", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axhline(1.0, color="#059669", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.text(n_samples[-2], 2.05, "O(n²)", fontsize=7, color="#9ca3af", ha="right")
    ax.text(n_samples[-2], 1.05, "O(n)", fontsize=7, color="#059669", ha="right")
    ax.set_ylim(0, 2.5)

    leg = ax.legend(frameon=True, facecolor="white", edgecolor=COL["border"],
                    framealpha=0.95, fontsize=7, borderpad=0.5, loc="upper right")
    leg.get_frame().set_linewidth(0.4)
    ax.set_title("Scaling exponent (slope of log-log)", loc="left")

    fig.suptitle("Scaling Comparison — cxt vs fastcxt vs fastcxt + tsinfer",
                 fontsize=16, fontweight="bold", color=COL["text"], y=0.99)
    fig.text(0.50, 0.95,
             "Theoretical scaling calibrated to measured per-pair / per-node runtimes · "
             "1 Mb segments · CUDA inference",
             ha="center", fontsize=8.5, color=COL["muted"])

    fig.savefig(outdir / "08_scaling_comparison.png", facecolor=COL["bg"],
                bbox_inches="tight", pad_inches=0.4)
    plt.close(fig)
    print("  ✓ 08_scaling_comparison.png")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def _load_real_atlas(atlas_dir: str, pop_map: str | None):
    """Load a real atlas from disk with optional population metadata."""
    atlas = TimeAtlas.load(atlas_dir)

    pop_sample_map: dict[str, list[int]] = {}
    sample_pop: dict[int, str] = {}

    if pop_map and Path(pop_map).exists():
        import json as _json
        with open(pop_map) as f:
            raw = _json.load(f)
        pop_sample_map = {k: [int(x) for x in v] for k, v in raw.items()}
        for pop, samples in pop_sample_map.items():
            for s in samples:
                sample_pop[s] = pop
    else:
        first_arm = list(atlas.arms.values())[0]
        all_samples = sorted(set(first_arm.pairs[:, 0]) | set(first_arm.pairs[:, 1]))
        n = len(all_samples)
        pops = list(POPULATIONS.keys())
        chunk = max(1, n // len(pops))
        for i, pop in enumerate(pops):
            start, end = i * chunk, (i + 1) * chunk if i < len(pops) - 1 else n
            pop_sample_map[pop] = [int(s) for s in all_samples[start:end]]
            for s in all_samples[start:end]:
                sample_pop[int(s)] = pop

    return atlas, pop_sample_map, sample_pop


def main():
    parser = argparse.ArgumentParser(description="TimeAtlas showcase plots")
    parser.add_argument("--outdir", default="figures", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-pop", type=int, default=4)
    parser.add_argument("--window-size", type=int, default=50_000)
    parser.add_argument("--atlas-dir", default=None,
                        help="Path to a saved TimeAtlas directory (uses real data)")
    parser.add_argument("--pop-map", default=None,
                        help="Path to JSON mapping pop codes → sample ID lists")
    parser.add_argument("--benchmark-json", default=None,
                        help="Path to fastcxt_scaling.json from benchmark stage")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("━" * 64)
    print("  fastcxt TimeAtlas Showcase")
    print("━" * 64)

    if args.atlas_dir and Path(args.atlas_dir).exists():
        print(f"  Loading real atlas from {args.atlas_dir} ...")
        atlas, psm, sp = _load_real_atlas(args.atlas_dir, args.pop_map)
        data_label = "real"
    else:
        print("  No atlas provided — generating simulated placeholder data")
        rng = np.random.default_rng(args.seed)
        atlas, psm, sp = build_simulated_atlas(
            rng, n_per_pop=args.samples_per_pop, window_size=args.window_size)
        data_label = "simulated"

    s = atlas.summary()
    print(f"  Atlas ({data_label}): {atlas}")
    print(f"  {s['n_arms']} arms · {atlas.total_pairs:,} pairs · "
          f"{atlas.total_windows:,} windows")
    print()

    plot_collection_sites(atlas, psm, sp, outdir)
    plot_connectivity_map(atlas, psm, sp, outdir)
    plot_genome_landscape(atlas, psm, sp, outdir)
    plot_population_heatmap(atlas, psm, sp, outdir)
    plot_sweep_panel(atlas, psm, sp, outdir)
    plot_tmrca_raster(atlas, psm, sp, outdir)
    plot_composite_dashboard(atlas, psm, sp, outdir)
    plot_scaling_comparison(outdir, benchmark_json=args.benchmark_json)

    print()
    print("━" * 64)
    print(f"  All {data_label}-data figures saved to {outdir}/")
    print("━" * 64)


if __name__ == "__main__":
    main()
