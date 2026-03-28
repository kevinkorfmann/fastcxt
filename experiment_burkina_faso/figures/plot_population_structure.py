#!/usr/bin/env python3
"""PCA and geographic correlation from population-level TMRCA summaries."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Config (mirrors generate_all_figures.py)
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("/tmp/het_results")
FIG_DIR = Path(__file__).parent / "population_structure"
FIG_DIR.mkdir(exist_ok=True)

POPULATIONS = [
    "Burkina_Faso", "Cameroon", "Central_African_Republic",
    "Democratic_Republic_of_the_Congo", "Equatorial_Guinea", "Gabon",
    "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Mali",
    "Mayotte", "Mozambique", "Tanzania", "Uganda",
]

ARMS = ["2L", "2R", "3L", "3R", "X"]
ARM_GROUPS = {
    "2L": ["2La_hom_standard", "2La_heterozygous", "2La_hom_inverted"],
    "2R": ["2Rb_hom_standard", "2Rb_heterozygous", "2Rb_hom_inverted"],
    "3L": ["all"], "3R": ["all"], "X": ["all"],
}
PAIR_TYPES = ["intra", "inter"]

POP_COORDS = {
    "Burkina_Faso": (11.269, -4.037),
    "Cameroon": (5.627, 13.631),
    "Central_African_Republic": (4.367, 18.583),
    "Democratic_Republic_of_the_Congo": (4.283, 21.017),
    "Equatorial_Guinea": (3.700, 8.700),
    "Gabon": (0.384, 9.455),
    "Gambia": (13.567, -14.917),
    "Ghana": (5.940, -0.246),
    "Guinea": (8.870, -9.774),
    "Guinea-Bissau": (12.272, -14.222),
    "Kenya": (-3.511, 39.909),
    "Mali": (11.670, -8.042),
    "Mayotte": (-12.857, 45.137),
    "Mozambique": (-23.716, 35.299),
    "Tanzania": (-3.451, 35.285),
    "Uganda": (0.072, 32.041),
}

REGION = {
    "Burkina_Faso": "West", "Cameroon": "Central", "Central_African_Republic": "Central",
    "Democratic_Republic_of_the_Congo": "Central", "Equatorial_Guinea": "Central",
    "Gabon": "Central", "Gambia": "West", "Ghana": "West", "Guinea": "West",
    "Guinea-Bissau": "West", "Kenya": "East", "Mali": "West",
    "Mayotte": "East", "Mozambique": "East", "Tanzania": "East", "Uganda": "East",
}

REGION_CLR = {"West": "#2563eb", "Central": "#8b5cf6", "East": "#ef4444"}

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

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fafafa",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_group(pop, arm, group, pair_type):
    d = RESULTS_DIR / pop / arm / f"{group}_{pair_type}"
    if not (d / "means.npz").exists():
        return None
    return float(np.mean(np.load(d / "means.npz")["means"]))


def build_feature_matrix():
    """Build (n_pops, n_features) matrix of mean log(TMRCA) per group."""
    cols = []
    col_labels = []
    for arm in ARMS:
        for group in ARM_GROUPS[arm]:
            for pt in PAIR_TYPES:
                cols.append((arm, group, pt))
                col_labels.append(f"{arm}_{group}_{pt}")

    pops_with_data = []
    rows = []
    for pop in POPULATIONS:
        row = []
        has_any = False
        for arm, group, pt in cols:
            val = load_group(pop, arm, group, pt)
            row.append(val if val is not None else np.nan)
            if val is not None:
                has_any = True
        if has_any:
            pops_with_data.append(pop)
            rows.append(row)

    matrix = np.array(rows)
    return matrix, pops_with_data, col_labels


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def main():
    matrix, pops, col_labels = build_feature_matrix()
    print(f"Feature matrix: {matrix.shape} ({len(pops)} populations x {matrix.shape[1]} features)")

    # --- Impute NaN columns with column mean for PCA ---
    mat_imputed = matrix.copy()
    for j in range(mat_imputed.shape[1]):
        col = mat_imputed[:, j]
        mask = np.isfinite(col)
        if mask.any() and not mask.all():
            mat_imputed[~mask, j] = np.nanmean(col)

    # Drop columns that are all NaN
    valid_cols = np.any(np.isfinite(mat_imputed), axis=0)
    mat_clean = mat_imputed[:, valid_cols]
    labels_clean = [col_labels[i] for i in range(len(col_labels)) if valid_cols[i]]

    # Use only inversion-free arms for cleaner demographic signal
    neutral_mask = np.array(["3L" in l or "3R" in l or "X_" in l for l in labels_clean])
    mat_neutral = mat_clean[:, neutral_mask]

    # =====================================================================
    # Figure 1: PCA — all features
    # =====================================================================
    pca = PCA(n_components=min(3, mat_clean.shape[1], mat_clean.shape[0]))
    scores = pca.fit_transform(mat_clean)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PC1 vs PC2
    ax = axes[0]
    for i, pop in enumerate(pops):
        c = REGION_CLR.get(REGION.get(pop, ""), "#888")
        ax.scatter(scores[i, 0], scores[i, 1], c=c, s=80, zorder=3, edgecolors="white", lw=0.8)
        ax.annotate(pop.replace("_", " "), (scores[i, 0], scores[i, 1]),
                    fontsize=7.5, ha="left", va="bottom", xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of coalescence profiles (all arms + karyotypes)", fontweight="bold")

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=REGION_CLR[r],
               markersize=8, label=f"{r} Africa") for r in ["West", "Central", "East"]]
    ax.legend(handles=handles, fontsize=9)

    # PC1 vs PC3 if available
    ax = axes[1]
    if scores.shape[1] >= 3:
        for i, pop in enumerate(pops):
            c = REGION_CLR.get(REGION.get(pop, ""), "#888")
            ax.scatter(scores[i, 0], scores[i, 2], c=c, s=80, zorder=3, edgecolors="white", lw=0.8)
            ax.annotate(pop.replace("_", " "), (scores[i, 0], scores[i, 2]),
                        fontsize=7.5, ha="left", va="bottom", xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
        ax.set_title("PC1 vs PC3", fontweight="bold")
    else:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_all_features.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "pca_all_features.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved pca_all_features.png")

    # =====================================================================
    # Figure 2: PCA — neutral arms only (3L, 3R, X)
    # =====================================================================
    if mat_neutral.shape[1] >= 2:
        pca_n = PCA(n_components=min(2, mat_neutral.shape[1], mat_neutral.shape[0]))
        scores_n = pca_n.fit_transform(mat_neutral)

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, pop in enumerate(pops):
            c = REGION_CLR.get(REGION.get(pop, ""), "#888")
            ax.scatter(scores_n[i, 0], scores_n[i, 1], c=c, s=100, zorder=3, edgecolors="white", lw=0.8)
            ax.annotate(pop.replace("_", " "), (scores_n[i, 0], scores_n[i, 1]),
                        fontsize=8, ha="left", va="bottom", xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel(f"PC1 ({pca_n.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca_n.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title("PCA of coalescence profiles (neutral arms: 3L, 3R, X)", fontweight="bold")
        ax.legend(handles=handles, fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "pca_neutral_arms.png", dpi=150, bbox_inches="tight")
        fig.savefig(FIG_DIR / "pca_neutral_arms.pdf", bbox_inches="tight")
        plt.close()
        print("  Saved pca_neutral_arms.png")

    # =====================================================================
    # Figure 3: Geographic distance vs coalescence distance (Mantel-like)
    # =====================================================================
    n = len(pops)
    geo_dists = []
    coal_dists = []

    for i in range(n):
        for j in range(i+1, n):
            lat_i, lon_i = POP_COORDS[pops[i]]
            lat_j, lon_j = POP_COORDS[pops[j]]
            geo_dists.append(haversine_km(lat_i, lon_i, lat_j, lon_j))
            # Euclidean distance in coalescence feature space
            diff = mat_clean[i] - mat_clean[j]
            coal_dists.append(np.sqrt(np.nansum(diff**2)))

    geo_dists = np.array(geo_dists)
    coal_dists = np.array(coal_dists)

    r_pearson, p_pearson = pearsonr(geo_dists, coal_dists)
    r_spearman, p_spearman = spearmanr(geo_dists, coal_dists)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Color by region pair
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            ri, rj = REGION.get(pops[i], ""), REGION.get(pops[j], "")
            if ri == rj:
                c = REGION_CLR.get(ri, "#888")
                marker = "o"
            else:
                c = "#999"
                marker = "x"
            ax.scatter(geo_dists[idx], coal_dists[idx], c=c, s=40, alpha=0.7,
                       marker=marker, zorder=3, edgecolors="none")
            idx += 1

    # Fit line
    z = np.polyfit(geo_dists, coal_dists, 1)
    x_line = np.linspace(geo_dists.min(), geo_dists.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), color="#dc2626", lw=1.5, ls="--", alpha=0.7)

    ax.set_xlabel("Geographic distance (km)")
    ax.set_ylabel("Coalescence profile distance (Euclidean)")
    ax.set_title(f"Isolation by distance — r={r_pearson:.3f} (p={p_pearson:.1e}), "
                 f"Spearman={r_spearman:.3f}", fontweight="bold")

    # Legend
    within = Line2D([0], [0], marker="o", color="w", markerfacecolor="#666", markersize=8, label="Within region")
    between = Line2D([0], [0], marker="x", color="w", markerfacecolor="#999",
                     markeredgecolor="#999", markersize=8, label="Between regions")
    ax.legend(handles=[within, between], fontsize=9)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "geo_vs_coal_distance.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "geo_vs_coal_distance.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved geo_vs_coal_distance.png (r={r_pearson:.3f}, p={p_pearson:.1e})")

    # =====================================================================
    # Figure 4: PCA projected onto Africa map
    # =====================================================================
    fig, ax = plt.subplots(figsize=(12, 10))
    poly = Polygon([(lon, lat) for lon, lat in AFRICA_OUTLINE],
                   closed=True, facecolor="#f5f5f4", edgecolor="#bbb", lw=0.8)
    ax.add_patch(poly)
    ax.set_xlim(-20, 52)
    ax.set_ylim(-37, 40)
    ax.set_aspect("equal")

    # Use PC1 for color, PC2 for size
    pc1 = scores[:, 0]
    pc1_norm = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-10)

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    cmap = plt.cm.RdYlBu_r
    norm = Normalize(vmin=pc1.min(), vmax=pc1.max())

    for i, pop in enumerate(pops):
        lat, lon = POP_COORDS[pop]
        c = cmap(norm(pc1[i]))
        ax.scatter(lon, lat, c=[c], s=150, zorder=5, edgecolors="white", lw=1.2)
        ax.annotate(pop.replace("_", " "), (lon, lat),
                    fontsize=8, ha="left", va="bottom", xytext=(5, 5),
                    textcoords="offset points", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    sm = ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
    cbar.set_label("PC1 score (coalescence profile)", fontsize=10)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Coalescence PCA projected onto geography", fontsize=13, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_color("#ddd")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_on_map.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "pca_on_map.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved pca_on_map.png")

    # =====================================================================
    # Figure 5: Hierarchical clustering dendrogram
    # =====================================================================
    from scipy.cluster.hierarchy import linkage, dendrogram

    # Use coalescence distance matrix
    dist_matrix = squareform(coal_dists)
    Z = linkage(coal_dists, method="average")

    fig, ax = plt.subplots(figsize=(12, 5))
    pop_labels = [p.replace("_", " ") for p in pops]
    dn = dendrogram(Z, labels=pop_labels, ax=ax, leaf_rotation=45,
                    leaf_font_size=10, above_threshold_color="#999")

    # Color labels by region
    xlbls = ax.get_xticklabels()
    for lbl in xlbls:
        name = lbl.get_text().replace(" ", "_")
        c = REGION_CLR.get(REGION.get(name, ""), "#333")
        lbl.set_color(c)
        lbl.set_fontweight("bold")

    ax.set_ylabel("Coalescence profile distance")
    ax.set_title("Hierarchical clustering of populations by coalescence profile", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "dendrogram.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "dendrogram.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved dendrogram.png")


if __name__ == "__main__":
    main()
