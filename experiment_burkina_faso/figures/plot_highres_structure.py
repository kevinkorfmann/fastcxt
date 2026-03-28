#!/usr/bin/env python3
"""PCA and UMAP on high-resolution per-window TMRCA profiles."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

RESULTS_DIR = Path("/tmp/het_results")
FIG_DIR = Path(__file__).parent / "population_structure"
FIG_DIR.mkdir(exist_ok=True)

POPULATIONS = [
    "Burkina_Faso", "Cameroon", "Central_African_Republic",
    "Democratic_Republic_of_the_Congo", "Equatorial_Guinea", "Gabon",
    "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Mali",
    "Mayotte", "Mozambique", "Tanzania", "Uganda",
]

REGION = {
    "Burkina_Faso": "West", "Cameroon": "Central", "Central_African_Republic": "Central",
    "Democratic_Republic_of_the_Congo": "Central", "Equatorial_Guinea": "Central",
    "Gabon": "Central", "Gambia": "West", "Ghana": "West", "Guinea": "West",
    "Guinea-Bissau": "West", "Kenya": "East", "Mali": "West",
    "Mayotte": "East", "Mozambique": "East", "Tanzania": "East", "Uganda": "East",
}
REGION_CLR = {"West": "#2563eb", "Central": "#16a34a", "East": "#ef4444"}

POP_COORDS = {
    "Burkina_Faso": (11.269, -4.037), "Cameroon": (5.627, 13.631),
    "Central_African_Republic": (4.367, 18.583), "Democratic_Republic_of_the_Congo": (4.283, 21.017),
    "Equatorial_Guinea": (3.700, 8.700), "Gabon": (0.384, 9.455),
    "Gambia": (13.567, -14.917), "Ghana": (5.940, -0.246),
    "Guinea": (8.870, -9.774), "Guinea-Bissau": (12.272, -14.222),
    "Kenya": (-3.511, 39.909), "Mali": (11.670, -8.042),
    "Mayotte": (-12.857, 45.137), "Mozambique": (-23.716, 35.299),
    "Tanzania": (-3.451, 35.285), "Uganda": (0.072, 32.041),
}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fafafa",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
})


def reconstruct_profile(pop, arm, group, pair_type="intra"):
    """Reconstruct (n_pairs, n_total_windows) from block-pair layout."""
    d = RESULTS_DIR / pop / arm / f"{group}_{pair_type}"
    if not (d / "means.npz").exists():
        return None, None
    means = np.load(d / "means.npz")["means"]
    imap = np.load(d / "index_map.npy")
    blocks = json.load(open(d / "blocks.json"))
    cfg = json.load(open(d / "config.json"))
    n_pairs, n_blocks, wpb = cfg["n_pairs"], len(blocks), means.shape[1]
    n_total = n_blocks * wpb
    pm = np.zeros((n_pairs, n_total), dtype=np.float32)
    for r in range(means.shape[0]):
        bi, pi = imap[r]
        s = bi * wpb
        pm[pi, s:s+wpb] = means[r]
    ws = np.zeros(n_total, dtype=np.int64)
    for b in blocks:
        for w in range(wpb):
            ws[b["idx"]*wpb + w] = b["start"] + w * 200
    return pm, ws


def get_pop_median_profile(pop, arm="3L", group="all"):
    """Get median TMRCA profile across all intra pairs for a population."""
    pm, ws = reconstruct_profile(pop, arm, group)
    if pm is None:
        return None, None
    # Downsample to block resolution (median across 500 windows per block)
    n_blocks = pm.shape[1] // 500
    block_medians = np.zeros((pm.shape[0], n_blocks))
    for b in range(n_blocks):
        block_medians[:, b] = np.median(pm[:, b*500:(b+1)*500], axis=1)
    # Population median across pairs
    pop_profile = np.median(block_medians, axis=0)
    block_mids = ws[::500][:n_blocks] / 1e6
    return pop_profile, block_mids


def main():
    # =====================================================================
    # Build high-res feature matrix: populations x blocks (3L + 3R)
    # =====================================================================
    profiles = {}
    for pop in POPULATIONS:
        parts = []
        for arm in ["3L", "3R"]:
            prof, mids = get_pop_median_profile(pop, arm, "all")
            if prof is not None:
                parts.append(prof)
        if parts:
            profiles[pop] = np.concatenate(parts)

    pops = list(profiles.keys())
    mat = np.array([profiles[p] for p in pops])
    print(f"High-res feature matrix: {mat.shape} ({len(pops)} pops x {mat.shape[1]} blocks)")

    # =====================================================================
    # Figure 1: PCA on high-res profiles
    # =====================================================================
    pca = PCA(n_components=min(3, len(pops)-1))
    scores = pca.fit_transform(mat)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=REGION_CLR[r],
               markersize=8, label=f"{r} Africa") for r in ["West", "Central", "East"]]

    for ax_idx, (pc_x, pc_y, title) in enumerate([(0, 1, "PC1 vs PC2"), (0, 2, "PC1 vs PC3")]):
        ax = axes[ax_idx]
        if pc_y >= scores.shape[1]:
            ax.set_visible(False)
            continue
        for i, pop in enumerate(pops):
            c = REGION_CLR.get(REGION.get(pop, ""), "#888")
            ax.scatter(scores[i, pc_x], scores[i, pc_y], c=c, s=100, zorder=3,
                       edgecolors="white", lw=0.8)
            ax.annotate(pop.replace("_", " "), (scores[i, pc_x], scores[i, pc_y]),
                        fontsize=7.5, ha="left", va="bottom", xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel(f"PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)")
        ax.set_ylabel(f"PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)")
        ax.set_title(title, fontweight="bold")
        ax.legend(handles=handles, fontsize=9)

    fig.suptitle(f"PCA on block-resolution TMRCA profiles (3L + 3R, {mat.shape[1]} blocks)",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_highres.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "pca_highres.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved pca_highres.png")

    # =====================================================================
    # Figure 2: PCA loadings + accessibility + gene annotations
    # =====================================================================
    loadings = pca.components_[0]
    prof_3l, mids_3l = get_pop_median_profile(pops[0], "3L", "all")
    prof_3r, mids_3r = get_pop_median_profile(pops[0], "3R", "all")
    n_3l = len(prof_3l)

    # Build x-axis in Mb: 3L positions then 3R positions offset by gap
    gap = 5  # Mb visual gap between arms
    x_3l = mids_3l
    x_3r = mids_3r + mids_3l[-1] + gap

    x_mb = np.concatenate([x_3l, x_3r])

    fig, (ax_load, ax_acc) = plt.subplots(2, 1, figsize=(18, 5),
        height_ratios=[3, 1], sharex=True, gridspec_kw={"hspace": 0.08})

    # --- Top: loadings ---
    ax_load.fill_between(x_mb[:n_3l], loadings[:n_3l], color="#059669", alpha=0.4, label="3L")
    ax_load.fill_between(x_mb[n_3l:], loadings[n_3l:], color="#d97706", alpha=0.4, label="3R")
    ax_load.axhline(0, color="black", lw=0.5)
    ax_load.axvline(x_3l[-1] + gap/2, color="#ccc", lw=1, ls="--")

    # Annotate top loading peaks with nearest genes
    genes_data = None
    genes_file = RESULTS_DIR / "genes.json"
    if genes_file.exists():
        genes_data = json.load(open(genes_file))

    def find_gene_at(arm, pos_bp):
        if genes_data is None or arm not in genes_data:
            return None
        best_dist, best = 50_000, None
        for g in genes_data[arm]:
            mid = (g["start"] + g["end"]) / 2
            dist = abs(mid - pos_bp)
            if dist < best_dist:
                best_dist = dist
                best = g
        if best is None:
            return None
        return best.get("symbol") or best.get("description", "")[:20] or best.get("id", "")

    # Find top 8 peaks by absolute loading
    abs_loadings = np.abs(loadings)
    # Exclude peaks within 5 blocks of each other
    peak_indices = []
    sorted_idx = np.argsort(-abs_loadings)
    for idx in sorted_idx:
        if len(peak_indices) >= 8:
            break
        if all(abs(idx - p) > 5 for p in peak_indices):
            peak_indices.append(idx)

    texts = []
    for pi in peak_indices:
        if pi < n_3l:
            arm = "3L"
            pos_bp = int(mids_3l[pi] * 1e6)
        else:
            arm = "3R"
            pos_bp = int(mids_3r[pi - n_3l] * 1e6)

        gene = find_gene_at(arm, pos_bp)
        if gene:
            label = f"{gene}\n({x_mb[pi]:.1f} Mb)"
        else:
            label = f"{x_mb[pi]:.1f} Mb"

        t = ax_load.annotate(label, (x_mb[pi], loadings[pi]),
                    fontsize=7, ha="center", va="bottom" if loadings[pi] > 0 else "top",
                    xytext=(0, 8 if loadings[pi] > 0 else -8),
                    textcoords="offset points", color="#333",
                    arrowprops=dict(arrowstyle="-", color="#999", lw=0.5),
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ddd", alpha=0.8))

    ax_load.set_ylabel("PC1 loading", fontsize=11)
    ax_load.set_title("PC1 loadings — genomic regions driving population differentiation",
                      fontweight="bold", fontsize=12)
    ax_load.legend(fontsize=10, loc="upper right")

    # --- Bottom: accessibility/missingness track ---
    acc_file = RESULTS_DIR / "accessibility_100kb.npz"
    if acc_file.exists():
        acc = np.load(acc_file)
        for arm, x_offset, color in [("3L", 0, "#059669"), ("3R", mids_3l[-1] + gap, "#d97706")]:
            frac = acc[f"{arm}_frac"]
            amids = acc[f"{arm}_mids"]
            missing = 1.0 - frac
            kernel = 3
            if len(missing) > kernel:
                missing = np.convolve(missing, np.ones(kernel)/kernel, mode="same")
            ax_acc.fill_between(amids + x_offset, 0, missing, color="#E53935", alpha=0.3)
            ax_acc.plot(amids + x_offset, missing, color="#E53935", lw=0.6, alpha=0.6)

    ax_acc.axvline(x_3l[-1] + gap/2, color="#ccc", lw=1, ls="--")
    ax_acc.set_ylim(0, 0.8)
    ax_acc.invert_yaxis()
    ax_acc.set_ylabel("Missing", fontsize=9, color="#E53935")
    ax_acc.set_xlabel("Position (Mb)", fontsize=11)
    for spine in ax_acc.spines.values():
        spine.set_color("#ddd")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_loadings.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "pca_loadings.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved pca_loadings.png")

    # =====================================================================
    # Figure 3: UMAP
    # =====================================================================
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=min(5, len(pops)-1), min_dist=0.3,
                            random_state=42, metric="euclidean")
        embedding = reducer.fit_transform(mat)

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, pop in enumerate(pops):
            c = REGION_CLR.get(REGION.get(pop, ""), "#888")
            ax.scatter(embedding[i, 0], embedding[i, 1], c=c, s=120, zorder=3,
                       edgecolors="white", lw=0.8)
            ax.annotate(pop.replace("_", " "), (embedding[i, 0], embedding[i, 1]),
                        fontsize=8, ha="left", va="bottom", xytext=(4, 4),
                        textcoords="offset points")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"UMAP of block-resolution TMRCA profiles (3L + 3R, {mat.shape[1]} blocks)",
                     fontweight="bold")
        ax.legend(handles=handles, fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "umap_highres.png", dpi=150, bbox_inches="tight")
        fig.savefig(FIG_DIR / "umap_highres.pdf", bbox_inches="tight")
        plt.close()
        print("  Saved umap_highres.png")
    except ImportError:
        print("  UMAP skipped (pip install umap-learn)")

    # =====================================================================
    # Figure 4: Per-pair PCA (all pairs from all populations on 3L)
    # =====================================================================
    pair_profiles = []
    pair_labels = []
    pair_pops = []
    for pop in pops:
        pm, ws = reconstruct_profile(pop, "3L", "all")
        if pm is None:
            continue
        # Downsample each pair to block resolution
        n_blocks = pm.shape[1] // 500
        for pi in range(pm.shape[0]):
            block_vals = np.array([np.median(pm[pi, b*500:(b+1)*500]) for b in range(n_blocks)])
            pair_profiles.append(block_vals)
            pair_labels.append(pop)
            pair_pops.append(pop)

    pair_mat = np.array(pair_profiles)
    print(f"Per-pair matrix: {pair_mat.shape} ({len(pair_profiles)} pairs x {pair_mat.shape[1]} blocks)")

    pca_pairs = PCA(n_components=2)
    pair_scores = pca_pairs.fit_transform(pair_mat)

    fig, ax = plt.subplots(figsize=(10, 8))
    for pop in pops:
        mask = np.array([p == pop for p in pair_pops])
        c = REGION_CLR.get(REGION.get(pop, ""), "#888")
        ax.scatter(pair_scores[mask, 0], pair_scores[mask, 1], c=c, s=15, alpha=0.5,
                   edgecolors="none", label=pop.replace("_", " "))
    ax.set_xlabel(f"PC1 ({pca_pairs.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_pairs.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Per-pair PCA on chr3L ({len(pair_profiles)} pairs across {len(pops)} populations)",
                 fontweight="bold")
    ax.legend(fontsize=7, ncol=2, loc="best", markerscale=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_perpair.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "pca_perpair.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved pca_perpair.png")


if __name__ == "__main__":
    main()
