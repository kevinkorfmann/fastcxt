#!/usr/bin/env python3
"""PCA and UMAP on high-resolution per-window TMRCA profiles."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
ARM_CLR = {"2L": "#2563eb", "2R": "#7c3aed", "3L": "#059669", "3R": "#d97706", "X": "#dc2626"}
ARM_GROUPS = {"2L": "2La_hom_standard", "2R": "2Rb_hom_standard",
              "3L": "all", "3R": "all", "X": "all"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fafafa",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
})

HANDLES = [Line2D([0], [0], marker="o", color="w", markerfacecolor=REGION_CLR[r],
           markersize=8, label=f"{r} Africa") for r in ["West", "Central", "East"]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reconstruct_profile(pop, arm, group, pair_type="intra"):
    d = RESULTS_DIR / pop / arm / f"{group}_{pair_type}"
    if not (d / "means.npz").exists():
        return None, None
    means = np.load(d / "means.npz")["means"]
    imap = np.load(d / "index_map.npy")
    blocks = json.load(open(d / "blocks.json"))
    cfg = json.load(open(d / "config.json"))
    np_, nb, wpb = cfg["n_pairs"], len(blocks), means.shape[1]
    nt = nb * wpb
    pm = np.zeros((np_, nt), dtype=np.float32)
    for r in range(means.shape[0]):
        bi, pi = imap[r]
        s = bi * wpb
        pm[pi, s:s+wpb] = means[r]
    ws = np.zeros(nt, dtype=np.int64)
    for b in blocks:
        for w in range(wpb):
            ws[b["idx"]*wpb + w] = b["start"] + w * 200
    return pm, ws


def get_pop_median_profile(pop, arm, group):
    pm, ws = reconstruct_profile(pop, arm, group)
    if pm is None:
        return None, None
    nb = pm.shape[1] // 500
    block_med = np.zeros((pm.shape[0], nb))
    for b in range(nb):
        block_med[:, b] = np.median(pm[:, b*500:(b+1)*500], axis=1)
    return np.median(block_med, axis=0), ws[::500][:nb] / 1e6


def find_gene(genes_data, arm, pos_bp):
    if genes_data is None or arm not in genes_data:
        return None
    best_dist, best = 50_000, None
    for g in genes_data[arm]:
        d = abs((g["start"] + g["end"]) / 2 - pos_bp)
        if d < best_dist:
            best_dist, best = d, g
    if best is None:
        return None
    sym = best.get("symbol", "").strip()
    return sym if sym else best.get("id", "")


def build_multi_arm_matrix(pops, arms):
    """Build (n_pops, total_blocks) matrix across multiple arms."""
    arm_data = {}
    for arm in arms:
        profs = []
        mids = None
        for pop in pops:
            p, m = get_pop_median_profile(pop, arm, ARM_GROUPS[arm])
            if p is not None:
                profs.append(p)
                mids = m
            else:
                profs.append(None)
        if mids is not None:
            arm_data[arm] = (profs, mids)

    # Build matrix, impute missing populations with column mean
    rows = []
    for pi in range(len(pops)):
        parts = []
        for arm in arms:
            if arm in arm_data:
                prof = arm_data[arm][0][pi]
                if prof is not None:
                    parts.append(prof)
                else:
                    parts.append(np.full(len(arm_data[arm][1]), np.nan))
        rows.append(np.concatenate(parts))

    mat = np.array(rows)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        mask = np.isfinite(col)
        if mask.any() and not mask.all():
            mat[~mask, j] = np.nanmean(col)

    # Build segments: (arm, start_idx, n_blocks, mids_mb)
    segments = []
    offset = 0
    for arm in arms:
        if arm in arm_data:
            n = len(arm_data[arm][1])
            segments.append((arm, offset, n, arm_data[arm][1]))
            offset += n

    return mat, segments


def plot_loadings(pca_obj, segments, fname, title_suffix=""):
    """Plot PC1 loadings with gene annotations and accessibility track."""
    loadings = pca_obj.components_[0]
    gap = 3  # Mb visual gap between arms

    # Build x-axis in Mb
    x_mb = np.zeros(len(loadings))
    arm_bounds = []
    cum = 0
    for arm, si, nb, mids in segments:
        for i in range(nb):
            x_mb[si + i] = mids[i] + cum
        arm_bounds.append((arm, cum, cum + mids[-1], si, nb))
        cum += mids[-1] + gap

    fig, (ax_l, ax_a) = plt.subplots(2, 1, figsize=(22, 5.5),
        height_ratios=[3, 1], sharex=True, gridspec_kw={"hspace": 0.08})

    # Loadings per arm
    for arm, mb0, mb1, si, nb in arm_bounds:
        sl = slice(si, si + nb)
        ax_l.fill_between(x_mb[sl], loadings[sl], color=ARM_CLR.get(arm, "#888"), alpha=0.4, label=arm)
    ax_l.axhline(0, color="black", lw=0.5)

    # Separators
    for i, (arm, mb0, mb1, _, _) in enumerate(arm_bounds):
        if i > 0:
            mid = (arm_bounds[i-1][2] + mb0) / 2
            ax_l.axvline(mid, color="#ccc", lw=1, ls="--")
            ax_a.axvline(mid, color="#ccc", lw=1, ls="--")

    # Gene annotations at top 25 peaks
    genes_file = RESULTS_DIR / "genes.json"
    gd = json.load(open(genes_file)) if genes_file.exists() else None

    abs_l = np.abs(loadings)
    peaks = []
    for idx in np.argsort(-abs_l):
        if len(peaks) >= 25:
            break
        if all(abs(idx - p) > 3 for p in peaks):
            peaks.append(idx)

    texts = []
    for pi in peaks:
        arm_name = None
        for arm, mb0, mb1, si, nb in arm_bounds:
            if si <= pi < si + nb:
                arm_name = arm
                local = pi - si
                for sa, so, sn, sm in segments:
                    if sa == arm:
                        pos_bp = int(sm[local] * 1e6)
                        break
                break
        gene = find_gene(gd, arm_name, pos_bp) if arm_name else None
        label = f"{gene} ({x_mb[pi]:.0f})" if gene else f"{x_mb[pi]:.0f} Mb"
        t = ax_l.text(x_mb[pi], loadings[pi], label, fontsize=6, ha="center",
                      va="bottom" if loadings[pi] > 0 else "top", color="#333",
                      bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="#ddd", alpha=0.8))
        texts.append(t)

    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax_l, arrowprops=dict(arrowstyle="-", color="#999", lw=0.5),
                    force_text=(0.5, 0.8), force_points=(0.3, 0.3), expand_text=(1.2, 1.4))
    except ImportError:
        pass

    ax_l.set_ylabel("PC1 loading", fontsize=11)
    ax_l.set_title(f"PC1 loadings {title_suffix} ({pca_obj.explained_variance_ratio_[0]*100:.1f}% variance)",
                   fontweight="bold", fontsize=12)
    ax_l.legend(fontsize=9, loc="upper right", ncol=5)

    # Accessibility
    acc_file = RESULTS_DIR / "accessibility_100kb.npz"
    if acc_file.exists():
        acc = np.load(acc_file)
        for arm, mb0, mb1, si, nb in arm_bounds:
            if f"{arm}_frac" in acc:
                frac, amids = acc[f"{arm}_frac"], acc[f"{arm}_mids"]
                missing = 1.0 - frac
                c = 0
                for sa, so, sn, sm in segments:
                    if sa == arm:
                        break
                    c += sm[-1] + gap
                ax_a.fill_between(amids + c, 0, missing, color="#E53935", alpha=0.3)
                ax_a.plot(amids + c, missing, color="#E53935", lw=0.4, alpha=0.5)

    ax_a.set_ylim(0, 0.8)
    ax_a.invert_yaxis()
    ax_a.set_ylabel("Missing", fontsize=9, color="#E53935")
    ax_a.set_xlabel("Position (Mb)", fontsize=11)
    for sp in ax_a.spines.values():
        sp.set_color("#ddd")

    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{fname}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{fname}.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Gather populations with data
    pops = [p for p in POPULATIONS if (RESULTS_DIR / p / "3L").exists()]
    print(f"Populations with data: {len(pops)}")

    # =====================================================================
    # Neutral arms (3L + 3R) feature matrix
    # =====================================================================
    mat_neutral, seg_neutral = build_multi_arm_matrix(pops, ["3L", "3R"])
    print(f"Neutral feature matrix: {mat_neutral.shape}")

    pca = PCA(n_components=min(3, len(pops)-1))
    scores = pca.fit_transform(mat_neutral)

    # --- PCA scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax_i, (px, py, t) in enumerate([(0, 1, "PC1 vs PC2"), (0, 2, "PC1 vs PC3")]):
        ax = axes[ax_i]
        if py >= scores.shape[1]:
            ax.set_visible(False); continue
        for i, pop in enumerate(pops):
            c = REGION_CLR.get(REGION.get(pop, ""), "#888")
            ax.scatter(scores[i, px], scores[i, py], c=c, s=100, zorder=3, edgecolors="white", lw=0.8)
            ax.annotate(pop.replace("_", " "), (scores[i, px], scores[i, py]),
                        fontsize=7.5, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel(f"PC{px+1} ({pca.explained_variance_ratio_[px]*100:.1f}%)")
        ax.set_ylabel(f"PC{py+1} ({pca.explained_variance_ratio_[py]*100:.1f}%)")
        ax.set_title(t, fontweight="bold"); ax.legend(handles=HANDLES, fontsize=9)
    fig.suptitle(f"PCA on block-resolution TMRCA (3L + 3R, {mat_neutral.shape[1]} blocks)",
                 fontweight="bold", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_highres.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "pca_highres.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved pca_highres.png")

    # --- Loadings: neutral arms ---
    plot_loadings(pca, seg_neutral, "pca_loadings", "(neutral: 3L + 3R)")

    # =====================================================================
    # All 5 arms feature matrix + loadings
    # =====================================================================
    mat_all, seg_all = build_multi_arm_matrix(pops, ["2L", "2R", "3L", "3R", "X"])
    print(f"All-arms feature matrix: {mat_all.shape}")

    pca_all = PCA(n_components=min(3, len(pops)-1))
    pca_all.fit(mat_all)
    plot_loadings(pca_all, seg_all, "pca_loadings_allarms", "(all 5 chromosome arms)")

    # =====================================================================
    # UMAP
    # =====================================================================
    try:
        import umap
        reducer = umap.UMAP(n_neighbors=min(5, len(pops)-1), min_dist=0.3,
                            random_state=42, metric="euclidean")
        emb = reducer.fit_transform(mat_neutral)
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, pop in enumerate(pops):
            c = REGION_CLR.get(REGION.get(pop, ""), "#888")
            ax.scatter(emb[i, 0], emb[i, 1], c=c, s=120, zorder=3, edgecolors="white", lw=0.8)
            ax.annotate(pop.replace("_", " "), (emb[i, 0], emb[i, 1]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_title(f"UMAP of block-resolution TMRCA (3L + 3R, {mat_neutral.shape[1]} blocks)", fontweight="bold")
        ax.legend(handles=HANDLES, fontsize=9)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "umap_highres.png", dpi=150, bbox_inches="tight")
        fig.savefig(FIG_DIR / "umap_highres.pdf", bbox_inches="tight")
        plt.close()
        print("  Saved umap_highres.png")
    except ImportError:
        print("  UMAP skipped (pip install umap-learn)")

    # =====================================================================
    # Per-pair PCA on 3L
    # =====================================================================
    pair_profiles, pair_pops_list = [], []
    for pop in pops:
        pm, ws = reconstruct_profile(pop, "3L", "all")
        if pm is None:
            continue
        nb = pm.shape[1] // 500
        for pi in range(pm.shape[0]):
            bv = np.array([np.median(pm[pi, b*500:(b+1)*500]) for b in range(nb)])
            pair_profiles.append(bv)
            pair_pops_list.append(pop)

    pair_mat = np.array(pair_profiles)
    print(f"Per-pair matrix: {pair_mat.shape}")
    pca_p = PCA(n_components=2)
    ps = pca_p.fit_transform(pair_mat)

    fig, ax = plt.subplots(figsize=(10, 8))
    for pop in pops:
        mask = np.array([p == pop for p in pair_pops_list])
        c = REGION_CLR.get(REGION.get(pop, ""), "#888")
        ax.scatter(ps[mask, 0], ps[mask, 1], c=c, s=15, alpha=0.5,
                   edgecolors="none", label=pop.replace("_", " "))
    ax.set_xlabel(f"PC1 ({pca_p.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca_p.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"Per-pair PCA on chr3L ({len(pair_profiles)} pairs)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2, markerscale=3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_perpair.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "pca_perpair.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved pca_perpair.png")


if __name__ == "__main__":
    main()
