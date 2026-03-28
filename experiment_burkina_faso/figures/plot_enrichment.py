#!/usr/bin/env python3
"""Permutation test for sweep enrichment at resistance loci +
GO-like functional enrichment of PC1 loading peaks."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/tmp/het_results")
FIG_DIR = Path(__file__).parent / "enrichment"
FIG_DIR.mkdir(exist_ok=True)

POPULATIONS = [
    "Burkina_Faso", "Cameroon", "Central_African_Republic",
    "Democratic_Republic_of_the_Congo", "Equatorial_Guinea", "Gabon",
    "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Kenya", "Mali",
]

ARMS = ["2L", "2R", "3L", "3R", "X"]
ARM_GROUPS = {
    "2L": ["2La_hom_standard"], "2R": ["2Rb_hom_standard"],
    "3L": ["all"], "3R": ["all"], "X": ["all"],
}

# Known insecticide resistance gene regions (arm, start_bp, end_bp, name)
RESISTANCE_REGIONS = [
    ("2L", 2_300_000, 2_500_000, "Vgsc/kdr"),
    ("2L", 25_300_000, 25_500_000, "Rdl"),
    ("2R", 28_450_000, 28_550_000, "CYP6P4"),
    ("3R", 28_500_000, 28_700_000, "Gste2"),
    ("X", 15_200_000, 15_300_000, "CYP9K1"),
]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fafafa",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
})


def load_block_means(pop, arm, group):
    d = RESULTS_DIR / pop / arm / f"{group}_intra"
    if not (d / "means.npz").exists():
        return None, None
    means = np.load(d / "means.npz")["means"]
    imap = np.load(d / "index_map.npy")
    blocks = json.load(open(d / "blocks.json"))
    cfg = json.load(open(d / "config.json"))
    n_blocks = len(blocks)
    n_pairs = cfg["n_pairs"]
    n_win = means.shape[1]
    arr = np.full((n_blocks, n_pairs, n_win), np.nan)
    for i in range(len(imap)):
        arr[imap[i, 0], imap[i, 1], :] = means[i]
    bm = np.nanmean(arr, axis=(1, 2))  # (n_blocks,)
    mids_bp = np.array([(b["start"] + b["end"]) / 2 for b in blocks])
    return bm, mids_bp


# =========================================================================
# 1. Permutation test: are resistance loci enriched among TMRCA outliers?
# =========================================================================

def permutation_test():
    print("=" * 60)
    print("Permutation test: resistance loci enrichment in TMRCA outliers")
    print("=" * 60)

    n_perm = 10_000
    rng = np.random.RandomState(42)

    all_results = []

    for pop in POPULATIONS:
        pop_dir = RESULTS_DIR / pop
        if not pop_dir.exists():
            continue

        for arm in ARMS:
            for group in ARM_GROUPS[arm]:
                bm, mids_bp = load_block_means(pop, arm, group)
                if bm is None or len(bm) < 10:
                    continue

                # Define outlier blocks (bottom 5th percentile = young = sweep candidates)
                q05 = np.percentile(bm, 5)
                is_outlier = bm <= q05
                n_outlier = is_outlier.sum()
                if n_outlier == 0:
                    continue

                # Check how many outlier blocks overlap resistance regions
                resistance_mask = np.zeros(len(bm), dtype=bool)
                for r_arm, r_start, r_end, r_name in RESISTANCE_REGIONS:
                    if r_arm != arm:
                        continue
                    resistance_mask |= (mids_bp >= r_start) & (mids_bp <= r_end)

                n_resist_blocks = resistance_mask.sum()
                if n_resist_blocks == 0:
                    continue

                observed = (is_outlier & resistance_mask).sum()

                # Permutation: shuffle outlier labels
                perm_counts = np.zeros(n_perm)
                for p in range(n_perm):
                    perm_outlier = rng.permutation(is_outlier)
                    perm_counts[p] = (perm_outlier & resistance_mask).sum()

                p_value = (perm_counts >= observed).sum() / n_perm
                if observed > 0:
                    fold = observed / (n_outlier * n_resist_blocks / len(bm))
                else:
                    fold = 0

                all_results.append({
                    "pop": pop, "arm": arm, "group": group,
                    "n_blocks": len(bm), "n_outlier": n_outlier,
                    "n_resist": n_resist_blocks, "observed": observed,
                    "expected": n_outlier * n_resist_blocks / len(bm),
                    "fold": fold, "p_value": p_value,
                })

    # Print results
    sig_results = [r for r in all_results if r["observed"] > 0]
    print(f"\nTests with overlap (observed > 0): {len(sig_results)} / {len(all_results)}")
    print(f"\n{'Population':<30} {'Arm':>3} {'Obs':>4} {'Exp':>6} {'Fold':>6} {'p':>8}")
    print("-" * 65)
    for r in sorted(sig_results, key=lambda x: x["p_value"]):
        print(f"{r['pop'].replace('_',' '):<30} {r['arm']:>3} {r['observed']:>4} "
              f"{r['expected']:>6.2f} {r['fold']:>6.1f} {r['p_value']:>8.4f}")

    # --- Plot: permutation distribution for pooled test ---
    # Pool across all populations and arms
    total_observed = sum(r["observed"] for r in all_results)
    total_expected = sum(r["expected"] for r in all_results)

    # Pooled permutation
    pooled_perm = np.zeros(n_perm)
    for r in all_results:
        bm, mids_bp = load_block_means(r["pop"], r["arm"], r["group"])
        if bm is None:
            continue
        q05 = np.percentile(bm, 5)
        is_outlier = bm <= q05
        resistance_mask = np.zeros(len(bm), dtype=bool)
        for r_arm, r_start, r_end, r_name in RESISTANCE_REGIONS:
            if r_arm != r["arm"]:
                continue
            resistance_mask |= (mids_bp >= r_start) & (mids_bp <= r_end)
        if resistance_mask.sum() == 0:
            continue
        for p in range(n_perm):
            perm_outlier = rng.permutation(is_outlier)
            pooled_perm[p] += (perm_outlier & resistance_mask).sum()

    pooled_p = (pooled_perm >= total_observed).sum() / n_perm

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pooled_perm, bins=range(int(pooled_perm.max()) + 2), color="#93c5fd",
            edgecolor="white", lw=0.5, alpha=0.8, label="Permuted")
    ax.axvline(total_observed, color="#dc2626", lw=2, ls="--",
               label=f"Observed = {total_observed}")
    ax.axvline(total_expected, color="#666", lw=1, ls=":",
               label=f"Expected = {total_expected:.1f}")
    ax.set_xlabel("# outlier blocks overlapping resistance loci")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation test: resistance loci in TMRCA outliers\n"
                 f"(pooled across {len(all_results)} tests, {n_perm:,} permutations, p = {pooled_p:.4f})",
                 fontweight="bold")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "permutation_test.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "permutation_test.pdf", bbox_inches="tight")
    plt.close()
    print(f"\n  Pooled: observed={total_observed}, expected={total_expected:.1f}, p={pooled_p:.4f}")
    print(f"  Saved permutation_test.png")

    return all_results


# =========================================================================
# 2. Functional enrichment of PC1 loading peak genes
# =========================================================================

def functional_enrichment():
    print("\n" + "=" * 60)
    print("Functional enrichment of PC1 loading peaks")
    print("=" * 60)

    genes_data = json.load(open(RESULTS_DIR / "genes.json"))

    # Get block-level profiles and run PCA (same as plot_highres_structure.py)
    pops = [p for p in POPULATIONS if (RESULTS_DIR / p / "3L").exists()]
    arm_groups_map = {"2L": "2La_hom_standard", "2R": "2Rb_hom_standard",
                      "3L": "all", "3R": "all", "X": "all"}

    def get_pop_median(pop, arm, group):
        d = RESULTS_DIR / pop / arm / f"{group}_intra"
        if not (d / "means.npz").exists():
            return None, None
        m = np.load(d / "means.npz")["means"]
        im = np.load(d / "index_map.npy")
        bl = json.load(open(d / "blocks.json"))
        cfg = json.load(open(d / "config.json"))
        np_, nb, wpb = cfg["n_pairs"], len(bl), m.shape[1]
        nt = nb * wpb
        pm = np.zeros((np_, nt), np.float32)
        for r in range(m.shape[0]):
            bi, pi = im[r]
            s = bi * wpb
            pm[pi, s:s+wpb] = m[r]
        nblk = pm.shape[1] // 500
        bmed = np.zeros((pm.shape[0], nblk))
        for b in range(nblk):
            bmed[:, b] = np.median(pm[:, b*500:(b+1)*500], axis=1)
        ws = np.zeros(nt, np.int64)
        for b in bl:
            for w in range(wpb):
                ws[b["idx"]*wpb + w] = b["start"] + w * 200
        return np.median(bmed, axis=0), ws[::500][:nblk] / 1e6

    # Build all-arms matrix
    from sklearn.decomposition import PCA
    arm_data = {}
    for arm in ARMS:
        profs, mids = [], None
        for pop in pops:
            p, m = get_pop_median(pop, arm, arm_groups_map[arm])
            profs.append(p)
            if m is not None:
                mids = m
        if mids is not None:
            arm_data[arm] = (profs, mids)

    rows = []
    for pi in range(len(pops)):
        parts = []
        for arm in ARMS:
            if arm in arm_data:
                p = arm_data[arm][0][pi]
                parts.append(p if p is not None else np.full(len(arm_data[arm][1]), np.nan))
        rows.append(np.concatenate(parts))
    mat = np.array(rows)
    for j in range(mat.shape[1]):
        c = mat[:, j]
        m = np.isfinite(c)
        if m.any() and not m.all():
            mat[~m, j] = np.nanmean(c)

    pca = PCA(n_components=3)
    pca.fit(mat)
    loadings = np.abs(pca.components_[0])

    segments = []
    offset = 0
    for arm in ARMS:
        if arm in arm_data:
            n = len(arm_data[arm][1])
            segments.append((arm, offset, n, arm_data[arm][1]))
            offset += n

    # Top 100 loading peaks -> find overlapping genes
    peaks = []
    for idx in np.argsort(-loadings):
        if len(peaks) >= 100:
            break
        if all(abs(idx - p) > 2 for p in peaks):
            peaks.append(idx)

    # Map peaks to genes
    peak_genes = []
    for pi in peaks:
        for arm, si, nb, mids in segments:
            if si <= pi < si + nb:
                pos_bp = int(mids[pi - si] * 1e6)
                # Find all genes within 50 kb
                for g in genes_data[arm]:
                    d = abs((g["start"] + g["end"]) / 2 - pos_bp)
                    if d < 50_000:
                        peak_genes.append(g)
                break

    # Count description keywords across peak genes vs all genes
    def extract_keywords(desc):
        """Extract functional keywords from gene descriptions."""
        if not desc or desc == "unspecified product":
            return set()
        # Normalize
        desc = desc.lower().replace("%2c", ",").replace("%2C", ",")
        keywords = set()
        # Functional categories
        categories = {
            "receptor": "receptor",
            "kinase": "kinase",
            "protease": "protease",
            "transporter": "transporter",
            "channel": "ion channel",
            "oxidase": "oxidoreductase",
            "reductase": "oxidoreductase",
            "dehydrogenase": "oxidoreductase",
            "transferase": "transferase",
            "synthase": "synthase",
            "helicase": "helicase",
            "ribosom": "ribosomal",
            "heat shock": "heat shock/chaperone",
            "chaperone": "heat shock/chaperone",
            "dnaj": "heat shock/chaperone",
            "cuticular": "cuticle",
            "cytochrome": "cytochrome P450",
            "p450": "cytochrome P450",
            "odorant": "chemosensory",
            "gustatory": "chemosensory",
            "ionotropic": "chemosensory",
            "immune": "immunity",
            "thioester": "immunity",
            "defensin": "immunity",
            "lectin": "immunity",
            "toll": "immunity",
            "clip": "immunity",
            "serine protease": "immunity",
            "zinc finger": "transcription factor",
            "homeobox": "transcription factor",
            "transcription": "transcription factor",
            "adhesion": "cell adhesion",
            "integrin": "cell adhesion",
            "cadherin": "cell adhesion",
            "calcium": "calcium signaling",
            "calmodulin": "calcium signaling",
            "calumenin": "calcium signaling",
            "ubiquitin": "ubiquitin/proteasome",
            "proteasome": "ubiquitin/proteasome",
            "abc transporter": "ABC transporter",
            "glutathione": "detoxification",
            "superoxide": "oxidative stress",
            "peroxidase": "oxidative stress",
        }
        for keyword, category in categories.items():
            if keyword in desc:
                keywords.add(category)
        return keywords

    # Count categories in peak genes vs background
    peak_cats = {}
    for g in peak_genes:
        for cat in extract_keywords(g.get("description", "")):
            peak_cats[cat] = peak_cats.get(cat, 0) + 1

    all_genes = [g for arm in genes_data for g in genes_data[arm]]
    bg_cats = {}
    for g in all_genes:
        for cat in extract_keywords(g.get("description", "")):
            bg_cats[cat] = bg_cats.get(cat, 0) + 1

    # Compute enrichment (fold change) and Fisher's exact test
    from scipy.stats import fisher_exact

    n_peak = len(peak_genes)
    n_bg = len(all_genes)

    enrichment_results = []
    for cat in sorted(set(list(peak_cats.keys()) + list(bg_cats.keys()))):
        a = peak_cats.get(cat, 0)  # in peaks, in category
        b = n_peak - a             # in peaks, not in category
        c = bg_cats.get(cat, 0)    # not in peaks, in category
        d = n_bg - n_peak - c      # not in peaks, not in category
        if a == 0:
            continue
        odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        fold = (a / n_peak) / (c / n_bg) if c > 0 else float("inf")
        enrichment_results.append({
            "category": cat, "peak_count": a, "bg_count": c,
            "peak_frac": a / n_peak, "bg_frac": c / n_bg,
            "fold": fold, "p_value": p,
        })

    enrichment_results.sort(key=lambda x: x["p_value"])

    print(f"\nPeak genes: {n_peak}, Background genes: {n_bg}")
    print(f"\n{'Category':<25} {'Peak':>5} {'BG':>6} {'Fold':>6} {'p-value':>10}")
    print("-" * 60)
    for r in enrichment_results:
        sig = "*" if r["p_value"] < 0.05 else " "
        print(f"{r['category']:<25} {r['peak_count']:>5} {r['bg_count']:>6} "
              f"{r['fold']:>6.1f} {r['p_value']:>9.4f} {sig}")

    # --- Plot: enrichment barplot ---
    cats_to_plot = [r for r in enrichment_results if r["peak_count"] >= 2]
    if cats_to_plot:
        fig, ax = plt.subplots(figsize=(10, max(4, len(cats_to_plot) * 0.5)))
        cats = [r["category"] for r in cats_to_plot]
        folds = [r["fold"] for r in cats_to_plot]
        pvals = [r["p_value"] for r in cats_to_plot]
        colors = ["#2563eb" if p < 0.05 else "#93c5fd" if p < 0.1 else "#cbd5e1"
                  for p in pvals]

        y_pos = range(len(cats))
        bars = ax.barh(y_pos, folds, color=colors, edgecolor="white", lw=0.5, height=0.7)

        # Add count labels
        for i, (bar, r) in enumerate(zip(bars, cats_to_plot)):
            ax.text(bar.get_width() + 0.1, i, f"n={r['peak_count']} (p={r['p_value']:.3f})",
                    va="center", fontsize=8, color="#333")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(cats, fontsize=10)
        ax.set_xlabel("Fold enrichment (peak genes / background)")
        ax.axvline(1, color="#999", lw=0.8, ls="--")
        ax.set_title("Functional enrichment at PC1 loading peaks\n"
                     "(blue = p < 0.05, light blue = p < 0.1, gray = n.s.)",
                     fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "functional_enrichment.png", dpi=150, bbox_inches="tight")
        fig.savefig(FIG_DIR / "functional_enrichment.pdf", bbox_inches="tight")
        plt.close()
        print(f"\n  Saved functional_enrichment.png")

    # --- Uncharacterized gene count ---
    n_unspec = sum(1 for g in peak_genes
                   if g.get("description", "") == "unspecified product"
                   and not g.get("symbol", "").strip())
    n_known = n_peak - n_unspec
    print(f"\nPeak gene annotation status:")
    print(f"  Known function: {n_known} ({n_known/n_peak*100:.0f}%)")
    print(f"  Uncharacterized: {n_unspec} ({n_unspec/n_peak*100:.0f}%)")

    # Pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie([n_known, n_unspec],
           labels=[f"Annotated\n(n={n_known})", f"Uncharacterized\n(n={n_unspec})"],
           colors=["#2563eb", "#e5e7eb"],
           autopct="%1.0f%%", startangle=90,
           textprops={"fontsize": 12},
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Genes at PC1 loading peaks:\nannotated vs uncharacterized",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "peak_annotation_pie.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "peak_annotation_pie.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved peak_annotation_pie.png")


if __name__ == "__main__":
    permutation_test()
    functional_enrichment()
