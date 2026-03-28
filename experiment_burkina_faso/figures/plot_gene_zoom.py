#!/usr/bin/env python3
"""Zoom into gene regions and plot TMRCA profiles across populations."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

RESULTS_DIR = Path("/tmp/het_results")
FIG_DIR = Path(__file__).parent / "gene_zoom"
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

ARM_GROUPS = {"2L": "2La_hom_standard", "2R": "2Rb_hom_standard",
              "3L": "all", "3R": "all", "X": "all"}

# Genes to zoom into: (symbol, arm, center_bp, flank_kb, description)
ZOOM_GENES = [
    ("para/Vgsc", "2L", 2_395_000, 500, "Voltage-gated Na+ channel (pyrethroid resistance)"),
    ("Rdl",       "2L", 25_400_000, 500, "GABA receptor (dieldrin resistance)"),
    ("dpr2",      "3R", 362_000, 500, "Defective proboscis extension response 2"),
    ("Tep1",      "3L", 3_600_000, 500, "Thioester-containing protein 1 (immunity)"),
    ("CuSOD3",    "3L", 1_920_000, 500, "Copper-zinc superoxide dismutase 3"),
    ("CYP9K1",    "X",  15_241_000, 500, "Cytochrome P450 (metabolic resistance)"),
]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#fafafa",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
})


def load_window_profile(pop, arm, group, start_bp, end_bp):
    """Load TMRCA at window resolution for a genomic region."""
    d = RESULTS_DIR / pop / arm / f"{group}_intra"
    if not (d / "means.npz").exists():
        return None, None
    means = np.load(d / "means.npz")["means"]
    imap = np.load(d / "index_map.npy")
    blocks = json.load(open(d / "blocks.json"))
    cfg = json.load(open(d / "config.json"))
    n_pairs, wpb = cfg["n_pairs"], means.shape[1]

    # Find blocks overlapping our region
    block_size = 100_000
    profiles = []  # per-pair profiles in the region
    positions = []

    for b in blocks:
        b_start, b_end = b["start"], b["end"]
        if b_end < start_bp or b_start > end_bp:
            continue
        # Get all pairs' data for this block
        block_rows = []
        for r in range(imap.shape[0]):
            if imap[r, 0] == b["idx"]:
                block_rows.append(means[r])
        if not block_rows:
            continue
        block_data = np.array(block_rows)  # (n_pairs_in_block, 500)
        # Window positions within this block
        win_pos = np.array([b_start + w * 200 for w in range(wpb)])
        # Mask to our region
        mask = (win_pos >= start_bp) & (win_pos <= end_bp)
        if mask.sum() == 0:
            continue
        profiles.append(block_data[:, mask])
        positions.append(win_pos[mask])

    if not profiles:
        return None, None

    # Concatenate across blocks
    all_prof = np.concatenate(profiles, axis=1)  # (n_pairs, n_windows_in_region)
    all_pos = np.concatenate(positions) / 1e6  # Mb

    # Sort by position
    order = np.argsort(all_pos)
    return all_prof[:, order], all_pos[order]


def main():
    genes_data = json.load(open(RESULTS_DIR / "genes.json")) if (RESULTS_DIR / "genes.json").exists() else None

    # =====================================================================
    # Multi-panel figure: one row per gene
    # =====================================================================
    n_genes = len(ZOOM_GENES)
    fig, axes = plt.subplots(n_genes, 1, figsize=(18, n_genes * 3.5),
                              sharex=False)

    for gi, (gene_name, arm, center_bp, flank_kb, desc) in enumerate(ZOOM_GENES):
        ax = axes[gi]
        start_bp = center_bp - flank_kb * 1000
        end_bp = center_bp + flank_kb * 1000
        group = ARM_GROUPS[arm]

        any_data = False
        for pop in POPULATIONS:
            prof, pos_mb = load_window_profile(pop, arm, group, start_bp, end_bp)
            if prof is None:
                continue
            any_data = True

            # Median across pairs, light smoothing
            med = np.median(np.exp(prof), axis=0)
            if len(med) > 10:
                med = gaussian_filter1d(med, sigma=3)

            c = REGION_CLR.get(REGION.get(pop, ""), "#888")
            label = pop.replace("_", " ")
            ax.plot(pos_mb, med, color=c, lw=1.0, alpha=0.7, label=label)

        if not any_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14, color="#bbb")
            continue

        # Mark gene position
        ax.axvline(center_bp / 1e6, color="#333", lw=1, ls=":", alpha=0.5)

        # Draw gene body if we can find it
        if genes_data and arm in genes_data:
            for g in genes_data[arm]:
                gs, ge = g["start"] / 1e6, g["end"] / 1e6
                if ge < start_bp / 1e6 or gs > end_bp / 1e6:
                    continue
                sym = g.get("symbol", "").strip()
                if sym:
                    ax.axvspan(gs, ge, alpha=0.08, color="#333", zorder=0)
                    ax.text((gs + ge) / 2, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0,
                            sym, fontsize=7, ha="center", va="bottom", color="#666",
                            fontstyle="italic")

        ax.set_ylabel("TMRCA (gen)", fontsize=10)
        ax.set_title(f"{gene_name} — {arm}:{start_bp/1e6:.1f}-{end_bp/1e6:.1f} Mb — {desc}",
                     fontsize=11, fontweight="bold")

        # Legend only on first panel
        if gi == 0:
            from matplotlib.lines import Line2D
            handles = [Line2D([0], [0], color=REGION_CLR[r], lw=2, label=f"{r} Africa")
                       for r in ["West", "Central", "East"]]
            ax.legend(handles=handles, fontsize=8, loc="upper right")

    axes[-1].set_xlabel("Position (Mb)", fontsize=11)
    fig.suptitle("TMRCA profiles at candidate gene regions across populations",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "gene_zoom_multi.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / "gene_zoom_multi.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved gene_zoom_multi.png ({n_genes} panels)")

    # =====================================================================
    # Individual gene panels (larger, one per file)
    # =====================================================================
    for gene_name, arm, center_bp, flank_kb, desc in ZOOM_GENES:
        start_bp = center_bp - flank_kb * 1000
        end_bp = center_bp + flank_kb * 1000
        group = ARM_GROUPS[arm]

        fig, (ax_main, ax_var) = plt.subplots(2, 1, figsize=(14, 6),
            height_ratios=[3, 1], sharex=True, gridspec_kw={"hspace": 0.08})

        pop_profiles = {}
        for pop in POPULATIONS:
            prof, pos_mb = load_window_profile(pop, arm, group, start_bp, end_bp)
            if prof is None:
                continue
            pop_profiles[pop] = (prof, pos_mb)

        if not pop_profiles:
            plt.close()
            continue

        # Main panel: median per population
        for pop, (prof, pos_mb) in pop_profiles.items():
            med = np.median(np.exp(prof), axis=0)
            q25, q75 = np.percentile(np.exp(prof), [25, 75], axis=0)
            if len(med) > 10:
                med = gaussian_filter1d(med, 3)
                q25 = gaussian_filter1d(q25, 3)
                q75 = gaussian_filter1d(q75, 3)

            c = REGION_CLR.get(REGION.get(pop, ""), "#888")
            ax_main.fill_between(pos_mb, q25, q75, color=c, alpha=0.05)
            ax_main.plot(pos_mb, med, color=c, lw=1.2, alpha=0.8,
                         label=pop.replace("_", " "))

        ax_main.axvline(center_bp / 1e6, color="#dc2626", lw=1.5, ls="--", alpha=0.5)
        ax_main.set_ylabel("TMRCA (generations)")
        ax_main.set_title(f"{gene_name} ({arm}) — {desc}", fontweight="bold", fontsize=12)
        ax_main.legend(fontsize=7, ncol=3, loc="best")

        # Bottom panel: coefficient of variation across populations
        # Shows where populations diverge most
        ref_pos = list(pop_profiles.values())[0][1]
        all_meds = []
        for pop, (prof, pos_mb) in pop_profiles.items():
            med = np.median(np.exp(prof), axis=0)
            if len(med) > 10:
                med = gaussian_filter1d(med, 3)
            all_meds.append(med)
        all_meds = np.array(all_meds)
        cv = np.std(all_meds, axis=0) / (np.mean(all_meds, axis=0) + 1)
        ax_var.fill_between(ref_pos, cv, color="#8b5cf6", alpha=0.3)
        ax_var.plot(ref_pos, cv, color="#8b5cf6", lw=0.8)
        ax_var.axvline(center_bp / 1e6, color="#dc2626", lw=1.5, ls="--", alpha=0.5)
        ax_var.set_ylabel("CV across\npopulations", fontsize=9)
        ax_var.set_xlabel("Position (Mb)")

        # Gene annotations
        if genes_data and arm in genes_data:
            for g in genes_data[arm]:
                gs, ge = g["start"] / 1e6, g["end"] / 1e6
                if ge < start_bp / 1e6 or gs > end_bp / 1e6:
                    continue
                sym = g.get("symbol", "").strip()
                if sym:
                    ax_main.axvspan(gs, ge, alpha=0.06, color="#333")
                    ylim = ax_main.get_ylim()
                    ax_main.text((gs + ge) / 2, ylim[1] * 0.98, sym,
                                 fontsize=8, ha="center", va="top", color="#333",
                                 fontstyle="italic",
                                 bbox=dict(fc="white", ec="none", alpha=0.7, pad=1))

        fig.tight_layout()
        safe_name = gene_name.replace("/", "_")
        fig.savefig(FIG_DIR / f"zoom_{safe_name}_{arm}.png", dpi=150, bbox_inches="tight")
        fig.savefig(FIG_DIR / f"zoom_{safe_name}_{arm}.pdf", bbox_inches="tight")
        plt.close()
        print(f"  Saved zoom_{safe_name}_{arm}.png")


if __name__ == "__main__":
    main()
