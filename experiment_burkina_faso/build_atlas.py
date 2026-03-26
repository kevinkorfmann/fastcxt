#!/usr/bin/env python3
"""Build a TimeAtlas from the inference results and take it for a spin.

Loads the Burkina Faso intra-individual results, reconstructs per-pair
genome-wide TMRCA profiles, builds a TimeAtlas, saves it, and runs
some example queries to demonstrate the API.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastcxt.atlas import TimeAtlas

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("/tmp/het_results")
POPULATION = "Burkina_Faso"
PAIR_TYPE = "intra"  # use intra for within-individual pairs
ATLAS_OUT = Path("/tmp/het_results/atlas_burkina_faso")

ARMS = ["2L", "2R", "3L", "3R", "X"]
ARM_GROUPS = {
    "2L": "2La_hom_standard",
    "2R": "2Rb_hom_standard",
    "3L": "all",
    "3R": "all",
    "X":  "all",
}

WINDOW_SIZE_BP = 200  # each of the 500 windows in a block = 100kb/500
MUTATION_RATE = 3.5e-9


def load_and_reshape(pop, arm, group, pair_type):
    """Load results and reshape from (n_blocks*n_pairs, 500) to (n_pairs, n_total_windows)."""
    d = RESULTS_DIR / pop / arm / f"{group}_{pair_type}"
    if not d.exists():
        return None, None, None, None

    means = np.load(d / "means.npz")["means"]
    variances = np.load(d / "variances.npz")["variances"]
    index_map = np.load(d / "index_map.npy")
    blocks = json.load(open(d / "blocks.json"))
    config = json.load(open(d / "config.json"))

    n_pairs = config["n_pairs"]
    n_blocks = len(blocks)
    windows_per_block = means.shape[1]  # 500

    # Reshape: rows are ordered by (block_idx, pair_idx)
    # Reconstruct (n_pairs, n_blocks * windows_per_block) by gathering blocks per pair
    n_total_windows = n_blocks * windows_per_block
    pair_means = np.zeros((n_pairs, n_total_windows), dtype=np.float32)
    pair_vars = np.zeros((n_pairs, n_total_windows), dtype=np.float32)

    for row_idx in range(means.shape[0]):
        block_idx = index_map[row_idx, 0]
        pair_idx = index_map[row_idx, 1]
        w_start = block_idx * windows_per_block
        w_end = w_start + windows_per_block
        pair_means[pair_idx, w_start:w_end] = means[row_idx]
        pair_vars[pair_idx, w_start:w_end] = variances[row_idx]

    # Build pairs array: for intra, pairs are (2*i, 2*i+1) haplotypes
    pairs = np.array([[i, i] for i in range(n_pairs)], dtype=np.int32)
    # Actually, these are haplotype pairs within individuals -- use pair index as ID
    pairs = np.column_stack([np.arange(n_pairs), np.arange(n_pairs) + n_pairs]).astype(np.int32)

    # Build window starts
    window_starts = np.zeros(n_total_windows, dtype=np.int64)
    for b in blocks:
        bidx = b["idx"]
        for w in range(windows_per_block):
            window_starts[bidx * windows_per_block + w] = b["start"] + w * WINDOW_SIZE_BP

    return pair_means, pair_vars, pairs, window_starts


def main():
    atlas = TimeAtlas()
    atlas.metadata = {
        "population": POPULATION,
        "pair_type": PAIR_TYPE,
        "source": "Ag1000G Ag3.0 tsinfer+tsdate",
        "description": "Intra-individual TMRCA atlas for Burkina Faso hom-standard karyotype",
    }

    for arm in ARMS:
        group = ARM_GROUPS[arm]
        print(f"Loading {arm} ({group}_{PAIR_TYPE})...", end=" ")
        pair_means, pair_vars, pairs, window_starts = load_and_reshape(
            POPULATION, arm, group, PAIR_TYPE
        )
        if pair_means is None:
            print("SKIPPED (no data)")
            continue

        atlas.add_arm(
            arm,
            means=pair_means,
            variances=pair_vars,
            pairs=pairs,
            block_starts=window_starts,
            window_size=WINDOW_SIZE_BP,
            mutation_rate=MUTATION_RATE,
        )
        print(f"{pair_means.shape[0]} pairs x {pair_means.shape[1]} windows")

    # Save
    print(f"\nSaving atlas to {ATLAS_OUT}...")
    atlas.save(ATLAS_OUT)
    print(f"Saved: {atlas}")

    # --- Take it for a spin ---
    print("\n" + "=" * 70)
    print("TimeAtlas Demo Queries")
    print("=" * 70)

    # Summary
    summary = atlas.summary()
    print(f"\nSummary:")
    print(f"  Arms: {summary['arms']}")
    print(f"  Total pairs: {summary['total_pairs']}")
    print(f"  Total windows: {summary['total_windows']}")
    for arm, info in summary["per_arm"].items():
        print(f"  {arm}: {info['n_pairs']} pairs, {info['n_windows']} windows, "
              f"arm_length={info['arm_length_bp']/1e6:.1f} Mb, "
              f"mean_log_tmrca={info['mean_log_tmrca']:.2f}")

    # Query a single pair across 2L
    print(f"\nQuery pair (0, {pairs[0][1]}) on 2L:")
    result = atlas.query_pair("2L", 0, pairs[0][1])
    if result:
        m, v = result
        print(f"  Profile length: {len(m)} windows")
        print(f"  Mean log(TMRCA): {m.mean():.3f} (= {np.exp(m.mean()):.0f} generations)")
        print(f"  Min log(TMRCA): {m.min():.3f} at window {m.argmin()}")
        print(f"  Max log(TMRCA): {m.max():.3f} at window {m.argmax()}")

    # Query In(2L)a inversion region (20.5-42.2 Mb)
    print(f"\nQuery In(2L)a inversion region (20.5-42.2 Mb):")
    result = atlas.query_region("2L", 20_500_000, 42_200_000)
    if result:
        pairs_q, m_region, v_region = result
        mean_inside = np.exp(m_region).mean()
        print(f"  Pairs: {len(pairs_q)}, Windows in region: {m_region.shape[1]}")
        print(f"  Mean TMRCA inside inversion: {mean_inside:.0f} generations")

    # Compare inside vs outside inversion
    result_outside_left = atlas.query_region("2L", 0, 20_000_000)
    result_outside_right = atlas.query_region("2L", 43_000_000, 49_000_000)
    if result_outside_left and result_outside_right:
        _, m_left, _ = result_outside_left
        _, m_right, _ = result_outside_right
        m_outside = np.concatenate([m_left, m_right], axis=1)
        mean_outside = np.exp(m_outside).mean()
        print(f"  Mean TMRCA outside inversion: {mean_outside:.0f} generations")
        print(f"  Ratio inside/outside: {mean_inside/mean_outside:.2f}x")

    # Deepest pairs at a known sweep locus: Rdl gene (~28.5 Mb on 2L)
    RDL_POS = 28_500_000
    print(f"\nDeepest 5 pairs at Rdl locus ({RDL_POS/1e6:.1f} Mb on 2L):")
    deep = atlas.deepest_pairs("2L", RDL_POS, k=5)
    for i, (a, b) in enumerate(deep):
        result = atlas.query_pair("2L", int(a), int(b))
        if result:
            w = atlas.arms["2L"].window_at(RDL_POS)
            print(f"  #{i+1}: pair ({a}, {b}), TMRCA = {np.exp(result[0][w]):.0f} gen")

    # Shallowest (most recent coalescence) at Rdl
    print(f"\nShallowest 5 pairs at Rdl locus:")
    shallow = atlas.shallowest_pairs("2L", RDL_POS, k=5)
    for i, (a, b) in enumerate(shallow):
        result = atlas.query_pair("2L", int(a), int(b))
        if result:
            w = atlas.arms["2L"].window_at(RDL_POS)
            print(f"  #{i+1}: pair ({a}, {b}), TMRCA = {np.exp(result[0][w]):.0f} gen")

    # Window query: all pairwise TMRCAs at the vgsc (kdr) locus (~2.4 Mb on 2L)
    VGSC_POS = 2_400_000
    print(f"\nAll pairwise TMRCAs at vgsc/kdr locus ({VGSC_POS/1e6:.1f} Mb on 2L):")
    result = atlas.query_window("2L", VGSC_POS)
    if result:
        pairs_w, m_w, v_w = result
        print(f"  {len(pairs_w)} pairs")
        print(f"  TMRCA range: {np.exp(m_w.min()):.0f} - {np.exp(m_w.max()):.0f} generations")
        print(f"  Mean TMRCA: {np.exp(m_w).mean():.0f} generations")
        print(f"  Std TMRCA: {np.exp(m_w).std():.0f} generations")

    # Per-pair genome-wide mean TMRCA
    print(f"\nPer-pair genome-wide mean TMRCA (natural scale) on 3L:")
    mean_tmrca_3l = atlas.mean_tmrca("3L")
    print(f"  {len(mean_tmrca_3l)} pairs")
    print(f"  Range: {mean_tmrca_3l.min():.0f} - {mean_tmrca_3l.max():.0f} generations")
    print(f"  Mean: {mean_tmrca_3l.mean():.0f} generations")
    print(f"  Std: {mean_tmrca_3l.std():.0f} generations")

    # Compare arms
    print(f"\nCross-arm comparison (genome-wide mean TMRCA):")
    for arm in ARMS:
        if arm in atlas.arms:
            mt = atlas.mean_tmrca(arm)
            print(f"  {arm}: {mt.mean():.0f} +/- {mt.std():.0f} generations")

    # Reload and verify
    print(f"\nReloading atlas from disk...")
    atlas2 = TimeAtlas.load(ATLAS_OUT)
    print(f"  Loaded: {atlas2}")
    assert atlas2.total_pairs == atlas.total_pairs
    assert atlas2.total_windows == atlas.total_windows
    print("  Round-trip verified!")

    print("\nDone!")


if __name__ == "__main__":
    main()
