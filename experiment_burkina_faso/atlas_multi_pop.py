#!/usr/bin/env python3
"""Build a multi-population TimeAtlas on 3L (no inversions) and compare.

Demonstrates cross-population queries using the TimeAtlas API with
real inference results from Burkina Faso and Cameroon.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastcxt.atlas import TimeAtlas

RESULTS_DIR = Path("/tmp/het_results")
WINDOW_SIZE_BP = 200
MUTATION_RATE = 3.5e-9
ARM = "3L"
GROUP = "all"
PAIR_TYPE = "intra"


def load_and_reshape(pop, arm, group, pair_type, pair_offset=0):
    """Load results and reshape. pair_offset shifts pair IDs for multi-pop atlas."""
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
    wpb = means.shape[1]
    n_total = n_blocks * wpb

    pair_means = np.zeros((n_pairs, n_total), dtype=np.float32)
    pair_vars = np.zeros((n_pairs, n_total), dtype=np.float32)

    for row_idx in range(means.shape[0]):
        bi = index_map[row_idx, 0]
        pi = index_map[row_idx, 1]
        ws = bi * wpb
        pair_means[pi, ws:ws + wpb] = means[row_idx]
        pair_vars[pi, ws:ws + wpb] = variances[row_idx]

    pairs = np.column_stack([
        np.arange(n_pairs) + pair_offset,
        np.arange(n_pairs) + pair_offset + 10000,  # second haplotype in different range
    ]).astype(np.int32)

    window_starts = np.zeros(n_total, dtype=np.int64)
    for b in blocks:
        for w in range(wpb):
            window_starts[b["idx"] * wpb + w] = b["start"] + w * WINDOW_SIZE_BP

    return pair_means, pair_vars, pairs, window_starts


def main():
    populations = ["Burkina_Faso", "Cameroon"]
    pop_data = {}

    for i, pop in enumerate(populations):
        print(f"Loading {pop} {ARM} {GROUP}_{PAIR_TYPE}...", end=" ")
        m, v, p, ws = load_and_reshape(pop, ARM, GROUP, PAIR_TYPE, pair_offset=i * 1000)
        if m is None:
            print("SKIPPED")
            continue
        pop_data[pop] = (m, v, p, ws)
        print(f"{m.shape[0]} pairs x {m.shape[1]} windows")

    # Build per-population atlases for comparison
    print("\n" + "=" * 70)
    print("Per-Population Comparison on 3L (no inversions)")
    print("=" * 70)

    for pop, (m, v, p, ws) in pop_data.items():
        atlas = TimeAtlas()
        atlas.add_arm(ARM, m, v, p, ws, window_size=WINDOW_SIZE_BP, mutation_rate=MUTATION_RATE)

        mean_tmrca = atlas.mean_tmrca(ARM)
        print(f"\n{pop}:")
        print(f"  Pairs: {len(mean_tmrca)}")
        print(f"  Genome-wide mean TMRCA: {mean_tmrca.mean():.0f} +/- {mean_tmrca.std():.0f} gen")
        print(f"  Range: {mean_tmrca.min():.0f} - {mean_tmrca.max():.0f} gen")

        # Distribution of TMRCA at a few positions
        for pos_mb in [5, 15, 30]:
            pos = int(pos_mb * 1e6)
            result = atlas.query_window(ARM, pos)
            if result:
                _, m_w, _ = result
                print(f"  At {pos_mb} Mb: mean={np.exp(m_w).mean():.0f}, "
                      f"std={np.exp(m_w).std():.0f} gen")

    # IICR comparison (demographic inference proxy)
    print("\n" + "=" * 70)
    print("IICR (Ne proxy) Comparison at Different Time Depths")
    print("=" * 70)

    for pop, (m, v, p, ws) in pop_data.items():
        # Flatten all means across pairs
        all_tmrca = np.exp(m).flatten()
        # Compute IICR at different quantiles as a simple Ne proxy
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        print(f"\n{pop} TMRCA distribution:")
        for q in quantiles:
            val = np.quantile(all_tmrca, q)
            print(f"  {q*100:.0f}th percentile: {val:.0f} generations")

    # Heterogeneity along the chromosome
    print("\n" + "=" * 70)
    print("TMRCA Heterogeneity Along 3L (10 Mb bins)")
    print("=" * 70)

    for pop, (m, v, p, ws) in pop_data.items():
        n_windows = m.shape[1]
        bin_size = n_windows // 4  # ~4 bins across the arm
        print(f"\n{pop}:")
        for bi in range(4):
            start_w = bi * bin_size
            end_w = min((bi + 1) * bin_size, n_windows)
            region_mean = np.exp(m[:, start_w:end_w]).mean()
            region_std = np.exp(m[:, start_w:end_w]).std()
            start_mb = ws[start_w] / 1e6
            end_mb = ws[min(end_w - 1, n_windows - 1)] / 1e6
            print(f"  {start_mb:.1f}-{end_mb:.1f} Mb: "
                  f"mean={region_mean:.0f}, std={region_std:.0f} gen")

    print("\nDone!")


if __name__ == "__main__":
    main()
