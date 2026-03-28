#!/usr/bin/env python3
"""Run tsinfer + tsdate on focal regions and compare with fastcxt predictions.

Focal regions:
  1. Vgsc/kdr: 2L:1.9-2.9 Mb (pyrethroid resistance sweep)
  2. Rdl:      2L:24.9-25.9 Mb (dieldrin resistance)
  3. Neutral:  3L:20-21 Mb (no known selection targets)

Populations: Burkina Faso, Cameroon, Kenya (one per geographic region)

Usage (on poppy):
    # First install latest tsinfer + tsdate:
    pixi run pip install --upgrade tsinfer tsdate

    # Then run:
    pixi run python experiment_burkina_faso/validation/run_tsdate_comparison.py \
        --vcf-dir /sietch_colab/data_share/Ag1000G/Ag3.0/vcf/phased_vcf/gamb \
        --meta /sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/gamb.meta.tsinfer.csv \
        --accessibility /sietch_colab/data_share/Ag1000G/Ag3.0/vcf/agp3.is_accessible.txt.npz \
        --fastcxt-results /sietch_colab/kkor/fastcxt/results/het_allpop \
        --out-dir /sietch_colab/kkor/fastcxt/results/tsdate_validation
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")

# ---------------------------------------------------------------------------
# Focal regions: (name, arm, start_bp, end_bp)
# ---------------------------------------------------------------------------
FOCAL_REGIONS = [
    ("Vgsc_kdr", "2L", 1_900_000, 2_900_000),
    ("Rdl",      "2L", 24_900_000, 25_900_000),
    ("Neutral",  "3L", 20_000_000, 21_000_000),
]

# Populations to validate (one per region)
VALIDATION_POPS = ["Burkina Faso", "Cameroon", "Kenya"]

MUTATION_RATE = 3.5e-9
RECOMBINATION_RATE = 1e-8
BLOCK_SIZE = 100_000  # 100 kb blocks matching fastcxt
MAX_SAMPLES = 40  # cap diploid individuals for tractability


def load_vcf_region(vcf_path, arm, start, end, sample_ids=None):
    """Load genotypes + positions from a VCF for a genomic region using cyvcf2."""
    try:
        from cyvcf2 import VCF
    except ImportError:
        log.error("cyvcf2 not installed. Run: pixi run pip install cyvcf2")
        raise

    vcf = VCF(str(vcf_path))
    if sample_ids is not None:
        vcf.set_samples(sample_ids)

    positions = []
    genotypes = []
    alleles = []

    region = f"{arm}:{start}-{end}"
    for variant in vcf(region):
        if variant.is_snp and len(variant.ALT) == 1:
            gt = variant.genotype.array()[:, :2]  # (n_samples, 2)
            if np.any(gt < 0):
                continue  # skip sites with missing data
            positions.append(variant.POS)
            genotypes.append(gt.flatten())  # haploid view
            alleles.append((variant.REF, variant.ALT[0]))

    if not positions:
        return None, None, None

    positions = np.array(positions)
    genotypes = np.array(genotypes, dtype=np.int8)  # (n_sites, n_haploids)
    return positions, genotypes, alleles


def run_tsinfer_tsdate(positions, genotypes, alleles, sequence_length,
                       mutation_rate=MUTATION_RATE, recombination_rate=RECOMBINATION_RATE):
    """Run tsinfer + tsdate on genotype data. Returns dated tree sequence."""
    import tsinfer
    import tsdate

    n_sites, n_haploids = genotypes.shape
    log.info(f"  tsinfer: {n_sites} sites, {n_haploids} haploids, {sequence_length/1e6:.1f} Mb")

    # Build tsinfer SampleData
    with tsinfer.SampleData(sequence_length=sequence_length) as sd:
        for i in range(n_sites):
            sd.add_site(
                position=positions[i],
                genotypes=genotypes[i],
                alleles=alleles[i],
            )

    # Infer tree sequence
    t0 = time.time()
    ts_inferred = tsinfer.infer(sd)
    log.info(f"  tsinfer done in {time.time()-t0:.1f}s ({ts_inferred.num_trees} trees)")

    # Date with tsdate
    t0 = time.time()
    ts_dated = tsdate.date(
        ts_inferred,
        mutation_rate=mutation_rate,
        method="variational_gamma",
    )
    log.info(f"  tsdate done in {time.time()-t0:.1f}s")

    return ts_dated


def extract_tsdate_tmrca(ts, pairs, block_starts, block_size=BLOCK_SIZE):
    """Extract per-block mean TMRCA from a dated tree sequence for given pairs.

    Returns (n_blocks, n_pairs) array of log(TMRCA).
    """
    n_blocks = len(block_starts)
    n_pairs = len(pairs)
    tmrca = np.full((n_blocks, n_pairs), np.nan)

    for bi, bs in enumerate(block_starts):
        block_mid = bs + block_size // 2
        try:
            tree = ts.at(block_mid)
        except Exception:
            continue

        for pi, (sa, sb) in enumerate(pairs):
            try:
                mrca = tree.mrca(sa, sb)
                if mrca != -1:
                    t = tree.time(mrca)
                    if t > 0:
                        tmrca[bi, pi] = np.log(t)
            except Exception:
                continue

    return tmrca


def load_fastcxt_predictions(results_dir, pop, arm, group, start_bp, end_bp):
    """Load fastcxt block-level predictions for a region."""
    d = results_dir / pop.replace(" ", "_") / arm / f"{group}_intra"
    if not (d / "means.npz").exists():
        return None, None, None

    means = np.load(d / "means.npz")["means"]
    imap = np.load(d / "index_map.npy")
    blocks = json.load(open(d / "blocks.json"))
    cfg = json.load(open(d / "config.json"))

    n_pairs = cfg["n_pairs"]
    # Get per-block mean across windows and pairs
    block_means = {}
    block_pair_data = {}  # block_idx -> list of per-pair means

    for r in range(imap.shape[0]):
        bi, pi = imap[r]
        b = blocks[bi]
        if b["start"] < start_bp or b["end"] > end_bp:
            continue
        if bi not in block_pair_data:
            block_pair_data[bi] = []
        block_pair_data[bi].append(np.mean(means[r]))

    if not block_pair_data:
        return None, None, None

    sorted_blocks = sorted(block_pair_data.keys())
    block_starts = np.array([blocks[bi]["start"] for bi in sorted_blocks])
    # Mean across pairs per block
    fastcxt_means = np.array([np.mean(block_pair_data[bi]) for bi in sorted_blocks])
    # All pair values per block for correlation
    fastcxt_all = np.array([block_pair_data[bi] for bi in sorted_blocks])

    return fastcxt_means, block_starts, fastcxt_all


def main():
    parser = argparse.ArgumentParser(description="tsinfer+tsdate validation on focal regions")
    parser.add_argument("--vcf-dir", type=Path, required=True,
                        help="Dir with gamb.{arm}.phased.n1470.derived.vcf.gz")
    parser.add_argument("--meta", type=Path, required=True,
                        help="gamb.meta.tsinfer.csv")
    parser.add_argument("--accessibility", type=Path, default=None,
                        help="agp3.is_accessible.txt.npz (optional)")
    parser.add_argument("--fastcxt-results", type=Path, required=True,
                        help="fastcxt results dir (het_allpop/)")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta = pd.read_csv(args.meta)
    log.info(f"Metadata: {len(meta)} samples")

    results = []

    for region_name, arm, start_bp, end_bp in FOCAL_REGIONS:
        log.info(f"\n{'='*60}")
        log.info(f"Region: {region_name} ({arm}:{start_bp/1e6:.1f}-{end_bp/1e6:.1f} Mb)")
        log.info(f"{'='*60}")

        vcf_path = args.vcf_dir / f"gamb.{arm}.phased.n1470.derived.vcf.gz"
        if not vcf_path.exists():
            log.warning(f"VCF not found: {vcf_path}")
            continue

        for pop in VALIDATION_POPS:
            log.info(f"\nPopulation: {pop}")

            # Get sample IDs for this population
            pop_meta = meta[meta["country"] == pop]
            if len(pop_meta) == 0:
                # Try without exact match
                pop_meta = meta[meta["country"].str.contains(pop, case=False, na=False)]
            if len(pop_meta) == 0:
                log.warning(f"  No samples for {pop}")
                continue

            # Cap samples
            if len(pop_meta) > MAX_SAMPLES:
                pop_meta = pop_meta.sample(MAX_SAMPLES, random_state=42)
            sample_ids = pop_meta["sample_id"].tolist()
            log.info(f"  {len(sample_ids)} samples")

            # Load VCF region
            try:
                positions, genotypes, alleles = load_vcf_region(
                    vcf_path, arm, start_bp, end_bp, sample_ids
                )
            except Exception as e:
                log.error(f"  VCF loading failed: {e}")
                continue

            if positions is None or len(positions) == 0:
                log.warning(f"  No variants in region")
                continue

            log.info(f"  {len(positions)} variants loaded")

            # Run tsinfer + tsdate
            try:
                ts_dated = run_tsinfer_tsdate(
                    positions, genotypes, alleles,
                    sequence_length=end_bp,
                    mutation_rate=MUTATION_RATE,
                )
            except Exception as e:
                log.error(f"  tsinfer/tsdate failed: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Extract TMRCA from tsdate
            block_starts = np.arange(start_bp, end_bp, BLOCK_SIZE)
            # Use intra-individual pairs (haplotype 0 vs 1 for each diploid)
            n_haploids = ts_dated.num_samples
            pairs = [(i*2, i*2+1) for i in range(n_haploids // 2)]

            tsdate_tmrca = extract_tsdate_tmrca(ts_dated, pairs, block_starts)
            tsdate_block_means = np.nanmean(tsdate_tmrca, axis=1)
            log.info(f"  tsdate TMRCA: {tsdate_tmrca.shape}, "
                     f"mean={np.nanmean(tsdate_block_means):.2f}")

            # Load fastcxt predictions
            group = "2La_hom_standard" if arm == "2L" else "all"
            fcxt_means, fcxt_starts, _ = load_fastcxt_predictions(
                args.fastcxt_results, pop, arm, group, start_bp, end_bp
            )

            if fcxt_means is not None:
                log.info(f"  fastcxt TMRCA: {len(fcxt_means)} blocks, "
                         f"mean={np.mean(fcxt_means):.2f}")

                # Align blocks
                common_starts = np.intersect1d(block_starts, fcxt_starts)
                if len(common_starts) > 0:
                    ts_idx = np.array([np.where(block_starts == s)[0][0] for s in common_starts])
                    fc_idx = np.array([np.where(fcxt_starts == s)[0][0] for s in common_starts])

                    ts_vals = tsdate_block_means[ts_idx]
                    fc_vals = fcxt_means[fc_idx]

                    # Remove NaN
                    valid = np.isfinite(ts_vals) & np.isfinite(fc_vals)
                    if valid.sum() > 2:
                        from scipy.stats import pearsonr
                        r, p = pearsonr(ts_vals[valid], fc_vals[valid])
                        log.info(f"  Correlation: r={r:.3f}, p={p:.1e}, n={valid.sum()} blocks")

                        results.append({
                            "region": region_name, "arm": arm, "pop": pop,
                            "n_blocks": int(valid.sum()), "r": float(r), "p": float(p),
                            "tsdate_mean": float(np.mean(ts_vals[valid])),
                            "fastcxt_mean": float(np.mean(fc_vals[valid])),
                        })

            # Save tsdate results
            out_file = args.out_dir / f"tsdate_{region_name}_{pop.replace(' ', '_')}_{arm}.npz"
            np.savez_compressed(out_file,
                                block_starts=block_starts,
                                tsdate_block_means=tsdate_block_means,
                                tsdate_tmrca=tsdate_tmrca)
            log.info(f"  Saved {out_file.name}")

    # Save summary
    if results:
        summary_file = args.out_dir / "validation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)
        log.info(f"\nSummary saved to {summary_file}")

        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"{'Region':<15} {'Pop':<20} {'n_blocks':>8} {'r':>8} {'p':>10}")
        print("-" * 65)
        for r in results:
            print(f"{r['region']:<15} {r['pop']:<20} {r['n_blocks']:>8} "
                  f"{r['r']:>8.3f} {r['p']:>10.1e}")


if __name__ == "__main__":
    main()
