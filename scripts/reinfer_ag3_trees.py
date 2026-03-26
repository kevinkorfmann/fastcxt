#!/usr/bin/env python3
"""Reinfer Ag3 tree sequences from polarized VCFs using tsinfer + tsdate.

The existing trees on sietch were inferred with tsinfer 0.2.3 using default
mismatch_ratio and tsdate mutation_rate=1e-9 (too low for AnoGam). This
script reinfers with proper parameters.

Prerequisites:
    pip install tsinfer tsdate pysam

Usage:
    # Reinfer a single arm from the existing polarized VCF
    python scripts/reinfer_ag3_trees.py --arm 2L --out-dir ./data/ag3_trees

    # All arms
    python scripts/reinfer_ag3_trees.py --out-dir ./data/ag3_trees

    # Custom mismatch ratio
    python scripts/reinfer_ag3_trees.py --arm 2L --out-dir ./data/ag3_trees --mismatch-ratio 1.0
"""

import argparse
from pathlib import Path

import numpy as np

ARMS = ["2L", "2R", "3L", "3R", "X"]

# Default paths on sietch
DEFAULT_VCF_DIR = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/phased_vcf/gamb"
DEFAULT_METADATA = "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/gamb.meta.tsinfer.csv"
DEFAULT_ACCESSIBILITY = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/agp3.is_accessible.txt.npz"

# AnoGam parameters
MUTATION_RATE = 3.5e-9
RECOMBINATION_RATE = 1e-8

# Chromosome arm lengths (AgamP4)
ARM_LENGTHS = {
    "2L": 49_364_325,
    "2R": 61_545_105,
    "3L": 41_963_435,
    "3R": 53_200_684,
    "X": 24_393_108,
}


def vcf_to_sample_data(vcf_path, arm, arm_length, accessibility_path=None):
    """Convert polarized VCF to tsinfer SampleData."""
    import tsinfer
    import pysam

    vcf = pysam.VariantFile(vcf_path)
    sample_names = list(vcf.header.samples)
    n_samples = len(sample_names)

    # Load accessibility mask if available
    is_accessible = None
    if accessibility_path and Path(accessibility_path).exists():
        masks = np.load(accessibility_path)
        for key in [f"is_accessible_{arm}", f"access_{arm}", arm]:
            if key in masks:
                is_accessible = masks[key]
                break

    with tsinfer.SampleData(
        sequence_length=arm_length,
        num_samples=n_samples * 2,  # diploid -> 2 haplotypes per sample
    ) as sd:
        # Add individuals
        for name in sample_names:
            sd.add_individual(ploidy=2)

        # Add sites from VCF
        n_sites = 0
        n_skipped = 0
        for rec in vcf.fetch(arm):
            # Skip non-biallelic or non-SNP
            if len(rec.alts) != 1 or len(rec.ref) != 1 or len(rec.alts[0]) != 1:
                n_skipped += 1
                continue

            # Get ancestral allele
            aa = rec.info.get("AA", None)
            if aa is None:
                n_skipped += 1
                continue

            # Determine ancestral/derived
            if aa == rec.ref:
                ancestral = rec.ref
                derived = rec.alts[0]
            elif aa == rec.alts[0]:
                ancestral = rec.alts[0]
                derived = rec.ref
            else:
                n_skipped += 1
                continue

            # Get genotypes (already phased)
            genotypes = np.zeros(n_samples * 2, dtype=np.int32)
            for j, s in enumerate(rec.samples.values()):
                gt = s["GT"]
                if gt[0] is not None:
                    # If AA == ALT, flip: REF(0) becomes derived(1), ALT(1) becomes ancestral(0)
                    if aa == rec.alts[0]:
                        genotypes[2 * j] = 1 - gt[0]
                        genotypes[2 * j + 1] = 1 - gt[1]
                    else:
                        genotypes[2 * j] = gt[0]
                        genotypes[2 * j + 1] = gt[1]

            sd.add_site(
                position=rec.pos,
                genotypes=genotypes,
                alleles=[ancestral, derived],
            )
            n_sites += 1

        vcf.close()
        print(f"  Added {n_sites:,} sites, skipped {n_skipped:,}")

    return sd


def reinfer_arm(
    arm, vcf_dir, out_dir, metadata_path, accessibility_path,
    mismatch_ratio=None, num_threads=8,
):
    """Reinfer tree sequence for one chromosome arm."""
    import tsinfer
    import tsdate

    vcf_path = f"{vcf_dir}/gamb.{arm}.phased.n1470.derived.vcf.gz"
    if not Path(vcf_path).exists():
        print(f"  VCF not found: {vcf_path}")
        return

    arm_length = ARM_LENGTHS[arm]
    print(f"\n{'='*60}")
    print(f"  {arm} ({arm_length/1e6:.1f} Mb)")
    print(f"{'='*60}")

    # Step 1: VCF -> SampleData
    print(f"  Step 1: Converting VCF to SampleData...")
    sd = vcf_to_sample_data(vcf_path, arm, arm_length, accessibility_path)
    sd_path = out_dir / f"{arm}.samples"
    sd.copy(sd_path)
    print(f"  Saved SampleData: {sd.num_sites:,} sites, {sd.num_samples} haplotypes")

    # Step 2: tsinfer
    print(f"  Step 2: Running tsinfer (mismatch_ratio={mismatch_ratio})...")
    sd = tsinfer.load(sd_path)

    infer_kwargs = dict(num_threads=num_threads, progress_monitor=True)
    if mismatch_ratio is not None:
        infer_kwargs["mismatch_ratio"] = mismatch_ratio
        infer_kwargs["recombination_rate"] = RECOMBINATION_RATE

    ts = tsinfer.infer(sd, **infer_kwargs)
    print(f"  Inferred: {ts.num_trees:,} trees, {ts.num_sites:,} sites")

    # Step 3: Simplify
    ts = ts.simplify()
    print(f"  Simplified: {ts.num_trees:,} trees")

    # Step 4: tsdate
    print(f"  Step 3: Running tsdate (mutation_rate={MUTATION_RATE})...")
    ts_dated = tsdate.date(
        ts,
        mutation_rate=MUTATION_RATE,
        progress=True,
        num_threads=num_threads,
    )

    # Save
    out_path = out_dir / f"gamb.{arm}.trees"
    ts_dated.dump(out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  Trees: {ts_dated.num_trees:,}, Sites: {ts_dated.num_sites:,}")

    # Add metadata from CSV if available
    if Path(metadata_path).exists():
        import pandas as pd
        meta = pd.read_csv(metadata_path)
        print(f"  Metadata: {len(meta)} individuals")

    return ts_dated


def main():
    parser = argparse.ArgumentParser(description="Reinfer Ag3 trees with tsinfer + tsdate")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arm", nargs="+", default=ARMS, choices=ARMS)
    parser.add_argument("--vcf-dir", default=DEFAULT_VCF_DIR)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--accessibility", default=DEFAULT_ACCESSIBILITY)
    parser.add_argument("--mismatch-ratio", type=float, default=None,
                        help="tsinfer mismatch ratio (default: tsinfer's default)")
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for arm in args.arm:
        reinfer_arm(
            arm, args.vcf_dir, args.out_dir,
            args.metadata, args.accessibility,
            mismatch_ratio=args.mismatch_ratio,
            num_threads=args.num_threads,
        )

    print(f"\nDone. Trees saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
