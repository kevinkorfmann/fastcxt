#!/usr/bin/env python3
"""Polarize Ag3 genotype data using outgroup-inferred ancestral alleles.

Reads phased genotype data from the Ag3 zarr store on sietch, applies
ancestral allele annotations (AA field) to polarize genotypes so that
0 = ancestral, 1 = derived — matching msprime/fastcxt convention.

The AA field in the zarr was computed by Scott Small's pipeline using
est-sfs with 3 outgroup species (An. arabiensis, An. merus, An. melas).

Usage:
    # Polarize all chromosome arms
    python scripts/polarize_ag3.py --out-dir /sietch_colab/kkor/fastcxt/data/ag3_polarized

    # Single arm
    python scripts/polarize_ag3.py --out-dir ./data/ag3_polarized --arms 2L

    # Only high-confidence sites (AAProb >= 0.95)
    python scripts/polarize_ag3.py --out-dir ./data/ag3_polarized --min-aa-prob 0.95

    # Test mode: first 100kb of 2L only, compare against tsinfer trees
    python scripts/polarize_ag3.py --out-dir ./data/ag3_test --arms 2L --test-region 0:100000
"""

import argparse
import sys
from pathlib import Path

import numpy as np

CHROMOSOME_ARMS = ["2L", "2R", "3L", "3R", "X"]

# Default paths on sietch
DEFAULT_ZARR = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/AgamP3.phased.zarr"
DEFAULT_METADATA = "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/gamb.meta.tsinfer.csv"
DEFAULT_ACCESSIBILITY = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/agp3.is_accessible.txt.npz"


def polarize_arm(
    zarr_path: str,
    arm: str,
    out_dir: Path,
    min_aa_prob: float = 0.0,
    region: tuple[int, int] | None = None,
):
    """Polarize genotypes for one chromosome arm.

    Reads the zarr store, filters to biallelic SNPs with confident
    ancestral allele calls, flips genotypes where REF != ancestral,
    and saves as compressed npz.

    Returns dict with summary stats.
    """
    import zarr

    z = zarr.open(f"{zarr_path}/{arm}", "r")
    n_sites_total = z["variants/POS"].shape[0]

    # Determine slice range — only load positions first to find bounds
    if region is not None:
        start, end = region
        pos_all = z["variants/POS"][:]
        i_start = int(np.searchsorted(pos_all, start))
        i_end = int(np.searchsorted(pos_all, end))
        sl = slice(i_start, i_end)
        del pos_all
    else:
        sl = slice(None)

    # Load only the region slice
    pos = z["variants/POS"][sl]
    ref = z["variants/REF"][sl]
    alt_all = z["variants/ALT"][sl]  # (n_sites, 3)
    aa = z["variants/AA"][sl]
    aa_prob = z["variants/AAProb"][sl, 0]
    aa_cond = z["variants/AAcond"][sl]
    is_snp = z["variants/is_snp"][sl]
    n_slice = len(pos)

    # Filter: biallelic SNPs only
    mask_snp = is_snp

    # Filter: AA must match either REF or ALT[0]
    alt = alt_all[:, 0]
    aa_is_ref = aa == ref
    aa_is_alt = aa == alt
    mask_aa_valid = aa_is_ref | aa_is_alt

    # Filter: confidence threshold
    mask_prob = np.ones(n_slice, dtype=bool)
    if min_aa_prob > 0:
        valid_prob = ~np.isnan(aa_prob)
        mask_prob = valid_prob & (aa_prob >= min_aa_prob)

    # Combined filter
    keep = mask_snp & mask_aa_valid & mask_prob
    n_keep = keep.sum()

    print(f"  {arm}: {n_sites_total:,} total sites, {n_slice:,} in region")
    print(f"    Biallelic SNP: {mask_snp.sum():,}")
    print(f"    AA valid (REF or ALT): {(mask_snp & mask_aa_valid).sum():,}")
    print(f"    AA prob >= {min_aa_prob}: {n_keep:,}")

    if n_keep == 0:
        print(f"    WARNING: No sites passed filters for {arm}")
        return {"arm": arm, "n_sites": 0}

    # Identify which sites need flipping (AA == ALT, not REF)
    needs_flip = aa_is_alt[keep]
    n_flip = needs_flip.sum()
    print(f"    Sites needing flip (AA=ALT): {n_flip:,} ({100 * n_flip / n_keep:.1f}%)")

    # Load genotypes only for kept sites within the slice
    print(f"    Loading genotypes...")
    kept_local = np.where(keep)[0]
    # Convert local indices back to global zarr indices
    if region is not None:
        kept_indices = kept_local + i_start
    else:
        kept_indices = kept_local

    n_samples = z["calldata/GT"].shape[1]
    n_haplotypes = n_samples * 2
    haplotypes = np.empty((n_haplotypes, n_keep), dtype=np.int8)

    chunk_size = 50_000
    for chunk_start in range(0, n_keep, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_keep)
        idx = kept_indices[chunk_start:chunk_end]
        gt_chunk = z["calldata/GT"].get_orthogonal_selection((idx, slice(None), slice(None)))
        gt_flat = gt_chunk.reshape(len(idx), -1).T.astype(np.int8)
        haplotypes[:, chunk_start:chunk_end] = gt_flat

    # Flip genotypes where AA == ALT (so 0 becomes 1 and vice versa)
    if n_flip > 0:
        flip_cols = np.where(needs_flip)[0]
        haplotypes[:, flip_cols] = 1 - haplotypes[:, flip_cols]

    # Get positions and metadata for kept sites
    positions = pos[keep]
    aa_kept = aa[keep]
    aa_prob_kept = aa_prob[keep]
    aa_cond_kept = aa_cond[keep]

    # Save
    out_path = out_dir / f"{arm}.npz"
    np.savez_compressed(
        out_path,
        haplotypes=haplotypes,      # (n_haplotypes, n_sites), 0=ancestral, 1=derived
        positions=positions,         # (n_sites,)
        aa_prob=aa_prob_kept,        # (n_sites,) confidence
        aa_cond=aa_cond_kept,        # (n_sites,) condition string
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"    Saved {out_path} ({size_mb:.1f} MB)")
    print(f"    Shape: {haplotypes.shape} (haplotypes × sites)")

    return {
        "arm": arm,
        "n_sites": n_keep,
        "n_flip": n_flip,
        "n_haplotypes": n_haplotypes,
        "size_mb": size_mb,
    }


def save_metadata(zarr_path: str, metadata_csv: str, out_dir: Path):
    """Save sample metadata alongside the genotype data."""
    import pandas as pd

    if Path(metadata_csv).exists():
        df = pd.read_csv(metadata_csv)
        meta_path = out_dir / "sample_metadata.csv"
        df.to_csv(meta_path, index=False)
        print(f"Saved metadata ({len(df)} samples) to {meta_path}")
    else:
        # Extract sample IDs from zarr
        import zarr
        arm = "2L"
        z = zarr.open(f"{zarr_path}/{arm}", "r")
        samples = z["samples"][:]
        meta_path = out_dir / "samples.txt"
        with open(meta_path, "w") as f:
            for s in samples:
                f.write(f"{s}\n")
        print(f"Saved {len(samples)} sample IDs to {meta_path}")


def save_accessibility(mask_path: str, out_dir: Path, arms: list[str]):
    """Copy accessibility mask if it exists."""
    src = Path(mask_path)
    if src.exists():
        import shutil
        dst = out_dir / "accessibility_mask.npz"
        shutil.copy2(src, dst)
        print(f"Copied accessibility mask to {dst}")
    else:
        print(f"  Accessibility mask not found at {mask_path}, skipping")


DEFAULT_POLARIZED_VCF_DIR = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/phased_vcf/gamb"


def validate_against_vcf(
    out_dir: Path,
    arm: str = "2L",
    region: tuple[int, int] = (0, 100_000),
    vcf_dir: str = DEFAULT_POLARIZED_VCF_DIR,
):
    """Compare polarized output against existing polarized VCF.

    Uses bcftools to quickly extract the AA field and derived allele
    frequencies from the existing VCF for the same region.
    """
    vcf_path = f"{vcf_dir}/gamb.{arm}.phased.n1470.derived.vcf.gz"
    print(f"\n=== Validation: comparing against {Path(vcf_path).name} ({arm}:{region[0]}-{region[1]}) ===")

    if not Path(vcf_path).exists():
        print(f"  VCF not found at {vcf_path}, skipping")
        return

    # Load our polarized data
    data = np.load(out_dir / f"{arm}.npz", allow_pickle=True)
    our_pos = data["positions"]
    our_hap = data["haplotypes"]
    our_prob = data["aa_prob"]

    # Extract AA and AF from the existing polarized VCF using pysam
    import pysam

    vcf_pos = []
    vcf_derived_freq = []
    vcf = pysam.VariantFile(vcf_path)
    for rec in vcf.fetch(arm, region[0], region[1]):
        vcf_pos.append(rec.pos)
        # Compute derived freq from genotypes directly (not AF field).
        # In the "derived" VCF, AA field tells us the ancestral allele.
        # GT 0=REF, 1=ALT. If AA==REF, then ALT=derived, so derived=GT.
        # If AA==ALT, then REF=derived, so derived=1-GT.
        aa_vcf = rec.info.get("AA", ".")
        gts = []
        for s in rec.samples.values():
            gts.extend(s["GT"])
        gts = np.array([g for g in gts if g is not None], dtype=np.int8)
        if aa_vcf == rec.ref:
            # ALT is derived, GT=1 means derived
            vcf_derived_freq.append(gts.mean())
        else:
            # REF is derived, GT=0 means derived
            vcf_derived_freq.append(1.0 - gts.mean())
    vcf.close()

    vcf_pos = np.array(vcf_pos)
    vcf_derived_freq = np.array(vcf_derived_freq)

    # Find overlapping positions
    common_pos = np.intersect1d(our_pos, vcf_pos)
    print(f"  Our sites: {len(our_pos)}")
    print(f"  VCF sites: {len(vcf_pos)}")
    print(f"  Overlapping: {len(common_pos)}")

    if len(common_pos) == 0:
        print("  No overlapping sites")
        return

    # Get derived allele freq from our data and VCF at common positions
    our_idx = np.searchsorted(our_pos, common_pos)
    vcf_idx = np.searchsorted(vcf_pos, common_pos)

    our_freq = our_hap[:, our_idx].mean(axis=0)
    ref_freq = vcf_derived_freq[vcf_idx]

    # Compare
    freq_diff = np.abs(our_freq - ref_freq)
    concordant = freq_diff < 0.01
    flipped = np.abs(our_freq - (1 - ref_freq)) < 0.01

    print(f"  Compared {len(common_pos)} sites:")
    print(f"    Concordant (freq diff < 1%): {concordant.sum()} ({100 * concordant.mean():.1f}%)")
    print(f"    Opposite polarization: {flipped.sum()} ({100 * flipped.mean():.1f}%)")
    print(f"    Mean |freq diff|: {freq_diff.mean():.4f}")
    print(f"    Median |freq diff|: {np.median(freq_diff):.4f}")

    # High-confidence subset
    our_prob_common = our_prob[our_idx]
    high_conf = our_prob_common >= 0.95
    if high_conf.sum() > 0:
        print(f"  High-confidence (AAProb >= 0.95): {high_conf.sum()}")
        print(f"    Concordant: {concordant[high_conf].sum()} ({100 * concordant[high_conf].mean():.1f}%)")
        print(f"    Mean |freq diff|: {freq_diff[high_conf].mean():.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Polarize Ag3 genotype data using ancestral allele annotations"
    )
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory")
    parser.add_argument("--zarr", default=DEFAULT_ZARR,
                        help=f"Path to Ag3 phased zarr store (default: {DEFAULT_ZARR})")
    parser.add_argument("--arms", nargs="+", default=CHROMOSOME_ARMS,
                        choices=CHROMOSOME_ARMS, help="Chromosome arms")
    parser.add_argument("--min-aa-prob", type=float, default=0.0,
                        help="Minimum ancestral allele probability (0-1)")
    parser.add_argument("--metadata", default=DEFAULT_METADATA,
                        help="Path to sample metadata CSV")
    parser.add_argument("--accessibility", default=DEFAULT_ACCESSIBILITY,
                        help="Path to accessibility mask npz")
    parser.add_argument("--vcf-dir", default=DEFAULT_POLARIZED_VCF_DIR,
                        help="Path to existing polarized VCFs (for validation)")
    parser.add_argument("--test-region", default=None,
                        help="Test region as start:end (e.g., 0:100000)")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation against existing polarized VCF")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Parse test region
    region = None
    if args.test_region:
        start, end = args.test_region.split(":")
        region = (int(start), int(end))
        print(f"Test mode: region {region[0]:,}-{region[1]:,}")
        args.validate = True  # auto-enable validation in test mode

    # Save metadata
    save_metadata(args.zarr, args.metadata, args.out_dir)

    # Save accessibility mask
    save_accessibility(args.accessibility, args.out_dir, args.arms)

    # Polarize each arm
    results = []
    for arm in args.arms:
        print(f"\nPolarizing {arm}...")
        stats = polarize_arm(
            args.zarr, arm, args.out_dir,
            min_aa_prob=args.min_aa_prob,
            region=region,
        )
        results.append(stats)

    # Validation
    if args.validate:
        test_region = region or (0, 100_000)
        for arm in args.arms:
            validate_against_vcf(
                args.out_dir,
                arm=arm, region=test_region,
                vcf_dir=args.vcf_dir,
            )

    # Summary
    print("\n=== Summary ===")
    total_sites = 0
    total_mb = 0
    for r in results:
        if r["n_sites"] > 0:
            print(f"  {r['arm']}: {r['n_sites']:,} sites, "
                  f"{r['n_flip']:,} flipped, "
                  f"{r['n_haplotypes']:,} haplotypes, "
                  f"{r['size_mb']:.1f} MB")
            total_sites += r["n_sites"]
            total_mb += r["size_mb"]
    print(f"  Total: {total_sites:,} sites, {total_mb:.1f} MB")
    print(f"  Output: {args.out_dir}")


if __name__ == "__main__":
    main()
