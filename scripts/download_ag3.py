#!/usr/bin/env python3
"""Download Ag3 (Anopheles gambiae) data from MalariaGEN for fastcxt.

Downloads haplotype data, accessibility masks, and sample metadata
from the MalariaGEN Ag3 release via their public GCS-backed API.

Data is saved as compressed zarr arrays per chromosome arm, plus
metadata as a parquet file.

Usage:
    # Download all arms (default: gamb_colu analysis, Ag3.0 samples)
    python scripts/download_ag3.py --out-dir /sietch_colab/kkor/fastcxt/data/ag3

    # Download a single arm
    python scripts/download_ag3.py --out-dir ./data/ag3 --arms 2L

    # Filter to a specific country
    python scripts/download_ag3.py --out-dir ./data/ag3 --sample-query "country == 'Burkina Faso'"

    # Dry run (show sizes, don't download)
    python scripts/download_ag3.py --out-dir ./data/ag3 --dry-run
"""

import argparse
import sys
from pathlib import Path

import numpy as np

CHROMOSOME_ARMS = ["2L", "2R", "3L", "3R", "X"]
DEFAULT_ANALYSIS = "gamb_colu"
DEFAULT_SITE_MASK = "gamb_colu"
DEFAULT_SAMPLE_SETS = "3.0"


def get_ag3():
    """Initialize the Ag3 API client."""
    import malariagen_data
    return malariagen_data.Ag3()


def download_metadata(ag3, out_dir: Path, sample_sets: str, sample_query: str | None):
    """Download and save sample metadata."""
    print("Downloading sample metadata...")
    df = ag3.sample_metadata(sample_sets=sample_sets)
    if sample_query:
        df = df.query(sample_query)

    meta_path = out_dir / "sample_metadata.parquet"
    df.to_parquet(meta_path)
    print(f"  Saved {len(df)} samples to {meta_path}")

    # Also save a human-readable summary
    summary_path = out_dir / "sample_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Total samples: {len(df)}\n\n")
        f.write(f"Species:\n{df['taxon'].value_counts().to_string()}\n\n")
        f.write(f"Countries:\n{df['country'].value_counts().to_string()}\n\n")
    print(f"  Summary written to {summary_path}")

    return df


def download_accessibility(ag3, out_dir: Path, arms: list[str], site_mask: str):
    """Download and save accessibility masks per arm."""
    mask_dir = out_dir / "accessibility"
    mask_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    for arm in arms:
        print(f"  Downloading accessibility mask for {arm}...")
        acc = ag3.is_accessible(region=arm, site_mask=site_mask)
        acc_np = np.array(acc)
        arrays[f"is_accessible_{arm}"] = acc_np
        n_accessible = acc_np.sum()
        n_total = len(acc_np)
        print(f"    {arm}: {n_accessible:,}/{n_total:,} accessible sites ({100*n_accessible/n_total:.1f}%)")

    mask_path = out_dir / "accessibility_mask.npz"
    np.savez_compressed(mask_path, **arrays)
    size_mb = mask_path.stat().st_size / 1e6
    print(f"  Saved all masks to {mask_path} ({size_mb:.1f} MB)")


def download_haplotypes(
    ag3, out_dir: Path, arms: list[str],
    analysis: str, sample_sets: str,
    sample_query: str | None,
):
    """Download phased haplotype data per chromosome arm as compressed npz."""
    import zarr

    hap_dir = out_dir / "haplotypes"
    hap_dir.mkdir(parents=True, exist_ok=True)

    for arm in arms:
        print(f"\nDownloading haplotypes for {arm} (analysis={analysis})...")
        ds = ag3.haplotypes(
            region=arm,
            analysis=analysis,
            sample_sets=sample_sets,
            sample_query=sample_query,
        )

        # Get dimensions
        n_variants = ds.dims["variants"]
        n_samples = ds.dims["samples"]
        n_haplotypes = n_samples * 2  # diploid
        print(f"  {arm}: {n_variants:,} variants × {n_haplotypes:,} haplotypes")

        # Download positions (small)
        positions = ds["variant_position"].values

        # Download genotypes in chunks to manage memory
        print(f"  Downloading genotype data (this may take a while)...")
        gt = ds["call_genotype"].values  # (variants, samples, ploidy)

        # Reshape to (n_haplotypes, n_variants) for fastcxt
        # gt shape: (variants, samples, 2) -> we want (samples*2, variants)
        haplotypes = gt.reshape(n_variants, -1).T.astype(np.int8)

        # Save as compressed npz
        out_path = hap_dir / f"{arm}.npz"
        np.savez_compressed(
            out_path,
            haplotypes=haplotypes,   # (n_haplotypes, n_variants)
            positions=positions,      # (n_variants,)
        )
        size_mb = out_path.stat().st_size / 1e6
        print(f"  Saved {out_path} ({size_mb:.1f} MB)")

        # Free memory
        del gt, haplotypes, positions


def dry_run(ag3, arms: list[str], analysis: str, sample_sets: str,
            sample_query: str | None, site_mask: str):
    """Estimate download sizes without downloading."""
    print("=== DRY RUN ===\n")

    # Metadata
    df = ag3.sample_metadata(sample_sets=sample_sets)
    if sample_query:
        df = df.query(sample_query)
    n_samples = len(df)
    print(f"Samples: {n_samples} ({n_samples * 2} haplotypes)")
    print(f"  Species: {dict(df['taxon'].value_counts())}")
    print()

    total_est_gb = 0
    for arm in arms:
        # Get variant count from haplotype sites
        ds = ag3.haplotypes(
            region=arm, analysis=analysis,
            sample_sets=sample_sets, sample_query=sample_query,
        )
        n_variants = ds.dims["variants"]
        n_haplotypes = n_samples * 2

        raw_gb = n_variants * n_haplotypes / 1e9
        compressed_est_gb = raw_gb * 0.05  # ~20x compression typical for sparse genotypes
        total_est_gb += compressed_est_gb

        print(f"  {arm}: {n_variants:,} variants × {n_haplotypes:,} haplotypes")
        print(f"       Raw: {raw_gb:.1f} GB, Compressed estimate: {compressed_est_gb:.1f} GB")

    print(f"\nEstimated total download: ~{total_est_gb:.1f} GB compressed")
    print("(Actual size depends on compression ratio of genotype data)")


def main():
    parser = argparse.ArgumentParser(description="Download Ag3 data for fastcxt")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for downloaded data")
    parser.add_argument("--arms", nargs="+", default=CHROMOSOME_ARMS,
                        choices=CHROMOSOME_ARMS,
                        help="Chromosome arms to download (default: all)")
    parser.add_argument("--analysis", default=DEFAULT_ANALYSIS,
                        help=f"Phasing analysis (default: {DEFAULT_ANALYSIS})")
    parser.add_argument("--site-mask", default=DEFAULT_SITE_MASK,
                        help=f"Site mask for accessibility (default: {DEFAULT_SITE_MASK})")
    parser.add_argument("--sample-sets", default=DEFAULT_SAMPLE_SETS,
                        help=f"Sample set release (default: {DEFAULT_SAMPLE_SETS})")
    parser.add_argument("--sample-query", default=None,
                        help="Pandas query to filter samples (e.g. \"country == 'Burkina Faso'\")")
    parser.add_argument("--dry-run", action="store_true",
                        help="Estimate sizes without downloading")
    parser.add_argument("--skip-haplotypes", action="store_true",
                        help="Only download metadata and accessibility masks")
    args = parser.parse_args()

    ag3 = get_ag3()

    if args.dry_run:
        dry_run(ag3, args.arms, args.analysis, args.sample_sets,
                args.sample_query, args.site_mask)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Metadata
    download_metadata(ag3, args.out_dir, args.sample_sets, args.sample_query)

    # 2. Accessibility masks
    download_accessibility(ag3, args.out_dir, args.arms, args.site_mask)

    # 3. Haplotype data
    if not args.skip_haplotypes:
        download_haplotypes(ag3, args.out_dir, args.arms, args.analysis,
                            args.sample_sets, args.sample_query)

    print("\n=== Download complete ===")
    print(f"Data saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
