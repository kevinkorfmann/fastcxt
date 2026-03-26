#!/usr/bin/env python3
"""End-to-end pipeline: download AG3 data from MalariaGEN → polarize → fastcxt-ready output.

Downloads phased haplotypes for An. gambiae/coluzzii (focal) and An. arabiensis
(outgroup) from MalariaGEN's cloud-hosted AG3 release. Polarizes genotypes using
parsimony with the arabiensis outgroup so that 0=ancestral, 1=derived — matching
the msprime/fastcxt convention.

Prerequisites:
    pip install malariagen_data pysam
    gcloud auth application-default login   # one-time GCS authentication

Usage:
    # Full pipeline: download + polarize all arms
    python scripts/ag3_pipeline.py --out-dir /path/to/ag3_polarized

    # Single arm, high-confidence only
    python scripts/ag3_pipeline.py --out-dir ./data/ag3 --arms 2L --min-aa-prob 0.9

    # Subset to one country
    python scripts/ag3_pipeline.py --out-dir ./data/ag3 --sample-query "country == 'Burkina Faso'"

    # Skip download if zarr cache already exists
    python scripts/ag3_pipeline.py --out-dir ./data/ag3 --skip-download

    # Download only (no polarization)
    python scripts/ag3_pipeline.py --out-dir ./data/ag3 --download-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np

CHROMOSOME_ARMS = ["2L", "2R", "3L", "3R", "X"]
DEFAULT_SAMPLE_SETS = "3.0"
DEFAULT_ANALYSIS = "gamb_colu"
DEFAULT_SITE_MASK = "gamb_colu"
# Minimum outgroup allele frequency to call ancestral with confidence
DEFAULT_MIN_OUTGROUP_FREQ = 0.95


# ---------------------------------------------------------------------------
# Step 1: Download from MalariaGEN
# ---------------------------------------------------------------------------

def download_ag3(
    out_dir: Path,
    arms: list[str],
    sample_sets: str,
    analysis: str,
    site_mask: str,
    sample_query: str | None,
):
    """Download phased haplotypes + arabiensis outgroup from MalariaGEN AG3."""
    import malariagen_data

    ag3 = malariagen_data.Ag3()
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Sample metadata ---
    print("Downloading sample metadata...")
    df_all = ag3.sample_metadata(sample_sets=sample_sets)
    meta_path = out_dir / "sample_metadata_all.csv"
    df_all.to_csv(meta_path, index=False)

    # Focal species (gambiae + coluzzii)
    df_focal = df_all[df_all["taxon"].isin(["gambiae", "coluzzii"])]
    if sample_query:
        df_focal = df_focal.query(sample_query)
    focal_ids = set(df_focal["sample_id"])
    print(f"  Focal samples: {len(df_focal)} ({dict(df_focal['taxon'].value_counts())})")

    # Outgroup (arabiensis)
    df_arab = df_all[df_all["taxon"] == "arabiensis"]
    arab_ids = set(df_arab["sample_id"])
    print(f"  Outgroup (arabiensis): {len(df_arab)}")

    df_focal.to_csv(out_dir / "sample_metadata_focal.csv", index=False)
    df_arab.to_csv(out_dir / "sample_metadata_arabiensis.csv", index=False)

    # --- Accessibility mask ---
    print("Downloading accessibility masks...")
    mask_arrays = {}
    for arm in arms:
        acc = ag3.is_accessible(region=arm, site_mask=site_mask)
        mask_arrays[f"is_accessible_{arm}"] = np.array(acc)
        n_acc = mask_arrays[f"is_accessible_{arm}"].sum()
        n_tot = len(mask_arrays[f"is_accessible_{arm}"])
        print(f"  {arm}: {n_acc:,}/{n_tot:,} accessible ({100*n_acc/n_tot:.1f}%)")
    np.savez_compressed(out_dir / "accessibility_mask.npz", **mask_arrays)

    # --- Per-arm: focal haplotypes + arabiensis SNP calls ---
    for arm in arms:
        print(f"\nDownloading {arm}...")
        arm_cache = cache_dir / arm
        arm_cache.mkdir(exist_ok=True)

        # Focal haplotypes (phased)
        focal_cache = arm_cache / "focal_haplotypes.npz"
        if focal_cache.exists():
            print(f"  Focal haplotypes cached, skipping")
        else:
            print(f"  Focal haplotypes ({analysis} analysis)...")
            ds = ag3.haplotypes(
                region=arm,
                analysis=analysis,
                sample_sets=sample_sets,
                sample_query=sample_query,
            )
            focal_pos = ds["variant_position"].values
            focal_alleles = ds["variant_allele"].values  # (n_variants, n_alleles)
            focal_gt = ds["call_genotype"].values  # (n_variants, n_samples, 2)
            focal_samples = ds["sample_id"].values
            n_var, n_samp = focal_gt.shape[:2]
            print(f"    {n_var:,} variants × {n_samp} samples ({n_samp*2} haplotypes)")
            np.savez_compressed(
                focal_cache,
                positions=focal_pos,
                alleles=focal_alleles,
                genotypes=focal_gt.astype(np.int8),
                samples=focal_samples,
            )
            size_mb = focal_cache.stat().st_size / 1e6
            print(f"    Saved ({size_mb:.1f} MB)")

        # Arabiensis SNP calls (for outgroup polarization)
        arab_cache = arm_cache / "arab_snp_calls.npz"
        if arab_cache.exists():
            print(f"  Arabiensis SNP calls cached, skipping")
        else:
            print(f"  Arabiensis SNP calls...")
            ds_arab = ag3.snp_calls(
                region=arm,
                sample_sets=sample_sets,
                sample_query="taxon == 'arabiensis'",
                site_mask=site_mask,
            )
            arab_pos = ds_arab["variant_position"].values
            arab_alleles = ds_arab["variant_allele"].values
            arab_gt = ds_arab["call_genotype"].values.astype(np.int8)
            arab_samples = ds_arab["sample_id"].values
            print(f"    {len(arab_pos):,} variants × {len(arab_samples)} samples")
            np.savez_compressed(
                arab_cache,
                positions=arab_pos,
                alleles=arab_alleles,
                genotypes=arab_gt,
                samples=arab_samples,
            )
            size_mb = arab_cache.stat().st_size / 1e6
            print(f"    Saved ({size_mb:.1f} MB)")

    print("\nDownload complete.")


# ---------------------------------------------------------------------------
# Step 2: Polarize using arabiensis outgroup
# ---------------------------------------------------------------------------

def polarize_arm(
    cache_dir: Path,
    arm: str,
    out_dir: Path,
    min_outgroup_freq: float = DEFAULT_MIN_OUTGROUP_FREQ,
    min_aa_prob: float = 0.0,
    region: tuple[int, int] | None = None,
):
    """Polarize one chromosome arm using arabiensis as outgroup.

    For each biallelic SNP in the focal (gambiae/coluzzii) haplotypes:
    1. Look up the arabiensis allele frequency at the same position
    2. If arabiensis is fixed/nearly-fixed for one allele → that's ancestral
    3. Flip genotypes so 0=ancestral, 1=derived
    """
    arm_cache = cache_dir / arm

    # Load focal haplotypes
    print(f"  Loading focal haplotypes...")
    focal = np.load(arm_cache / "focal_haplotypes.npz", allow_pickle=True)
    focal_pos = focal["positions"]
    focal_alleles = focal["alleles"]  # (n_variants, n_alleles)
    focal_gt = focal["genotypes"]     # (n_variants, n_samples, 2)
    focal_samples = focal["samples"]

    # Load arabiensis outgroup
    print(f"  Loading arabiensis outgroup...")
    arab = np.load(arm_cache / "arab_snp_calls.npz", allow_pickle=True)
    arab_pos = arab["positions"]
    arab_gt = arab["genotypes"]  # (n_variants, n_arab_samples, 2)

    # Region filter
    if region is not None:
        start, end = region
        mask = (focal_pos >= start) & (focal_pos < end)
        focal_pos = focal_pos[mask]
        focal_alleles = focal_alleles[mask]
        focal_gt = focal_gt[mask]

    n_focal = len(focal_pos)
    n_haplotypes = focal_gt.shape[1] * 2
    print(f"  {arm}: {n_focal:,} focal variants, {n_haplotypes} haplotypes")

    # Find overlapping positions between focal and arabiensis
    arab_pos_set = set(arab_pos.tolist())
    arab_pos_idx = {int(p): i for i, p in enumerate(arab_pos)}

    # For each focal site, determine ancestral allele from arabiensis
    aa_allele = np.full(n_focal, -1, dtype=np.int8)  # -1 = unknown
    aa_prob = np.full(n_focal, np.nan, dtype=np.float32)
    n_found = 0
    n_polarized = 0

    for i in range(n_focal):
        p = int(focal_pos[i])
        if p not in arab_pos_idx:
            continue
        n_found += 1

        # Get arabiensis genotypes at this site
        j = arab_pos_idx[p]
        arab_gts = arab_gt[j].reshape(-1)
        # Filter missing (-1)
        arab_valid = arab_gts[arab_gts >= 0]
        if len(arab_valid) == 0:
            continue

        # Allele 0 = REF, allele 1 = first ALT in the focal data
        # arabiensis uses same REF/ALT encoding (same VCF)
        ref_freq = (arab_valid == 0).mean()
        alt_freq = (arab_valid == 1).mean()

        # Determine ancestral: whichever allele is at higher freq in arabiensis
        if ref_freq >= min_outgroup_freq:
            aa_allele[i] = 0  # REF is ancestral
            aa_prob[i] = ref_freq
            n_polarized += 1
        elif alt_freq >= min_outgroup_freq:
            aa_allele[i] = 1  # ALT is ancestral
            aa_prob[i] = alt_freq
            n_polarized += 1
        else:
            # Ambiguous — arabiensis is polymorphic at this site
            # Still assign based on majority, but with lower confidence
            if ref_freq >= alt_freq:
                aa_allele[i] = 0
                aa_prob[i] = ref_freq
            else:
                aa_allele[i] = 1
                aa_prob[i] = alt_freq
            n_polarized += 1

    print(f"    Arabiensis overlap: {n_found:,}/{n_focal:,} ({100*n_found/n_focal:.1f}%)")
    print(f"    Polarized: {n_polarized:,}")

    # Filter: only keep sites that could be polarized and pass confidence threshold
    keep = aa_allele >= 0
    if min_aa_prob > 0:
        keep &= aa_prob >= min_aa_prob
    n_keep = keep.sum()

    high_conf = keep & (aa_prob >= min_outgroup_freq)
    print(f"    High-confidence (outgroup freq >= {min_outgroup_freq}): {high_conf.sum():,}")
    print(f"    Kept after filters: {n_keep:,}")

    if n_keep == 0:
        print(f"    WARNING: No sites passed filters")
        return {"arm": arm, "n_sites": 0}

    # Build polarized haplotype matrix
    kept_gt = focal_gt[keep]  # (n_kept, n_samples, 2)
    kept_aa = aa_allele[keep]
    kept_prob = aa_prob[keep]
    kept_pos = focal_pos[keep]

    # Reshape to (n_haplotypes, n_kept)
    haplotypes = kept_gt.reshape(n_keep, -1).T.astype(np.int8)

    # Flip sites where ALT is ancestral
    needs_flip = kept_aa == 1
    n_flip = needs_flip.sum()
    if n_flip > 0:
        flip_cols = np.where(needs_flip)[0]
        haplotypes[:, flip_cols] = 1 - haplotypes[:, flip_cols]
    print(f"    Flipped {n_flip:,} sites ({100*n_flip/n_keep:.1f}%)")

    # Assign condition labels
    aa_cond = np.where(kept_prob >= min_outgroup_freq, "outgroup_fixed", "outgroup_poly")

    # Save
    out_path = out_dir / f"{arm}.npz"
    np.savez_compressed(
        out_path,
        haplotypes=haplotypes,    # (n_haplotypes, n_sites), 0=ancestral, 1=derived
        positions=kept_pos,        # (n_sites,)
        aa_prob=kept_prob,         # (n_sites,)
        aa_cond=aa_cond,           # (n_sites,)
        samples=focal_samples,     # (n_samples,) — each appears twice (diploid)
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
        "pct_high_conf": 100 * high_conf.sum() / n_keep if n_keep > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Step 3: Validate against existing polarized VCF (optional)
# ---------------------------------------------------------------------------

DEFAULT_POLARIZED_VCF_DIR = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/phased_vcf/gamb"


def validate_against_vcf(
    out_dir: Path,
    arm: str = "2L",
    region: tuple[int, int] = (0, 100_000),
    vcf_dir: str = DEFAULT_POLARIZED_VCF_DIR,
):
    """Compare our polarization against the existing polarized VCF on sietch."""
    import pysam

    vcf_path = f"{vcf_dir}/gamb.{arm}.phased.n1470.derived.vcf.gz"
    print(f"\n=== Validation: {Path(vcf_path).name} ({arm}:{region[0]}-{region[1]}) ===")

    if not Path(vcf_path).exists():
        print(f"  VCF not found, skipping validation")
        return

    data = np.load(out_dir / f"{arm}.npz", allow_pickle=True)
    our_pos = data["positions"]
    our_hap = data["haplotypes"]
    our_prob = data["aa_prob"]

    vcf_pos = []
    vcf_derived_freq = []
    vcf = pysam.VariantFile(vcf_path)
    for rec in vcf.fetch(arm, region[0], region[1]):
        vcf_pos.append(rec.pos)
        aa_vcf = rec.info.get("AA", ".")
        gts = []
        for s in rec.samples.values():
            gts.extend(s["GT"])
        gts = np.array([g for g in gts if g is not None], dtype=np.int8)
        if aa_vcf == rec.ref:
            vcf_derived_freq.append(gts.mean())
        else:
            vcf_derived_freq.append(1.0 - gts.mean())
    vcf.close()

    vcf_pos = np.array(vcf_pos)
    vcf_derived_freq = np.array(vcf_derived_freq)

    common_pos = np.intersect1d(our_pos, vcf_pos)
    print(f"  Our sites: {len(our_pos)}, VCF sites: {len(vcf_pos)}, Overlap: {len(common_pos)}")

    if len(common_pos) == 0:
        return

    our_idx = np.searchsorted(our_pos, common_pos)
    vcf_idx = np.searchsorted(vcf_pos, common_pos)

    our_freq = our_hap[:, our_idx].mean(axis=0)
    ref_freq = vcf_derived_freq[vcf_idx]

    freq_diff = np.abs(our_freq - ref_freq)
    concordant = freq_diff < 0.01
    print(f"  Concordant: {concordant.sum()}/{len(common_pos)} ({100*concordant.mean():.1f}%)")
    print(f"  Mean |freq diff|: {freq_diff.mean():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download AG3 data from MalariaGEN, polarize, and prepare for fastcxt"
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=CHROMOSOME_ARMS,
                        choices=CHROMOSOME_ARMS)
    parser.add_argument("--sample-sets", default=DEFAULT_SAMPLE_SETS)
    parser.add_argument("--analysis", default=DEFAULT_ANALYSIS,
                        help="Phasing analysis for haplotypes")
    parser.add_argument("--site-mask", default=DEFAULT_SITE_MASK)
    parser.add_argument("--sample-query", default=None,
                        help="Filter focal samples (e.g. \"country == 'Burkina Faso'\")")
    parser.add_argument("--min-outgroup-freq", type=float, default=DEFAULT_MIN_OUTGROUP_FREQ,
                        help="Min arabiensis allele freq to call ancestral (default: 0.95)")
    parser.add_argument("--min-aa-prob", type=float, default=0.0,
                        help="Min ancestral probability to keep site")
    parser.add_argument("--test-region", default=None,
                        help="Restrict to region (e.g. 0:100000)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing cache")
    parser.add_argument("--download-only", action="store_true",
                        help="Download only, don't polarize")
    parser.add_argument("--validate", action="store_true",
                        help="Validate against existing polarized VCF on sietch")
    parser.add_argument("--vcf-dir", default=DEFAULT_POLARIZED_VCF_DIR,
                        help="Path to existing polarized VCFs for validation")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / "cache"

    region = None
    if args.test_region:
        start, end = args.test_region.split(":")
        region = (int(start), int(end))
        print(f"Region filter: {region[0]:,}-{region[1]:,}")

    # Step 1: Download
    if not args.skip_download:
        print("=" * 60)
        print("STEP 1: Download from MalariaGEN AG3")
        print("=" * 60)
        download_ag3(
            args.out_dir, args.arms, args.sample_sets,
            args.analysis, args.site_mask, args.sample_query,
        )

    if args.download_only:
        return

    # Step 2: Polarize
    print("\n" + "=" * 60)
    print("STEP 2: Polarize using arabiensis outgroup")
    print("=" * 60)

    results = []
    for arm in args.arms:
        print(f"\nPolarizing {arm}...")
        stats = polarize_arm(
            cache_dir, arm, args.out_dir,
            min_outgroup_freq=args.min_outgroup_freq,
            min_aa_prob=args.min_aa_prob,
            region=region,
        )
        results.append(stats)

    # Step 3: Validate (optional)
    if args.validate:
        print("\n" + "=" * 60)
        print("STEP 3: Validate against existing polarized VCF")
        print("=" * 60)
        test_region = region or (0, 1_000_000)
        for arm in args.arms:
            validate_against_vcf(args.out_dir, arm, test_region, args.vcf_dir)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_sites = 0
    total_mb = 0
    for r in results:
        if r["n_sites"] > 0:
            print(f"  {r['arm']}: {r['n_sites']:,} sites, "
                  f"{r['n_flip']:,} flipped, "
                  f"{r['n_haplotypes']:,} haplotypes, "
                  f"{r['pct_high_conf']:.0f}% high-conf, "
                  f"{r['size_mb']:.1f} MB")
            total_sites += r["n_sites"]
            total_mb += r["size_mb"]
    print(f"\n  Total: {total_sites:,} polarized sites, {total_mb:.1f} MB")
    print(f"  Output: {args.out_dir}")
    print(f"\n  Output format (per arm .npz):")
    print(f"    haplotypes: (n_haplotypes, n_sites) int8, 0=ancestral 1=derived")
    print(f"    positions:  (n_sites,) int32, bp coordinates")
    print(f"    aa_prob:    (n_sites,) float32, outgroup confidence")
    print(f"    samples:    (n_samples,) sample IDs")


if __name__ == "__main__":
    main()
