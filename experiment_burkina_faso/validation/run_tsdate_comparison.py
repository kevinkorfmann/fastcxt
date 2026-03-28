#!/usr/bin/env python3
"""Re-infer focal regions from VCF subsets with latest tsinfer + tsdate.

VCF subsets are pre-created by run_validation.sh using tabix.
Each subset is ~1 Mb and contains only a few thousand sites, making
tsinfer + tsdate fast (seconds per population).

Usage:
    pixi run python -u experiment_burkina_faso/validation/run_tsdate_comparison.py \
        --vcf-dir results/tsdate_validation/vcf_subsets \
        --meta gamb.meta.tsinfer.csv \
        --fastcxt-results results/het_allpop \
        --out-dir results/tsdate_validation
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

FOCAL_REGIONS = [
    ("Vgsc_kdr", "2L", 1_900_000, 2_900_000, "2L_Vgsc_kdr.vcf.gz"),
    ("Rdl",      "2L", 24_900_000, 25_900_000, "2L_Rdl.vcf.gz"),
    ("Neutral",  "3L", 20_000_000, 21_000_000, "3L_Neutral.vcf.gz"),
]

VALIDATION_POPS = ["Burkina Faso", "Cameroon", "Kenya"]
MUTATION_RATE = 3.5e-9
BLOCK_SIZE = 100_000
MAX_SAMPLES = 40


def load_vcf_samples(vcf_path):
    """Get sample IDs from VCF header."""
    import gzip
    with gzip.open(str(vcf_path), "rt") as f:
        for line in f:
            if line.startswith("#CHROM"):
                fields = line.strip().split("\t")
                return fields[9:]  # sample IDs
    return []


def run_tsinfer_tsdate_from_vcf(vcf_path, sample_ids, sequence_length):
    """Run tsinfer + tsdate on a VCF subset for specific samples."""
    import tsinfer
    import tsdate
    import allel

    # Load VCF with scikit-allel
    try:
        callset = allel.read_vcf(str(vcf_path), samples=sample_ids,
                                  fields=["variants/POS", "calldata/GT", "variants/REF", "variants/ALT"])
    except Exception:
        # Fallback: manual VCF parsing
        return _run_tsinfer_tsdate_manual(vcf_path, sample_ids, sequence_length)

    if callset is None:
        return None

    positions = callset["variants/POS"]
    gt = callset["calldata/GT"]  # (n_sites, n_samples, ploidy)
    ref = callset["variants/REF"]
    alt = callset["variants/ALT"][:, 0]

    # Filter biallelic SNPs without missing data
    mask = np.all(gt >= 0, axis=(1, 2))
    positions = positions[mask]
    gt = gt[mask]
    ref = ref[mask]
    alt = alt[mask]

    n_sites = len(positions)
    n_samples = gt.shape[1]
    haplotypes = gt.reshape(n_sites, n_samples * 2)

    log.info(f"    {n_sites} sites, {n_samples} diploids ({n_samples*2} haploids)")

    # tsinfer
    t0 = time.time()
    with tsinfer.SampleData(sequence_length=sequence_length) as sd:
        for i in range(n_sites):
            sd.add_site(position=positions[i], genotypes=haplotypes[i],
                        alleles=[ref[i], alt[i]])
    ts = tsinfer.infer(sd)
    log.info(f"    tsinfer: {time.time()-t0:.1f}s, {ts.num_trees} trees")

    # tsdate
    t0 = time.time()
    ts_dated = tsdate.date(ts, mutation_rate=MUTATION_RATE, method="variational_gamma")
    log.info(f"    tsdate: {time.time()-t0:.1f}s")

    return ts_dated


def _run_tsinfer_tsdate_manual(vcf_path, sample_ids, sequence_length):
    """Fallback VCF parser without scikit-allel."""
    import gzip
    import tsinfer
    import tsdate

    all_samples = load_vcf_samples(vcf_path)
    sample_indices = [all_samples.index(s) for s in sample_ids if s in all_samples]
    if len(sample_indices) < 2:
        return None

    positions, haplotypes, alleles_list = [], [], []
    opener = gzip.open if str(vcf_path).endswith(".gz") else open
    with opener(str(vcf_path), "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            pos = int(fields[1])
            ref, alt = fields[3], fields[4]
            if "," in alt:
                continue  # skip multiallelic

            gts = []
            skip = False
            for si in sample_indices:
                gt_str = fields[9 + si].split(":")[0]
                for a in gt_str.replace("|", "/").split("/"):
                    if a == ".":
                        skip = True
                        break
                    gts.append(int(a))
                if skip:
                    break
            if skip or not gts:
                continue

            positions.append(pos)
            haplotypes.append(gts)
            alleles_list.append((ref, alt))

    if not positions:
        return None

    positions = np.array(positions)
    haplotypes = np.array(haplotypes, dtype=np.int8)
    n_sites, n_haploids = haplotypes.shape

    log.info(f"    {n_sites} sites, {n_haploids} haploids (manual parser)")

    t0 = time.time()
    with tsinfer.SampleData(sequence_length=sequence_length) as sd:
        for i in range(n_sites):
            sd.add_site(position=positions[i], genotypes=haplotypes[i],
                        alleles=alleles_list[i])
    ts = tsinfer.infer(sd)
    log.info(f"    tsinfer: {time.time()-t0:.1f}s, {ts.num_trees} trees")

    t0 = time.time()
    ts_dated = tsdate.date(ts, mutation_rate=MUTATION_RATE, method="variational_gamma")
    log.info(f"    tsdate: {time.time()-t0:.1f}s")

    return ts_dated


def extract_tmrca(ts, block_starts, block_size=BLOCK_SIZE):
    """Extract per-block mean log(TMRCA) for intra-individual pairs."""
    import tskit

    n_haploids = ts.num_samples
    pairs = [(i*2, i*2+1) for i in range(n_haploids // 2)]
    n_blocks = len(block_starts)

    tmrca = np.full((n_blocks, len(pairs)), np.nan)
    for bi, bs in enumerate(block_starts):
        mid = bs + block_size // 2
        if mid >= ts.sequence_length or mid < 0:
            continue
        try:
            tree = ts.at(mid)
        except Exception:
            continue
        for pi, (sa, sb) in enumerate(pairs):
            try:
                m = tree.mrca(sa, sb)
                if m != tskit.NULL:
                    t = tree.time(m)
                    if t > 0:
                        tmrca[bi, pi] = np.log(t)
            except Exception:
                continue

    return tmrca, pairs


def load_fastcxt_predictions(results_dir, pop, arm, group, start_bp, end_bp):
    """Load fastcxt block-level mean predictions for a region."""
    d = results_dir / pop.replace(" ", "_") / arm / f"{group}_intra"
    if not (d / "means.npz").exists():
        return None, None

    means = np.load(d / "means.npz")["means"]
    imap = np.load(d / "index_map.npy")
    blocks = json.load(open(d / "blocks.json"))

    block_pair_data = {}
    for r in range(imap.shape[0]):
        bi = imap[r, 0]
        b = blocks[bi]
        if b["start"] < start_bp or b["end"] > end_bp:
            continue
        if bi not in block_pair_data:
            block_pair_data[bi] = []
        block_pair_data[bi].append(np.mean(means[r]))

    if not block_pair_data:
        return None, None

    sorted_bi = sorted(block_pair_data.keys())
    starts = np.array([blocks[bi]["start"] for bi in sorted_bi])
    vals = np.array([np.mean(block_pair_data[bi]) for bi in sorted_bi])
    return vals, starts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf-dir", type=Path, required=True)
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--fastcxt-results", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.meta)
    log.info(f"Metadata: {len(meta)} samples")

    try:
        import tsdate
        log.info(f"tsdate version: {tsdate.__version__}")
    except ImportError:
        log.error("tsdate not installed"); return

    results = []

    for region_name, arm, start_bp, end_bp, vcf_name in FOCAL_REGIONS:
        log.info(f"\n{'='*60}")
        log.info(f"Region: {region_name} ({arm}:{start_bp/1e6:.1f}-{end_bp/1e6:.1f} Mb)")
        log.info(f"{'='*60}")

        vcf_path = args.vcf_dir / vcf_name
        if not vcf_path.exists():
            log.warning(f"VCF subset not found: {vcf_path}")
            continue

        for pop in VALIDATION_POPS:
            log.info(f"\n  Population: {pop}")

            pop_meta = meta[meta["country"] == pop]
            if len(pop_meta) == 0:
                pop_meta = meta[meta["country"].str.contains(pop, case=False, na=False)]
            if len(pop_meta) == 0:
                log.warning(f"    No samples"); continue

            if len(pop_meta) > MAX_SAMPLES:
                pop_meta = pop_meta.sample(MAX_SAMPLES, random_state=42)

            sample_ids = pop_meta["sample_id"].tolist()
            log.info(f"    {len(sample_ids)} samples selected")

            # Run tsinfer + tsdate
            ts_dated = run_tsinfer_tsdate_from_vcf(vcf_path, sample_ids, end_bp)
            if ts_dated is None:
                log.warning(f"    Failed"); continue

            # Extract TMRCA
            block_starts = np.arange(start_bp, end_bp, BLOCK_SIZE)
            tsdate_tmrca, pairs = extract_tmrca(ts_dated, block_starts)
            tsdate_means = np.nanmean(tsdate_tmrca, axis=1)
            log.info(f"    tsdate blocks: {len(block_starts)}, "
                     f"mean log(TMRCA)={np.nanmean(tsdate_means):.2f}")

            # Compare with fastcxt
            group = "2La_hom_standard" if arm == "2L" else "all"
            fcxt_vals, fcxt_starts = load_fastcxt_predictions(
                args.fastcxt_results, pop, arm, group, start_bp, end_bp)

            if fcxt_vals is not None:
                common = np.intersect1d(block_starts, fcxt_starts)
                if len(common) > 2:
                    ti = np.array([np.where(block_starts == s)[0][0] for s in common])
                    fi = np.array([np.where(fcxt_starts == s)[0][0] for s in common])
                    tv, fv = tsdate_means[ti], fcxt_vals[fi]
                    valid = np.isfinite(tv) & np.isfinite(fv)
                    if valid.sum() > 2:
                        from scipy.stats import pearsonr
                        r, p = pearsonr(tv[valid], fv[valid])
                        log.info(f"    Correlation: r={r:.3f}, p={p:.1e}, n={valid.sum()}")
                        results.append({
                            "region": region_name, "arm": arm, "pop": pop,
                            "n_blocks": int(valid.sum()), "r": float(r), "p": float(p),
                            "tsdate_mean": float(np.mean(tv[valid])),
                            "fastcxt_mean": float(np.mean(fv[valid])),
                        })

            # Save
            out = args.out_dir / f"tsdate_{region_name}_{pop.replace(' ', '_')}.npz"
            np.savez_compressed(out, block_starts=block_starts,
                                tsdate_means=tsdate_means, tsdate_tmrca=tsdate_tmrca)
            log.info(f"    Saved {out.name}")

    if results:
        with open(args.out_dir / "validation_summary.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n" + "=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"{'Region':<15} {'Pop':<20} {'n':>4} {'r':>8} {'p':>10}")
        print("-" * 60)
        for r in results:
            print(f"{r['region']:<15} {r['pop']:<20} {r['n_blocks']:>4} "
                  f"{r['r']:>8.3f} {r['p']:>10.1e}")


if __name__ == "__main__":
    main()
