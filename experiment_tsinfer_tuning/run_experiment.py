#!/usr/bin/env python3
"""Tsinfer parameter tuning experiment.

Subsets a small region from the AG3 polarized VCF, runs tsinfer with
different mismatch_ratio values, and evaluates inferred tree quality
against simulated ground truth + real data metrics.

Two modes:
  1. Simulated: generate msprime trees, extract genotypes, run tsinfer,
     compare inferred vs true (KC distance, tree count, branch length).
  2. Real data: subset AG3 VCF, run tsinfer, measure tree statistics.

Usage:
    # Simulated benchmark (fast, has ground truth)
    python run_experiment.py --mode sim --out-dir results/sim

    # Real data (subset of 2L)
    python run_experiment.py --mode real --out-dir results/real --region 2L:1000000-2000000

    # Both
    python run_experiment.py --mode both --out-dir results
"""

import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MISMATCH_RATIOS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, None]  # None = tsinfer default

# Simulation parameters (AnoGam-like)
SIM_SEQ_LEN = 100_000  # 100 kb (fast iteration)
SIM_SAMPLE_SIZE = 50
SIM_NE = 1e6
SIM_MU = 3.5e-9
SIM_RECOMB = 1.3e-8  # AnoGam 2L/2R/3L rate

# Real data paths on sietch
DEFAULT_VCF = "/sietch_colab/data_share/Ag1000G/Ag3.0/vcf/phased_vcf/gamb/gamb.2L.phased.n1470.derived.vcf.gz"
DEFAULT_REGION = "2L:5000000-6000000"  # 1 Mb region

BLUE = "#2166ac"
GREEN = "#1b7837"
RED = "#b2182b"
ORANGE = "#e08214"
BLACK = "#252525"
GREY = "#636363"


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": BLACK, "axes.linewidth": 0.6,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300,
    })


# ---------------------------------------------------------------------------
# Mode 1: Simulated data (ground truth available)
# ---------------------------------------------------------------------------

def run_sim_experiment(out_dir: Path, n_threads: int = 4):
    """Run tsinfer on simulated data and compare against true trees."""
    import msprime
    import tsinfer
    import tskit

    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== Simulated Experiment ===")
    print(f"  {SIM_SEQ_LEN/1e6:.0f} Mb, n={SIM_SAMPLE_SIZE}, Ne={SIM_NE:.0e}")

    # Simulate true trees
    print("  Simulating...")
    ts_true = msprime.sim_ancestry(
        samples=SIM_SAMPLE_SIZE,
        population_size=SIM_NE,
        recombination_rate=SIM_RECOMB,
        sequence_length=SIM_SEQ_LEN,
        random_seed=42,
    )
    ts_true = msprime.sim_mutations(ts_true, rate=SIM_MU, random_seed=42)
    print(f"  True: {ts_true.num_trees:,} trees, {ts_true.num_sites:,} sites")

    # Create SampleData from true trees
    sd = tsinfer.SampleData.from_tree_sequence(ts_true, use_sites_time=False)

    results = []
    for mr in MISMATCH_RATIOS:
        label = f"mr={mr}" if mr is not None else "default"
        print(f"\n  --- {label} ---")

        t0 = time.time()
        # Must call steps separately so mismatch_ratio applies to both stages
        ancestors = tsinfer.generate_ancestors(sd, num_threads=n_threads)

        ma_kwargs = dict(num_threads=n_threads)
        ms_kwargs = dict(num_threads=n_threads)
        if mr is not None:
            ma_kwargs["recombination_rate"] = SIM_RECOMB
            ma_kwargs["mismatch_ratio"] = mr
            ms_kwargs["recombination_rate"] = SIM_RECOMB
            ms_kwargs["mismatch_ratio"] = mr

        ancestors_ts = tsinfer.match_ancestors(sd, ancestors, **ma_kwargs)
        ts_inferred = tsinfer.match_samples(sd, ancestors_ts, **ms_kwargs)
        ts_inferred = ts_inferred.simplify()
        elapsed = time.time() - t0

        # Metrics
        n_trees = ts_inferred.num_trees
        total_bl = sum(t.total_branch_length for t in ts_inferred.trees())
        mean_bl = total_bl / n_trees

        # KC distance (topology comparison with true trees)
        # Requires single-root trees — tsinfer can produce multi-root
        try:
            kc_topo = ts_true.kc_distance(ts_inferred, lambda_=0)
            kc_bl = ts_true.kc_distance(ts_inferred, lambda_=1)
        except Exception:
            print(f"    KC distance failed (multi-root trees), skipping")
            kc_topo = float("nan")
            kc_bl = float("nan")

        # Robinson-Foulds-like: count matching breakpoints
        true_breaks = set(int(t.interval.left) for t in ts_true.trees())
        inferred_breaks = set(int(t.interval.left) for t in ts_inferred.trees())
        shared_breaks = len(true_breaks & inferred_breaks)

        r = {
            "mismatch_ratio": mr if mr is not None else "default",
            "n_trees": n_trees,
            "n_trees_true": ts_true.num_trees,
            "mean_branch_length": mean_bl,
            "kc_topo": kc_topo,
            "kc_bl": kc_bl,
            "shared_breakpoints": shared_breaks,
            "true_breakpoints": len(true_breaks),
            "breakpoint_recall": shared_breaks / len(true_breaks) if true_breaks else 0,
            "time_s": elapsed,
        }
        results.append(r)
        print(f"    trees: {n_trees:,} (true: {ts_true.num_trees:,})")
        print(f"    KC(topo): {kc_topo:.2f}, KC(bl): {kc_bl:.2f}")
        print(f"    breakpoint recall: {shared_breaks}/{len(true_breaks)} ({r['breakpoint_recall']:.1%})")
        print(f"    time: {elapsed:.1f}s")

        # Save inferred tree
        ts_inferred.dump(out_dir / f"inferred_{label.replace('=', '')}.trees")

    ts_true.dump(out_dir / "true.trees")

    # Save results
    import json
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_sim_results(results, out_dir)
    return results


def plot_sim_results(results, out_dir):
    """Plot mismatch_ratio vs quality metrics."""
    mrs = [r["mismatch_ratio"] for r in results]
    # Convert "default" to a placeholder for plotting
    x_labels = [str(mr) for mr in mrs]
    x = range(len(mrs))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")

    # KC distance (topology) — may be NaN if multi-root
    ax = axes[0, 0]
    kc_vals = [r["kc_topo"] for r in results]
    colors_kc = [BLUE if not np.isnan(v) else GREY for v in kc_vals]
    kc_vals_plot = [v if not np.isnan(v) else 0 for v in kc_vals]
    ax.bar(x, kc_vals_plot, color=colors_kc, edgecolor=BLACK, linewidth=0.5)
    ax.set_ylabel("KC distance (topology)")
    ax.set_title("Topology accuracy (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)

    # KC distance (branch lengths)
    ax = axes[0, 1]
    kc_bl_vals = [r["kc_bl"] for r in results]
    colors_bl = [GREEN if not np.isnan(v) else GREY for v in kc_bl_vals]
    kc_bl_plot = [v if not np.isnan(v) else 0 for v in kc_bl_vals]
    ax.bar(x, kc_bl_plot, color=colors_bl, edgecolor=BLACK, linewidth=0.5)
    ax.set_ylabel("KC distance (branch lengths)")
    ax.set_title("Branch length accuracy (lower = better)")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)

    # Number of trees
    ax = axes[1, 0]
    true_n = results[0]["n_trees_true"]
    ax.bar(x, [r["n_trees"] for r in results], color=ORANGE, edgecolor=BLACK, linewidth=0.5)
    ax.axhline(true_n, color=RED, ls="--", lw=1, label=f"True ({true_n:,})")
    ax.set_ylabel("Number of trees")
    ax.set_title("Tree count")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)
    ax.legend()

    # Breakpoint recall
    ax = axes[1, 1]
    ax.bar(x, [r["breakpoint_recall"] for r in results], color=RED, edgecolor=BLACK, linewidth=0.5)
    ax.set_ylabel("Breakpoint recall")
    ax.set_title("Fraction of true breakpoints recovered")
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45)

    fig.suptitle("tsinfer mismatch_ratio tuning (simulated AnoGam-like data)", fontsize=12)
    for ax in axes.flat:
        ax.set_xlabel("mismatch_ratio")

    fig.savefig(out_dir / "mismatch_ratio_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {out_dir / 'mismatch_ratio_comparison.png'}")


# ---------------------------------------------------------------------------
# Mode 2: Real AG3 data
# ---------------------------------------------------------------------------

def run_real_experiment(out_dir: Path, vcf_path: str, region: str, n_threads: int = 4):
    """Run tsinfer on a subset of real AG3 data."""
    import tsinfer
    import pysam

    out_dir.mkdir(parents=True, exist_ok=True)
    print("=== Real Data Experiment ===")
    print(f"  VCF: {vcf_path}")
    print(f"  Region: {region}")

    arm, coords = region.split(":")
    start, end = [int(x) for x in coords.split("-")]
    region_len = end - start

    # Extract genotypes from VCF region
    print(f"  Extracting genotypes from {region}...")
    vcf = pysam.VariantFile(vcf_path)
    sample_names = list(vcf.header.samples)
    n_samples = len(sample_names)

    positions = []
    genotypes = []
    alleles_list = []
    for rec in vcf.fetch(arm, start, end):
        if len(rec.alts) != 1 or len(rec.ref) != 1 or len(rec.alts[0]) != 1:
            continue
        aa = rec.info.get("AA", None)
        if aa is None:
            continue

        pos = rec.pos - start  # relative position
        gts = np.zeros(n_samples * 2, dtype=np.int32)
        for j, s in enumerate(rec.samples.values()):
            gt = s["GT"]
            if gt[0] is not None:
                if aa == rec.alts[0]:
                    gts[2 * j] = 1 - gt[0]
                    gts[2 * j + 1] = 1 - gt[1]
                else:
                    gts[2 * j] = gt[0]
                    gts[2 * j + 1] = gt[1]

        if aa == rec.ref:
            alleles = [rec.ref, rec.alts[0]]
        else:
            alleles = [rec.alts[0], rec.ref]

        positions.append(pos)
        genotypes.append(gts)
        alleles_list.append(alleles)

    vcf.close()
    print(f"  Extracted {len(positions):,} biallelic SNPs, {n_samples} samples")

    # Create SampleData
    # Build a temporary VCF for the region, then use VariantData
    # Alternatively, build SampleData in memory (avoid deprecated file format)
    import zarr
    sd_path = out_dir / "region.samples.zarr"
    if sd_path.exists():
        import shutil
        shutil.rmtree(sd_path)

    # Use tsinfer.SampleData in-memory (still works, just deprecated for file I/O)
    with tsinfer.SampleData(sequence_length=region_len) as sd:
        for _ in sample_names:
            sd.add_individual(ploidy=2)
        for pos, gts, alleles in zip(positions, genotypes, alleles_list):
            sd.add_site(position=pos, genotypes=gts, alleles=alleles)

    results = []
    for mr in MISMATCH_RATIOS:
        label = f"mr={mr}" if mr is not None else "default"
        print(f"\n  --- {label} ---")

        t0 = time.time()
        # Must call steps separately so mismatch_ratio applies to both stages
        ancestors = tsinfer.generate_ancestors(sd, num_threads=n_threads)

        ma_kwargs = dict(num_threads=n_threads)
        ms_kwargs = dict(num_threads=n_threads)
        if mr is not None:
            ma_kwargs["recombination_rate"] = SIM_RECOMB
            ma_kwargs["mismatch_ratio"] = mr
            ms_kwargs["recombination_rate"] = SIM_RECOMB
            ms_kwargs["mismatch_ratio"] = mr

        ancestors_ts = tsinfer.match_ancestors(sd, ancestors, **ma_kwargs)
        ts = tsinfer.match_samples(sd, ancestors_ts, **ms_kwargs)
        ts = ts.simplify()
        elapsed = time.time() - t0

        n_trees = ts.num_trees
        total_bl = sum(t.total_branch_length for t in ts.trees())
        mean_bl = total_bl / n_trees
        mean_span = region_len / n_trees

        r = {
            "mismatch_ratio": mr if mr is not None else "default",
            "n_trees": n_trees,
            "mean_branch_length": mean_bl,
            "mean_span_bp": mean_span,
            "time_s": elapsed,
        }
        results.append(r)
        print(f"    trees: {n_trees:,}, mean span: {mean_span:.0f} bp")
        print(f"    mean branch length: {mean_bl:.1f}")
        print(f"    time: {elapsed:.1f}s")

        ts.dump(out_dir / f"inferred_{label.replace('=', '')}.trees")

    import json
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    plot_real_results(results, out_dir, region)
    return results


def plot_real_results(results, out_dir, region):
    """Plot mismatch_ratio effects on real data."""
    mrs = [str(r["mismatch_ratio"]) for r in results]
    x = range(len(mrs))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")

    ax = axes[0]
    ax.bar(x, [r["n_trees"] for r in results], color=BLUE, edgecolor=BLACK, linewidth=0.5)
    ax.set_ylabel("Number of trees")
    ax.set_title("Tree count")
    ax.set_xticks(x); ax.set_xticklabels(mrs, rotation=45)
    ax.set_xlabel("mismatch_ratio")

    ax = axes[1]
    ax.bar(x, [r["mean_span_bp"] for r in results], color=GREEN, edgecolor=BLACK, linewidth=0.5)
    ax.set_ylabel("Mean tree span (bp)")
    ax.set_title("Average span per tree")
    ax.set_xticks(x); ax.set_xticklabels(mrs, rotation=45)
    ax.set_xlabel("mismatch_ratio")

    ax = axes[2]
    ax.bar(x, [r["time_s"] for r in results], color=ORANGE, edgecolor=BLACK, linewidth=0.5)
    ax.set_ylabel("Time (s)")
    ax.set_title("Runtime")
    ax.set_xticks(x); ax.set_xticklabels(mrs, rotation=45)
    ax.set_xlabel("mismatch_ratio")

    fig.suptitle(f"tsinfer mismatch_ratio tuning — {region}", fontsize=12)
    fig.savefig(out_dir / "mismatch_ratio_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {out_dir / 'mismatch_ratio_comparison.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_style()
    parser = argparse.ArgumentParser(description="Tsinfer parameter tuning experiment")
    parser.add_argument("--mode", choices=["sim", "real", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--vcf", default=DEFAULT_VCF)
    parser.add_argument("--region", default=DEFAULT_REGION,
                        help="Region to subset (e.g. 2L:5000000-6000000)")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()

    if args.mode in ("sim", "both"):
        run_sim_experiment(args.out_dir / "sim", args.num_threads)

    if args.mode in ("real", "both"):
        run_real_experiment(args.out_dir / "real", args.vcf, args.region, args.num_threads)


if __name__ == "__main__":
    main()
