#!/usr/bin/env python3
"""Date AG1000G populations individually using the pairwise FastCxtModel.

Loads a tsinfer tree sequence for a given chromosome arm, identifies
populations from individual metadata, and runs pairwise TMRCA inference
for each population separately.

Usage:
    # Date all populations on 2L
    python run_population_dating.py --arm 2L --checkpoint /path/to/best.ckpt

    # Date a specific population
    python run_population_dating.py --arm 2L --checkpoint best.ckpt --populations "Burkina Faso"

    # Use a custom tree path
    python run_population_dating.py --arm 2L --checkpoint best.ckpt \
        --trees /path/to/2L.trees

    # Re-plot from cached results
    python run_population_dating.py --arm 2L --from-cache --out-dir results/2L
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.serialization

from config import (
    EXPERIMENT_DIR,
    TREES_DIR,
    TREE_FILE_TEMPLATE,
    ACCESSIBILITY_MASK,
    METADATA_CSV,
    DEFAULT_CHECKPOINT,
    BLOCK_SIZE,
    WINDOW_SIZE,
    N_WINDOWS,
    MUTATION_RATE,
    CHROMOSOME_ARMS,
    BATCH_SIZE,
    BUILD_WORKERS,
    MAX_PAIRS_PER_POP,
)

# ---------------------------------------------------------------------------
# Imports (deferred for cluster compatibility)
# ---------------------------------------------------------------------------


def _import_fastcxt():
    from fastcxt.config import FastCxtConfig, TrainingConfig
    from fastcxt.train import LitFastCxt
    from fastcxt.translate import translate_from_genotype_matrix
    from fastcxt.mosquito import AccessibilityMask, generate_blocks

    torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])
    return LitFastCxt, translate_from_genotype_matrix, AccessibilityMask, generate_blocks


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(ckpt_path: str, device: str = "cpu"):
    LitFastCxt, *_ = _import_fastcxt()
    lit = LitFastCxt.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    model = lit.model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Population extraction from tsinfer tree sequence
# ---------------------------------------------------------------------------


def get_populations(ts) -> dict[str, list[int]]:
    """Extract population → haploid node IDs from tree sequence metadata.

    Returns dict mapping population name (country) to sorted list of
    sample node IDs belonging to that population.
    """
    pop_nodes: dict[str, list[int]] = {}

    for ind in ts.individuals():
        meta = json.loads(ind.metadata.decode()) if ind.metadata else {}
        country = meta.get("country", "unknown")
        if country not in pop_nodes:
            pop_nodes[country] = []
        pop_nodes[country].extend(ind.nodes)

    # Sort nodes within each population
    for k in pop_nodes:
        pop_nodes[k] = sorted(pop_nodes[k])

    return pop_nodes


def build_pairs(n_haploids: int, max_pairs: int | None = None) -> list[tuple[int, int]]:
    """Build within-individual + between-individual pairs.

    Haploid indices are local (0-based after simplification).
    """
    n_individuals = n_haploids // 2
    within = [(2 * i, 2 * i + 1) for i in range(n_individuals)]
    between = [(2 * i, 2 * j) for i, j in itertools.combinations(range(n_individuals), 2)]
    pairs = within + between

    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pairs), max_pairs, replace=False)
        idx.sort()
        pairs = [pairs[int(i)] for i in idx]

    return pairs


# ---------------------------------------------------------------------------
# Core dating routine
# ---------------------------------------------------------------------------


MAX_HAPLOIDS = 200  # must match model's max_samples


def _subsample_gm(gm: np.ndarray, pivot_a: int, pivot_b: int,
                   max_haploids: int, rng: np.random.Generator):
    """Subsample genotype matrix to max_haploids, always keeping the pivots.

    Returns (subsampled_gm, new_pivot_a, new_pivot_b).
    """
    n = gm.shape[0]
    if n <= max_haploids:
        return gm, pivot_a, pivot_b

    # Always keep the two pivots; randomly sample the rest
    others = [i for i in range(n) if i != pivot_a and i != pivot_b]
    keep = rng.choice(others, max_haploids - 2, replace=False)
    keep = np.sort(keep)

    # Build new index array: pivots first, then context samples
    selected = np.concatenate([[pivot_a, pivot_b], keep])
    new_gm = gm[selected]
    return new_gm, 0, 1


def date_population(
    ts,
    pop_name: str,
    pop_nodes: list[int],
    model,
    arm: str,
    blocks: list,
    mask,
    device: str,
    max_pairs: int | None,
    out_dir: Path,
):
    """Run pairwise dating for a single population.

    For populations with >200 haploids, the genotype matrix is subsampled
    to 200 haploids per pair (always keeping the two pivots) so the SFS
    frequency spectrum matches what the model was trained on.
    """
    _, translate_from_genotype_matrix, *_ = _import_fastcxt()

    n_haploids = len(pop_nodes)
    n_individuals = n_haploids // 2
    needs_subsample = n_haploids > MAX_HAPLOIDS
    print(f"\n{'='*60}")
    print(f"Population: {pop_name}")
    print(f"  Individuals: {n_individuals}, Haploids: {n_haploids}")
    if needs_subsample:
        print(f"  NOTE: exceeds max_samples={MAX_HAPLOIDS}, will subsample per pair")

    if n_individuals < 2:
        print(f"  SKIPPING: need at least 2 individuals")
        return None

    # Simplify tree to this population
    print(f"  Simplifying tree sequence...")
    ts_pop = ts.simplify(samples=pop_nodes)
    gm_full = ts_pop.genotype_matrix().T  # (n_haploids, n_sites)
    positions = ts_pop.tables.sites.position.astype(np.float64)
    print(f"  Sites: {len(positions):,}, Samples: {gm_full.shape[0]}")

    # Apply accessibility mask
    if mask is not None:
        pos_int = positions.astype(np.int64)
        valid = pos_int < len(mask.mask)
        accessible = np.zeros(len(positions), dtype=bool)
        accessible[valid] = mask.mask[pos_int[valid]]
        gm_full = gm_full[:, accessible]
        positions = positions[accessible]
        print(f"  After accessibility mask: {len(positions):,} sites")

    # Build pairs
    pairs = build_pairs(n_haploids, max_pairs)
    print(f"  Pairs: {len(pairs)} (max_pairs={max_pairs})")

    block_tuples = [(b.start, b.end) for b in blocks]
    print(f"  Blocks: {len(block_tuples)}")
    print(f"  Running pairwise inference...")

    if not needs_subsample:
        # Small population: single call with full genotype matrix
        means, variances, index_map = translate_from_genotype_matrix(
            gm=gm_full,
            positions=positions,
            model=model,
            blocks=block_tuples,
            pivot_pairs=pairs,
            mutation_rate=MUTATION_RATE,
            device=device,
            batch_size=BATCH_SIZE,
            build_workers=BUILD_WORKERS,
            progress=True,
        )
    else:
        # Large population: subsample per pair so SFS has correct frequency scale.
        # Process pairs one at a time, each with a fresh subsample of context
        # haploids drawn around the two pivots.
        rng = np.random.default_rng(42)
        all_means = []
        all_vars = []
        all_idx = []

        for pi, (a, b) in enumerate(pairs):
            sub_gm, new_a, new_b = _subsample_gm(
                gm_full, a, b, MAX_HAPLOIDS, rng
            )
            m, v, idx = translate_from_genotype_matrix(
                gm=sub_gm,
                positions=positions,
                model=model,
                blocks=block_tuples,
                pivot_pairs=[(new_a, new_b)],
                mutation_rate=MUTATION_RATE,
                device=device,
                batch_size=BATCH_SIZE,
                build_workers=BUILD_WORKERS,
                progress=False,
            )
            # Remap pair index in index_map to global pair index
            idx_remapped = idx.copy()
            idx_remapped[:, 1] = pi
            all_means.append(m)
            all_vars.append(v)
            all_idx.append(idx_remapped)

            if (pi + 1) % 20 == 0 or pi + 1 == len(pairs):
                print(f"    Pair {pi+1}/{len(pairs)}")

        means = np.concatenate(all_means, axis=0)
        variances = np.concatenate(all_vars, axis=0)
        index_map = np.concatenate(all_idx, axis=0)

    # Save results
    pop_dir = out_dir / _safe_name(pop_name)
    pop_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        pop_dir / "results.npz",
        means=means,
        variances=variances,
        index_map=index_map,
        pivot_pairs=np.array(pairs),
        block_starts=np.array([b.start for b in blocks]),
        block_ends=np.array([b.end for b in blocks]),
        n_individuals=n_individuals,
        n_haploids=n_haploids,
        subsampled=needs_subsample,
    )

    # Save summary stats
    summary = {
        "population": pop_name,
        "arm": arm,
        "n_individuals": n_individuals,
        "n_haploids": n_haploids,
        "n_sites": int(len(positions)),
        "n_pairs": len(pairs),
        "n_blocks": len(blocks),
        "subsampled": needs_subsample,
        "mean_log_tmrca": float(np.nanmean(means)),
        "std_log_tmrca": float(np.nanstd(means)),
    }
    import json as json_mod
    with open(pop_dir / "summary.json", "w") as f:
        json_mod.dump(summary, f, indent=2)

    print(f"  Mean log(TMRCA): {summary['mean_log_tmrca']:.3f}")
    print(f"  Saved to {pop_dir}/")

    return summary


def _safe_name(name: str) -> str:
    """Convert population name to filesystem-safe directory name."""
    return name.lower().replace(" ", "_").replace("'", "").replace(".", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Date AG1000G populations using pairwise FastCxtModel"
    )
    ap.add_argument("--arm", type=str, default="2L",
                    choices=list(CHROMOSOME_ARMS.keys()),
                    help="Chromosome arm (default: 2L)")
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                    help="Path to trained base_anogam checkpoint")
    ap.add_argument("--trees", type=str, default=None,
                    help="Path to tsinfer tree sequence (auto-detected if not given)")
    ap.add_argument("--mask", type=str, default=str(ACCESSIBILITY_MASK),
                    help="Path to accessibility mask .npz")
    ap.add_argument("--populations", nargs="*", default=None,
                    help="Specific populations to date (default: all)")
    ap.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_POP,
                    help="Max pairs per population")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Output directory (default: results/<arm>)")
    ap.add_argument("--from-cache", action="store_true",
                    help="Skip inference, only regenerate summary from cached results")
    ap.add_argument("--list-populations", action="store_true",
                    help="List populations in the tree and exit")
    args = ap.parse_args()

    if args.checkpoint is None and not args.from_cache and not args.list_populations:
        ap.error("--checkpoint is required (or use --from-cache / --list-populations)")

    # Resolve paths
    arm = args.arm
    trees_path = args.trees or str(TREES_DIR / TREE_FILE_TEMPLATE.format(arm=arm))
    out_dir = Path(args.out_dir) if args.out_dir else EXPERIMENT_DIR / "results" / arm
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load tree sequence
    import tskit
    print(f"Loading tree sequence: {trees_path}")
    ts = tskit.load(trees_path)
    print(f"  Samples: {ts.num_samples}, Trees: {ts.num_trees:,}, "
          f"Sites: {ts.num_sites:,}, Length: {ts.sequence_length/1e6:.1f} Mb")

    # Get populations
    pop_nodes = get_populations(ts)
    pop_summary = {k: len(v) // 2 for k, v in pop_nodes.items()}
    print(f"\nPopulations ({len(pop_nodes)}):")
    for name, n_ind in sorted(pop_summary.items(), key=lambda x: -x[1]):
        print(f"  {name}: {n_ind} individuals ({len(pop_nodes[name])} haploids)")

    if args.list_populations:
        return

    # Filter to requested populations
    if args.populations:
        missing = [p for p in args.populations if p not in pop_nodes]
        if missing:
            print(f"\nWARNING: populations not found: {missing}")
            print(f"Available: {sorted(pop_nodes.keys())}")
        pop_nodes = {k: v for k, v in pop_nodes.items() if k in args.populations}

    if args.from_cache:
        print(f"\n--from-cache: skipping inference, reading from {out_dir}")
        _print_cached_summary(out_dir, pop_nodes)
        return

    # Load model
    print(f"\nLoading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device=device)

    # Load accessibility mask
    _, _, AccessibilityMask, generate_blocks = _import_fastcxt()
    mask = None
    if Path(args.mask).exists():
        mask = AccessibilityMask.from_npz(args.mask, arm)
        print(f"Accessibility mask: {mask.accessible_fraction:.3f} accessible")
    else:
        print(f"WARNING: mask not found at {args.mask}, proceeding without masking")

    # Generate blocks
    arm_length = CHROMOSOME_ARMS[arm]
    blocks = generate_blocks(arm, arm_length, block_size=BLOCK_SIZE)
    print(f"Blocks: {len(blocks)} x {BLOCK_SIZE/1e3:.0f} kb")

    # Date each population
    all_summaries = []
    for pop_name in sorted(pop_nodes.keys()):
        nodes = pop_nodes[pop_name]
        summary = date_population(
            ts, pop_name, nodes, model, arm, blocks, mask, device,
            max_pairs=args.max_pairs, out_dir=out_dir,
        )
        if summary is not None:
            all_summaries.append(summary)

    # Save combined summary
    import json as json_mod
    with open(out_dir / "all_populations_summary.json", "w") as f:
        json_mod.dump(all_summaries, f, indent=2)

    # Print final summary table
    print(f"\n{'='*70}")
    print(f"{'Population':<25} {'Ind':>5} {'Pairs':>6} {'Mean log(T)':>12} {'Std':>8}")
    print(f"{'-'*70}")
    for s in sorted(all_summaries, key=lambda x: x["mean_log_tmrca"]):
        print(f"{s['population']:<25} {s['n_individuals']:>5} "
              f"{s['n_pairs']:>6} {s['mean_log_tmrca']:>12.3f} "
              f"{s['std_log_tmrca']:>8.3f}")
    print(f"{'='*70}")
    print(f"Results saved to {out_dir}/")


def _print_cached_summary(out_dir: Path, pop_nodes: dict):
    """Print summary from cached results."""
    import json as json_mod

    summary_path = out_dir / "all_populations_summary.json"
    if summary_path.exists():
        summaries = json_mod.loads(summary_path.read_text())
        print(f"\n{'Population':<25} {'Ind':>5} {'Pairs':>6} {'Mean log(T)':>12}")
        print(f"{'-'*55}")
        for s in sorted(summaries, key=lambda x: x["mean_log_tmrca"]):
            print(f"{s['population']:<25} {s['n_individuals']:>5} "
                  f"{s['n_pairs']:>6} {s['mean_log_tmrca']:>12.3f}")
    else:
        print(f"  No cached summary found at {summary_path}")
        # Check individual population dirs
        for pop_dir in sorted(out_dir.iterdir()):
            if pop_dir.is_dir() and (pop_dir / "summary.json").exists():
                s = json_mod.loads((pop_dir / "summary.json").read_text())
                print(f"  {s['population']}: mean log(TMRCA) = {s['mean_log_tmrca']:.3f}")


if __name__ == "__main__":
    main()
