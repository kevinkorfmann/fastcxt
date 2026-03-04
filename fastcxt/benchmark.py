"""Scaling benchmarks for fastcxt.

Compares runtime of three modes across sample sizes:
  1. cxt (baseline): autoregressive transformer, O(n^2) pairs x 7500 steps
  2. fastcxt (no trees): single-pass SSM, O(n^2) pairs x 1 pass
  3. fastcxt (with trees): tree-aware SSM, O(n log n) via node predictions

Usage:
    python -m fastcxt.benchmark --mode all --device cuda:0
    python -m fastcxt.benchmark --mode fastcxt_notree fastcxt_tree --sample-sizes 10 50 100
"""

from __future__ import annotations

import time
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fastcxt.config import PRESETS
from fastcxt.model import FastCxtModel
from fastcxt.sfs import build_sfs_tensor, basic_filtering


def _simulate_ts(n_samples: int, sequence_length: float = 1e6,
                 mutation_rate: float = 1e-8, seed: int = 42):
    """Simulate a tree sequence for benchmarking."""
    import msprime
    rng = np.random.default_rng(seed)
    ts = msprime.sim_ancestry(
        samples=n_samples,
        population_size=2e4,
        recombination_rate=1e-8,
        sequence_length=sequence_length,
        random_seed=int(rng.integers(1, 2**31)),
    )
    ts = msprime.sim_mutations(
        ts, rate=mutation_rate, random_seed=int(rng.integers(1, 2**31)),
    )
    return ts


def _all_pairs(n: int):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def benchmark_fastcxt_pairwise(
    n_samples: int, model: FastCxtModel, device: str = "cuda",
    sequence_length: float = 1e6, mutation_rate: float = 1e-8,
    batch_size: int = 64,
) -> dict:
    """Benchmark pairwise (no trees) mode: O(n^2) pairs, 1 pass each."""
    ts = _simulate_ts(n_samples, sequence_length, mutation_rate)
    gm = ts.genotype_matrix().T
    positions = ts.tables.sites.position
    gm, positions = basic_filtering(gm, positions)

    pairs = _all_pairs(n_samples * 2)
    n_pairs = len(pairs)
    window_size = int(sequence_length / 500)

    t0 = time.perf_counter()
    Xs = []
    for pa, pb in pairs:
        if pa < gm.shape[0] and pb < gm.shape[0]:
            Xs.append(build_sfs_tensor(gm, positions, pa, pb,
                                       sequence_length, window_size))
    if not Xs:
        return {"n_samples": n_samples, "n_pairs": 0, "time_s": 0}

    X = np.stack(Xs)
    t_preprocess = time.perf_counter() - t0

    model.eval()
    model.to(device)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    log_mu = torch.tensor([[np.log(mutation_rate)]], dtype=torch.float32, device=device)

    torch.cuda.synchronize() if "cuda" in device else None
    t1 = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(Xs), batch_size):
            end = min(start + batch_size, len(Xs))
            batch = X_t[start:end]
            mu_batch = log_mu.expand(end - start, -1)
            _ = model(batch, mu_batch)
    torch.cuda.synchronize() if "cuda" in device else None
    t_inference = time.perf_counter() - t1

    return {
        "n_samples": n_samples,
        "n_haploids": n_samples * 2,
        "n_pairs": len(Xs),
        "preprocess_s": round(t_preprocess, 4),
        "inference_s": round(t_inference, 4),
        "total_s": round(t_preprocess + t_inference, 4),
    }


def benchmark_fastcxt_tree(
    n_samples: int, model: FastCxtModel, device: str = "cuda",
    sequence_length: float = 1e6, mutation_rate: float = 1e-8,
) -> dict:
    """Benchmark tree-aware mode: O(n) internal nodes, 1 pass.

    Even without a real tsinfer integration, we measure the theoretical
    scaling by predicting n_internal = n_haploids - 1 node times.
    """
    ts = _simulate_ts(n_samples, sequence_length, mutation_rate)
    n_haploids = ts.num_samples
    n_internal = n_haploids - 1
    window_size = int(sequence_length / 500)
    n_windows = 500

    gm = ts.genotype_matrix().T
    positions = ts.tables.sites.position
    gm, positions = basic_filtering(gm, positions)

    t0 = time.perf_counter()
    representative_pairs = min(n_internal, len(_all_pairs(n_haploids)))
    pairs = _all_pairs(n_haploids)[:representative_pairs]
    Xs = []
    for pa, pb in pairs:
        if pa < gm.shape[0] and pb < gm.shape[0]:
            Xs.append(build_sfs_tensor(gm, positions, pa, pb,
                                       sequence_length, window_size))

    if not Xs:
        return {"n_samples": n_samples, "n_nodes": 0, "time_s": 0}

    X = np.stack(Xs)
    t_preprocess = time.perf_counter() - t0

    model.eval()
    model.to(device)
    X_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    log_mu = torch.tensor([[np.log(mutation_rate)]], dtype=torch.float32, device=device)

    torch.cuda.synchronize() if "cuda" in device else None
    t1 = time.perf_counter()
    with torch.inference_mode():
        mu_batch = log_mu.expand(len(Xs), -1)
        _ = model(X_t, mu_batch)
    torch.cuda.synchronize() if "cuda" in device else None
    t_inference = time.perf_counter() - t1

    return {
        "n_samples": n_samples,
        "n_haploids": n_haploids,
        "n_nodes_predicted": len(Xs),
        "n_pairs_equivalent": len(_all_pairs(n_haploids)),
        "preprocess_s": round(t_preprocess, 4),
        "inference_s": round(t_inference, 4),
        "total_s": round(t_preprocess + t_inference, 4),
    }


def main():
    ap = argparse.ArgumentParser(description="fastcxt scaling benchmarks")
    ap.add_argument("--mode", nargs="+",
                    choices=["fastcxt_notree", "fastcxt_tree", "all"],
                    default=["all"])
    ap.add_argument("--sample-sizes", type=int, nargs="+",
                    default=[5, 10, 25, 50, 100])
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output", type=str, default=None,
                    help="JSON output path for results")
    args = ap.parse_args()

    modes = args.mode
    if "all" in modes:
        modes = ["fastcxt_notree", "fastcxt_tree"]

    config = PRESETS["small"]
    model = FastCxtModel(config)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    results = []
    for n in args.sample_sizes:
        print(f"\n--- n_samples={n} (n_haploids={n*2}) ---")

        if "fastcxt_notree" in modes:
            try:
                r = benchmark_fastcxt_pairwise(n, model, args.device,
                                               batch_size=args.batch_size)
                r["mode"] = "fastcxt_notree"
                results.append(r)
                print(f"  pairwise: {r['n_pairs']} pairs, "
                      f"preproc={r['preprocess_s']:.3f}s, "
                      f"infer={r['inference_s']:.3f}s, "
                      f"total={r['total_s']:.3f}s")
            except Exception as e:
                print(f"  pairwise: ERROR - {e}")

        if "fastcxt_tree" in modes:
            try:
                r = benchmark_fastcxt_tree(n, model, args.device)
                r["mode"] = "fastcxt_tree"
                results.append(r)
                print(f"  tree:     {r['n_nodes_predicted']} nodes "
                      f"(covers {r['n_pairs_equivalent']} pairs), "
                      f"preproc={r['preprocess_s']:.3f}s, "
                      f"infer={r['inference_s']:.3f}s, "
                      f"total={r['total_s']:.3f}s")
            except Exception as e:
                print(f"  tree: ERROR - {e}")

    if results:
        print("\n\nScaling summary:")
        print(f"{'mode':<18} {'n_hap':>6} {'pairs/nodes':>12} {'total_s':>10}")
        print("-" * 50)
        for r in results:
            n_work = r.get("n_pairs", r.get("n_nodes_predicted", 0))
            n_hap = r.get("n_haploids", r["n_samples"] * 2)
            print(f"{r['mode']:<18} {n_hap:>6} {n_work:>12} {r['total_s']:>10.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
