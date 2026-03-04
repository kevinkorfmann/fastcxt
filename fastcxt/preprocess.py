"""Preprocessing pipeline for fastcxt.

Clean separation of concerns:
  - SFS feature building   (delegated to fastcxt.sfs)
  - TMRCA target extraction (windowed span-averages)
  - Pair selection          (deterministic, reproducible)
  - Missing-data handling   (accessibility masks / bitmasks)
  - Tree topology features  (optional, delegated to fastcxt.tree_utils)
  - Parallel I/O            (multiprocessing with dataclass job specs)

Usage:
    python -m fastcxt.preprocess --base-dir ./sims --out-subdir processed
    python -m fastcxt.preprocess --base-dir ./sims --extract-trees --accessibility-mask mask.npz
"""

from __future__ import annotations

import os
import json
import argparse
import pathlib
from dataclasses import dataclass
from typing import List

import numpy as np
import tskit
import multiprocessing as mp

from fastcxt.sfs import build_sfs_tensor, basic_filtering


# ---------------------------------------------------------------------------
# TMRCA targets: exact length-weighted window averages
# ---------------------------------------------------------------------------

def windowed_tmrca(
    ts: tskit.TreeSequence,
    sample_a: int,
    sample_b: int,
    window_size: int,
    sequence_length: int | None = None,
) -> np.ndarray:
    """Exact span-weighted average TMRCA per genomic window for one pair."""
    if sequence_length is None:
        sequence_length = int(ts.sequence_length)

    lefts, rights, vals = [], [], []
    for tree in ts.trees():
        l, r = tree.interval
        m = tree.mrca(sample_a, sample_b)
        vals.append(np.nan if m == tskit.NULL else tree.time(m))
        lefts.append(l)
        rights.append(r)

    lefts = np.asarray(lefts, dtype=np.int64)
    rights = np.asarray(rights, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)

    starts = np.arange(0, sequence_length, window_size, dtype=np.int64)
    ends = starts + window_size
    n_w = len(starts)
    numer = np.zeros(n_w, dtype=np.float64)

    si, wi = 0, 0
    while si < len(lefts) and wi < n_w:
        a = max(lefts[si], starts[wi])
        b = min(rights[si], ends[wi])
        if b > a:
            numer[wi] += (b - a) * vals[si]
        if rights[si] <= ends[wi]:
            si += 1
        else:
            wi += 1
    return numer / window_size


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def choose_pairs(n_samples: int, num_pairs: int, seed: int) -> np.ndarray:
    """Deterministically choose pairs of haploid sample indices."""
    all_pairs = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]
    if len(all_pairs) <= num_pairs:
        return np.asarray(all_pairs, dtype=np.int32)
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(all_pairs), size=num_pairs, replace=False)
    return np.asarray([all_pairs[k] for k in sel], dtype=np.int32)


# ---------------------------------------------------------------------------
# Missing data / accessibility mask
# ---------------------------------------------------------------------------

def apply_accessibility_mask(
    gm: np.ndarray,
    positions: np.ndarray,
    mask: np.ndarray,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Zero out genotypes in inaccessible regions.

    Parameters
    ----------
    mask : 1-d bool array of length ``sequence_length``.
           True = accessible, False = masked.
    """
    pos_int = positions.astype(np.int64)
    accessible = mask[np.clip(pos_int, 0, len(mask) - 1)]
    return gm[:, accessible], positions[accessible]


# ---------------------------------------------------------------------------
# Process one tree sequence
# ---------------------------------------------------------------------------

def process_one_ts(
    ts: tskit.TreeSequence,
    pairs: np.ndarray,
    window_size: int = 2000,
    sequence_length: int | None = None,
    accessibility_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SFS features and log-TMRCA targets for a set of pairs.

    Returns
    -------
    X : (P, 2, n_scales, n_windows, n_samples)  float16
    y : (P, n_windows)  float16  (log-TMRCA)
    """
    if sequence_length is None:
        sequence_length = int(ts.sequence_length)

    gm = ts.genotype_matrix().T
    positions = ts.tables.sites.position
    gm, positions = basic_filtering(gm, positions)

    if accessibility_mask is not None:
        gm, positions = apply_accessibility_mask(gm, positions, accessibility_mask, sequence_length)

    P = len(pairs)
    n_windows = int(np.ceil(sequence_length / window_size))

    probe = build_sfs_tensor(gm, positions, int(pairs[0][0]), int(pairs[0][1]),
                             float(sequence_length), window_size)
    X = np.empty((P,) + probe.shape, dtype=np.float16)
    y = np.empty((P, n_windows), dtype=np.float16)

    X[0] = probe
    y0 = windowed_tmrca(ts, int(pairs[0][0]), int(pairs[0][1]), window_size, sequence_length)
    y[0] = np.log(np.clip(y0, 1e-10, None)).astype(np.float16)

    for k in range(1, P):
        pa, pb = int(pairs[k][0]), int(pairs[k][1])
        X[k] = build_sfs_tensor(gm, positions, pa, pb, float(sequence_length), window_size)
        yk = windowed_tmrca(ts, pa, pb, window_size, sequence_length)
        y[k] = np.log(np.clip(yk, 1e-10, None)).astype(np.float16)

    return X, y


# ---------------------------------------------------------------------------
# Mutation rate estimation
# ---------------------------------------------------------------------------

def estimate_mutation_rate(ts: tskit.TreeSequence) -> float:
    """Estimate per-bp per-generation mutation rate from a tree sequence."""
    total_branch = sum(
        tree.total_branch_length * (tree.interval.right - tree.interval.left)
        for tree in ts.trees()
    )
    return ts.num_mutations / max(total_branch, 1.0)


# ---------------------------------------------------------------------------
# File discovery and train/test splitting
# ---------------------------------------------------------------------------

def find_ts_files(root: pathlib.Path) -> list[pathlib.Path]:
    exts = {".trees", ".ts", ".tsk", ".tskit"}
    out = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for fn in filenames:
            p = pathlib.Path(dirpath) / fn
            if p.suffix.lower() in exts:
                out.append(p)
    out.sort(key=lambda p: str(p.relative_to(root)))
    return out


def scenario_from_path(p: pathlib.Path, base: pathlib.Path) -> str:
    rel = p.relative_to(base).parent
    return str(rel) if str(rel) != "." else "default"


def deterministic_split(
    files: list[pathlib.Path],
    base: pathlib.Path,
    train_ratio: float = 0.9,
    seed: int = 12345,
) -> set[int]:
    """Split files into train/test, stratified by scenario directory."""
    rng = np.random.default_rng(seed)
    by_scenario: dict[str, list[int]] = {}
    for i, f in enumerate(files):
        scn = scenario_from_path(f, base)
        by_scenario.setdefault(scn, []).append(i)

    train_idx: set[int] = set()
    for idxs in by_scenario.values():
        arr = np.array(idxs, dtype=int)
        rng.shuffle(arr)
        n = len(arr)
        cut = max(1, int(np.floor(train_ratio * n)))
        if n >= 2 and cut == n:
            cut = n - 1
        train_idx.update(arr[:cut].tolist())
    return train_idx


# ---------------------------------------------------------------------------
# Worker job specification (dataclass, not tuple)
# ---------------------------------------------------------------------------

@dataclass
class PreprocessJob:
    idx: int
    ts_path: str
    base_dir: str
    out_root: str
    split: str
    window_size: int
    sequence_length: int
    num_pairs: int
    global_seed: int
    skip_existing: bool
    simplify_n: int
    extract_trees: bool
    mutation_rate_override: float | None
    accessibility_mask_path: str | None


def _run_job(job: PreprocessJob) -> str:
    f = pathlib.Path(job.ts_path)
    base = pathlib.Path(job.base_dir)
    out_root = pathlib.Path(job.out_root)

    scenario = scenario_from_path(f, base)
    short_id = f"{f.stem}_i{job.idx}"
    out_dir = out_root / job.split / scenario / short_id
    out_dir.mkdir(parents=True, exist_ok=True)

    if job.skip_existing and (out_dir / "X.npy").exists() and (out_dir / "y.npy").exists():
        return f"[skip] {job.split}/{scenario}/{short_id}"

    try:
        ts = tskit.load(str(f))
        if job.simplify_n > 0 and ts.num_samples > job.simplify_n:
            ts = ts.simplify(samples=list(range(job.simplify_n)))
    except Exception as e:
        return f"[error] load {f}: {e}"

    acc_mask = None
    if job.accessibility_mask_path:
        try:
            data = np.load(job.accessibility_mask_path)
            key = list(data.keys())[0]
            full_mask = data[key].astype(bool)
            rng = np.random.default_rng(
                int(np.int64(job.global_seed) + np.int64(job.idx) * 7919)
            )
            if len(full_mask) > job.sequence_length:
                start = rng.integers(0, len(full_mask) - job.sequence_length)
                acc_mask = full_mask[start:start + job.sequence_length]
            else:
                acc_mask = full_mask[:job.sequence_length]
        except Exception as e:
            return f"[warn] mask load failed: {e}"

    pair_seed = int(np.int64(job.global_seed) + np.int64(job.idx) * 10007)

    try:
        pairs = choose_pairs(ts.num_samples, job.num_pairs, seed=pair_seed)
        X, y = process_one_ts(
            ts, pairs, job.window_size, job.sequence_length,
            accessibility_mask=acc_mask,
        )

        np.save(out_dir / "X.npy", X)
        np.save(out_dir / "y.npy", y)
        np.save(out_dir / "pairs.npy", pairs)

        mu_rate = job.mutation_rate_override
        if mu_rate is None:
            mu_rate = estimate_mutation_rate(ts)

        meta = {
            "source_file": str(f.relative_to(base)),
            "scenario": scenario,
            "split": job.split,
            "num_pairs": int(len(pairs)),
            "window_size": job.window_size,
            "sequence_length": job.sequence_length,
            "num_samples": int(ts.num_samples),
            "mutation_rate": float(mu_rate),
            "has_accessibility_mask": acc_mask is not None,
        }
        with open(out_dir / "meta.json", "w") as fp:
            json.dump(meta, fp, indent=2)

        if job.extract_trees:
            try:
                from fastcxt.tree_utils import extract_topology_features
                tf = extract_topology_features(
                    ts, n_windows=X.shape[3],
                    window_size=job.window_size,
                    max_internal=ts.num_samples - 1,
                )
                tf_all = np.tile(tf[np.newaxis], (len(pairs), 1, 1))
                np.save(out_dir / "tree_feats.npy", tf_all.astype(np.float32))
            except Exception as e:
                return f"[warn] tree extraction failed for {f}: {e}"

        return f"[ok] {job.split}/{scenario}/{short_id}  X{X.shape}  y{y.shape}"
    except Exception as e:
        return f"[error] {f}: {e}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Preprocess tree sequences into training data for fastcxt",
    )
    ap.add_argument("--base-dir", required=True,
                    help="Root dir containing simulated .trees files")
    ap.add_argument("--out-subdir", default="processed",
                    help="Output subdirectory name under base-dir")
    ap.add_argument("--window-size", type=int, default=2000)
    ap.add_argument("--sequence-length", type=int, default=1_000_000)
    ap.add_argument("--num-pairs", type=int, default=200)
    ap.add_argument("--simplify-n", type=int, default=0,
                    help="Simplify to first N samples (0 = keep all)")
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--global-seed", type=int, default=12345)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--num-workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--extract-trees", action="store_true",
                    help="Compute tsinfer topology features")
    ap.add_argument("--mutation-rate", type=float, default=None,
                    help="Override mutation rate (default: estimate from tree sequence)")
    ap.add_argument("--accessibility-mask", type=str, default=None,
                    help="Path to .npz accessibility mask (for missing-data regions)")
    args = ap.parse_args()

    base = pathlib.Path(args.base_dir)
    out_root = base / args.out_subdir
    out_root.mkdir(parents=True, exist_ok=True)

    files = find_ts_files(base)
    if not files:
        print(f"No tree-sequence files found under {base}")
        return
    print(f"Found {len(files)} tree-sequence files")

    train_membership = deterministic_split(files, base, args.train_ratio,
                                           seed=args.global_seed)

    jobs = []
    for idx, f in enumerate(files):
        jobs.append(PreprocessJob(
            idx=idx,
            ts_path=str(f),
            base_dir=str(base),
            out_root=str(out_root),
            split="train" if idx in train_membership else "test",
            window_size=args.window_size,
            sequence_length=args.sequence_length,
            num_pairs=args.num_pairs,
            global_seed=args.global_seed,
            skip_existing=args.skip_existing,
            simplify_n=args.simplify_n,
            extract_trees=args.extract_trees,
            mutation_rate_override=args.mutation_rate,
            accessibility_mask_path=args.accessibility_mask,
        ))

    with mp.Pool(processes=args.num_workers) as pool:
        for msg in pool.imap_unordered(_run_job, jobs, chunksize=1):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
