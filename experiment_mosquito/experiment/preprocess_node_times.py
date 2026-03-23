#!/usr/bin/env python3
"""Extract node-time features and targets from tree sequences.

Scans a sims directory for .trees files and produces numpy arrays
ready for NodeTimeModel training.  Use --use-tsinfer to extract
topology features from tsinfer-inferred trees (targets are always
ground-truth node times).

Usage:
    python preprocess_node_times.py
    python preprocess_node_times.py --use-tsinfer --out-dir outputs/node_features_tsinfer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tskit
from tqdm.auto import tqdm

from fastcxt.tree_utils import (
    FEATS_PER_NODE,
    extract_topology_features,
    extract_node_times,
)
from fastcxt.preprocess import estimate_mutation_rate
from config import (
    SIMS_DIR, OUTPUTS_DIR, MAX_SAMPLES, N_WINDOWS, WINDOW_SIZE,
)


def preprocess(
    sims_dir: Path,
    out_dir: Path,
    max_samples: int = MAX_SAMPLES,
    use_tsinfer: bool = False,
):
    ts_files = sorted(sims_dir.rglob("*.trees"))
    if not ts_files:
        raise FileNotFoundError(f"No .trees files found under {sims_dir}")
    print(f"Preprocessing {len(ts_files)} tree sequences (tsinfer={use_tsinfer}) ...")

    max_internal = max_samples - 1
    tree_feat_dim = max_internal * FEATS_PER_NODE
    N = len(ts_files)

    all_feats = np.zeros((N, N_WINDOWS, tree_feat_dim), dtype=np.float32)
    all_times = np.zeros((N, N_WINDOWS, max_internal), dtype=np.float32)
    all_mu = np.zeros(N, dtype=np.float32)
    all_masks = np.zeros((N, max_internal), dtype=np.float32)

    for i, f in enumerate(tqdm(ts_files, desc="Extracting")):
        ts = tskit.load(str(f))

        # Features: from tsinfer-inferred or ground-truth trees
        if use_tsinfer:
            import tsinfer
            sd = tsinfer.SampleData.from_tree_sequence(ts, use_sites_time=False)
            inferred_ts = tsinfer.infer(sd)
            all_feats[i] = extract_topology_features(
                inferred_ts, n_windows=N_WINDOWS,
                window_size=WINDOW_SIZE, max_internal=max_internal,
            )
        else:
            all_feats[i] = extract_topology_features(
                ts, n_windows=N_WINDOWS,
                window_size=WINDOW_SIZE, max_internal=max_internal,
            )

        # Targets: always ground-truth node times
        all_times[i] = extract_node_times(
            ts, n_windows=N_WINDOWS,
            window_size=WINDOW_SIZE, max_internal=max_internal,
        )
        all_mu[i] = np.log(estimate_mutation_rate(ts))
        all_masks[i, : ts.num_samples - 1] = 1.0

    # Shuffle and split 90/10
    rng = np.random.default_rng(42)
    idx = rng.permutation(N)
    all_feats, all_times = all_feats[idx], all_times[idx]
    all_mu, all_masks = all_mu[idx], all_masks[idx]

    split = int(0.9 * N)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_feats.npy", all_feats[:split])
    np.save(out_dir / "train_times.npy", all_times[:split])
    np.save(out_dir / "train_mu.npy", all_mu[:split])
    np.save(out_dir / "train_masks.npy", all_masks[:split])
    np.save(out_dir / "val_feats.npy", all_feats[split:])
    np.save(out_dir / "val_times.npy", all_times[split:])
    np.save(out_dir / "val_mu.npy", all_mu[split:])
    np.save(out_dir / "val_masks.npy", all_masks[split:])

    print(f"  Saved {split} train / {N - split} val samples to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims-dir", type=str, default=str(SIMS_DIR))
    ap.add_argument("--out-dir", type=str, default=str(OUTPUTS_DIR / "node_features"))
    ap.add_argument("--max-samples", type=int, default=MAX_SAMPLES)
    ap.add_argument("--use-tsinfer", action="store_true")
    args = ap.parse_args()

    preprocess(
        Path(args.sims_dir), Path(args.out_dir),
        max_samples=args.max_samples, use_tsinfer=args.use_tsinfer,
    )


if __name__ == "__main__":
    main()
