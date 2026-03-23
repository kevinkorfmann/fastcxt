#!/usr/bin/env python3
"""Run inference, build TimeAtlas, and generate visualizations.

Uses the checkpoint from 6 epochs of training (epoch=5-step=2112.ckpt).
Steps 4-6 from the quickstart documentation.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import tskit
from lightning.pytorch import LightningModule

from fastcxt.config import PRESETS
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_ts
from fastcxt.atlas import TimeAtlas

# Paths
EXPERIMENT_DIR = Path(__file__).resolve().parent
CHECKPOINT = EXPERIMENT_DIR / "lightning_logs/version_2/checkpoints/epoch=5-step=2112.ckpt"
TS_FILE = EXPERIMENT_DIR / "sims/constant/ts_00000000.trees"
ATLAS_DIR = EXPERIMENT_DIR / "my_atlas"
FIGURES_DIR = EXPERIMENT_DIR / "figures"


def main():
    print("Loading model from checkpoint (6 epochs trained)...")
    lit_model: LightningModule = LitFastCxt.load_from_checkpoint(
        str(CHECKPOINT),
        map_location="cpu",
    )
    model = lit_model.model

    print("Loading tree sequence...")
    ts = tskit.load(str(TS_FILE))
    n_samples = ts.num_samples

    # Use a few representative pairs
    pivot_pairs = [(0, 1), (0, 2), (1, 2)]
    blocks = [(0, int(ts.sequence_length))]

    print("Running inference...")
    means, variances, index_map = translate_from_ts(
        ts, model,
        blocks=blocks,
        pivot_pairs=pivot_pairs,
        mutation_rate=1e-8,
        device="cuda:0" if __import__("torch").cuda.is_available() else "cpu",
    )

    # index_map: (N, 2) -> [block_idx, pivot_idx]; we want pairs array (N, 2)
    pairs_arr = np.array(
        [pivot_pairs[int(i)] for _, i in index_map],
        dtype=np.int32,
    )

    print("Building TimeAtlas...")
    atlas = TimeAtlas()
    atlas.add_arm(
        "chr1",
        means, variances, pairs_arr,
        window_size=2000,
        mutation_rate=1e-8,
    )
    ATLAS_DIR.mkdir(parents=True, exist_ok=True)
    atlas.save(str(ATLAS_DIR))
    print(f"  Saved to {ATLAS_DIR}")

    # Query demo
    loaded = TimeAtlas.load(str(ATLAS_DIR))
    m, v = loaded.query_pair("chr1", sample_a=0, sample_b=1)
    print(f"  Query pair (0,1): {len(m)} windows, mean[0]={m[0]:.4f}")

    print("Generating experiment figures...")
    subprocess.run(
        [sys.executable, str(EXPERIMENT_DIR / "plot_experiment_figures.py")],
        check=True,
        cwd=str(EXPERIMENT_DIR),
    )
    print(f"  Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
