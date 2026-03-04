"""TimeAtlas -- a data structure for at-scale coalescence time maps.

The TimeAtlas stores predicted pairwise TMRCA landscapes across an entire
genome (or set of chromosome arms), organized for efficient queries:

  - "What is the TMRCA between samples i and j across chromosome 2L?"
  - "What are all pairwise TMRCAs at a specific genomic window?"
  - "Which pairs have the deepest/shallowest coalescence at position X?"
  - "What is the genome-wide mean TMRCA between populations A and B?"

Storage layout:
    Each chromosome arm stores a dense (n_pairs, n_windows) matrix of
    predicted log-TMRCA means and variances, along with metadata mapping
    pair indices to sample IDs and windows to genomic coordinates.

Serialization:
    Saves to a directory of .npz files (one per arm) plus a manifest.json.
    Designed for zarr or HDF5 backends when datasets get very large.

Usage:
    atlas = TimeAtlas()
    atlas.add_arm("2L", means, variances, pairs, blocks, mutation_rate=3.5e-9)
    atlas.save("my_atlas/")

    atlas = TimeAtlas.load("my_atlas/")
    tmrca = atlas.query_pair("2L", sample_a=0, sample_b=5)
    window_view = atlas.query_window("2L", position_bp=5_000_000)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Per-arm data container
# ---------------------------------------------------------------------------

@dataclass
class ArmData:
    """TMRCA predictions for one chromosome arm."""

    arm: str
    means: np.ndarray           # (n_pairs, n_windows) log-TMRCA means
    variances: np.ndarray       # (n_pairs, n_windows) log-TMRCA variances
    pairs: np.ndarray           # (n_pairs, 2) sample index pairs
    window_starts: np.ndarray   # (n_windows,) genomic start positions
    window_size: int
    mutation_rate: float

    @property
    def n_pairs(self) -> int:
        return self.means.shape[0]

    @property
    def n_windows(self) -> int:
        return self.means.shape[1]

    @property
    def arm_length(self) -> int:
        return int(self.window_starts[-1] + self.window_size)

    def pair_index(self, sample_a: int, sample_b: int) -> int | None:
        """Find the row index for a given pair, or None if absent."""
        a, b = min(sample_a, sample_b), max(sample_a, sample_b)
        for i, (pa, pb) in enumerate(self.pairs):
            if pa == a and pb == b:
                return i
        return None

    def window_at(self, position_bp: int) -> int:
        """Map a base-pair position to a window index."""
        idx = int(position_bp // self.window_size)
        return min(idx, self.n_windows - 1)


# ---------------------------------------------------------------------------
# TimeAtlas
# ---------------------------------------------------------------------------

class TimeAtlas:
    """Genome-wide collection of pairwise TMRCA predictions.

    Organizes results by chromosome arm and provides query methods for
    extracting TMRCA profiles, window snapshots, and summary statistics.
    """

    def __init__(self):
        self.arms: dict[str, ArmData] = {}
        self.metadata: dict = {}

    # --- building ---

    def add_arm(
        self,
        arm: str,
        means: np.ndarray,
        variances: np.ndarray,
        pairs: np.ndarray,
        block_starts: np.ndarray | list | None = None,
        window_size: int = 2000,
        mutation_rate: float = 1e-8,
    ) -> None:
        n_windows = means.shape[1]
        if block_starts is None:
            block_starts = np.arange(n_windows) * window_size
        self.arms[arm] = ArmData(
            arm=arm,
            means=np.asarray(means, dtype=np.float32),
            variances=np.asarray(variances, dtype=np.float32),
            pairs=np.asarray(pairs, dtype=np.int32),
            window_starts=np.asarray(block_starts, dtype=np.int64),
            window_size=window_size,
            mutation_rate=mutation_rate,
        )

    # --- queries ---

    def query_pair(
        self, arm: str, sample_a: int, sample_b: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Get (mean, variance) TMRCA profile for one pair across an arm.

        Returns (means_1d, variances_1d) or None if pair not found.
        """
        ad = self.arms.get(arm)
        if ad is None:
            return None
        idx = ad.pair_index(sample_a, sample_b)
        if idx is None:
            return None
        return ad.means[idx], ad.variances[idx]

    def query_window(
        self, arm: str, position_bp: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Get all pairwise TMRCAs at one genomic position.

        Returns (pairs, means_at_window, variances_at_window) or None.
        """
        ad = self.arms.get(arm)
        if ad is None:
            return None
        w = ad.window_at(position_bp)
        return ad.pairs, ad.means[:, w], ad.variances[:, w]

    def query_region(
        self, arm: str, start_bp: int, end_bp: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Get all pairwise TMRCAs across a genomic region.

        Returns (pairs, means_slice, variances_slice) or None.
        """
        ad = self.arms.get(arm)
        if ad is None:
            return None
        w_start = ad.window_at(start_bp)
        w_end = ad.window_at(end_bp) + 1
        return ad.pairs, ad.means[:, w_start:w_end], ad.variances[:, w_start:w_end]

    def mean_tmrca(self, arm: str) -> np.ndarray:
        """Per-pair genome-wide mean TMRCA (in natural scale) for an arm."""
        ad = self.arms[arm]
        return np.exp(ad.means).mean(axis=1)

    def deepest_pairs(self, arm: str, position_bp: int, k: int = 10) -> np.ndarray:
        """Return the k pairs with the deepest TMRCA at a position."""
        ad = self.arms[arm]
        w = ad.window_at(position_bp)
        order = np.argsort(-ad.means[:, w])[:k]
        return ad.pairs[order]

    def shallowest_pairs(self, arm: str, position_bp: int, k: int = 10) -> np.ndarray:
        """Return the k pairs with the shallowest TMRCA at a position."""
        ad = self.arms[arm]
        w = ad.window_at(position_bp)
        order = np.argsort(ad.means[:, w])[:k]
        return ad.pairs[order]

    # --- summary stats ---

    @property
    def total_pairs(self) -> int:
        return sum(ad.n_pairs for ad in self.arms.values())

    @property
    def total_windows(self) -> int:
        return sum(ad.n_windows for ad in self.arms.values())

    def summary(self) -> dict:
        return {
            "n_arms": len(self.arms),
            "arms": list(self.arms.keys()),
            "total_pairs": self.total_pairs,
            "total_windows": self.total_windows,
            "per_arm": {
                arm: {
                    "n_pairs": ad.n_pairs,
                    "n_windows": ad.n_windows,
                    "arm_length_bp": ad.arm_length,
                    "mean_log_tmrca": float(ad.means.mean()),
                    "mutation_rate": ad.mutation_rate,
                }
                for arm, ad in self.arms.items()
            },
        }

    # --- iteration ---

    def iter_pairs(self, arm: str) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
        """Iterate over (sample_a, sample_b, means, variances) for an arm."""
        ad = self.arms[arm]
        for i, (a, b) in enumerate(ad.pairs):
            yield int(a), int(b), ad.means[i], ad.variances[i]

    # --- serialization ---

    def save(self, directory: str | Path) -> None:
        """Save the atlas to a directory of .npz files + manifest.json."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)

        manifest = {"version": 1, "arms": {}}
        for arm, ad in self.arms.items():
            fname = f"{arm}.npz"
            np.savez_compressed(
                d / fname,
                means=ad.means,
                variances=ad.variances,
                pairs=ad.pairs,
                window_starts=ad.window_starts,
            )
            manifest["arms"][arm] = {
                "file": fname,
                "window_size": ad.window_size,
                "mutation_rate": ad.mutation_rate,
                "n_pairs": ad.n_pairs,
                "n_windows": ad.n_windows,
            }

        manifest["metadata"] = self.metadata
        with open(d / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def load(cls, directory: str | Path) -> TimeAtlas:
        """Load an atlas from a saved directory."""
        d = Path(directory)
        with open(d / "manifest.json") as f:
            manifest = json.load(f)

        atlas = cls()
        atlas.metadata = manifest.get("metadata", {})

        for arm, info in manifest["arms"].items():
            data = np.load(d / info["file"])
            atlas.arms[arm] = ArmData(
                arm=arm,
                means=data["means"],
                variances=data["variances"],
                pairs=data["pairs"],
                window_starts=data["window_starts"],
                window_size=info["window_size"],
                mutation_rate=info["mutation_rate"],
            )
        return atlas

    def __repr__(self) -> str:
        arms_str = ", ".join(
            f"{arm}({ad.n_pairs}p x {ad.n_windows}w)"
            for arm, ad in self.arms.items()
        )
        return f"TimeAtlas({arms_str})"
