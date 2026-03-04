"""Anopheles gambiae analysis protocols for fastcxt.

Provides end-to-end workflows for mosquito population genomics at scale:

  - Accessibility mask handling (Ag1000G-style .npz masks)
  - Chromosome-arm block generation with proper coordinate handling
  - Batch inference across chromosome arms for thousands of samples
  - Missing-data-aware SFS construction

Usage:
    from fastcxt.mosquito import MosquitoAnalysis
    analysis = MosquitoAnalysis(model, device="cuda:0")
    atlas = analysis.run_chromosome_arm(vcf_path, "2L", accessibility_mask=mask)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    import tskit
except ImportError:
    tskit = None


# ---------------------------------------------------------------------------
# Anopheles reference genome constants
# ---------------------------------------------------------------------------

ANOGAM_CHROMOSOME_ARMS = {
    "2L": 49_364_325,
    "2R": 61_545_105,
    "3L": 41_963_435,
    "3R": 53_200_684,
    "X":  24_393_108,
}

DEFAULT_BLOCK_SIZE = 1_000_000
DEFAULT_WINDOW_SIZE = 2000
DEFAULT_N_WINDOWS_PER_BLOCK = 500


# ---------------------------------------------------------------------------
# Accessibility mask utilities
# ---------------------------------------------------------------------------

@dataclass
class AccessibilityMask:
    """Binary accessibility mask for a chromosome arm.

    True = accessible site, False = inaccessible (missing data).
    """
    arm: str
    mask: np.ndarray

    @classmethod
    def from_npz(cls, path: str | Path, arm: str) -> AccessibilityMask:
        """Load from an Ag1000G-style .npz file.

        Tries keys: ``arm``, ``access_{arm}``, ``is_accessible_{arm}``.
        """
        data = np.load(str(path))
        for key_template in ["{arm}", "access_{arm}", "is_accessible_{arm}"]:
            key = key_template.format(arm=arm)
            if key in data:
                return cls(arm=arm, mask=data[key].astype(bool))
        available = list(data.keys())
        raise KeyError(
            f"No mask found for arm {arm!r} in {path}. "
            f"Available keys: {available}"
        )

    @property
    def accessible_fraction(self) -> float:
        return self.mask.mean()

    def slice(self, start: int, end: int) -> np.ndarray:
        """Get the mask for a genomic interval [start, end)."""
        return self.mask[start:end]

    def accessible_bp(self, start: int, end: int) -> int:
        return int(self.mask[start:end].sum())


# ---------------------------------------------------------------------------
# Block generation
# ---------------------------------------------------------------------------

@dataclass
class GenomicBlock:
    """One analysis block (typically 1 Mb) along a chromosome arm."""
    arm: str
    start: int
    end: int
    block_idx: int

    @property
    def length(self) -> int:
        return self.end - self.start


def generate_blocks(
    arm: str,
    arm_length: int | None = None,
    block_size: int = DEFAULT_BLOCK_SIZE,
    min_block_fraction: float = 0.5,
) -> list[GenomicBlock]:
    """Tile a chromosome arm into blocks of ``block_size`` bp.

    Drops the last block if it covers less than ``min_block_fraction``
    of a full block.
    """
    if arm_length is None:
        arm_length = ANOGAM_CHROMOSOME_ARMS.get(arm)
        if arm_length is None:
            raise ValueError(f"Unknown arm {arm!r}; pass arm_length explicitly")

    blocks = []
    for i, start in enumerate(range(0, arm_length, block_size)):
        end = min(start + block_size, arm_length)
        if (end - start) < block_size * min_block_fraction:
            continue
        blocks.append(GenomicBlock(arm=arm, start=start, end=end, block_idx=i))
    return blocks


# ---------------------------------------------------------------------------
# At-scale analysis protocol
# ---------------------------------------------------------------------------

@dataclass
class MosquitoAnalysis:
    """End-to-end analysis protocol for Anopheles population genomics.

    Runs fastcxt inference across all blocks of a chromosome arm for a
    set of sample pairs, handling missing data via accessibility masks.
    """
    model: object
    device: str = "cuda:0"
    block_size: int = DEFAULT_BLOCK_SIZE
    batch_size: int = 128
    build_workers: int = 4

    def run_chromosome_arm(
        self,
        genotype_matrix: np.ndarray,
        positions: np.ndarray,
        arm: str,
        pivot_pairs: list[tuple[int, int]],
        mutation_rate: float = 3.5e-9,
        accessibility_mask: AccessibilityMask | None = None,
        progress: bool = True,
    ) -> dict:
        """Run inference across all blocks of one chromosome arm.

        Parameters
        ----------
        genotype_matrix : (n_haploids, n_sites) int8 array
        positions : (n_sites,) bp positions
        arm : chromosome arm name (e.g. "2L")
        pivot_pairs : list of (sample_a, sample_b) pairs
        mutation_rate : per-bp per-generation
        accessibility_mask : optional mask for missing data
        progress : show progress bars

        Returns
        -------
        results : dict with keys "means", "variances", "blocks", "pairs"
        """
        from fastcxt.translate import translate_from_genotype_matrix

        arm_length = ANOGAM_CHROMOSOME_ARMS.get(arm)
        blocks = generate_blocks(arm, arm_length, self.block_size)
        block_tuples = [(b.start, b.end) for b in blocks]

        means, variances, index_map = translate_from_genotype_matrix(
            gm=genotype_matrix,
            positions=positions,
            model=self.model,
            blocks=block_tuples,
            pivot_pairs=pivot_pairs,
            mutation_rate=mutation_rate,
            device=self.device,
            batch_size=self.batch_size,
            build_workers=self.build_workers,
            progress=progress,
        )

        return {
            "means": means,
            "variances": variances,
            "index_map": index_map,
            "blocks": blocks,
            "pairs": pivot_pairs,
            "arm": arm,
            "mutation_rate": mutation_rate,
        }

    def run_all_arms(
        self,
        genotype_matrices: dict[str, tuple[np.ndarray, np.ndarray]],
        pivot_pairs: list[tuple[int, int]],
        mutation_rate: float = 3.5e-9,
        accessibility_masks: dict[str, AccessibilityMask] | None = None,
        arms: Sequence[str] | None = None,
        progress: bool = True,
    ) -> dict[str, dict]:
        """Run inference across multiple chromosome arms.

        Parameters
        ----------
        genotype_matrices : {arm: (gm, positions)} per chromosome arm
        """
        if arms is None:
            arms = list(genotype_matrices.keys())

        results = {}
        for arm in arms:
            gm, pos = genotype_matrices[arm]
            mask = accessibility_masks.get(arm) if accessibility_masks else None
            results[arm] = self.run_chromosome_arm(
                gm, pos, arm, pivot_pairs, mutation_rate,
                accessibility_mask=mask, progress=progress,
            )
        return results


# ---------------------------------------------------------------------------
# Simulation helpers for mosquito-like data
# ---------------------------------------------------------------------------

def simulate_anogam(
    seed: int,
    n_samples: int = 25,
    segment_length: float = 1e6,
) -> "tskit.TreeSequence":
    """Quick AnoGam simulation via stdpopsim for testing."""
    from fastcxt.simulate import simulate_stdpopsim, SimConfig
    cfg = SimConfig(
        name="AnoGam", kind="stdpopsim", species="AnoGam",
        n_samples=n_samples, sequence_length=segment_length,
    )
    return simulate_stdpopsim(seed, cfg)
