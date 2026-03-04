"""Centralized path configuration for fastcxt.

All cluster/sietch paths and data locations are defined here, overridable
via environment variables.  Import this module wherever you need data paths
instead of hardcoding strings.

Usage:
    from fastcxt.paths import PATHS

    ts_dir = PATHS.ag1000g_data_dir
    mask   = PATHS.ag1000g_accessibility_mask
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class FastCxtPaths:
    """Cluster path registry.  All values come from env vars with sietch defaults."""

    # --- Root ---
    base_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("FASTCXT_BASE_DIR",
                       os.environ.get("BASE_DIR",
                                      "/sietch_colab/data_share/cxt_scratch"))))

    # --- Anopheles gambiae / Ag1000G ---
    ag1000g_data_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("AG1000G_DATA_DIR",
                       "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/tsinfer_data_v2")))

    ag1000g_accessibility_mask: Path = field(default_factory=lambda: Path(
        os.environ.get("AG1000G_ACCESSIBILITY",
                       os.environ.get("BITMASK",
                                      "/sietch_colab/data_share/Ag1000G/Ag3.0/args_trees/singer/agp3.is_accessible.txt.npz"))))

    # --- Human 1000 Genomes ---
    hg1kg_trees_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("HG1KG_TSZ_DIR",
                       "/sietch_colab/data_share/hg1kg/tsinfer-trees/working")))

    # --- Checkpoints ---
    checkpoint_cache: Path = field(default_factory=lambda: Path(
        os.environ.get("CXT_CHECKPOINT_CACHE",
                       os.environ.get("FASTCXT_CHECKPOINT_CACHE",
                                      "/sietch_colab/data_share/cxt/models"))))

    # --- Benchmarks ---
    benchmark_dir: Path = field(default_factory=lambda: Path(
        os.environ.get("FASTCXT_BENCHMARK_DIR",
                       "/sietch_colab/data_share/cxt/mosquito/benchmarks")))

    # --- Derived paths ---

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def lightning_logs(self) -> Path:
        return self.base_dir / "lightning_logs"

    @property
    def figures_dir(self) -> Path:
        return self.base_dir / "figures" / "output"

    # --- Ag1000G chromosome-arm tree-sequence files ---

    def ag1000g_arm_trees(self, arm: str) -> Path:
        """Path to tsinfer tree-sequence file for an Ag1000G chromosome arm."""
        return self.ag1000g_data_dir / f"{arm}.trees"

    # --- Convenience ---

    def ensure_dirs(self) -> None:
        """Create key directories if they don't exist."""
        for d in [self.data_dir, self.processed_dir, self.lightning_logs, self.figures_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def exists_report(self) -> dict[str, bool]:
        """Check which paths actually exist on the current system."""
        return {
            "base_dir": self.base_dir.exists(),
            "ag1000g_data_dir": self.ag1000g_data_dir.exists(),
            "ag1000g_accessibility_mask": self.ag1000g_accessibility_mask.exists(),
            "hg1kg_trees_dir": self.hg1kg_trees_dir.exists(),
            "checkpoint_cache": self.checkpoint_cache.exists(),
            "benchmark_dir": self.benchmark_dir.exists(),
        }


PATHS = FastCxtPaths()
