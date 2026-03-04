"""Tests for fastcxt.paths -- centralized path configuration."""

import os

import pytest

from fastcxt.paths import FastCxtPaths, PATHS


class TestFastCxtPaths:
    def test_singleton_available(self):
        assert PATHS is not None
        assert isinstance(PATHS, FastCxtPaths)

    def test_default_paths_are_paths(self):
        from pathlib import Path
        assert isinstance(PATHS.base_dir, Path)
        assert isinstance(PATHS.ag1000g_data_dir, Path)
        assert isinstance(PATHS.ag1000g_accessibility_mask, Path)
        assert isinstance(PATHS.hg1kg_trees_dir, Path)
        assert isinstance(PATHS.checkpoint_cache, Path)
        assert isinstance(PATHS.benchmark_dir, Path)

    def test_derived_paths(self):
        assert "data" in str(PATHS.data_dir)
        assert "processed" in str(PATHS.processed_dir)
        assert "lightning" in str(PATHS.lightning_logs)
        assert "figures" in str(PATHS.figures_dir)

    def test_ag1000g_arm_trees(self):
        p = PATHS.ag1000g_arm_trees("2L")
        assert str(p).endswith("2L.trees")

    def test_exists_report_returns_dict(self):
        report = PATHS.exists_report()
        assert isinstance(report, dict)
        assert "base_dir" in report
        assert "ag1000g_data_dir" in report

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("FASTCXT_BASE_DIR", "/tmp/test_fastcxt")
        p = FastCxtPaths()
        assert str(p.base_dir) == "/tmp/test_fastcxt"

    def test_ag1000g_env_override(self, monkeypatch):
        monkeypatch.setenv("AG1000G_DATA_DIR", "/custom/ag1000g")
        p = FastCxtPaths()
        assert str(p.ag1000g_data_dir) == "/custom/ag1000g"

    def test_bitmask_env_override(self, monkeypatch):
        monkeypatch.setenv("BITMASK", "/custom/mask.npz")
        # Clear AG1000G_ACCESSIBILITY to test fallback to BITMASK
        monkeypatch.delenv("AG1000G_ACCESSIBILITY", raising=False)
        p = FastCxtPaths()
        assert str(p.ag1000g_accessibility_mask) == "/custom/mask.npz"

    def test_frozen(self):
        with pytest.raises(AttributeError):
            PATHS.base_dir = "/other"
