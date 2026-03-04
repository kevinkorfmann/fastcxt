"""Tests for fastcxt.preprocess -- new preprocessing pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fastcxt.preprocess import (
    windowed_tmrca,
    choose_pairs,
    apply_accessibility_mask,
    process_one_ts,
    estimate_mutation_rate,
    find_ts_files,
    deterministic_split,
    PreprocessJob,
)
from fastcxt.simulate import simulate_constant, SimConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_ts():
    cfg = SimConfig(name="constant", kind="msprime", n_samples=10,
                    sequence_length=1e5)
    return simulate_constant(42, cfg)


@pytest.fixture
def medium_ts():
    cfg = SimConfig(name="constant", kind="msprime", n_samples=15,
                    sequence_length=5e5)
    return simulate_constant(42, cfg)


# ---------------------------------------------------------------------------
# windowed_tmrca
# ---------------------------------------------------------------------------

class TestWindowedTmrca:
    def test_output_shape(self, small_ts):
        y = windowed_tmrca(small_ts, 0, 1, window_size=2000,
                           sequence_length=int(small_ts.sequence_length))
        expected_windows = int(np.ceil(small_ts.sequence_length / 2000))
        assert y.shape == (expected_windows,)

    def test_positive_values(self, small_ts):
        y = windowed_tmrca(small_ts, 0, 1, 2000,
                           int(small_ts.sequence_length))
        assert np.all(y > 0)

    def test_different_pairs_differ(self, small_ts):
        y01 = windowed_tmrca(small_ts, 0, 1, 2000, int(small_ts.sequence_length))
        y02 = windowed_tmrca(small_ts, 0, 2, 2000, int(small_ts.sequence_length))
        assert not np.allclose(y01, y02)

    def test_auto_sequence_length(self, small_ts):
        y = windowed_tmrca(small_ts, 0, 1, 2000)
        assert len(y) > 0


# ---------------------------------------------------------------------------
# choose_pairs
# ---------------------------------------------------------------------------

class TestChoosePairs:
    def test_basic(self):
        pairs = choose_pairs(10, 5, seed=42)
        assert pairs.shape == (5, 2)
        assert pairs.dtype == np.int32

    def test_all_pairs_when_small(self):
        pairs = choose_pairs(4, 100, seed=42)
        assert len(pairs) == 6  # 4 choose 2

    def test_deterministic(self):
        p1 = choose_pairs(20, 10, seed=42)
        p2 = choose_pairs(20, 10, seed=42)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds(self):
        p1 = choose_pairs(50, 10, seed=1)
        p2 = choose_pairs(50, 10, seed=2)
        assert not np.array_equal(p1, p2)

    def test_valid_indices(self):
        n = 15
        pairs = choose_pairs(n, 20, seed=42)
        assert np.all(pairs[:, 0] < pairs[:, 1])
        assert np.all(pairs >= 0)
        assert np.all(pairs < n)


# ---------------------------------------------------------------------------
# apply_accessibility_mask
# ---------------------------------------------------------------------------

class TestApplyAccessibilityMask:
    def test_half_masked(self):
        rng = np.random.default_rng(42)
        gm = rng.integers(0, 2, (10, 100), dtype=np.int8)
        positions = np.linspace(0, 999_999, 100)
        mask = np.ones(1_000_000, dtype=bool)
        mask[500_000:] = False

        gm_out, pos_out = apply_accessibility_mask(gm, positions, mask, 1_000_000)
        assert len(pos_out) < 100
        assert np.all(pos_out < 500_000)

    def test_all_accessible(self):
        gm = np.ones((5, 20), dtype=np.int8)
        positions = np.arange(20, dtype=float) * 100
        mask = np.ones(10_000, dtype=bool)
        gm_out, pos_out = apply_accessibility_mask(gm, positions, mask, 10_000)
        assert len(pos_out) == 20

    def test_none_accessible(self):
        gm = np.ones((5, 20), dtype=np.int8)
        positions = np.arange(20, dtype=float) * 100
        mask = np.zeros(10_000, dtype=bool)
        gm_out, pos_out = apply_accessibility_mask(gm, positions, mask, 10_000)
        assert len(pos_out) == 0


# ---------------------------------------------------------------------------
# process_one_ts
# ---------------------------------------------------------------------------

class TestProcessOneTs:
    def test_output_shapes(self, small_ts):
        pairs = choose_pairs(small_ts.num_samples, 5, seed=42)
        X, y = process_one_ts(small_ts, pairs, window_size=2000,
                              sequence_length=int(small_ts.sequence_length))
        assert X.shape[0] == 5
        assert X.shape[1] == 2   # xor/xnor
        assert X.shape[2] == 4   # scales
        assert y.shape[0] == 5
        assert X.shape[3] == y.shape[1]  # windows match

    def test_output_dtypes(self, small_ts):
        pairs = choose_pairs(small_ts.num_samples, 3, seed=42)
        X, y = process_one_ts(small_ts, pairs, window_size=2000,
                              sequence_length=int(small_ts.sequence_length))
        assert X.dtype == np.float16
        assert y.dtype == np.float16

    def test_y_is_log_scale(self, small_ts):
        pairs = choose_pairs(small_ts.num_samples, 3, seed=42)
        X, y = process_one_ts(small_ts, pairs, window_size=2000,
                              sequence_length=int(small_ts.sequence_length))
        assert np.all(np.isfinite(y.astype(np.float32)))

    def test_with_accessibility_mask(self, small_ts):
        seq_len = int(small_ts.sequence_length)
        mask = np.ones(seq_len, dtype=bool)
        mask[seq_len // 2:] = False

        pairs = choose_pairs(small_ts.num_samples, 3, seed=42)
        X, y = process_one_ts(small_ts, pairs, window_size=2000,
                              sequence_length=seq_len,
                              accessibility_mask=mask)
        assert X.shape[0] == 3


# ---------------------------------------------------------------------------
# estimate_mutation_rate
# ---------------------------------------------------------------------------

class TestEstimateMutationRate:
    def test_positive(self, small_ts):
        mu = estimate_mutation_rate(small_ts)
        assert mu > 0

    def test_order_of_magnitude(self, small_ts):
        mu = estimate_mutation_rate(small_ts)
        assert 1e-10 < mu < 1e-5


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

class TestFileDiscovery:
    def test_find_ts_files(self, tmp_path):
        (tmp_path / "a.trees").touch()
        (tmp_path / "b.ts").touch()
        (tmp_path / "c.txt").touch()
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.trees").touch()

        files = find_ts_files(tmp_path)
        names = [f.name for f in files]
        assert "a.trees" in names
        assert "b.ts" in names
        assert "d.trees" in names
        assert "c.txt" not in names

    def test_empty_dir(self, tmp_path):
        assert find_ts_files(tmp_path) == []


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

class TestDeterministicSplit:
    def test_basic_split(self, tmp_path):
        files = [tmp_path / f"ts_{i}.trees" for i in range(10)]
        for f in files:
            f.touch()

        train_idx = deterministic_split(files, tmp_path, train_ratio=0.8, seed=42)
        assert len(train_idx) == 8
        test_idx = set(range(10)) - train_idx
        assert len(test_idx) == 2

    def test_deterministic(self, tmp_path):
        files = [tmp_path / f"ts_{i}.trees" for i in range(20)]
        for f in files:
            f.touch()

        s1 = deterministic_split(files, tmp_path, seed=42)
        s2 = deterministic_split(files, tmp_path, seed=42)
        assert s1 == s2

    def test_at_least_one_test(self, tmp_path):
        files = [tmp_path / f"ts_{i}.trees" for i in range(5)]
        for f in files:
            f.touch()

        train_idx = deterministic_split(files, tmp_path, train_ratio=0.99, seed=42)
        test_idx = set(range(5)) - train_idx
        assert len(test_idx) >= 1


# ---------------------------------------------------------------------------
# PreprocessJob dataclass
# ---------------------------------------------------------------------------

class TestPreprocessJob:
    def test_creation(self):
        job = PreprocessJob(
            idx=0, ts_path="/tmp/ts.trees", base_dir="/tmp",
            out_root="/tmp/out", split="train", window_size=2000,
            sequence_length=1_000_000, num_pairs=100, global_seed=42,
            skip_existing=False, simplify_n=0, extract_trees=False,
            mutation_rate_override=None, accessibility_mask_path=None,
        )
        assert job.idx == 0
        assert job.accessibility_mask_path is None
