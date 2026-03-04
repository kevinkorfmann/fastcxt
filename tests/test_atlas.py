"""Tests for fastcxt.atlas -- TimeAtlas data structure."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fastcxt.atlas import TimeAtlas, ArmData


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_pairs(n_samples: int) -> np.ndarray:
    return np.array(
        [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)],
        dtype=np.int32,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_atlas(rng):
    atlas = TimeAtlas()
    n_samples = 10
    pairs = _make_pairs(n_samples)
    n_pairs = len(pairs)  # 45

    for arm in ["2L", "2R", "3L"]:
        n_windows = 500
        atlas.add_arm(
            arm,
            means=rng.normal(8.0, 1.0, (n_pairs, n_windows)).astype(np.float32),
            variances=rng.exponential(0.5, (n_pairs, n_windows)).astype(np.float32),
            pairs=pairs,
            window_size=2000,
            mutation_rate=3.5e-9,
        )
    return atlas


# ---------------------------------------------------------------------------
# ArmData tests
# ---------------------------------------------------------------------------

class TestArmData:
    def test_basic_properties(self, rng):
        pairs = _make_pairs(8)
        n_p, n_w = len(pairs), 100
        ad = ArmData(
            arm="2L",
            means=rng.normal(0, 1, (n_p, n_w)).astype(np.float32),
            variances=np.ones((n_p, n_w), dtype=np.float32),
            pairs=pairs,
            window_starts=np.arange(n_w, dtype=np.int64) * 2000,
            window_size=2000,
            mutation_rate=1e-8,
        )
        assert ad.n_pairs == n_p
        assert ad.n_windows == n_w
        assert ad.arm_length == (n_w - 1) * 2000 + 2000

    def test_pair_index_found(self):
        pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)
        ad = ArmData("X", np.zeros((3, 10)), np.zeros((3, 10)),
                     pairs, np.arange(10) * 2000, 2000, 1e-8)
        assert ad.pair_index(0, 1) == 0
        assert ad.pair_index(1, 2) == 2
        # Reversed order should also work
        assert ad.pair_index(2, 1) == 2

    def test_pair_index_not_found(self):
        pairs = np.array([[0, 1]], dtype=np.int32)
        ad = ArmData("X", np.zeros((1, 10)), np.zeros((1, 10)),
                     pairs, np.arange(10) * 2000, 2000, 1e-8)
        assert ad.pair_index(5, 6) is None

    def test_window_at(self):
        ad = ArmData("X", np.zeros((1, 500)), np.zeros((1, 500)),
                     np.array([[0, 1]]), np.arange(500) * 2000, 2000, 1e-8)
        assert ad.window_at(0) == 0
        assert ad.window_at(1999) == 0
        assert ad.window_at(2000) == 1
        assert ad.window_at(999_999) == 499
        assert ad.window_at(10_000_000) == 499  # clamp


# ---------------------------------------------------------------------------
# TimeAtlas construction
# ---------------------------------------------------------------------------

class TestTimeAtlasConstruction:
    def test_add_arm(self):
        atlas = TimeAtlas()
        pairs = np.array([[0, 1], [0, 2]], dtype=np.int32)
        atlas.add_arm("2L", np.zeros((2, 100)), np.ones((2, 100)), pairs)
        assert "2L" in atlas.arms
        assert atlas.arms["2L"].n_pairs == 2
        assert atlas.arms["2L"].n_windows == 100

    def test_multiple_arms(self, sample_atlas):
        assert len(sample_atlas.arms) == 3
        assert set(sample_atlas.arms.keys()) == {"2L", "2R", "3L"}

    def test_custom_block_starts(self, rng):
        atlas = TimeAtlas()
        pairs = np.array([[0, 1]], dtype=np.int32)
        starts = np.array([0, 1_000_000, 2_000_000])
        atlas.add_arm("2L", rng.normal(0, 1, (1, 3)),
                      np.ones((1, 3)), pairs, block_starts=starts,
                      window_size=1_000_000)
        assert atlas.arms["2L"].window_starts[1] == 1_000_000


# ---------------------------------------------------------------------------
# TimeAtlas queries
# ---------------------------------------------------------------------------

class TestTimeAtlasQueries:
    def test_query_pair_exists(self, sample_atlas):
        result = sample_atlas.query_pair("2L", 0, 1)
        assert result is not None
        means, variances = result
        assert means.shape == (500,)
        assert variances.shape == (500,)

    def test_query_pair_reversed(self, sample_atlas):
        r1 = sample_atlas.query_pair("2L", 0, 3)
        r2 = sample_atlas.query_pair("2L", 3, 0)
        assert r1 is not None and r2 is not None
        np.testing.assert_array_equal(r1[0], r2[0])

    def test_query_pair_missing_arm(self, sample_atlas):
        assert sample_atlas.query_pair("X", 0, 1) is None

    def test_query_pair_missing_pair(self, sample_atlas):
        assert sample_atlas.query_pair("2L", 99, 100) is None

    def test_query_window(self, sample_atlas):
        result = sample_atlas.query_window("2L", 100_000)
        assert result is not None
        pairs, means, variances = result
        assert len(pairs) == 45
        assert means.shape == (45,)

    def test_query_window_missing_arm(self, sample_atlas):
        assert sample_atlas.query_window("X", 0) is None

    def test_query_region(self, sample_atlas):
        result = sample_atlas.query_region("2L", 100_000, 200_000)
        assert result is not None
        pairs, means, variances = result
        assert means.shape[0] == 45
        assert means.shape[1] > 0  # at least one window

    def test_query_region_missing_arm(self, sample_atlas):
        assert sample_atlas.query_region("X", 0, 100_000) is None


# ---------------------------------------------------------------------------
# TimeAtlas analytics
# ---------------------------------------------------------------------------

class TestTimeAtlasAnalytics:
    def test_mean_tmrca(self, sample_atlas):
        mt = sample_atlas.mean_tmrca("2L")
        assert mt.shape == (45,)
        assert np.all(mt > 0)  # exp of real numbers

    def test_deepest_pairs(self, sample_atlas):
        deep = sample_atlas.deepest_pairs("2L", 100_000, k=5)
        assert deep.shape == (5, 2)

    def test_shallowest_pairs(self, sample_atlas):
        shallow = sample_atlas.shallowest_pairs("2L", 100_000, k=3)
        assert shallow.shape == (3, 2)

    def test_deepest_k_exceeds_n_pairs(self, sample_atlas):
        deep = sample_atlas.deepest_pairs("2L", 100_000, k=100)
        assert deep.shape == (45, 2)  # clamped to actual count


# ---------------------------------------------------------------------------
# TimeAtlas summary and properties
# ---------------------------------------------------------------------------

class TestTimeAtlasSummary:
    def test_total_pairs(self, sample_atlas):
        assert sample_atlas.total_pairs == 45 * 3

    def test_total_windows(self, sample_atlas):
        assert sample_atlas.total_windows == 500 * 3

    def test_summary_keys(self, sample_atlas):
        s = sample_atlas.summary()
        assert "n_arms" in s
        assert "total_pairs" in s
        assert "per_arm" in s
        assert "2L" in s["per_arm"]
        assert "mutation_rate" in s["per_arm"]["2L"]

    def test_repr(self, sample_atlas):
        r = repr(sample_atlas)
        assert "TimeAtlas" in r
        assert "2L" in r

    def test_iter_pairs(self, sample_atlas):
        count = 0
        for a, b, m, v in sample_atlas.iter_pairs("2L"):
            assert isinstance(a, int)
            assert isinstance(b, int)
            assert m.shape == (500,)
            count += 1
        assert count == 45


# ---------------------------------------------------------------------------
# TimeAtlas serialization
# ---------------------------------------------------------------------------

class TestTimeAtlasSerialization:
    def test_save_load_roundtrip(self, sample_atlas):
        with tempfile.TemporaryDirectory() as d:
            sample_atlas.save(d)

            # Check files exist
            assert (Path(d) / "manifest.json").exists()
            assert (Path(d) / "2L.npz").exists()
            assert (Path(d) / "2R.npz").exists()
            assert (Path(d) / "3L.npz").exists()

            loaded = TimeAtlas.load(d)
            assert len(loaded.arms) == 3

    def test_data_integrity_after_roundtrip(self, sample_atlas):
        with tempfile.TemporaryDirectory() as d:
            sample_atlas.save(d)
            loaded = TimeAtlas.load(d)

            for arm in ["2L", "2R", "3L"]:
                np.testing.assert_allclose(
                    sample_atlas.arms[arm].means,
                    loaded.arms[arm].means,
                )
                np.testing.assert_allclose(
                    sample_atlas.arms[arm].variances,
                    loaded.arms[arm].variances,
                )
                np.testing.assert_array_equal(
                    sample_atlas.arms[arm].pairs,
                    loaded.arms[arm].pairs,
                )

    def test_query_after_roundtrip(self, sample_atlas):
        with tempfile.TemporaryDirectory() as d:
            sample_atlas.save(d)
            loaded = TimeAtlas.load(d)

            r1 = sample_atlas.query_pair("2L", 0, 1)
            r2 = loaded.query_pair("2L", 0, 1)
            assert r1 is not None and r2 is not None
            np.testing.assert_allclose(r1[0], r2[0])
            np.testing.assert_allclose(r1[1], r2[1])

    def test_metadata_preserved(self, sample_atlas):
        sample_atlas.metadata = {"species": "AnoGam", "version": "0.1"}
        with tempfile.TemporaryDirectory() as d:
            sample_atlas.save(d)
            loaded = TimeAtlas.load(d)
            assert loaded.metadata["species"] == "AnoGam"

    def test_empty_atlas_roundtrip(self):
        atlas = TimeAtlas()
        with tempfile.TemporaryDirectory() as d:
            atlas.save(d)
            loaded = TimeAtlas.load(d)
            assert len(loaded.arms) == 0
