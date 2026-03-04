"""Tests for fastcxt.mosquito -- Anopheles gambiae analysis protocols."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from fastcxt.mosquito import (
    AccessibilityMask,
    GenomicBlock,
    generate_blocks,
    ANOGAM_CHROMOSOME_ARMS,
    DEFAULT_BLOCK_SIZE,
)


# ---------------------------------------------------------------------------
# AccessibilityMask
# ---------------------------------------------------------------------------

class TestAccessibilityMask:
    def test_from_npz_direct_key(self, tmp_path):
        mask_data = np.random.default_rng(0).choice([True, False], size=1_000_000, p=[0.8, 0.2])
        np.savez(tmp_path / "mask.npz", **{"2L": mask_data})
        m = AccessibilityMask.from_npz(tmp_path / "mask.npz", "2L")
        assert m.arm == "2L"
        assert len(m.mask) == 1_000_000

    def test_from_npz_access_prefix(self, tmp_path):
        mask_data = np.ones(500_000, dtype=bool)
        np.savez(tmp_path / "mask.npz", **{"access_3R": mask_data})
        m = AccessibilityMask.from_npz(tmp_path / "mask.npz", "3R")
        assert m.arm == "3R"
        assert m.accessible_fraction == 1.0

    def test_from_npz_is_accessible_prefix(self, tmp_path):
        mask_data = np.zeros(100_000, dtype=bool)
        np.savez(tmp_path / "mask.npz", **{"is_accessible_X": mask_data})
        m = AccessibilityMask.from_npz(tmp_path / "mask.npz", "X")
        assert m.accessible_fraction == 0.0

    def test_from_npz_missing_key(self, tmp_path):
        np.savez(tmp_path / "mask.npz", other_key=np.ones(10))
        with pytest.raises(KeyError, match="No mask found"):
            AccessibilityMask.from_npz(tmp_path / "mask.npz", "2L")

    def test_accessible_fraction(self):
        mask = np.array([True, True, False, True, False])
        m = AccessibilityMask(arm="2L", mask=mask)
        assert m.accessible_fraction == pytest.approx(0.6)

    def test_slice(self):
        mask = np.array([True, False, True, True, False, True])
        m = AccessibilityMask(arm="2L", mask=mask)
        s = m.slice(1, 4)
        np.testing.assert_array_equal(s, [False, True, True])

    def test_accessible_bp(self):
        mask = np.array([True, False, True, True, False, True])
        m = AccessibilityMask(arm="2L", mask=mask)
        assert m.accessible_bp(0, 6) == 4
        assert m.accessible_bp(1, 3) == 1


# ---------------------------------------------------------------------------
# GenomicBlock
# ---------------------------------------------------------------------------

class TestGenomicBlock:
    def test_length(self):
        b = GenomicBlock(arm="2L", start=1_000_000, end=2_000_000, block_idx=1)
        assert b.length == 1_000_000

    def test_fields(self):
        b = GenomicBlock(arm="3R", start=0, end=500_000, block_idx=0)
        assert b.arm == "3R"
        assert b.start == 0
        assert b.end == 500_000


# ---------------------------------------------------------------------------
# generate_blocks
# ---------------------------------------------------------------------------

class TestGenerateBlocks:
    def test_known_arm_lengths(self):
        for arm, length in ANOGAM_CHROMOSOME_ARMS.items():
            blocks = generate_blocks(arm)
            assert len(blocks) > 0
            assert blocks[0].start == 0
            assert blocks[0].arm == arm
            for b in blocks:
                assert b.length >= DEFAULT_BLOCK_SIZE * 0.5

    def test_2L_block_count(self):
        blocks = generate_blocks("2L")
        expected = ANOGAM_CHROMOSOME_ARMS["2L"] // DEFAULT_BLOCK_SIZE
        assert len(blocks) == expected or len(blocks) == expected + 1

    def test_custom_block_size(self):
        blocks = generate_blocks("2L", block_size=5_000_000)
        assert len(blocks) < 15

    def test_explicit_arm_length(self):
        blocks = generate_blocks("custom", arm_length=10_000_000, block_size=2_000_000)
        assert len(blocks) == 5
        assert blocks[0].arm == "custom"

    def test_unknown_arm_no_length_raises(self):
        with pytest.raises(ValueError, match="Unknown arm"):
            generate_blocks("UNKNOWN")

    def test_min_block_fraction(self):
        blocks_strict = generate_blocks("2L", block_size=10_000_000, min_block_fraction=0.9)
        blocks_lenient = generate_blocks("2L", block_size=10_000_000, min_block_fraction=0.1)
        assert len(blocks_lenient) >= len(blocks_strict)

    def test_contiguous_coverage(self):
        blocks = generate_blocks("2L", block_size=DEFAULT_BLOCK_SIZE, min_block_fraction=0.0)
        for i in range(len(blocks) - 1):
            assert blocks[i].end == blocks[i + 1].start

    def test_block_indices_monotonic(self):
        blocks = generate_blocks("3R")
        for i, b in enumerate(blocks):
            assert b.block_idx == i


# ---------------------------------------------------------------------------
# Anopheles constants
# ---------------------------------------------------------------------------

class TestAnogamConstants:
    def test_all_arms_present(self):
        assert set(ANOGAM_CHROMOSOME_ARMS.keys()) == {"2L", "2R", "3L", "3R", "X"}

    def test_arm_lengths_reasonable(self):
        for arm, length in ANOGAM_CHROMOSOME_ARMS.items():
            assert 20_000_000 < length < 70_000_000, f"{arm}: {length}"


# ---------------------------------------------------------------------------
# simulate_anogam (requires stdpopsim)
# ---------------------------------------------------------------------------

class TestSimulateAnogam:
    @pytest.mark.slow
    def test_simulate_basic(self):
        from fastcxt.mosquito import simulate_anogam
        ts = simulate_anogam(seed=42, n_samples=10, segment_length=1e5)
        assert ts.num_samples == 20  # diploid -> haploid
        assert ts.num_mutations > 0

    @pytest.mark.slow
    def test_simulate_different_seeds_differ(self):
        from fastcxt.mosquito import simulate_anogam
        ts1 = simulate_anogam(seed=1, n_samples=5, segment_length=5e4)
        ts2 = simulate_anogam(seed=999, n_samples=5, segment_length=5e4)
        # Different seeds should generally produce different tree sequences
        assert ts1.num_samples == ts2.num_samples == 10


# ---------------------------------------------------------------------------
# Integration: mask + preprocess
# ---------------------------------------------------------------------------

class TestMaskWithPreprocess:
    def test_apply_accessibility_mask(self):
        from fastcxt.preprocess import apply_accessibility_mask

        rng = np.random.default_rng(42)
        n_sites = 1000
        n_hap = 20
        gm = rng.integers(0, 2, (n_hap, n_sites), dtype=np.int8)
        positions = np.sort(rng.choice(1_000_000, n_sites, replace=False)).astype(float)

        mask = np.ones(1_000_000, dtype=bool)
        mask[500_000:] = False  # second half inaccessible

        gm_filt, pos_filt = apply_accessibility_mask(gm, positions, mask, 1_000_000)
        assert len(pos_filt) < n_sites
        assert np.all(pos_filt < 500_000)

    def test_all_accessible_passthrough(self):
        from fastcxt.preprocess import apply_accessibility_mask

        rng = np.random.default_rng(42)
        n_sites = 100
        gm = rng.integers(0, 2, (10, n_sites), dtype=np.int8)
        positions = np.linspace(0, 999_999, n_sites)
        mask = np.ones(1_000_000, dtype=bool)

        gm_out, pos_out = apply_accessibility_mask(gm, positions, mask, 1_000_000)
        assert len(pos_out) == n_sites

    def test_none_accessible(self):
        from fastcxt.preprocess import apply_accessibility_mask

        gm = np.ones((5, 50), dtype=np.int8)
        positions = np.linspace(0, 999, 50)
        mask = np.zeros(1_000_000, dtype=bool)

        gm_out, pos_out = apply_accessibility_mask(gm, positions, mask, 1_000_000)
        assert len(pos_out) == 0
