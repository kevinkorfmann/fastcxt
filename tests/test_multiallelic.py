"""Tests for multi-allelic site support (decompose-to-biallelic path)."""

import numpy as np

from fastcxt.sfs import decompose_multiallelic, basic_filtering, build_sfs_tensor


# A 6-sample matrix: site0 bi-allelic, site1 tri-allelic {0,1,2}, site2 monomorphic.
GM = np.array(
    [[0, 0, 0],
     [1, 1, 0],
     [0, 2, 0],
     [1, 0, 0],
     [0, 1, 0],
     [1, 2, 0]],
    dtype=np.int8,
)
POS = np.array([10.0, 20.0, 30.0])


def test_decompose_expands_multiallelic():
    gmd, posd = decompose_multiallelic(GM, POS)
    # site0 (1 col) + site1 -> 2 alt cols + site2 (1 col) = 4 columns
    assert gmd.shape == (6, 4)
    # positions carried and sorted; the tri-allelic site appears twice at pos 20
    assert list(posd) == [10.0, 20.0, 20.0, 30.0]
    # output is strictly 0/1
    assert set(np.unique(gmd).tolist()).issubset({0, 1})
    # conservation: total non-reference observations preserved
    assert int(gmd.sum()) == int((GM > 0).sum()) == 7


def test_basic_filtering_decompose_vs_drop():
    gm_d, pos_d = basic_filtering(GM, POS, multiallelic="decompose")
    gm_x, pos_x = basic_filtering(GM, POS, multiallelic="drop")
    # decompose keeps site0 + 2 alt columns (site2 dropped as monomorphic) = 3
    assert gm_d.shape[1] == 3
    # drop discards the tri-allelic site entirely, keeps only site0 = 1
    assert gm_x.shape[1] == 1
    # decompose retains strictly more informative columns
    assert gm_d.shape[1] > gm_x.shape[1]
    # everything returned is biallelic 0/1 and non-fixed
    assert set(np.unique(gm_d).tolist()).issubset({0, 1})
    freq = gm_d.sum(0)
    assert np.all((freq > 0) & (freq < gm_d.shape[0]))


def test_build_sfs_tensor_on_decomposed():
    """XOR/XNOR pipeline runs unchanged on decomposed biallelic columns."""
    gm_d, pos_d = basic_filtering(GM, POS, multiallelic="decompose")
    X = build_sfs_tensor(gm_d, pos_d, pivot_a=0, pivot_b=1,
                         sequence_length=40, window_size=40)
    assert X.shape == (2, 1, 6)
    assert np.isfinite(X).all()


def test_bad_mode_raises():
    try:
        basic_filtering(GM, POS, multiallelic="nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown multiallelic mode")
