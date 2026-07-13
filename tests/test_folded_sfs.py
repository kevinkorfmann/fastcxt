"""Tests for the folded (unpolarized) SFS representation.

The core scientific claim of the unpolarized pivot: folding the SFS frequency
axis to the minor-allele count makes the feature tensor invariant to the
ancestral/derived (0/1) coding, so a folded model needs no polarization and
works unchanged on a standard REF/ALT VCF.
"""

import numpy as np

from fastcxt.sfs import build_sfs_tensor


def _random_biallelic_gm(rng, num_samples, n_sites):
    """Genotype matrix where every site is polymorphic (count in [1, N-1])."""
    gm = np.zeros((num_samples, n_sites), dtype=np.int8)
    for s in range(n_sites):
        k = int(rng.integers(1, num_samples))  # 1 .. N-1
        idx = rng.choice(num_samples, size=k, replace=False)
        gm[idx, s] = 1
    return gm


def test_folded_is_polarization_invariant():
    """Folded features are identical under arbitrary per-site allele recoding."""
    rng = np.random.default_rng(0)
    num_samples, n_sites = 12, 400
    seqlen, window = 1e5, 2000

    gm = _random_biallelic_gm(rng, num_samples, n_sites)
    positions = np.sort(rng.uniform(0, seqlen, size=n_sites))

    # Flip the 0/1 coding at a random subset of sites: this is exactly what an
    # unpolarized (REF/ALT instead of ancestral/derived) encoding does.
    flip = rng.random(n_sites) < 0.5
    gm_flipped = gm.copy()
    gm_flipped[:, flip] = 1 - gm_flipped[:, flip]

    pa, pb = 0, 1
    Xf = build_sfs_tensor(gm, positions, pa, pb, seqlen, window, folded=True)
    Xf_flip = build_sfs_tensor(gm_flipped, positions, pa, pb, seqlen, window, folded=True)

    # Folded: byte-identical regardless of allele coding -> polarization not needed.
    np.testing.assert_array_equal(Xf, Xf_flip)

    # Sanity: the UNfolded features genuinely change under recoding, so the
    # invariance above is a real property of folding and not a trivial pass.
    Xu = build_sfs_tensor(gm, positions, pa, pb, seqlen, window, folded=False)
    Xu_flip = build_sfs_tensor(gm_flipped, positions, pa, pb, seqlen, window, folded=False)
    assert not np.array_equal(Xu, Xu_flip)


def test_folded_bins_are_minor_allele_count():
    """Folded frequency bins equal min(f, N-f); shape and total mass preserved."""
    num_samples = 10
    seqlen, window = 2000, 2000

    # Three sites with derived counts 2, 8, 5 -> folded minor counts 2, 2, 5.
    gm = np.zeros((num_samples, 3), dtype=np.int8)
    gm[:2, 0] = 1
    gm[:8, 1] = 1
    gm[:5, 2] = 1
    positions = np.array([10.0, 20.0, 30.0])

    # Pivots 0 and 1 are identical at all three sites -> all land in the XNOR
    # channel (index 1), so we can read the frequency axis directly.
    X = build_sfs_tensor(gm, positions, 0, 1, seqlen, window, folded=True)
    assert X.shape == (2, 1, num_samples)  # shape unchanged by folding

    counts = np.expm1(X[1, 0].astype(np.float64)).round().astype(int)
    assert counts[2] == 2   # sites with folded count 2 (derived 2 and 8)
    assert counts[5] == 1   # site with folded count 5
    assert counts.sum() == 3


def test_folded_shape_matches_unfolded():
    """Dual-mode: folded and unfolded produce the same tensor shape (drop-in)."""
    rng = np.random.default_rng(1)
    gm = _random_biallelic_gm(rng, 8, 100)
    positions = np.sort(rng.uniform(0, 5e4, size=100))
    Xf = build_sfs_tensor(gm, positions, 0, 1, 5e4, 2000, folded=True)
    Xu = build_sfs_tensor(gm, positions, 0, 1, 5e4, 2000, folded=False)
    assert Xf.shape == Xu.shape
