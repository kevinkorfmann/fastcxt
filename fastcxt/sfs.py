"""Site-frequency spectrum computation for fastcxt.

Single-scale SFS binning per genomic window.  Multi-scale feature
extraction is handled by learned convolutions inside the model
(see modules.MultiScaleInputProjection).
"""

from __future__ import annotations

import numpy as np


def calculate_window_sfs(
    positions: np.ndarray,
    pivot_frequencies: np.ndarray,
    window_size: int = 2000,
    sequence_length: float = 1e6,
    num_samples: int = 50,
    step_size: int = 2000,
    folded: bool = False,
) -> np.ndarray:
    """Bin site frequencies into genomic windows.

    Parameters
    ----------
    folded : bool
        If True, fold the frequency axis to the minor-allele count
        ``min(f, num_samples - f)``.  This makes the spectrum invariant to
        the ancestral/derived (0/1) coding, i.e. it removes the polarization
        requirement (see ``build_sfs_tensor``).

    Returns
    -------
    sfs : ndarray (n_windows, num_samples)
    """
    n_windows = int(np.ceil(sequence_length / step_size))

    if len(positions) == 0:
        return np.zeros((n_windows, num_samples), dtype=np.int32)

    # O(n_sites) integer division instead of O(n_sites * n_windows) broadcasting
    win_idx = np.clip(
        (positions // step_size).astype(np.int64), 0, n_windows - 1
    )
    freqs = pivot_frequencies.astype(np.int64)
    if folded:
        freqs = np.minimum(freqs, num_samples - freqs)
    freq_clipped = np.clip(freqs, 0, num_samples - 1)
    flat = win_idx * num_samples + freq_clipped
    sfs = np.bincount(flat, minlength=n_windows * num_samples).astype(np.int32)
    return sfs.reshape(n_windows, num_samples)


def decompose_multiallelic(
    gm: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expand multi-allelic sites into per-alt-allele bi-allelic pseudo-sites.

    Allele ``0`` is treated as the reference/ancestral state.  A site carrying
    alleles ``{0, k1, k2, ...}`` becomes one 0/1 column per alt allele present
    (``1`` where ``gm == k``), each inheriting the site's base-pair position.
    Bi-allelic and monomorphic sites pass through as a single 0/1 column.

    This retains multi-allelic variation (which ``basic_filtering(..., "drop")``
    discards): a pivot pair carrying two different alt alleles at a site then
    contributes a difference in each alt column, matching the multiple
    mutations that separate the two lineages there.  Columns are returned
    sorted by position so downstream windowing is unaffected.
    """
    gm = np.asarray(gm)
    positions = np.asarray(positions)
    num_samples, n_sites = gm.shape
    if n_sites == 0:
        return gm.astype(np.int8), positions

    max_per_site = gm.max(axis=0)
    bi = max_per_site <= 1

    # Bi-allelic / monomorphic sites: coerce to 0/1 in place (vectorised).
    bi_cols = (gm[:, bi] > 0).astype(np.int8)
    bi_pos = positions[bi]

    # Multi-allelic sites are the minority: expand each alt allele.
    extra_cols: list[np.ndarray] = []
    extra_pos: list[float] = []
    for s in np.nonzero(~bi)[0]:
        col = gm[:, s]
        for k in range(1, int(col.max()) + 1):
            present = col == k
            if present.any():
                extra_cols.append(present.astype(np.int8))
                extra_pos.append(positions[s])

    if extra_cols:
        gm_out = np.concatenate([bi_cols, np.stack(extra_cols, axis=1)], axis=1)
        pos_out = np.concatenate([bi_pos, np.asarray(extra_pos, dtype=positions.dtype)])
    else:
        gm_out, pos_out = bi_cols, bi_pos

    order = np.argsort(pos_out, kind="stable")
    return gm_out[:, order], pos_out[order]


def basic_filtering(
    gm: np.ndarray, positions: np.ndarray, multiallelic: str = "drop"
) -> tuple[np.ndarray, np.ndarray]:
    """Remove fixed sites; handle multi-allelic sites per ``multiallelic``.

    Parameters
    ----------
    multiallelic : {"drop", "decompose"}
        ``"drop"`` (default): discard any site with an allele index >= 2
        (strictly bi-allelic, original behaviour).
        ``"decompose"``: expand multi-allelic sites into per-alt bi-allelic
        pseudo-sites (see :func:`decompose_multiallelic`) so they are retained.
    """
    num_samples = gm.shape[0]
    if multiallelic == "decompose":
        gm, positions = decompose_multiallelic(gm, positions)
        drop_multi = np.zeros(gm.shape[1], dtype=bool)
    elif multiallelic == "drop":
        drop_multi = np.any(gm >= 2, axis=0)
    else:
        raise ValueError(
            f"multiallelic must be 'drop' or 'decompose', got {multiallelic!r}"
        )
    freq = gm.sum(0)
    fixed = (freq == 0) | (freq >= num_samples)
    keep = ~(drop_multi | fixed)
    return gm[:, keep], positions[keep]


def build_sfs_tensor(
    gm: np.ndarray,
    positions: np.ndarray,
    pivot_a: int,
    pivot_b: int,
    sequence_length: float = 1e6,
    window_size: int = 2000,
    folded: bool = False,
) -> np.ndarray:
    """Build single-scale SFS tensor for one pivot pair.

    Sites are split into XOR (pivots differ) and XNOR (pivots agree)
    channels and binned at the base window resolution.  Multi-scale
    aggregation is delegated to the model's convolutional stem.

    The XOR/XNOR channel assignment (``gm[a] ^ gm[b]``) is already invariant
    to the 0/1 allele coding.  The only polarization-dependent part is the
    frequency axis (``gm.sum(0)`` = count of the allele coded ``1``).  With
    ``folded=True`` this axis is folded to the minor-allele count
    ``min(f, N - f)``, which is invariant to swapping ancestral/derived, so a
    model trained on folded features works unchanged on unpolarized inputs
    (e.g. a standard REF/ALT VCF).  Shape is preserved: folded mass simply
    occupies the lower half of the frequency axis.

    Returns
    -------
    X : (2, n_windows, num_samples)  float16, log1p-transformed
    """
    step_size = window_size
    num_samples = gm.shape[0]
    n_windows = int(np.ceil(sequence_length / step_size))

    # Compute window indices and frequencies once for all sites
    win_idx = np.clip(
        (positions // step_size).astype(np.int64), 0, n_windows - 1
    )
    freqs = gm.sum(0).astype(np.int64)
    if folded:
        freqs = np.minimum(freqs, num_samples - freqs)
    freqs = np.clip(freqs, 0, num_samples - 1)
    xor_mask = (gm[pivot_a] ^ gm[pivot_b]).astype(bool)

    # Flat bincount for XOR and XNOR channels
    flat_size = n_windows * num_samples

    flat_xor = win_idx[xor_mask] * num_samples + freqs[xor_mask]
    sfs_xor = np.bincount(flat_xor, minlength=flat_size).reshape(n_windows, num_samples)

    xnor_mask = ~xor_mask
    flat_xnor = win_idx[xnor_mask] * num_samples + freqs[xnor_mask]
    sfs_xnor = np.bincount(flat_xnor, minlength=flat_size).reshape(n_windows, num_samples)

    X = np.stack([sfs_xor, sfs_xnor], axis=0).astype(np.float16)
    return np.log1p(X)
