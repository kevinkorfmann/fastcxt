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
) -> np.ndarray:
    """Bin site frequencies into genomic windows.

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
    freq_clipped = np.clip(pivot_frequencies.astype(np.int64), 0, num_samples - 1)
    flat = win_idx * num_samples + freq_clipped
    sfs = np.bincount(flat, minlength=n_windows * num_samples).astype(np.int32)
    return sfs.reshape(n_windows, num_samples)


def basic_filtering(
    gm: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove non-biallelic and fixed sites."""
    num_samples = gm.shape[0]
    non_bial = np.any(gm >= 2, axis=0)
    freq = gm.sum(0)
    fixed = (freq == 0) | (freq >= num_samples)
    keep = ~(non_bial | fixed)
    return gm[:, keep], positions[keep]


def build_sfs_tensor(
    gm: np.ndarray,
    positions: np.ndarray,
    pivot_a: int,
    pivot_b: int,
    sequence_length: float = 1e6,
    window_size: int = 2000,
) -> np.ndarray:
    """Build single-scale SFS tensor for one pivot pair.

    Sites are split into XOR (pivots differ) and XNOR (pivots agree)
    channels and binned at the base window resolution.  Multi-scale
    aggregation is delegated to the model's convolutional stem.

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
    freqs = np.clip(gm.sum(0).astype(np.int64), 0, num_samples - 1)
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
