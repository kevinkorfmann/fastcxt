"""Site-frequency spectrum computation for fastcxt.

Cleaned-up version of cxt/sfs.py with the same core algorithm.
"""

from __future__ import annotations

import numpy as np

W_MULTIPLIERS = (2, 8, 32, 64)


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
    window_starts = np.arange(n_windows) * step_size
    window_ends = np.minimum(window_starts + window_size, sequence_length)

    site_in_window = (
        (positions[:, np.newaxis] >= window_starts)
        & (positions[:, np.newaxis] < window_ends)
    )

    sfs = np.zeros((n_windows, num_samples), dtype=np.int32)
    for i in range(n_windows):
        wf = pivot_frequencies[site_in_window[:, i]]
        if wf.size:
            sfs[i] = np.bincount(wf, minlength=num_samples)[:num_samples]
    return sfs


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
    """Build multi-scale SFS tensor for one pivot pair.

    Returns
    -------
    X : (2, n_scales, n_windows, num_samples)  float16, log1p-transformed
    """
    step_size = window_size
    num_samples = gm.shape[0]
    n_windows = int(np.ceil(sequence_length / step_size))

    xor_mask = (gm[pivot_a] ^ gm[pivot_b]).astype(bool)
    freqs = gm.sum(0).astype(np.int32)

    pos_xor, freq_xor = positions[xor_mask], freqs[xor_mask]
    pos_xnor, freq_xnor = positions[~xor_mask], freqs[~xor_mask]

    Xs_xor = np.zeros((len(W_MULTIPLIERS), n_windows, num_samples), dtype=np.int32)
    Xs_xnor = np.zeros_like(Xs_xor)

    for i, m in enumerate(W_MULTIPLIERS):
        ws = window_size * m
        if pos_xor.size:
            Xs_xor[i] = calculate_window_sfs(
                pos_xor.astype(np.float32), freq_xor.astype(np.int32),
                window_size=ws, sequence_length=sequence_length,
                num_samples=num_samples, step_size=step_size,
            )
        if pos_xnor.size:
            Xs_xnor[i] = calculate_window_sfs(
                pos_xnor.astype(np.float32), freq_xnor.astype(np.int32),
                window_size=ws, sequence_length=sequence_length,
                num_samples=num_samples, step_size=step_size,
            )

    X = np.stack([Xs_xor, Xs_xnor], axis=0).astype(np.float16)
    return np.log1p(X)
