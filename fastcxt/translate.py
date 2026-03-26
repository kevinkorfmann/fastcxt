"""Unified inference API for fastcxt.

Single forward pass per pair -- no autoregressive loop, no sampling.
Returns (mean, variance) of log-TMRCA per window directly.
"""

from __future__ import annotations

import gc
from typing import Sequence

import numpy as np
import torch
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from fastcxt.sfs import build_sfs_tensor, basic_filtering


# ---------------------------------------------------------------------------
# Source building
# ---------------------------------------------------------------------------

def _build_one(task: dict) -> np.ndarray:
    return build_sfs_tensor(**task)


def _build_sources(
    gm: np.ndarray,
    positions: np.ndarray,
    blocks: list[tuple],
    pivot_pairs: list[tuple],
    window_size: int = 2000,
    workers: int = 4,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build SFS source tensors for all (block, pair) combinations.

    Pre-slices the genotype matrix per block so each task only pickles
    the small block slice rather than the full matrix.
    """
    gm_filt, pos_filt = basic_filtering(gm, positions)

    # Pre-slice per block — each block's gm/pos is small and cheap to pickle
    block_data = []
    for bstart, bend in blocks:
        mask = (pos_filt >= bstart) & (pos_filt < bend)
        block_data.append((gm_filt[:, mask], pos_filt[mask] - bstart, float(bend - bstart)))

    tasks = []
    index_map = []
    for b_idx, (bgm, bpos, seq_len) in enumerate(block_data):
        for p_idx, (pA, pB) in enumerate(pivot_pairs):
            tasks.append(dict(
                gm=bgm, positions=bpos, pivot_a=pA, pivot_b=pB,
                sequence_length=seq_len, window_size=window_size,
            ))
            index_map.append([b_idx, p_idx])

    if workers > 1:
        results = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_build_one, t): i for i, t in enumerate(tasks)}
            it = as_completed(futs)
            if progress:
                it = tqdm(it, total=len(tasks), desc="Building sources", leave=False)
            for fut in it:
                results[futs[fut]] = fut.result()
        X = np.stack(results)
    else:
        it = tasks
        if progress:
            it = tqdm(it, total=len(tasks), desc="Building sources", leave=False)
        X = np.stack([_build_one(t) for t in it])

    return X, np.array(index_map)


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(
    model,
    src: torch.Tensor,
    mutation_rate: float,
    batch_size: int = 128,
    device: str = "cuda",
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run single-pass inference on source tensors.

    Returns
    -------
    means : (N, W) predicted log-TMRCA means
    variances : (N, W) predicted log-TMRCA variances
    """
    model.eval()
    model.to(device)
    N = src.shape[0]

    log_mu = torch.tensor([[np.log(mutation_rate)]], dtype=torch.float32)
    log_mu = log_mu.to(device)

    all_means = []
    all_vars = []

    chunk_iter = range(0, N, batch_size)
    if progress:
        chunk_iter = tqdm(chunk_iter, total=(N + batch_size - 1) // batch_size,
                          desc="Inference", leave=False)

    with torch.inference_mode():
        for start in chunk_iter:
            end = min(start + batch_size, N)
            batch_src = src[start:end].to(device, non_blocking=True)
            batch_mu = log_mu.expand(end - start, -1)

            out = model(batch_src, batch_mu)
            all_means.append(out[..., 0].cpu().numpy())
            all_vars.append(torch.exp(torch.clamp(out[..., 1], -10, 10)).cpu().numpy())

    means = np.concatenate(all_means, axis=0)
    variances = np.concatenate(all_vars, axis=0)
    return means, variances


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def translate_from_genotype_matrix(
    gm: np.ndarray,
    positions: np.ndarray,
    model,
    blocks: list[tuple] = [(0, 1_000_000)],
    pivot_pairs: list[tuple] = [(0, 1)],
    mutation_rate: float = 1e-8,
    device: str = "cuda",
    batch_size: int = 128,
    build_workers: int = 4,
    progress: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer pairwise TMRCA from a genotype matrix.

    Returns
    -------
    means : (N, W) predicted log-TMRCA means
    variances : (N, W) predicted variances
    index_map : (N, 2) mapping each row to [block_idx, pivot_idx]
    """
    a, b = blocks[0]
    n_win = getattr(model, 'config', None) and model.config.n_windows or 500
    window_size = int((b - a) / n_win)

    X, index_map = _build_sources(
        gm, positions, blocks, pivot_pairs,
        window_size=window_size, workers=build_workers, progress=progress,
    )

    param_dtype = next(model.parameters()).dtype
    X_t = torch.as_tensor(X, dtype=param_dtype)
    if torch.cuda.is_available():
        X_t = X_t.pin_memory()

    means, variances = predict(
        model, X_t, mutation_rate=mutation_rate,
        batch_size=batch_size, device=device, progress=progress,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return means, variances, index_map


def translate_from_ts(ts, model, **kwargs):
    """Infer TMRCA from a tree sequence."""
    positions = ts.tables.sites.position
    gm = ts.genotype_matrix().T
    return translate_from_genotype_matrix(
        gm=gm, positions=positions, model=model, **kwargs,
    )


def translate(
    input_data,
    model,
    blocks: list[tuple] = [(0, 1_000_000)],
    pivot_pairs: list[tuple] = [(0, 1)],
    data_type: str | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unified inference entry point.

    Parameters
    ----------
    input_data : tree sequence or (gm, positions) tuple
    model : loaded FastCxtModel
    blocks : list of (start, end) genomic intervals in bp
    pivot_pairs : list of (sample_A, sample_B) haploid indices
    data_type : "ts" or "gm". Auto-detected if None.

    Returns
    -------
    means, variances, index_map
    """
    if data_type is None:
        if isinstance(input_data, tuple) and len(input_data) == 2:
            data_type = "gm"
        else:
            data_type = "ts"

    if data_type == "ts":
        return translate_from_ts(input_data, model, blocks=blocks,
                                 pivot_pairs=pivot_pairs, **kwargs)
    elif data_type == "gm":
        gm, positions = input_data
        return translate_from_genotype_matrix(
            gm=gm, positions=positions, model=model,
            blocks=blocks, pivot_pairs=pivot_pairs, **kwargs,
        )
    else:
        raise ValueError(f"data_type must be 'ts' or 'gm', got {data_type!r}")
