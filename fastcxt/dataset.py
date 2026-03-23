"""Training datasets for fastcxt.

PairDataset:             continuous log-TMRCA targets + mutation rate metadata
"""

from __future__ import annotations

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset


class PairDataset(Dataset):
    """One pair per item with continuous regression targets.

    Directory layout (per simulation):
        <split>/<scenario>/<id>/
            X.npy          (P, 2, n_windows, n_samples)  float16
            y.npy          (P, n_windows)                float16  (log-TMRCA)
            meta.json      { ..., "mutation_rate": float, ... }

    Parameters
    ----------
    root : str
        Root directory containing ``train/`` and ``test/`` subdirectories.
    split : str
        ``"train"`` or ``"test"``.
    max_samples : int
        Pad/truncate SFS sample dimension to this size.
    """

    def __init__(self, root: str, split: str = "train", max_samples: int = 200):
        self.root = root
        self.split = split
        self.max_samples = max_samples

        self.items: list[tuple[str, str, int, float]] = []

        split_dir = os.path.join(root, split)
        for dirpath, _dirnames, filenames in os.walk(split_dir):
            if "X.npy" not in filenames or "y.npy" not in filenames:
                continue

            X_path = os.path.join(dirpath, "X.npy")
            y_path = os.path.join(dirpath, "y.npy")

            mutation_rate = 1e-8
            meta_path = os.path.join(dirpath, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                mutation_rate = meta.get("mutation_rate", 1e-8)

            Y = np.load(y_path, mmap_mode="r")
            P = int(Y.shape[0])
            del Y
            for p_idx in range(P):
                self.items.append((X_path, y_path, p_idx, mutation_rate))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        X_path, y_path, p_idx, mutation_rate = self.items[i]

        X = np.load(X_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")

        Xi = torch.as_tensor(np.array(X[p_idx]), dtype=torch.float32)
        Xi = torch.log1p(Xi)

        N = Xi.shape[-1]
        if N < self.max_samples:
            Xi = torch.nn.functional.pad(Xi, (0, self.max_samples - N))
        elif N > self.max_samples:
            Xi = Xi[..., :self.max_samples]

        yi = torch.as_tensor(np.array(y[p_idx]), dtype=torch.float32)

        mu_rate = torch.tensor([np.log(mutation_rate)], dtype=torch.float32)

        return Xi, yi, mu_rate


class LazyPairDataset(Dataset):
    """Computes SFS on-the-fly from stored genotype matrices.

    Layout per sim:
        genotypes.npy   (n_samples, n_sites) int8
        positions.npy   (n_sites,) float64
        y.npy           (P, n_windows) float16
        pairs.npy       (P, 2) int32
        meta.json       { "storage_mode": "lazy_sfs", ... }
    """

    def __init__(self, root: str, split: str = "train", max_samples: int = 200):
        self.root = root
        self.split = split
        self.max_samples = max_samples

        self.items: list[tuple[str, str, str, str, int, float, float]] = []

        split_dir = os.path.join(root, split)
        for dirpath, _dirnames, filenames in os.walk(split_dir):
            if "genotypes.npy" not in filenames or "y.npy" not in filenames:
                continue

            gm_path = os.path.join(dirpath, "genotypes.npy")
            pos_path = os.path.join(dirpath, "positions.npy")
            y_path = os.path.join(dirpath, "y.npy")
            pairs_path = os.path.join(dirpath, "pairs.npy")

            mutation_rate = 1e-8
            sequence_length = 1e6
            meta_path = os.path.join(dirpath, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                mutation_rate = meta.get("mutation_rate", 1e-8)
                sequence_length = float(meta.get("sequence_length", 1e6))
            window_size = float(meta.get("window_size", 2000))

            Y = np.load(y_path, mmap_mode="r")
            P = int(Y.shape[0])
            del Y
            for p_idx in range(P):
                self.items.append((
                    gm_path, pos_path, y_path, pairs_path,
                    p_idx, mutation_rate, sequence_length, window_size,
                ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from fastcxt.sfs import build_sfs_tensor

        (gm_path, pos_path, y_path, pairs_path,
         p_idx, mutation_rate, sequence_length, window_size) = self.items[i]

        gm = np.load(gm_path, mmap_mode="r")
        positions = np.load(pos_path, mmap_mode="r")
        pairs = np.load(pairs_path, mmap_mode="r")
        y = np.load(y_path, mmap_mode="r")

        pa, pb = int(pairs[p_idx, 0]), int(pairs[p_idx, 1])

        # build_sfs_tensor returns (2, n_windows, n_samples) with log1p already applied
        Xi = build_sfs_tensor(
            np.array(gm), np.array(positions),
            pivot_a=pa, pivot_b=pb,
            sequence_length=sequence_length,
            window_size=int(window_size),
        )
        Xi = torch.as_tensor(Xi, dtype=torch.float32)

        N = Xi.shape[-1]
        if N < self.max_samples:
            Xi = torch.nn.functional.pad(Xi, (0, self.max_samples - N))
        elif N > self.max_samples:
            Xi = Xi[..., :self.max_samples]

        yi = torch.as_tensor(np.array(y[p_idx]), dtype=torch.float32)
        mu_rate = torch.tensor([np.log(mutation_rate)], dtype=torch.float32)

        return Xi, yi, mu_rate
