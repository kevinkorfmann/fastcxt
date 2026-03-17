"""Training datasets for fastcxt.

PairDataset:             continuous log-TMRCA targets + mutation rate metadata
TreeAugmentedPairDataset: adds tsinfer topology features alongside SFS
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
            [tree_feats.npy]   optional (P, n_windows, tree_feat_dim) float32

    Parameters
    ----------
    root : str
        Root directory containing ``train/`` and ``test/`` subdirectories.
    split : str
        ``"train"`` or ``"test"``.
    max_samples : int
        Pad/truncate SFS sample dimension to this size.
    use_trees : bool
        If True, also load tree_feats.npy when available.
    """

    def __init__(self, root: str, split: str = "train", max_samples: int = 200,
                 use_trees: bool = False):
        self.root = root
        self.split = split
        self.max_samples = max_samples
        self.use_trees = use_trees

        self.items: list[tuple[str, str, int, float, str | None]] = []

        split_dir = os.path.join(root, split)
        for dirpath, _dirnames, filenames in os.walk(split_dir):
            if "X.npy" not in filenames or "y.npy" not in filenames:
                continue

            X_path = os.path.join(dirpath, "X.npy")
            y_path = os.path.join(dirpath, "y.npy")
            tree_path = os.path.join(dirpath, "tree_feats.npy") if self.use_trees else None
            if tree_path and not os.path.exists(tree_path):
                tree_path = None

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
                self.items.append((X_path, y_path, p_idx, mutation_rate, tree_path))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        X_path, y_path, p_idx, mutation_rate, tree_path = self.items[i]

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

        if tree_path is not None:
            tree_feats = np.load(tree_path, mmap_mode="r")
            tf = torch.as_tensor(np.array(tree_feats[p_idx]), dtype=torch.float32)
            return Xi, yi, mu_rate, tf

        return Xi, yi, mu_rate


class LazyPairDataset(Dataset):
    """Computes SFS on-the-fly from stored genotype matrices.

    Layout per sim:
        genotypes.npy   (n_samples, n_sites) int8
        positions.npy   (n_sites,) float64
        y.npy           (P, n_windows) float16
        pairs.npy       (P, 2) int32
        meta.json       { "storage_mode": "lazy_sfs", ... }
        tree_feats.npy  (n_windows, tree_feat_dim) float32  [shared across pairs]
    """

    def __init__(self, root: str, split: str = "train", max_samples: int = 200,
                 use_trees: bool = False, tree_feat_dim: int = 597):
        self.root = root
        self.split = split
        self.max_samples = max_samples
        self.use_trees = use_trees
        self.tree_feat_dim = tree_feat_dim
        # LRU cache for tree_feats — each sim's tree_feats (~15 MB) is shared
        # across all its pairs; caching avoids redundant disk reads.
        # 64 entries × ~15 MB ≈ 1 GB per worker.
        self._tree_cache_keys: list[str] = []
        self._tree_cache: dict[str, torch.Tensor] = {}
        self._tree_cache_max = 64

        self.items: list[tuple[str, str, str, str, int, float, str | None, float]] = []

        split_dir = os.path.join(root, split)
        for dirpath, _dirnames, filenames in os.walk(split_dir):
            if "genotypes.npy" not in filenames or "y.npy" not in filenames:
                continue

            gm_path = os.path.join(dirpath, "genotypes.npy")
            pos_path = os.path.join(dirpath, "positions.npy")
            y_path = os.path.join(dirpath, "y.npy")
            pairs_path = os.path.join(dirpath, "pairs.npy")
            tree_path = os.path.join(dirpath, "tree_feats.npy") if self.use_trees else None
            if tree_path and not os.path.exists(tree_path):
                tree_path = None

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
                    p_idx, mutation_rate, tree_path, sequence_length, window_size,
                ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from fastcxt.sfs import build_sfs_tensor

        (gm_path, pos_path, y_path, pairs_path,
         p_idx, mutation_rate, tree_path, sequence_length, window_size) = self.items[i]

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

        if tree_path is not None:
            # tree_feats is shared across all pairs for a sim — use LRU cache
            tf = self._tree_cache.get(tree_path)
            if tf is None:
                tree_feats = np.load(tree_path, mmap_mode="r")
                tf = torch.as_tensor(np.array(tree_feats), dtype=torch.float32)
                if len(self._tree_cache_keys) >= self._tree_cache_max:
                    evict = self._tree_cache_keys.pop(0)
                    del self._tree_cache[evict]
                self._tree_cache[tree_path] = tf
                self._tree_cache_keys.append(tree_path)
            if tf.ndim == 3:
                tf = tf[p_idx]
            return Xi, yi, mu_rate, tf

        if self.use_trees:
            n_windows = yi.shape[0]
            tf = torch.zeros(n_windows, self.tree_feat_dim, dtype=torch.float32)
            return Xi, yi, mu_rate, tf

        return Xi, yi, mu_rate


class TreeAugmentedPairDataset(PairDataset):
    """PairDataset that always loads tree topology features.

    Falls back to zero features if tree_feats.npy is missing for a sample.
    """

    def __init__(self, root: str, split: str = "train", max_samples: int = 200,
                 tree_feat_dim: int = 597):
        super().__init__(root, split, max_samples, use_trees=True)
        self.tree_feat_dim = tree_feat_dim

    def __getitem__(self, i):
        result = super().__getitem__(i)
        if len(result) == 4:
            return result
        Xi, yi, mu_rate = result
        n_windows = yi.shape[0]
        tf = torch.zeros(n_windows, self.tree_feat_dim, dtype=torch.float32)
        return Xi, yi, mu_rate, tf
