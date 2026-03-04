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

        yi = torch.as_tensor(np.array(y[p_idx]), dtype=torch.float32)

        mu_rate = torch.tensor([np.log(mutation_rate)], dtype=torch.float32)

        if tree_path is not None:
            tree_feats = np.load(tree_path, mmap_mode="r")
            tf = torch.as_tensor(np.array(tree_feats[p_idx]), dtype=torch.float32)
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
