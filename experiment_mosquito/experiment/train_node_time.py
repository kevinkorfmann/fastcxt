#!/usr/bin/env python3
"""Train a NodeTimeModel from pre-extracted features.

Usage:
    python train_node_time.py --features-dir outputs/node_features
    python train_node_time.py --features-dir outputs/node_features_tsinfer --out outputs/node_time_tsinfer.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from fastcxt.tree_utils import FEATS_PER_NODE
from fastcxt.node_time_model import NodeTimeModel
from config import OUTPUTS_DIR, MAX_SAMPLES, N_WINDOWS, NODE_D_MODEL, NODE_N_LAYERS

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class NodeTimeDataset(Dataset):
    def __init__(self, feats, targets, mu, masks):
        self.feats = feats
        self.targets = targets
        self.mu = mu
        self.masks = masks

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return (
            torch.as_tensor(self.feats[i], dtype=torch.float32),
            torch.as_tensor(self.targets[i], dtype=torch.float32),
            torch.as_tensor([self.mu[i]], dtype=torch.float32),
            torch.as_tensor(self.masks[i], dtype=torch.float32),
        )


def masked_mse_loss(pred, target, mask):
    """MSE loss only on real (non-padded) nodes."""
    mask_3d = mask.unsqueeze(1).expand_as(pred)
    diff = (pred - target) ** 2
    return (diff * mask_3d).sum() / mask_3d.sum().clamp(min=1)


def train(features_dir: Path, out_path: Path, epochs: int = 30, batch_size: int = 32):
    train_feats = np.load(features_dir / "train_feats.npy", mmap_mode="r")
    train_times = np.load(features_dir / "train_times.npy", mmap_mode="r")
    train_mu = np.load(features_dir / "train_mu.npy")
    train_masks = np.load(features_dir / "train_masks.npy")
    val_feats = np.load(features_dir / "val_feats.npy", mmap_mode="r")
    val_times = np.load(features_dir / "val_times.npy", mmap_mode="r")
    val_mu = np.load(features_dir / "val_mu.npy")
    val_masks = np.load(features_dir / "val_masks.npy")

    max_internal = MAX_SAMPLES - 1
    tree_feat_dim = max_internal * FEATS_PER_NODE

    model = NodeTimeModel(
        tree_feat_dim=tree_feat_dim,
        n_internal=max_internal,
        d_model=NODE_D_MODEL,
        n_layers=NODE_N_LAYERS,
        n_windows=N_WINDOWS,
    ).to(DEVICE)

    train_ds = NodeTimeDataset(train_feats, train_times, train_mu, train_masks)
    val_ds = NodeTimeDataset(val_feats, val_times, val_mu, val_masks)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Training NodeTimeModel ({n_params:,} params) on {DEVICE}")
    print(f"  {len(train_ds)} train / {len(val_ds)} val, {epochs} epochs\n")

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for feats, targets, mu, mask in train_loader:
            feats, targets = feats.to(DEVICE), targets.to(DEVICE)
            mu, mask = mu.to(DEVICE), mask.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = masked_mse_loss(model(feats, mu), targets, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, targets, mu, mask in val_loader:
                feats, targets = feats.to(DEVICE), targets.to(DEVICE)
                mu, mask = mu.to(DEVICE), mask.to(DEVICE)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    loss = masked_mse_loss(model(feats, mu), targets, mask)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        marker = " *" if val_loss < best_val else ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_path)
        print(f"  epoch {epoch:2d}  train={train_loss:.4f}  val={val_loss:.4f}  rmse={val_loss**0.5:.4f}{marker}")

    print(f"\nSaved best model to {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", type=str, default=str(OUTPUTS_DIR / "node_features"))
    ap.add_argument("--out", type=str, default=str(OUTPUTS_DIR / "node_time_true.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    train(Path(args.features_dir), out_path, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
