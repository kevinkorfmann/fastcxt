#!/usr/bin/env python3
"""Train a NodeTimeModel and benchmark against pair-based FastCxtModel.

Demonstrates that predicting node times once + LCA lookups breaks the
O(n_pairs) scaling of the pair-based approach.

Usage:
    CUDA_VISIBLE_DEVICES=2 python train_and_benchmark_node_times.py
"""

from __future__ import annotations

import time
from pathlib import Path
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import tskit
from tqdm.auto import tqdm

from fastcxt.tree_utils import (
    extract_topology_features,
    extract_node_times,
    lca_lookup_batch,
)
from fastcxt.preprocess import estimate_mutation_rate
from fastcxt.node_time_model import NodeTimeModel
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_ts

EXPERIMENT_DIR = Path(__file__).resolve().parent
SIMS_DIR = EXPERIMENT_DIR / "sims/constant"
CHECKPOINT_BASE = EXPERIMENT_DIR / "lightning_logs/version_9/checkpoints/epoch=19-step=4680.ckpt"
DEVICE = "cuda:0"
WINDOW_SIZE = 2000
SEQ_LEN = 1_000_000
N_WINDOWS = SEQ_LEN // WINDOW_SIZE
MUTATION_RATE = 1e-8


# ── Dataset ──────────────────────────────────────────────────────────────

class NodeTimeDataset(Dataset):
    def __init__(self, feats: np.ndarray, targets: np.ndarray,
                 mutation_rates: np.ndarray):
        self.feats = feats
        self.targets = targets
        self.mutation_rates = mutation_rates

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, i):
        return (
            torch.as_tensor(self.feats[i], dtype=torch.float32),
            torch.as_tensor(self.targets[i], dtype=torch.float32),
            torch.as_tensor([self.mutation_rates[i]], dtype=torch.float32),
        )


# ── Preprocessing ────────────────────────────────────────────────────────

def preprocess(max_files: int = 1000):
    """Extract tree topology features and node times from tree sequences."""
    ts_files = sorted(SIMS_DIR.glob("*.trees"))[:max_files]
    print(f"Preprocessing {len(ts_files)} tree sequences ...")

    ts0 = tskit.load(str(ts_files[0]))
    n_samples = ts0.num_samples
    max_internal = n_samples - 1
    from fastcxt.tree_utils import FEATS_PER_NODE
    tree_feat_dim = max_internal * FEATS_PER_NODE

    all_feats = np.zeros((len(ts_files), N_WINDOWS, tree_feat_dim), dtype=np.float32)
    all_times = np.zeros((len(ts_files), N_WINDOWS, max_internal), dtype=np.float32)
    all_mu = np.zeros(len(ts_files), dtype=np.float32)

    for i, f in enumerate(tqdm(ts_files, desc="Extracting")):
        ts = tskit.load(str(f))
        all_feats[i] = extract_topology_features(
            ts, n_windows=N_WINDOWS, window_size=WINDOW_SIZE, max_internal=max_internal,
        )
        all_times[i] = extract_node_times(
            ts, n_windows=N_WINDOWS, window_size=WINDOW_SIZE, max_internal=max_internal,
        )
        all_mu[i] = np.log(estimate_mutation_rate(ts))

    split = int(0.9 * len(ts_files))
    return (
        all_feats[:split], all_times[:split], all_mu[:split],
        all_feats[split:], all_times[split:], all_mu[split:],
        n_samples, max_internal, tree_feat_dim,
    )


# ── Training ─────────────────────────────────────────────────────────────

def train_model(
    model: NodeTimeModel,
    train_feats, train_times, train_mu,
    val_feats, val_times, val_mu,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
):
    model.to(DEVICE)
    train_ds = NodeTimeDataset(train_feats, train_times, train_mu)
    val_ds = NodeTimeDataset(val_feats, val_times, val_mu)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining NodeTimeModel ({n_params:,} params)")
    print(f"  {len(train_ds)} train / {len(val_ds)} val samples, {epochs} epochs\n")

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for feats, targets, mu in train_loader:
            feats = feats.to(DEVICE)
            targets = targets.to(DEVICE)
            mu = mu.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(feats, mu)
                loss = nn.functional.mse_loss(pred, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, targets, mu in val_loader:
                feats = feats.to(DEVICE)
                targets = targets.to(DEVICE)
                mu = mu.to(DEVICE)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model(feats, mu)
                    loss = nn.functional.mse_loss(pred, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        rmse = val_loss ** 0.5
        marker = " *" if val_loss < best_val else ""
        best_val = min(best_val, val_loss)
        print(f"  epoch {epoch:2d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_rmse={rmse:.4f}{marker}")

    return model


# ── Benchmark ────────────────────────────────────────────────────────────

def benchmark(
    node_model: NodeTimeModel,
    base_checkpoint: Path,
    ts_path: Path,
    n_samples: int,
    max_internal: int,
    tree_feat_dim: int,
):
    print("\n" + "=" * 80)
    print("BENCHMARK: pair-based O(P) vs node-time O(1) inference")
    print("=" * 80)

    ts = tskit.load(str(ts_path))
    seq_len = int(ts.sequence_length)
    blocks = [(0, seq_len)]

    lit_base = LitFastCxt.load_from_checkpoint(str(base_checkpoint), map_location="cpu")
    base_model = lit_base.model

    tree_feats_np = extract_topology_features(
        ts, n_windows=N_WINDOWS, window_size=WINDOW_SIZE, max_internal=max_internal,
    )
    tree_feats_t = torch.as_tensor(tree_feats_np[np.newaxis], dtype=torch.float32).to(DEVICE)
    log_mu_t = torch.tensor([[np.log(MUTATION_RATE)]], dtype=torch.float32).to(DEVICE)

    node_model.eval()
    base_model.to(DEVICE)
    base_model.eval()
    with torch.inference_mode():
        _ = node_model(tree_feats_t, log_mu_t)
    translate_from_ts(
        ts, base_model, blocks=blocks, pivot_pairs=[(0, 1)],
        mutation_rate=MUTATION_RATE, device=DEVICE, progress=False,
    )

    all_pairs = list(combinations(range(n_samples), 2))
    pair_counts = [1, 3, 10, 25, 50, 100, 250, 500, 1000, len(all_pairs)]
    pair_counts = sorted(set(p for p in pair_counts if p <= len(all_pairs)))

    REPEATS = 3
    results = []

    for n_pairs in pair_counts:
        pivot_pairs = all_pairs[:n_pairs]

        times_base = []
        for rep in range(REPEATS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            translate_from_ts(
                ts, base_model, blocks=blocks, pivot_pairs=list(pivot_pairs),
                mutation_rate=MUTATION_RATE, device=DEVICE, progress=False,
            )
            torch.cuda.synchronize()
            times_base.append(time.perf_counter() - t0)

        times_node = []
        for rep in range(REPEATS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                node_times_pred = node_model(tree_feats_t, log_mu_t)
            node_times_np = node_times_pred[0].cpu().numpy()
            _ = lca_lookup_batch(
                ts, node_times_np, list(pivot_pairs),
                n_windows=N_WINDOWS, window_size=WINDOW_SIZE,
            )
            torch.cuda.synchronize()
            times_node.append(time.perf_counter() - t0)

        mb = np.mean(times_base) * 1000
        sb = np.std(times_base) * 1000
        mn = np.mean(times_node) * 1000
        sn = np.std(times_node) * 1000
        speedup = mb / mn if mn > 0 else float("inf")
        results.append((n_pairs, mb, sb, mn, sn, speedup))
        print(f"  {n_pairs:5d} pairs: pair-based {mb:7.0f} ± {sb:3.0f} ms | "
              f"node-time {mn:7.0f} ± {sn:3.0f} ms | {speedup:5.1f}x speedup")

    print("\n" + "=" * 80)
    print(f"{'pairs':>6} | {'pair-based (ms)':>16} | {'node-time (ms)':>16} | {'speedup':>8}")
    print("-" * 80)
    for n_pairs, mb, sb, mn, sn, speedup in results:
        print(f"{n_pairs:6d} | {mb:8.0f} ± {sb:5.0f} | {mn:8.0f} ± {sn:5.0f} | {speedup:6.1f}x")
    print("=" * 80)

    max_pairs = results[-1]
    print(f"\nAt {max_pairs[0]} pairs (all pairs for {n_samples} samples):")
    print(f"  pair-based: {max_pairs[1]:.0f} ms")
    print(f"  node-time:  {max_pairs[3]:.0f} ms")
    print(f"  speedup:    {max_pairs[5]:.1f}x")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    (train_feats, train_times, train_mu,
     val_feats, val_times, val_mu,
     n_samples, max_internal, tree_feat_dim) = preprocess(max_files=1000)

    model = NodeTimeModel(
        tree_feat_dim=tree_feat_dim,
        n_internal=max_internal,
        d_model=256,
        n_layers=4,
        n_windows=N_WINDOWS,
    )

    model = train_model(
        model,
        train_feats, train_times, train_mu,
        val_feats, val_times, val_mu,
        epochs=50,
    )
    torch.save(model.state_dict(), EXPERIMENT_DIR / "node_time_model.pt")
    print(f"\nSaved model to {EXPERIMENT_DIR / 'node_time_model.pt'}")

    benchmark(
        model,
        CHECKPOINT_BASE,
        SIMS_DIR / "ts_00000000.trees",
        n_samples, max_internal, tree_feat_dim,
    )


if __name__ == "__main__":
    main()
