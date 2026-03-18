#!/usr/bin/env python3
"""Cache all model predictions as .npz files for offline figure generation.

Run this ONCE on a machine with GPU + torch + msprime, then use
plot_figures.py anywhere (numpy + matplotlib only).

Usage:
    python paper/figures/cache_predictions.py

Outputs everything to paper/figures/cached_data/
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.serialization
import msprime

from fastcxt.config import FastCxtConfig, TrainingConfig
from fastcxt.train import LitFastCxt
from fastcxt.translate import translate_from_ts
from fastcxt.preprocess import windowed_tmrca

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
OUT = Path("paper/figures/cached_data")


def load_model(ckpt, device="cpu"):
    lit = LitFastCxt.load_from_checkpoint(str(ckpt), map_location="cpu")
    m = lit.model.to(device)
    m.eval()
    return m


def run_model(model, ts, use_tsinfer=False, mutation_rate=1e-8, window_size=2000):
    seq_len = int(ts.sequence_length)
    pairs = [(0, 1), (2, 3), (4, 5)]
    pairs = [(a, b) for a, b in pairs if b < ts.num_samples]
    blocks = [(0, seq_len)]
    kw = dict(blocks=blocks, pivot_pairs=pairs, mutation_rate=mutation_rate,
              device=DEVICE, progress=False)
    if use_tsinfer:
        kw["use_tsinfer"] = True
    means, varis, _ = translate_from_ts(ts, model, **kw)
    true_log = []
    for sa, sb in pairs:
        raw = windowed_tmrca(ts, sa, sb, window_size, seq_len)
        true_log.append(np.log(np.clip(raw, 1e-10, None)))
    return np.array(true_log), means, varis


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- Human-like simulations ----
    print("Simulating human-like test data ...")
    ts_human = msprime.sim_ancestry(
        50, sequence_length=1e6, recombination_rate=1e-8,
        population_size=10000, random_seed=42)
    ts_human = msprime.sim_mutations(ts_human, rate=1e-8, random_seed=42)

    ckpts = {
        "base": "experiment/lightning_logs/version_1/checkpoints/epoch=29-step=20880.ckpt",
        "tree_true": "experiment/tree_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt",
        "tree_tsinfer": "experiment/tsinfer_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt",
    }

    for key, ckpt in ckpts.items():
        print(f"  Running {key} ...")
        model = load_model(ckpt, device=DEVICE)
        use_tsinfer = (key == "tree_tsinfer")
        true_arr, pred_arr, var_arr = run_model(model, ts_human,
                                                 use_tsinfer=use_tsinfer)
        np.savez(OUT / f"human_{key}.npz",
                 true=true_arr, pred=pred_arr, var=var_arr,
                 window_size=2000, seq_len=1_000_000)
        print(f"    -> human_{key}.npz  shape={true_arr.shape}")

    # ---- AnoGam 100kb ----
    print("Simulating AnoGam 100kb test data ...")
    from fastcxt.mosquito import simulate_anogam
    ts_anogam = simulate_anogam(seed=123, n_samples=50, segment_length=100_000)

    anogam_ckpt = "experiment_mossies/tsinfer_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt"
    print("  Running AnoGam 100kb ...")
    model_a = load_model(anogam_ckpt, device=DEVICE)
    true_a, pred_a, var_a = run_model(model_a, ts_anogam, use_tsinfer=True,
                                       mutation_rate=3.5e-9, window_size=200)
    np.savez(OUT / "anogam_100kb.npz",
             true=true_a, pred=pred_a, var=var_a,
             window_size=200, seq_len=100_000)
    print(f"    -> anogam_100kb.npz  shape={true_a.shape}")

    # ---- AnoGam 400kb (copy from existing) ----
    print("Copying AnoGam 400kb predictions ...")
    src = Path("experiment-large-context/tmrca_predictions")
    all_true, all_pred, all_var = [], [], []
    for f in sorted(src.glob("pair_*.npz")):
        d = np.load(f)
        all_true.append(d["y_true"])
        all_pred.append(d["pred_mu"])
        all_var.append(d["pred_std"] ** 2)
    pos_kb = np.load(src / "genomic_pos_kb.npy")
    np.savez(OUT / "anogam_400kb.npz",
             true=np.stack(all_true), pred=np.stack(all_pred),
             var=np.stack(all_var), pos_kb=pos_kb,
             window_size=200, seq_len=400_000)
    print(f"    -> anogam_400kb.npz")

    print(f"\nAll cached data saved to {OUT}/")
    print("You can now run: python paper/figures/plot_figures.py")


if __name__ == "__main__":
    main()
