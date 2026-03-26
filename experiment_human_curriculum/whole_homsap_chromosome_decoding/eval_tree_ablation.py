"""Quick visual ablation: TMRCA along genome with vs without tree features.

Usage (on Betty):
    python eval_tree_ablation.py \
        --checkpoint /path/to/ckpt \
        --dataset-path /path/to/processed_lazy_tsinfer \
        --output /path/to/ablation_plot.png
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastcxt.config import PRESETS, FastCxtConfig, TrainingConfig
from fastcxt.train import LitFastCxt
from fastcxt.dataset import LazyPairDataset
from dataclasses import replace
import os

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])


def load_model(ckpt_path, model_cfg):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    fixed = {k.replace("._orig_mod", ""): v for k, v in state_dict.items()}
    ckpt["state_dict"] = fixed
    torch.save(ckpt, "/tmp/_fixed_ckpt.ckpt")
    return LitFastCxt.load_from_checkpoint(
        "/tmp/_fixed_ckpt.ckpt", model_config=model_cfg,
        training_config=TrainingConfig().__dict__,
    )


def predict(lit_model, batch, zero_trees=False, device="cuda"):
    lit_model.eval()
    lit_model.to(device)
    with torch.inference_mode():
        batch = [b.to(device) for b in batch]
        x, y, mu_rate, tree_feats = batch
        W = lit_model.model_config.n_windows
        x = x[:, :, :W, :]
        y = y[:, :W]
        tree_feats = tree_feats[:, :W, :]
        if zero_trees:
            tree_feats = torch.zeros_like(tree_feats)
        pred = lit_model.model(x, mu_rate, tree_feats)
    return pred.cpu(), y.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--output", default="ablation_plot.png")
    args = ap.parse_args()

    model_cfg = PRESETS["base_trees_5k"].for_training(batch_size=args.n_samples)
    for dirpath, _, filenames in os.walk(args.dataset_path):
        if "tree_feats.npy" in filenames:
            tf = np.load(os.path.join(dirpath, "tree_feats.npy"), mmap_mode="r")
            model_cfg = replace(model_cfg, tree_feat_dim=tf.shape[-1])
            print(f"Tree feature dim: {tf.shape[-1]}")
            break

    lit_model = load_model(args.checkpoint, model_cfg)
    print(f"Loaded checkpoint")

    test_ds = LazyPairDataset(
        root=args.dataset_path, split="test",
        max_samples=model_cfg.max_samples,
        use_trees=True, tree_feat_dim=model_cfg.tree_feat_dim,
    )
    print(f"Test samples: {len(test_ds)}")

    # Grab one batch
    from torch.utils.data import DataLoader
    loader = DataLoader(test_ds, batch_size=args.n_samples, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    pred_trees, y = predict(lit_model, batch, zero_trees=False, device=args.device)
    pred_notrees, _ = predict(lit_model, batch, zero_trees=True, device=args.device)

    # Plot
    n = min(args.n_samples, pred_trees.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    window_size_kb = 0.2  # 200 bp in kb
    for i, ax in enumerate(axes):
        W = y.shape[1]
        x_pos = np.arange(W) * window_size_kb / 1000  # Mb

        mu_trees = pred_trees[i, :, 0].numpy()
        mu_notrees = pred_notrees[i, :, 0].numpy()
        true_y = y[i].numpy()

        ax.plot(x_pos, true_y, color="gray", alpha=0.5, linewidth=0.5, label="true TMRCA")
        ax.plot(x_pos, mu_trees, color="#4C72B0", linewidth=0.8, label="with trees")
        ax.plot(x_pos, mu_notrees, color="#C44E52", linewidth=0.8, label="SFS only")

        rmse_t = np.sqrt(np.mean((mu_trees - true_y) ** 2))
        rmse_nt = np.sqrt(np.mean((mu_notrees - true_y) ** 2))
        ax.set_ylabel("log TMRCA")
        ax.set_title(f"Sample {i+1}  |  RMSE: trees={rmse_t:.3f}, SFS-only={rmse_nt:.3f}", fontsize=10)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    axes[-1].set_xlabel("Position (Mb)")
    fig.suptitle("Tree Feature Ablation — 1 Mb AnoGam (5k windows, 200 bp)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
