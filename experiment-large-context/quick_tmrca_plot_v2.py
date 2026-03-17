"""Quick TMRCA prediction along a genome from the large-context checkpoint."""
import sys, json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import replace

from fastcxt.config import FastCxtConfig, TrainingConfig
from fastcxt.train import LitFastCxt
from fastcxt.sfs import build_sfs_tensor

torch.serialization.add_safe_globals([FastCxtConfig, TrainingConfig])

CKPT = "tsinfer_4k_logs/lightning_logs/version_0/checkpoints/epoch=29-step=20880.ckpt"
TEST_DIR = "sims/processed_lazy_tsinfer/test/n100/ts_00000003_i203"
N_WINDOWS = 2000  # model's window count

# --- Load model ---
lit = LitFastCxt.load_from_checkpoint(CKPT, map_location="cpu")
model = lit.model.eval().cuda()
model_cfg = lit.hparams["model_config"]

# --- Load test sample ---
gm = np.load(f"{TEST_DIR}/genotypes.npy")
pos = np.load(f"{TEST_DIR}/positions.npy")
pairs = np.load(f"{TEST_DIR}/pairs.npy")
y_all = np.load(f"{TEST_DIR}/y.npy")
tf_all = np.load(f"{TEST_DIR}/tree_feats.npy")

with open(f"{TEST_DIR}/meta.json") as f:
    meta = json.load(f)

mutation_rate = meta["mutation_rate"]
seq_len = meta["sequence_length"]
window_size = meta["window_size"]

# Pick 3 diverse pairs
pair_indices = [0, 50, 150]

fig, axes = plt.subplots(len(pair_indices), 1, figsize=(14, 3.5 * len(pair_indices)),
                         sharex=True)

genomic_pos_kb = np.arange(N_WINDOWS) * window_size / 1000  # in kb

for ax, pidx in zip(axes, pair_indices):
    pa, pb = int(pairs[pidx, 0]), int(pairs[pidx, 1])

    # Build SFS
    Xi = build_sfs_tensor(gm, pos, pa, pb,
                          sequence_length=seq_len, window_size=window_size)
    Xi = torch.as_tensor(Xi, dtype=torch.float32).unsqueeze(0)  # (1, 2, W, N)

    # Truncate to model's window count
    Xi = Xi[:, :, :N_WINDOWS, :]
    N = Xi.shape[-1]
    if N < model_cfg.max_samples:
        Xi = torch.nn.functional.pad(Xi, (0, model_cfg.max_samples - N))

    # Tree features (shared, truncate to N_WINDOWS)
    tf = torch.as_tensor(tf_all[:N_WINDOWS], dtype=torch.float32).unsqueeze(0)

    mu_rate = torch.tensor([[np.log(mutation_rate)]], dtype=torch.float32)

    # Predict
    with torch.inference_mode():
        out = model(Xi.cuda(), mu_rate.cuda(), tf.cuda())
    pred_mu = out[0, :, 0].cpu().numpy()
    pred_std = torch.exp(0.5 * torch.clamp(out[0, :, 1], -10, 10)).cpu().numpy()

    # Ground truth (first N_WINDOWS)
    y_true = y_all[pidx, :N_WINDOWS].astype(np.float32)

    ax.plot(genomic_pos_kb, y_true, color="black", alpha=0.6, lw=0.8, label="True log-TMRCA")
    ax.plot(genomic_pos_kb, pred_mu, color="tab:blue", lw=0.8, label="Predicted")
    ax.fill_between(genomic_pos_kb,
                     pred_mu - 1.96 * pred_std,
                     pred_mu + 1.96 * pred_std,
                     alpha=0.2, color="tab:blue", label="95% CI")
    ax.set_ylabel("log-TMRCA")
    ax.set_title(f"Pair ({pa}, {pb})", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)

axes[-1].set_xlabel("Genomic position (kb)")
fig.suptitle(f"TMRCA prediction — large-context tsinfer (epoch 28)", fontsize=12)
fig.tight_layout()
fig.savefig("quick_tmrca_prediction_v2.png", dpi=150)
print(f"Saved quick_tmrca_prediction_v2.png")
