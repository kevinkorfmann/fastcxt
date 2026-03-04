"""Training script for fastcxt models.

Single-pass forward with Beta-NLL loss (Seitzer et al., 2022).
No autoregressive generation.

Usage:
    python -m fastcxt.train --model base --dataset-path /path/to/processed --gpus 0 1 2
    python -m fastcxt.train --model base_trees --dataset-path /path/to/processed --gpus 0
"""

from __future__ import annotations

import math
import argparse

import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader

from fastcxt.config import PRESETS, TrainingConfig
from fastcxt.model import FastCxtModel
from fastcxt.dataset import PairDataset, TreeAugmentedPairDataset


# ---------------------------------------------------------------------------
# Beta-NLL loss  (Seitzer et al., 2022)
# ---------------------------------------------------------------------------

def beta_nll_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Heteroscedastic Gaussian NLL with beta-weighting to prevent
    early variance inflation.

    When beta=0 this is standard Gaussian NLL.  beta=0.5 (default)
    down-weights the reconstruction term by detached sigma^beta,
    preventing the model from trivially reducing loss by inflating
    the predicted variance.

    Parameters
    ----------
    pred : (B, W, 2) where [..., 0] = mu, [..., 1] = log_sigma2
    target : (B, W)  continuous log-TMRCA
    beta : float in [0, 1], default 0.5

    Returns
    -------
    scalar loss
    """
    mu = pred[..., 0]
    log_sigma2 = torch.clamp(pred[..., 1], min=-10, max=10)
    sigma2 = torch.exp(log_sigma2)
    mse = (target - mu) ** 2
    weight = sigma2.detach() ** beta
    return 0.5 * (weight * (log_sigma2 + mse / sigma2)).mean()


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class LitFastCxt(L.LightningModule):
    def __init__(self, model_config, training_config: dict | None = None):
        super().__init__()
        self.model = FastCxtModel(model_config)
        self.training_config = training_config or TrainingConfig().__dict__
        if isinstance(self.training_config, TrainingConfig):
            self.training_config = self.training_config.__dict__
        self.save_hyperparameters(ignore=["model"])

    def _forward(self, batch):
        if len(batch) == 4:
            x, y, mu_rate, tree_feats = batch
        else:
            x, y, mu_rate = batch
            tree_feats = None
        pred = self.model(x, mu_rate, tree_feats)
        return pred, y

    def training_step(self, batch, batch_idx):
        pred, y = self._forward(batch)
        loss = beta_nll_loss(pred, y)
        mu = pred[..., 0]
        rmse = ((mu - y) ** 2).mean().sqrt()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_rmse", rmse, prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pred, y = self._forward(batch)
        loss = beta_nll_loss(pred, y)
        mu = pred[..., 0]
        rmse = ((mu - y) ** 2).mean().sqrt()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_rmse", rmse, prog_bar=True, sync_dist=True)

        log_sigma2 = torch.clamp(pred[..., 1], min=-10, max=10)
        sigma = torch.exp(0.5 * log_sigma2)
        coverage = ((y >= mu - 1.96 * sigma) & (y <= mu + 1.96 * sigma)).float().mean()
        self.log("val_coverage_95", coverage, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        tc = self.training_config
        opt = self.model.configure_optimizers(
            weight_decay=tc["weight_decay"],
            learning_rate=tc["max_lr"],
            betas=tc["betas"],
            device_type=self.device.type,
        )
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=self._lr_lambda)
        return [opt], [{"scheduler": sch, "interval": "step", "frequency": 1}]

    def _lr_lambda(self, step):
        tc = self.training_config
        if step < tc["warmup_iters"]:
            return float(step) / max(1, tc["warmup_iters"])
        if step > tc["lr_decay_iters"]:
            return tc["min_lr"] / tc["max_lr"]
        ratio = (step - tc["warmup_iters"]) / (tc["lr_decay_iters"] - tc["warmup_iters"])
        coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
        return tc["min_lr"] / tc["max_lr"] + coeff * (1.0 - tc["min_lr"] / tc["max_lr"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train fastcxt models")
    ap.add_argument("--model", type=str, default="base", choices=list(PRESETS))
    ap.add_argument("--dataset-path", type=str, required=True)
    ap.add_argument("--gpus", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--log-dir", type=str, default=None)
    args = ap.parse_args()

    model_cfg = PRESETS[args.model].for_training(batch_size=args.batch_size)

    DatasetCls = TreeAugmentedPairDataset if model_cfg.use_trees else PairDataset
    ds_kwargs = dict(root=args.dataset_path, max_samples=model_cfg.max_samples)
    if model_cfg.use_trees:
        ds_kwargs["tree_feat_dim"] = (model_cfg.max_samples - 1) * 3

    train_ds = DatasetCls(split="train", **ds_kwargs)
    test_ds = DatasetCls(split="test", **ds_kwargs)
    print(f"Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")

    n_gpus = len(args.gpus)
    steps_per_epoch = len(train_ds) // (args.batch_size * n_gpus * args.grad_accum)
    total_opt_steps = steps_per_epoch * args.epochs

    train_cfg = TrainingConfig(
        max_lr=args.lr,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        num_workers=args.workers,
        lr_decay_iters=total_opt_steps,
        warmup_iters=min(100, total_opt_steps // 10),
    )
    print(f"Scheduler: {train_cfg.warmup_iters} warmup -> cosine decay over {total_opt_steps} steps")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers, pin_memory=True,
        shuffle=True, persistent_workers=True,
        prefetch_factor=train_cfg.prefetch_factor, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=train_cfg.batch_size,
        num_workers=train_cfg.num_workers, pin_memory=True,
        shuffle=False, persistent_workers=True,
        prefetch_factor=train_cfg.prefetch_factor, drop_last=True,
    )

    if args.checkpoint:
        lit_model = LitFastCxt.load_from_checkpoint(
            args.checkpoint, model_config=model_cfg,
            training_config=train_cfg.__dict__,
        )
    else:
        lit_model = LitFastCxt(model_cfg, training_config=train_cfg.__dict__)

    torch.set_float32_matmul_precision("medium")
    trainer_kwargs = dict(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=args.gpus,
        precision="bf16-mixed",
        strategy="ddp" if len(args.gpus) > 1 else "auto",
        accumulate_grad_batches=train_cfg.grad_accum_steps,
    )
    if args.log_dir:
        trainer_kwargs["default_root_dir"] = args.log_dir

    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(model=lit_model, train_dataloaders=train_loader,
                val_dataloaders=test_loader)


if __name__ == "__main__":
    main()
