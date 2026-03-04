"""Configuration dataclasses for fastcxt models and training."""

from __future__ import annotations
from dataclasses import dataclass, replace


@dataclass
class FastCxtConfig:
    """Unified configuration for fastcxt Mamba encoder-decoder models."""

    # Mamba dimensions
    d_model: int = 256
    d_state: int = 64
    d_conv: int = 4
    expand: int = 2

    # Encoder / decoder depth
    n_enc_layers: int = 6
    n_dec_layers: int = 4

    # Input
    n_window_scales: int = 4
    n_channels: int = 2          # xor / xnor
    max_samples: int = 200       # zero-pad SFS to this size
    window_size: int = 2000      # base SFS window in bp
    n_windows: int = 500         # number of output windows per 1Mb block

    # Conditioning
    use_mutation_rate: bool = True

    # Tree topology
    use_trees: bool = False
    tree_embed_dim: int = 64

    # Output
    output_dim: int = 2          # (mu, log_sigma2) per window

    # Regularization
    dropout: float = 0.1

    # Runtime
    device: str = "cpu"
    batch_size: int = 128

    def for_inference(self, batch_size: int = 1, device: str = "cpu") -> FastCxtConfig:
        return replace(self, batch_size=batch_size, device=device)

    def for_training(self, batch_size: int = 128, device: str = "cuda") -> FastCxtConfig:
        return replace(self, batch_size=batch_size, device=device)


PRESETS: dict[str, FastCxtConfig] = {
    "small": FastCxtConfig(d_model=128, n_enc_layers=4, n_dec_layers=2),
    "base": FastCxtConfig(d_model=256, n_enc_layers=6, n_dec_layers=4),
    "large": FastCxtConfig(d_model=512, n_enc_layers=8, n_dec_layers=6),
    "base_trees": FastCxtConfig(d_model=256, n_enc_layers=6, n_dec_layers=4, use_trees=True),
}


@dataclass
class TrainingConfig:
    """Hyperparameters for Lightning training."""

    max_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 100
    lr_decay_iters: int = 150_000
    batch_size: int = 128
    grad_accum_steps: int = 4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    num_workers: int = 8
    prefetch_factor: int = 2
