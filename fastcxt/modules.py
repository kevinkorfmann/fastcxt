"""Core neural network modules for fastcxt.

BiMambaBlock     -- bidirectional Mamba (forward + backward + projection)
InputProjection  -- multi-scale SFS to d_model, variable sample size
FiLMLayer        -- feature-wise linear modulation conditioned on mutation rate
UncertaintyHead  -- regression head producing (mu, log_sigma2) per window
TreeEncoder      -- encode tsinfer coalescence topology as per-window features
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba2


# ---------------------------------------------------------------------------
# Bidirectional Mamba block
# ---------------------------------------------------------------------------

class BiMambaBlock(nn.Module):
    """Run Mamba forward and backward, merge via learned projection."""

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba2(d_model=d_model, d_state=d_state,
                                d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba2(d_model=d_model, d_state=d_state,
                                d_conv=d_conv, expand=expand)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        h = self.norm(x)
        h_fwd = self.mamba_fwd(h)
        h_bwd = self.mamba_bwd(h.flip(1)).flip(1)
        h = self.merge(torch.cat([h_fwd, h_bwd], dim=-1))
        return x + self.drop(h)


# ---------------------------------------------------------------------------
# Input projection: multi-scale SFS -> d_model
# ---------------------------------------------------------------------------

class InputProjection(nn.Module):
    """Project multi-scale SFS features into the model latent space.

    Handles variable sample sizes by zero-padding to max_samples.

    Input:  (B, n_channels, n_scales, n_windows, n_samples)
    Output: (B, n_windows, d_model)
    """

    def __init__(self, d_model: int, max_samples: int = 200,
                 n_channels: int = 2, n_scales: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.max_samples = max_samples
        self.n_channels = n_channels
        self.n_scales = n_scales

        flat_dim = n_channels * n_scales * max_samples
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, S, W, N = x.shape
        if N < self.max_samples:
            pad = x.new_zeros(B, C, S, W, self.max_samples - N)
            x = torch.cat([x, pad], dim=-1)
        elif N > self.max_samples:
            x = x[..., :self.max_samples]
        # (B, C, S, W, max_samples) -> (B, W, C*S*max_samples)
        x = x.permute(0, 3, 1, 2, 4).reshape(B, W, -1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# FiLM conditioning (Feature-wise Linear Modulation)
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Condition hidden states on a scalar (log mutation rate).

    Projects the scalar to per-dimension (gamma, beta) and applies
    h = gamma * h + beta.
    """

    def __init__(self, d_model: int, cond_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """h: (B, L, D), cond: (B, 1) -> (B, L, D)"""
        gb = self.net(cond)                   # (B, 2D)
        gamma, beta = gb.chunk(2, dim=-1)     # each (B, D)
        gamma = gamma.unsqueeze(1)            # (B, 1, D)
        beta = beta.unsqueeze(1)
        return gamma * h + beta


# ---------------------------------------------------------------------------
# Uncertainty head: (mu, log_sigma2) per window
# ---------------------------------------------------------------------------

class UncertaintyHead(nn.Module):
    """Predict mean and log-variance of log-TMRCA per window."""

    def __init__(self, d_model: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),   # (mu, log_sigma2)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, L, D) -> (B, L, 2)"""
        return self.head(h)


# ---------------------------------------------------------------------------
# Tree topology encoder
# ---------------------------------------------------------------------------

class TreeEncoder(nn.Module):
    """Encode tsinfer coalescence-order topology as per-window features.

    The tree topology is represented as a per-window tensor of coalescence
    rank vectors: for each local tree (spanning one or more windows), we
    store the sorted merge order of sample pairs.  This is a (B, W, tree_feat_dim)
    tensor produced during preprocessing.

    This module projects those features and adds them to the SFS embedding.
    """

    def __init__(self, tree_feat_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(tree_feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, tree_feats: torch.Tensor) -> torch.Tensor:
        """tree_feats: (B, W, tree_feat_dim) -> (B, W, d_model)"""
        return self.proj(tree_feats)
