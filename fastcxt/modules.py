"""Core neural network modules for fastcxt.

BiMambaBlock              -- bidirectional Mamba (forward + backward + projection)
MultiScaleInputProjection -- learned multi-scale SFS embedding via 1D convolutions
FiLMLayer                 -- feature-wise linear modulation conditioned on mutation rate
UncertaintyHead           -- regression head producing (mu, log_sigma2) per window
TreeEncoder               -- encode tsinfer coalescence topology as per-window features
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
    """Bidirectional Mamba with feedforward network.

    Sequence mixing (BiMamba) followed by channel mixing (FFN),
    each with pre-norm and residual connections.
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1, ffn_expand: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba_fwd = Mamba2(d_model=d_model, d_state=d_state,
                                d_conv=d_conv, expand=expand)
        self.mamba_bwd = Mamba2(d_model=d_model, d_state=d_state,
                                d_conv=d_conv, expand=expand)
        self.merge = nn.Linear(2 * d_model, d_model)
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        h = self.norm1(x)
        h_fwd = self.mamba_fwd(h)
        h_bwd = self.mamba_bwd(h.flip(1)).flip(1)
        h = self.merge(torch.cat([h_fwd, h_bwd], dim=-1))
        x = x + self.drop(h)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Input projection: single-scale SFS -> d_model via learned multi-scale convs
# ---------------------------------------------------------------------------

class MultiScaleInputProjection(nn.Module):
    """Embed single-scale SFS into d_model using learned multi-scale 1D convolutions.

    The frequency axis is compressed gradually: per-channel linear projection
    reduces the SFS bins first, preserving the XOR/XNOR structure.  The
    compressed channels are then merged and fed through parallel 1D conv
    branches at different genomic scales.

    Input:  (B, n_channels, n_windows, n_samples)
    Output: (B, n_windows, d_model)
    """

    def __init__(self, d_model: int, max_samples: int = 200,
                 n_channels: int = 2, stem_channels: int = 64,
                 conv_channels: int = 64,
                 kernel_sizes: tuple[int, ...] = (3, 11, 31, 63),
                 dropout: float = 0.1):
        super().__init__()
        self.max_samples = max_samples
        self.n_channels = n_channels

        freq_hidden = max(stem_channels // n_channels, 32)

        # Per-channel frequency compression: (max_samples) -> (freq_hidden)
        self.freq_compress = nn.Sequential(
            nn.Linear(max_samples, freq_hidden),
            nn.GELU(),
        )

        # Merge channels: (n_channels * freq_hidden) -> stem_channels
        self.channel_merge = nn.Sequential(
            nn.Conv1d(n_channels * freq_hidden, stem_channels, kernel_size=1),
            nn.GELU(),
        )

        # Parallel branches at different spatial scales
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            self.branches.append(nn.Sequential(
                nn.Conv1d(stem_channels, conv_channels, kernel_size=k,
                          padding=k // 2),
                nn.GELU(),
            ))

        total_ch = conv_channels * len(kernel_sizes)
        self.proj = nn.Sequential(
            nn.Linear(total_ch, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, W, N = x.shape

        if N < self.max_samples:
            pad = x.new_zeros(B, C, W, self.max_samples - N)
            x = torch.cat([x, pad], dim=-1)
        elif N > self.max_samples:
            x = x[..., :self.max_samples]

        # Per-channel frequency compression: (B, C, W, max_samples) -> (B, C, W, freq_hidden)
        x = self.freq_compress(x)
        # Reshape for Conv1d: (B, C * freq_hidden, W)
        x = x.reshape(B, -1, W)
        x = self.channel_merge(x)                      # (B, stem_ch, W)

        outs = [branch(x) for branch in self.branches]  # each (B, conv_ch, W)
        x = torch.cat(outs, dim=1)                      # (B, total_ch, W)

        x = x.permute(0, 2, 1)                          # (B, W, total_ch)
        return self.proj(x)                              # (B, W, d_model)


# ---------------------------------------------------------------------------
# FiLM conditioning (Feature-wise Linear Modulation)
# ---------------------------------------------------------------------------

class FiLMLayer(nn.Module):
    """Condition hidden states on a scalar (log mutation rate).

    Projects the scalar to per-dimension (gamma, beta) and applies
    an additive modulation: h = h + (gamma * h + beta), preserving
    the residual stream.
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
        return h + gamma * h + beta


# ---------------------------------------------------------------------------
# Uncertainty head: (mu, log_sigma2) per window
# ---------------------------------------------------------------------------

class UncertaintyHead(nn.Module):
    """Predict mean and log-variance of log-TMRCA per window.

    Separate MLP branches for mean and variance so they can
    specialize independently.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        mid = d_model // 2
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, mid), nn.GELU(), nn.Linear(mid, 1),
        )
        self.var_head = nn.Sequential(
            nn.Linear(d_model, mid), nn.GELU(), nn.Linear(mid, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, L, D) -> (B, L, 2)"""
        h = self.norm(h)
        return torch.cat([self.mu_head(h), self.var_head(h)], dim=-1)


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
