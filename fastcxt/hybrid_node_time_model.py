"""Hybrid node-time model: genetic signal + tree topology → node times.

Combines the accuracy of the pairwise model (which sees actual genetic
variation via the SFS) with the speed of the NodeTimeModel (which predicts
all internal node times in a single forward pass).

Key idea: instead of per-pair XOR/XNOR SFS (which costs O(n²) pairs), we
compute the *full-sample* allele frequency spectrum per window — an O(n)
operation that captures the same population-genetic signal.  This is fused
with tree topology features and processed through a BiMamba backbone to
predict all internal node times.  Pairwise TMRCA is then recovered via
O(log n) LCA lookups, exactly like the original NodeTimeModel.

Inputs
------
sfs_features : (B, W, n_samples)     per-window allele frequency spectrum
tree_feats   : (B, W, tree_feat_dim) coalescence-order topology features
mutation_rate: (B, 1)                 log mutation rate

Output
------
node_times   : (B, W, n_internal)    predicted log(time) per internal node
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastcxt.modules import BiMambaBlock, FiLMLayer


class HybridNodeTimeModel(nn.Module):
    """Predict internal node times from SFS + tree topology features.

    Two input branches (SFS and topology) are projected independently,
    fused additively, and processed through a BiMamba backbone with FiLM
    conditioning on mutation rate.
    """

    def __init__(
        self,
        n_samples: int,
        tree_feat_dim: int,
        n_internal: int,
        d_model: int = 256,
        n_layers: int = 4,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        n_windows: int = 500,
    ):
        super().__init__()
        self.n_internal = n_internal
        self.n_samples = n_samples

        self.sfs_proj = nn.Sequential(
            nn.Linear(n_samples, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.tree_proj = nn.Sequential(
            nn.Linear(tree_feat_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, n_windows, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            BiMambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        self._film_indices = {0, n_layers - 1}
        self.films = nn.ModuleDict({
            str(i): FiLMLayer(d_model) for i in self._film_indices
        })

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_internal),
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        sfs_features: torch.Tensor,
        tree_feats: torch.Tensor,
        mutation_rate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        sfs_features:  (B, W, n_samples)     allele frequency spectrum
        tree_feats:    (B, W, tree_feat_dim)  topology features
        mutation_rate: (B, 1)                 log mutation rate
        """
        h_sfs = self.sfs_proj(sfs_features)
        h_tree = self.tree_proj(tree_feats)

        h = h_sfs + h_tree
        W = h.shape[1]
        h = h + self.pos_embed[:, :W, :]

        for i, block in enumerate(self.blocks):
            h = block(h)
            if str(i) in self.films and mutation_rate is not None:
                h = self.films[str(i)](h, mutation_rate)

        return self.head(h)
