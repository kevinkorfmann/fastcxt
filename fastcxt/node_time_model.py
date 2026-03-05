"""Node-time prediction model for O(1) forward-pass scaling.

Instead of predicting pairwise TMRCA (one forward pass per pair), this model
predicts internal node times from tree topology features in a single forward
pass.  Pairwise TMRCA is then recovered via O(log n) LCA lookups per pair,
reducing the cost from O(n^2) model evaluations to O(1) + O(n^2 log n) tree
lookups.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from fastcxt.modules import BiMambaBlock, FiLMLayer


class NodeTimeModel(nn.Module):
    """Predict internal node times from tree topology features.

    Input:  tree_feats (B, W, tree_feat_dim) -- coalescence order features
            mutation_rate (B, 1)             -- log mutation rate (time-scale)
    Output: (B, W, n_internal)               -- predicted log(time) per node
    """

    def __init__(
        self,
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

        self.proj_in = nn.Sequential(
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

        self.film = FiLMLayer(d_model)

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
        tree_feats: torch.Tensor,
        mutation_rate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        tree_feats:    (B, W, tree_feat_dim)
        mutation_rate: (B, 1) log mutation rate -- calibrates the time scale
        """
        h = self.proj_in(tree_feats)
        W = h.shape[1]
        h = h + self.pos_embed[:, :W, :]
        for block in self.blocks:
            h = block(h)
        if mutation_rate is not None:
            h = self.film(h, mutation_rate)
        return self.head(h)
