"""FastCxtModel -- bidirectional Mamba encoder-decoder for TMRCA inference.

Single forward pass produces (mu, log_sigma2) for every genomic window.
No autoregressive generation, no discrete bins.
"""

from __future__ import annotations

import math
import inspect

import torch
import torch.nn as nn

from fastcxt.modules import (
    BiMambaBlock,
    MultiScaleInputProjection,
    FiLMLayer,
    UncertaintyHead,
    TreeEncoder,
)


class FastCxtModel(nn.Module):
    """Bidirectional Mamba encoder-decoder for pairwise coalescence times.

    Encoder: MultiScaleInputProjection -> [BiMambaBlock + FiLM] x n_enc_layers
    Decoder: [BiMambaBlock] x n_dec_layers  (with skip connections from encoder)
    Head:    UncertaintyHead -> (mu, log_sigma2) per window
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.d_model

        # --- input projection (learned multi-scale convolutions) ---
        self.input_proj = MultiScaleInputProjection(
            d_model=d,
            max_samples=config.max_samples,
            n_channels=config.n_channels,
            stem_channels=config.stem_channels,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            dropout=config.dropout,
        )

        # --- optional tree encoder ---
        self.tree_encoder = None
        if config.use_trees:
            self.tree_encoder = TreeEncoder(
                tree_feat_dim=config.tree_feat_dim,
                d_model=d,
                dropout=config.dropout,
            )

        # --- encoder ---
        self.enc_blocks = nn.ModuleList([
            BiMambaBlock(d, config.d_state, config.d_conv,
                         config.expand, config.dropout)
            for _ in range(config.n_enc_layers)
        ])
        # FiLM conditioning only at first and last encoder layers
        self._film_indices = {0, config.n_enc_layers - 1}
        self.enc_films = nn.ModuleDict({
            str(i): FiLMLayer(d) for i in self._film_indices
        })

        # --- decoder (with skip connections every 2 encoder layers) ---
        self.dec_blocks = nn.ModuleList([
            BiMambaBlock(d, config.d_state, config.d_conv,
                         config.expand, config.dropout)
            for _ in range(config.n_dec_layers)
        ])

        n_skips = min(config.n_dec_layers, config.n_enc_layers)
        self.skip_projs = nn.ModuleList([
            nn.Linear(d, d) for _ in range(n_skips)
        ])

        # --- positional encoding ---
        self.pos_embed = nn.Parameter(torch.zeros(1, config.n_windows, d))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # --- output head ---
        self.head = UncertaintyHead(d)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        mutation_rate: torch.Tensor,
        tree_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 2, W, N)  single-scale SFS (log1p-transformed)
        mutation_rate : (B, 1)  log mutation rate
        tree_feats : (B, W, tree_feat_dim) optional tsinfer topology features

        Returns
        -------
        out : (B, W, 2)  where out[..., 0] = mu, out[..., 1] = log_sigma2
        """
        h = self.input_proj(x)                       # (B, W, D)
        W = h.shape[1]
        h = h + self.pos_embed[:, :W, :]

        if self.tree_encoder is not None and tree_feats is not None:
            h = h + self.tree_encoder(tree_feats)

        # --- encoder with FiLM conditioning at first/last layers ---
        enc_hiddens = []
        for i, block in enumerate(self.enc_blocks):
            h = block(h)
            if str(i) in self.enc_films:
                h = self.enc_films[str(i)](h, mutation_rate)
            enc_hiddens.append(h)

        # --- decoder with additive skip connections ---
        for i, block in enumerate(self.dec_blocks):
            if i < len(self.skip_projs):
                enc_idx = len(enc_hiddens) - 1 - i
                h = h + self.skip_projs[i](enc_hiddens[enc_idx])
            h = block(h)

        return self.head(h)                           # (B, W, 2)

    def configure_optimizers(self, weight_decay: float, learning_rate: float,
                             betas: tuple, device_type: str):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else {}
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas,
                                 **extra_args)
