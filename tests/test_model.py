"""Unit tests for FastCxtModel and its component modules.

Tests cover: shape correctness, output ranges, gradient flow, config presets,
module isolation, edge cases, and reproducibility.

NOTE: BiMambaBlock / full model require CUDA (Mamba2 is GPU-only).
      Tests that need CUDA are marked with @pytest.mark.skipif.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fastcxt.config import FastCxtConfig, PRESETS
from fastcxt.modules import (
    FiLMLayer,
    MultiScaleInputProjection,
    UncertaintyHead,
    TreeEncoder,
)

CUDA = torch.cuda.is_available()
skipnocuda = pytest.mark.skipif(not CUDA, reason="CUDA required for Mamba2")


# ── helpers ──────────────────────────────────────────────────────────────────

def _small_config(**overrides) -> FastCxtConfig:
    """Minimal config for fast tests."""
    defaults = dict(
        d_model=64, d_state=16, d_conv=4, expand=2,
        n_enc_layers=2, n_dec_layers=2,
        n_channels=2, max_samples=32, n_windows=16,
        stem_channels=32, conv_channels=16,
        kernel_sizes=(3, 5),
        dropout=0.0, use_trees=False,
    )
    defaults.update(overrides)
    return FastCxtConfig(**defaults)


def _make_inputs(cfg: FastCxtConfig, B: int = 2, device: str = "cpu"):
    x = torch.randn(B, cfg.n_channels, cfg.n_windows, cfg.max_samples, device=device)
    mu = torch.randn(B, 1, device=device)
    return x, mu


# ── 1. FiLMLayer: output shape ──────────────────────────────────────────────

def test_film_output_shape():
    film = FiLMLayer(d_model=64)
    h = torch.randn(2, 10, 64)
    cond = torch.randn(2, 1)
    out = film(h, cond)
    assert out.shape == h.shape


# ── 2. FiLMLayer: identity at gamma=0, beta=0 ──────────────────────────────

def test_film_identity_when_zero():
    """When gamma=0 and beta=0, FiLM should return h (identity)."""
    film = FiLMLayer(d_model=32)
    # Force output to zero
    with torch.no_grad():
        film.net[-1].weight.zero_()
        film.net[-1].bias.zero_()
    h = torch.randn(1, 5, 32)
    cond = torch.randn(1, 1)
    out = film(h, cond)
    torch.testing.assert_close(out, h)


# ── 3. FiLMLayer: gradient flows through condition ─────────────────────────

def test_film_grad_flows_through_cond():
    film = FiLMLayer(d_model=32)
    h = torch.randn(1, 5, 32, requires_grad=True)
    cond = torch.randn(1, 1, requires_grad=True)
    out = film(h, cond)
    out.sum().backward()
    assert cond.grad is not None and cond.grad.abs().sum() > 0


# ── 4. UncertaintyHead: output shape (B, W, 2) ─────────────────────────────

def test_uncertainty_head_shape():
    head = UncertaintyHead(d_model=64)
    h = torch.randn(3, 20, 64)
    out = head(h)
    assert out.shape == (3, 20, 2)


# ── 5. UncertaintyHead: mu and log_sigma2 are independent branches ─────────

def test_uncertainty_head_branches_independent():
    """Zeroing mu_head weights should not affect var_head output."""
    head = UncertaintyHead(d_model=64)
    h = torch.randn(1, 5, 64)
    out_before = head(h).detach().clone()

    with torch.no_grad():
        for p in head.mu_head.parameters():
            p.zero_()

    out_after = head(h)
    # mu changed, but log_sigma2 should be the same
    assert not torch.allclose(out_after[..., 0], out_before[..., 0])
    torch.testing.assert_close(out_after[..., 1], out_before[..., 1])


# ── 6. TreeEncoder: output shape ────────────────────────────────────────────

def test_tree_encoder_shape():
    enc = TreeEncoder(tree_feat_dim=100, d_model=64)
    feats = torch.randn(2, 16, 100)
    out = enc(feats)
    assert out.shape == (2, 16, 64)


# ── 7. TreeEncoder: gradient flow ───────────────────────────────────────────

def test_tree_encoder_grad_flow():
    enc = TreeEncoder(tree_feat_dim=50, d_model=32)
    feats = torch.randn(1, 8, 50, requires_grad=True)
    out = enc(feats)
    out.sum().backward()
    assert feats.grad is not None and feats.grad.abs().sum() > 0


# ── 8. MultiScaleInputProjection: output shape ─────────────────────────────

def test_multiscale_proj_shape():
    proj = MultiScaleInputProjection(
        d_model=64, max_samples=32, n_channels=2,
        stem_channels=32, conv_channels=16,
        kernel_sizes=(3, 5), dropout=0.0,
    )
    x = torch.randn(2, 2, 16, 32)
    out = proj(x)
    assert out.shape == (2, 16, 64)


# ── 9. MultiScaleInputProjection: pads short samples ───────────────────────

def test_multiscale_proj_pads_short_input():
    proj = MultiScaleInputProjection(
        d_model=64, max_samples=32, n_channels=2,
        stem_channels=32, conv_channels=16,
        kernel_sizes=(3,), dropout=0.0,
    )
    x = torch.randn(1, 2, 8, 10)  # N=10 < max_samples=32
    out = proj(x)
    assert out.shape == (1, 8, 64)


# ── 10. MultiScaleInputProjection: truncates long samples ──────────────────

def test_multiscale_proj_truncates_long_input():
    proj = MultiScaleInputProjection(
        d_model=64, max_samples=32, n_channels=2,
        stem_channels=32, conv_channels=16,
        kernel_sizes=(3,), dropout=0.0,
    )
    x = torch.randn(1, 2, 8, 100)  # N=100 > max_samples=32
    out = proj(x)
    assert out.shape == (1, 8, 64)


# ── 11. MultiScaleInputProjection: deterministic in eval mode ───────────────

def test_multiscale_proj_deterministic_eval():
    proj = MultiScaleInputProjection(
        d_model=64, max_samples=32, n_channels=2,
        stem_channels=32, conv_channels=16,
        kernel_sizes=(3, 5), dropout=0.1,
    )
    proj.eval()
    x = torch.randn(1, 2, 8, 32)
    out1 = proj(x)
    out2 = proj(x)
    torch.testing.assert_close(out1, out2)


# ── 12. Config presets are valid ────────────────────────────────────────────

@pytest.mark.parametrize("preset_name", list(PRESETS.keys()))
def test_config_presets_valid(preset_name):
    cfg = PRESETS[preset_name]
    assert cfg.d_model > 0
    assert cfg.n_enc_layers > 0
    assert cfg.n_dec_layers > 0
    assert cfg.n_windows > 0


# ── 13. Config for_inference / for_training ─────────────────────────────────

def test_config_for_inference_and_training():
    cfg = FastCxtConfig()
    inf = cfg.for_inference(batch_size=1, device="cpu")
    assert inf.batch_size == 1 and inf.device == "cpu"
    tr = cfg.for_training(batch_size=64, device="cuda")
    assert tr.batch_size == 64 and tr.device == "cuda"


# ── 14. Full model: forward output shape ────────────────────────────────────

@skipnocuda
def test_model_forward_shape():
    from fastcxt.model import FastCxtModel
    cfg = _small_config()
    model = FastCxtModel(cfg).cuda()
    x, mu = _make_inputs(cfg, B=2, device="cuda")
    out = model(x, mu)
    assert out.shape == (2, cfg.n_windows, 2)


# ── 15. Full model: gradient flows end-to-end ──────────────────────────────

@skipnocuda
def test_model_gradient_flow():
    from fastcxt.model import FastCxtModel
    cfg = _small_config()
    model = FastCxtModel(cfg).cuda()
    x, mu = _make_inputs(cfg, B=1, device="cuda")
    out = model(x, mu)
    loss = out.sum()
    loss.backward()
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


# ── 16. Full model: eval mode is deterministic ─────────────────────────────

@skipnocuda
def test_model_eval_deterministic():
    from fastcxt.model import FastCxtModel
    cfg = _small_config(dropout=0.1)
    model = FastCxtModel(cfg).cuda().eval()
    x, mu = _make_inputs(cfg, B=1, device="cuda")
    out1 = model(x, mu)
    out2 = model(x, mu)
    torch.testing.assert_close(out1, out2)


# ── 17. Full model: train mode with dropout is non-deterministic ───────────

@skipnocuda
def test_model_train_stochastic():
    from fastcxt.model import FastCxtModel
    cfg = _small_config(dropout=0.3)
    model = FastCxtModel(cfg).cuda().train()
    x, mu = _make_inputs(cfg, B=2, device="cuda")
    out1 = model(x, mu)
    out2 = model(x, mu)
    # Very unlikely to be identical with high dropout
    assert not torch.allclose(out1, out2, atol=1e-6)


# ── 18. Full model with tree features ──────────────────────────────────────

@skipnocuda
def test_model_with_trees():
    from fastcxt.model import FastCxtModel
    cfg = _small_config(use_trees=True, tree_feat_dim=50)
    model = FastCxtModel(cfg).cuda()
    x, mu = _make_inputs(cfg, B=2, device="cuda")
    tree_feats = torch.randn(2, cfg.n_windows, 50, device="cuda")
    out = model(x, mu, tree_feats=tree_feats)
    assert out.shape == (2, cfg.n_windows, 2)


# ── 19. Full model: configure_optimizers returns AdamW ──────────────────────

@skipnocuda
def test_configure_optimizers():
    from fastcxt.model import FastCxtModel
    cfg = _small_config()
    model = FastCxtModel(cfg).cuda()
    opt = model.configure_optimizers(
        weight_decay=0.1, learning_rate=3e-4,
        betas=(0.9, 0.95), device_type="cuda",
    )
    assert isinstance(opt, torch.optim.AdamW)
    # Should have 2 param groups (decay + no-decay)
    assert len(opt.param_groups) == 2


# ── 20. Full model: parameter count is reasonable ───────────────────────────

@skipnocuda
def test_model_param_count():
    from fastcxt.model import FastCxtModel
    cfg = _small_config()
    model = FastCxtModel(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    # Small config should have between 10K and 10M params
    assert 10_000 < n_params < 10_000_000, f"Unexpected param count: {n_params}"
