"""fastcxt -- fast pairwise coalescence time inference with Mamba SSMs."""

__version__ = "0.1.0"

from fastcxt.config import FastCxtConfig, PRESETS, TrainingConfig
from fastcxt.atlas import TimeAtlas
from fastcxt.paths import PATHS


def _lazy_import(name):
    """Lazy import for modules requiring mamba_ssm (GPU dependency)."""
    import importlib
    return importlib.import_module(f"fastcxt.{name}")


def __getattr__(name):
    if name == "FastCxtModel":
        return _lazy_import("model").FastCxtModel
    if name == "translate":
        return _lazy_import("translate").translate
    raise AttributeError(f"module 'fastcxt' has no attribute {name!r}")
