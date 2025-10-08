# src/__init__.py
"""
CNNVision package initializer.

This module:
- Sets up a package-wide logger with sane defaults.
- Exposes top-level imports for config, dataset, model, train, evaluate, predict.
- Provides utility helpers (device selection, version, simple config loader).
- Implements minimal registries to enable plugin-like extensibility for models and datasets.
"""

from __future__ import annotations

import os
import sys
import json
import time
import importlib
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# ---------------------------
# Version and package metadata
# ---------------------------
__title__ = "CNNVision"
__version__ = "0.1.0"
__author__ = "Krishan"
__email__ = "kantkrishan0206@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/kantkrishan0206-crypto/CNNVision"

# ---------------------------
# Logging setup (package-wide)
# ---------------------------
import logging

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger(__title__)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler (if LOGS_DIR exists)
    try:
        from .config import LOGS_DIR
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOGS_DIR / "package.log", mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception:
        # Config may not be available yet; skip file logging
        pass

    logger.propagate = False
    return logger

logger = _setup_logger()
logger.info(f"Initialized {__title__} v{__version__}")

# ---------------------------
# Device utility
# ---------------------------
def get_device(prefer_mps: bool = False):
    """
    Returns a torch device. Prefers CUDA; can optionally try Apple MPS.
    """
    try:
        import torch
        if torch.cuda.is_available():
            d = torch.device("cuda")
            logger.info(f"[Device] Using CUDA: {torch.cuda.get_device_name(0)}")
            return d
        if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("[Device] Using Apple MPS")
            return torch.device("mps")
        logger.info("[Device] Using CPU")
        return torch.device("cpu")
    except Exception as e:
        logger.warning(f"[Device] Torch not available, defaulting to CPU ({e})")
        class _CPUDevice:
            type = "cpu"
        return _CPUDevice()

# ---------------------------
# Simple config loader (JSON/YAML)
# ---------------------------
def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON or YAML file, then apply overrides.
    """
    cfg: Dict[str, Any] = {}
    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        ext = p.suffix.lower()
        if ext in (".json",):
            cfg = json.loads(p.read_text(encoding="utf-8"))
        elif ext in (".yml", ".yaml"):
            try:
                import yaml  # type: ignore
            except Exception:
                raise RuntimeError("PyYAML not installed. Run: pip install pyyaml")
            cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        else:
            raise ValueError(f"Unsupported config format: {ext}")

    overrides = overrides or {}
    cfg.update(overrides)
    return cfg

# ---------------------------
# Lightweight registries
# ---------------------------
_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}
_DATASET_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register_model(name: str):
    """
    Decorator to register a model builder under a name.
    Usage:
        @register_model("cifar_cnn")
        def build_model(...): ...
    """
    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _MODEL_REGISTRY:
            logger.warning(f"[Registry] Model '{name}' already registered. Overwriting.")
        _MODEL_REGISTRY[name] = fn
        logger.info(f"[Registry] Registered model: {name}")
        return fn
    return _decorator

def get_model(name: str, **kwargs):
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not registered. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](**kwargs)

def register_dataset(name: str):
    """
    Decorator to register a dataset loader under a name.
    Usage:
        @register_dataset("cifar10")
        def get_dataloaders(...): ...
    """
    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _DATASET_REGISTRY:
            logger.warning(f"[Registry] Dataset '{name}' already registered. Overwriting.")
        _DATASET_REGISTRY[name] = fn
        logger.info(f"[Registry] Registered dataset: {name}")
        return fn
    return _decorator

def get_dataset(name: str, **kwargs):
    if name not in _DATASET_REGISTRY:
        raise KeyError(f"Dataset '{name}' not registered. Available: {list(_DATASET_REGISTRY.keys())}")
    return _DATASET_REGISTRY[name](**kwargs)

# ---------------------------
# Public API re-exports
# ---------------------------
# Make imports ergonomic: from src import config, dataset, model, train, evaluate, predict
try:
    from . import config
    from .dataset import get_dataloaders, save_stats, CIFAR_MEAN, CIFAR_STD
    from .model import build_model
    from .train import train_model
    from .evaluate import evaluate_model, collect_predictions, confusion_matrix_plot
    from .predict import predict_image
except Exception as e:
    # During partial setups, some modules may not import; keep package load resilient.
    logger.debug(f"[Init] Partial import due to: {e}")

# ---------------------------
# Auto-registration (optional)
# ---------------------------
# Auto-register default model and dataset names so CLI can refer to them symbolically.
try:
    # Register model
    from .model import build_model as _default_build_model
    register_model("cifar_cnn")(_default_build_model)

    # Register dataset
    from .dataset import get_dataloaders as _default_get_dataloaders
    register_dataset("cifar10")(_default_get_dataloaders)
except Exception as e:
    logger.debug(f"[Init] Registry setup skipped: {e}")

# ---------------------------
# Utility: experiment run ID
# ---------------------------
def make_run_id(prefix: str = "run") -> str:
    """
    Generates a short, sortable run ID, e.g., 'run_20251008_215830'.
    """
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"

# ---------------------------
# Convenience export for top-level usage
# ---------------------------
__all__ = [
    "__title__", "__version__", "logger",
    "get_device", "load_config",
    "register_model", "get_model",
    "register_dataset", "get_dataset",
    "make_run_id",
    # re-exported modules/functions
    "config", "get_dataloaders", "save_stats", "CIFAR_MEAN", "CIFAR_STD",
    "build_model", "train_model",
    "evaluate_model", "collect_predictions", "confusion_matrix_plot",
    "predict_image",
]
