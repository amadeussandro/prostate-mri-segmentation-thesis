"""Utility functions for the Prostate MRI 2D Segmentation Pipeline.

This module provides common helper functions for configuration parsing,
reproducibility, device selection, and filesystem directory management.
"""

import os
import random
from typing import Any, Dict, Optional
import numpy as np


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and parse a YAML experiment configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing parsed configuration parameters.

    Raises
    ------
    FileNotFoundError
        If the specified config file does not exist.
    ImportError
        If PyYAML is not installed.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    # Try importing PyYAML; raise informative error if not present in environment
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required to parse config files. "
            "Install it via: pip install pyyaml"
        ) from e

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # TODO: Add schema validation for mandatory configuration keys
    return config or {}


def set_seed(seed: int = 42) -> None:
    """Set random seed across standard library, NumPy, and PyTorch (if available).

    Ensures experiment reproducibility across runs.

    Parameters
    ----------
    seed : int, default=42
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch seeding (handled safely if torch is not yet installed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # PyTorch not installed yet in scaffolding phase
        pass


def get_device(prefer_cuda: bool = True) -> str:
    """Detect and return available compute device string ('cuda', 'mps', or 'cpu').

    Parameters
    ----------
    prefer_cuda : bool, default=True
        Whether to prioritize CUDA/GPU acceleration when available.

    Returns
    -------
    str
        Device identifier ('cuda', 'mps', or 'cpu').
    """
    if not prefer_cuda:
        return "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def ensure_dir(dir_path: str) -> str:
    """Ensure that a directory exists, creating parent directories if necessary.

    Parameters
    ----------
    dir_path : str
        Directory path to ensure.

    Returns
    -------
    str
        Absolute path to the created/existing directory.
    """
    abs_path = os.path.abspath(dir_path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

