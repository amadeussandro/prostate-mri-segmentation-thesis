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
        import yaml  # type: ignore
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
        import torch  # type: ignore

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
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def build_config_fingerprint(
    config: Dict[str, Any], device: Optional[str] = None
) -> Dict[str, Any]:
    """Summarize the scientifically-relevant settings of an experiment run.

    The purpose is auditability: printed at the start of every run and saved
    next to the results, so it is immediately obvious which factor a given
    experiment changed relative to the baseline -- and that everything else
    (optimizer, LR, batch size, seed, epochs, precision) was held fixed.

    Reads ONLY the config; it never inspects or mutates training state. Fields
    the pipeline does not implement (LR scheduler) are reported explicitly as
    OFF rather than omitted, so their absence is on the record. Augmentation is
    reported from the resolved `augmentation:` block: OFF for Baseline / E1
    (no block), and the full parameter set for E2.
    """
    from src.augment import AugmentationConfig  # local import: keeps utils light
    from src.losses import LossConfig  # local import: keeps utils torch-free on import

    experiment_cfg = config.get("experiment", {}) or {}
    model_cfg = config.get("model", {}) or {}
    training_cfg = config.get("training", {}) or {}
    data_cfg = config.get("data", {}) or {}
    preprocessing_cfg = config.get("preprocessing", {}) or {}

    loss_config = LossConfig.from_config(config)
    augmentation_config = AugmentationConfig.from_config(config)
    amp_enabled = bool(training_cfg.get("amp", False))

    return {
        "experiment": experiment_cfg.get("name", "<unnamed>"),
        "output_dir": experiment_cfg.get("output_dir"),
        "architecture": model_cfg.get("architecture", "ProstateUNet2D"),
        "in_channels": model_cfg.get("in_channels"),
        "out_channels": model_cfg.get("out_channels"),
        "init_features": model_cfg.get("init_features"),
        "dropout_rate": model_cfg.get("dropout_rate"),
        "loss": loss_config.describe(),
        "loss_spec": loss_config.to_dict(),
        "dice_classes": list(loss_config.dice_classes) if not loss_config.is_baseline_cross_entropy else None,
        "dice_epsilon": loss_config.epsilon if not loss_config.is_baseline_cross_entropy else None,
        "num_epochs": training_cfg.get("num_epochs"),
        "seed": training_cfg.get("seed", 42),
        "amp": amp_enabled,
        "precision": "AMP (mixed)" if amp_enabled else "FP32",
        "optimizer": training_cfg.get("optimizer"),
        "learning_rate": training_cfg.get("learning_rate"),
        "weight_decay": training_cfg.get("weight_decay"),
        "batch_size": training_cfg.get("batch_size"),
        "num_workers": training_cfg.get("num_workers", 0),
        "use_weighted_sampling": training_cfg.get("use_weighted_sampling", True),
        "augmentation": augmentation_config.describe(),
        "augmentation_spec": augmentation_config.to_dict(),
        "augmentation_enabled": augmentation_config.enabled,
        "scheduler": "OFF (constant learning rate)",
        "preprocessing": {
            "normalization": preprocessing_cfg.get("normalization"),
            "z_resample": preprocessing_cfg.get("z_resample"),
            "spatial_mode": preprocessing_cfg.get("spatial_mode"),
            "target_size": preprocessing_cfg.get("target_size"),
        },
        "target_mask": data_cfg.get("target_mask"),
        "slice_sampling": data_cfg.get("slice_sampling"),
        "references_test_split": bool(data_cfg.get("test_csv") or data_cfg.get("test_dir")),
        "device": device,
    }


def format_config_fingerprint(fingerprint: Dict[str, Any]) -> str:
    """Render `build_config_fingerprint` output as an aligned text block."""
    rows = [
        ("Experiment", fingerprint.get("experiment")),
        ("Architecture", f"{fingerprint.get('architecture')} "
                         f"(in={fingerprint.get('in_channels')}, out={fingerprint.get('out_channels')}, "
                         f"init_features={fingerprint.get('init_features')}, "
                         f"dropout={fingerprint.get('dropout_rate')})"),
        ("Loss", fingerprint.get("loss")),
    ]
    if fingerprint.get("dice_classes") is not None:
        rows.append(("Dice classes", f"{fingerprint['dice_classes']} (1=CG, 2=PZ; background excluded)"))
        rows.append(("Dice epsilon", fingerprint.get("dice_epsilon")))
    rows.extend([
        ("Epochs (max)", fingerprint.get("num_epochs")),
        ("Seed", fingerprint.get("seed")),
        ("AMP", "ON" if fingerprint.get("amp") else "OFF"),
        ("Precision", fingerprint.get("precision")),
        ("Optimizer", fingerprint.get("optimizer")),
        ("Learning rate", fingerprint.get("learning_rate")),
        ("Weight decay", fingerprint.get("weight_decay")),
        ("Batch size", fingerprint.get("batch_size")),
        ("DataLoader workers", fingerprint.get("num_workers")),
        ("Weighted sampling", "ON" if fingerprint.get("use_weighted_sampling") else "OFF"),
        ("Augmentation", fingerprint.get("augmentation")),
        ("Scheduler", fingerprint.get("scheduler")),
        ("Preprocessing", fingerprint.get("preprocessing")),
        ("Target mask", fingerprint.get("target_mask")),
        ("Slice sampling", fingerprint.get("slice_sampling")),
        ("Test split referenced", "YES" if fingerprint.get("references_test_split") else "NO"),
        ("Device", fingerprint.get("device")),
    ])

    width = max(len(label) for label, _ in rows)
    header = "=" * 78
    lines = [header, "EXPERIMENT CONFIGURATION FINGERPRINT", header]
    lines.extend(f"  {label.ljust(width)} : {value}" for label, value in rows)
    lines.append(header)
    return "\n".join(lines)


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

