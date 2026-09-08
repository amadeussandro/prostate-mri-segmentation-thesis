"""Evaluation metrics and qualitative assessment for Prostate MRI Segmentation.

This module provides functions for calculating segmentation metrics (Dice Similarity
Coefficient, Intersection over Union / Jaccard index, sensitivity, specificity)
and generating qualitative comparison visualizations.
"""

import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


def compute_dice(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute the Dice Similarity Coefficient (DSC) between two binary masks.

    DSC = (2 * |X ∩ Y|) / (|X| + |Y|)

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary mask array (values in {0, 1}).
    true_mask : np.ndarray
        Ground-truth binary mask array (values in {0, 1}).
    eps : float, default=1e-6
        Smoothing epsilon to avoid division by zero when both masks are empty.

    Returns
    -------
    float
        Dice similarity coefficient in [0.0, 1.0].
    """
    p = (pred_mask > 0).astype(np.float32)
    t = (true_mask > 0).astype(np.float32)

    intersection = np.sum(p * t)
    total_voxels = np.sum(p) + np.sum(t)

    if total_voxels == 0:
        # Both masks are completely empty: perfect agreement on negative slice
        return 1.0

    return float((2.0 * intersection + eps) / (total_voxels + eps))


def compute_iou(
    pred_mask: np.ndarray,
    true_mask: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Compute Intersection over Union (IoU / Jaccard Index) between two binary masks.

    IoU = |X ∩ Y| / |X ∪ Y|

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary mask array.
    true_mask : np.ndarray
        Ground-truth binary mask array.
    eps : float, default=1e-6
        Smoothing constant.

    Returns
    -------
    float
        IoU score in [0.0, 1.0].
    """
    p = (pred_mask > 0).astype(np.float32)
    t = (true_mask > 0).astype(np.float32)

    intersection = np.sum(p * t)
    union = np.sum(p) + np.sum(t) - intersection

    if union == 0:
        return 1.0

    return float((intersection + eps) / (union + eps))


def evaluate_patient(
    patient_id: str,
    model: Any,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Run full-volume 3D evaluation for a single patient by evaluating 2D slices.

    Parameters
    ----------
    patient_id : str
        Patient identifier string.
    model : Any
        Trained segmentation model.
    config : Dict[str, Any]
        Evaluation configuration.

    Returns
    -------
    Dict[str, float]
        Dictionary of summary metrics (volume-level Dice, mean slice Dice, slice IoU).
    """
    # Scaffolding placeholder
    # TODO: Load patient volume slices, run batched inference, reconstruct 3D prediction
    # TODO: Compute 3D volumetric Dice, 95% Hausdorff Distance, and Average Surface Distance
    raise NotImplementedError("evaluate_patient is a scaffolding placeholder.")


def plot_prediction_slice(
    mri_slice: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_path: str,
    title: Optional[str] = None,
) -> None:
    """Save a side-by-side qualitative comparison of MRI, ground truth, and prediction.

    Parameters
    ----------
    mri_slice : np.ndarray
        2D MRI slice (e.g. T2 or ADC) of shape (H, W).
    ground_truth : np.ndarray
        2D ground-truth binary mask of shape (H, W).
    prediction : np.ndarray
        2D predicted binary mask or probability map of shape (H, W).
    output_path : str
        Path where the figure will be saved.
    title : Optional[str], default=None
        Overall figure title.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(mri_slice, cmap="gray")
    axes[0].set_title("Input MRI Slice")
    axes[0].axis("off")

    axes[1].imshow(mri_slice, cmap="gray")
    gt_masked = np.ma.masked_where(ground_truth == 0, ground_truth)
    axes[1].imshow(gt_masked, cmap="Reds", alpha=0.6)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(mri_slice, cmap="gray")
    pred_masked = np.ma.masked_where(prediction == 0, prediction)
    axes[2].imshow(pred_masked, cmap="Blues", alpha=0.6)
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

