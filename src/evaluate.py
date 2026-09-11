"""Evaluation metrics and qualitative assessment for Prostate MRI Segmentation.

This module provides slice-level metric helpers (compute_dice/compute_iou, used
by src/train.py for cheap per-epoch validation) and the thesis-required
VOLUME-LEVEL, per-case, original-voxel-space evaluation pipeline
(evaluate_patient / evaluate_cases), built on src/reconstruction.py and
src/metrics.py. Per the thesis protocol, volume-level per-case evaluation is
primary; slice-level is only ever a supplement (see Experiment 3 / RQ-2).
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
import torch

from src.metrics import (
    CaseMetrics,
    aggregate_case_results,
    dice_score,
    evaluate_case_volume,
    iou_score,
    precision_recall,
)
from src.reconstruction import reconstruct_case_to_original_space, reconstruct_volume_from_slices


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
    dataset: Any,
    device: str,
    class_ids: Sequence[int] = (0, 1, 2),
) -> CaseMetrics:
    """Run full-volume, original-voxel-space evaluation for a single patient.

    Pipeline (per thesis protocol, Strategy A): run the model on every 2D
    slice of this case's preprocessed volume -> reconstruct by explicit slice
    index -> invert the preprocessing geometric transforms (nearest-neighbour)
    -> compare against the untouched original-space ground truth -> compute
    per-class Dice/IoU/HD95/ASD/precision/recall.

    Parameters
    ----------
    patient_id : str
        Patient identifier string. Must be one of `dataset.patient_ids`.
    model : torch.nn.Module
        Segmentation model (already on `device`).
    dataset : ProstateZonal2DDataset
        Dataset that already built the index for this patient (so
        dataset.case_geometry / dataset.case_transform_meta are populated).
    device : str
        Torch device string.
    class_ids : Sequence[int]
        Class indices to evaluate (default: background + 2 anatomy classes).

    Returns
    -------
    CaseMetrics
        Per-class metrics dict for this case, in original voxel space.
    """
    import nibabel as nib

    geometry, transform_meta = dataset.get_case_metadata(patient_id)

    case_sample_indices = [
        i for i, (pid, _) in enumerate(dataset.samples) if pid == patient_id
    ]
    if not case_sample_indices:
        raise ValueError(f"No samples found for patient '{patient_id}' in this dataset.")

    model.eval()
    pred_slices: List[np.ndarray] = []
    gt_slices: List[np.ndarray] = []
    slice_indices: List[int] = []

    with torch.no_grad():
        for i in case_sample_indices:
            sample = dataset[i]
            image = torch.from_numpy(sample["image"]).unsqueeze(0).float().to(device)
            logits = model(image)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

            pred_slices.append(pred)
            gt_slices.append(sample["mask"])
            slice_indices.append(sample["slice_idx"])

    pred_nii = reconstruct_case_to_original_space(pred_slices, slice_indices, transform_meta, geometry)
    gt_nii = reconstruct_case_to_original_space(gt_slices, slice_indices, transform_meta, geometry)

    pred_original = np.asarray(pred_nii.dataobj)
    gt_original = np.asarray(gt_nii.dataobj)

    return evaluate_case_volume(
        pred=pred_original,
        gt=gt_original,
        spacing=geometry.zooms,
        class_ids=class_ids,
        patient_id=patient_id,
    )


def evaluate_patient_protocols(
    patient_id: str,
    model: Any,
    dataset: Any,
    device: str,
    class_ids: Sequence[int] = (0, 1, 2),
) -> Dict[str, Any]:
    """Experiment 3 (RQ-2) support: score ONE model's predictions for a case
    under multiple evaluation protocols, without any retraining or re-running
    the model more than once. Isolates the *scoring protocol* as the only
    variable, per the thesis constraint that Experiment 3 is scoring-only.

    Returns a dict with:
      - "original_voxel_per_case": CaseMetrics in original voxel space
        (Strategy A -- the thesis's primary protocol).
      - "resampled_per_case": CaseMetrics scored directly in the preprocessed
        (resampled/cropped) grid, WITHOUT inverting the geometric transforms.
      - "per_slice_dice": {class_id: [dice(k) for k in range(depth)]} computed
        in original voxel space -- the raw material for both the per-slice
        aggregation supplement and the RQ-3 inter-slice consistency analysis.
    """
    geometry, transform_meta = dataset.get_case_metadata(patient_id)

    case_sample_indices = [i for i, (pid, _) in enumerate(dataset.samples) if pid == patient_id]
    if not case_sample_indices:
        raise ValueError(f"No samples found for patient '{patient_id}' in this dataset.")

    model.eval()
    pred_slices, gt_slices, slice_indices = [], [], []

    with torch.no_grad():
        for i in case_sample_indices:
            sample = dataset[i]
            image = torch.from_numpy(sample["image"]).unsqueeze(0).float().to(device)
            logits = model(image)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)
            pred_slices.append(pred)
            gt_slices.append(sample["mask"])
            slice_indices.append(sample["slice_idx"])

    # Protocol A (primary): reconstruct + invert to original voxel space.
    pred_nii = reconstruct_case_to_original_space(pred_slices, slice_indices, transform_meta, geometry)
    gt_nii = reconstruct_case_to_original_space(gt_slices, slice_indices, transform_meta, geometry)
    pred_original = np.asarray(pred_nii.dataobj)
    gt_original = np.asarray(gt_nii.dataobj)

    original_metrics = evaluate_case_volume(
        pred_original, gt_original, spacing=geometry.zooms, class_ids=class_ids, patient_id=patient_id,
    )

    # Protocol B (RQ-2 supplement): score directly in the preprocessed/resampled
    # grid, no inverse transform. Voxel spacing in this grid is NOT the
    # original spacing for axes that were resized, so HD95/ASD in mm would be
    # misleading there and are intentionally NOT computed for this protocol.
    # Dice/IoU/precision/recall are voxel-count ratios (spacing-independent),
    # so all four are reported here to match "resampled_per_case" documented
    # schema exactly.
    depth = transform_meta.post_resample_depth
    pred_resampled = reconstruct_volume_from_slices(pred_slices, slice_indices, depth)
    gt_resampled = reconstruct_volume_from_slices(gt_slices, slice_indices, depth)
    resampled_metrics = {}
    for c in class_ids:
        precision, recall = precision_recall(pred_resampled, gt_resampled, c)
        resampled_metrics[c] = {
            "dice": dice_score(pred_resampled, gt_resampled, c),
            "iou": iou_score(pred_resampled, gt_resampled, c),
            "precision": precision,
            "recall": recall,
        }

    # Per-slice Dice in original voxel space, for the per-slice-vs-per-case
    # aggregation comparison (RQ-2) and the RQ-3 inter-slice analysis.
    per_slice_dice = {
        c: [dice_score(pred_original[:, :, k], gt_original[:, :, k], c) for k in range(pred_original.shape[2])]
        for c in class_ids
    }

    return {
        "original_voxel_per_case": original_metrics,
        "resampled_per_case": resampled_metrics,
        "per_slice_dice": per_slice_dice,
    }


def evaluate_cases(
    model: Any,
    dataset: Any,
    device: str,
    patient_ids: Optional[Sequence[str]] = None,
    class_ids: Sequence[int] = (0, 1, 2),
) -> Tuple[List[CaseMetrics], Dict[int, Dict[str, dict]]]:
    """Evaluate every case in `patient_ids` (default: all cases in `dataset`)
    and aggregate to cohort-level mean +/- SD + bootstrap CI, per class.
    Metrics are computed per case BEFORE aggregation (never slice-averaged).
    """
    ids = list(patient_ids) if patient_ids is not None else list(dataset.patient_ids)
    case_results = [
        evaluate_patient(pid, model, dataset, device, class_ids=class_ids) for pid in ids
    ]
    summary = aggregate_case_results(case_results, class_ids=class_ids)
    return case_results, summary


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

