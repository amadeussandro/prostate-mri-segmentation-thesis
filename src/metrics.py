"""Volume-level segmentation metrics, computed per case in original voxel
space, per class, before any cohort-level aggregation.

Primary metric: Dice per class. Secondary: IoU, HD95, ASD (both mm-correct,
using the case's real voxel spacing), precision, recall. Reporting follows
the thesis protocol: mean +/- SD across cases, plus a bootstrap 95% CI
(formal significance tests are secondary given the small n=19/119 cohort).
"""

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def dice_score(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    p = pred == class_id
    t = gt == class_id
    denom = p.sum() + t.sum()
    if denom == 0:
        return 1.0  # both absent: trivially perfect agreement on a negative case
    intersection = np.logical_and(p, t).sum()
    return float(2.0 * intersection / denom)


def iou_score(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    p = pred == class_id
    t = gt == class_id
    union = np.logical_or(p, t).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(p, t).sum()
    return float(intersection / union)


def precision_recall(pred: np.ndarray, gt: np.ndarray, class_id: int) -> Tuple[float, float]:
    p = pred == class_id
    t = gt == class_id
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()

    precision = 1.0 if (tp + fp) == 0 else float(tp / (tp + fp))
    recall = 1.0 if (tp + fn) == 0 else float(tp / (tp + fn))
    return precision, recall


def _surface_voxel_coords_mm(mask: np.ndarray, spacing: Sequence[float]) -> np.ndarray:
    """Physical (mm) coordinates of the mask's boundary (surface) voxels."""
    if mask.sum() == 0:
        return np.empty((0, mask.ndim))
    eroded = ndimage.binary_erosion(mask, border_value=0)
    surface = mask & ~eroded
    if surface.sum() == 0:
        # A single-voxel or fully eroded object: treat every foreground voxel as surface.
        surface = mask
    coords = np.array(np.nonzero(surface)).T.astype(np.float64)
    return coords * np.array(spacing[: mask.ndim])


def _directed_surface_distances(surf_a_mm: np.ndarray, surf_b_mm: np.ndarray) -> np.ndarray:
    """Distance (mm) from every point in surf_a to its nearest point in surf_b."""
    tree = cKDTree(surf_b_mm)
    dists, _ = tree.query(surf_a_mm, k=1)
    return dists


def hausdorff_distance_95(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], class_id: int) -> float:
    """Symmetric HD95 in mm: max(P95(d(pred_surf, gt_surf)), P95(d(gt_surf, pred_surf))).

    Returns 0.0 if both masks are empty (trivial agreement) and NaN if only
    one is empty (undefined -- caller should exclude NaNs from aggregation).
    """
    p_mask = pred == class_id
    t_mask = gt == class_id
    if p_mask.sum() == 0 and t_mask.sum() == 0:
        return 0.0
    if p_mask.sum() == 0 or t_mask.sum() == 0:
        return float("nan")

    surf_p = _surface_voxel_coords_mm(p_mask, spacing)
    surf_t = _surface_voxel_coords_mm(t_mask, spacing)

    d_pt = _directed_surface_distances(surf_p, surf_t)
    d_tp = _directed_surface_distances(surf_t, surf_p)

    return float(max(np.percentile(d_pt, 95), np.percentile(d_tp, 95)))


def average_surface_distance(pred: np.ndarray, gt: np.ndarray, spacing: Sequence[float], class_id: int) -> float:
    """Symmetric ASD in mm: mean(mean(d(pred_surf, gt_surf)), mean(d(gt_surf, pred_surf)))."""
    p_mask = pred == class_id
    t_mask = gt == class_id
    if p_mask.sum() == 0 and t_mask.sum() == 0:
        return 0.0
    if p_mask.sum() == 0 or t_mask.sum() == 0:
        return float("nan")

    surf_p = _surface_voxel_coords_mm(p_mask, spacing)
    surf_t = _surface_voxel_coords_mm(t_mask, spacing)

    d_pt = _directed_surface_distances(surf_p, surf_t)
    d_tp = _directed_surface_distances(surf_t, surf_p)

    return float((d_pt.mean() + d_tp.mean()) / 2.0)


@dataclass
class CaseMetrics:
    patient_id: str
    per_class: Dict[int, Dict[str, float]]  # class_id -> {dice, iou, hd95, asd, precision, recall}


def evaluate_case_volume(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Sequence[float],
    class_ids: Sequence[int],
    patient_id: str = "",
) -> CaseMetrics:
    """Compute all metrics for one case, per class, in original voxel space.
    `pred` and `gt` must already be in the SAME (original) voxel grid -- i.e.
    reconstructed and inverse-transformed, per the thesis evaluation protocol
    (Strategy A: evaluate in original voxel space, never in resampled space).
    """
    per_class = {}
    for c in class_ids:
        precision, recall = precision_recall(pred, gt, c)
        per_class[c] = {
            "dice": dice_score(pred, gt, c),
            "iou": iou_score(pred, gt, c),
            "hd95_mm": hausdorff_distance_95(pred, gt, spacing, c),
            "asd_mm": average_surface_distance(pred, gt, spacing, c),
            "precision": precision,
            "recall": recall,
        }
    return CaseMetrics(patient_id=patient_id, per_class=per_class)


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of `values`,
    skipping NaNs (undefined per-case metrics, e.g. HD95 on an empty class)."""
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(arr, size=arr.size, replace=True).mean() for _ in range(n_boot)
    ])
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha)))
    return (lower, upper)


def inter_slice_dice_inconsistency(pred: np.ndarray, gt: np.ndarray, class_id: int) -> float:
    """RQ-3 supporting analysis (kept clearly separate from primary Dice/IoU/
    HD95/ASD metrics; NOT claimed as a novel metric). Measures axial
    discontinuity of the reconstructed 2D-slice predictions: the per-slice
    Dice(k) against ground truth is computed for every axial slice k, and the
    inconsistency score is mean(|Dice(k) - Dice(k+1)|) across consecutive
    slice pairs. A perfectly smooth (spatially consistent) prediction along z
    scores 0; large slice-to-slice Dice swings score higher.
    """
    depth = pred.shape[2]
    if depth < 2:
        return 0.0
    per_slice_dice = np.array([
        dice_score(pred[:, :, k], gt[:, :, k], class_id) for k in range(depth)
    ])
    return float(np.mean(np.abs(np.diff(per_slice_dice))))


def aggregate_per_slice_pooled(
    per_case_slice_dice: List[Dict[int, List[float]]],
    class_ids: Sequence[int],
) -> Dict[int, Dict[str, float]]:
    """Experiment 3 (RQ-2) supplement: pool every slice's Dice across ALL
    cases and average directly (the per-slice protocol) -- as opposed to
    `aggregate_case_results`, which averages per-case volume Dice (the
    thesis's PRIMARY per-case protocol). Per-slice pooling is a known biased
    estimator: cases with more slices contribute proportionally more to the
    mean. Comparing the two is exactly what RQ-2 measures; this function
    must never replace `aggregate_case_results` as the primary reported number.
    """
    pooled: Dict[int, Dict[str, float]] = {}
    for c in class_ids:
        all_slice_values: List[float] = []
        for case_slices in per_case_slice_dice:
            all_slice_values.extend(case_slices[c])
        arr = np.asarray(all_slice_values, dtype=np.float64)
        pooled[c] = {
            "mean": float(arr.mean()) if arr.size else float("nan"),
            "std": float(arr.std()) if arr.size else float("nan"),
            "n_slices": int(arr.size),
        }
    return pooled


def aggregate_case_results(
    case_results: List[CaseMetrics],
    class_ids: Sequence[int],
    metric_names: Sequence[str] = ("dice", "iou", "hd95_mm", "asd_mm", "precision", "recall"),
) -> Dict[int, Dict[str, dict]]:
    """Aggregate per-case metrics into cohort-level mean +/- SD + bootstrap CI,
    per class, per metric. NaNs (undefined per-case metrics) are excluded.
    """
    summary: Dict[int, Dict[str, dict]] = {}
    for c in class_ids:
        summary[c] = {}
        for metric in metric_names:
            values = [case.per_class[c][metric] for case in case_results]
            valid = [v for v in values if not np.isnan(v)]
            n_excluded = len(values) - len(valid)
            mean = float(np.mean(valid)) if valid else float("nan")
            std = float(np.std(valid)) if valid else float("nan")
            ci_low, ci_high = bootstrap_ci(valid) if valid else (float("nan"), float("nan"))
            summary[c][metric] = {
                "mean": mean,
                "std": std,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "n_cases": len(valid),
                "n_excluded_undefined": n_excluded,
            }
    return summary
