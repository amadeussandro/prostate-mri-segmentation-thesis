"""Evaluation metrics and qualitative assessment for Prostate MRI Segmentation.

This module provides slice-level metric helpers (compute_dice/compute_iou, used
by src/train.py for cheap per-epoch validation) and the thesis-required
VOLUME-LEVEL, per-case, original-voxel-space evaluation pipeline
(evaluate_patient / evaluate_cases), built on src/reconstruction.py and
src/metrics.py. Per the thesis protocol, volume-level per-case evaluation is
primary; slice-level is only ever a supplement (see Experiment 3 / RQ-2).
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import nibabel as nib
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
    pred_original, gt_original, geometry = predict_case_original_space(
        patient_id, model, dataset, device
    )

    return evaluate_case_volume(
        pred=pred_original,
        gt=gt_original,
        spacing=geometry.zooms,
        class_ids=class_ids,
        patient_id=patient_id,
    )


def load_original_space_ground_truth(
    dataset: Any,
    patient_id: str,
    geometry: Any,
) -> np.ndarray:
    """Load the UNTOUCHED, on-disk anatomy mask for a case as the ground truth
    for original-voxel-space evaluation.

    Why not reconstruct the GT from the preprocessed slices?
    -------------------------------------------------------
    Reconstructing the GT by inverting the preprocessing sends the ground truth
    through the SAME lossy transform as the prediction. For the baseline /
    crop_pad pipeline that inverse is exact (round-trip Dice = 1.0, see
    scripts/roundtrip_test.py), so it makes no difference. But for the
    Experiment 2-C `resize` arm the transform is deliberately lossy and NOT
    pixel-exact -- round-tripping the GT would degrade it identically to the
    prediction, so preprocessing errors partially cancel and the resize arm
    scores optimistically. That confounds the RQ-1 preprocessing ablation.

    Comparing every arm's reconstructed prediction against the identical,
    untouched original-space mask keeps the comparison fair: crop_pad is
    unaffected (its round-trip GT already equals this), and resize is scored
    honestly against the true anatomy in original voxel space.

    The on-disk mask is already in the case's original voxel grid and array
    order (the same grid `reconstruct_case_to_original_space` restores the
    prediction to), so the two are directly comparable.
    """
    mask_path = os.path.join(dataset.dataset_root, patient_id, dataset.mask_name)
    gt = np.round(nib.load(mask_path).get_fdata()).astype(np.int64)
    if tuple(gt.shape) != tuple(geometry.shape):
        raise ValueError(
            f"Untouched ground-truth mask shape {gt.shape} for patient "
            f"'{patient_id}' does not match the case geometry {geometry.shape}."
        )
    return gt


@dataclass
class CasePrediction:
    """Everything RQ2 needs from one case's inference + 2D->3D reconstruction.

    Produced by `predict_case_reconstructed`; the thin `predict_case_original_space`
    wrapper below returns only the three fields the Experiment-1 evaluation and
    the qualitative visualization already consumed, so their contract is unchanged.
    """

    patient_id: str
    pred_nii: Any                      # nib.Nifti1Image: prediction in ORIGINAL voxel space
    pred_original: np.ndarray          # np.asarray(pred_nii.dataobj), int label volume (H, W, D)
    pred_preprocessed: np.ndarray      # (H, W, D) argmax stack in PREPROCESSED space (pre-inverse)
    slice_indices: List[int]           # axial index of each predicted slice (explicit order)
    gt_original: np.ndarray            # untouched on-disk GT in the same original grid
    geometry: Any                      # VolumeGeometry for the case
    transform_meta: Any                # CaseTransformMeta (records the exact forward transform)


def predict_case_reconstructed(
    patient_id: str,
    model: Any,
    dataset: Any,
    device: str,
) -> CasePrediction:
    """Run the model on every axial slice of a case and reconstruct the
    prediction to original voxel space, returning the reconstructed NIfTI plus
    all provenance needed for RQ2 (2D stack, slice order, transform metadata).

    This is the single shared inference + 2D->3D reconstruction path. Inference
    only: `model.eval()` + `torch.no_grad()`; the checkpoint, dataset, and
    preprocessing are never modified here. The ground truth returned is the
    UNTOUCHED on-disk anatomy mask (see `load_original_space_ground_truth`),
    never the preprocessed mask round-tripped back.
    """
    geometry, transform_meta = dataset.get_case_metadata(patient_id)

    case_sample_indices = [
        i for i, (pid, _) in enumerate(dataset.samples) if pid == patient_id
    ]
    if not case_sample_indices:
        raise ValueError(f"No samples found for patient '{patient_id}' in this dataset.")

    model.eval()
    pred_slices: List[np.ndarray] = []
    slice_indices: List[int] = []

    with torch.no_grad():
        for i in case_sample_indices:
            sample = dataset[i]
            image = torch.from_numpy(sample["image"]).unsqueeze(0).float().to(device)
            logits = model(image)
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.int64)

            pred_slices.append(pred)
            slice_indices.append(sample["slice_idx"])

    # Preprocessed-space reassembly (the raw 2D predictions, explicit index),
    # then the full inverse-transform to original voxel space. Both are built
    # from the SAME predicted slices, so they can never disagree.
    pred_preprocessed = reconstruct_volume_from_slices(
        pred_slices, slice_indices, transform_meta.post_resample_depth
    )
    pred_nii = reconstruct_case_to_original_space(pred_slices, slice_indices, transform_meta, geometry)
    pred_original = np.asarray(pred_nii.dataobj)

    gt_original = load_original_space_ground_truth(dataset, patient_id, geometry)

    return CasePrediction(
        patient_id=patient_id,
        pred_nii=pred_nii,
        pred_original=pred_original,
        pred_preprocessed=pred_preprocessed,
        slice_indices=slice_indices,
        gt_original=gt_original,
        geometry=geometry,
        transform_meta=transform_meta,
    )


def predict_case_original_space(
    patient_id: str,
    model: Any,
    dataset: Any,
    device: str,
):
    """Run the model on every axial slice of a case, reconstruct the prediction
    to original voxel space, and pair it with the untouched original-space GT.

    This is the single shared inference + 2D->3D reconstruction path used by
    the baseline evaluation (`evaluate_patient` calls this) AND by the
    qualitative visualization (`scripts/visualize_baseline_cases.py`), so the
    two can never silently diverge. Inference only: `model.eval()` +
    `torch.no_grad()`; the checkpoint, dataset, and preprocessing are never
    modified here.

    The ground truth is the UNTOUCHED on-disk anatomy mask (see
    `load_original_space_ground_truth`), NOT the preprocessed mask
    inverse-transformed back -- so lossy preprocessing (Exp02-C resize) cannot
    inflate the score by degrading prediction and GT identically. For the
    lossless crop_pad baseline the two are bit-identical, so Experiment-1
    numbers are unchanged.

    Returns
    -------
    (pred_original, gt_original, geometry)
        pred_original : int label volume (H, W, D) -- the model prediction
            reconstructed and inverse-transformed to the case's ORIGINAL voxel
            grid (source affine restored).
        gt_original : int label volume (H, W, D) -- the untouched on-disk
            anatomy mask in that same original voxel grid.
        geometry : VolumeGeometry for the case (shape, affine, zooms, axcodes).
    """
    result = predict_case_reconstructed(patient_id, model, dataset, device)
    return result.pred_original, result.gt_original, result.geometry


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

    # Protocol A (primary): reconstruct + invert the PREDICTION to original
    # voxel space, and score it against the UNTOUCHED on-disk GT (never the
    # preprocessed mask round-tripped back -- see load_original_space_ground_truth
    # for why: a lossy resize round-trip would otherwise inflate this arm).
    pred_nii = reconstruct_case_to_original_space(pred_slices, slice_indices, transform_meta, geometry)
    pred_original = np.asarray(pred_nii.dataobj)
    gt_original = load_original_space_ground_truth(dataset, patient_id, geometry)

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


def _resolve_split_dataset_root(data_cfg: Dict[str, Any], split: str) -> str:
    """Return the dataset root directory to use for a given split.

    The official Prostate158 test cases live in a SEPARATE archive/directory
    from the 139-case train+validation deposit. So:
      * split == "test"  -> data.test_dir (REQUIRED; raises if unset)
      * train / val      -> data.dataset_root (unchanged; Exp-1 behavior)

    Requiring test_dir for the test split makes it impossible for a test run
    to silently fall back to -- and accidentally read -- the train/validation
    case directories.
    """
    if split == "test":
        test_dir = data_cfg.get("test_dir")
        if not test_dir:
            raise ValueError(
                "split='test' requires data.test_dir to point at the extracted, "
                "isolated test case-root (the directory that directly contains the "
                "19 test case folders). Refusing to fall back to the train/val "
                "dataset_root for the held-out test set."
            )
        return test_dir
    return data_cfg["dataset_root"]


def run_evaluation(
    config_path: str,
    checkpoint_path: str,
    output_dir: Optional[str] = None,
    split: str = "val",
    device: Optional[str] = None,
    class_ids: Sequence[int] = (0, 1, 2),
) -> Dict[str, Any]:
    """End-to-end baseline evaluation orchestration (inference only -- NO training).

    Loads a trained checkpoint, rebuilds the model and the *exact* preprocessing
    used to train it, evaluates every case of the requested official split at
    the volume level in original voxel space (per the thesis protocol), and
    writes machine-readable result files (per-case CSV, cohort summary CSV +
    JSON) to `output_dir`.

    Design decisions (all to keep evaluation faithful to how the weights were
    produced, and to avoid silently changing the evaluation):
      * The model architecture and the preprocessing config are taken from the
        checkpoint's own embedded `config` when present (that is literally what
        trained these weights), falling back to `config_path` otherwise.
      * `dataset_root`, the official split CSVs, `target_mask`, and the default
        `output_dir` come from `config_path`.
      * slice_sampling is FORCED to "all" -- evaluation must always see the
        complete slice population (never drop empty/background slices), no
        matter what the config says.
      * Nothing here trains, fine-tunes, or mutates the checkpoint. The model is
        put in eval() mode and run under torch.no_grad() (in evaluate_patient).

    Parameters
    ----------
    config_path : str
        Path to the experiment YAML (e.g. configs/config_baseline.yaml). Supplies
        the dataset root, official split CSVs, target mask, and default output_dir.
    checkpoint_path : str
        Path to the trained checkpoint (e.g. the Drive-backed Epoch-77
        best_model.pt). Only its weights + embedded config are read.
    output_dir : Optional[str]
        Where to write result files. Defaults to the config's
        experiment.output_dir.
    split : str
        Which official split to evaluate: "val"/"validation" (default), "test"
        (requires the 19-case test archive -- errors clearly if absent), or
        "train".
    device : Optional[str]
        Torch device string; auto-detected (cuda/mps/cpu) if None.
    class_ids : Sequence[int]
        Classes to score (default background + 2 anatomy classes).

    Returns
    -------
    Dict[str, Any]
        {"per_case_csv", "summary_csv", "summary_json", "summary",
         "case_results", "output_dir", "metadata"}.
    """
    import datetime

    from src.dataset import ProstateZonal2DDataset
    from src.metrics import (
        DEFAULT_CLASS_NAMES,
        write_case_metrics_csv,
        write_summary_csv,
        write_summary_json,
    )
    from src.model import build_model
    from src.splits import load_official_split
    from src.transforms import PreprocessingConfig
    from src.utils import ensure_dir, get_device, load_config

    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})

    if device is None:
        device = get_device()

    # --- Checkpoint (weights + the config that produced them) ---
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint '{checkpoint_path}' has no 'model_state_dict'. "
            f"Keys present: {sorted(checkpoint.keys())}"
        )
    ckpt_cfg = checkpoint.get("config", cfg)

    # Rebuild the model from the checkpoint's own config so the architecture
    # always matches the stored weights, then load the weights.
    model = build_model(ckpt_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Preprocessing MUST match training exactly (same crop/pad size etc.).
    preproc = PreprocessingConfig.from_dict(
        ckpt_cfg.get("preprocessing", cfg.get("preprocessing", {}))
    )

    # --- Resolve the official, case-level split ---
    if "train_csv" in data_cfg and "valid_csv" in data_cfg:
        official = load_official_split(
            train_csv=data_cfg["train_csv"],
            valid_csv=data_cfg["valid_csv"],
            test_csv=data_cfg.get("test_csv"),
            test_dir=data_cfg.get("test_dir"),
        )
        split_ids_map = {
            "train": official.train_ids,
            "val": official.val_ids,
            "validation": official.val_ids,
            "test": official.test_ids,
        }
    else:
        split_ids_map = {
            "train": data_cfg.get("patient_ids", []),
            "val": cfg.get("validation", {}).get("val_patient_ids", []),
            "validation": cfg.get("validation", {}).get("val_patient_ids", []),
            "test": None,
        }

    if split not in split_ids_map:
        raise ValueError(f"Unknown split '{split}'. Choose from {sorted(split_ids_map)}.")

    patient_ids = split_ids_map[split]
    if split == "test" and not patient_ids:
        raise ValueError(
            "Requested split='test' but the official 19-case Prostate158 test "
            "archive (DOI 10.5281/zenodo.6592345) is not available locally. "
            "Point data.test_csv/test_dir at it once obtained; evaluate on the "
            "validation split in the meantime."
        )
    if not patient_ids:
        raise ValueError(f"No case IDs resolved for split='{split}'.")

    is_official_test = split == "test"

    # The official test cases live in a SEPARATE archive/directory from
    # train/val. For the test split, require and use data.test_dir as the
    # dataset root, so the test split can NEVER silently fall back to reading
    # train/validation case directories. Train/val always use dataset_root,
    # so this branch does not affect the Experiment-1 validation evaluation.
    split_dataset_root = _resolve_split_dataset_root(data_cfg, split)

    print("=" * 78)
    print(f"BASELINE EVALUATION (inference only, no training)")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  ckpt epoch : {checkpoint.get('epoch', 'unknown')}  "
          f"(training-loop best metric: {checkpoint.get('best_metric', float('nan'))})")
    print(f"  split      : {split}  ({len(patient_ids)} cases)")
    print(f"  data root  : {split_dataset_root}")
    if not is_official_test:
        print("  NOTE       : this is the VALIDATION split, NOT the official held-out "
              "19-case test set. These numbers are not final test results.")
    print(f"  device     : {device}")
    print("=" * 78)

    dataset = ProstateZonal2DDataset(
        dataset_root=split_dataset_root,
        patient_ids=patient_ids,
        image_name="t2.nii.gz",
        mask_name=data_cfg.get("target_mask", "t2_anatomy_reader1.nii.gz"),
        slice_sampling="all",  # FORCED: evaluation must see the full slice population
        preprocessing_config=preproc,
        cache_data=True,
    )

    case_results, summary = evaluate_cases(
        model, dataset, device, patient_ids=patient_ids, class_ids=class_ids
    )

    out = ensure_dir(output_dir or cfg.get("experiment", {}).get("output_dir", "./results"))
    tag = f"eval_{split}"

    metadata = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "training_loop_val_metric_slice_level": float(checkpoint.get("best_metric", float("nan"))),
        "config_path": config_path,
        "split": split,
        "n_cases": len(case_results),
        "case_ids": [c.patient_id for c in case_results],
        "is_official_held_out_test": is_official_test,
        "class_ids": [int(c) for c in class_ids],
        "class_names": {int(c): DEFAULT_CLASS_NAMES.get(c, str(c)) for c in class_ids},
        "evaluation_space": "original_voxel",
        "ground_truth_source": "original_on_disk_mask_untouched",
        "aggregation": "per_case",
        "label_mapping_status": "unverified (see results/label_mapping/label_mapping_report.md)",
        "preprocessing": {
            "normalization": preproc.normalization,
            "z_resample": preproc.z_resample,
            "spatial_mode": preproc.spatial_mode,
            "target_size": list(preproc.target_size),
        },
        "primary_metric": "dice (per class, per case, mean +/- SD across cases)",
        "note": (
            "Volume-level, original-voxel-space, per-case metrics. Distinct from "
            "the training loop's slice-level val_dice (see "
            "training_loop_val_metric_slice_level above)."
        ),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    per_case_csv = write_case_metrics_csv(
        os.path.join(out, f"{tag}_per_case_metrics.csv"), case_results, class_ids
    )
    summary_csv = write_summary_csv(
        os.path.join(out, f"{tag}_summary.csv"), summary, class_ids
    )
    summary_json = write_summary_json(
        os.path.join(out, f"{tag}_summary.json"), summary, metadata
    )

    print(f"\nWrote:\n  {per_case_csv}\n  {summary_csv}\n  {summary_json}")
    for c in class_ids:
        d = summary[c]["dice"]
        print(f"  class {c} ({DEFAULT_CLASS_NAMES.get(c, c)}): "
              f"Dice mean={d['mean']:.4f} SD={d['std']:.4f} "
              f"95%CI=[{d['ci95_low']:.4f}, {d['ci95_high']:.4f}] n={d['n_cases']}")

    return {
        "per_case_csv": per_case_csv,
        "summary_csv": summary_csv,
        "summary_json": summary_json,
        "summary": summary,
        "case_results": case_results,
        "output_dir": out,
        "metadata": metadata,
    }
