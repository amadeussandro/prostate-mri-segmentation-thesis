"""RQ2 -- 2D->3D reconstruction of the FROZEN 2D U-Net's predictions (inference only).

Thesis RQ2:
    "Bagaimana Implementasi Rekonstruksi 3D Citra MRI Pasien Kanker Prostat dari
     hasil Segmentasi 2D U-Net?"

This is the single end-to-end RQ2 entry point. It does NOT train, retrain,
fine-tune, or modify the checkpoint -- it consumes the existing Experiment-1
(epoch-77) `best_model.pt` and, for every case of the official held-out TEST
split (19 cases):

    original MRI (t2.nii.gz)
      -> ProstateZonal2DDataset preprocessing (the SAME preprocessing that
         trained the checkpoint; read from the checkpoint's embedded config)
      -> 2D axial slices
      -> frozen 2D U-Net (eval / no_grad, argmax over 3 classes)
      -> predicted 2D masks
      -> explicit-index slice reassembly              (src.reconstruction)
      -> inverse preprocessing (nearest-neighbour)    (src.transforms)
      -> original voxel space, source affine restored (src.reconstruction)
      -> reconstructed 3D segmentation NIfTI (saved)
      -> geometry validation                          (src.reconstruction.validate_reconstruction_geometry)
      -> quantitative evaluation vs untouched GT      (src.metrics)
      -> qualitative visualization                    (scripts.visualize_baseline_cases + 3D projections)

Everything scientific is reused from src/: this script only composes those
building blocks and writes the RQ2 result artifacts + report.

IMPORTANT distinctions (see report.md):
  * The round-trip fidelity result (Dice = 1.0, scripts/roundtrip_test.py) proves
    the reconstruction TRANSFORM is loss-free. It is NOT the model's accuracy.
  * The REAL 3D quantitative numbers here are the model's per-case Dice/IoU/
    HD95/ASD/precision/recall in original voxel space -- a different quantity.

Label mapping (0=background, 1=CG central gland, 2=PZ peripheral zone) is the
mapping human-verified in 3D Slicer for this thesis; it is never changed here.

Typical Colab usage (checkpoint + test archive on Google Drive):

    python scripts/run_rq2_reconstruction.py \
        --config configs/config_baseline.yaml \
        --checkpoint /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt \
        --test-dir  /content/drive/MyDrive/THESIS_PROSTATE158/dataset/test/extracted/<case-root> \
        --output-dir /content/drive/MyDrive/THESIS_PROSTATE158/results/exp04_rq2_reconstruction \
        --split test
"""

import argparse
import csv
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import numpy as np

# Label mapping human-verified in 3D Slicer for this thesis (cases 020, 059).
# RQ2 reports under these names; the integer->zone mapping is never changed here.
RQ2_CLASS_NAMES: Dict[int, str] = {
    0: "background",
    1: "CG_central_gland",
    2: "PZ_peripheral_zone",
}
FOREGROUND_CLASS_IDS = (1, 2)

# Colorblind-safe overlay colors, consistent with scripts/visualize_baseline_cases.py.
CLASS_COLORS = {1: (0.95, 0.55, 0.10), 2: (0.10, 0.70, 0.90)}


# ---------------------------------------------------------------------------
# Objective representative-case selection (pure logic -- unit-testable)
# ---------------------------------------------------------------------------

def mean_foreground_dice(case_metric, class_ids: Sequence[int] = FOREGROUND_CLASS_IDS) -> float:
    """Mean Dice over the foreground classes for one CaseMetrics."""
    vals = [case_metric.per_class[c]["dice"] for c in class_ids if c in case_metric.per_class]
    return float(np.mean(vals)) if vals else float("nan")


def select_representative_cases(
    case_results,
    class_ids: Sequence[int] = FOREGROUND_CLASS_IDS,
) -> Dict[str, str]:
    """Objective, documented selection: rank cases by mean foreground Dice and
    return one 'good' (highest), one 'median' (middle rank), one 'challenging'
    (lowest). Ties are broken by patient_id for determinism. Returns an ordered
    {label -> patient_id} dict; labels collapse if fewer than 3 cases exist.
    """
    scored = sorted(
        ((mean_foreground_dice(cm, class_ids), cm.patient_id) for cm in case_results),
        key=lambda kv: (kv[0], kv[1]),
    )
    if not scored:
        return {}
    pids_asc = [pid for _, pid in scored]
    picks: Dict[str, str] = {}
    picks["challenging"] = pids_asc[0]
    picks["median"] = pids_asc[len(pids_asc) // 2]
    picks["good"] = pids_asc[-1]
    # Deduplicate while keeping the good/median/challenging labels meaningful.
    return picks


# ---------------------------------------------------------------------------
# Result serialization
# ---------------------------------------------------------------------------

def write_reconstruction_validation_csv(path: str, rows: List[Dict[str, Any]]) -> str:
    """Per-case geometry-validation table (RQ2 Step 5)."""
    fieldnames = [
        "patient_id", "all_ok",
        "shape_match", "recon_shape", "source_shape", "depth_match",
        "affine_match", "affine_max_abs_diff",
        "spacing_match", "recon_spacing_mm", "source_spacing_mm",
        "orientation_match", "recon_orientation", "source_orientation",
        "labels_found", "labels_valid",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    return path


# ---------------------------------------------------------------------------
# 3D (multi-plane projection) visualization -- matplotlib only, no extra deps
# ---------------------------------------------------------------------------

def render_3d_projections(
    t2: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    patient_id: str,
    out_path: str,
    class_colors: Dict[int, tuple] = CLASS_COLORS,
) -> str:
    """Pseudo-3D comparison: for each of the three anatomical planes (axial,
    coronal, sagittal) show the T2 maximum-intensity projection with the GT
    (top row) and predicted (bottom row) label silhouettes overlaid -- a
    foreground voxel of a class anywhere along the projected axis colors that
    pixel. Dependency-free (matplotlib), a lightweight stand-in for a full
    surface render (which can be produced from the saved NIfTI in 3D Slicer)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    axes_names = ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"]
    proj_axis = [2, 1, 0]

    fig, axgrid = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Patient {patient_id} -- 3D label projections (T2 MIP + label silhouette)\n"
        f"Top: Ground Truth   Bottom: Prediction   "
        f"(CG=orange, PZ=cyan; original voxel space)",
        fontsize=13,
    )

    for row, (vol, tag) in enumerate([(gt, "GT"), (pred, "Pred")]):
        for col, ax_id in enumerate(proj_axis):
            t2_mip = t2.max(axis=ax_id)
            lo, hi = (np.percentile(t2_mip, [1, 99]) if np.ptp(t2_mip) > 0 else (0.0, 1.0))
            ax = axgrid[row][col]
            ax.imshow(t2_mip.T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
            overlay = np.zeros((*t2_mip.shape, 4), dtype=float)
            for cid, color in class_colors.items():
                footprint = (vol == cid).any(axis=ax_id)
                overlay[footprint, 0], overlay[footprint, 1], overlay[footprint, 2] = color
                overlay[footprint, 3] = 0.55
            ax.imshow(np.transpose(overlay, (1, 0, 2)), origin="lower")
            ax.set_title(f"{tag} -- {axes_names[col]}")
            ax.axis("off")

    legend = [
        Patch(facecolor=class_colors[1], label="CG (Central Gland)"),
        Patch(facecolor=class_colors[2], label="PZ (Peripheral Zone)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_mean_sd(summary: Dict[int, Dict[str, dict]], class_id: int, metric: str) -> str:
    s = summary.get(class_id, {}).get(metric)
    if not s or s.get("mean") is None or (isinstance(s.get("mean"), float) and np.isnan(s["mean"])):
        return "n/a"
    return f"{s['mean']:.4f} +/- {s['std']:.4f}"


def write_report(path: str, ctx: Dict[str, Any]) -> str:
    """Render the 13-section RQ2 technical report from a completed run's context."""
    summary = ctx["summary"]
    md = []
    md.append("# RQ2 -- 3D Reconstruction of 2D U-Net Segmentation (Technical Report)\n")
    md.append(f"_Generated: {ctx['generated_utc']} | split: {ctx['split']} "
              f"({ctx['n_cases']} cases) | checkpoint epoch: {ctx['checkpoint_epoch']}_\n")

    md.append("## 1. Objective\n")
    md.append(
        "Answer RQ2: implement and validate the reconstruction of the frozen 2D "
        "U-Net's slice-wise predictions into the original 3D MRI voxel space, "
        "preserving the original geometry, and quantify the reconstructed 3D "
        "segmentation against the untouched ground truth. **RQ1 (the model and "
        "its training) is frozen; no retraining was performed.**\n")

    md.append("## 2. Input data\n")
    md.append(
        f"- Dataset: Prostate158, official held-out **{ctx['split']}** split "
        f"({ctx['n_cases']} cases: `{', '.join(ctx['case_ids'])}`).\n"
        f"- Input volume: `t2.nii.gz` (T2W, single channel).\n"
        f"- Ground truth: `{ctx['mask_name']}` (labels {{0,1,2}}), read untouched from disk.\n"
        f"- Data root (this split): `{ctx['split_root']}`.\n")

    md.append("## 3. Existing checkpoint used\n")
    md.append(
        f"- Path: `{ctx['checkpoint_path']}`\n"
        f"- Best epoch: **{ctx['checkpoint_epoch']}** "
        f"(training-loop slice-level best metric: {ctx['checkpoint_best_metric']}).\n"
        f"- Architecture: `{ctx['architecture']}`, in_channels={ctx['in_channels']}, "
        f"out_channels={ctx['out_channels']} (0=background, 1=CG, 2=PZ).\n"
        "- The checkpoint, its weights, and the training pipeline were not modified.\n")

    md.append("## 4. 2D inference procedure\n")
    md.append(
        "For each case, every axial slice of the preprocessed T2 volume is passed "
        "through the frozen U-Net under `model.eval()` + `torch.no_grad()`; the "
        "per-slice prediction is `argmax` over the 3 class logits. Preprocessing is "
        "rebuilt from the checkpoint's own embedded config, so inference sees exactly "
        "the pipeline the weights were trained on "
        "(`src.evaluate.predict_case_reconstructed`).\n")

    md.append("## 5. Reconstruction procedure\n")
    md.append(
        "Predicted 2D slices are reassembled into a (H, W, D) volume by **explicit "
        "slice index** (never append order); missing/duplicate indices raise "
        "(`src.reconstruction.reconstruct_volume_from_slices`).\n")

    md.append("## 6. Inverse preprocessing\n")
    md.append(
        "The preprocessed-space volume is inverse-transformed in reverse order "
        "(undo crop/pad -> undo z-resample -> undo RAS orientation), always with "
        "nearest-neighbour interpolation to preserve integer labels "
        "(`src.transforms.invert_preprocessing_mask`).\n")

    md.append("## 7. Original-space restoration\n")
    md.append(
        "The reconstructed volume is wrapped in a NIfTI with the **original source "
        "affine and voxel spacing restored exactly** (never a default identity "
        "affine); the geometry is then re-read from the output header and checked "
        "against the source (`src.reconstruction.reconstruct_case_to_original_space`, "
        "9-point checklist in `src.geometry`). Saved to `reconstructions_3d/`.\n")

    md.append("## 8. Geometry validation\n")
    n_ok = ctx["n_validation_ok"]
    md.append(
        f"Per-case machine-readable validation (`reconstruction_validation.csv`): "
        f"**{n_ok}/{ctx['n_cases']}** cases pass all checks (shape, affine, spacing, "
        f"orientation, depth/slice-order, labels subset of {{0,1,2}}). "
        f"`validate_reconstruction_geometry` reads geometry back from each output "
        f"NIfTI's own header.\n")

    md.append("## 9. Round-trip fidelity result\n")
    md.append(
        "Reconstruction-transform fidelity is established by `scripts/roundtrip_test.py` "
        "(and `tests/test_reconstruction.py` / `tests/test_rq2_reconstruction.py`): "
        "GT mask -> preprocess -> reconstruct -> inverse -> original space yields "
        "**Dice = 1.0 exactly** for classes 0/1/2 with shape and affine exact, on "
        "both the identity and baseline-(442,442) suites, **with no model involved**.\n\n"
        "> **This Dice = 1.0 is a property of the reconstruction transform only. It is "
        "NOT the segmentation accuracy of the model. The model's real accuracy is in "
        "Section 10.**\n")

    md.append("## 10. Real 3D quantitative results\n")
    md.append(
        "Model predictions reconstructed to original voxel space, scored per case "
        "against the untouched GT, then aggregated (mean +/- SD across "
        f"{ctx['n_cases']} cases; per-case first, never slice-pooled):\n\n")
    md.append("| Class | Dice | IoU | HD95 (mm) | ASD (mm) | Precision | Recall |\n")
    md.append("|---|---|---|---|---|---|---|\n")
    for c in (1, 2):
        md.append(
            f"| {RQ2_CLASS_NAMES[c]} | {_fmt_mean_sd(summary, c, 'dice')} | "
            f"{_fmt_mean_sd(summary, c, 'iou')} | {_fmt_mean_sd(summary, c, 'hd95_mm')} | "
            f"{_fmt_mean_sd(summary, c, 'asd_mm')} | {_fmt_mean_sd(summary, c, 'precision')} | "
            f"{_fmt_mean_sd(summary, c, 'recall')} |\n")
    md.append(
        "\nThese are volume-level, original-voxel-space, per-case metrics -- distinct "
        "from any slice-level training-loop `val_dice`.\n")

    md.append("## 11. Qualitative visualization cases\n")
    md.append(
        "Representative cases selected by an objective rule (rank by mean foreground "
        "Dice; pick highest = good, middle = median, lowest = challenging):\n\n")
    for label, info in ctx["representatives"].items():
        md.append(f"- **{label}**: patient `{info['patient_id']}` "
                  f"(mean foreground Dice {info['mean_fg_dice']:.4f}) -- "
                  f"axial panels + 3D projections under `visualizations/patient_{info['patient_id']}/`.\n")

    md.append("\n## 12. Limitations\n")
    md.append(
        "- 2D slice-wise model: no explicit inter-slice (z) context; residual axial "
        "discontinuities are possible (see `src.metrics.inter_slice_dice_inconsistency`).\n"
        "- PZ is a thin peripheral structure and scores lower / with higher variance "
        "than CG -- expected for zonal segmentation.\n"
        "- HD95/ASD are undefined for a class absent from a case (prediction or GT) "
        "and are excluded from that case's aggregate (NaN-skipping).\n"
        "- n = 19 (official test): report mean +/- SD and bootstrap CI, not strong "
        "significance claims.\n")

    md.append("## 13. Reproducibility instructions\n")
    md.append(
        "```bash\n"
        "# 1) Reconstruction-transform fidelity (no model, needs the local dataset):\n"
        "python scripts/roundtrip_test.py\n\n"
        "# 2) Torch-free RQ2 unit tests (geometry validation, synthetic round-trip):\n"
        "python -m pytest tests/test_rq2_reconstruction.py -q\n\n"
        "# 3) Full RQ2 run (Colab/GPU, checkpoint + test archive on Drive):\n"
        "python scripts/run_rq2_reconstruction.py \\\n"
        "    --config configs/config_baseline.yaml \\\n"
        f"    --checkpoint {ctx['checkpoint_path']} \\\n"
        "    --test-dir <extracted 19-case test root> \\\n"
        "    --output-dir results/exp04_rq2_reconstruction \\\n"
        "    --split test\n"
        "```\n\n"
        "**Frozen-scope statement:** RQ1 model/training was frozen; no retraining, no "
        "architecture/hyperparameter/split change, and `results/exp01_baseline` was "
        "not overwritten. RQ2 uses the existing 2D U-Net predictions only. Round-trip "
        "Dice = 1.0 is a reconstruction-fidelity result, not model segmentation accuracy.\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    return path


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def apply_data_path_overrides(
    cfg: dict,
    dataset_root: Optional[str] = None,
    train_csv: Optional[str] = None,
    valid_csv: Optional[str] = None,
    test_dir: Optional[str] = None,
    test_csv: Optional[str] = None,
) -> dict:
    """Overlay explicit dataset paths onto cfg["data"], returning the data dict.

    Only non-None overrides are applied, so when nothing is passed the config's
    own paths are preserved verbatim (backward-compatible local/non-Colab usage).
    On Colab the dataset lives on Google Drive (outside the repo), so the
    notebook passes absolute Drive paths here; no Drive path is hardcoded in
    source. This does not change the split or evaluation methodology -- it only
    changes WHERE the same official CSVs / case directories are read from.
    """
    data_cfg = cfg.setdefault("data", {})
    for key, value in (
        ("dataset_root", dataset_root),
        ("train_csv", train_csv),
        ("valid_csv", valid_csv),
        ("test_dir", test_dir),
        ("test_csv", test_csv),
    ):
        if value:
            data_cfg[key] = value
    return data_cfg


def validate_data_paths(data_cfg: dict, split: str) -> None:
    """Fail early and clearly if the dataset paths needed for `split` are absent.

    load_official_split always reads train_csv + valid_csv (to enforce the
    official 119/20 counts and the leakage check), so both must exist for every
    split. The 'test' split additionally needs test_dir (the extracted 19-case
    case-root). Raises FileNotFoundError naming the missing path and the CLI flag
    that supplies it -- instead of a downstream pandas/nibabel error.
    """
    missing = []
    for key, flag in (("train_csv", "--train-csv"), ("valid_csv", "--valid-csv")):
        path = data_cfg.get(key)
        if not path or not os.path.isfile(path):
            missing.append(f"{key} ({flag}): {path!r}")
    if split == "test":
        test_dir = data_cfg.get("test_dir")
        if not test_dir or not os.path.isdir(test_dir):
            missing.append(f"test_dir (--test-dir): {test_dir!r}")
    if missing:
        raise FileNotFoundError(
            "Required dataset path(s) not found. On Colab the Prostate158 data "
            "lives on Google Drive, not in the repo -- pass the Drive path(s) "
            "explicitly. Missing:\n  - " + "\n  - ".join(missing)
        )


def _resolve_split_ids(cfg: dict, split: str):
    from src.splits import load_official_split

    data_cfg = cfg.get("data", {})
    official = load_official_split(
        train_csv=data_cfg["train_csv"],
        valid_csv=data_cfg["valid_csv"],
        test_csv=data_cfg.get("test_csv"),
        test_dir=data_cfg.get("test_dir"),
    )
    mapping = {
        "train": official.train_ids,
        "val": official.val_ids,
        "validation": official.val_ids,
        "test": official.test_ids,
    }
    if split not in mapping:
        raise ValueError(f"Unknown split '{split}'. Choose from {sorted(mapping)}.")
    ids = mapping[split]
    if not ids:
        raise ValueError(
            f"No case IDs resolved for split='{split}'. For 'test', provide the "
            "19-case archive via --test-dir/--test-csv (data.test_dir/test_csv)."
        )
    return list(ids)


def run(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    split: str = "test",
    test_dir: Optional[str] = None,
    test_csv: Optional[str] = None,
    dataset_root: Optional[str] = None,
    train_csv: Optional[str] = None,
    valid_csv: Optional[str] = None,
    device: Optional[str] = None,
    class_ids: Sequence[int] = (0, 1, 2),
    make_viz: bool = True,
    n_slices: int = 3,
) -> Dict[str, Any]:
    import nibabel as nib
    import torch

    from src.dataset import ProstateZonal2DDataset
    from src.evaluate import _resolve_split_dataset_root, predict_case_reconstructed
    from src.metrics import (
        aggregate_case_results,
        evaluate_case_volume,
        write_case_metrics_csv,
        write_summary_csv,
        write_summary_json,
    )
    from src.model import build_model
    from src.reconstruction import validate_reconstruction_geometry
    from src.transforms import PreprocessingConfig
    from src.utils import ensure_dir, get_device, load_config
    from visualize_baseline_cases import render_case_slices, select_slice_indices

    cfg = load_config(config_path)
    # Overlay explicit dataset paths (e.g. Google Drive) onto the config, then
    # fail early with a clear message if anything required for this split is
    # missing. When no overrides are passed, the config's paths are used as-is.
    data_cfg = apply_data_path_overrides(
        cfg,
        dataset_root=dataset_root,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_dir=test_dir,
        test_csv=test_csv,
    )
    validate_data_paths(data_cfg, split)
    if device is None:
        device = get_device()

    # --- Checkpoint (weights + the config that produced them) ---
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint '{checkpoint_path}' has no 'model_state_dict'.")
    ckpt_cfg = checkpoint.get("config", cfg)

    model = build_model(ckpt_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Step 2 -- checkpoint / architecture verification (fail loudly, never train).
    in_ch = int(getattr(model, "in_channels", ckpt_cfg.get("model", {}).get("in_channels", -1)))
    out_ch = int(getattr(model, "out_channels", ckpt_cfg.get("model", {}).get("out_channels", -1)))
    if in_ch != 1:
        raise ValueError(f"Expected a T2W single-channel model (in_channels=1), got {in_ch}.")
    if out_ch != 3:
        raise ValueError(f"Expected a 3-class model (0=bg,1=CG,2=PZ), got out_channels={out_ch}.")

    preproc = PreprocessingConfig.from_dict(
        ckpt_cfg.get("preprocessing", cfg.get("preprocessing", {}))
    )

    split_ids = _resolve_split_ids(cfg, split)
    split_root = _resolve_split_dataset_root(data_cfg, split)
    mask_name = data_cfg.get("target_mask", "t2_anatomy_reader1.nii.gz")

    out_root = ensure_dir(output_dir)
    recon_dir = ensure_dir(os.path.join(out_root, "reconstructions_3d"))
    pred2d_dir = ensure_dir(os.path.join(out_root, "predictions_2d"))
    meta_dir = ensure_dir(os.path.join(out_root, "metadata"))
    viz_dir = ensure_dir(os.path.join(out_root, "visualizations"))

    print("=" * 78)
    print("RQ2 -- 3D RECONSTRUCTION FROM FROZEN 2D U-NET (inference only, no training)")
    print(f"  checkpoint : {checkpoint_path}  (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"  split      : {split}  ({len(split_ids)} cases)")
    print(f"  data root  : {split_root}")
    print(f"  output     : {out_root}")
    print(f"  device     : {device}")
    print("=" * 78)

    case_results = []
    validation_rows: List[Dict[str, Any]] = []
    cached_arrays: Dict[str, Dict[str, np.ndarray]] = {}

    for pid in split_ids:
        dataset = ProstateZonal2DDataset(
            dataset_root=split_root,
            patient_ids=[pid],
            image_name="t2.nii.gz",
            mask_name=mask_name,
            slice_sampling="all",
            preprocessing_config=preproc,
            cache_data=True,
        )
        cp = predict_case_reconstructed(pid, model, dataset, device)

        # Step 4 -- save reconstructed 3D prediction (original voxel space).
        recon_path = os.path.join(recon_dir, f"pred_{pid}.nii.gz")
        nib.save(cp.pred_nii, recon_path)

        # Step 3 provenance -- save the raw reassembled 2D prediction stack + order.
        np.savez_compressed(
            os.path.join(pred2d_dir, f"pred_{pid}.npz"),
            pred_preprocessed=cp.pred_preprocessed.astype(np.uint8),
            slice_indices=np.asarray(cp.slice_indices, dtype=np.int64),
        )

        # Step 5 -- geometry validation (never raises; records a per-case row).
        val = validate_reconstruction_geometry(cp.pred_nii, cp.geometry, valid_labels=class_ids)
        val_row = {"patient_id": pid, **val}
        validation_rows.append(val_row)

        # Step 7 -- real 3D quantitative metrics vs untouched GT.
        cm = evaluate_case_volume(
            pred=cp.pred_original, gt=cp.gt_original,
            spacing=cp.geometry.zooms, class_ids=class_ids, patient_id=pid,
        )
        case_results.append(cm)

        # Per-case metadata (enough to reconstruct + audit).
        with open(os.path.join(meta_dir, f"meta_{pid}.json"), "w") as f:
            json.dump({
                "patient_id": pid,
                "original_shape": [int(s) for s in cp.geometry.shape],
                "affine": np.asarray(cp.geometry.affine).tolist(),
                "spacing_mm": [float(z) for z in cp.geometry.zooms],
                "orientation": list(cp.geometry.axcodes),
                "n_slices": len(cp.slice_indices),
                "preprocessing": {
                    "normalization": preproc.normalization,
                    "z_resample": preproc.z_resample,
                    "spatial_mode": preproc.spatial_mode,
                    "target_size": list(preproc.target_size),
                },
                "transform_meta": {
                    "spatial_mode": cp.transform_meta.spatial_mode,
                    "target_size": list(cp.transform_meta.target_size),
                    "pad_crop_h": vars(cp.transform_meta.pad_crop_h),
                    "pad_crop_w": vars(cp.transform_meta.pad_crop_w),
                    "pre_resample_depth": cp.transform_meta.pre_resample_depth,
                    "post_resample_depth": cp.transform_meta.post_resample_depth,
                },
                "geometry_validation": val,
                "reconstruction_path": recon_path,
            }, f, indent=2, default=str)

        dice_fg = mean_foreground_dice(cm)
        print(f"  {pid}: recon={val['all_ok']}  "
              f"Dice CG={cm.per_class[1]['dice']:.4f} PZ={cm.per_class[2]['dice']:.4f} "
              f"(mean FG {dice_fg:.4f}) -> {os.path.basename(recon_path)}")

    # --- Aggregate + write result files (reuse existing writers, verified CG/PZ names) ---
    summary = aggregate_case_results(case_results, class_ids=class_ids)

    per_case_csv = write_case_metrics_csv(
        os.path.join(out_root, "metrics_per_case.csv"), case_results, class_ids, RQ2_CLASS_NAMES)
    summary_csv = write_summary_csv(
        os.path.join(out_root, "metrics_summary.csv"), summary, class_ids, RQ2_CLASS_NAMES)
    validation_csv = write_reconstruction_validation_csv(
        os.path.join(out_root, "reconstruction_validation.csv"), validation_rows)

    n_validation_ok = sum(1 for r in validation_rows if r["all_ok"])

    metadata = {
        "task": "RQ2_3d_reconstruction",
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "training_loop_val_metric_slice_level": float(checkpoint.get("best_metric", float("nan"))),
        "config_path": config_path,
        "split": split,
        "n_cases": len(case_results),
        "case_ids": [c.patient_id for c in case_results],
        "is_official_held_out_test": split == "test",
        "class_ids": [int(c) for c in class_ids],
        "class_names": {int(c): RQ2_CLASS_NAMES.get(c, str(c)) for c in class_ids},
        "evaluation_space": "original_voxel",
        "ground_truth_source": "original_on_disk_mask_untouched",
        "aggregation": "per_case",
        "label_mapping_status": "verified_3dslicer (0=background, 1=CG, 2=PZ)",
        "reconstruction_geometry_all_ok": f"{n_validation_ok}/{len(case_results)}",
        "roundtrip_fidelity_note": (
            "Dice=1.0 in scripts/roundtrip_test.py is reconstruction-transform "
            "fidelity (no model), NOT model segmentation accuracy."
        ),
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    summary_json = write_summary_json(
        os.path.join(out_root, "summary.json"), summary, metadata, RQ2_CLASS_NAMES)

    # --- Objective representative selection + qualitative visualization ---
    representatives: Dict[str, Any] = {}
    reps = select_representative_cases(case_results)
    if make_viz and reps:
        # Re-run inference only for the (<=3) selected cases to obtain arrays for rendering.
        pid_to_label: Dict[str, List[str]] = {}
        for label, pid in reps.items():
            pid_to_label.setdefault(pid, []).append(label)
        for pid in pid_to_label:
            dataset = ProstateZonal2DDataset(
                dataset_root=split_root, patient_ids=[pid], image_name="t2.nii.gz",
                mask_name=mask_name, slice_sampling="all",
                preprocessing_config=preproc, cache_data=True,
            )
            cp = predict_case_reconstructed(pid, model, dataset, device)
            t2 = np.asarray(nib.load(os.path.join(split_root, pid, "t2.nii.gz")).get_fdata())
            if t2.shape != cp.gt_original.shape:
                raise ValueError(f"T2/GT geometry mismatch for {pid}: {t2.shape} vs {cp.gt_original.shape}")
            patient_dir = ensure_dir(os.path.join(viz_dir, f"patient_{pid}"))
            slice_idx = select_slice_indices(cp.gt_original, cp.pred_original, n=n_slices)
            axial = render_case_slices(
                t2, cp.gt_original, cp.pred_original, slice_idx, pid, patient_dir,
                subtitle="(CG=orange, PZ=cyan; original voxel space; official TEST split)",
                class_labels={1: "CG (Central Gland)", 2: "PZ (Peripheral Zone)"},
            )
            proj = render_3d_projections(
                t2, cp.gt_original, cp.pred_original, pid,
                os.path.join(patient_dir, f"patient_{pid}_3d_projections.png"))
            for label in pid_to_label[pid]:
                cm = next(c for c in case_results if c.patient_id == pid)
                representatives[label] = {
                    "patient_id": pid,
                    "mean_fg_dice": mean_foreground_dice(cm),
                    "axial_figures": axial,
                    "projection_figure": proj,
                }
    elif reps:
        for label, pid in reps.items():
            cm = next(c for c in case_results if c.patient_id == pid)
            representatives[label] = {"patient_id": pid, "mean_fg_dice": mean_foreground_dice(cm)}

    # --- Report ---
    report_ctx = {
        "generated_utc": metadata["generated_utc"],
        "split": split,
        "n_cases": len(case_results),
        "case_ids": [c.patient_id for c in case_results],
        "mask_name": mask_name,
        "split_root": split_root,
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": metadata["checkpoint_epoch"],
        "checkpoint_best_metric": metadata["training_loop_val_metric_slice_level"],
        "architecture": ckpt_cfg.get("model", {}).get("architecture", "ProstateUNet2D"),
        "in_channels": in_ch,
        "out_channels": out_ch,
        "summary": summary,
        "n_validation_ok": n_validation_ok,
        "representatives": representatives,
    }
    report_path = write_report(os.path.join(out_root, "report.md"), report_ctx)

    print("\nRQ2 reconstruction complete.")
    print(f"  reconstructions : {recon_dir}")
    print(f"  metrics (case)  : {per_case_csv}")
    print(f"  metrics (summary): {summary_csv}")
    print(f"  geometry valid. : {validation_csv}  ({n_validation_ok}/{len(case_results)} all_ok)")
    print(f"  summary JSON    : {summary_json}")
    print(f"  report          : {report_path}")
    for c in (1, 2):
        s = summary[c]["dice"]
        print(f"  {RQ2_CLASS_NAMES[c]} Dice: mean={s['mean']:.4f} SD={s['std']:.4f} n={s['n_cases']}")

    return {
        "output_dir": out_root,
        "reconstructions_dir": recon_dir,
        "per_case_csv": per_case_csv,
        "summary_csv": summary_csv,
        "validation_csv": validation_csv,
        "summary_json": summary_json,
        "report_path": report_path,
        "summary": summary,
        "representatives": representatives,
        "n_validation_ok": n_validation_ok,
        "n_cases": len(case_results),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ2 3D reconstruction from the frozen 2D U-Net (inference only).")
    parser.add_argument("--config", default="configs/config_baseline.yaml")
    parser.add_argument("--checkpoint", required=True, help="Path to the frozen best_model.pt (epoch 77).")
    parser.add_argument("--output-dir", default="results/exp04_rq2_reconstruction")
    parser.add_argument("--split", default="test", choices=["test", "val", "validation", "train"])
    parser.add_argument("--test-dir", default=None, help="Extracted 19-case test root (data.test_dir).")
    parser.add_argument("--test-csv", default=None, help="CSV of test case IDs (data.test_csv).")
    parser.add_argument("--dataset-root", default=None,
                        help="Override data.dataset_root (train/val split root); e.g. a Google Drive path.")
    parser.add_argument("--train-csv", default=None,
                        help="Override data.train_csv (official 119-case split); e.g. a Google Drive path.")
    parser.add_argument("--valid-csv", default=None,
                        help="Override data.valid_csv (official 20-case split); e.g. a Google Drive path.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-viz", action="store_true", help="Skip qualitative visualization.")
    parser.add_argument("--n-slices", type=int, default=3)
    args = parser.parse_args()

    run(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        test_dir=args.test_dir,
        test_csv=args.test_csv,
        dataset_root=args.dataset_root,
        train_csv=args.train_csv,
        valid_csv=args.valid_csv,
        device=args.device,
        make_viz=not args.no_viz,
        n_slices=args.n_slices,
    )


if __name__ == "__main__":
    main()
