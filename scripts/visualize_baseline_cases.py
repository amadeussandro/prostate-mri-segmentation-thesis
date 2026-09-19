"""Qualitative visualization of the Experiment 1 baseline (inference only).

Renders axial GT-vs-prediction panels for selected Prostate158 cases, using
the SAME production inference + preprocessing + 2D->3D reconstruction path as
the baseline evaluation:

    checkpoint -> ProstateZonal2DDataset (preprocessing) -> per-slice inference
    -> reconstruct_case_to_original_space (inverse transforms) -> original
       voxel space -> compare GT vs prediction

Nothing here trains, resumes, or modifies the checkpoint/dataset. The model is
run under eval()/no_grad() via src.evaluate.predict_case_original_space, the
single shared inference+reconstruction entry point (so this can never diverge
from how the baseline was evaluated).

IMPORTANT (labels): the integer->anatomy mapping of Class 1 / Class 2 is
UNVERIFIED in this repository, so every figure/label says exactly "Class 1" /
"Class 2" -- never CG/TZ/PZ.

IMPORTANT (split): --split val is the official 20-case VALIDATION split, which
is NOT the held-out 19-case TEST split. Validation results are never test
results.

Typical usage (Colab, checkpoint on Drive):

    python scripts/visualize_baseline_cases.py \
        --config configs/config_baseline.yaml \
        --checkpoint /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt \
        --metrics-csv /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/eval_val_per_case_metrics.csv \
        --output-dir /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/visualizations \
        --split val --mode worst --n 3

    # or explicit patients:
        ... --patients 60 77 87
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

# Neutral, verified-safe display labels. The Class 1 / Class 2 -> anatomy
# mapping is UNVERIFIED (see results/label_mapping/), so we never write CG/TZ/PZ.
CLASS_DISPLAY_LABELS: Dict[int, str] = {
    0: "Background",
    1: "Class 1",
    2: "Class 2",
}
FOREGROUND_CLASS_IDS = (1, 2)


# ---------------------------------------------------------------------------
# Case selection (pure logic -- unit-testable without torch / the dataset)
# ---------------------------------------------------------------------------

def normalize_patient_id(raw: str, zero_pad: int = 3) -> str:
    """Normalize a user-supplied patient id to the repo's zero-padded form.
    Accepts '60', '060', 60 -> '060'. Non-numeric ids are returned stripped
    (some cohorts may use non-numeric ids)."""
    s = str(raw).strip()
    if s.isdigit():
        return s.zfill(zero_pad)
    return s


def load_per_case_dice(metrics_csv: str, class_ids: Sequence[int] = FOREGROUND_CLASS_IDS) -> Dict[str, float]:
    """Read eval_*_per_case_metrics.csv (long form: patient_id, class_id, ...,
    dice, ...) and return {patient_id -> mean Dice over the given foreground
    classes}. Uses only the actual CSV contents (no hardcoded patient ids)."""
    import pandas as pd

    df = pd.read_csv(metrics_csv)
    for col in ("patient_id", "class_id", "dice"):
        if col not in df.columns:
            raise ValueError(f"metrics CSV missing required column '{col}'; found {list(df.columns)}")

    df = df[df["class_id"].isin(list(class_ids))].copy()
    df["patient_id"] = df["patient_id"].apply(normalize_patient_id)
    means = df.groupby("patient_id")["dice"].mean()
    return {pid: float(v) for pid, v in means.items()}


def select_patients_by_mode(
    metrics_csv: str,
    mode: str,
    n: int,
    class_ids: Sequence[int] = FOREGROUND_CLASS_IDS,
) -> List[str]:
    """Select n patient ids from the metrics CSV by mean foreground Dice.
    mode: 'worst' (lowest first), 'best' (highest first), 'median' (n cases
    centered on the median)."""
    dice_by_pid = load_per_case_dice(metrics_csv, class_ids)
    if not dice_by_pid:
        raise ValueError(f"No usable rows in {metrics_csv} for classes {tuple(class_ids)}.")

    # Sort ascending by dice, tie-broken by patient id for determinism.
    ordered = sorted(dice_by_pid.items(), key=lambda kv: (kv[1], kv[0]))
    pids_asc = [pid for pid, _ in ordered]

    n = max(1, min(n, len(pids_asc)))
    if mode == "worst":
        return pids_asc[:n]
    if mode == "best":
        return pids_asc[::-1][:n]
    if mode == "median":
        mid = len(pids_asc) // 2
        half = n // 2
        start = max(0, mid - half)
        end = min(len(pids_asc), start + n)
        start = max(0, end - n)
        return pids_asc[start:end]
    raise ValueError(f"Unknown mode '{mode}'. Choose from: worst, best, median.")


# ---------------------------------------------------------------------------
# Slice selection (pure logic -- unit-testable on synthetic volumes)
# ---------------------------------------------------------------------------

def select_slice_indices(
    gt: np.ndarray,
    pred: np.ndarray,
    n: int = 3,
    foreground_class_ids: Sequence[int] = FOREGROUND_CLASS_IDS,
) -> List[int]:
    """Pick up to `n` informative axial slice indices (last axis) from a case:
      1. the slice with the most GT foreground (representative),
      2. the slice with the most GT/pred disagreement (error-revealing),
      3. the middle slice of the GT foreground z-range (central prostate).
    Deduplicated, returned in ascending order. If the case has no GT
    foreground at all, falls back to the volume's middle slice."""
    if gt.shape != pred.shape:
        raise ValueError(f"gt shape {gt.shape} != pred shape {pred.shape}")

    depth = gt.shape[2]
    gt_fg = np.isin(gt, list(foreground_class_ids))
    pred_fg = np.isin(pred, list(foreground_class_ids))

    gt_fg_per_slice = gt_fg.reshape(-1, depth).sum(axis=0)
    disagree = (gt != pred) & (gt_fg | pred_fg)
    disagree_per_slice = disagree.reshape(-1, depth).sum(axis=0)

    picks: List[int] = []
    if gt_fg_per_slice.max() > 0:
        picks.append(int(np.argmax(gt_fg_per_slice)))
        fg_slices = np.where(gt_fg_per_slice > 0)[0]
        picks.append(int(fg_slices[len(fg_slices) // 2]))  # central foreground slice
    else:
        picks.append(depth // 2)

    if disagree_per_slice.max() > 0:
        picks.append(int(np.argmax(disagree_per_slice)))

    # Deduplicate preserving informativeness, then sort ascending, cap at n.
    seen = set()
    unique = []
    for p in picks:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return sorted(unique)[:max(1, n)]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _discrete_mask_rgba(mask2d: np.ndarray, class_colors: Dict[int, tuple]) -> np.ndarray:
    """Map a 2D integer label slice to an RGBA image with background transparent."""
    h, w = mask2d.shape
    rgba = np.zeros((h, w, 4), dtype=float)
    for class_id, color in class_colors.items():
        if class_id == 0:
            continue
        sel = mask2d == class_id
        rgba[sel, 0], rgba[sel, 1], rgba[sel, 2] = color
        rgba[sel, 3] = 1.0
    return rgba


def render_case_slices(
    t2: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    slice_indices: Sequence[int],
    patient_id: str,
    out_dir: str,
    subtitle: Optional[str] = None,
    class_labels: Optional[Dict[int, str]] = None,
) -> List[str]:
    """Render one figure per selected slice: T2 | GT | Pred | GT/Pred overlay |
    error map. Returns the list of written PNG paths. Import of matplotlib is
    local so the selection helpers stay importable without a display backend.

    subtitle / class_labels default to the Experiment-1 baseline caption
    ("Class 1 / Class 2", mapping UNVERIFIED, VALIDATION split). RQ2 passes the
    verified CG/PZ names and the correct split so its test-set figures are not
    mislabeled -- existing callers are unaffected (defaults preserve behavior)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    os.makedirs(out_dir, exist_ok=True)

    # Class 1 = orange, Class 2 = cyan (colorblind-safe, distinct on grayscale MRI).
    class_colors = {1: (0.95, 0.55, 0.10), 2: (0.10, 0.70, 0.90)}
    if subtitle is None:
        subtitle = "(Class1/Class2 anatomy mapping UNVERIFIED; VALIDATION split, not test)"
    labels = class_labels or {1: "Class 1", 2: "Class 2"}
    written: List[str] = []

    for k in slice_indices:
        t2_s = t2[:, :, k].astype(np.float32)
        gt_s = gt[:, :, k].astype(np.int64)
        pred_s = pred[:, :, k].astype(np.int64)

        # Robust intensity window for display only (does not affect any metric).
        lo, hi = np.percentile(t2_s, [1, 99]) if np.ptp(t2_s) > 0 else (0.0, 1.0)

        fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
        fig.suptitle(
            f"Patient {patient_id} -- axial slice {k}   {subtitle}",
            fontsize=12,
        )

        axes[0].imshow(t2_s.T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        axes[0].set_title("T2 MRI")

        axes[1].imshow(t2_s.T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        axes[1].imshow(np.transpose(_discrete_mask_rgba(gt_s, class_colors), (1, 0, 2)), origin="lower")
        axes[1].set_title("Ground Truth")

        axes[2].imshow(t2_s.T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        axes[2].imshow(np.transpose(_discrete_mask_rgba(pred_s, class_colors), (1, 0, 2)), origin="lower")
        axes[2].set_title("Prediction")

        # Overlay: GT filled (alpha) + prediction boundary contours for direct comparison.
        axes[3].imshow(t2_s.T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        gt_rgba = _discrete_mask_rgba(gt_s, class_colors)
        gt_rgba[..., 3] *= 0.45
        axes[3].imshow(np.transpose(gt_rgba, (1, 0, 2)), origin="lower")
        for class_id, color in class_colors.items():
            if np.any(pred_s == class_id):
                axes[3].contour((pred_s == class_id).T, levels=[0.5], colors=[color], linewidths=1.2, origin="lower")
        axes[3].set_title("GT (fill) vs Pred (outline)")

        # Error map: agreement vs FP vs FN on the foreground union.
        gt_fg = np.isin(gt_s, list(class_colors))
        pred_fg = np.isin(pred_s, list(class_colors))
        err = np.zeros((*gt_s.shape, 4), dtype=float)
        agree = gt_fg & pred_fg & (gt_s == pred_s)
        fp = pred_fg & ~gt_fg
        fn = gt_fg & ~pred_fg
        mismatch = gt_fg & pred_fg & (gt_s != pred_s)  # both foreground but wrong class
        err[agree] = (0.20, 0.80, 0.20, 0.9)     # green  = correct
        err[fp] = (0.95, 0.20, 0.20, 0.9)         # red    = false positive
        err[fn] = (0.20, 0.35, 0.95, 0.9)         # blue   = false negative (missed)
        err[mismatch] = (0.95, 0.85, 0.10, 0.95)  # yellow = wrong class
        axes[4].imshow(t2_s.T, cmap="gray", origin="lower", vmin=lo, vmax=hi)
        axes[4].imshow(np.transpose(err, (1, 0, 2)), origin="lower")
        axes[4].set_title("Error (G=correct R=FP B=miss Y=wrong-class)")

        for ax in axes:
            ax.axis("off")

        legend = [
            Patch(facecolor=class_colors[1], label=labels.get(1, "Class 1")),
            Patch(facecolor=class_colors[2], label=labels.get(2, "Class 2")),
        ]
        fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False)
        fig.tight_layout(rect=(0, 0.04, 1, 0.95))

        out_path = os.path.join(out_dir, f"patient_{patient_id}_slice_{k:03d}.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _resolve_split_patient_ids(cfg: dict, split: str) -> List[str]:
    from src.splits import load_official_split

    data_cfg = cfg.get("data", {})
    if "train_csv" in data_cfg and "valid_csv" in data_cfg:
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
    else:
        mapping = {
            "train": data_cfg.get("patient_ids", []),
            "val": cfg.get("validation", {}).get("val_patient_ids", []),
            "validation": cfg.get("validation", {}).get("val_patient_ids", []),
            "test": None,
        }
    if split not in mapping:
        raise ValueError(f"Unknown split '{split}'. Choose from {sorted(mapping)}.")
    ids = mapping[split]
    if split == "test" and not ids:
        raise ValueError(
            "Requested split='test' but the official 19-case Prostate158 test "
            "archive is not available locally. Use --split val in the meantime."
        )
    if not ids:
        raise ValueError(f"No case IDs resolved for split='{split}'.")
    return list(ids)


def run(
    config_path: str,
    checkpoint_path: str,
    output_dir: str,
    split: str = "val",
    patients: Optional[Sequence[str]] = None,
    mode: Optional[str] = None,
    n: int = 3,
    n_slices: int = 3,
    metrics_csv: Optional[str] = None,
    device: Optional[str] = None,
    class_ids: Sequence[int] = (0, 1, 2),
) -> dict:
    """Build the model+dataset from the checkpoint (same construction as
    scripts/run_evaluation.py), select cases, render qualitative panels, and
    write a visualization_summary.json. Inference only."""
    import torch

    from src.dataset import ProstateZonal2DDataset
    from src.evaluate import predict_case_original_space
    from src.model import build_model
    from src.transforms import PreprocessingConfig
    from src.utils import ensure_dir, get_device, load_config

    cfg = load_config(config_path)
    data_cfg = cfg.get("data", {})
    if device is None:
        device = get_device()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint '{checkpoint_path}' has no 'model_state_dict'.")
    ckpt_cfg = checkpoint.get("config", cfg)

    model = build_model(ckpt_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preproc = PreprocessingConfig.from_dict(
        ckpt_cfg.get("preprocessing", cfg.get("preprocessing", {}))
    )

    split_ids = _resolve_split_patient_ids(cfg, split)
    split_set = set(split_ids)

    # --- choose which patients to visualize ---
    if patients:
        requested = [normalize_patient_id(p) for p in patients]
        selected, invalid = [], []
        for pid in requested:
            (selected if pid in split_set else invalid).append(pid)
        if invalid:
            print(f"WARNING: ignoring patient id(s) not in split '{split}': {invalid}")
        if not selected:
            raise ValueError(
                f"None of the requested patients {requested} are in split '{split}'. "
                f"Available (first 25): {sorted(split_set)[:25]}"
            )
    elif mode:
        if not metrics_csv or not os.path.exists(metrics_csv):
            raise ValueError(
                f"--mode {mode} needs --metrics-csv pointing at an existing "
                f"eval_*_per_case_metrics.csv (got: {metrics_csv})."
            )
        selected = select_patients_by_mode(metrics_csv, mode, n)
        selected = [pid for pid in selected if pid in split_set]
        if not selected:
            raise ValueError(f"Mode '{mode}' selected no patients present in split '{split}'.")
    else:
        raise ValueError("Provide either --patients or --mode.")

    print("=" * 78)
    print("BASELINE QUALITATIVE VISUALIZATION (inference only -- no training)")
    print(f"  checkpoint : {checkpoint_path}  (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"  split      : {split}  ({len(split_ids)} cases)  "
          f"[VALIDATION is not the held-out test set]")
    print(f"  selected   : {selected}")
    print(f"  device     : {device}")
    print("=" * 78)

    out_root = ensure_dir(output_dir)

    # Optional: pull existing quantitative metrics for the summary (never recomputed here).
    metrics_lookup: Dict[str, dict] = {}
    if metrics_csv and os.path.exists(metrics_csv):
        import pandas as pd
        mdf = pd.read_csv(metrics_csv)
        mdf["patient_id"] = mdf["patient_id"].apply(normalize_patient_id)
        for pid, sub in mdf.groupby("patient_id"):
            metrics_lookup[pid] = {
                str(int(r["class_id"])): {
                    k: (None if (isinstance(r[k], float) and np.isnan(r[k])) else float(r[k]))
                    for k in ("dice", "iou", "hd95_mm", "asd_mm", "precision", "recall")
                    if k in sub.columns
                }
                for _, r in sub.iterrows()
            }

    summary = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": split,
        "is_official_held_out_test": split == "test",
        "label_mapping_status": "unverified (see results/label_mapping/label_mapping_report.md)",
        "class_display_labels": {str(c): CLASS_DISPLAY_LABELS.get(c, str(c)) for c in class_ids},
        "output_dir": out_root,
        "cases": [],
    }

    for pid in selected:
        dataset = ProstateZonal2DDataset(
            dataset_root=data_cfg["dataset_root"],
            patient_ids=[pid],
            image_name="t2.nii.gz",
            mask_name=data_cfg.get("target_mask", "t2_anatomy_reader1.nii.gz"),
            slice_sampling="all",  # full slice population, matches evaluation
            preprocessing_config=preproc,
            cache_data=True,
        )

        # Shared production inference + reconstruction path (original voxel space).
        pred, gt, geometry = predict_case_original_space(pid, model, dataset, device)

        # Original on-disk T2 lives in the SAME original voxel grid as gt/pred.
        import nibabel as nib
        t2_path = os.path.join(data_cfg["dataset_root"], pid, "t2.nii.gz")
        t2 = np.asarray(nib.load(t2_path).get_fdata())
        if t2.shape != gt.shape:
            raise ValueError(
                f"Geometry mismatch for patient {pid}: T2 {t2.shape} vs reconstructed GT {gt.shape}. "
                f"Refusing to render a spatially inconsistent figure."
            )

        slice_idx = select_slice_indices(gt, pred, n=n_slices)
        patient_dir = os.path.join(out_root, f"patient_{pid}")
        pngs = render_case_slices(t2, gt, pred, slice_idx, pid, patient_dir)

        summary["cases"].append({
            "patient_id": pid,
            "geometry_shape": list(geometry.shape),
            "geometry_spacing_mm": [float(z) for z in geometry.zooms],
            "geometry_orientation": list(geometry.axcodes),
            "selected_slice_indices": slice_idx,
            "existing_metrics": metrics_lookup.get(pid, {}),
            "figures": pngs,
        })
        print(f"  patient {pid}: slices {slice_idx} -> {len(pngs)} figure(s)")

    summary_dir = ensure_dir(os.path.join(out_root, "summary"))
    summary_path = os.path.join(summary_dir, "visualization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["summary_path"] = summary_path
    print(f"\nWrote summary: {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline qualitative visualization (inference only).")
    parser.add_argument("--config", default="configs/config_baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/exp01_baseline/visualizations",
        help="Where figures + summary are written (kept out of Git; see .gitignore).",
    )
    parser.add_argument("--split", default="val", choices=["val", "validation", "test", "train"])
    parser.add_argument("--patients", nargs="+", default=None, help="Explicit patient ids (e.g. 60 77 87).")
    parser.add_argument("--mode", default=None, choices=["worst", "best", "median"],
                        help="Auto-select from the metrics CSV instead of --patients.")
    parser.add_argument("--n", type=int, default=3, help="How many patients to auto-select.")
    parser.add_argument("--n-slices", type=int, default=3, help="Max slices to render per patient.")
    parser.add_argument("--metrics-csv", default=None,
                        help="eval_*_per_case_metrics.csv (required for --mode; used for summary metrics).")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.patients and not args.mode:
        parser.error("provide either --patients or --mode")

    run(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        patients=args.patients,
        mode=args.mode,
        n=args.n,
        n_slices=args.n_slices,
        metrics_csv=args.metrics_csv,
        device=args.device,
    )


if __name__ == "__main__":
    main()
