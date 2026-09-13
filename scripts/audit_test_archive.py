"""Audit the official Prostate158 held-out TEST archive (READ-ONLY, no evaluation).

This inspects the test ZIP / extracted test directory and reports everything
needed to decide whether the final Experiment-1 test evaluation can run:
  - number of test cases and their ids,
  - per-case presence of t2.nii.gz and t2_anatomy_reader1.nii.gz,
  - NIfTI shape / spacing / orientation / unique label values (where GT exists),
  - overlap against the 119 train + 20 validation cases (leakage check),
  - whether src.splits.load_official_split resolves exactly 19 test ids from
    the extracted directory (loader compatibility).

It does NOT: load any model, run inference, compute segmentation metrics,
train, or touch the Experiment-1 checkpoint. The original ZIP is never
modified; extraction (if requested) writes only into --extract-dir.

Typical Colab usage:

    python scripts/audit_test_archive.py \
        --zip /content/drive/MyDrive/THESIS_PROSTATE158/dataset/test/prostate158_test.zip \
        --extract-dir /content/drive/MyDrive/THESIS_PROSTATE158/dataset/test/extracted \
        --train-csv dataset/prostate158_train/train.csv \
        --valid-csv dataset/prostate158_train/valid.csv

(If the archive is already extracted, pass only --extract-dir and skip --zip.)
"""

import argparse
import json
import os
import sys
import zipfile
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import nibabel as nib
import numpy as np

T2_NAME = "t2.nii.gz"
GT_NAME = "t2_anatomy_reader1.nii.gz"


def _normalize_id(raw: str, zero_pad: int = 3) -> str:
    s = str(raw).strip()
    return s.zfill(zero_pad) if s.isdigit() else s


def extract_if_needed(zip_path: Optional[str], extract_dir: str) -> None:
    """Extract the ZIP into extract_dir if that dir doesn't already have content.
    The ZIP itself is opened read-only and never modified."""
    if zip_path is None:
        if not os.path.isdir(extract_dir):
            raise FileNotFoundError(f"--extract-dir not found and no --zip given: {extract_dir}")
        return
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"ZIP not found: {zip_path}")
    os.makedirs(extract_dir, exist_ok=True)
    if any(os.scandir(extract_dir)):
        print(f"[extract] {extract_dir} already non-empty; skipping extraction (ZIP untouched).")
        return
    print(f"[extract] extracting {zip_path} -> {extract_dir} (ZIP left untouched)")
    with zipfile.ZipFile(zip_path, "r") as zf:  # read-only
        zf.extractall(extract_dir)


def find_case_root(root: str, max_depth: int = 4) -> str:
    """Find the directory that DIRECTLY contains per-case subfolders (a case
    subfolder is one containing t2.nii.gz). Descends through wrapper folders
    (common in Zenodo archives) up to max_depth."""
    current = root
    for _ in range(max_depth):
        subdirs = [d for d in sorted(os.listdir(current))
                   if os.path.isdir(os.path.join(current, d))]
        # If any immediate subdir looks like a case (has a t2), this is the root.
        if any(os.path.exists(os.path.join(current, d, T2_NAME)) for d in subdirs):
            return current
        # Otherwise, if there's exactly one wrapper subdir, descend into it.
        if len(subdirs) == 1:
            current = os.path.join(current, subdirs[0])
            continue
        break
    return current  # best effort; caller reports what it found


def inspect_case(case_dir: str) -> Dict:
    info: Dict = {"has_t2": False, "has_gt": False}
    t2_path = os.path.join(case_dir, T2_NAME)
    gt_path = os.path.join(case_dir, GT_NAME)
    info["has_t2"] = os.path.exists(t2_path)
    info["has_gt"] = os.path.exists(gt_path)

    if info["has_t2"]:
        img = nib.load(t2_path)
        info["t2_shape"] = tuple(int(s) for s in img.shape[:3])
        info["t2_spacing"] = tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])
        info["t2_orientation"] = "".join(nib.aff2axcodes(img.affine))

    if info["has_gt"]:
        gt = nib.load(gt_path)
        arr = np.round(gt.get_fdata()).astype(np.int64)
        info["gt_shape"] = tuple(int(s) for s in gt.shape[:3])
        info["gt_orientation"] = "".join(nib.aff2axcodes(gt.affine))
        info["gt_unique_labels"] = sorted(int(v) for v in np.unique(arr))
        # image/mask geometry agreement (shape + affine), if both present
        if info["has_t2"]:
            img = nib.load(t2_path)
            info["gt_matches_t2_geometry"] = (
                img.shape[:3] == gt.shape[:3]
                and bool(np.allclose(img.affine, gt.affine, atol=1e-4))
            )
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the official Prostate158 test archive (read-only).")
    parser.add_argument("--zip", default=None, help="Path to prostate158_test.zip (optional if already extracted).")
    parser.add_argument("--extract-dir", required=True, help="Isolated extraction directory (ZIP left untouched).")
    parser.add_argument("--train-csv", default="dataset/prostate158_train/train.csv")
    parser.add_argument("--valid-csv", default="dataset/prostate158_train/valid.csv")
    parser.add_argument("--audit-json", default=None,
                        help="Where to write the audit summary JSON (default: <extract-dir>/test_archive_audit.json).")
    args = parser.parse_args()

    extract_if_needed(args.zip, args.extract_dir)
    case_root = find_case_root(args.extract_dir)
    print(f"[case-root] {case_root}")

    case_ids = sorted(d for d in os.listdir(case_root)
                      if os.path.isdir(os.path.join(case_root, d)))
    print(f"[cases] found {len(case_ids)} case directories: {case_ids}")

    per_case = {cid: inspect_case(os.path.join(case_root, cid)) for cid in case_ids}

    n_t2 = sum(1 for c in per_case.values() if c["has_t2"])
    n_gt = sum(1 for c in per_case.values() if c["has_gt"])
    if n_gt == len(case_ids) and len(case_ids) > 0:
        gt_status = "A: t2_anatomy_reader1 present for ALL cases"
    elif n_gt > 0:
        gt_status = f"B: t2_anatomy_reader1 present for SOME cases ({n_gt}/{len(case_ids)})"
    else:
        gt_status = "C: t2_anatomy_reader1 ABSENT for all cases"

    # Overlap with train/validation (leakage check).
    overlap: Dict[str, List[str]] = {"train": [], "val": []}
    loader_ok = None
    loader_msg = ""
    try:
        import pandas as pd
        train_ids = {_normalize_id(v) for v in pd.read_csv(args.train_csv)["ID"]}
        val_ids = {_normalize_id(v) for v in pd.read_csv(args.valid_csv)["ID"]}
        norm_test = {_normalize_id(c) for c in case_ids}
        overlap["train"] = sorted(norm_test & train_ids)
        overlap["val"] = sorted(norm_test & val_ids)

        # Loader compatibility: does load_official_split resolve 19 test ids?
        from src.splits import load_official_split
        split = load_official_split(args.train_csv, args.valid_csv, test_dir=case_root)
        loader_ok = (split.test_ids is not None and len(split.test_ids) == 19)
        loader_msg = f"load_official_split resolved {len(split.test_ids or [])} test ids (no leakage)."
    except Exception as e:  # pragma: no cover - reported, not raised
        loader_ok = False
        loader_msg = f"load_official_split / overlap check failed: {type(e).__name__}: {e}"

    # Geometry summary (over cases that have a T2).
    shapes = sorted({c.get("t2_shape") for c in per_case.values() if c.get("t2_shape")}, key=str)
    orients = sorted({c.get("t2_orientation") for c in per_case.values() if c.get("t2_orientation")})
    labels_seen = sorted({tuple(c["gt_unique_labels"]) for c in per_case.values() if c.get("gt_unique_labels")}, key=str)

    print("\n================ TEST ARCHIVE AUDIT ================")
    print(f"case-root              : {case_root}")
    print(f"num cases              : {len(case_ids)} (expected 19)")
    print(f"cases with t2.nii.gz   : {n_t2}/{len(case_ids)}")
    print(f"cases with GT          : {n_gt}/{len(case_ids)}  -> {gt_status}")
    print(f"unique t2 shapes       : {shapes}")
    print(f"unique t2 orientations : {orients}")
    print(f"unique GT label sets   : {labels_seen if labels_seen else 'n/a (no GT)'}")
    print(f"overlap with train     : {overlap['train'] or 'NONE'}")
    print(f"overlap with val       : {overlap['val'] or 'NONE'}")
    print(f"loader compatible      : {loader_ok} -- {loader_msg}")
    print("===================================================")

    audit = {
        "case_root": case_root,
        "num_cases": len(case_ids),
        "case_ids": case_ids,
        "num_with_t2": n_t2,
        "num_with_gt": n_gt,
        "gt_status": gt_status,
        "unique_t2_shapes": [list(s) for s in shapes],
        "unique_t2_orientations": orients,
        "unique_gt_label_sets": [list(l) for l in labels_seen],
        "overlap_with_train": overlap["train"],
        "overlap_with_val": overlap["val"],
        "loader_resolves_19_test_ids": loader_ok,
        "loader_message": loader_msg,
        "per_case": per_case,
        "note": "READ-ONLY audit. No model, no evaluation, no training, no metrics.",
    }
    audit_json = args.audit_json or os.path.join(args.extract_dir, "test_archive_audit.json")
    with open(audit_json, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\nWrote audit summary: {audit_json}")

    ready = (len(case_ids) == 19 and n_t2 == 19 and n_gt == 19
             and not overlap["train"] and not overlap["val"] and loader_ok)
    print(f"\nFINAL TEST EVALUATION READY: {ready}")
    if not ready:
        print("  (If GT is absent/partial, Dice/IoU/HD95/ASD cannot be computed for those cases.)")


if __name__ == "__main__":
    main()
