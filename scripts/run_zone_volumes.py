"""RQ2 task C: anatomical zone volumes in millilitres.

Why this exists
---------------
Dice answers "do the voxels overlap"; it does not answer "how big is the
gland". The clinical quantities downstream of zonal segmentation are volumes:
whole-gland volume is the denominator of **PSA density** (serum PSA / gland
volume, the threshold commonly cited around 0.15 ng/mL/cc), and the CG:PZ split
is what targeted-biopsy planning is drawn on. Reporting millilitres is what
turns a segmentation result into a clinical one, and it is the standing
supervisor request from 2026-09-22 for a medical justification of RQ2.

It also makes the geometry work legible. A volume in mL is the product of a
voxel count and the affine determinant, so a pipeline that loses the affine
reports a wrong millilitre figure from perfectly correct voxels -- exactly the
failure the reconstruction ablation measures at ~47%.

Two modes
---------
Predictions only (runs anywhere the reconstructions are):

    python scripts/run_zone_volumes.py --recon-dir results/rq2_e1_epoch88/reconstructions_3d

With ground truth (needs the dataset, i.e. Colab/Drive) adds per-case error,
bias and limits of agreement:

    python scripts/run_zone_volumes.py \
        --recon-dir  <DRIVE>/results/rq2_e1_epoch88/reconstructions_3d \
        --gt-root    <extracted 19-case test root> \
        --output-dir results/rq2_zone_volumes
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
from typing import Dict, List, Optional

import nibabel as nib
import numpy as np

CLASS_NAMES = {1: "CG_central_gland", 2: "PZ_peripheral_zone"}
MM3_PER_ML = 1000.0


def volume_ml(vol: np.ndarray, affine: np.ndarray, class_id: int) -> float:
    """Physical volume of a class, in mL.

    The voxel volume is |det(affine[:3,:3])| rather than the product of the
    header zooms, so that a sheared or rotated acquisition is handled correctly
    instead of being silently approximated by its axis-aligned spacing.
    """
    voxel_mm3 = float(abs(np.linalg.det(np.asarray(affine)[:3, :3])))
    return float((vol == class_id).sum() * voxel_mm3 / MM3_PER_ML)


def zone_volumes(vol: np.ndarray, affine: np.ndarray) -> Dict[str, float]:
    cg = volume_ml(vol, affine, 1)
    pz = volume_ml(vol, affine, 2)
    return {
        "cg_ml": cg,
        "pz_ml": pz,
        "whole_gland_ml": cg + pz,
        "pz_fraction": (pz / (cg + pz)) if (cg + pz) > 0 else float("nan"),
    }


def _stats(values: List[float]) -> Dict[str, float]:
    a = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {k: float("nan") for k in ("mean", "std", "min", "median", "max")}
    return {"mean": float(a.mean()), "std": float(a.std()),  # ddof=0, repo convention
            "min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def run(recon_dir: str, output_dir: str, gt_root: str = None,
        mask_name: str = "t2_anatomy_reader1.nii.gz") -> dict:
    files = sorted(glob.glob(os.path.join(recon_dir, "pred_*.nii.gz")))
    if not files:
        raise FileNotFoundError(f"No reconstructions found in {recon_dir}")
    os.makedirs(output_dir, exist_ok=True)

    rows: List[Dict[str, object]] = []
    missing_gt: List[str] = []

    for f in files:
        pid = os.path.basename(f)[len("pred_"):-len(".nii.gz")]
        nii = nib.load(f)
        pred = np.asarray(nii.dataobj)
        row: Dict[str, object] = {"patient_id": pid}
        pv = zone_volumes(pred, nii.affine)
        row.update({f"pred_{k}": v for k, v in pv.items()})

        if gt_root:
            gt_path = os.path.join(gt_root, pid, mask_name)
            if not os.path.exists(gt_path):
                missing_gt.append(pid)
            else:
                gt_nii = nib.load(gt_path)
                gt = np.round(gt_nii.get_fdata()).astype(np.int64)
                if gt.shape != pred.shape:
                    raise ValueError(
                        f"GT shape {gt.shape} != prediction shape {pred.shape} for case {pid}")
                gv = zone_volumes(gt, gt_nii.affine)
                row.update({f"gt_{k}": v for k, v in gv.items()})
                for key in ("cg_ml", "pz_ml", "whole_gland_ml"):
                    err = pv[key] - gv[key]
                    row[f"error_{key}"] = err
                    row[f"abs_error_{key}"] = abs(err)
                    row[f"pct_error_{key}"] = (err / gv[key] * 100.0) if gv[key] > 0 else float("nan")
        rows.append(row)

    if missing_gt:
        raise FileNotFoundError(
            f"--gt-root was given but the mask is missing for case(s) {missing_gt}. "
            "Refusing to report a partial cohort.")

    per_case = os.path.join(output_dir, "zone_volumes_per_case.csv")
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "patient_id", k))
    with open(per_case, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    metrics = [k for k in fields if k != "patient_id"]
    summary = {m: _stats([float(r[m]) for r in rows if m in r]) for m in metrics}

    # Bland-Altman style agreement, only meaningful when GT was supplied.
    agreement = {}
    if gt_root:
        for key in ("cg_ml", "pz_ml", "whole_gland_ml"):
            d = np.array([float(r[f"error_{key}"]) for r in rows], dtype=float)
            bias, sd = float(d.mean()), float(d.std(ddof=1))
            agreement[key] = {
                "bias_ml": bias, "sd_ml": sd,
                "loa_lower_ml": bias - 1.96 * sd, "loa_upper_ml": bias + 1.96 * sd,
                "note": "Bland-Altman bias and 95% limits of agreement (SD uses ddof=1, "
                        "the standard convention for limits of agreement).",
            }

    payload = {
        "task": "RQ2_zone_volumes_ml",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "recon_dir": recon_dir,
        "gt_root": gt_root,
        "mode": "prediction_vs_ground_truth" if gt_root else "prediction_only",
        "n_cases": len(rows),
        "clinical_note": (
            "Whole-gland volume is the denominator of PSA density (serum PSA / gland "
            "volume). Volumes are computed as voxel count x |det(affine)|, so they are "
            "only valid because the reconstruction restores the source affine exactly; "
            "see the reconstruction ablation for what the same voxels yield without it."),
        "summary": summary,
        "agreement": agreement,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    summary_csv = os.path.join(output_dir, "zone_volumes_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "mean", "std", "min", "median", "max"])
        for m, s in summary.items():
            w.writerow([m, s["mean"], s["std"], s["min"], s["median"], s["max"]])

    report = write_report(os.path.join(output_dir, "report.md"), payload)
    return {"per_case": per_case, "summary": summary_csv, "report": report,
            "n_cases": len(rows), "payload": payload}


def _f(x, nd=2):
    return "n/a" if (x is None or not np.isfinite(x)) else f"{x:.{nd}f}"


def write_report(path: str, payload: dict) -> str:
    s = payload["summary"]
    md = [
        "# RQ2 -- anatomical zone volumes (mL)\n",
        f"_Generated: {payload['generated_utc']} | {payload['n_cases']} cases | "
        f"mode: **{payload['mode']}**_\n",
        "## 1. Why volumes\n",
        payload["clinical_note"] + "\n",
        "## 2. Predicted volumes\n",
        "| Quantity | Mean | SD | Min | Median | Max |\n|---|---|---|---|---|---|\n",
    ]
    labels = [("pred_cg_ml", "CG (mL)"), ("pred_pz_ml", "PZ (mL)"),
              ("pred_whole_gland_ml", "Whole gland (mL)"), ("pred_pz_fraction", "PZ fraction")]
    for key, label in labels:
        if key in s:
            st = s[key]
            nd = 3 if "fraction" in key else 2
            md.append(f"| {label} | {_f(st['mean'], nd)} | {_f(st['std'], nd)} | "
                      f"{_f(st['min'], nd)} | {_f(st['median'], nd)} | {_f(st['max'], nd)} |\n")

    if payload["mode"] == "prediction_only":
        md.append(
            "\n> **Prediction-only mode.** These are the volumes the model produces. "
            "Accuracy against the expert annotation is NOT established here -- rerun "
            "with `--gt-root` pointing at the test cases to add per-case error, bias and "
            "limits of agreement. Do not present these as validated volumes until then.\n")
    else:
        md.append("\n## 3. Agreement with ground truth\n")
        md.append("| Quantity | Bias (mL) | SD (mL) | 95% limits of agreement (mL) |\n")
        md.append("|---|---|---|---|\n")
        for key, label in [("cg_ml", "CG"), ("pz_ml", "PZ"), ("whole_gland_ml", "Whole gland")]:
            a = payload["agreement"][key]
            md.append(f"| {label} | {_f(a['bias_ml'])} | {_f(a['sd_ml'])} | "
                      f"{_f(a['loa_lower_ml'])} to {_f(a['loa_upper_ml'])} |\n")
        md.append(
            "\nBias is mean(prediction - ground truth): positive means the model "
            "over-estimates. The limits of agreement, not the bias, are what determine "
            "whether a volume is usable for an individual patient -- a near-zero bias "
            "with wide limits means errors cancel across the cohort while remaining "
            "large case by case.\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="RQ2 zone volumes in mL (CPU, no inference).")
    ap.add_argument("--recon-dir", default="results/rq2_e1_epoch88/reconstructions_3d")
    ap.add_argument("--gt-root", default=None,
                    help="Test case root holding <id>/<mask_name>. Omit for prediction-only mode.")
    ap.add_argument("--mask-name", default="t2_anatomy_reader1.nii.gz")
    ap.add_argument("--output-dir", default="results/rq2_zone_volumes")
    args = ap.parse_args()
    res = run(args.recon_dir, args.output_dir, args.gt_root, args.mask_name)
    print(f"Zone volumes computed for {res['n_cases']} cases "
          f"({res['payload']['mode']}).")
    for k in ("per_case", "summary", "report"):
        print(f"  {k:9s}: {res[k]}")


if __name__ == "__main__":
    main()
