"""Regenerate the RQ2 qualitative figures WITHOUT re-running inference.

Why this exists
---------------
The 3D projection figures saved by the original RQ2 run were drawn with square
voxels, which compresses the coronal and sagittal panels by the in-plane to
through-plane spacing ratio (~6.4x on Prostate158) and makes a geometrically
valid reconstruction look like a flat disc. The renderer was fixed; the figures
on disk still predate it.

Re-running the whole RQ2 pipeline to redraw three pictures would mean loading
the checkpoint and re-running inference on every slice of every case, for
output that is already saved. The predictions live in
`reconstructions_3d/pred_<id>.nii.gz` in original voxel space -- exactly what
the renderers consume. So this script needs no model, no checkpoint, no GPU and
no torch: it reads the saved predictions, the T2 and the annotation, and
redraws.

It deliberately recomputes the representative cases from `metrics_per_case.csv`
under the same documented rule the report uses (rank by mean foreground Dice:
highest = good, middle = median, lowest = challenging) rather than taking them
on faith, so the figures cannot drift away from the table they illustrate.

Usage (Colab, after mounting Drive)
-----------------------------------
    python scripts/regenerate_rq2_figures.py \
        --rq2-dir  <DRIVE>/results/rq2_e1_epoch88 \
        --gt-root  <extracted 19-case test root>
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Sequence

import nibabel as nib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from run_rq2_reconstruction import render_3d_projections  # noqa: E402
from visualize_baseline_cases import render_case_slices, select_slice_indices  # noqa: E402

SUBTITLE = "(CG=orange, PZ=cyan; original voxel space; official TEST split)"
CLASS_LABELS = {1: "CG (Central Gland)", 2: "PZ (Peripheral Zone)"}


def representative_cases(metrics_csv: str) -> Dict[str, str]:
    """Reproduce the report's representative-case rule from the per-case metrics.

    Ranks by mean foreground Dice (CG and PZ only, background excluded) and
    returns the highest, middle and lowest. Ties break on patient id, matching
    the deterministic selection used elsewhere.
    """
    rows = list(csv.DictReader(open(metrics_csv, encoding="utf-8")))
    per_case: Dict[str, List[float]] = {}
    for r in rows:
        if int(r["class_id"]) in (1, 2):
            per_case.setdefault(r["patient_id"], []).append(float(r["dice"]))
    if not per_case:
        raise ValueError(f"No foreground rows found in {metrics_csv}")
    ranked = sorted(per_case.items(), key=lambda kv: (sum(kv[1]) / len(kv[1]), kv[0]))
    return {
        "challenging": ranked[0][0],
        "median": ranked[len(ranked) // 2][0],
        "good": ranked[-1][0],
    }


def regenerate_case(pid: str, rq2_dir: str, gt_root: str, mask_name: str,
                    n_slices: int) -> Dict[str, object]:
    pred_path = os.path.join(rq2_dir, "reconstructions_3d", f"pred_{pid}.nii.gz")
    t2_path = os.path.join(gt_root, pid, "t2.nii.gz")
    gt_path = os.path.join(gt_root, pid, mask_name)
    for p in (pred_path, t2_path, gt_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing input for case {pid}: {p}")

    pred_nii = nib.load(pred_path)
    pred = np.asarray(pred_nii.dataobj)
    t2 = np.asarray(nib.load(t2_path).get_fdata())
    gt = np.round(nib.load(gt_path).get_fdata()).astype(np.int64)

    # The three volumes must share one grid, or the overlays are meaningless.
    if not (t2.shape == gt.shape == pred.shape):
        raise ValueError(
            f"Geometry mismatch for case {pid}: t2 {t2.shape}, gt {gt.shape}, "
            f"pred {pred.shape}. Refusing to draw an overlay on mismatched grids.")

    patient_dir = os.path.join(rq2_dir, "visualizations", f"patient_{pid}")
    os.makedirs(patient_dir, exist_ok=True)

    slice_idx = select_slice_indices(gt, pred, n=n_slices)
    axial = render_case_slices(t2, gt, pred, slice_idx, pid, patient_dir,
                               subtitle=SUBTITLE, class_labels=CLASS_LABELS)
    zooms = nib.load(pred_path).header.get_zooms()[:3]
    proj = render_3d_projections(
        t2, gt, pred, pid,
        os.path.join(patient_dir, f"patient_{pid}_3d_projections.png"),
        spacing=zooms)
    return {"patient_id": pid, "axial_figures": axial, "projection_figure": proj,
            "spacing_mm": [round(float(z), 6) for z in zooms]}


def run(rq2_dir: str, gt_root: str, mask_name: str = "t2_anatomy_reader1.nii.gz",
        cases: Sequence[str] = None, n_slices: int = 3) -> dict:
    if cases:
        reps = {f"case_{c}": c for c in cases}
    else:
        reps = representative_cases(os.path.join(rq2_dir, "metrics_per_case.csv"))

    written = []
    for label, pid in reps.items():
        info = regenerate_case(pid, rq2_dir, gt_root, mask_name, n_slices)
        info["label"] = label
        written.append(info)
        print(f"  {label:12s} patient {pid}: {len(info['axial_figures'])} axial + "
              f"1 projection  (spacing {info['spacing_mm']})")
    return {"representatives": reps, "written": written}


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Redraw the RQ2 qualitative figures from saved predictions (no inference).")
    ap.add_argument("--rq2-dir", default="results/rq2_e1_epoch88")
    ap.add_argument("--gt-root", required=True,
                    help="Test case root holding <id>/t2.nii.gz and <id>/<mask-name>.")
    ap.add_argument("--mask-name", default="t2_anatomy_reader1.nii.gz")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Explicit case ids. Default: recompute the report's representatives.")
    ap.add_argument("--n-slices", type=int, default=3)
    args = ap.parse_args()

    print("Regenerating RQ2 figures from saved predictions -- no model, no inference.")
    res = run(args.rq2_dir, args.gt_root, args.mask_name, args.cases, args.n_slices)
    print(f"\nDone. Representatives: {res['representatives']}")


if __name__ == "__main__":
    main()
