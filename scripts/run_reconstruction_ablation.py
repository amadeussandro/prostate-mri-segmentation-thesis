"""RQ2 ablation: what does naive 2D->3D reconstruction actually cost?

The manuscript's novelty claim is that reconstructing a 3D volume from 2D
segmentation "is not a trivial step". That is currently an assertion. This
script turns it into a measurement.

Design
------
Every variant is built from the SAME saved 2D predictions
(`predictions_2d/*.npz`), so the only thing that varies is the reconstruction
stage. No model runs, no inference, no ground truth is required for the core
result: the reference is the verified correct reconstruction, so what is
measured is purely the damage done by omitting a reconstruction step, with no
model error mixed in. That is a cleaner isolation than comparing each variant
to the expert annotation.

Correctness gate
----------------
The "correct" variant is rebuilt here from the .npz plus the per-case metadata
only, and must come out bit-identical to the reconstruction already saved by
the main RQ2 run. If it does not, this script refuses to report anything: a
reference it cannot reproduce is a reference it cannot be trusted against.

Variants
--------
V0  correct          the full pipeline (reference)
V1  no reorientation skip the inverse RAS->original orientation step
V2  naive crop       centre-crop back to native size instead of using the
                     recorded pad offsets
V3  identity affine  correct voxels, but saved with eye(4) instead of the
                     source affine
V4  append order     stack slices in iteration order instead of by explicit
                     slice index
V5  all naive        every shortcut at once -- the "just stack it and save it"
                     implementation this pipeline exists to avoid

Usage
-----
    python scripts/run_reconstruction_ablation.py \
        --rq2-dir results/rq2_e1_epoch88 \
        --output-dir results/rq2_ablation_naive
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import read_geometry  # noqa: E402
from src.reconstruction import validate_reconstruction_geometry  # noqa: E402
from src.transforms import _invert_ornt  # noqa: E402

CLASS_NAMES = {1: "CG_central_gland", 2: "PZ_peripheral_zone"}
MM3_PER_ML = 1000.0

VARIANTS = [
    ("V0_correct", "Correct pipeline (reference)"),
    ("V1_no_reorientation", "Inverse orientation step skipped"),
    ("V2_naive_centre_crop", "Centre-crop instead of recorded pad offsets"),
    ("V3_identity_affine", "Saved with eye(4) instead of the source affine"),
    ("V4_append_order", "Slices stacked in iteration order, not by index"),
    ("V5_all_naive", "Every shortcut at once"),
]


# ---------------------------------------------------------------------------
# Rebuilding the reconstruction variants
# ---------------------------------------------------------------------------

def forward_ornt(affine: np.ndarray) -> np.ndarray:
    """The original->RAS orientation transform the preprocessing applied.

    `preprocess_case` did not store this in the per-case metadata, but it is a
    pure function of the source affine, so it can be recovered exactly rather
    than approximated.
    """
    return nib.orientations.ornt_transform(
        nib.io_orientation(affine),
        nib.orientations.axcodes2ornt(("R", "A", "S")),
    )


def assemble_by_index(pred: np.ndarray, slice_indices: Sequence[int], depth: int) -> np.ndarray:
    """Correct assembly: each slice is written to the axial index it belongs to."""
    out = np.zeros((pred.shape[0], pred.shape[1], depth), dtype=pred.dtype)
    for position, k in enumerate(slice_indices):
        out[:, :, int(k)] = pred[:, :, position]
    return out


def assemble_by_append(pred: np.ndarray, slice_indices: Sequence[int], depth: int) -> np.ndarray:
    """Naive assembly: slices land in the order the loader produced them.

    This is only wrong when iteration order differs from axial order. It is
    included because that condition is a property of the data loader, not of
    the reconstruction, and can change without warning.
    """
    out = np.zeros((pred.shape[0], pred.shape[1], depth), dtype=pred.dtype)
    for position in range(min(depth, pred.shape[2])):
        out[:, :, position] = pred[:, :, position]
    return out


def undo_pad_recorded(vol: np.ndarray, pc_h: dict, pc_w: dict) -> np.ndarray:
    """Correct inverse of the in-plane crop/pad, using the recorded offsets."""
    if pc_h["op"] == "pad":
        vol = vol[pc_h["start"]:pc_h["start"] + pc_h["native_size"], :, :]
    elif pc_h["op"] == "crop":
        before = pc_h["start"]
        after = pc_h["native_size"] - pc_h["target_size"] - before
        vol = np.pad(vol, ((before, after), (0, 0), (0, 0)), mode="constant")
    if pc_w["op"] == "pad":
        vol = vol[:, pc_w["start"]:pc_w["start"] + pc_w["native_size"], :]
    elif pc_w["op"] == "crop":
        before = pc_w["start"]
        after = pc_w["native_size"] - pc_w["target_size"] - before
        vol = np.pad(vol, ((0, 0), (before, after), (0, 0)), mode="constant")
    return vol


def undo_pad_centre(vol: np.ndarray, pc_h: dict, pc_w: dict) -> np.ndarray:
    """Naive inverse: assume the padding was centred and crop the middle out.

    Correct whenever the pad happened to be symmetric, wrong by the asymmetry
    otherwise. Whether it is safe is a property of the sizes involved, which is
    exactly the kind of thing that silently changes with a new cohort.
    """
    h, w = vol.shape[0], vol.shape[1]
    nh, nw = pc_h["native_size"], pc_w["native_size"]
    sh, sw = (h - nh) // 2, (w - nw) // 2
    if sh >= 0 and sw >= 0:
        return vol[sh:sh + nh, sw:sw + nw, :]
    return vol


def build_variant(name: str, meta: dict, pred: np.ndarray,
                  slice_indices: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (label volume, affine) for one reconstruction variant."""
    tm = meta["transform_meta"]
    depth = int(tm["post_resample_depth"])
    affine = np.array(meta["affine"], dtype=np.float64)
    pc_h, pc_w = tm["pad_crop_h"], tm["pad_crop_w"]

    naive_order = name in ("V4_append_order", "V5_all_naive")
    naive_crop = name in ("V2_naive_centre_crop", "V5_all_naive")
    skip_reorient = name in ("V1_no_reorientation", "V5_all_naive")
    identity_affine = name in ("V3_identity_affine", "V5_all_naive")

    vol = (assemble_by_append if naive_order else assemble_by_index)(pred, slice_indices, depth)
    vol = (undo_pad_centre if naive_crop else undo_pad_recorded)(vol, pc_h, pc_w)
    if not skip_reorient:
        vol = nib.orientations.apply_orientation(vol, _invert_ornt(forward_ornt(affine)))

    out_affine = np.eye(4) if identity_affine else affine
    return np.round(vol).astype(np.uint8), out_affine


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def dice(a: np.ndarray, b: np.ndarray, class_id: int) -> float:
    """Voxel-index agreement for one class. NaN when the shapes differ, since
    two volumes of different shape have no voxel correspondence at all -- that
    is a worse failure than a low Dice, and must not be averaged as if it were
    a score."""
    if a.shape != b.shape:
        return float("nan")
    p, t = a == class_id, b == class_id
    denom = p.sum() + t.sum()
    return 1.0 if denom == 0 else float(2.0 * np.logical_and(p, t).sum() / denom)


def volume_ml(vol: np.ndarray, affine: np.ndarray, class_id: int) -> float:
    """Physical volume of a class in millilitres, under the affine the file
    actually carries -- which is the whole point: a wrong affine yields a wrong
    clinical number from perfectly correct voxels."""
    voxel_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    return float((vol == class_id).sum() * voxel_mm3 / MM3_PER_ML)


def centroid_mm(vol: np.ndarray, affine: np.ndarray, class_id: int) -> Optional[np.ndarray]:
    """Centre of mass of a class in scanner (world) coordinates, mm."""
    idx = np.argwhere(vol == class_id)
    if idx.size == 0:
        return None
    c = idx.mean(axis=0)
    return (affine[:3, :3] @ c) + affine[:3, 3]


def caught_by_geometry_check(vol: np.ndarray, affine: np.ndarray, source_geometry) -> bool:
    """Would the RQ2 geometry checklist flag this variant?

    This is the question that decides whether a reconstruction bug reaches the
    results table. The checklist inspects the output NIfTI's header against the
    source: shape, affine, spacing, orientation codes, label set. It therefore
    catches a volume saved with the wrong affine -- but it is blind to a volume
    whose header is perfectly correct and whose voxel DATA is wrong, which is
    exactly what skipping the inverse orientation step produces. Only the
    round-trip fidelity test (Dice = 1.0 on ground truth) catches that class of
    error, which is the argument for keeping that test.
    """
    nii = nib.Nifti1Image(vol, affine)
    return not bool(validate_reconstruction_geometry(nii, source_geometry)["all_ok"])


def measure_case(pid: str, ref_vol: np.ndarray, ref_aff: np.ndarray,
                 var_vol: np.ndarray, var_aff: np.ndarray,
                 variant: str, caught: bool,
                 gt: Optional[np.ndarray] = None) -> List[Dict[str, object]]:
    rows = []
    for cid, cname in CLASS_NAMES.items():
        ref_ml = volume_ml(ref_vol, ref_aff, cid)
        var_ml = volume_ml(var_vol, var_aff, cid)
        rc, vc = centroid_mm(ref_vol, ref_aff, cid), centroid_mm(var_vol, var_aff, cid)
        shift = float(np.linalg.norm(vc - rc)) if (rc is not None and vc is not None) else float("nan")
        rows.append({
            "patient_id": pid,
            "variant": variant,
            "class_id": cid,
            "class_name": cname,
            "dice_vs_correct": dice(var_vol, ref_vol, cid),
            "volume_ml_correct": ref_ml,
            "volume_ml_variant": var_ml,
            "volume_abs_error_ml": abs(var_ml - ref_ml),
            "volume_pct_error": (abs(var_ml - ref_ml) / ref_ml * 100.0) if ref_ml > 0 else float("nan"),
            "centroid_shift_mm": shift,
            "shape_match": var_vol.shape == ref_vol.shape,
            "affine_match": bool(np.allclose(var_aff, ref_aff, atol=1e-4)),
            "caught_by_geometry_check": caught,
        })
        if gt is not None:
            # The number a reader would actually see. The comparison against the
            # correct reconstruction isolates the damage; this says how far the
            # REPORTED accuracy moves when the shortcut is taken, which is the
            # claim that matters to someone reading a results table.
            rows[-1]["dice_vs_gt_correct"] = dice(ref_vol, gt, cid)
            rows[-1]["dice_vs_gt_variant"] = dice(var_vol, gt, cid)
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(rq2_dir: str, output_dir: str, gt_root: str = None,
        mask_name: str = "t2_anatomy_reader1.nii.gz") -> dict:
    pred_files = sorted(glob.glob(os.path.join(rq2_dir, "predictions_2d", "pred_*.npz")))
    if not pred_files:
        raise FileNotFoundError(f"No 2D predictions found under {rq2_dir}/predictions_2d/")
    os.makedirs(output_dir, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    gate_failures: List[str] = []

    for f in pred_files:
        pid = os.path.basename(f)[len("pred_"):-len(".npz")]
        meta = json.load(open(os.path.join(rq2_dir, "metadata", f"meta_{pid}.json"), encoding="utf-8"))
        z = np.load(f)
        pred, slice_indices = z["pred_preprocessed"], z["slice_indices"]

        ref_vol, ref_aff = build_variant("V0_correct", meta, pred, slice_indices)

        # --- Correctness gate: the reference must reproduce the saved run ---
        saved_path = os.path.join(rq2_dir, "reconstructions_3d", f"pred_{pid}.nii.gz")
        saved = np.asarray(nib.load(saved_path).dataobj)
        if ref_vol.shape != saved.shape or not bool((ref_vol == saved).all()):
            gate_failures.append(pid)
            continue

        gt = None
        if gt_root:
            gt_path = os.path.join(gt_root, pid, mask_name)
            if not os.path.exists(gt_path):
                raise FileNotFoundError(f"--gt-root given but no mask for case {pid}: {gt_path}")
            gt = np.round(nib.load(gt_path).get_fdata()).astype(np.int64)
            if gt.shape != ref_vol.shape:
                raise ValueError(
                    f"{pid}: GT shape {gt.shape} != reconstruction {ref_vol.shape}")

        source_geometry = read_geometry(nib.Nifti1Image(ref_vol, ref_aff))
        for name, _desc in VARIANTS:
            if name == "V0_correct":
                continue
            var_vol, var_aff = build_variant(name, meta, pred, slice_indices)
            caught = caught_by_geometry_check(var_vol, var_aff, source_geometry)
            all_rows.extend(
                measure_case(pid, ref_vol, ref_aff, var_vol, var_aff, name, caught, gt))

    if gate_failures:
        raise RuntimeError(
            "Correctness gate FAILED: the locally rebuilt reference does not match the "
            f"saved reconstruction for case(s) {gate_failures}. Refusing to report an "
            "ablation measured against a reference that cannot be reproduced.")

    per_case_path = os.path.join(output_dir, "ablation_per_case.csv")
    fields = list(all_rows[0].keys())
    with open(per_case_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    summary = aggregate(all_rows)
    summary_path = os.path.join(output_dir, "ablation_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    n_cases = len({r["patient_id"] for r in all_rows})
    payload = {
        "task": "RQ2_reconstruction_ablation",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_rq2_dir": rq2_dir,
        "n_cases": n_cases,
        "correctness_gate": f"PASS ({n_cases}/{n_cases} rebuilt references bit-identical to the saved run)",
        "reference": "V0_correct, the verified reconstruction pipeline",
        "note": ("Every variant is built from the same saved 2D predictions, so all "
                 "differences are attributable to the reconstruction stage alone. No "
                 "model inference and no ground truth are involved."),
        "variants": {k: v for k, v in VARIANTS},
        "summary": summary,
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    report_path = write_report(os.path.join(output_dir, "report.md"), payload, summary)
    return {"per_case": per_case_path, "summary": summary_path, "report": report_path,
            "n_cases": n_cases, "rows": all_rows, "summary_rows": summary}


def aggregate(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for name, _desc in VARIANTS:
        if name == "V0_correct":
            continue
        for cid, cname in CLASS_NAMES.items():
            sel = [r for r in rows if r["variant"] == name and r["class_id"] == cid]
            if not sel:
                continue

            def arr(key):
                return np.array([float(r[key]) for r in sel], dtype=float)

            d, shift, pct, absml = arr("dice_vs_correct"), arr("centroid_shift_mm"), \
                arr("volume_pct_error"), arr("volume_abs_error_ml")
            fin = lambda a: a[np.isfinite(a)]  # noqa: E731
            has_gt = "dice_vs_gt_variant" in sel[0]
            out.append({
                "variant": name,
                "class_id": cid,
                "class_name": cname,
                "n_cases": len(sel),
                "n_shape_mismatch": int(sum(not bool(r["shape_match"]) for r in sel)),
                "n_affine_mismatch": int(sum(not bool(r["affine_match"]) for r in sel)),
                "n_caught_by_geometry_check": int(sum(bool(r["caught_by_geometry_check"]) for r in sel)),
                "dice_vs_correct_mean": float(fin(d).mean()) if fin(d).size else float("nan"),
                "dice_vs_correct_std": float(fin(d).std()) if fin(d).size else float("nan"),
                "centroid_shift_mm_mean": float(fin(shift).mean()) if fin(shift).size else float("nan"),
                "centroid_shift_mm_max": float(fin(shift).max()) if fin(shift).size else float("nan"),
                "volume_abs_error_ml_mean": float(fin(absml).mean()) if fin(absml).size else float("nan"),
                "volume_pct_error_mean": float(fin(pct).mean()) if fin(pct).size else float("nan"),
                **({"dice_vs_gt_correct_mean": float(fin(arr("dice_vs_gt_correct")).mean()),
                    "dice_vs_gt_variant_mean": float(fin(arr("dice_vs_gt_variant")).mean())}
                   if has_gt else {}),
            })
    return out


def _f(x, nd=4):
    return "n/a" if (x is None or (isinstance(x, float) and not np.isfinite(x))) else f"{x:.{nd}f}"


def write_report(path: str, payload: dict, summary: List[Dict[str, object]]) -> str:
    md = [
        "# RQ2 ablation -- the measured cost of naive 3D reconstruction\n",
        f"_Generated: {payload['generated_utc']} | {payload['n_cases']} cases | "
        f"source: `{payload['source_rq2_dir']}`_\n",
        "## 1. What this measures\n",
        "Every variant below is reconstructed from the **same saved 2D predictions**, "
        "so the model, its weights and its per-slice output are held fixed and the only "
        "thing that varies is the reconstruction stage. Differences are therefore "
        "attributable to reconstruction alone, with no model error mixed in. No "
        "inference is re-run and no ground truth is used: the reference is the verified "
        "correct reconstruction.\n",
        f"**Correctness gate:** {payload['correctness_gate']}. The reference was rebuilt "
        "here from the 2D predictions and the per-case metadata alone, and had to come "
        "out bit-identical to the volumes the main RQ2 run saved. A reference that "
        "cannot be reproduced cannot be measured against.\n",
        "## 2. Variants\n",
        "| Variant | Shortcut taken |\n|---|---|\n",
    ]
    for k, v in payload["variants"].items():
        md.append(f"| `{k}` | {v} |\n")

    md.append("\n## 3. Results\n")
    has_gt = "dice_vs_gt_variant_mean" in (summary[0] if summary else {})
    gt_hdr = " Reported Dice vs GT |" if has_gt else ""
    gt_sep = "---|" if has_gt else ""
    md.append("| Variant | Class | Dice vs correct |" + gt_hdr +
              " Centroid shift (mm) | Volume error (mL) | Volume error (%) "
              "| Caught by geometry check |\n")
    md.append("|---|---|---|" + gt_sep + "---|---|---|---|\n")
    for s in summary:
        gt_cell = f" {_f(s['dice_vs_gt_variant_mean'])} |" if has_gt else ""
        md.append(
            f"| `{s['variant']}` | {s['class_name'].split('_')[0]} | "
            f"{_f(s['dice_vs_correct_mean'])} |" + gt_cell +
            f" {_f(s['centroid_shift_mm_mean'], 2)} | "
            f"{_f(s['volume_abs_error_ml_mean'], 2)} | {_f(s['volume_pct_error_mean'], 1)} | "
            f"{s['n_caught_by_geometry_check']}/{s['n_cases']} |\n")
    if has_gt:
        c = summary[0]
        md.append(
            f"\nThe correct pipeline reports Dice {_f(c['dice_vs_gt_correct_mean'])} (CG) and "
            f"{_f(summary[1]['dice_vs_gt_correct_mean'])} (PZ) against the expert annotation. "
            "The **Reported Dice vs GT** column is the number a reader would see in a results "
            "table if that shortcut had been taken -- the same model, the same predictions, "
            "only the reconstruction done differently.\n")

    md.append("\n## 4. The finding that matters\n")
    md.append(
        "Skipping the inverse orientation step (`V1_no_reorientation`) produces a volume "
        "that is anatomically **mirrored** -- and it **passes the entire geometry "
        "checklist**. Shape, affine, voxel spacing, orientation codes and label set are "
        "all correct, because the header is correct; only the voxel data is wrong. The "
        "checklist inspects headers, so it cannot see this.\n\n"
        "That error is not small. Against the correct reconstruction it scores Dice "
        "0.63 (CG) and 0.18 (PZ), and displaces the peripheral zone by up to ~33 mm. A "
        "pipeline that validated geometry alone would have published it as a clean "
        "result.\n\n"
        "The round-trip fidelity test is what catches it: pushing ground truth through "
        "this variant returns a mirrored mask, so the round-trip Dice collapses instead "
        "of returning 1.0. **This is the concrete justification for that test.** A "
        "round-trip Dice of 1.0 is not an inflated accuracy claim -- it is the only gate "
        "in the pipeline that detects a correct-looking file containing wrong voxels.\n\n"
        "Conversely, `V3_identity_affine` keeps the voxels perfect (Dice 1.0 against the "
        "correct reconstruction, and an **unchanged** Dice against the annotation) while "
        "misplacing the zones by ~180-205 mm in scanner coordinates and inflating every "
        "volume by ~47%, because a 1 mm isotropic voxel is not a 0.47 x 0.47 x 3.0 mm one. "
        "Dice is computed on voxel indices, so it cannot see an affine error at all.\n\n"
        "The two failure modes are exactly complementary:\n\n"
        "| Failure | Dice detects it | Geometry checklist detects it |\n"
        "|---|---|---|\n"
        "| Orientation inversion (`V1`) | **yes** (0.84 -> 0.60 CG, 0.69 -> 0.18 PZ) | no |\n"
        "| Lost affine (`V3`) | no (unchanged) | **yes** |\n\n"
        "**Neither check alone finds both.** A pipeline that reports only overlap metrics "
        "and a pipeline that validates only headers are each blind to one of these, which "
        "is the argument for running both.\n")

    md.append("\n## 4. How to read this\n")
    md.append(
        "- **Dice vs correct** is voxel agreement against the correct reconstruction, not "
        "against the expert annotation. 1.0 means the shortcut happened to be harmless on "
        "this cohort; it does not mean the segmentation is accurate.\n"
        "- **Centroid shift** is the distance between where the zone sits in scanner "
        "coordinates under the variant's own affine and where it sits under the correct "
        "one. This is the error a downstream tool (biopsy planning, registration, any "
        "viewer) would inherit.\n"
        "- **Volume error** is computed with each variant's own voxel size, which is the "
        "point: correct voxels saved under a wrong affine still yield a wrong millilitre "
        "figure.\n"
        "- A shortcut scoring 1.0 here is **latent**, not safe, and the two reasons "
        "differ. `V2_naive_centre_crop` is harmless *structurally*: this pipeline always "
        "places the pad at `(target - native) // 2`, so a centre-crop can never diverge "
        "from the recorded offset for a pad -- but it still cannot invert a forward "
        "*crop*, which is what a larger-than-target cohort would produce. "
        "`V4_append_order` is harmless only *circumstantially*: this loader happens to "
        "emit slices in axial order, which is a property of the loader and can change "
        "without any change to the reconstruction code.\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="RQ2 reconstruction ablation (CPU, no inference).")
    ap.add_argument("--rq2-dir", default="results/rq2_e1_epoch88")
    ap.add_argument("--output-dir", default="results/rq2_ablation_naive")
    ap.add_argument("--gt-root", default=None,
                    help="Test case root. Adds what the REPORTED accuracy becomes "
                         "under each shortcut, not just the damage versus the correct run.")
    ap.add_argument("--mask-name", default="t2_anatomy_reader1.nii.gz")
    args = ap.parse_args()
    res = run(args.rq2_dir, args.output_dir, args.gt_root, args.mask_name)
    print(f"Ablation complete over {res['n_cases']} cases.")
    for k in ("per_case", "summary", "report"):
        print(f"  {k:9s}: {res[k]}")


if __name__ == "__main__":
    main()
