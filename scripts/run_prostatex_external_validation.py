"""Task D -- external validation of the frozen E1 model on PROSTATEx.

Inference only. No training, no fine-tuning, no threshold search, no model
selection. PROSTATEx is an INDEPENDENT cohort: different scanner population,
different annotators, different acquisition geometry. It is touched exactly
once, with the model already frozen by Prostate158 validation.

Expect lower numbers than the Prostate158 test set. That is a finding about
generalization, not a failure, and it is what JMIR asks for when it requires
independent dataset validation. The framing is fixed here, before the numbers
exist, so it cannot be rationalized afterwards.

What this script is careful about
---------------------------------
1. **Mask harmonization.** PROSTATEx ships PZ and "rest" as two separate binary
   masks; the model's convention is 0=background, 1=CG, 2=PZ. The combination
   is built here (never stored in `raw/`) with `mask > 0`, not `mask == 1`:
   case ProstateX-0086 carries a single stray voxel of value 2 in its PZ mask,
   and an equality test would silently drop it. Disjointness is re-measured per
   case rather than taken from the manifest.

2. **Non-canonical identifiers.** 15 source files use non-canonical stems
   (13 three-digit, 2 with a `VOLUME-` prefix). Masks are resolved through the
   manifest's recorded paths, never by reconstructing a filename.

3. **In-plane cropping.** Prostate158 is 232x232 and the model's crop_pad target
   is 442, so Prostate158 cases are always PADDED. PROSTATEx reaches 640x640,
   which the same transform will CROP -- and a crop can cut anatomy off. Every
   case is checked for foreground lost to the crop, and any loss is reported
   per case instead of being averaged away.

4. **Scale shift is reported, not hidden.** The pipeline has no in-plane
   resampling: crop_pad preserves physical scale, so a 0.3 mm/px acquisition
   presents the prostate at ~1.6x the pixel size the model trained on
   (Prostate158 is ~0.47 mm/px). This is a genuine covariate shift and is
   recorded per case so the results can be read against it.

Usage (Colab, after mounting Drive)
-----------------------------------
    python scripts/run_prostatex_external_validation.py \
        --prostatex-root <DRIVE>/dataset/external/prostatex \
        --checkpoint     <DRIVE>/results/exp_e1_dice_ce/best_model.pt \
        --output-dir     results/external_prostatex_e1
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EXPECTED_E1_SHA256 = "2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2"
EXPECTED_CASES = 204
CLASS_IDS = (0, 1, 2)
CLASS_NAMES = {0: "background", 1: "CG_central_gland", 2: "PZ_peripheral_zone"}
PROSTATE158_INPLANE_MM = 0.46875  # the scale the model was trained at


# ---------------------------------------------------------------------------
# Phase 1 -- harmonize PROSTATEx into the layout the shared dataset expects
# ---------------------------------------------------------------------------

def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(prostatex_root: str) -> List[dict]:
    path = os.path.join(prostatex_root, "metadata", "prostatex_manifest.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Manifest not found at {path}. It is REQUIRED: 15 cases have non-canonical "
            "mask filenames that cannot be derived from the case id, so masks must be "
            "resolved through the manifest's recorded paths.")
    return list(csv.DictReader(open(path, encoding="utf-8")))


def harmonize_case(row: dict, prostatex_root: str, out_root: str) -> dict:
    """Write <out_root>/<case_id>/{t2.nii.gz, anatomy.nii.gz} for one case.

    anatomy.nii.gz is the model's 0/1/2 convention, built as
    `rest > 0 -> 1`, `pz > 0 -> 2`. The two source masks are mutually exclusive
    (re-measured here, not assumed), so no precedence rule is needed.
    """
    cid = row["case_id"]
    case_dir_existing = os.path.join(out_root, cid)
    done_marker = os.path.join(case_dir_existing, "harmonized.json")
    if os.path.exists(done_marker) and \
            os.path.exists(os.path.join(case_dir_existing, "anatomy.nii.gz")) and \
            os.path.exists(os.path.join(case_dir_existing, "t2.nii.gz")):
        # Already built and recorded by an earlier run. Harmonization is a pure
        # function of the source masks, so rebuilding it would reproduce the
        # same bytes at the cost of re-reading every mask.
        return json.load(open(done_marker, encoding="utf-8"))

    t2_path = os.path.join(prostatex_root, row["t2_path"])
    pz_path = os.path.join(prostatex_root, row["pz_mask_path"])
    rest_path = os.path.join(prostatex_root, row["rest_mask_path"])
    for p in (t2_path, pz_path, rest_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{cid}: missing source file {p}")

    t2_nii = nib.load(t2_path)
    pz = np.asarray(nib.load(pz_path).dataobj)
    rest = np.asarray(nib.load(rest_path).dataobj)

    if not (t2_nii.shape == pz.shape == rest.shape):
        raise ValueError(
            f"{cid}: geometry mismatch -- t2 {t2_nii.shape}, pz {pz.shape}, rest {rest.shape}")

    pz_bin, rest_bin = pz > 0, rest > 0
    overlap = int(np.logical_and(pz_bin, rest_bin).sum())
    if overlap:
        raise ValueError(
            f"{cid}: PZ and rest overlap in {overlap} voxels. The harmonization assumes "
            "mutual exclusivity; a precedence rule would have to be chosen and justified.")

    combined = np.zeros(t2_nii.shape, dtype=np.uint8)
    combined[rest_bin] = 1
    combined[pz_bin] = 2

    case_dir = os.path.join(out_root, cid)
    os.makedirs(case_dir, exist_ok=True)
    nib.save(nib.Nifti1Image(combined, t2_nii.affine, t2_nii.header),
             os.path.join(case_dir, "anatomy.nii.gz"))

    # Link the T2 rather than copying ~4 MB x 204; fall back to a copy where
    # symlinks are unavailable (Windows without privilege).
    link = os.path.join(case_dir, "t2.nii.gz")
    if not os.path.exists(link):
        try:
            os.symlink(os.path.abspath(t2_path), link)
        except (OSError, NotImplementedError, AttributeError):
            import shutil
            shutil.copy2(t2_path, link)

    zooms = [float(z) for z in t2_nii.header.get_zooms()[:3]]
    record = {
        "case_id": cid,
        "shape": tuple(int(s) for s in t2_nii.shape),
        "spacing_mm": zooms,
        "inplane_mm": zooms[0],
        "scale_vs_prostate158": PROSTATE158_INPLANE_MM / zooms[0] if zooms[0] else float("nan"),
        "orientation": "".join(nib.orientations.aff2axcodes(t2_nii.affine)),
        "cg_voxels": int((combined == 1).sum()),
        "pz_voxels": int((combined == 2).sum()),
        "pz_rest_overlap": overlap,
        "stray_label_values_in_pz": sorted(int(v) for v in np.unique(pz) if v not in (0, 1)),
    }
    with open(done_marker, "w", encoding="utf-8") as fh:
        json.dump(record, fh, default=list)
    return record


def prepare(prostatex_root: str, work_dir: str, expected_cases: int) -> List[dict]:
    rows = read_manifest(prostatex_root)
    usable = [r for r in rows if r.get("paired_ok", "True") == "True"]
    if len(usable) != expected_cases:
        raise ValueError(
            f"Manifest lists {len(usable)} usable cases, expected {expected_cases}. "
            "Refusing to run external validation on an incomplete cohort -- a partial "
            "upload is the likely cause; re-check the Drive transfer.")

    cases_root = os.path.join(work_dir, "cases")
    os.makedirs(cases_root, exist_ok=True)
    records = [harmonize_case(r, prostatex_root, cases_root) for r in usable]
    print(f"  harmonized {len(records)} cases -> {cases_root}")
    return records


# ---------------------------------------------------------------------------
# Phase 2/3 -- inference and evaluation, on the SAME shared path as RQ1/RQ2
# ---------------------------------------------------------------------------

def crop_loss_report(gt_original: np.ndarray, transform_meta) -> Dict[str, int]:
    """Foreground voxels lost to the in-plane crop, per class.

    Prostate158 cases are always PADDED by this transform, so the question never
    arises there. PROSTATEx reaches 640x640 against a 442 target, where the same
    transform CROPS -- and a crop can cut anatomy off before the model ever sees
    it. Reported per case rather than averaged: one case losing the gland edge
    is a different problem from every case losing a little.

    Recomputed from the recorded forward transform rather than read out of the
    dataset's private cache, so it stays correct if the caching changes.
    """
    pc_h, pc_w = transform_meta.pad_crop_h, transform_meta.pad_crop_w
    oriented = nib.orientations.apply_orientation(gt_original, transform_meta.orientation_fwd)

    cropped = oriented
    if pc_h.op == "crop":
        cropped = cropped[pc_h.start:pc_h.start + pc_h.target_size, :, :]
    if pc_w.op == "crop":
        cropped = cropped[:, pc_w.start:pc_w.start + pc_w.target_size, :]

    out = {}
    for cid in (1, 2):
        before = int((oriented == cid).sum())
        after = int((cropped == cid).sum())
        out[f"class{cid}_before"] = before
        out[f"class{cid}_after"] = after
        out[f"class{cid}_lost"] = max(0, before - after)
    return out


def run(prostatex_root: str, checkpoint_path: str, output_dir: str,
        work_dir: str = None, expected_cases: int = EXPECTED_CASES,
        device: str = None, skip_sha_check: bool = False) -> dict:
    import torch

    from src.dataset import ProstateZonal2DDataset
    from src.evaluate import predict_case_reconstructed
    from src.metrics import (
        CaseMetrics,
        aggregate_case_results,
        evaluate_case_volume,
        write_case_metrics_csv,
        write_summary_csv,
    )
    from src.model import build_model
    from src.reconstruction import validate_reconstruction_geometry
    from src.transforms import PreprocessingConfig
    from src.utils import get_device

    os.makedirs(output_dir, exist_ok=True)
    work_dir = work_dir or os.path.join(output_dir, "_harmonized")

    # --- Gate 1: checkpoint identity. A filename proves nothing; the hash does.
    digest = sha256_of(checkpoint_path)
    print(f"checkpoint : {checkpoint_path}\nSHA-256    : {digest}")
    if not skip_sha_check and digest != EXPECTED_E1_SHA256:
        raise RuntimeError(
            "SHA-256 MISMATCH -- this is not the frozen E1 checkpoint.\n"
            f"  expected {EXPECTED_E1_SHA256}\n  got      {digest}\n"
            "External validation must use the same weights that produced the RQ1 result.")

    # --- Gate 2: cohort completeness + harmonization
    print("\nHarmonizing PROSTATEx (PZ + rest -> 0/1/2) ...")
    prep = prepare(prostatex_root, work_dir, expected_cases)
    cases_root = os.path.join(work_dir, "cases")
    case_ids = [r["case_id"] for r in prep]

    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_cfg = checkpoint.get("config", {})
    model = build_model(ckpt_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    # --- Gate 3: architecture
    epoch = checkpoint.get("epoch", "unknown")
    exp_name = ckpt_cfg.get("experiment", {}).get("name")
    print(f"experiment : {exp_name}  epoch {epoch}")
    if int(getattr(model, "out_channels", -1)) != 3:
        raise RuntimeError("Expected a 3-class model (0=bg, 1=CG, 2=PZ).")

    preproc = PreprocessingConfig.from_dict(ckpt_cfg.get("preprocessing", {}))
    print(f"preprocessing rebuilt from the checkpoint: target_size={preproc.target_size}, "
          f"z_resample={preproc.z_resample}, mode={preproc.spatial_mode}")

    # Saved per case so a long run is resumable and so the predicted volumes
    # survive for later analysis (zone volumes, connected-component diagnosis)
    # without paying for inference again.
    recon_dir = os.path.join(output_dir, "reconstructions_3d")
    cache_dir = os.path.join(output_dir, "_case_metrics")
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    case_results, geometry_rows = [], []
    for i, cid in enumerate(case_ids, 1):
        cache_path = os.path.join(cache_dir, f"{cid}.json")
        recon_path = os.path.join(recon_dir, f"pred_{cid}.nii.gz")

        # Resume: a case whose metrics AND volume both survived an earlier run
        # is replayed from disk. Inference is deterministic (eval mode, no TTA,
        # fixed weights), so a replayed case is identical to a recomputed one.
        if os.path.exists(cache_path) and os.path.exists(recon_path):
            cached = json.load(open(cache_path, encoding="utf-8"))
            case_results.append(CaseMetrics(
                patient_id=cid,
                per_class={int(k): v for k, v in cached["per_class"].items()}))
            geometry_rows.append(cached["geometry_row"])
            if i % 25 == 0 or i == len(case_ids):
                print(f"  {i}/{len(case_ids)} cases (resumed)")
            continue

        dataset = ProstateZonal2DDataset(
            dataset_root=cases_root, patient_ids=[cid],
            image_name="t2.nii.gz", mask_name="anatomy.nii.gz",
            slice_sampling="all", preprocessing_config=preproc, cache_data=True,
        )
        cp = predict_case_reconstructed(cid, model, dataset, device)

        labels = sorted(int(v) for v in np.unique(cp.pred_original))
        if not set(labels).issubset(set(CLASS_IDS)):
            raise RuntimeError(f"{cid}: prediction carries labels outside {CLASS_IDS}: {labels}")

        # Keep the volume: it is what makes zone volumes and any post-hoc error
        # analysis possible without re-running inference over the whole cohort.
        nib.save(cp.pred_nii, recon_path)

        cm = evaluate_case_volume(pred=cp.pred_original, gt=cp.gt_original,
                                  spacing=cp.geometry.zooms, class_ids=CLASS_IDS,
                                  patient_id=cid)
        case_results.append(cm)

        rec = next(r for r in prep if r["case_id"] == cid)
        loss = crop_loss_report(cp.gt_original, cp.transform_meta)
        # Geometry validation on a cohort whose orientation, spacing and
        # crop/pad branch all differ from Prostate158 is stronger evidence for
        # the reconstruction's generality than 19 homogeneous cases.
        val = validate_reconstruction_geometry(cp.pred_nii, cp.geometry, valid_labels=CLASS_IDS)
        row = {
            "case_id": cid, "shape": str(rec["shape"]),
            "inplane_mm": rec["inplane_mm"], "through_plane_mm": rec["spacing_mm"][2],
            "scale_vs_prostate158": rec["scale_vs_prostate158"],
            "orientation": rec["orientation"],
            "inplane_exceeds_target": max(rec["shape"][0], rec["shape"][1]) > preproc.target_size[0],
            "geometry_all_ok": val["all_ok"],
            "affine_max_abs_diff": val["affine_max_abs_diff"],
            **loss,
        }
        geometry_rows.append(row)

        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump({"per_class": cm.per_class, "geometry_row": row}, fh, default=float)

        if i % 25 == 0 or i == len(case_ids):
            print(f"  {i}/{len(case_ids)} cases", flush=True)

    summary = aggregate_case_results(case_results, class_ids=CLASS_IDS)
    per_case = write_case_metrics_csv(os.path.join(output_dir, "metrics_per_case.csv"),
                                      case_results, CLASS_IDS, CLASS_NAMES)
    summary_csv = write_summary_csv(os.path.join(output_dir, "metrics_summary.csv"),
                                    summary, CLASS_IDS, CLASS_NAMES)

    geom_csv = os.path.join(output_dir, "cohort_geometry.csv")
    with open(geom_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(geometry_rows[0].keys()))
        w.writeheader()
        w.writerows(geometry_rows)

    payload = {
        "task": "external_validation_prostatex",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "cohort": "PROSTATEx (Cuocolo et al. annotations)",
        "n_cases": len(case_results),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": digest,
        "checkpoint_epoch": epoch,
        "experiment": exp_name,
        "usage": "inference only -- no training, tuning, threshold or model selection",
        "label_harmonization": "rest > 0 -> 1 (CG), pz > 0 -> 2 (PZ); disjointness re-measured per case",
        "framing": ("PROSTATEx is an independent cohort with different scanners, annotators "
                    "and geometry. Lower scores than the Prostate158 test set are a finding "
                    "about generalization, not a failure."),
        "summary": {str(c): {"class_name": CLASS_NAMES[c], "metrics": summary[c]} for c in CLASS_IDS},
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    report = write_report(os.path.join(output_dir, "report.md"), payload, geometry_rows)
    return {"per_case": per_case, "summary": summary_csv, "geometry": geom_csv,
            "report": report, "n_cases": len(case_results), "payload": payload}


def _ms(summary: dict, c: int, m: str) -> str:
    s = summary[str(c)]["metrics"].get(m)
    if not s or not np.isfinite(s.get("mean", float("nan"))):
        return "n/a"
    return f"{s['mean']:.4f} +/- {s['std']:.4f}"


def write_report(path: str, payload: dict, geometry_rows: List[dict]) -> str:
    s = payload["summary"]
    cropped = [g for g in geometry_rows if g.get("inplane_exceeds_target")]
    lost = [g for g in geometry_rows if g.get("class1_lost", 0) or g.get("class2_lost", 0)]
    scales = [float(g["scale_vs_prostate158"]) for g in geometry_rows]

    md = [
        "# External validation on PROSTATEx -- frozen E1 model\n",
        f"_Generated: {payload['generated_utc']} | {payload['n_cases']} cases | "
        f"{payload['experiment']} epoch {payload['checkpoint_epoch']}_\n",
        "## 1. Status of this cohort\n",
        payload["usage"].capitalize() + ". " + payload["framing"] + "\n",
        f"Checkpoint SHA-256 `{payload['checkpoint_sha256']}` -- verified identical to the "
        "model that produced the RQ1 held-out test result.\n",
        "## 2. Results\n",
        "| Metric | CG | PZ |\n|---|---|---|\n",
    ]
    for m, label in [("dice", "Dice"), ("iou", "IoU"), ("hd95_mm", "HD95 (mm)"),
                     ("asd_mm", "ASD (mm)"), ("precision", "Precision"), ("recall", "Recall")]:
        md.append(f"| {label} | {_ms(s, 1, m)} | {_ms(s, 2, m)} |\n")
    md.append(f"\nBackground reported separately: {_ms(s, 0, 'dice')} Dice. Macro Dice is the "
              "mean of CG and PZ only.\n")

    md.append("\n## 3. Domain shift, measured\n")
    md.append(
        f"- In-plane scale relative to Prostate158 (~{PROSTATE158_INPLANE_MM} mm/px): "
        f"{min(scales):.2f}x to {max(scales):.2f}x. The pipeline does no in-plane "
        "resampling, so the prostate is presented to the model at a different pixel size "
        "than it was trained on. This is a genuine covariate shift and part of what any "
        "score drop measures.\n"
        f"- Cases whose in-plane size exceeds the crop_pad target: **{len(cropped)}**. "
        "Prostate158 cases are always padded by this transform; these are cropped instead.\n"
        f"- Cases losing annotated foreground to that crop: **{len(lost)}**. "
        + ("Any such case has had anatomy cut off before the model ever saw it, and its "
           "score is a measure of the crop as much as of the model -- see "
           "`cohort_geometry.csv` for the per-case counts.\n"
           if lost else "No case lost annotated foreground to cropping.\n"))

    md.append("\n## 4. How to read this\n")
    md.append(
        "These numbers are not comparable to the Prostate158 test result as a like-for-like "
        "contrast: the annotators, scanners, acquisition geometry and zone definitions all "
        "differ. They answer a different question -- does a model trained on one cohort "
        "retain useful zonal segmentation on an independent one -- and the honest unit of "
        "that answer is the gap, reported alongside the shift in Section 3 that partly "
        "explains it.\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="PROSTATEx external validation (inference only).")
    ap.add_argument("--prostatex-root", default="dataset/external/prostatex")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", default="results/external_prostatex_e1")
    ap.add_argument("--work-dir", default=None,
                    help="Where the harmonized 0/1/2 view is built. Default: <output-dir>/_harmonized")
    ap.add_argument("--expected-cases", type=int, default=EXPECTED_CASES)
    ap.add_argument("--device", default=None)
    ap.add_argument("--prepare-only", action="store_true",
                    help="Harmonize and verify the cohort, then stop. No model needed.")
    ap.add_argument("--skip-sha-check", action="store_true",
                    help="Escape hatch; never use for a reported run.")
    args = ap.parse_args()

    if args.prepare_only:
        work = args.work_dir or os.path.join(args.output_dir, "_harmonized")
        recs = prepare(args.prostatex_root, work, args.expected_cases)
        print(f"Prepared {len(recs)} cases. No inference performed.")
        return

    res = run(args.prostatex_root, args.checkpoint, args.output_dir,
              args.work_dir, args.expected_cases, args.device, args.skip_sha_check)
    print(f"\nExternal validation complete over {res['n_cases']} cases.")
    for k in ("per_case", "summary", "geometry", "report"):
        print(f"  {k:9s}: {res[k]}")


if __name__ == "__main__":
    main()
