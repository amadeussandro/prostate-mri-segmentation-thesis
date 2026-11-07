"""Frozen-E1 inference with the pre-specified LCC rule (results/lcc_prespecification.md).

Phase `valid`: Prostate158 validation split (20 cases). Applies R1/R2 and
               evaluates the pre-specified safety check. Writes decision.json.
Phase `nci`  : NCI-ISBI 2013 (confirmatory external cohort). Unmodified pipeline
               is the primary outcome; R1/R2 are reported only if they passed
               the validation check.

Inference only. CPU is enough. Reconstructions are cached per case, so a run
can be resumed.

Usage:
    python scripts/run_nci_external_validation.py --phase valid
    python scripts/run_nci_external_validation.py --phase nci
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import shutil
import sys

import nibabel as nib
import numpy as np
from scipy import ndimage, stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.metrics import dice_score, hausdorff_distance_95  # noqa: E402

CKPT = os.path.join(ROOT, "results/exp_e1_dice_ce/best_model.pt")
E1_SHA = "2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2"
PRESPEC = os.path.join(ROOT, "results/lcc_prespecification.md")
P158 = os.path.join(ROOT, "dataset/prostate158_train")
NCI = os.path.join(ROOT, "dataset/external/nci_isbi_2013/nifti")
OUT = os.path.join(ROOT, "results/external_nci_isbi_e1")
CG, PZ = 1, 2
DICE_TOL = 0.005
EXCLUDE = {"Prostate3T-01-0019"}  # whole-gland-only label (prespec Amendment 1)


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def largest_cc(mask: np.ndarray) -> np.ndarray:
    lab, n = ndimage.label(mask)  # 6-connectivity, as pre-specified
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def apply_rule(pred: np.ndarray, rule: str) -> np.ndarray:
    out = pred.copy()
    if rule in ("R1", "R2"):
        cg = pred == CG
        out[cg & ~largest_cc(cg)] = 0
    if rule == "R2":
        pz = out == PZ
        out[pz & ~largest_cc(pz)] = 0
    return out


def case_metrics(pred, gt, spacing, cid, extra=None) -> dict:
    row = {"case_id": cid, **(extra or {})}
    for rule in ("none", "R1", "R2"):
        p = apply_rule(pred, rule)
        for cls, name in ((CG, "cg"), (PZ, "pz")):
            if not (gt == cls).any():
                row[f"{name}_dice_{rule}"] = row[f"{name}_hd95_{rule}"] = float("nan")
                continue
            row[f"{name}_dice_{rule}"] = float(dice_score(p, gt, cls))
            row[f"{name}_hd95_{rule}"] = float(hausdorff_distance_95(p, gt, spacing, cls))
    for cls, name in ((CG, "cg"), (PZ, "pz")):
        row[f"{name}_n_components"] = int(ndimage.label(pred == cls)[1])
    return row


def load_model(device="cpu"):
    import torch
    from src.model import build_model
    from src.transforms import PreprocessingConfig
    digest = sha256(CKPT)
    if digest != E1_SHA:
        raise RuntimeError(f"checkpoint SHA mismatch: {digest}")
    torch.set_num_threads(os.cpu_count() or 4)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    model = build_model(cfg).to(device)
    model.load_state_dict(ck["model_state_dict"], strict=True)
    model.eval()
    return model, PreprocessingConfig.from_dict(cfg.get("preprocessing", {})), ck.get("epoch")


def infer(cases_root, ids, mask_name, out_dir, extra_of=None):
    from src.dataset import ProstateZonal2DDataset
    from src.evaluate import predict_case_reconstructed
    from src.reconstruction import validate_reconstruction_geometry
    model, preproc, epoch = load_model()
    rec_dir, cache = os.path.join(out_dir, "reconstructions_3d"), os.path.join(out_dir, "_case_metrics")
    os.makedirs(rec_dir, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    rows = []
    for i, cid in enumerate(ids, 1):
        cp_json = os.path.join(cache, f"{cid}.json")
        if os.path.exists(cp_json):
            rows.append(json.load(open(cp_json)))
            continue
        ds = ProstateZonal2DDataset(dataset_root=cases_root, patient_ids=[cid], image_name="t2.nii.gz",
                                    mask_name=mask_name, slice_sampling="all",
                                    preprocessing_config=preproc, cache_data=True)
        cp = predict_case_reconstructed(cid, model, ds, "cpu")
        nib.save(cp.pred_nii, os.path.join(rec_dir, f"pred_{cid}.nii.gz"))
        val = validate_reconstruction_geometry(cp.pred_nii, cp.geometry, valid_labels=(0, 1, 2))
        gt = np.asarray(cp.gt_original).astype(np.int64)
        extra = {"geometry_ok": bool(val["all_ok"]), **((extra_of or {}).get(cid, {}))}
        row = case_metrics(np.asarray(cp.pred_original).astype(np.int64), gt, cp.geometry.zooms, cid, extra)
        json.dump(row, open(cp_json, "w"))
        rows.append(row)
        print(f"  {i}/{len(ids)} {cid} CG {row['cg_dice_none']:.3f} PZ {row['pz_dice_none']:.3f}", flush=True)
    with open(os.path.join(out_dir, "per_case.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows, epoch


def boot_ci(x, n=10000, seed=0):
    x = np.asarray([v for v in x if np.isfinite(v)])
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def describe(rows, key):
    x = np.array([r[key] for r in rows], float)
    x = x[np.isfinite(x)]
    lo, hi = boot_ci(x)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "median": float(np.median(x)), "ci95": [lo, hi]}


def paired(rows, a, b):
    pairs = [(r[a], r[b]) for r in rows if np.isfinite(r[a]) and np.isfinite(r[b])]
    x, y = np.array(pairs).T
    d = y - x
    nz = d[d != 0]
    p = float(stats.wilcoxon(nz).pvalue) if len(nz) >= 1 else 1.0
    return {"n": int(len(d)), "mean_diff": float(d.mean()), "n_changed": int(len(nz)), "wilcoxon_p": p}


def phase_valid():
    ids = [f"{int(r['ID']):03d}" for r in csv.DictReader(open(os.path.join(P158, "valid.csv")))]
    assert len(ids) == 20
    out = os.path.join(OUT, "valid_prostate158")
    rows, epoch = infer(os.path.join(P158, "train"), ids, "t2_anatomy_reader1.nii.gz", out)
    dec = {"prespec_sha256": sha256(PRESPEC), "epoch": epoch, "n": len(rows), "rules": {}}
    for rule, zones in (("R1", ["cg"]), ("R2", ["cg", "pz"])):
        res, ok = {}, True
        for z in zones:
            dd = paired(rows, f"{z}_dice_none", f"{z}_dice_{rule}")
            dh = paired(rows, f"{z}_hd95_none", f"{z}_hd95_{rule}")
            res[z] = {"dice": dd, "hd95": dh}
            ok &= dd["mean_diff"] >= -DICE_TOL and dh["mean_diff"] <= 0
        dec["rules"][rule] = {"passed": bool(ok), **res}
    for k in ("cg_dice_none", "pz_dice_none", "cg_hd95_none", "pz_hd95_none"):
        dec[k] = describe(rows, k)
    json.dump(dec, open(os.path.join(out, "decision.json"), "w"), indent=2)
    print(json.dumps(dec["rules"], indent=1))


def harmonize_nci(work):
    man = json.load(open(os.path.join(NCI, "manifest.json")))
    ok = [r for r in man if r["status"] == "ok" and r["patient_id"] not in EXCLUDE]
    extra = {}
    for r in ok:
        pid = r["patient_id"]
        d = os.path.join(work, pid)
        os.makedirs(d, exist_ok=True)
        if not os.path.exists(os.path.join(d, "anatomy.nii.gz")):
            src = nib.load(os.path.join(NCI, pid, "anatomy.nii.gz"))
            a = np.asarray(src.dataobj).astype(np.uint8)
            m = np.zeros_like(a)
            m[a == 1] = PZ  # NCI 1 = PZ (verified anatomically)
            m[a == 2] = CG  # NCI 2 = CG
            nib.save(nib.Nifti1Image(m, src.affine, src.header), os.path.join(d, "anatomy.nii.gz"))
            shutil.copy2(os.path.join(NCI, pid, "t2.nii.gz"), os.path.join(d, "t2.nii.gz"))
        extra[pid] = {"site": r["site"], "inplane_mm": r["spacing_mm"][0]}
    return [r["patient_id"] for r in ok], extra


def phase_nci():
    dec_path = os.path.join(OUT, "valid_prostate158", "decision.json")
    if not os.path.exists(dec_path):
        raise RuntimeError("run --phase valid first: the rule must pass the internal check before NCI")
    dec = json.load(open(dec_path))
    if dec["prespec_sha256"] != sha256(PRESPEC):
        raise RuntimeError("pre-specification changed after the validation check")
    work = os.path.join(OUT, "_harmonized")
    ids, extra = harmonize_nci(work)
    out = os.path.join(OUT, "nci")
    rows, epoch = infer(work, ids, "anatomy.nii.gz", out, extra)
    s = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
         "checkpoint_sha256": E1_SHA, "epoch": epoch, "prespec_sha256": dec["prespec_sha256"],
         "n_cases": len(rows), "geometry_ok": sum(r["geometry_ok"] for r in rows),
         "rules_passed_internal_check": {k: v["passed"] for k, v in dec["rules"].items()},
         "primary": {}, "by_site": {}, "secondary": {}}
    for k in ("cg_dice_none", "pz_dice_none", "cg_hd95_none", "pz_hd95_none"):
        s["primary"][k] = describe(rows, k)
    for site in sorted({r["site"] for r in rows}):
        sub = [r for r in rows if r["site"] == site]
        s["by_site"][site] = {k: describe(sub, k) for k in ("cg_dice_none", "pz_dice_none", "cg_hd95_none", "pz_hd95_none")}
    for rule, zones in (("R1", ["cg"]), ("R2", ["cg", "pz"])):
        if not dec["rules"][rule]["passed"]:
            s["secondary"][rule] = "not applied: failed the internal check"
            continue
        s["secondary"][rule] = {z: {"dice": paired(rows, f"{z}_dice_none", f"{z}_dice_{rule}"),
                                    "hd95": paired(rows, f"{z}_hd95_none", f"{z}_hd95_{rule}"),
                                    "hd95_after": describe(rows, f"{z}_hd95_{rule}"),
                                    "dice_after": describe(rows, f"{z}_dice_{rule}")} for z in zones}
    json.dump(s, open(os.path.join(out, "summary.json"), "w"), indent=2)
    print(json.dumps({k: s[k] for k in ("n_cases", "geometry_ok", "primary", "secondary")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["valid", "nci"], required=True)
    a = ap.parse_args()
    phase_valid() if a.phase == "valid" else phase_nci()
