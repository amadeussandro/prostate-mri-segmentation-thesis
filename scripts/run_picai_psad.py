"""PI-CAI: PSA density from the final model's whole-gland volume.

Implements results/picai_psad_prespecification.md, whose SHA-256 (LF line
endings) was locked in results/picai_psad/prespec_lock.json before any PI-CAI
file was downloaded. Every phase checks that lock first and stops if the
prespecification has changed.

Phases, run in this order (each one can be resumed):
  cohort     eligibility from the marksheet (no image is read)    -> cohort.csv, eligibility.json
  prepare    each eligible study's axial T2W from the archives, MHA -> NIfTI, corner check
             -> conversion.csv
  overlap    duplicates of PROSTATEx, NCI-ISBI 2013 and MSD images (voxel-wise correlation)
             -> overlap.json
  infer      final model + frozen reconstruction: zone volumes, rule R1, geometry check,
             Bosma22b volume and Dice                              -> per_case.csv
  roundtrip  Bosma22b masks through the pipeline's own forward and inverse steps
             -> roundtrip.csv
  analyze    prespecified endpoints                                -> summary.json, report.md

Archives are read member by member (never extracted wholesale), and only names
of the form <patient>/<patient>_<study>_t2w.mha are used.

Usage:
    python scripts/run_picai_psad.py --phase cohort
    python scripts/run_picai_psad.py --phase prepare
    python scripts/run_picai_psad.py --phase infer [--device cuda] [--limit 5]
"""
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

PRESPEC = os.path.join(ROOT, "results", "picai_psad_prespecification.md")
OUT = os.path.join(ROOT, "results", "picai_psad")
LOCK = os.path.join(OUT, "prespec_lock.json")
SELECTION = os.path.join(ROOT, "results", "e1_seed_selection", "selection.json")
DATA = "D:/1.AmadeusKuliah/Skripsi/PICAI"
PROSTATEX_T2 = "G:/My Drive/THESIS_PROSTATE158/dataset/external/prostatex/raw/images/t2"
NCI = os.path.join(ROOT, "dataset", "external", "nci_isbi_2013", "nifti")
MSD = os.path.join(ROOT, "dataset", "external", "msd_task05", "extracted", "Task05_Prostate")

MARGIN = -0.05          # noninferiority margin on the AUC difference
THRESH = 0.15           # PSA density threshold, ng/mL/mL
N_BOOT = 10000
SEED = 0
CORNER_TOL_MM = 0.01    # MHA -> NIfTI conversion check
DUP_R = 0.99            # duplicate images: Pearson r on the same RAS array shape
CG, PZ = 1, 2
T2W = re.compile(r"^(\d{5})/(\d{5})_(\d{7})_t2w\.mha$")


def p(*parts):
    return os.path.join(DATA, *parts)


def labels_dir():
    return p("picai_labels")


def bosma_path(case):
    return os.path.join(labels_dir(), "anatomical_delineations", "whole_gland", "AI", "Bosma22b",
                        f"{case}.nii.gz")


# ----------------------------------------------------------------------------- helpers
def sha_lf(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def check_lock() -> dict:
    lock = json.load(open(LOCK))
    got = sha_lf(PRESPEC)
    if got != lock["sha256_lf"]:
        raise RuntimeError(f"prespecification changed after locking: {got} != {lock['sha256_lf']}")
    return lock


def fnum(s):
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


# ----------------------------------------------------------------------------- cohort
def phase_cohort():
    lock = check_lock()
    ms = os.path.join(labels_dir(), "clinical_information", "marksheet.csv")
    rows = read_csv(ms)
    commit = subprocess.run(["git", "-C", labels_dir(), "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()
    if any(not r["mri_date"].strip() for r in rows):
        raise RuntimeError("a study has no mri_date; the earliest-study rule needs a decision")
    by_pat = {}
    for r in rows:
        by_pat.setdefault(r["patient_id"], []).append(r)
    first = [sorted(v, key=lambda r: (r["mri_date"], int(r["study_id"])))[0] for v in by_pat.values()]
    excl = {"later study of the same patient": len(rows) - len(first), "no PSA": 0,
            "no csPCa label": 0}
    cohort = []
    for r in sorted(first, key=lambda r: int(r["patient_id"])):
        psa, lab = fnum(r["psa"]), r["case_csPCa"].strip().upper()
        if psa is None or psa <= 0:
            excl["no PSA"] += 1
            continue
        if lab not in ("YES", "NO"):
            excl["no csPCa label"] += 1
            continue
        vol = fnum(r["prostate_volume"])
        cohort.append({"case": f"{r['patient_id']}_{r['study_id']}", "patient_id": r["patient_id"],
                       "study_id": r["study_id"], "mri_date": r["mri_date"], "center": r["center"],
                       "psa": psa, "report_volume_ml": vol if vol is not None and vol > 0 else "",
                       "psad_marksheet": r["psad"], "cspca": int(lab == "YES")})
    os.makedirs(OUT, exist_ok=True)
    write_csv(os.path.join(OUT, "cohort.csv"), cohort)
    e = {"generated_utc": now(), "prespec_sha256_lf": lock["sha256_lf"],
         "labels_repo": "https://github.com/DIAGNijmegen/picai_labels", "labels_commit": commit,
         "marksheet_sha256": sha_file(ms), "n_studies": len(rows), "n_patients": len(by_pat),
         "excluded": excl, "n_eligible": len(cohort),
         "n_with_report_volume": sum(1 for c in cohort if c["report_volume_ml"] != ""),
         "n_cspca": sum(c["cspca"] for c in cohort)}
    json.dump(e, open(os.path.join(OUT, "eligibility.json"), "w"), indent=2)
    print(json.dumps(e, indent=1))


# ----------------------------------------------------------------------------- prepare
def corner_check(img, nii):
    """Max distance (mm) between SimpleITK (LPS) and nibabel (RAS) world coordinates of the 8 corners."""
    size = img.GetSize()
    worst = 0.0
    for i in (0, size[0] - 1):
        for j in (0, size[1] - 1):
            for k in (0, size[2] - 1):
                lps = np.array(img.TransformIndexToPhysicalPoint((int(i), int(j), int(k))))
                ras = (nii.affine @ np.array([i, j, k, 1.0]))[:3]
                worst = max(worst, float(np.abs(ras - lps * np.array([-1.0, -1.0, 1.0])).max()))
    return worst


def phase_prepare(limit=None):
    check_lock()
    import nibabel as nib
    import SimpleITK as sitk
    cohort = read_csv(os.path.join(OUT, "cohort.csv"))
    zips = sorted(glob.glob(p("zips", "picai_public_images_fold*.zip")))
    if not zips:
        raise RuntimeError("no archives found")
    handles, where = {}, {}
    for z in list(zips):
        try:
            zf = zipfile.ZipFile(z)
        except zipfile.BadZipFile:  # still downloading: its studies wait for a later run
            print(f"  skipping incomplete archive {os.path.basename(z)}")
            zips.remove(z)
            continue
        handles[z] = zf
        for n in zf.namelist():
            m = T2W.match(n)
            if m and m.group(1) == m.group(2):
                where[f"{m.group(2)}_{m.group(3)}"] = (z, n)
    done = 0
    os.makedirs(p("conversions"), exist_ok=True)
    for c in cohort:
        case = c["case"]
        d = p("work", case)
        rec_path = p("conversions", f"{case}.json")
        if os.path.exists(rec_path):
            rec = json.load(open(rec_path))
            # Done: failed for good, already inferred, or its NIfTI is still on disk.
            if (rec["status"] != "ok" or os.path.exists(p("cache_infer", f"{case}.json"))
                    or os.path.exists(os.path.join(d, "blank.nii.gz"))):
                continue
        if case not in where:
            if len(zips) == 5:  # every archive is here, so the study really has no axial T2W
                json.dump({"case": case, "status": "no T2W in archives"}, open(rec_path, "w"), indent=1)
            continue  # otherwise its archive is not here (yet): folds can be processed one at a time
        if limit is not None and done >= limit:
            break
        os.makedirs(d, exist_ok=True)
        z, member = where[case]
        rec = {"case": case, "archive": os.path.basename(z), "member": member}
        mha = os.path.join(d, "t2w.mha")
        with open(mha, "wb") as fh:
            fh.write(handles[z].read(member))
        try:
            img = sitk.ReadImage(mha)
        except RuntimeError as ex:
            rec.update(status="unreadable", error=str(ex)[:200])
        else:
            if img.GetDimension() != 3:
                rec.update(status="not 3D")
            else:
                t2 = os.path.join(d, "t2.nii.gz")
                sitk.WriteImage(img, t2)
                nii = nib.load(t2)
                if tuple(nii.shape[:3]) != tuple(img.GetSize()):
                    raise RuntimeError(f"{case}: NIfTI shape {nii.shape} != MHA size {img.GetSize()}")
                worst = corner_check(img, nii)
                rec.update(shape=list(nii.shape[:3]),
                           spacing=[round(float(v), 4) for v in nii.header.get_zooms()[:3]],
                           axcodes="".join(nib.aff2axcodes(nii.affine)), corner_max_diff_mm=worst)
                if worst > CORNER_TOL_MM:
                    json.dump({**rec, "status": "corner check failed"}, open(rec_path, "w"), indent=1)
                    raise RuntimeError(f"{case}: conversion moved a corner by {worst:.4f} mm; "
                                       "the prespecification says to stop and find the cause")
                hdr = nii.header.copy()
                hdr.set_data_dtype(np.uint8)
                blank = nib.Nifti1Image(np.zeros(nii.shape[:3], np.uint8), nii.affine, hdr)
                nib.save(blank, os.path.join(d, "blank.nii.gz"))  # placeholder: the loader needs a mask
                rec["status"] = "ok"
        os.remove(mha)
        json.dump(rec, open(rec_path, "w"), indent=1)
        done += 1
        if done % 50 == 0:
            print(f"  prepared {done}", flush=True)
    rows = list(load_records().values())
    missing = len(cohort) - len(rows)
    if rows:
        write_csv(os.path.join(OUT, "conversion.csv"), rows)
    st = {}
    for r in rows:
        st[r["status"]] = st.get(r["status"], 0) + 1
    print(f"prepared {done} now; status {st}; not yet available {missing}; archives {len(zips)}")


def load_records():
    """Conversion records of the cohort, read from the per-case JSON files (all folds so far)."""
    out = {}
    for c in read_csv(os.path.join(OUT, "cohort.csv")):
        rp = p("conversions", f"{c['case']}.json")
        if os.path.exists(rp):
            out[c["case"]] = json.load(open(rp))
    return out


def conversions():
    return {r["case"]: r for r in read_csv(os.path.join(OUT, "conversion.csv"))}


# ----------------------------------------------------------------------------- overlap
def canonical_shape(img):
    import nibabel as nib
    ornt = nib.orientations.io_orientation(img.affine)
    out = [0, 0, 0]
    for i in range(3):
        out[int(ornt[i, 0])] = int(img.shape[i])
    return tuple(out)


def canonical_array(img):
    import nibabel as nib
    return np.asarray(nib.as_closest_canonical(img).dataobj, dtype=np.float32)


def phase_overlap():
    check_lock()
    import nibabel as nib
    others = []
    for f in sorted(glob.glob(os.path.join(PROSTATEX_T2, "*.nii.gz"))):
        others.append(("PROSTATEx", os.path.basename(f).split("_")[0], f, None))
    for f in sorted(glob.glob(os.path.join(NCI, "*", "t2.nii.gz"))):
        others.append(("NCI-ISBI", os.path.basename(os.path.dirname(f)), f, None))
    for sub in ("imagesTr", "imagesTs"):
        for f in sorted(glob.glob(os.path.join(MSD, sub, "prostate_*.nii.gz"))):
            if not os.path.basename(f).startswith("._"):
                others.append(("MSD", os.path.basename(f)[:-7], f, 0))
    shapes = {}
    for src, oid, f, ch in others:
        img = nib.load(f)
        if ch is not None:
            img = nib.four_to_three(img)[ch]
        shapes.setdefault(canonical_shape(img), []).append((src, oid, img))
    counts = {s: sum(1 for o in others if o[0] == s) for s in ("PROSTATEx", "NCI-ISBI", "MSD")}
    if counts["PROSTATEx"] == 0:
        raise RuntimeError(f"no PROSTATEx images found at {PROSTATEX_T2}")
    conv = conversions()
    arrays, pairs, dups = {}, 0, []
    for case, r in sorted(conv.items()):
        if r["status"] != "ok":
            continue
        img = nib.load(p("work", case, "t2.nii.gz"))
        cands = shapes.get(canonical_shape(img), [])
        if not cands:
            continue
        a = canonical_array(img).ravel()
        for src, oid, oimg in cands:
            key = (src, oid)
            if key not in arrays:
                arrays[key] = canonical_array(oimg).ravel()
            pairs += 1
            rr = float(np.corrcoef(a, arrays[key])[0, 1])
            if rr >= DUP_R:
                dups.append({"case": case, "other": f"{src}:{oid}", "r": round(rr, 5)})
    res = {"generated_utc": now(), "method": f"same RAS array shape and Pearson r >= {DUP_R}",
           "picai_checked": sum(1 for r in conv.values() if r["status"] == "ok"),
           "others": counts, "candidate_pairs": pairs, "duplicates": dups,
           "note": "PROSTATEx = the 204 T2 images used in the study (Cuocolo et al. subset)"}
    json.dump(res, open(os.path.join(OUT, "overlap.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "duplicates"}, indent=1),
          f"\nduplicates: {len(dups)}")


# ----------------------------------------------------------------------------- infer
def phase_infer(device="cpu", limit=None):
    check_lock()
    import nibabel as nib
    import torch
    import run_nci_external_validation as nci  # the same R1 rule (6-connectivity) as before
    from src.dataset import ProstateZonal2DDataset
    from src.ensemble import load_final_model
    from src.evaluate import predict_case_reconstructed
    from src.reconstruction import validate_reconstruction_geometry
    from src.transforms import PreprocessingConfig
    if device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)
    else:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    model, cfg, ident = load_final_model(SELECTION, device)
    preproc = PreprocessingConfig.from_dict(cfg.get("preprocessing", {}))
    cache, preds = p("cache_infer"), p("predictions")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(preds, exist_ok=True)
    json.dump({**ident, "device": device, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None},
              open(os.path.join(OUT, "model_identity.json"), "w"), indent=2)
    recs = load_records()
    cases = [c for c, r in sorted(recs.items()) if r["status"] == "ok"]
    todo = [c for c in cases if not os.path.exists(os.path.join(cache, f"{c}.json"))
            and os.path.exists(p("work", c, "blank.nii.gz"))]
    if limit is not None:
        todo = todo[:limit]
    print(f"infer: {len(cases)} converted so far, {len(todo)} to do now, device {device}", flush=True)
    for i, case in enumerate(todo, 1):
        t0 = time.time()
        ds = ProstateZonal2DDataset(dataset_root=p("work"), patient_ids=[case], image_name="t2.nii.gz",
                                    mask_name="blank.nii.gz", slice_sampling="all",
                                    preprocessing_config=preproc, cache_data=True)
        cp = predict_case_reconstructed(case, model, ds, device)
        nib.save(cp.pred_nii, os.path.join(preds, f"pred_{case}.nii.gz"))
        val = validate_reconstruction_geometry(cp.pred_nii, cp.geometry, valid_labels=(0, 1, 2))
        pred = np.asarray(cp.pred_original).astype(np.int64)
        ml = float(np.prod(cp.geometry.zooms[:3])) / 1000.0
        r1 = nci.apply_rule(pred, "R1")
        row = {"case": case, "n_slices": int(pred.shape[2]), "shape": "x".join(map(str, pred.shape)),
               "geometry_ok": bool(val["all_ok"]), "affine_max_abs_diff": val.get("affine_max_abs_diff"),
               "cg_ml": (pred == CG).sum() * ml, "pz_ml": (pred == PZ).sum() * ml,
               "wg_ml": (pred > 0).sum() * ml, "wg_ml_R1": (r1 > 0).sum() * ml,
               "empty": bool((pred > 0).sum() == 0)}
        bp = bosma_path(case)
        if os.path.exists(bp):
            b = nib.load(bp)
            bm = np.asarray(b.dataobj) > 0.5
            row["bosma_ml"] = float(bm.sum() * np.prod(b.header.get_zooms()[:3]) / 1000.0)
            same = tuple(b.shape[:3]) == tuple(pred.shape) and np.allclose(b.affine, cp.geometry.affine,
                                                                          atol=1e-3)
            row["bosma_same_grid"] = bool(same)
            if same:
                a = pred > 0
                den = a.sum() + bm.sum()
                row["dice_vs_bosma"] = float(2 * (a & bm).sum() / den) if den else 1.0
        row["seconds"] = round(time.time() - t0, 1)
        row = {k: (float(v) if isinstance(v, np.floating) else v) for k, v in row.items()}
        json.dump(row, open(os.path.join(cache, f"{case}.json"), "w"))
        print(f"  {i}/{len(todo)} {case} WG {row['wg_ml']:.1f} mL geom {row['geometry_ok']} "
              f"({row['seconds']:.0f} s)", flush=True)
    rows = [json.load(open(os.path.join(cache, f"{c}.json"))) for c in cases
            if os.path.exists(os.path.join(cache, f"{c}.json"))]
    if rows:
        write_csv(os.path.join(OUT, "per_case.csv"), rows)
    print(f"per_case.csv: {len(rows)} inferred of {len(cases)} converted")


# ----------------------------------------------------------------------------- roundtrip
def phase_roundtrip(limit=None):
    """Level-2 test on each Bosma22b mask's own grid: the mask is both the image
    and the label, so only the geometric forward and inverse steps matter."""
    check_lock()
    import nibabel as nib
    from src.dataset import ProstateZonal2DDataset
    from src.reconstruction import reconstruct_case_to_original_space
    from src.ensemble import load_member
    from src.transforms import PreprocessingConfig
    sel = json.load(open(SELECTION))
    from src.ensemble import resolve_member_path
    _, info = load_member(resolve_member_path(ROOT, sel["final_model"]["members"][0]["path"]))
    preproc = PreprocessingConfig.from_dict(info["preprocessing"])
    conv = conversions()
    cases = [c for c, r in sorted(conv.items()) if r["status"] == "ok" and os.path.exists(bosma_path(c))]
    cache = p("cache_roundtrip")
    os.makedirs(cache, exist_ok=True)
    done = 0
    for case in cases:
        cj = os.path.join(cache, f"{case}.json")
        if os.path.exists(cj):
            continue
        if limit is not None and done >= limit:
            break
        bp = bosma_path(case)
        orig = np.round(nib.load(bp).get_fdata()).astype(np.int64)
        rec = {"case": case, "foreground_voxels": int((orig > 0).sum())}
        if rec["foreground_voxels"] == 0:
            rec.update(status="empty mask")
        else:
            root, pid = os.path.dirname(os.path.dirname(bp)), os.path.basename(os.path.dirname(bp))
            ds = ProstateZonal2DDataset(dataset_root=root, patient_ids=[pid], image_name=f"{case}.nii.gz",
                                        mask_name=f"{case}.nii.gz", slice_sampling="all",
                                        preprocessing_config=preproc, cache_data=True)
            geom, meta = ds.get_case_metadata(pid)
            sl, idx = [], []
            for i in range(len(ds)):
                s = ds[i]
                sl.append(s["mask"])
                idx.append(s["slice_idx"])
            back = np.asarray(reconstruct_case_to_original_space(sl, idx, meta, geom).dataobj).astype(np.int64)
            same = back.shape == orig.shape
            nd = int((back != orig).sum()) if same else int(orig.size)
            rec.update(status="ok", same_shape=same, voxels_different=nd, passed=bool(same and nd == 0),
                       foreground_lost=int(((orig > 0) & (back == 0)).sum()) if same else None)
        json.dump(rec, open(cj, "w"))
        done += 1
    rows = [json.load(open(os.path.join(cache, f"{c}.json"))) for c in cases
            if os.path.exists(os.path.join(cache, f"{c}.json"))]
    if rows:
        write_csv(os.path.join(OUT, "roundtrip.csv"), rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"roundtrip: {len(rows)} of {len(cases)}; passed {sum(1 for r in ok if r['passed'] in (True, 'True'))}"
          f" of {len(ok)} non-empty")


# ----------------------------------------------------------------------------- analyze
def auc(score, y):
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def compare(a, b, y, seed=SEED, n=N_BOOT):
    """Paired AUC difference a - b, percentile CI from a bootstrap stratified by outcome."""
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    da, db = np.empty(n), np.empty(n)
    for k in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        da[k], db[k] = auc(a[idx], y[idx]), auc(b[idx], y[idx])
    d = da - db
    lo, hi = (float(v) for v in np.percentile(d, [2.5, 97.5]))
    return {"n": int(len(y)), "n_pos": int(y.sum()), "auc_a": auc(a, y), "auc_b": auc(b, y),
            "auc_a_ci95": [float(v) for v in np.percentile(da, [2.5, 97.5])],
            "auc_b_ci95": [float(v) for v in np.percentile(db, [2.5, 97.5])],
            "diff": auc(a, y) - auc(b, y), "diff_ci95": [lo, hi],
            "noninferior": bool(lo > MARGIN), "superior": bool(lo > 0)}


def agreement(x, ref):
    from scipy.stats import spearmanr
    d = x - ref
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return {"n": int(len(d)), "mean_x": float(x.mean()), "mean_ref": float(ref.mean()), "bias": bias,
            "sd": sd, "loa95": [bias - 1.96 * sd, bias + 1.96 * sd],
            "median_rel_diff": float(np.median(d / ref)), "spearman": float(spearmanr(x, ref)[0])}


def at_threshold(score, y, t=THRESH):
    pos = score >= t
    return {"sensitivity": float((pos & (y == 1)).sum() / (y == 1).sum()),
            "specificity": float((~pos & (y == 0)).sum() / (y == 0).sum()), "n_positive": int(pos.sum())}


def psad(psa, vol):
    with np.errstate(divide="ignore"):
        return np.where(vol > 0, psa / np.where(vol > 0, vol, 1.0), np.inf)


def phase_analyze():
    lock = check_lock()
    elig = json.load(open(os.path.join(OUT, "eligibility.json")))
    cohort = {c["case"]: c for c in read_csv(os.path.join(OUT, "cohort.csv"))}
    conv = conversions()
    pc = {r["case"]: r for r in read_csv(os.path.join(OUT, "per_case.csv"))}
    ov = json.load(open(os.path.join(OUT, "overlap.json")))
    rt = read_csv(os.path.join(OUT, "roundtrip.csv")) if os.path.exists(os.path.join(OUT, "roundtrip.csv")) else []
    missing = [c for c in cohort if c not in conv]
    if missing:
        raise RuntimeError(f"{len(missing)} eligible studies not prepared yet; finish all archives first")
    not_ok = {c: r["status"] for c, r in conv.items() if r["status"] != "ok"}
    not_inferred = [c for c in cohort if c not in not_ok and c not in pc]
    if not_inferred:
        raise RuntimeError(f"{len(not_inferred)} studies not inferred yet")
    tb = lambda v: str(v) in ("True", "true", "1")  # noqa: E731
    cases = [c for c in cohort if c in pc]
    excl = dict(elig["excluded"])
    for s in not_ok.values():
        excl[f"image: {s}"] = excl.get(f"image: {s}", 0) + 1
    prim = [c for c in cases if cohort[c]["report_volume_ml"] != ""]
    excl["no reported prostate volume"] = len(cases) - len(prim)

    def arr(keys, f):
        return np.array([f(k) for k in keys], float)

    y = arr(prim, lambda c: int(cohort[c]["cspca"])).astype(int)
    psa = arr(prim, lambda c: float(cohort[c]["psa"]))
    vrep = arr(prim, lambda c: float(cohort[c]["report_volume_ml"]))
    vmod = arr(prim, lambda c: float(pc[c]["wg_ml"]))
    vr1 = arr(prim, lambda c: float(pc[c]["wg_ml_R1"]))
    d_rep, d_mod, d_r1 = psad(psa, vrep), psad(psa, vmod), psad(psa, vr1)
    s = {"generated_utc": now(), "prespec_sha256_lf": lock["sha256_lf"],
         "model_identity": json.load(open(os.path.join(OUT, "model_identity.json"))),
         "labels_commit": elig["labels_commit"], "marksheet_sha256": elig["marksheet_sha256"],
         "n_studies": elig["n_studies"], "n_patients": elig["n_patients"], "n_inferred": len(cases),
         "n_primary": len(prim), "excluded": excl,
         "share_of_patients_not_in_primary": 1 - len(prim) / elig["n_patients"],
         "n_empty_model": int(sum(tb(pc[c]["empty"]) for c in cases)),
         "geometry_ok": int(sum(tb(pc[c]["geometry_ok"]) for c in cases)),
         "corner_max_diff_mm": max(float(conv[c]["corner_max_diff_mm"]) for c in cases)}
    s["over_20pct_removed"] = s["share_of_patients_not_in_primary"] > 0.20
    s["primary"] = compare(d_mod, d_rep, y)
    s["primary"]["labels"] = ["PSAD model", "PSAD report"]
    s["volume_model_vs_report"] = agreement(vmod, vrep)
    bos = [c for c in cases if pc[c].get("bosma_ml", "") != ""]
    s["volume_model_vs_bosma"] = agreement(arr(bos, lambda c: float(pc[c]["wg_ml"])),
                                           arr(bos, lambda c: float(pc[c]["bosma_ml"])))
    dic = arr([c for c in bos if pc[c].get("dice_vs_bosma", "") != ""], lambda c: float(pc[c]["dice_vs_bosma"]))
    s["dice_vs_bosma"] = ({"n": int(len(dic)), "mean": float(dic.mean()), "median": float(np.median(dic)),
                           "q25_q75": [float(v) for v in np.percentile(dic, [25, 75])]} if len(dic) else None)
    s["bosma_same_grid"] = int(sum(tb(pc[c].get("bosma_same_grid")) for c in bos))
    s["reclassification"] = {"threshold": THRESH, "model": at_threshold(d_mod, y),
                             "report": at_threshold(d_rep, y),
                             "model_pos_report_neg": int(((d_mod >= THRESH) & (d_rep < THRESH)).sum()),
                             "model_neg_report_pos": int(((d_mod < THRESH) & (d_rep >= THRESH)).sum())}
    s["reclassification"]["discordant"] = (s["reclassification"]["model_pos_report_neg"]
                                           + s["reclassification"]["model_neg_report_pos"])
    pb = [c for c in prim if pc[c].get("bosma_ml", "") != ""]
    if pb:
        yb = arr(pb, lambda c: int(cohort[c]["cspca"])).astype(int)
        pbs = arr(pb, lambda c: float(cohort[c]["psa"]))
        s["bosma_psad_vs_report"] = compare(psad(pbs, arr(pb, lambda c: float(pc[c]["bosma_ml"]))),
                                            psad(pbs, arr(pb, lambda c: float(cohort[c]["report_volume_ml"]))), yb)
        s["reclassification"]["bosma"] = at_threshold(psad(pbs, arr(pb, lambda c: float(pc[c]["bosma_ml"]))), yb)
    s["by_center"] = {}
    for cen in sorted({cohort[c]["center"] for c in prim}):
        m = np.array([cohort[c]["center"] == cen for c in prim])
        s["by_center"][cen] = compare(d_mod[m], d_rep[m], y[m])
    s["sensitivity"] = {"rule_R1": compare(d_r1, d_rep, y)}
    dup = {d["case"] for d in ov["duplicates"]}
    m = np.array([c not in dup for c in prim])
    s["sensitivity"]["without_duplicates"] = {"n_removed": int((~m).sum()), **compare(d_mod[m], d_rep[m], y[m])}
    m = np.array([tb(pc[c]["geometry_ok"]) for c in prim])
    s["sensitivity"]["without_geometry_failures"] = ({"n_removed": int((~m).sum()), **compare(d_mod[m], d_rep[m], y[m])}
                                                     if (~m).any() else "no geometry failures")
    s["overlap"] = {k: ov[k] for k in ("method", "others", "candidate_pairs")}
    s["overlap"]["n_duplicates"] = len(ov["duplicates"])
    s["overlap"]["by_source"] = {}
    for d in ov["duplicates"]:
        src = d["other"].split(":")[0]
        s["overlap"]["by_source"][src] = s["overlap"]["by_source"].get(src, 0) + 1
    ok = [r for r in rt if r["status"] == "ok"]
    s["roundtrip_bosma"] = {"n_tested": len(ok), "passed": int(sum(tb(r["passed"]) for r in ok)),
                            "n_empty": len(rt) - len(ok)}
    json.dump(s, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    write_report(s)
    print(json.dumps({k: s[k] for k in ("n_primary", "primary", "excluded")}, indent=1))


def write_report(s):
    f3 = lambda v: f"{v:.3f}"  # noqa: E731
    pr = s["primary"]
    L = ["# PI-CAI: PSA density from the final model's whole-gland volume", "",
         f"Prespecification SHA-256 (LF): `{s['prespec_sha256_lf']}`; labels commit "
         f"`{s['labels_commit']}`; generated {s['generated_utc']}.", "",
         f"{s['n_studies']} studies from {s['n_patients']} patients; {s['n_inferred']} first studies "
         f"inferred; {s['n_primary']} in the primary comparison "
         f"({s['n_primary'] and pr['n_pos']} with csPCa).", "", "## Exclusions", "",
         "| Reason | n |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in s["excluded"].items()]
    L += ["", "## Primary endpoint", "",
          "| | AUC (95% CI) |", "|---|---|",
          f"| PSA density, model volume | {f3(pr['auc_a'])} ({f3(pr['auc_a_ci95'][0])}-{f3(pr['auc_a_ci95'][1])}) |",
          f"| PSA density, report volume | {f3(pr['auc_b'])} ({f3(pr['auc_b_ci95'][0])}-{f3(pr['auc_b_ci95'][1])}) |",
          f"| Difference (model - report) | {f3(pr['diff'])} ({f3(pr['diff_ci95'][0])} to {f3(pr['diff_ci95'][1])}) |",
          "", f"Noninferiority (margin {MARGIN}): {'shown' if pr['noninferior'] else 'not shown'}; "
          f"superiority: {'shown' if pr['superior'] else 'not shown'}.", "",
          "## Volume agreement", "", "| Comparison | n | Bias (mL) | 95% LoA (mL) | Median rel. diff | Spearman |",
          "|---|---|---|---|---|---|"]
    for k, lab in (("volume_model_vs_report", "Model vs report"), ("volume_model_vs_bosma", "Model vs Bosma22b")):
        a = s[k]
        L.append(f"| {lab} | {a['n']} | {a['bias']:.1f} | {a['loa95'][0]:.1f} to {a['loa95'][1]:.1f} | "
                 f"{a['median_rel_diff']:.3f} | {a['spearman']:.3f} |")
    if s["dice_vs_bosma"]:
        dv = s["dice_vs_bosma"]
        L += ["", f"Whole-gland Dice vs Bosma22b (same grid, n = {dv['n']}): mean {f3(dv['mean'])}, "
              f"median {f3(dv['median'])}."]
    rc = s["reclassification"]
    L += ["", f"## Threshold {rc['threshold']} ng/mL/mL", "", "| PSA density from | Sensitivity | Specificity |",
          "|---|---|---|"]
    for k in ("model", "report", "bosma"):
        if k in rc:
            L.append(f"| {k} | {f3(rc[k]['sensitivity'])} | {f3(rc[k]['specificity'])} |")
    L += ["", f"Discordant patients: {rc['discordant']} (model positive only {rc['model_pos_report_neg']}, "
          f"report positive only {rc['model_neg_report_pos']}).", "", "## By center", "",
          "| Center | n (csPCa) | AUC model | AUC report | Difference (95% CI) |", "|---|---|---|---|---|"]
    for cen, c in s["by_center"].items():
        L.append(f"| {cen} | {c['n']} ({c['n_pos']}) | {f3(c['auc_a'])} | {f3(c['auc_b'])} | "
                 f"{f3(c['diff'])} ({f3(c['diff_ci95'][0])} to {f3(c['diff_ci95'][1])}) |")
    L += ["", "## Sensitivity analyses", ""]
    for k, c in s["sensitivity"].items():
        if isinstance(c, str):
            L.append(f"- {k}: {c}")
        else:
            L.append(f"- {k}: difference {f3(c['diff'])} ({f3(c['diff_ci95'][0])} to {f3(c['diff_ci95'][1])}), "
                     f"n = {c['n']}" + (f", removed {c['n_removed']}" if "n_removed" in c else ""))
    if "bosma_psad_vs_report" in s:
        b = s["bosma_psad_vs_report"]
        L.append(f"- Bosma22b PSA density (reference): AUC {f3(b['auc_a'])}, vs report {f3(b['diff'])} "
                 f"({f3(b['diff_ci95'][0])} to {f3(b['diff_ci95'][1])})")
    ov, rt = s["overlap"], s["roundtrip_bosma"]
    L += ["", "## Verification", "",
          f"- Conversion corner check: max {s['corner_max_diff_mm']:.5f} mm.",
          f"- Header-based geometry check: {s['geometry_ok']} of {s['n_inferred']} passed.",
          f"- Round-trip of Bosma22b masks: {rt['passed']} of {rt['n_tested']} exact ({rt['n_empty']} empty masks).",
          f"- Empty model segmentations: {s['n_empty_model']}.",
          f"- Duplicates ({ov['method']}): {ov['n_duplicates']} {ov['by_source']}.",
          f"- Share of patients not in the primary comparison: {s['share_of_patients_not_in_primary']:.1%}"
          + (" (over 20%: stated as a limitation)." if s["over_20pct_removed"] else ".")]
    open(os.path.join(OUT, "report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["cohort", "prepare", "overlap", "infer", "roundtrip", "analyze"])
    ap.add_argument("--data", default=DATA, help="folder with zips/ and picai_labels/")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="process at most N new cases (testing)")
    a = ap.parse_args()
    DATA = os.path.abspath(a.data)
    {"cohort": phase_cohort, "prepare": lambda: phase_prepare(a.limit), "overlap": phase_overlap,
     "infer": lambda: phase_infer(a.device, a.limit), "roundtrip": lambda: phase_roundtrip(a.limit),
     "analyze": phase_analyze}[a.phase]()
