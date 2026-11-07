"""Model-free reconstruction ablation across datasets (CPU, no model, no training).

The expert annotation itself is used as the "prediction". It is pushed through
the production preprocessing (orientation to RAS, in-plane crop/pad to 442x442,
axial slicing with stored indices) and then reconstructed back to original
space by each variant below. Because the input is the annotation, the correct
pipeline must return it bit-identically (this IS the round-trip test), and
every deviation is attributable to the reconstruction stage alone.

Variants
--------
V0 correct            production pipeline (src.reconstruction)
V1 no_reorientation   inverse orientation step skipped
V2 centre_crop        centre crop/pad instead of the recorded offsets
V3 identity_affine    correct voxels written with eye(4)
V4 append_order       slices stacked in loader order (loader emits axial order)
V4b shuffled_order    slices stacked in loader order, loader order shuffled
                      (e.g. multi-worker or shuffle=True inference loader)
V6 spacing_only       correct voxels written with diag(spacing): voxel size kept,
                      direction and origin lost (a common "save it" shortcut)
V5 all_naive          V1 + V2 + V3 + V4 together

Checks applied to every variant
-------------------------------
L1 geometry checklist   header vs source: shape, affine, spacing, orientation,
                        label set (src.reconstruction.validate_reconstruction_geometry)
L2 round-trip           voxel-exact equality with the annotation
L3 anatomical rules     need no reference at all:
   L3a  peripheral zone centroid lies posterior to the central gland centroid
        in world coordinates (RAS+, so smaller y = more posterior)
   L3b  gland present on a contiguous run of slices (no empty slice inside the
        gland's axial extent) and no in-plane centroid jump > JUMP_MM between
        neighbouring slices
   Thresholds are fixed here before any result is read. Their false-alarm
   rate is reported on V0, where the volume is known to be correct.

Usage
-----
    python scripts/run_modelfree_ablation.py --datasets prostate158 msd_task05 prostatex nci_isbi_2013
Outputs go to results/modelfree_ablation/.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.geometry import read_geometry  # noqa: E402
from src.reconstruction import (  # noqa: E402
    reconstruct_case_to_original_space,
    validate_reconstruction_geometry,
)
from src.transforms import (  # noqa: E402
    CaseTransformMeta,
    _apply_pad_crop_2d,
    _compute_axis_pad_crop,
    _invert_ornt,
    _invert_pad_crop_2d,
    _standardize_orientation,
)

TARGET = (442, 442)  # configs/config_baseline.yaml
JUMP_MM = 15.0       # L3b threshold, fixed before results are read
SHUFFLE_SEED = 42
CLASSES = {1: "CG", 2: "PZ"}
VARIANTS = ["V0_correct", "V1_no_reorientation", "V2_centre_crop", "V3_identity_affine",
            "V4_append_order", "V4b_shuffled_order", "V6_spacing_only", "V5_all_naive"]

P158_DIRS = [
    os.path.join(ROOT, "dataset/prostate158_train/train"),
    os.path.join(ROOT, "dataset/prostate158_train/valid"),
    "G:/My Drive/THESIS_PROSTATE158/dataset/test/extracted/prostate158_test/test",
]
MSD_DIR = os.path.join(ROOT, "dataset/external/msd_task05/extracted/Task05_Prostate/labelsTr")
PROSTATEX_DIR = os.path.join(ROOT, "results/external_prostatex_e1/external_prostatex_e1/_harmonized/cases")
NCI_DIR = os.path.join(ROOT, "dataset/external/nci_isbi_2013/nifti")


# ---------------------------------------------------------------------------
# Dataset loaders: yield (case_id, mask image in the 0/1=CG/2=PZ convention)
# ---------------------------------------------------------------------------

def _relabel(img: nib.Nifti1Image, mapping: Dict[int, int]) -> nib.Nifti1Image:
    a = np.round(np.asarray(img.dataobj)).astype(np.int16)
    out = np.zeros_like(a)
    for src, dst in mapping.items():
        out[a == src] = dst
    return nib.Nifti1Image(out, img.affine, img.header)


def load_prostate158() -> Iterator[Tuple[str, nib.Nifti1Image]]:
    for d in P158_DIRS:
        for case in sorted(glob.glob(os.path.join(d, "*", "t2_anatomy_reader1.nii.gz"))):
            yield os.path.basename(os.path.dirname(case)), nib.load(case)


def load_msd() -> Iterator[Tuple[str, nib.Nifti1Image]]:
    # MSD Task05: 1 = PZ, 2 = TZ  ->  1 = CG (TZ), 2 = PZ
    for f in sorted(glob.glob(os.path.join(MSD_DIR, "prostate_*.nii.gz"))):
        yield os.path.basename(f).replace(".nii.gz", ""), _relabel(nib.load(f), {1: 2, 2: 1})


def load_prostatex() -> Iterator[Tuple[str, nib.Nifti1Image]]:
    for f in sorted(glob.glob(os.path.join(PROSTATEX_DIR, "*", "anatomy.nii.gz"))):
        yield os.path.basename(os.path.dirname(f)), nib.load(f)


def load_nci() -> Iterator[Tuple[str, nib.Nifti1Image]]:
    # NCI-ISBI 2013: label 1 lies posterior to label 2 in 78/78 cases
    # (scripts/prepare_nci_isbi.py), so 1 = PZ, 2 = CG  ->  1 = CG, 2 = PZ
    for f in sorted(glob.glob(os.path.join(NCI_DIR, "*", "anatomy.nii.gz"))):
        yield os.path.basename(os.path.dirname(f)), _relabel(nib.load(f), {1: 2, 2: 1})


LOADERS: Dict[str, Callable[[], Iterator[Tuple[str, nib.Nifti1Image]]]] = {
    "prostate158": load_prostate158,
    "msd_task05": load_msd,
    "prostatex": load_prostatex,
    "nci_isbi_2013": load_nci,
}


# ---------------------------------------------------------------------------
# Forward pass (production code path for a label map)
# ---------------------------------------------------------------------------

def forward(mask_img: nib.Nifti1Image):
    data = np.round(np.asarray(mask_img.dataobj)).astype(np.int64)
    ras, ornt_fwd = _standardize_orientation(data, mask_img.affine, order=0)
    pc_h = _compute_axis_pad_crop(ras.shape[0], TARGET[0])
    pc_w = _compute_axis_pad_crop(ras.shape[1], TARGET[1])
    pre = _apply_pad_crop_2d(ras, pc_h, pc_w, pad_value=0)
    meta = CaseTransformMeta(
        original_geometry=read_geometry(mask_img), orientation_fwd=ornt_fwd,
        reoriented_shape=ras.shape, pad_crop_h=pc_h, pad_crop_w=pc_w,
        spatial_mode="crop_pad", target_size=TARGET, z_resample_applied=False,
        z_resample_factor=1.0, pre_resample_depth=ras.shape[2], post_resample_depth=ras.shape[2],
    )
    slices = [pre[:, :, k] for k in range(pre.shape[2])]
    return slices, list(range(pre.shape[2])), meta


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def _centre_uncrop(vol: np.ndarray, meta: CaseTransformMeta) -> np.ndarray:
    nh, nw = meta.pad_crop_h.native_size, meta.pad_crop_w.native_size
    h, w = vol.shape[:2]
    out = np.zeros((nh, nw, vol.shape[2]), dtype=vol.dtype)
    # centre-align the two grids, whichever is larger
    sh, sw = (h - nh) // 2, (w - nw) // 2
    src = vol[max(sh, 0):max(sh, 0) + min(h, nh), max(sw, 0):max(sw, 0) + min(w, nw)]
    oh, ow = max(-sh, 0), max(-sw, 0)
    out[oh:oh + src.shape[0], ow:ow + src.shape[1]] = src
    return out


def build(variant: str, slices, indices, meta, src_img: nib.Nifti1Image, rng) -> nib.Nifti1Image:
    if variant == "V0_correct":
        return reconstruct_case_to_original_space(slices, indices, meta, read_geometry(src_img))

    naive_order = variant in ("V4_append_order", "V4b_shuffled_order", "V5_all_naive")
    shuffled = variant == "V4b_shuffled_order"
    naive_crop = variant in ("V2_centre_crop", "V5_all_naive")
    skip_reorient = variant in ("V1_no_reorientation", "V5_all_naive")
    identity = variant in ("V3_identity_affine", "V5_all_naive")
    spacing_only = variant == "V6_spacing_only"

    order = list(range(len(slices)))
    if shuffled:
        rng.shuffle(order)
    emitted = [slices[i] for i in order]          # what the loader hands over
    emitted_idx = [indices[i] for i in order]
    depth = meta.post_resample_depth
    vol = np.zeros(emitted[0].shape + (depth,), dtype=np.int64)
    for pos, (s, k) in enumerate(zip(emitted, emitted_idx)):
        vol[:, :, pos if naive_order else k] = s

    vol = _centre_uncrop(vol, meta) if naive_crop else _invert_pad_crop_2d(vol, meta.pad_crop_h, meta.pad_crop_w)
    if not skip_reorient:
        vol = nib.orientations.apply_orientation(vol, _invert_ornt(meta.orientation_fwd))
    vol = np.round(vol).astype(np.uint8)

    if identity:
        aff = np.eye(4)
    elif spacing_only:
        aff = np.diag(list(src_img.header.get_zooms()[:3]) + [1.0])
    else:
        aff = src_img.affine
    return nib.Nifti1Image(vol, aff)


# ---------------------------------------------------------------------------
# Measurements and checks
# ---------------------------------------------------------------------------

def world_centroid(vol, aff, c) -> Optional[np.ndarray]:
    idx = np.argwhere(vol == c)
    if idx.size == 0:
        return None
    return aff[:3, :3] @ idx.mean(axis=0) + aff[:3, 3]


def dice(a, b, c) -> float:
    if a.shape != b.shape:
        return float("nan")
    p, t = a == c, b == c
    s = p.sum() + t.sum()
    return 1.0 if s == 0 else float(2 * np.logical_and(p, t).sum() / s)


def l3_anatomy(vol: np.ndarray, aff: np.ndarray) -> Dict[str, object]:
    """Reference-free anatomical rules. Return pass flags plus raw values."""
    cg, pz = world_centroid(vol, aff, 1), world_centroid(vol, aff, 2)
    # L3a: in RAS+ world space, posterior = smaller y
    l3a = bool(cg is not None and pz is not None and pz[1] < cg[1])
    # L3b: find the slice axis in world terms = the voxel axis closest to world z
    zaxis = int(np.argmax(np.abs(aff[:3, :3][2, :]))) if np.any(aff[:3, :3]) else 2
    gland = vol > 0
    moved = np.moveaxis(gland, zaxis, -1)
    present = np.where(moved.reshape(-1, moved.shape[-1]).any(axis=0))[0]
    gaps = 0 if present.size == 0 else int((present.max() - present.min() + 1) - present.size)
    zooms = np.sqrt((aff[:3, :3] ** 2).sum(axis=0))
    inplane = [i for i in range(3) if i != zaxis]
    jumps = []
    prev = None
    for k in range(moved.shape[-1]):
        sl = moved[..., k]
        if not sl.any():
            prev = None
            continue
        c = np.argwhere(sl).mean(axis=0) * zooms[inplane]
        if prev is not None:
            jumps.append(float(np.linalg.norm(c - prev)))
        prev = c
    max_jump = max(jumps) if jumps else 0.0
    l3b = bool(gaps == 0 and max_jump <= JUMP_MM)
    return {"L3a_pz_posterior": l3a, "L3b_slice_continuity": l3b,
            "slice_gaps": gaps, "max_centroid_jump_mm": max_jump}


def run_case(ds: str, cid: str, mask_img: nib.Nifti1Image, rng) -> List[Dict[str, object]]:
    gt = np.round(np.asarray(mask_img.dataobj)).astype(np.uint8)
    src_geom = read_geometry(mask_img)
    slices, indices, meta = forward(mask_img)
    vox_ml = abs(np.linalg.det(mask_img.affine[:3, :3])) / 1000.0
    rows = []
    for v in VARIANTS:
        img = build(v, slices, indices, meta, mask_img, rng)
        vol = np.asarray(img.dataobj)
        aff = img.affine
        l1 = validate_reconstruction_geometry(img, src_geom)
        l2 = bool(vol.shape == gt.shape and np.array_equal(vol, gt))
        l3 = l3_anatomy(vol, aff)
        var_ml = abs(np.linalg.det(aff[:3, :3])) / 1000.0
        row = {
            "dataset": ds, "case": cid, "variant": v,
            "orientation": "".join(nib.aff2axcodes(mask_img.affine)),
            "inplane_op": f"{meta.pad_crop_h.op}/{meta.pad_crop_w.op}",
            "L1_geometry_flags": not bool(l1["all_ok"]),
            "L2_roundtrip_flags": not l2,
            "L3a_flags": not l3["L3a_pz_posterior"],
            "L3b_flags": not l3["L3b_slice_continuity"],
            "slice_gaps": l3["slice_gaps"], "max_centroid_jump_mm": l3["max_centroid_jump_mm"],
        }
        for c, name in CLASSES.items():
            row[f"dice_{name}"] = dice(vol, gt, c)
            ref_c, var_c = world_centroid(gt, mask_img.affine, c), world_centroid(vol, aff, c)
            row[f"shift_{name}_mm"] = (float(np.linalg.norm(var_c - ref_c))
                                       if ref_c is not None and var_c is not None else float("nan"))
            ref_v = (gt == c).sum() * vox_ml
            row[f"vol_err_{name}_pct"] = (((vol == c).sum() * var_ml - ref_v) / ref_v * 100) if ref_v else float("nan")
        rows.append(row)
    return rows


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["dataset", "variant"], sort=False)
    out = g.agg(
        n=("case", "count"),
        L1_detect=("L1_geometry_flags", "mean"),
        L2_detect=("L2_roundtrip_flags", "mean"),
        L3a_detect=("L3a_flags", "mean"),
        L3b_detect=("L3b_flags", "mean"),
        dice_CG=("dice_CG", "mean"), dice_PZ=("dice_PZ", "mean"),
        shift_CG_mm=("shift_CG_mm", "mean"), shift_PZ_mm_max=("shift_PZ_mm", "max"),
        vol_err_CG_pct=("vol_err_CG_pct", "mean"),
    ).reset_index()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(LOADERS))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/modelfree_ablation"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rows, skipped = [], []
    for ds in args.datasets:
        rng = np.random.default_rng(SHUFFLE_SEED)
        n = 0
        for cid, img in LOADERS[ds]():
            labels = set(np.unique(np.round(np.asarray(img.dataobj)).astype(int)).tolist())
            if not {1, 2}.issubset(labels):
                skipped.append({"dataset": ds, "case": cid, "labels": sorted(labels)})
                continue
            rows.extend(run_case(ds, cid, img, rng))
            n += 1
        print(f"{ds}: {n} cases")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "per_case.csv"), index=False)
    s = summarise(df)
    s.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    # Gate: V0 must be an exact round trip in every case, in every dataset.
    v0 = df[df.variant == "V0_correct"]
    gate = {
        "v0_cases": int(len(v0)),
        "v0_roundtrip_exact": int((~v0.L2_roundtrip_flags).sum()),
        "v0_geometry_pass": int((~v0.L1_geometry_flags).sum()),
        "v0_L3a_false_alarms": int(v0.L3a_flags.sum()),
        "v0_L3b_false_alarms": int(v0.L3b_flags.sum()),
        "jump_threshold_mm": JUMP_MM,
        "skipped": skipped,
    }
    with open(os.path.join(args.out, "gate.json"), "w") as fh:
        json.dump(gate, fh, indent=2)
    pd.set_option("display.width", 250)
    print(s.round(3).to_string(index=False))
    print(json.dumps(gate, indent=2))
    if gate["v0_roundtrip_exact"] != gate["v0_cases"]:
        raise SystemExit("GATE FAILED: the correct pipeline did not round-trip every case exactly.")


if __name__ == "__main__":
    main()
