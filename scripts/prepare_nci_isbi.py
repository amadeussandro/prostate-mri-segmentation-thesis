"""Convert NCI-ISBI 2013 (DICOM series + NRRD masks) to NIfTI.

Writes dataset/external/nci_isbi_2013/nifti/<case>/{t2.nii.gz, anatomy.nii.gz}
and a manifest. The mask keeps its source labels (1 and 2) here. Their meaning
(CG vs PZ) is checked anatomically by `--verify`, and the result is recorded
in the manifest, rather than assumed from a third-party description.

Usage:
    python scripts/prepare_nci_isbi.py
"""
from __future__ import annotations

import glob
import json
import os
import zipfile

import nibabel as nib
import numpy as np
import SimpleITK as sitk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "dataset/external/nci_isbi_2013")
ZIPS = os.path.join(BASE, "download/series_zip")
DICOM = os.path.join(BASE, "dicom")
MASKS = os.path.join(BASE, "masks")
OUT = os.path.join(BASE, "nifti")


def mask_index() -> dict:
    idx = {}
    for f in glob.glob(os.path.join(MASKS, "*", "*.nrrd")):
        name = os.path.basename(f).replace("_truth", "").replace("_correctedLabels", "").replace(".nrrd", "")
        idx[name] = f
    return idx


def extract(zpath: str, dest: str) -> None:
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():
            # flat archives expected; refuse anything that escapes the folder
            if os.path.isabs(m) or ".." in m.replace("\\", "/").split("/"):
                raise ValueError(f"unsafe path {m} in {zpath}")
        z.extractall(dest)


def world_centroid(arr_zyx: np.ndarray, img: sitk.Image, label: int):
    idx = np.argwhere(arr_zyx == label)
    if idx.size == 0:
        return None
    c = idx.mean(axis=0)[::-1]  # z,y,x -> x,y,z (ITK index order)
    return np.array(img.TransformContinuousIndexToPhysicalPoint(c.tolist()))  # LPS mm


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    masks = mask_index()
    manifest = []
    for zpath in sorted(glob.glob(os.path.join(ZIPS, "*.zip"))):
        uid = os.path.basename(zpath)[:-4]
        ddir = os.path.join(DICOM, uid)
        if not os.path.isdir(ddir):
            extract(zpath, ddir)
        reader = sitk.ImageSeriesReader()
        files = reader.GetGDCMSeriesFileNames(ddir)
        reader.SetFileNames(files)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        img = reader.Execute()
        pid = reader.GetMetaData(0, "0010|0020").strip()
        rec = {"series_uid": uid, "patient_id": pid, "n_slices": len(files)}
        if pid not in masks:
            rec["status"] = "no mask"
            manifest.append(rec)
            continue
        m = sitk.ReadImage(masks[pid])
        same_size = (m.GetSize() == img.GetSize()
                     and np.allclose(m.GetSpacing(), img.GetSpacing(), atol=1e-3)
                     and np.allclose(m.GetDirection(), img.GetDirection(), atol=1e-4))
        if not same_size:
            # The mask was drawn on a different series (different slice count or
            # thickness). There is no voxel correspondence to trust: exclude.
            rec["status"] = "excluded: mask grid differs from the image series"
            rec["mask_size"], rec["image_size"] = list(m.GetSize()), list(img.GetSize())
            manifest.append(rec)
            print(pid, rec["status"])
            continue
        offset = np.array(img.TransformPhysicalPointToContinuousIndex(m.GetOrigin()))
        rec["mask_origin_offset_vox"] = [round(float(v), 3) for v in offset]
        m_arr = sitk.GetArrayFromImage(m)
        if np.abs(offset).max() > 1e-3:
            # Same size, spacing and direction, origin offset only. Measure what a
            # physical-space resample would have done, then keep the voxel-aligned
            # copy (the annotation was drawn on these voxels).
            naive = sitk.GetArrayFromImage(
                sitk.Resample(m, img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, m.GetPixelID()))
            for lab in (1, 2):
                a, b = m_arr == lab, naive == lab
                s = a.sum() + b.sum()
                rec[f"naive_resample_dice_label{lab}"] = float(2 * (a & b).sum() / s) if s else 1.0
            rec["mask_origin_fixed_by_voxel_copy"] = True
        m = sitk.GetImageFromArray(m_arr)
        m.CopyInformation(img)
        m = sitk.Cast(m, sitk.sitkUInt8)
        arr = sitk.GetArrayFromImage(m)
        c1, c2 = world_centroid(arr, m, 1), world_centroid(arr, m, 2)
        # LPS: larger y = more posterior
        rec["label2_posterior_to_label1"] = (bool(c2[1] > c1[1]) if c1 is not None and c2 is not None else None)
        rec["labels"] = sorted(int(v) for v in np.unique(arr))
        case_dir = os.path.join(OUT, pid)
        os.makedirs(case_dir, exist_ok=True)
        sitk.WriteImage(img, os.path.join(case_dir, "t2.nii.gz"))
        sitk.WriteImage(m, os.path.join(case_dir, "anatomy.nii.gz"))
        a = nib.load(os.path.join(case_dir, "t2.nii.gz"))
        b = nib.load(os.path.join(case_dir, "anatomy.nii.gz"))
        rec["nifti_grid_match"] = bool(a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-3))
        rec["orientation"] = "".join(nib.aff2axcodes(a.affine))
        rec["spacing_mm"] = [round(float(z), 4) for z in a.header.get_zooms()[:3]]
        rec["shape"] = list(a.shape)
        rec["site"] = "Radboud 3T" if pid.startswith("Prostate3T") else "BMC 1.5T endorectal"
        rec["status"] = "ok"
        manifest.append(rec)
        print(pid, rec["orientation"], rec["shape"], rec["labels"], "L2 posterior:", rec["label2_posterior_to_label1"],
              rec.get("mask_origin_offset_vox"), rec.get("naive_resample_dice_label1", ""))
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    ok = [r for r in manifest if r["status"] == "ok"]
    post = [r["label2_posterior_to_label1"] for r in ok if r["label2_posterior_to_label1"] is not None]
    fixed = [r for r in ok if r.get("mask_origin_fixed_by_voxel_copy")]
    print(f"\n{len(ok)} cases converted; label 2 posterior to label 1 in {sum(post)}/{len(post)}; "
          f"{len(fixed)} masks with an origin offset fixed by voxel copy; "
          f"{sum(1 for r in manifest if r['status'] != 'ok')} excluded")
    for lab in (1, 2):
        d = [r[f"naive_resample_dice_label{lab}"] for r in fixed]
        if d:
            print(f"  physical resample instead of voxel copy, label {lab}: Dice mean {np.mean(d):.4f} "
                  f"min {np.min(d):.4f} (n={len(d)})")


if __name__ == "__main__":
    main()
