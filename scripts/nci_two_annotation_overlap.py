"""Model vs two independent annotations on the 18 NCI-ISBI images also in MSD Task05.

Reproduces the ad hoc computation behind results/external_overlap/model_vs_two_annotations.json
as a script, so it can be repeated for any model's NCI-ISBI reconstructions.
MSD labels are 1=PZ, 2=TZ; they are swapped to the model's 1=CG, 2=PZ.

    python scripts/nci_two_annotation_overlap.py \
        --nci-dir results/external_nci_isbi_final/nci \
        --out results/external_overlap_final/model_vs_two_annotations.json
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import nibabel as nib
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.metrics import dice_score  # noqa: E402

COLS = ["nci_vs_msd_cg", "nci_vs_msd_pz", "model_vs_nci_cg", "model_vs_nci_pz",
        "model_vs_msd_cg", "model_vs_msd_pz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nci-dir", required=True, help="folder with per_case.csv and reconstructions_3d/")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {r["case_id"]: r for r in csv.DictReader(open(os.path.join(a.nci_dir, "per_case.csv")))}
    pairs = [x for x in json.load(open(os.path.join(ROOT, "results/external_overlap/nci_vs_msd_labels.json")))
             if x["nci"] != "Prostate3T-01-0019"]
    msd = {os.path.basename(f)[:-7]: f for f in
           glob.glob(os.path.join(ROOT, "dataset/external/msd_task05/**/labelsTr/prostate_*.nii.gz"), recursive=True)}
    res = []
    for x in pairs:
        p = nib.as_closest_canonical(nib.load(os.path.join(a.nci_dir, "reconstructions_3d", f"pred_{x['nci']}.nii.gz")))
        m = nib.as_closest_canonical(nib.load(msd[x["msd"]]))
        P, M = np.asarray(p.dataobj).astype(int), np.asarray(m.dataobj).astype(int)
        Mr = np.zeros_like(M)
        Mr[M == 1], Mr[M == 2] = 2, 1
        r = rows[x["nci"]]
        res.append((x["dice_TZ"], x["dice_PZ"], float(r["cg_dice_none"]), float(r["pz_dice_none"]),
                    dice_score(P, Mr, 1), dice_score(P, Mr, 2)))
    arr = np.array(res)
    print("n", len(arr))
    print("NCI vs MSD   CG %.3f PZ %.3f" % tuple(arr[:, :2].mean(0)))
    print("model vs NCI CG %.3f PZ %.3f" % tuple(arr[:, 2:4].mean(0)))
    print("model vs MSD CG %.3f PZ %.3f" % tuple(arr[:, 4:].mean(0)))
    for i, j, n in ((2, 0, "CG"), (3, 1, "PZ")):
        d = arr[:, i] - arr[:, j]
        print(n, "model(vs NCI) - inter-annotation", round(d.mean(), 3), "P=%.3f" % stats.wilcoxon(d).pvalue)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"n": len(arr), "cols": COLS, "rows": arr.tolist(), "nci_dir": a.nci_dir}, open(a.out, "w"))


if __name__ == "__main__":
    main()
