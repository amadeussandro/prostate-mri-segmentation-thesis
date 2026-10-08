"""Why does central-gland HD95 have a heavy tail on the external cohort?

The external result shows a median HD95 near the internal value but a mean
pulled far out by a minority of cases, while Dice stays high in exactly those
cases. That pattern is what small volumes of prediction far from the gland look
like: too little volume to move an overlap metric, far enough to dominate a
boundary metric. The manuscript states that as a hypothesis. This tests it.

The test is a connected-component decomposition of each predicted class. If the
hypothesis holds then (a) the affected cases carry more than one component,
(b) the extra components are small and distant, and (c) discarding everything
but the largest component collapses HD95 while leaving Dice essentially intact.

Point (c) is the decisive one, and it is also why this is a diagnostic rather
than a proposed post-processing step: keeping only the largest component would
be a change to the method, and choosing it after seeing the external scores
would be tuning on the external cohort. It is reported here as an explanation
of the tail, not as a result.

Usage:
    python scripts/analyse_external_hd95_tail.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, PROJECT)

from src.metrics import dice_score, hausdorff_distance_95  # noqa: E402

CG, PZ = 1, 2


def largest_component_only(mask: np.ndarray) -> np.ndarray:
    """Keep the largest 6-connected component of a binary mask."""
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def component_profile(mask: np.ndarray, spacing) -> Dict[str, float]:
    """Describe a class's connected components: how many, how big, how far."""
    lab, n = ndimage.label(mask)
    if n == 0:
        return {"n_components": 0, "largest_frac": float("nan"),
                "n_satellites": 0, "max_satellite_dist_mm": 0.0,
                "satellite_vol_frac": 0.0}
    sizes = np.array(ndimage.sum(mask, lab, range(1, n + 1)))
    main = int(np.argmax(sizes)) + 1
    total = float(sizes.sum())
    sp = np.asarray(spacing[:3], dtype=float)
    main_centroid = np.array(ndimage.center_of_mass(lab == main)) * sp

    far, sat_vox = 0.0, 0.0
    for c in range(1, n + 1):
        if c == main:
            continue
        sat_vox += float(sizes[c - 1])
        cen = np.array(ndimage.center_of_mass(lab == c)) * sp
        far = max(far, float(np.linalg.norm(cen - main_centroid)))
    return {
        "n_components": int(n),
        "largest_frac": float(sizes[main - 1] / total),
        "n_satellites": int(n - 1),
        "max_satellite_dist_mm": far,
        "satellite_vol_frac": float(sat_vox / total),
    }


def run(recon_dir: str, gt_root: str, out_dir: str,
        mask_name: str = "anatomy.nii.gz") -> dict:
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(recon_dir) if f.endswith(".nii.gz"))
    if not files:
        raise FileNotFoundError(f"no reconstructions in {recon_dir}")

    rows: List[dict] = []
    for i, fn in enumerate(files, 1):
        cid = fn[len("pred_"):-len(".nii.gz")]
        nii = nib.load(os.path.join(recon_dir, fn))
        pred = np.asarray(nii.dataobj)
        spacing = nii.header.get_zooms()[:3]
        gt_path = os.path.join(gt_root, cid, mask_name)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"{cid}: no ground truth at {gt_path}")
        gt = np.round(nib.load(gt_path).get_fdata()).astype(np.int64)

        row = {"case_id": cid}
        for cls, name in ((CG, "cg"), (PZ, "pz")):
            m = pred == cls
            prof = component_profile(m, spacing)
            cleaned = np.where(largest_component_only(m), cls, 0).astype(pred.dtype)
            row.update({f"{name}_{k}": v for k, v in prof.items()})
            row[f"{name}_hd95"] = hausdorff_distance_95(pred, gt, spacing, cls)
            row[f"{name}_hd95_largest_only"] = hausdorff_distance_95(cleaned, gt, spacing, cls)
            row[f"{name}_dice"] = dice_score(pred, gt, cls)
            row[f"{name}_dice_largest_only"] = dice_score(cleaned, gt, cls)
        rows.append(row)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} cases", flush=True)

    per_case = os.path.join(out_dir, "hd95_tail_per_case.csv")
    with open(per_case, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    summary = summarise(rows)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    report = write_report(os.path.join(out_dir, "report.md"), summary)
    return {"per_case": per_case, "report": report, "summary": summary, "n": len(rows)}


def summarise(rows: List[dict], tail_mm: float = 30.0) -> dict:
    def arr(k):
        return np.array([float(r[k]) for r in rows], dtype=float)

    hd = arr("cg_hd95")
    hd_clean = arr("cg_hd95_largest_only")
    dice, dice_clean = arr("cg_dice"), arr("cg_dice_largest_only")
    ncomp = arr("cg_n_components")
    tail = hd > tail_mm

    def block(sel, label):
        return {
            "label": label, "n": int(sel.sum()),
            "hd95_mean": float(hd[sel].mean()) if sel.any() else float("nan"),
            "hd95_after_mean": float(hd_clean[sel].mean()) if sel.any() else float("nan"),
            "dice_mean": float(dice[sel].mean()) if sel.any() else float("nan"),
            "dice_after_mean": float(dice_clean[sel].mean()) if sel.any() else float("nan"),
            "components_mean": float(ncomp[sel].mean()) if sel.any() else float("nan"),
            "multi_component_frac": float((ncomp[sel] > 1).mean()) if sel.any() else float("nan"),
            "satellite_vol_frac_mean": float(arr("cg_satellite_vol_frac")[sel].mean())
            if sel.any() else float("nan"),
            "max_satellite_dist_mean": float(arr("cg_max_satellite_dist_mm")[sel].mean())
            if sel.any() else float("nan"),
        }

    return {
        "n_cases": len(rows), "tail_threshold_mm": tail_mm,
        "tail": block(tail, f"HD95 > {tail_mm:.0f} mm"),
        "rest": block(~tail, f"HD95 <= {tail_mm:.0f} mm"),
        "overall": {
            "hd95_mean": float(hd.mean()), "hd95_median": float(np.median(hd)),
            "hd95_after_mean": float(hd_clean.mean()),
            "hd95_after_median": float(np.median(hd_clean)),
            "dice_mean": float(dice.mean()), "dice_after_mean": float(dice_clean.mean()),
            "multi_component_frac": float((ncomp > 1).mean()),
            "corr_components_hd95": float(np.corrcoef(ncomp, hd)[0, 1]),
        },
    }


def write_report(path: str, s: dict) -> str:
    o, t, r = s["overall"], s["tail"], s["rest"]
    # The hypothesis predicts that removing the satellites collapses the boundary
    # metric without costing overlap. A small Dice *gain* is part of that
    # prediction, not evidence against it: the satellites are false positives, so
    # dropping them can only help overlap slightly. The test is therefore that
    # HD95 falls sharply and Dice does not degrade.
    hd_collapsed = o["hd95_after_mean"] < o["hd95_mean"] * 0.6
    dice_not_hurt = o["dice_after_mean"] >= o["dice_mean"] - 0.005
    verdict = "SUPPORTED" if (hd_collapsed and dice_not_hurt) else "NOT SUPPORTED"
    md = [
        "# Why central-gland HD95 has a heavy tail on PROSTATEx\n",
        f"_{s['n_cases']} cases. Diagnostic only: nothing here changes the reported method._\n",
        "## Hypothesis\n",
        "The tail is caused by small volumes of predicted central gland lying far from the "
        "gland, not by a boundary that is uniformly wrong. Such components carry too little "
        "volume to move Dice but dominate a 95th-percentile boundary distance.\n",
        "## Test\n",
        "Decompose each predicted class into connected components, then discard everything "
        "but the largest and re-measure. If the hypothesis holds, HD95 collapses while Dice "
        "barely moves.\n",
        "## Result\n",
        "| | All cases | Tail (" + t["label"] + ") | Rest |\n|---|---|---|---|\n",
        f"| Cases | {s['n_cases']} | {t['n']} | {r['n']} |\n",
        f"| HD95 (mm) | {o['hd95_mean']:.1f} | {t['hd95_mean']:.1f} | {r['hd95_mean']:.1f} |\n",
        f"| HD95, largest component only | {o['hd95_after_mean']:.1f} | "
        f"{t['hd95_after_mean']:.1f} | {r['hd95_after_mean']:.1f} |\n",
        f"| Dice | {o['dice_mean']:.4f} | {t['dice_mean']:.4f} | {r['dice_mean']:.4f} |\n",
        f"| Dice, largest component only | {o['dice_after_mean']:.4f} | "
        f"{t['dice_after_mean']:.4f} | {r['dice_after_mean']:.4f} |\n",
        f"| Mean components | {t['components_mean']:.2f} (tail) | {t['components_mean']:.2f} | "
        f"{r['components_mean']:.2f} |\n",
        f"| Cases with >1 component | {o['multi_component_frac']*100:.0f}% | "
        f"{t['multi_component_frac']*100:.0f}% | {r['multi_component_frac']*100:.0f}% |\n",
        f"| Satellite volume share | — | {t['satellite_vol_frac_mean']*100:.1f}% | "
        f"{r['satellite_vol_frac_mean']*100:.1f}% |\n",
        f"| Mean distance to farthest satellite | — | {t['max_satellite_dist_mean']:.1f} mm | "
        f"{r['max_satellite_dist_mean']:.1f} mm |\n",
        f"\n## Verdict: {verdict}\n",
        f"Discarding all but the largest component takes mean HD95 from {o['hd95_mean']:.1f} mm "
        f"to {o['hd95_after_mean']:.1f} mm while mean Dice moves from {o['dice_mean']:.4f} to "
        f"{o['dice_after_mean']:.4f}. Correlation between component count and HD95 is "
        f"{o['corr_components_hd95']:+.2f}.\n",
        "\n## Why this is not applied\n",
        "Keeping only the largest component would be a change to the method, and selecting it "
        "after seeing the external scores would be tuning on the external cohort. It is used "
        "here to explain the tail, not to improve the reported numbers. Adopting it would "
        "require pre-specifying it and validating it on the internal validation split first.\n",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        # Each entry already carries its own newline; joining with another one
        # would put a blank line between table rows and break the table.
        fh.write("".join(md))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose the external HD95 tail (CPU, no inference).")
    ap.add_argument("--recon-dir",
                    default="results/external_prostatex_e1_v2/reconstructions_3d")
    ap.add_argument("--gt-root",
                    default="results/external_prostatex_e1/external_prostatex_e1/_harmonized/cases")
    ap.add_argument("--out-dir", default="results/external_hd95_tail")
    args = ap.parse_args()
    res = run(args.recon_dir, args.gt_root, args.out_dir)
    print(f"\nDone over {res['n']} cases.")
    print(f"  per case : {res['per_case']}")
    print(f"  report   : {res['report']}")


if __name__ == "__main__":
    main()
