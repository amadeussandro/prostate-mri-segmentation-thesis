"""Inter-reader benchmark for the JMIR manuscript (CPU only, no training).

Compares, per held-out test case and in original voxel space:
  - reader 1 vs reader 2   (human inter-reader agreement)
  - model    vs reader 1   (the reported result)
  - model    vs reader 2   (model against the second expert)
Also reports zone volumes for all three.

Usage:
    python scripts/jmir_reader2_benchmark.py
Outputs go to results/jmir_reader2/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.metrics import (  # noqa: E402
    average_surface_distance,
    dice_score,
    hausdorff_distance_95,
)

TEST_DIR = Path("G:/My Drive/THESIS_PROSTATE158/dataset/test/extracted/prostate158_test/test")
PRED_DIR = ROOT / "results/rq2_e1_epoch88/reconstructions_3d"
OUT_DIR = ROOT / "results/jmir_reader2"
CLASSES = {1: "CG", 2: "PZ"}
PAIRS = [("r1", "r2"), ("model", "r1"), ("model", "r2")]


def voxel_ml(img: nib.Nifti1Image) -> float:
    return abs(np.linalg.det(img.affine[:3, :3])) / 1000.0


def load(path: Path):
    img = nib.load(str(path))
    return img, np.asarray(img.dataobj).astype(np.int16)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_dir in sorted(p for p in TEST_DIR.iterdir() if p.is_dir()):
        cid = case_dir.name
        r1_img, r1 = load(case_dir / "t2_anatomy_reader1.nii.gz")
        r2_path = case_dir / "t2_anatomy_reader2.nii.gz"
        pred_path = PRED_DIR / f"pred_{cid}.nii.gz"
        if not r2_path.exists() or not pred_path.exists():
            print(f"skip {cid}: missing reader 2 or prediction")
            continue
        r2_img, r2 = load(r2_path)
        p_img, pred = load(pred_path)
        assert r1.shape == r2.shape == pred.shape, cid
        assert np.allclose(r1_img.affine, r2_img.affine) and np.allclose(r1_img.affine, p_img.affine), cid
        spacing = r1_img.header.get_zooms()[:3]
        vml = voxel_ml(r1_img)
        masks = {"r1": r1, "r2": r2, "model": pred}
        row = {"case": cid}
        for c, name in CLASSES.items():
            for src in masks:
                row[f"vol_{name}_{src}"] = float((masks[src] == c).sum() * vml)
            for a, b in PAIRS:
                key = f"{a}_vs_{b}"
                row[f"dice_{name}_{key}"] = dice_score(masks[a], masks[b], c)
                row[f"hd95_{name}_{key}"] = hausdorff_distance_95(masks[a], masks[b], spacing, c)
                row[f"asd_{name}_{key}"] = average_surface_distance(masks[a], masks[b], spacing, c)
        for src in masks:
            row[f"vol_WG_{src}"] = row[f"vol_CG_{src}"] + row[f"vol_PZ_{src}"]
        rows.append(row)
        print(f"{cid}: CG r1-r2 {row['dice_CG_r1_vs_r2']:.3f} model-r1 {row['dice_CG_model_vs_r1']:.3f} | "
              f"PZ r1-r2 {row['dice_PZ_r1_vs_r2']:.3f} model-r1 {row['dice_PZ_model_vs_r1']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "per_case.csv", index=False)

    summary = {"n_cases": int(len(df))}
    for name in CLASSES.values():
        for metric in ["dice", "hd95", "asd"]:
            for a, b in PAIRS:
                col = f"{metric}_{name}_{a}_vs_{b}"
                v = df[col].dropna()
                summary[col] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "median": float(v.median())}
        # model vs reader 1 against human agreement: paired difference
        for metric in ["dice", "hd95"]:
            m = df[f"{metric}_{name}_model_vs_r1"]
            h = df[f"{metric}_{name}_r1_vs_r2"]
            d = (m - h).dropna()
            se = d.std(ddof=1) / np.sqrt(len(d))
            ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=se)
            summary[f"{metric}_{name}_model_minus_interreader"] = {
                "mean_diff": float(d.mean()),
                "ci95": [float(ci[0]), float(ci[1])],
                "wilcoxon_p": float(stats.wilcoxon(m, h).pvalue),
                "ratio_of_means": float(m.mean() / h.mean()),
            }
    # volume agreement: reader 2 vs reader 1 and model vs reader 1 (Bland-Altman)
    for name in ["CG", "PZ", "WG"]:
        for src in ["r2", "model"]:
            diff = df[f"vol_{name}_{src}"] - df[f"vol_{name}_r1"]
            sd = diff.std(ddof=1)
            summary[f"vol_{name}_{src}_minus_r1"] = {
                "bias_ml": float(diff.mean()),
                "loa_ml": [float(diff.mean() - 1.96 * sd), float(diff.mean() + 1.96 * sd)],
                "bias_pct": float((diff / df[f"vol_{name}_r1"]).mean() * 100),
            }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", default=str(PRED_DIR), help="model reconstructions (pred_<id>.nii.gz)")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    a = ap.parse_args()
    PRED_DIR, OUT_DIR = Path(a.pred_dir), Path(a.out_dir)
    main()
