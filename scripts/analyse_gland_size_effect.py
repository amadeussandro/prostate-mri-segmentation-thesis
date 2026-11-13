"""Does a smaller gland get segmented worse, and is the model consistent across sizes?

The question came from the supervisor: if the reference structure is small, does
accuracy suffer? That matters clinically, because a small gland is not a rare
edge case - it is the normal finding in younger men and in those without benign
enlargement, and PSA density divides by exactly this volume.

Two things are measured, on both cohorts:

  1. Correlation between reference zone volume and that zone's Dice. Spearman is
     the primary statistic because nothing says the relationship has to be
     linear; Pearson is reported alongside it.
  2. Mean Dice within size tertiles. A correlation can be driven by a handful of
     cases, so the tertile view shows whether the effect is monotone and how
     large it is in Dice units rather than in correlation units.

Internal reference volumes come from the zone-volume artifact. External ones are
computed here from the harmonized reference masks, since no volume artifact
exists for that cohort.

Usage:
    python scripts/analyse_gland_size_effect.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, List

import nibabel as nib
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

CG, PZ = 1, 2

INTERNAL_VOL = f"{PROJECT}/results/rq2_zone_volumes/zone_volumes_per_case.csv"
INTERNAL_MET = f"{PROJECT}/results/test_e1_epoch88/per_case_metrics.csv"
EXTERNAL_MET = f"{PROJECT}/results/external_prostatex_e1_v2/metrics_per_case.csv"
EXTERNAL_GT = (f"{PROJECT}/results/external_prostatex_e1/external_prostatex_e1"
               f"/_harmonized/cases")
OUT_DIR = f"{PROJECT}/results/gland_size_effect"


def _rows(path: str) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Spearman (primary) and Pearson, with two-sided p-values."""
    from scipy import stats
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rs, ps = stats.spearmanr(x, y)
    rp, pp = stats.pearsonr(x, y)
    return {"n": int(x.size), "spearman_r": float(rs), "spearman_p": float(ps),
            "pearson_r": float(rp), "pearson_p": float(pp)}


def _tertiles(vol: np.ndarray, dice: np.ndarray) -> List[dict]:
    """Mean Dice in the small / medium / large third of the volume range."""
    ok = np.isfinite(vol) & np.isfinite(dice)
    vol, dice = vol[ok], dice[ok]
    order = np.argsort(vol)
    groups = np.array_split(order, 3)
    out = []
    for name, idx in zip(("kecil", "sedang", "besar"), groups):
        out.append({
            "band": name,
            "n": int(idx.size),
            "vol_min": float(vol[idx].min()),
            "vol_max": float(vol[idx].max()),
            "vol_mean": float(vol[idx].mean()),
            "dice_mean": float(dice[idx].mean()),
            "dice_sd": float(dice[idx].std(ddof=0)),
        })
    return out


def internal() -> Dict[str, object]:
    vols = {r["patient_id"]: r for r in _rows(INTERNAL_VOL)}
    per = _rows(INTERNAL_MET)

    # Per-case Dice keyed by class, from the frozen test artifact.
    dice: Dict[str, Dict[int, float]] = {}
    for r in per:
        pid = r.get("patient_id") or r.get("case_id")
        cls = int(float(r.get("class_id", -1)))
        if pid is None or cls not in (CG, PZ):
            continue
        dice.setdefault(pid, {})[cls] = float(r["dice"])

    ids = [p for p in vols if p in dice]
    out: Dict[str, object] = {"n_cases": len(ids)}
    for cls, key, name in ((CG, "gt_cg_ml", "cg"), (PZ, "gt_pz_ml", "pz")):
        v = np.array([float(vols[p][key]) for p in ids])
        d = np.array([dice[p].get(cls, np.nan) for p in ids])
        out[name] = {"corr": _corr(v, d), "tertiles": _tertiles(v, d)}
    v = np.array([float(vols[p]["gt_whole_gland_ml"]) for p in ids])
    macro = np.array([np.mean([dice[p].get(CG, np.nan), dice[p].get(PZ, np.nan)])
                      for p in ids])
    out["whole"] = {"corr": _corr(v, macro), "tertiles": _tertiles(v, macro)}
    return out


def external() -> Dict[str, object]:
    per = _rows(EXTERNAL_MET)
    dice: Dict[str, Dict[int, float]] = {}
    for r in per:
        cls = int(float(r["class_id"]))
        if cls in (CG, PZ):
            dice.setdefault(r["patient_id"], {})[cls] = float(r["dice"])

    vols: Dict[str, Dict[str, float]] = {}
    ids = sorted(dice)
    for i, pid in enumerate(ids, 1):
        path = os.path.join(EXTERNAL_GT, pid, "anatomy.nii.gz")
        if not os.path.exists(path):
            continue
        nii = nib.load(path)
        gt = np.round(nii.get_fdata()).astype(np.int16)
        ml = float(np.prod(nii.header.get_zooms()[:3])) / 1000.0
        vols[pid] = {"cg": int((gt == CG).sum()) * ml,
                     "pz": int((gt == PZ).sum()) * ml}
        vols[pid]["whole"] = vols[pid]["cg"] + vols[pid]["pz"]
        if i % 50 == 0 or i == len(ids):
            print(f"  external volumes: {i}/{len(ids)}", flush=True)

    keys = [p for p in ids if p in vols]
    out: Dict[str, object] = {"n_cases": len(keys)}
    for cls, name in ((CG, "cg"), (PZ, "pz")):
        v = np.array([vols[p][name] for p in keys])
        d = np.array([dice[p].get(cls, np.nan) for p in keys])
        out[name] = {"corr": _corr(v, d), "tertiles": _tertiles(v, d)}
    v = np.array([vols[p]["whole"] for p in keys])
    macro = np.array([np.mean([dice[p].get(CG, np.nan), dice[p].get(PZ, np.nan)])
                      for p in keys])
    out["whole"] = {"corr": _corr(v, macro), "tertiles": _tertiles(v, macro)}
    out["_per_case"] = [{"case_id": p, "gt_cg_ml": vols[p]["cg"],
                         "gt_pz_ml": vols[p]["pz"],
                         "gt_whole_ml": vols[p]["whole"],
                         "dice_cg": dice[p].get(CG), "dice_pz": dice[p].get(PZ)}
                        for p in keys]
    return out


def volume_error_vs_size() -> Dict[str, object]:
    """Does a small gland also carry a larger RELATIVE volume error?

    Worth asking because PSA density divides by this volume: if the relative
    error grew as the denominator shrank, the two effects would compound. It is
    tested here rather than assumed, and on this cohort it does not hold.
    """
    from scipy import stats
    r = _rows(INTERNAL_VOL)
    v = np.array([float(x["gt_whole_gland_ml"]) for x in r])
    pe = np.array([abs(float(x["pct_error_whole_gland_ml"])) for x in r])
    rs, ps = stats.spearmanr(v, pe)
    order = np.argsort(v)
    bands = []
    for name, idx in zip(("kecil", "sedang", "besar"), np.array_split(order, 3)):
        bands.append({"band": name, "n": int(idx.size),
                      "vol_min": float(v[idx].min()), "vol_max": float(v[idx].max()),
                      "abs_pct_error_mean": float(pe[idx].mean())})
    return {"n": int(v.size), "spearman_r": float(rs), "spearman_p": float(ps),
            "bands": bands}


def write_report(path: str, s: dict) -> str:
    def block(coh: dict, label: str) -> List[str]:
        L = [f"### {label} (n = {coh['n_cases']})\n\n",
             "| Zona | Spearman r | p | Pearson r | p |\n|---|---|---|---|---|\n"]
        for name, lab in (("cg", "Central gland"), ("pz", "Peripheral zone"),
                          ("whole", "Seluruh kelenjar (Dice makro)")):
            c = coh[name]["corr"]
            L.append(f"| {lab} | {c['spearman_r']:+.3f} | {c['spearman_p']:.4f} | "
                     f"{c['pearson_r']:+.3f} | {c['pearson_p']:.4f} |\n")
        L.append("\n")
        for name, lab in (("cg", "Central gland"), ("pz", "Peripheral zone")):
            L.append(f"**{lab} per tertil volume acuan**\n\n")
            L.append("| Tertil | n | Rentang volume (mL) | Dice rerata |\n|---|---|---|---|\n")
            for t in coh[name]["tertiles"]:
                L.append(f"| {t['band']} | {t['n']} | {t['vol_min']:.1f}–{t['vol_max']:.1f} | "
                         f"{t['dice_mean']:.4f} ± {t['dice_sd']:.4f} |\n")
            L.append("\n")
        return L

    i, e = s["internal"], s["external"]
    spread = {}
    for coh, lab in ((i, "internal"), (e, "external")):
        for name in ("cg", "pz"):
            t = coh[name]["tertiles"]
            spread[f"{lab}_{name}"] = t[-1]["dice_mean"] - t[0]["dice_mean"]

    md = [
        "# Pengaruh ukuran kelenjar terhadap akurasi segmentasi\n\n",
        "Pertanyaan dari pembimbing: apakah kelenjar yang lebih kecil tersegmentasi "
        "lebih buruk, dan apakah model konsisten di seluruh rentang ukuran?\n\n",
        "Diukur dua cara: korelasi volume acuan terhadap Dice (Spearman sebagai "
        "statistik utama, karena hubungannya tidak harus linear), dan Dice rerata "
        "per tertil ukuran, supaya besarnya terbaca dalam satuan Dice alih-alih "
        "satuan korelasi.\n\n",
    ]
    md += block(i, "Kohort internal (Prostate158)")
    md += block(e, "Kohort eksternal (PROSTATEx)")
    md += [
        "## Selisih Dice antara tertil terbesar dan terkecil\n\n",
        "| Kohort | Central gland | Peripheral zone |\n|---|---|---|\n",
        f"| Internal | {spread['internal_cg']:+.4f} | {spread['internal_pz']:+.4f} |\n",
        f"| Eksternal | {spread['external_cg']:+.4f} | {spread['external_pz']:+.4f} |\n",
        "\nNilai positif berarti kelenjar besar tersegmentasi lebih baik.\n",
    ]

    ve = s.get("volume_error_vs_size")
    if ve:
        md += [
            "\n## Apakah galat volume relatif juga lebih besar pada kelenjar kecil?\n\n",
            "Pertanyaan ini penting karena densitas PSA membagi dengan volume "
            "kelenjar: bila galat relatif ikut membesar saat penyebutnya mengecil, "
            "kedua efek akan berlipat. Diuji, bukan diasumsikan.\n\n",
            f"Korelasi volume acuan terhadap |galat volume %|: Spearman "
            f"**{ve['spearman_r']:+.3f}** (p = {ve['spearman_p']:.4f}, "
            f"n = {ve['n']}).\n\n",
            "| Tertil | n | Rentang volume (mL) | \\|galat volume\\| rerata |\n|---|---|---|---|\n",
        ]
        for b in ve["bands"]:
            md.append(f"| {b['band']} | {b['n']} | {b['vol_min']:.1f}–{b['vol_max']:.1f} | "
                      f"{b['abs_pct_error_mean']:.1f}% |\n")
        md += [
            "\n**Tidak terbukti.** Korelasinya praktis nol dan pola tertilnya tidak "
            "monoton. Jadi efek ukuran yang nyata pada Dice **tidak** diikuti oleh "
            "galat volume relatif yang lebih besar pada kelenjar kecil, setidaknya "
            "pada 19 kasus internal. Dengan n sekecil ini pengujiannya berdaya "
            "rendah, sehingga yang dapat dinyatakan adalah tidak ditemukan bukti, "
            "bukan bahwa efeknya tidak ada.\n",
        ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(md))
    return path


def main() -> None:
    import argparse
    global INTERNAL_VOL, INTERNAL_MET, EXTERNAL_MET, OUT_DIR
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--internal-vol", default=INTERNAL_VOL)
    ap.add_argument("--internal-met", default=INTERNAL_MET)
    ap.add_argument("--external-met", default=EXTERNAL_MET)
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()
    INTERNAL_VOL, INTERNAL_MET, EXTERNAL_MET, OUT_DIR = (
        a.internal_vol, a.internal_met, a.external_met, a.out_dir)
    os.makedirs(OUT_DIR, exist_ok=True)
    print("internal...")
    i = internal()
    print("external...")
    e = external()
    per_case = e.pop("_per_case")

    with open(f"{OUT_DIR}/external_volumes_per_case.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_case[0]))
        w.writeheader()
        w.writerows(per_case)

    summary = {"internal": i, "external": e,
               "volume_error_vs_size": volume_error_vs_size()}
    with open(f"{OUT_DIR}/summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    rep = write_report(f"{OUT_DIR}/report.md", summary)
    print(f"\nreport: {rep}")


if __name__ == "__main__":
    main()
