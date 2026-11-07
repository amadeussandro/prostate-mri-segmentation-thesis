"""Numbers for the JMIR revision, layered on top of `manuscript_numbers`.

`manuscript_numbers.all_numbers()` stays untouched because the thesis build
(Indonesian) still reads it. This module adds what the revision needs and
switches the reported dispersion to the SAMPLE standard deviation (ddof=1),
the convention JMIR reviewers expect. Every value is read from a result
artifact; nothing is typed in.

Sources added here:
  results/jmir_stats/            TOST, Holm, sample-SD test table
  results/jmir_reader2/          inter-reader benchmark
  results/modelfree_ablation/    4-dataset model-free ablation
  results/external_nci_isbi_e1/  second external cohort + pre-specified LCC
  results/external_overlap/      overlap between the external cohorts
  results/rq2_zone_volumes/      PSA-density sensitivity analysis
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Dict

import numpy as np

import manuscript_numbers as base

P = base.PROJECT


def _j(rel: str):
    with open(os.path.join(P, rel), encoding="utf-8") as fh:
        return json.load(fh)


def _rows(rel: str):
    with open(os.path.join(P, rel), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x))).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _desc(x) -> Dict[str, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    lo, hi = _boot(x)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "median": float(np.median(x)),
            "min": float(x.min()), "max": float(x.max()), "n": int(len(x)), "ci_low": lo, "ci_high": hi}


def test_sample_sd() -> Dict[str, dict]:
    """Held-out test metrics per zone, sample SD, bootstrap CI."""
    path = base.TEST_PER_CASE
    out = {}
    for name, cls in (("cg", base.CLASS_CG), ("pz", base.CLASS_PZ), ("bg", base.CLASS_BG)):
        out[name] = {m: _desc(base._per_case(path, cls, m))
                     for m in ("dice", "iou", "hd95_mm", "asd_mm", "precision", "recall")}
    cg = base._per_case(path, base.CLASS_CG, "dice")
    pz = base._per_case(path, base.CLASS_PZ, "dice")
    out["macro"] = _desc((cg + pz) / 2)
    return out


def external_sample_sd() -> Dict[str, dict]:
    """PROSTATEx metrics with sample SD (the stored summary uses ddof=0)."""
    path = os.path.join(P, "results/external_prostatex_e1/external_prostatex_e1/metrics_per_case.csv")
    out = {}
    for name, cls in (("cg", base.CLASS_CG), ("pz", base.CLASS_PZ)):
        out[name] = {m: _desc(base._per_case(path, cls, m))
                     for m in ("dice", "iou", "hd95_mm", "asd_mm", "precision", "recall")}
    cg = base._per_case(path, base.CLASS_CG, "dice")
    pz = base._per_case(path, base.CLASS_PZ, "dice")
    out["macro"] = _desc((cg + pz) / 2)
    out["n"] = int(len(cg))
    return out


def psa_sensitivity(threshold: float = 0.15, grey=(4.0, 10.0)) -> Dict[str, float]:
    """How often would the model's gland volume move PSA density across 0.15?

    No PSA values exist for these patients, so the question is asked over a
    range instead of for one value. A patient is reclassified at PSA x when
    x / V_ref and x / V_pred fall on opposite sides of the threshold, which
    happens for x between 0.15 * V_pred and 0.15 * V_ref. Reported: the width
    of that PSA interval per patient, and the share of (patient, PSA) pairs
    reclassified when PSA is spread evenly over the diagnostic grey zone.
    """
    rows = _rows("results/rq2_zone_volumes/zone_volumes_per_case.csv")
    gt = np.array([float(r["gt_whole_gland_ml"]) for r in rows])
    pr = np.array([float(r["pred_whole_gland_ml"]) for r in rows])
    lo, hi = np.minimum(gt, pr) * threshold, np.maximum(gt, pr) * threshold
    width = hi - lo
    g0, g1 = grey
    overlap = np.clip(np.minimum(hi, g1) - np.maximum(lo, g0), 0, None)
    frac = overlap / (g1 - g0)
    # PSA grid, 0.1 ng/mL steps across the grey zone: patients ever reclassified
    grid = np.arange(g0, g1 + 1e-9, 0.1)
    ever = [bool(((grid >= l) & (grid <= h)).any()) for l, h in zip(lo, hi)]
    up = int((pr < gt).sum())
    return {"n": int(len(gt)), "threshold": threshold, "grey": list(grey),
            "width_median": float(np.median(width)), "width_max": float(width.max()),
            "width_min": float(width.min()),
            "share_pairs_reclassified_pct": float(frac.mean() * 100),
            "n_ever_reclassified": int(sum(ever)),
            "n_overestimated": up}


def volumes_sample_sd() -> Dict[str, dict]:
    rows = _rows("results/rq2_zone_volumes/zone_volumes_per_case.csv")
    out = {"n": len(rows)}
    for key, name in (("cg_ml", "cg"), ("pz_ml", "pz"), ("whole_gland_ml", "whole")):
        gt = np.array([float(r[f"gt_{key}"]) for r in rows])
        pr = np.array([float(r[f"pred_{key}"]) for r in rows])
        d = pr - gt
        bias, sd = float(d.mean()), float(d.std(ddof=1))
        out[name] = {"gt_mean": float(gt.mean()), "gt_sd": float(gt.std(ddof=1)),
                     "pred_mean": float(pr.mean()), "pred_sd": float(pr.std(ddof=1)),
                     "bias": bias, "pct_of_gt": float(bias / gt.mean() * 100),
                     "pct_per_case_mean": float((d / gt).mean() * 100),
                     "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
                     "under_in": int((d < 0).sum())}
    ratio = np.array([float(r["gt_whole_gland_ml"]) for r in rows]) / \
        np.array([float(r["pred_whole_gland_ml"]) for r in rows])
    out["psad"] = {"mean_pct": float((ratio.mean() - 1) * 100), "max_pct": float((ratio.max() - 1) * 100),
                   "over_in": int((ratio > 1).sum())}
    return out


def modelfree() -> Dict[str, object]:
    s = _rows("results/modelfree_ablation/summary_all.csv")
    pc = _rows("results/modelfree_ablation/per_case_all.csv")
    gate = _j("results/modelfree_ablation/gate.json")
    gate_nci = _j("results/modelfree_ablation_nci/gate.json")
    out = {"summary": {}, "n_cases": {}}
    for r in s:
        d = out["summary"].setdefault(r["dataset"], {})
        d[r["variant"]] = {k: float(v) for k, v in r.items() if k not in ("dataset", "variant")}
        out["n_cases"][r["dataset"]] = int(float(r["n"]))
    v0 = [r for r in pc if r["variant"] == "V0_correct"]
    out["v0_n"] = len(v0)
    out["v0_roundtrip_exact"] = gate["v0_roundtrip_exact"] + gate_nci["v0_roundtrip_exact"]
    out["v0_L3a_false"] = gate["v0_L3a_false_alarms"] + gate_nci["v0_L3a_false_alarms"]
    out["v0_L3b_false"] = gate["v0_L3b_false_alarms"] + gate_nci["v0_L3b_false_alarms"]
    t = lambda x: str(x).strip().lower() in ("true", "1", "1.0")
    dmg, undmg = [], []
    for r in pc:
        if r["variant"] == "V0_correct":
            continue
        damaged = (float(r["dice_CG"]) < 1 or float(r["dice_PZ"]) < 1 or float(r["shift_CG_mm"]) > 0.01
                   or float(r["shift_PZ_mm"]) > 0.01 or abs(float(r["vol_err_CG_pct"])) > 0.01)
        (dmg if damaged else undmg).append(r)
    l12 = lambda r: t(r["L1_geometry_flags"]) or t(r["L2_roundtrip_flags"])
    l3 = lambda r: t(r["L3a_flags"]) or t(r["L3b_flags"])
    out["n_damaged"] = len(dmg)
    out["n_undamaged"] = len(undmg)
    out["damaged_caught_L1L2"] = sum(l12(r) for r in dmg)
    out["damaged_caught_L1"] = sum(t(r["L1_geometry_flags"]) for r in dmg)
    out["damaged_caught_L2"] = sum(t(r["L2_roundtrip_flags"]) for r in dmg)
    out["damaged_caught_L3"] = sum(l3(r) for r in dmg)
    out["undamaged_flagged_L1L2"] = sum(l12(r) for r in undmg)
    man = _j("dataset/external/nci_isbi_2013/nifti/manifest.json")
    fixed = [r for r in man if r.get("mask_origin_fixed_by_voxel_copy")]
    out["nci_halfvoxel"] = {
        "n": len(fixed),
        "pz_dice_mean": float(np.mean([r["naive_resample_dice_label1"] for r in fixed])),
        "pz_dice_min": float(np.min([r["naive_resample_dice_label1"] for r in fixed])),
        "cg_dice_mean": float(np.mean([r["naive_resample_dice_label2"] for r in fixed])),
        "cg_dice_min": float(np.min([r["naive_resample_dice_label2"] for r in fixed])),
    }
    return out


def nci() -> Dict[str, object]:
    s = _j("results/external_nci_isbi_e1/nci/summary.json")
    dec = _j("results/external_nci_isbi_e1/valid_prostate158/decision.json")
    rows = _rows("results/external_nci_isbi_e1/nci/per_case.csv")
    f = lambda k: np.array([float(r[k]) for r in rows])
    with open(os.path.join(P, "results/lcc_prespecification.md"), "rb") as fh:
        prespec_sha = hashlib.sha256(fh.read()).hexdigest()
    if prespec_sha != s["prespec_sha256"] or prespec_sha != dec["prespec_sha256"]:
        raise RuntimeError("pre-specification file no longer matches the hash recorded by the run")
    out = {"n": s["n_cases"], "geometry_ok": s["geometry_ok"], "decision": dec,
           "primary": s["primary"], "by_site": s["by_site"], "secondary": s["secondary"],
           "n_sites": {site: sum(r["site"] == site for r in rows) for site in s["by_site"]},
           "cg_tail_before": int((f("cg_hd95_none") > 30).sum()),
           "cg_tail_after": int((f("cg_hd95_R1") > 30).sum()),
           "cg_multicomp_pct": float((f("cg_n_components") > 1).mean() * 100),
           "macro_dice": _desc((f("cg_dice_none") + f("pz_dice_none")) / 2)}
    man = [r for r in _j("dataset/external/nci_isbi_2013/nifti/manifest.json") if r["status"] == "ok"]
    ids = {r["case_id"] for r in rows}
    sp = [r["spacing_mm"] for r in man if r["patient_id"] in ids]
    out["inplane_min"] = float(min(x[0] for x in sp))
    out["inplane_max"] = float(max(x[0] for x in sp))
    out["through_min"] = float(min(x[2] for x in sp))
    out["through_max"] = float(max(x[2] for x in sp))
    out["n_converted"] = len(man)
    out["n_total"] = len(_j("dataset/external/nci_isbi_2013/nifti/manifest.json"))
    for site in s["by_site"]:
        sub = [r for r in rows if r["site"] == site]
        out["by_site"][site]["cg_hd95_R1_mean"] = float(np.mean([float(r["cg_hd95_R1"]) for r in sub]))
    return out


def overlap() -> Dict[str, object]:
    ov = _j("results/external_overlap/nci_overlap.json")
    lab = [x for x in _j("results/external_overlap/nci_vs_msd_labels.json") if x["nci"] != "Prostate3T-01-0019"]
    two = _j("results/external_overlap/model_vs_two_annotations.json")
    a = np.array(two["rows"])
    from scipy.stats import wilcoxon
    return {"n_msd": sum(m["other"].startswith("MSD") for m in ov),
            "n_px": sum(m["other"].startswith("PX") for m in ov),
            "msd_vs_nci_cg": float(np.mean([x["dice_TZ"] for x in lab])),
            "msd_vs_nci_pz": float(np.mean([x["dice_PZ"] for x in lab])),
            "n_two": int(two["n"]),
            "model_vs_nci_cg": float(a[:, 2].mean()), "model_vs_nci_pz": float(a[:, 3].mean()),
            "model_vs_msd_cg": float(a[:, 4].mean()), "model_vs_msd_pz": float(a[:, 5].mean()),
            "gap_cg": float((a[:, 2] - a[:, 0]).mean()), "gap_pz": float((a[:, 3] - a[:, 1]).mean()),
            "gap_cg_p": float(wilcoxon(a[:, 2] - a[:, 0]).pvalue),
            "gap_pz_p": float(wilcoxon(a[:, 3] - a[:, 1]).pvalue)}


def jmir_numbers() -> Dict[str, object]:
    N = base.all_numbers()
    N["t_s"] = test_sample_sd()
    N["e_s"] = external_sample_sd()
    N["vol_s"] = volumes_sample_sd()
    N["psa"] = psa_sensitivity()
    N["stats"] = _j("results/jmir_stats/summary.json")
    N["reader2"] = _j("results/jmir_reader2/summary.json")
    N["mf"] = modelfree()
    N["nci"] = nci()
    N["ov"] = overlap()
    return N


if __name__ == "__main__":
    import pprint
    N = jmir_numbers()
    for k in ("psa", "mf", "nci", "ov"):
        d = N[k]
        if k == "mf":
            d = {kk: vv for kk, vv in d.items() if kk != "summary"}
        if k == "nci":
            d = {kk: vv for kk, vv in d.items() if kk in ("n", "geometry_ok", "cg_tail_before", "cg_tail_after", "cg_multicomp_pct", "n_sites", "macro_dice")}
        print("==", k)
        pprint.pp(d, width=120, compact=True)
    print("== stats keys", list(N["stats"]))
    print("== t_s cg dice", N["t_s"]["cg"]["dice"], "\n   e_s cg dice", N["e_s"]["cg"]["dice"])
