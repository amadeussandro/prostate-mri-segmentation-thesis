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

Two models are carried (see MODELS below). The top-level keys describe the
FINAL model, the softmax ensemble of 3 seeds with left-right flip averaging
frozen by results/e1_seed_selection/selection.json. `N["single"]` holds the same
quantities for the seed-42 model the earlier versions reported, because the
prespecified baseline comparison and the confirmatory NCI-ISBI test were run
with it and must stay reported as they were run.
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

MODELS = {
    "single": {
        "test": "results/rq2_e1_epoch88",
        "ablation": "results/rq2_ablation_naive/ablation_summary.csv",
        "volumes": "results/rq2_zone_volumes/zone_volumes_per_case.csv",
        "stats": "results/jmir_stats/summary.json",
        "reader2": "results/jmir_reader2/summary.json",
        "prostatex": "results/external_prostatex_e1_v2",
        "hd95_tail": "results/external_hd95_tail/summary.json",
        "anterior_stroma": "results/external_anterior_stroma/summary.json",
        "gland_size": "results/gland_size_effect/summary.json",
        "nci": "results/external_nci_isbi_e1",
        "two_annotations": "results/external_overlap/model_vs_two_annotations.json",
    },
    "final": {
        "test": "results/final_ensemble_test",
        "ablation": "results/final_ensemble_ablation/ablation_summary.csv",
        "volumes": "results/final_ensemble_zone_volumes/zone_volumes_per_case.csv",
        "stats": "results/final_ensemble_stats/summary.json",
        "reader2": "results/final_ensemble_reader2/summary.json",
        "prostatex": "results/external_prostatex_final",
        "hd95_tail": "results/final_external_hd95_tail/summary.json",
        "anterior_stroma": "results/final_external_anterior_stroma/summary.json",
        "gland_size": "results/final_gland_size_effect/summary.json",
        "nci": "results/external_nci_isbi_final",
        "two_annotations": "results/external_overlap_final/model_vs_two_annotations.json",
    },
}
SELECTION = "results/e1_seed_selection/selection.json"
SEED_PRESPEC = "results/seed_25d_prespecification.md"
LIT = "results/literature_audit"   # audit of how published 2D studies report the 2D-to-3D step


def literature_audit() -> Dict[str, object]:
    """Counts from the literature audit (scripts/literature_audit.py), checked against its files."""
    s, srch = _j(f"{LIT}/summary.json"), _j(f"{LIT}/search.json")
    raw = open(os.path.join(P, LIT, "protocol.md"), "rb").read().replace(b"\r\n", b"\n")
    if not (hashlib.sha256(raw).hexdigest() == srch["protocol_sha256_lf"] == s["protocol_sha256_lf"]):
        raise RuntimeError("the literature-audit protocol changed after the search")
    rows = _rows(f"{LIT}/extraction.csv")
    if len(rows) != s["n_included"]:
        raise RuntimeError("extraction.csv and summary.json disagree; rerun literature_audit.py summarize")
    c = s["counts"]
    depend = sum(1 for r in rows if r["metric_level"] in ("volume", "both") or r["surface_metric_mm"] == "yes")
    return {
        "n_records": s["n_records"], "n": s["n_included"], "excl": s["exclusion_reasons"],
        "run_date": srch["run_utc"][:10],
        "dim": c["dimensionality"],
        "reasm_yes": c["reassembly_described"].get("yes", 0),
        "inv_yes": s["inverse"]["k"], "inv_n": s["inverse"]["n"],
        "geo_k": s["geometry_check"]["k"], "geo_hi": s["geometry_check"]["ci95"][1],
        "space": c["metric_space"], "mm_yes": c["surface_metric_mm"].get("yes", 0),
        "code_yes": c["code_available"].get("yes", 0), "n_depend": depend,
    }


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


def test_sample_sd(test_dir: str = MODELS["single"]["test"]) -> Dict[str, dict]:
    """Held-out test metrics per zone, sample SD, bootstrap CI."""
    path = os.path.join(P, test_dir, "metrics_per_case.csv")
    out = {}
    for name, cls in (("cg", base.CLASS_CG), ("pz", base.CLASS_PZ), ("bg", base.CLASS_BG)):
        out[name] = {m: _desc(base._per_case(path, cls, m))
                     for m in ("dice", "iou", "hd95_mm", "asd_mm", "precision", "recall")}
    cg = base._per_case(path, base.CLASS_CG, "dice")
    pz = base._per_case(path, base.CLASS_PZ, "dice")
    out["macro"] = _desc((cg + pz) / 2)
    return out


def external_sample_sd(px_dir: str = "results/external_prostatex_e1/external_prostatex_e1") -> Dict[str, dict]:
    """PROSTATEx metrics with sample SD (the stored summary uses ddof=0)."""
    path = os.path.join(P, px_dir, "metrics_per_case.csv")
    out = {}
    for name, cls in (("cg", base.CLASS_CG), ("pz", base.CLASS_PZ)):
        out[name] = {m: _desc(base._per_case(path, cls, m))
                     for m in ("dice", "iou", "hd95_mm", "asd_mm", "precision", "recall")}
    cg = base._per_case(path, base.CLASS_CG, "dice")
    pz = base._per_case(path, base.CLASS_PZ, "dice")
    out["macro"] = _desc((cg + pz) / 2)
    out["n"] = int(len(cg))
    return out


def psa_sensitivity(threshold: float = 0.15, grey=(4.0, 10.0),
                    volumes: str = MODELS["single"]["volumes"]) -> Dict[str, float]:
    """How often would the model's gland volume move PSA density across 0.15?

    No PSA values exist for these patients, so the question is asked over a
    range instead of for one value. A patient is reclassified at PSA x when
    x / V_ref and x / V_pred fall on opposite sides of the threshold, which
    happens for x between 0.15 * V_pred and 0.15 * V_ref. Reported: the width
    of that PSA interval per patient, and the share of (patient, PSA) pairs
    reclassified when PSA is spread evenly over the diagnostic grey zone.
    """
    rows = _rows(volumes)
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


def volumes_sample_sd(volumes: str = MODELS["single"]["volumes"]) -> Dict[str, dict]:
    rows = _rows(volumes)
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


def nci(nci_dir: str = MODELS["single"]["nci"]) -> Dict[str, object]:
    s = _j(f"{nci_dir}/nci/summary.json")
    dec = _j(f"{nci_dir}/valid_prostate158/decision.json")
    rows = _rows(f"{nci_dir}/nci/per_case.csv")
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
    # Sensitivity: drop the NCI-ISBI case that is also a PROSTATEx examination.
    shared = [m["nci"] for m in _j("results/external_overlap/nci_overlap.json")
              if m["other"].startswith("PX")]
    kept = [r for r in rows if r["case_id"] not in shared]
    if len(rows) - len(kept) != len(shared):
        raise RuntimeError("shared PROSTATEx case not found among the analysed NCI-ISBI cases")
    out["excl_px"] = {"n_dropped": len(shared),
                      "cg_dice": float(np.mean([float(r["cg_dice_none"]) for r in kept])),
                      "cg_hd95_R1": float(np.mean([float(r["cg_hd95_R1"]) for r in kept]))}
    return out


def overlap(two_annotations: str = MODELS["single"]["two_annotations"]) -> Dict[str, object]:
    ov = _j("results/external_overlap/nci_overlap.json")
    lab = [x for x in _j("results/external_overlap/nci_vs_msd_labels.json") if x["nci"] != "Prostate3T-01-0019"]
    two = _j(two_annotations)
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


def seeds() -> Dict[str, object]:
    """The prespecified seed / 2.5D selection, read from its frozen record."""
    sel = _j(SELECTION)
    with open(os.path.join(P, SEED_PRESPEC), "rb") as fh:
        sha = hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()
    if sha != sel["prespecification_sha256"]:
        raise RuntimeError("seed/2.5D prespecification no longer matches the hash in selection.json")
    if sel["test_split_read"] is not False or sel["split"] != "val":
        raise RuntimeError("selection record does not attest a validation-only selection")
    per = {arm: {int(k): v for k, v in d.items()} for arm, d in sel["step1_single_seed"].items()}
    macro2d = [v["macro"] for v in per["2d"].values()]
    macro25 = [v["macro"] for v in per["2.5d"].values()]
    return {"per_seed": per, "arm": sel["arm_stats"], "diff": sel["diff_25d_minus_2d"],
            "margin": sel["margin"], "adopted": sel["adopted_arm"], "n_val": sel["n_cases"],
            "ensemble": sel["step3"]["ensemble"], "ensemble_tta": sel["step3"]["ensemble_tta"],
            "tta_kept": sel["tta_kept"], "members": sel["final_model"]["members"],
            "seed_list": sorted(per["2d"]),
            "range_2d": float(max(macro2d) - min(macro2d)),
            "range_25d": float(max(macro25) - min(macro25)),
            "prespec_sha256": sha}


def _paired(final_csv: str, single_csv: str) -> Dict[str, dict]:
    """Final model minus single-seed model on the same cases (paired Wilcoxon)."""
    from scipy.stats import wilcoxon

    def load(path):
        d = {}
        for r in _rows(path):
            d.setdefault(r["patient_id"], {})[int(r["class_id"])] = r
        return d

    f, s_ = load(final_csv), load(single_csv)
    ids = sorted(set(f) & set(s_))
    if len(ids) != len(f) or len(ids) != len(s_):
        raise RuntimeError(f"case sets differ: {final_csv} vs {single_csv}")
    out = {"n": len(ids)}
    col = lambda d, cls, m: np.array([float(d[i][cls][m]) for i in ids])
    for name, cls in (("cg", base.CLASS_CG), ("pz", base.CLASS_PZ)):
        for m in ("dice", "hd95_mm"):
            a, b = col(f, cls, m), col(s_, cls, m)
            ok = np.isfinite(a) & np.isfinite(b)
            d = a[ok] - b[ok]
            out[f"{name}_{m}"] = {"mean_diff": float(d.mean()), "p": float(wilcoxon(d).pvalue),
                                  "n_better": int((d > 0).sum() if m == "dice" else (d < 0).sum()),
                                  "n": int(ok.sum())}
    d = ((col(f, base.CLASS_CG, "dice") + col(f, base.CLASS_PZ, "dice")) / 2
         - (col(s_, base.CLASS_CG, "dice") + col(s_, base.CLASS_PZ, "dice")) / 2)
    lo, hi = _boot(d)
    out["macro"] = {"mean_diff": float(d.mean()), "p": float(wilcoxon(d).pvalue),
                    "n_better": int((d > 0).sum()), "n": len(d), "ci_low": lo, "ci_high": hi}
    return out


def _nci_paired(final_dir: str, single_dir: str) -> Dict[str, dict]:
    from scipy.stats import wilcoxon
    f = {r["case_id"]: r for r in _rows(f"{final_dir}/nci/per_case.csv")}
    s_ = {r["case_id"]: r for r in _rows(f"{single_dir}/nci/per_case.csv")}
    ids = sorted(set(f) & set(s_))
    if len(ids) != len(f) or len(ids) != len(s_):
        raise RuntimeError("NCI-ISBI case sets differ between the two models")
    out = {"n": len(ids)}
    for k in ("cg_dice_none", "pz_dice_none", "cg_hd95_none", "pz_hd95_none"):
        a = np.array([float(f[i][k]) for i in ids])
        b = np.array([float(s_[i][k]) for i in ids])
        ok = np.isfinite(a) & np.isfinite(b)
        d = a[ok] - b[ok]
        out[k] = {"mean_diff": float(d.mean()), "p": float(wilcoxon(d).pvalue),
                  "n_better": int((d > 0).sum() if "dice" in k else (d < 0).sum()),
                  "n": int(ok.sum())}
    return out


def model_numbers(which: str) -> Dict[str, object]:
    """Every model-dependent quantity, for one of MODELS."""
    m = MODELS[which]
    return {
        "t_s": test_sample_sd(m["test"]),
        "e_s": external_sample_sd(m["prostatex"]),
        "vol_s": volumes_sample_sd(m["volumes"]),
        "psa": psa_sensitivity(volumes=m["volumes"]),
        "stats": _j(m["stats"]),
        "reader2": _j(m["reader2"]),
        "geometry": base.geometry_validation(os.path.join(P, m["test"], "reconstruction_validation.csv")),
        "ablation": base.ablation(os.path.join(P, m["ablation"])),
        "external": base.external(os.path.join(P, m["prostatex"])),
        "hd95_tail": base.hd95_tail(os.path.join(P, m["hd95_tail"])),
        "anterior_stroma": base.anterior_stroma(os.path.join(P, m["anterior_stroma"])),
        "gland_size": base.gland_size(os.path.join(P, m["gland_size"])),
        "nci": nci(m["nci"]),
        "ov": overlap(m["two_annotations"]),
    }


PICAI = "results/picai_psad"


def picai_numbers() -> Dict[str, object]:
    """PSA density on PI-CAI (results/picai_psad_prespecification.md).

    The cohort, the conversion record and the lock exist before any model output.
    `results` stays None until `run_picai_psad.py --phase analyze` has written
    summary.json, and the manuscript then prints [[PENDING ...]] markers instead.
    """
    from collections import Counter
    import run_picai_psad as rp     # the analysis constants, read rather than retyped
    elig = _j(f"{PICAI}/eligibility.json")
    lock = _j(f"{PICAI}/prespec_lock.json")
    if lock["sha256_lf"] != elig["prespec_sha256_lf"]:
        raise RuntimeError("eligibility.json was not written under the locked prespecification")
    cohort = _rows(f"{PICAI}/cohort.csv")
    conv = [r for r in _rows(f"{PICAI}/conversion.csv") if r["case"] in {c["case"] for c in cohort}]
    ok = [r for r in conv if r["status"] == "ok"]
    if len(cohort) != elig["n_eligible"] or sum(c["cspca"] == "1" for c in cohort) != elig["n_cspca"]:
        raise RuntimeError("cohort.csv and eligibility.json disagree")
    inplane = [json.loads(r["spacing"])[0] for r in ok]
    # Laptop CPU vs Colab GPU on the same studies: the table in cpu_vs_gpu.md.
    rows = [ln for ln in open(os.path.join(P, PICAI, "cpu_check", "cpu_vs_gpu.md"), encoding="utf-8")
            if ln.startswith("| 1")]
    summary = os.path.join(P, PICAI, "summary.json")
    # Duplicates of images already used: distinct PI-CAI studies, by source and center.
    dup, merge = None, None
    if os.path.exists(os.path.join(P, PICAI, "overlap.json")):
        pairs = _j(f"{PICAI}/overlap.json")["duplicates"]
        center = {c["case"]: c["center"] for c in cohort}
        cases = sorted({d["case"] for d in pairs})
        dup = {"n_pairs": len(pairs), "n_studies": len(cases),
               "by_source": {s: len({d["case"] for d in pairs if d["other"].startswith(s)})
                             for s in ("PROSTATEx", "NCI-ISBI", "MSD")},
               "centers": dict(Counter(center[c] for c in cases)),
               "r_min": min(d["r"] for d in pairs)}
    if os.path.exists(os.path.join(P, PICAI, "colab_merge.json")):
        merge = _j(f"{PICAI}/colab_merge.json")
    return {
        "dup": dup, "merge": merge,
        "elig": elig, "lock": lock,
        "n_studies": elig["n_studies"], "n_patients": elig["n_patients"],
        "n_eligible": len(cohort), "n_cspca": elig["n_cspca"],
        "n_with_report_volume": elig["n_with_report_volume"],
        "excluded": elig["excluded"],
        "centers": dict(Counter(c["center"] for c in cohort)),
        "n_converted": len(ok), "n_conversion_failed": len(conv) - len(ok),
        "axcodes": dict(Counter(r["axcodes"] for r in ok)),
        "inplane_min": min(inplane), "inplane_max": max(inplane),
        "corner_max_diff_mm": max(float(r["corner_max_diff_mm"]) for r in ok),
        "cpu_gpu_n": len(rows), "cpu_gpu_identical": sum("| 0 of" in r for r in rows),
        "margin": rp.MARGIN, "threshold": rp.THRESH, "n_boot": rp.N_BOOT,
        "corner_tol_mm": rp.CORNER_TOL_MM, "dup_r": rp.DUP_R,
        "results": _j(f"{PICAI}/summary.json") if os.path.exists(summary) else None,
    }


def jmir_numbers(picai: bool = False) -> Dict[str, object]:
    """`picai=True` adds the PI-CAI analysis (manuscript v10); without it the v9 text is built."""
    N = base.all_numbers()          # validation arms and the thesis-era keys
    single, final = model_numbers("single"), model_numbers("final")
    N.update(final)                 # top-level keys describe the FINAL model
    N["single"] = single
    N["seeds"] = seeds()
    N["mf"] = modelfree()
    N["lit"] = literature_audit()
    # The prespecified baseline comparison and the validation TOSTs were run with
    # the single-seed model; they stay under their own key.
    N["stats_single"] = single["stats"]
    fm, sm = MODELS["final"], MODELS["single"]
    N["final_vs_single"] = {
        "test": _paired(f"{fm['test']}/metrics_per_case.csv", f"{sm['test']}/metrics_per_case.csv"),
        "prostatex": _paired(f"{fm['prostatex']}/metrics_per_case.csv",
                             f"{sm['prostatex']}/metrics_per_case.csv"),
        "nci": _nci_paired(fm["nci"], sm["nci"]),
    }
    N["picai"] = picai_numbers() if picai else None
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
