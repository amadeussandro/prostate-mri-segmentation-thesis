"""Every quantitative value the manuscript reports, loaded from result artifacts.

Nothing here is typed in by hand. The manuscript text references these fields,
so a number in the paper cannot drift away from the artifact that produced it,
and re-running this module after a new experiment surfaces any change
immediately.

Conventions fixed here and used throughout:
  * standard deviation is **population SD (ddof = 0)**, matching `src/metrics.py`
    and every arm's own summary. `results/test_e1_epoch88/summary.json` carries
    ddof = 1 values from an earlier convention and is deliberately not used.
  * macro Dice is the mean of CG and PZ only; background is reported separately.
  * Bland-Altman limits of agreement use ddof = 1, which is the standard
    convention for limits of agreement specifically.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVE = "G:/My Drive/THESIS_PROSTATE158"
TEST_ROOT = f"{DRIVE}/dataset/test/extracted/prostate158_test/test"

CLASS_CG, CLASS_PZ, CLASS_BG = 1, 2, 0
PROSTATE158_INPLANE_MM = 0.46875


def _rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _first_existing(*paths: str) -> str:
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"none of these exist: {paths}")


def _per_case(path: str, class_id: int, metric: str) -> np.ndarray:
    return np.array([float(r[metric]) for r in _rows(path)
                     if int(r["class_id"]) == class_id], dtype=float)


def _ms(vals: np.ndarray) -> Dict[str, float]:
    """mean and population SD, the repo convention."""
    v = vals[np.isfinite(vals)]
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=0)), "n": int(v.size)}


# ---------------------------------------------------------------------------
# Model selection on the validation split
# ---------------------------------------------------------------------------

ARMS = {
    "baseline": ("exp01_baseline", "Cross-entropy (baseline)", 77),
    "e1": ("exp_e1_dice_ce", "+ soft Dice on CG and PZ", 88),
    "e2": ("exp_e2_ce_aug", "+ geometric and intensity augmentation", 70),
    "e3": ("exp_e3_dice_ce_pzweight", "+ class-weighted cross-entropy", 99),
}


def validation_arms() -> Dict[str, dict]:
    out = {}
    for key, (d, label, epoch) in ARMS.items():
        p = _first_existing(f"{DRIVE}/results/{d}/eval_val_per_case_metrics.csv",
                            f"{DRIVE}/results/{d}/{d}/eval_val_per_case_metrics.csv")
        cg, pz = _per_case(p, CLASS_CG, "dice"), _per_case(p, CLASS_PZ, "dice")
        macro = (cg + pz) / 2
        out[key] = {
            "label": label, "epoch": epoch, "n": len(macro),
            "cg": float(cg.mean()), "pz": float(pz.mean()),
            "macro": float(macro.mean()), "macro_sd": float(macro.std(ddof=0)),
            "_macro_per_case": macro,
        }
    return out


def validation_tests(arms: Dict[str, dict]) -> Dict[str, dict]:
    """Paired Wilcoxon between arms on the shared 20 validation cases.

    These are EXPLORATORY. The validation split is what the model was selected
    on, so a significance test computed on it is descriptive rather than
    confirmatory, and the manuscript must say so where it reports them.
    """
    from scipy.stats import wilcoxon
    pairs = [("e1", "baseline"), ("e1", "e2"), ("e3", "e1"), ("e2", "baseline")]
    out = {}
    for a, b in pairs:
        w, p = wilcoxon(arms[a]["_macro_per_case"], arms[b]["_macro_per_case"])
        out[f"{a}_vs_{b}"] = {
            "delta": float(arms[a]["macro"] - arms[b]["macro"]),
            "W": float(w), "p": float(p),
        }
    return out


# ---------------------------------------------------------------------------
# Held-out test, the frozen E1 model
# ---------------------------------------------------------------------------

TEST_PER_CASE = f"{PROJECT}/results/rq2_e1_epoch88/metrics_per_case.csv"
BASELINE_TEST_PER_CASE = f"{PROJECT}/results/exp04_reconstruction/metrics_per_case.csv"
METRICS = ("dice", "iou", "hd95_mm", "asd_mm", "precision", "recall")


def test_metrics() -> Dict[str, dict]:
    summary = json.load(open(f"{PROJECT}/results/rq2_e1_epoch88/summary.json", encoding="utf-8"))
    out = {"per_class": {}, "n_cases": summary["metadata"]["n_cases"],
           "case_ids": summary["metadata"]["case_ids"]}
    for cls, name in ((CLASS_BG, "background"), (CLASS_CG, "cg"), (CLASS_PZ, "pz")):
        node = {}
        for m in METRICS:
            node[m] = _ms(_per_case(TEST_PER_CASE, cls, m))
            ci = summary["per_class"][str(cls)]["metrics"][m]
            node[m]["ci_low"] = float(ci["ci95_low"])
            node[m]["ci_high"] = float(ci["ci95_high"])
        out["per_class"][name] = node
    cg = _per_case(TEST_PER_CASE, CLASS_CG, "dice")
    pz = _per_case(TEST_PER_CASE, CLASS_PZ, "dice")
    macro = (cg + pz) / 2
    out["macro"] = {"mean": float(macro.mean()), "sd": float(macro.std(ddof=0)),
                    "min": float(macro.min()), "max": float(macro.max())}
    ids = sorted({r["patient_id"] for r in _rows(TEST_PER_CASE)})
    order = np.argsort(macro)
    out["representatives"] = {
        "challenging": ids[int(order[0])], "median": ids[int(order[len(order) // 2])],
        "good": ids[int(order[-1])],
    }
    out["cg_range"] = (float(cg.min()), float(cg.max()))
    out["pz_range"] = (float(pz.min()), float(pz.max()))
    return out


def test_vs_baseline() -> Dict[str, dict]:
    """E1 against the baseline on the same 19 held-out cases.

    The macro comparison is the pre-specified one. The per-zone tests are POST
    HOC and are labelled as such wherever they appear.
    """
    from scipy.stats import wilcoxon

    def load(p):
        d = {}
        for r in _rows(p):
            d.setdefault(r["patient_id"], {})[int(r["class_id"])] = float(r["dice"])
        return d

    b, e = load(BASELINE_TEST_PER_CASE), load(TEST_PER_CASE)
    ids = sorted(set(b) & set(e))
    out = {"n": len(ids)}
    for name, cls in (("cg", CLASS_CG), ("pz", CLASS_PZ)):
        bv = np.array([b[i][cls] for i in ids]); ev = np.array([e[i][cls] for i in ids])
        w, p = wilcoxon(ev, bv)
        out[name] = {"delta": float(ev.mean() - bv.mean()), "W": float(w), "p": float(p),
                     "better_in": int((ev > bv).sum())}
    bm = np.array([(b[i][CLASS_CG] + b[i][CLASS_PZ]) / 2 for i in ids])
    em = np.array([(e[i][CLASS_CG] + e[i][CLASS_PZ]) / 2 for i in ids])
    w, p = wilcoxon(em, bm)
    out["macro"] = {"delta": float(em.mean() - bm.mean()), "W": float(w), "p": float(p),
                    "better_in": int((em > bm).sum()),
                    "baseline_mean": float(bm.mean()), "e1_mean": float(em.mean())}
    return out


# ---------------------------------------------------------------------------
# Reconstruction: geometry, ablation, volumes
# ---------------------------------------------------------------------------

def geometry_validation() -> Dict[str, object]:
    rows = _rows(f"{PROJECT}/results/rq2_e1_epoch88/reconstruction_validation.csv")
    diffs = [float(r["affine_max_abs_diff"]) for r in rows]
    return {
        "n": len(rows),
        "n_all_ok": sum(r["all_ok"] == "True" for r in rows),
        "affine_max_abs_diff": max(diffs),
        "orientations": sorted({r["recon_orientation"] for r in rows}),
        "labels_found": sorted({r["labels_found"] for r in rows}),
    }


def ablation() -> Dict[str, dict]:
    rows = _rows(f"{PROJECT}/results/rq2_ablation_naive/ablation_summary.csv")
    out = {}
    for r in rows:
        v = out.setdefault(r["variant"], {})
        cls = "cg" if r["class_name"].startswith("CG") else "pz"
        v[cls] = {
            "dice_vs_correct": float(r["dice_vs_correct_mean"]),
            "dice_vs_gt": float(r.get("dice_vs_gt_variant_mean", "nan")),
            "dice_vs_gt_correct": float(r.get("dice_vs_gt_correct_mean", "nan")),
            "centroid_shift_mm": float(r["centroid_shift_mm_mean"]),
            "centroid_shift_max_mm": float(r["centroid_shift_mm_max"]),
            "volume_pct_error": float(r["volume_pct_error_mean"]),
            "volume_abs_error_ml": float(r["volume_abs_error_ml_mean"]),
            "n_caught": int(r["n_caught_by_geometry_check"]),
            "n_cases": int(r["n_cases"]),
        }
    return out


def zone_volumes() -> Dict[str, object]:
    rows = _rows(f"{PROJECT}/results/rq2_zone_volumes/zone_volumes_per_case.csv")
    out = {"n": len(rows)}
    for key, name in (("cg_ml", "cg"), ("pz_ml", "pz"), ("whole_gland_ml", "whole")):
        gt = np.array([float(r[f"gt_{key}"]) for r in rows])
        pr = np.array([float(r[f"pred_{key}"]) for r in rows])
        d = pr - gt
        bias, sd = float(d.mean()), float(d.std(ddof=1))
        out[name] = {
            "gt_mean": float(gt.mean()), "gt_sd": float(gt.std(ddof=0)),
            "pred_mean": float(pr.mean()), "pred_sd": float(pr.std(ddof=0)),
            "bias": bias, "pct_of_gt": float(bias / gt.mean() * 100),
            "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
            "under_in": int((d < 0).sum()),
        }
    gt = np.array([float(r["gt_whole_gland_ml"]) for r in rows])
    pr = np.array([float(r["pred_whole_gland_ml"]) for r in rows])
    ratio = gt / pr  # PSA density scales with the inverse of the volume
    out["psad"] = {
        "mean_pct": float((ratio.mean() - 1) * 100),
        "max_pct": float((ratio.max() - 1) * 100),
        "over_in": int((ratio > 1).sum()),
        "threshold_equivalent": float(0.15 / ratio.mean()),
    }
    return out


# ---------------------------------------------------------------------------
# External validation
# ---------------------------------------------------------------------------

def external(results_dir: str = None) -> Dict[str, object]:
    # Prefer the run that also saved volumes and per-case geometry validation;
    # fall back to the earlier run while that one is still in progress. Check a
    # file the run writes at the END, so a half-finished directory is not used.
    if results_dir is None:
        candidates = [f"{PROJECT}/results/external_prostatex_e1_v2",
                      f"{PROJECT}/results/external_prostatex_e1/external_prostatex_e1"]
        base = next((c for c in candidates
                     if os.path.exists(f"{c}/cohort_geometry.csv")), None)
        if base is None:
            raise FileNotFoundError(
                "No completed external-validation run found. Looked for "
                "cohort_geometry.csv under: " + ", ".join(candidates))
    else:
        base = results_dir
    per_case = f"{base}/metrics_per_case.csv"
    geom = _rows(f"{base}/cohort_geometry.csv")
    out = {"n_cases": len({r["patient_id"] for r in _rows(per_case)}), "dir": base}
    for cls, name in ((CLASS_BG, "background"), (CLASS_CG, "cg"), (CLASS_PZ, "pz")):
        out[name] = {m: _ms(_per_case(per_case, cls, m)) for m in METRICS}
    cg = _per_case(per_case, CLASS_CG, "dice"); pz = _per_case(per_case, CLASS_PZ, "dice")
    macro = (cg + pz) / 2
    out["macro"] = {"mean": float(macro.mean()), "sd": float(macro.std(ddof=0))}

    hd = _per_case(per_case, CLASS_CG, "hd95_mm")
    prec = _per_case(per_case, CLASS_CG, "precision")
    out["hd95_cg"] = {
        "median": float(np.median(hd)), "mean": float(hd.mean()), "max": float(hd.max()),
        "p90": float(np.percentile(hd, 90)),
        "n_over_30": int((hd > 30).sum()), "n_over_50": int((hd > 50).sum()),
        "corr_with_precision": float(np.corrcoef(prec, hd)[0, 1]),
    }
    import ast
    fov = np.array([ast.literal_eval(g["shape"])[0] * float(g["inplane_mm"]) for g in geom])
    scale = np.array([float(g["scale_vs_prostate158"]) for g in geom])
    inplane = np.array([float(g["inplane_mm"]) for g in geom])
    through = np.array([float(g["through_plane_mm"]) for g in geom])
    out["geometry"] = {
        "fov_min": float(fov.min()), "fov_max": float(fov.max()), "fov_mean": float(fov.mean()),
        "fov_internal": 232 * PROSTATE158_INPLANE_MM,
        "fov_ratio": float(fov.mean() / (232 * PROSTATE158_INPLANE_MM)),
        "inplane_min": float(inplane.min()), "inplane_max": float(inplane.max()),
        "through_min": float(through.min()), "through_max": float(through.max()),
        "scale_min": float(scale.min()), "scale_max": float(scale.max()),
        "n_cropped": int(sum(g["inplane_exceeds_target"] == "True" for g in geom)),
        "n_losing_foreground": int(sum(
            int(g.get("class1_lost", 0) or 0) + int(g.get("class2_lost", 0) or 0) > 0
            for g in geom)),
        "orientations": sorted({g["orientation"] for g in geom}),
    }
    if geom and "geometry_all_ok" in geom[0]:
        out["geometry"]["n_all_ok"] = int(sum(g["geometry_all_ok"] == "True" for g in geom))
    return out



def hd95_tail() -> Dict[str, object]:
    """Connected-component diagnosis of the external boundary-error tail.

    Produced by `scripts/analyse_external_hd95_tail.py`. Loaded rather than
    recomputed so the manuscript quotes the same summary the report does.
    """
    path = f"{PROJECT}/results/external_hd95_tail/summary.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No HD95 tail analysis found. Run scripts/analyse_external_hd95_tail.py first.")
    return json.load(open(path, encoding="utf-8"))


def anterior_stroma() -> Dict[str, object]:
    """Where along the gland the external central-gland gap actually sits.

    Produced by `scripts/analyse_anterior_stroma.py`. It tests whether the
    anterior fibromuscular stroma in the external reference explains the
    external CG shortfall, using the internal cohort as the control, and finds
    that it does not. Loaded rather than recomputed so the manuscript quotes the
    same summary the report does.
    """
    path = f"{PROJECT}/results/external_anterior_stroma/summary.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No anterior-stroma analysis found. Run scripts/analyse_anterior_stroma.py first.")
    return json.load(open(path, encoding="utf-8"))


def gland_size() -> Dict[str, object]:
    """Does accuracy depend on how large the reference zone is?

    Produced by `scripts/analyse_gland_size_effect.py`. Answers a question put by
    the supervisor, and the answer is a limitation rather than a strength: the
    model is markedly weaker on small glands. Loaded rather than recomputed so
    the manuscript quotes the same summary the report does.
    """
    path = f"{PROJECT}/results/gland_size_effect/summary.json"
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No gland-size analysis found. Run scripts/analyse_gland_size_effect.py first.")
    return json.load(open(path, encoding="utf-8"))


# ---------------------------------------------------------------------------
# Extended version only
# ---------------------------------------------------------------------------

def _val_path(d: str) -> str:
    """The per-case validation CSV, from the same source validation_arms() uses."""
    return _first_existing(f"{DRIVE}/results/{d}/eval_val_per_case_metrics.csv",
                           f"{DRIVE}/results/{d}/{d}/eval_val_per_case_metrics.csv")


def _training_log(d: str) -> str:
    return _first_existing(f"{DRIVE}/results/{d}/metrics.csv",
                           f"{DRIVE}/results/{d}/{d}/metrics.csv",
                           f"{PROJECT}/results/{d}/metrics.csv",
                           f"{PROJECT}/results/{d}/{d}/metrics.csv")


def _network_input_size() -> List[int]:
    import yaml
    cfg = yaml.safe_load(open(f"{PROJECT}/configs/experiments/exp_e1_dice_ce.yaml",
                              encoding="utf-8"))

    def find(node):
        if isinstance(node, dict):
            if "target_size" in node:
                return node["target_size"]
            for v in node.values():
                hit = find(v)
                if hit is not None:
                    return hit
        return None

    size = find(cfg)
    if not size:
        raise RuntimeError("target_size not found in the E1 config")
    return [int(s) for s in size]


def extended() -> Dict[str, object]:
    """Numbers used only by the extended version of the manuscript.

    Called by build_manuscript.py with --extended and by nothing else, so the
    submission build reads exactly what it read before this existed. Every value
    is loaded from an artifact; where two artifacts must agree, a guard raises
    rather than letting the text quote one of them silently.
    """
    out: Dict[str, object] = {}

    # Training history. Baseline kept none; E1's first 69 epochs were
    # overwritten by a resume. Both facts are stated, not hidden.
    hist = {}
    for key in ("e1", "e2", "e3"):
        rows = _rows(_training_log(ARMS[key][0]))
        ep = sorted(int(r["epoch"]) for r in rows)
        hist[key] = {"first": ep[0], "last": ep[-1], "n": len(ep)}
    out["training"] = hist

    # Selection. The training loop maximised the batch-level foreground Dice
    # (val_dice_proxy); the reported metric is case-level. Guard that the
    # criterion's maximum is the epoch the manuscript says was frozen.
    e1_rows = _rows(_training_log(ARMS["e1"][0]))
    by_ep = {int(r["epoch"]): r for r in e1_rows}

    def bmacro(r):
        return (float(r["val_dice_class1_CG"]) + float(r["val_dice_class2_PZ"])) / 2

    best_crit = max(by_ep, key=lambda e: float(by_ep[e]["val_dice_proxy"]))
    best_macro = max(by_ep, key=lambda e: bmacro(by_ep[e]))
    if best_crit != ARMS["e1"][2]:
        raise RuntimeError(f"criterion maximum is epoch {best_crit}, but the manuscript "
                           f"states epoch {ARMS['e1'][2]} was frozen")
    out["selection"] = {
        "epoch": best_crit, "crit_value": float(by_ep[best_crit]["val_dice_proxy"]),
        "macro_at_selected": bmacro(by_ep[best_crit]),
        "macro_best_epoch": best_macro, "macro_best_value": bmacro(by_ep[best_macro]),
        "window": (min(by_ep), max(by_ep)),
    }

    # E3 mechanism: the operating point moved, overlap did not.
    pr = {}
    per_case = {}
    for key, (d, _, _) in ARMS.items():
        rows = [r for r in _rows(_val_path(d)) if int(r["class_id"]) == CLASS_PZ]
        per_case[key] = {r["patient_id"].zfill(3): r for r in rows}
        pr[key] = {
            "precision": float(np.mean([float(r["precision"]) for r in rows])),
            "recall": float(np.mean([float(r["recall"]) for r in rows])),
            "dice": float(np.mean([float(r["dice"]) for r in rows])),
        }
    e1c, e3c = per_case["e1"], per_case["e3"]
    shared = sorted(set(e1c) & set(e3c))
    out["pz_operating_point"] = pr
    out["e3_vs_e1"] = {
        "n": len(shared),
        "recall_up": sum(float(e3c[c]["recall"]) > float(e1c[c]["recall"]) for c in shared),
        "precision_down": sum(float(e3c[c]["precision"]) < float(e1c[c]["precision"])
                              for c in shared),
        "dice_up": sum(float(e3c[c]["dice"]) > float(e1c[c]["dice"]) for c in shared),
    }

    # The hardest validation case, per arm.
    hardest = {}
    for key, (d, _, _) in ARMS.items():
        cases: Dict[str, List[float]] = {}
        for r in _rows(_val_path(d)):
            if int(r["class_id"]) in (CLASS_CG, CLASS_PZ):
                cases.setdefault(r["patient_id"].zfill(3), []).append(float(r["dice"]))
        worst = min(cases, key=lambda c: float(np.mean(cases[c])))
        hardest[key] = {"case": worst, "macro": float(np.mean(cases[worst]))}
    out["hardest_validation"] = hardest
    out["hardest_same_in_all"] = len({h["case"] for h in hardest.values()}) == 1

    # Per-case held-out test, and the representatives' ranks.
    rows = _rows(TEST_PER_CASE)
    table: Dict[str, Dict[int, dict]] = {}
    for r in rows:
        cid = int(r["class_id"])
        if cid in (CLASS_CG, CLASS_PZ):
            table.setdefault(r["patient_id"], {})[cid] = r
    ids = sorted(table)
    per = [{"case": c,
            "cg_dice": float(table[c][CLASS_CG]["dice"]),
            "pz_dice": float(table[c][CLASS_PZ]["dice"]),
            "cg_hd95": float(table[c][CLASS_CG]["hd95_mm"]),
            "pz_hd95": float(table[c][CLASS_PZ]["hd95_mm"])} for c in ids]
    out["test_per_case"] = per
    macro = {p["case"]: (p["cg_dice"] + p["pz_dice"]) / 2 for p in per}
    ranked = sorted(macro, key=macro.get)
    out["test_ranks"] = {c: ranked.index(c) + 1 for c in ranked}
    out["test_n"] = len(ranked)
    out["lowest_pz_case"] = min(per, key=lambda p: p["pz_dice"])["case"]

    # Per-case change against the baseline, from the very files test_vs_baseline()
    # reads, with a guard that the two agree on how many cases improved.
    base_rows: Dict[str, Dict[int, float]] = {}
    for r in _rows(BASELINE_TEST_PER_CASE):
        base_rows.setdefault(r["patient_id"], {})[int(r["class_id"])] = float(r["dice"])
    delta = {c: macro[c] - (base_rows[c][CLASS_CG] + base_rows[c][CLASS_PZ]) / 2
             for c in macro if c in base_rows}
    better = sum(d > 0 for d in delta.values())
    expected = test_vs_baseline()["macro"]["better_in"]
    if better != expected:
        raise RuntimeError(f"per-case count {better} disagrees with test_vs_baseline {expected}")
    worst = min(delta, key=delta.get)
    out["vs_baseline_worst"] = {"case": worst, "delta": float(delta[worst])}

    # External: bootstrap intervals and the geometry the main text does not list.
    base = external()["dir"]
    ci = {}
    for r in _rows(f"{base}/metrics_summary.csv"):
        cid = int(r["class_id"])
        if cid in (CLASS_CG, CLASS_PZ):
            ci.setdefault("cg" if cid == CLASS_CG else "pz", {})[r["metric"]] = {
                k: float(r[k]) for k in ("mean", "std", "ci95_low", "ci95_high")}
    out["external_ci"] = ci

    import ast
    geom = _rows(f"{base}/cohort_geometry.csv")
    matrix = sorted({ast.literal_eval(g["shape"])[0] for g in geom})
    cropped = [g for g in geom if g["inplane_exceeds_target"] == "True"]
    out["geometry_extra"] = {
        "n": len(geom),
        "matrix_min": matrix[0], "matrix_max": matrix[-1],
        "network_input": _network_input_size(),
        "cropped_matrix": sorted({ast.literal_eval(g["shape"])[0] for g in cropped}),
        "cropped_inplane": sorted({round(float(g["inplane_mm"]), 3) for g in cropped}),
        "lost_in_cropped": int(sum(int(g["class1_lost"]) + int(g["class2_lost"])
                                   for g in cropped)),
        "affine_max": float(max(float(g["affine_max_abs_diff"]) for g in geom)),
        "reference_inplane": PROSTATE158_INPLANE_MM,
    }
    return out


def all_numbers() -> Dict[str, object]:
    arms = validation_arms()
    spread = max(a["macro"] for a in arms.values()) - min(a["macro"] for a in arms.values())
    sds = [a["macro_sd"] for a in arms.values()]
    return {
        "arms": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                 for k, v in arms.items()},
        "arm_spread": float(spread),
        "arm_sd_range": (float(min(sds)), float(max(sds))),
        "validation_tests": validation_tests(arms),
        "test": test_metrics(),
        "test_vs_baseline": test_vs_baseline(),
        "geometry": geometry_validation(),
        "ablation": ablation(),
        "volumes": zone_volumes(),
        "external": external(),
        "hd95_tail": hd95_tail(),
        "anterior_stroma": anterior_stroma(),
        "gland_size": gland_size(),
    }


if __name__ == "__main__":
    import pprint
    pprint.pp(all_numbers(), width=110, compact=True)
