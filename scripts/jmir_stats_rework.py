"""Statistics rework for the JMIR manuscript (CPU only, no training).

Recomputes from existing per-case result files:
  1. Held-out test metrics with sample SD (ddof=1) and 95% bootstrap CIs.
  2. Equivalence tests (TOST, paired t) for the configuration comparisons,
     with the margin fixed below before any result is read.
  3. Wilcoxon signed-rank tests for E1 vs baseline on the test set, with
     Holm correction across the three zone-level comparisons.

Usage:
    python scripts/jmir_stats_rework.py
Outputs go to results/jmir_stats/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "jmir_stats"

# Equivalence margin in Dice units. Justification to state in the Methods:
# smaller than the inter-reader SD of CG Dice (0.033) and of PZ Dice (0.068)
# on the same test cases, so a difference inside it is below the spread
# between two experts. Agree this value with the supervisor before reporting.
MARGIN = 0.02
N_BOOT = 2000
SEED = 42

VAL = {
    "baseline": RES / "exp01_baseline/eval_val_per_case_metrics.csv",
    "E1": RES / "exp_e1_dice_ce/eval_val_per_case_metrics.csv",
    "E2": RES / "exp_e2_ce_aug/eval_val_per_case_metrics.csv",
    "E3": RES / "exp_e3_dice_ce_pzweight/exp_e3_dice_ce_pzweight/eval_val_per_case_metrics.csv",
}
TEST = {
    "baseline": RES / "exp01_baseline/eval_test_per_case_metrics.csv",
    "E1": RES / "test_e1_epoch88/per_case_metrics.csv",
}
METRICS = ["dice", "iou", "hd95_mm", "asd_mm", "precision", "recall"]


def load(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, dtype={"patient_id": str})
    return d[d.class_id.isin([1, 2])]


def zone_table(d: pd.DataFrame, metric: str = "dice") -> pd.DataFrame:
    p = d.pivot(index="patient_id", columns="class_id", values=metric)
    p.columns = ["CG", "PZ"]
    p["macro"] = p[["CG", "PZ"]].mean(axis=1)
    return p


def boot_ci(v: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(N_BOOT)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def tost(diff: pd.Series, margin: float) -> dict:
    d = diff.dropna().to_numpy()
    n = len(d)
    mu, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    p_lower = 1 - stats.t.cdf((mu + margin) / se, n - 1)
    p_upper = stats.t.cdf((mu - margin) / se, n - 1)
    ci90 = stats.t.interval(0.90, n - 1, loc=mu, scale=se)
    ci95 = stats.t.interval(0.95, n - 1, loc=mu, scale=se)
    return {
        "n": n,
        "mean_diff": float(mu),
        "sd_diff": float(sd),
        "ci90": [float(ci90[0]), float(ci90[1])],
        "ci95": [float(ci95[0]), float(ci95[1])],
        "tost_p": float(max(p_lower, p_upper)),
        "equivalent": bool(max(p_lower, p_upper) < 0.05),
        "cohen_dz": float(mu / sd) if sd > 0 else float("nan"),
    }


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[k] = running
    return adjusted


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    out: dict = {"margin": MARGIN}

    # 1. Held-out test table, sample SD + bootstrap CI
    test_e1 = load(TEST["E1"])
    table = []
    for metric in METRICS:
        zt = zone_table(test_e1, metric)
        for zone in ["CG", "PZ"]:
            v = zt[zone].dropna().to_numpy()
            table.append({
                "metric": metric, "zone": zone, "n": len(v),
                "mean": v.mean(), "sd_sample": v.std(ddof=1), "sd_population": v.std(ddof=0),
                "ci95_low": boot_ci(v)[0], "ci95_high": boot_ci(v)[1],
            })
    pd.DataFrame(table).to_csv(OUT / "test_table_sample_sd.csv", index=False)

    # 2. Equivalence tests
    val = {k: zone_table(load(v)) for k, v in VAL.items()}
    test = {k: zone_table(load(v)) for k, v in TEST.items()}
    comparisons = {
        "val_E2_minus_baseline": (val["E2"], val["baseline"]),
        "val_E3_minus_E1": (val["E3"], val["E1"]),
        "val_E1_minus_baseline": (val["E1"], val["baseline"]),
        "test_E1_minus_baseline": (test["E1"], test["baseline"]),
    }
    out["tost"] = {
        name: {zone: tost(a[zone] - b[zone], MARGIN) for zone in ["macro", "CG", "PZ"]}
        for name, (a, b) in comparisons.items()
    }

    # 3. Wilcoxon on test, E1 vs baseline, Holm across zones
    a, b = test["E1"], test["baseline"]
    raw = {}
    wil = {}
    for zone in ["macro", "CG", "PZ"]:
        r = stats.wilcoxon(a[zone], b[zone])
        raw[zone] = float(r.pvalue)
        wil[zone] = {
            "W": float(r.statistic), "p_raw": float(r.pvalue),
            "n_better": int((a[zone] > b[zone]).sum()), "n": int(len(a)),
            "mean_diff": float((a[zone] - b[zone]).mean()),
        }
    adj = holm({z: raw[z] for z in ["CG", "PZ"]})
    for z in ["CG", "PZ"]:
        wil[z]["p_holm_zones"] = adj[z]
    wil["macro"]["note"] = "pre-specified primary comparison, not adjusted"
    out["wilcoxon_test_E1_vs_baseline"] = wil

    (OUT / "summary.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(table).round(4).to_string(index=False))
    for name, res in out["tost"].items():
        for zone, r in res.items():
            print(f"{name:26s} {zone:5s} diff={r['mean_diff']:+.4f} 90%CI=({r['ci90'][0]:+.4f},{r['ci90'][1]:+.4f}) "
                  f"TOST P={r['tost_p']:.4f} equiv={r['equivalent']}")
    print(json.dumps(wil, indent=2))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-model", default=str(TEST["E1"]),
                    help="per-case test metrics of the reported model (default: frozen E1)")
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args()
    TEST["E1"], OUT = Path(a.test_model), Path(a.out_dir)
    main()
