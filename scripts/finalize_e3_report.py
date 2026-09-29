"""Assemble the Experiment E3 deliverables: summary.json, training_report.md, plots.

E3 = E1 + median-frequency class-weighted Cross Entropy. The ONE factor that
changed relative to E1 is `loss.ce_class_weights = [0.0456, 1.0, 2.8477]`
(background, CG, PZ). Everything this script reports is framed against E1 for
that reason, with the Baseline and E2 carried along as context.

Reads only artifacts that already exist on disk -- it NEVER trains, never runs
inference, and never touches the official 19-case test set. Specifically it
consumes:

  * <e3_dir>/metrics.csv                      -- per-epoch history from run_training
  * <e3_dir>/training_history.json            -- same, richer (optional)
  * <e3_dir>/config_fingerprint.json          -- the run's resolved settings (optional)
  * <e3_dir>/eval_val_summary.json            -- 20-case volume-level validation eval
  * <e3_dir>/eval_val_per_case_metrics.csv    -- per-case rows, for the paired test
  * <e1_dir>/eval_val_summary.json  + per-case CSV   -- the E3 reference arm
  * <baseline_dir>/eval_val_summary.json      -- context anchor
  * <e2_dir>/eval_val_summary.json            -- context anchor (optional)

and writes:

  * <e3_dir>/summary.json           -- machine-readable E3 result, deltas, statistics
  * <e3_dir>/training_report.md      -- human-readable report for the thesis
  * <e3_dir>/e3_vs_e1_paired_cases.csv -- the per-case paired table used for the test
  * <e3_dir>/plots/training_loss.png
  * <e3_dir>/plots/validation_dice.png
  * <e3_dir>/plots/validation_metrics.png
  * <e3_dir>/plots/e3_vs_e1_paired.png

Comparison metric (unchanged from Baseline / E1 / E2):

    validation macro Dice = (mean CG Dice + mean PZ Dice) / 2

computed per case, at volume level, in original voxel space, over the 20
official validation cases.

STATISTICS -- nothing new is invented for E3
--------------------------------------------
* bootstrap 95% CI: `src.metrics.bootstrap_ci`, the SAME percentile bootstrap
  (2000 resamples, seed 42) the evaluation already uses for every cohort mean.
* paired comparison vs E1: two-sided Wilcoxon signed-rank over the 20 paired
  per-case values, via `scipy.stats.wilcoxon` -- reported only when SciPy is
  installed, and reported as a descriptive p-value on n = 20, never as a
  gate on the pre-committed success criteria.

PRE-COMMITTED SUCCESS CRITERIA
------------------------------
Fixed BEFORE the run and hard-coded below so they cannot be adjusted after
seeing the numbers. See E3_CRITERIA / classify_outcome.

SCOPE LIMIT -- deliberately enforced here
-----------------------------------------
This script does NOT select a final model, does NOT run the 2x2 factorial
analysis, and refuses outright to read a test-split evaluation.

Usage
-----
    python scripts/finalize_e3_report.py \
        --e3-dir results/exp_e3_dice_ce_pzweight \
        --e1-dir results/exp_e1_dice_ce \
        --baseline-dir results/exp01_baseline \
        --e2-dir results/exp_e2_ce_aug
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Foreground classes. Label identity (1=CG, 2=PZ) is the documented -- still
# unverified -- hypothesis; see results/label_mapping/label_mapping_report.md.
CLASS_CG = "1"
CLASS_PZ = "2"
CLASS_LABELS = {CLASS_CG: "CG (class 1)", CLASS_PZ: "PZ (class 2)"}

REPORT_METRICS = ("dice", "iou", "hd95_mm", "asd_mm", "precision", "recall")
# Metrics where a LOWER value is better (distances in mm).
LOWER_IS_BETTER = {"hd95_mm", "asd_mm"}

# Arms, in report order. "e3" is always required; "e1" is the reference arm.
ARMS = ("baseline", "e1", "e2", "e3")
ARM_LABELS = {
    "baseline": "Baseline (CE, no aug)",
    "e1": "E1 (Dice+CE, no aug)",
    "e2": "E2 (CE, aug)",
    "e3": "E3 (Dice+CE, CE class-weighted)",
}
ARM_INTERVENTIONS = {
    "baseline": "none (reference configuration)",
    "e1": "loss: + 1.0 x SoftDice(CG, PZ)",
    "e2": "mild geometric/intensity augmentation (train only)",
    "e3": "CE class weights [0.0456, 1.0, 2.8477] (bg, CG, PZ)",
}

# ---------------------------------------------------------------------------
# PRE-COMMITTED E3 SUCCESS CRITERIA -- fixed before the run. DO NOT EDIT after
# seeing results; doing so invalidates the experiment.
# ---------------------------------------------------------------------------
E3_CRITERIA = {
    "e1_reference": {
        "macro_dice": 0.8094,
        "cg_dice": 0.8662,
        "pz_dice": 0.7527,
        "pz_recall": 0.7161,
        "pz_precision": 0.8073,
    },
    "improvement": {"macro_dice_min": 0.8194, "cg_dice_min": 0.8612},
    "mechanistic_success": {"pz_recall_min": 0.7461, "macro_dice_tolerance": 0.010},
    "similar": {"macro_dice_tolerance": 0.010, "pz_recall_gain_below": 0.030},
    "regression": {"macro_dice_max": 0.7994, "cg_dice_max_drop_vs_e1": 0.005},
}

OUTCOME_LABELS = {
    "improvement": "IMPROVEMENT",
    "mechanistic_success": "MECHANISTIC SUCCESS (metric-neutral)",
    "similar": "SIMILAR",
    "regression": "REGRESSION",
    "indeterminate": "INDETERMINATE (matches no pre-committed category)",
}


# --------------------------------------------------------------------------- IO


def _load_summary(path: str, what: str, required: bool = True) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError(
                f"{what} evaluation summary not found: {path}\n"
                "Run src/evaluate.py::run_evaluation(..., split='val') for that arm first."
            )
        return None
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    metadata = summary.get("metadata", {})
    # Hard integrity guard: this script must never be fed a test-set evaluation.
    if metadata.get("is_official_held_out_test") or metadata.get("split") == "test":
        raise ValueError(
            f"REFUSING to build the E3 comparison from a TEST-set evaluation ({path}). "
            "E3 model selection and comparison must use the 20-case validation split "
            "only; the official 19-case test set is reserved for the single final "
            "evaluation after the RQ-1 configuration is chosen."
        )
    return summary


def _load_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_metrics_csv(path: str) -> List[Dict[str, Any]]:
    """Load the per-epoch metrics CSV, coercing numeric columns to float/int."""
    if not os.path.isfile(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: Dict[str, Any] = {}
            for key, value in raw.items():
                if value is None or value == "":
                    row[key] = None
                elif key == "epoch":
                    row[key] = int(value)
                elif key == "amp":
                    row[key] = str(value).strip().lower() == "true"
                else:
                    try:
                        row[key] = float(value)
                    except ValueError:
                        row[key] = value
            rows.append(row)
    rows.sort(key=lambda r: r.get("epoch", 0))
    return rows


def read_per_case_metrics(path: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load a per-case metrics CSV as {patient_id: {class_id: {metric: value}}}."""
    table: Dict[str, Dict[str, Dict[str, float]]] = {}
    if not os.path.isfile(path):
        return table
    with open(path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            case = str(row.get("patient_id", "")).strip()
            class_id = str(row.get("class_id", "")).strip()
            if not case or not class_id:
                continue
            metrics: Dict[str, float] = {}
            for metric in REPORT_METRICS:
                raw = row.get(metric)
                if raw is None or raw == "":
                    continue
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if not math.isnan(value):
                    metrics[metric] = value
            table.setdefault(case, {})[class_id] = metrics
    return table


def git_commit(repo_dir: str = project_root) -> Optional[str]:
    """Current HEAD commit, or None outside a git checkout. Never raises."""
    try:
        out = subprocess.run(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# ------------------------------------------------------------------ COMPARISON


def _metric(summary: Optional[Dict[str, Any]], class_id: str, metric: str) -> Optional[float]:
    if not summary:
        return None
    try:
        return summary["per_class"][class_id]["metrics"][metric]["mean"]
    except (KeyError, TypeError):
        return None


def _metric_sd(summary: Optional[Dict[str, Any]], class_id: str, metric: str) -> Optional[float]:
    if not summary:
        return None
    per_class = (summary.get("per_class") or {}).get(class_id) or {}
    cell = (per_class.get("metrics") or {}).get(metric) or {}
    for key in ("std", "sd", "stdev"):
        if cell.get(key) is not None:
            return cell[key]
    return None


def _macro_dice(summary: Optional[Dict[str, Any]]) -> Optional[float]:
    cg = _metric(summary, CLASS_CG, "dice")
    pz = _metric(summary, CLASS_PZ, "dice")
    if cg is None or pz is None:
        return None
    return (cg + pz) / 2.0


def build_comparison(summaries: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
    """Per-class, per-metric table across every available arm, plus macro Dice.

    Deltas are computed against **E1**, because E1 is the arm E3 is nested in:
    E3 is E1 with one added factor. A delta against the Baseline is also carried
    for continuity with the earlier reports.
    """
    comparison: Dict[str, Any] = {"per_class": {}, "macro": {}}

    for class_id in (CLASS_CG, CLASS_PZ):
        comparison["per_class"][class_id] = {}
        for metric in REPORT_METRICS:
            cell: Dict[str, Any] = {
                arm: _metric(summaries.get(arm), class_id, metric) for arm in ARMS
            }
            cell["e3_sd"] = _metric_sd(summaries.get("e3"), class_id, metric)
            cell["e1_sd"] = _metric_sd(summaries.get("e1"), class_id, metric)
            cell["delta_vs_e1"] = (
                cell["e3"] - cell["e1"]
                if cell["e3"] is not None and cell["e1"] is not None else None
            )
            cell["delta_vs_baseline"] = (
                cell["e3"] - cell["baseline"]
                if cell["e3"] is not None and cell["baseline"] is not None else None
            )
            cell["lower_is_better"] = metric in LOWER_IS_BETTER
            comparison["per_class"][class_id][metric] = cell

    macro: Dict[str, Any] = {arm: _macro_dice(summaries.get(arm)) for arm in ARMS}
    macro["delta_vs_e1"] = (
        macro["e3"] - macro["e1"] if macro["e3"] is not None and macro["e1"] is not None else None
    )
    macro["delta_vs_baseline"] = (
        macro["e3"] - macro["baseline"]
        if macro["e3"] is not None and macro["baseline"] is not None else None
    )
    comparison["macro"] = macro
    return comparison


# ---------------------------------------------------------------- PAIRED STATS


def per_case_macro_dice(
    per_case: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, float]:
    """Per-case macro Dice = (CG Dice + PZ Dice) / 2, for every case with both."""
    macro: Dict[str, float] = {}
    for case, classes in per_case.items():
        cg = (classes.get(CLASS_CG) or {}).get("dice")
        pz = (classes.get(CLASS_PZ) or {}).get("dice")
        if cg is not None and pz is not None:
            macro[case] = (cg + pz) / 2.0
    return macro


def _bootstrap_ci(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Percentile bootstrap CI via the project's existing helper (same seed/n)."""
    if len(values) < 2:
        return None, None
    try:
        from src.metrics import bootstrap_ci
    except ImportError:
        return None, None
    low, high = bootstrap_ci(values)
    if low != low or high != high:  # NaN
        return None, None
    return float(low), float(high)


def _wilcoxon(differences: List[float]) -> Dict[str, Any]:
    """Two-sided Wilcoxon signed-rank on paired differences. Never raises.

    Returns {"available", "statistic", "p_value", "n", "n_nonzero", "note"}.
    """
    result: Dict[str, Any] = {
        "test": "Wilcoxon signed-rank (two-sided, paired per-case)",
        "available": False,
        "statistic": None,
        "p_value": None,
        "n": len(differences),
        "n_nonzero": sum(1 for d in differences if d != 0.0),
        "note": None,
    }
    if result["n_nonzero"] == 0:
        result["note"] = "all paired differences are exactly zero; the test is undefined"
        return result
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        result["note"] = "SciPy is not installed; the paired test was not run"
        return result
    try:
        statistic, p_value = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
    except (ValueError, TypeError) as exc:
        result["note"] = f"scipy.stats.wilcoxon could not run: {exc}"
        return result
    result.update({"available": True, "statistic": float(statistic), "p_value": float(p_value)})
    result["note"] = (
        f"n = {result['n']} paired validation cases; descriptive, not a gate on the "
        "pre-committed success criteria"
    )
    return result


def paired_comparison(
    e3_per_case: Dict[str, Dict[str, Dict[str, float]]],
    e1_per_case: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Any]:
    """Case-by-case E3 vs E1 comparison: wins/losses, deltas, CI, Wilcoxon."""
    payload: Dict[str, Any] = {
        "available": False,
        "reason": None,
        "n_paired_cases": 0,
        "case_ids": [],
        "metrics": {},
        "rows": [],
    }
    if not e3_per_case:
        payload["reason"] = "no E3 per-case metrics found"
        return payload
    if not e1_per_case:
        payload["reason"] = "no E1 per-case metrics found (needed for the paired test)"
        return payload

    shared = sorted(set(e3_per_case) & set(e1_per_case))
    if not shared:
        payload["reason"] = "E3 and E1 per-case tables share no case IDs"
        return payload

    payload["available"] = True
    payload["n_paired_cases"] = len(shared)
    payload["case_ids"] = shared

    # ---- macro Dice, plus every per-class metric ----
    e3_macro = per_case_macro_dice(e3_per_case)
    e1_macro = per_case_macro_dice(e1_per_case)

    targets: List[Tuple[str, Any, Any]] = [
        ("macro_dice", lambda case: e3_macro.get(case), lambda case: e1_macro.get(case))
    ]
    for class_id in (CLASS_CG, CLASS_PZ):
        for metric in REPORT_METRICS:
            label = f"class{class_id}_{metric}"
            targets.append((
                label,
                lambda case, c=class_id, m=metric: (e3_per_case.get(case, {}).get(c) or {}).get(m),
                lambda case, c=class_id, m=metric: (e1_per_case.get(case, {}).get(c) or {}).get(m),
            ))

    for label, get_e3, get_e1 in targets:
        pairs = [
            (case, get_e3(case), get_e1(case)) for case in shared
            if get_e3(case) is not None and get_e1(case) is not None
        ]
        if not pairs:
            continue
        e3_values = [p[1] for p in pairs]
        e1_values = [p[2] for p in pairs]
        diffs = [a - b for a, b in zip(e3_values, e1_values)]
        lower_better = any(label.endswith(m) for m in LOWER_IS_BETTER)
        # "Won" = moved in the beneficial direction for this metric.
        won = sum(1 for d in diffs if (d < 0 if lower_better else d > 0))
        lost = sum(1 for d in diffs if (d > 0 if lower_better else d < 0))
        ci_low, ci_high = _bootstrap_ci(diffs)
        payload["metrics"][label] = {
            "n": len(pairs),
            "e3_mean": statistics.fmean(e3_values),
            "e1_mean": statistics.fmean(e1_values),
            "e3_sd": statistics.stdev(e3_values) if len(e3_values) > 1 else 0.0,
            "e1_sd": statistics.stdev(e1_values) if len(e1_values) > 1 else 0.0,
            "mean_difference": statistics.fmean(diffs),
            "median_difference": statistics.median(diffs),
            "sd_difference": statistics.stdev(diffs) if len(diffs) > 1 else 0.0,
            "difference_bootstrap_ci95": [ci_low, ci_high],
            "cases_won": won,
            "cases_lost": lost,
            "cases_tied": len(pairs) - won - lost,
            "lower_is_better": lower_better,
            "wilcoxon": _wilcoxon(diffs),
        }

    # ---- the per-case table the report prints and writes to CSV ----
    for case in shared:
        row = {"patient_id": case}
        row["e1_macro_dice"] = e1_macro.get(case)
        row["e3_macro_dice"] = e3_macro.get(case)
        row["delta_macro_dice"] = (
            e3_macro[case] - e1_macro[case]
            if case in e3_macro and case in e1_macro else None
        )
        for class_id, name in ((CLASS_CG, "cg"), (CLASS_PZ, "pz")):
            for metric in ("dice", "recall", "precision"):
                e1_value = (e1_per_case.get(case, {}).get(class_id) or {}).get(metric)
                e3_value = (e3_per_case.get(case, {}).get(class_id) or {}).get(metric)
                row[f"e1_{name}_{metric}"] = e1_value
                row[f"e3_{name}_{metric}"] = e3_value
                row[f"delta_{name}_{metric}"] = (
                    e3_value - e1_value if e1_value is not None and e3_value is not None else None
                )
        payload["rows"].append(row)

    return payload


def write_paired_csv(rows: List[Dict[str, Any]], path: str) -> Optional[str]:
    if not rows:
        return None
    fieldnames = list(rows[0].keys())
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)
    return path


# ------------------------------------------------------- PRE-COMMITTED VERDICT


def classify_outcome(
    macro_e3: Optional[float],
    cg_dice_e3: Optional[float],
    pz_recall_e3: Optional[float],
) -> Dict[str, Any]:
    """Apply the PRE-COMMITTED E3 criteria, in a fixed precedence order.

    Precedence (documented, not discovered after the fact):
        1. REGRESSION            -- a hard failure takes priority over any gain
        2. IMPROVEMENT
        3. MECHANISTIC SUCCESS   -- PZ recall up, overall metric neutral
        4. SIMILAR
        5. INDETERMINATE         -- fits none of the above

    Thresholds are the literal constants in `E3_CRITERIA`, measured against the
    E1 reference values recorded there, NOT against a re-read of E1's artifacts.
    That is deliberate: the criteria are frozen, so they cannot drift if an
    upstream artifact is regenerated.
    """
    reference = E3_CRITERIA["e1_reference"]
    checks: Dict[str, Any] = {}

    if macro_e3 is None or cg_dice_e3 is None or pz_recall_e3 is None:
        return {
            "category": "indeterminate",
            "label": OUTCOME_LABELS["indeterminate"],
            "reason": "one or more of macro Dice / CG Dice / PZ recall is missing",
            "checks": checks,
        }

    macro_delta = macro_e3 - reference["macro_dice"]
    cg_delta = cg_dice_e3 - reference["cg_dice"]
    pz_recall_delta = pz_recall_e3 - reference["pz_recall"]

    regression_macro = macro_e3 <= E3_CRITERIA["regression"]["macro_dice_max"]
    regression_cg = cg_delta < -E3_CRITERIA["regression"]["cg_dice_max_drop_vs_e1"]
    improvement = (
        macro_e3 >= E3_CRITERIA["improvement"]["macro_dice_min"]
        and cg_dice_e3 >= E3_CRITERIA["improvement"]["cg_dice_min"]
    )
    macro_neutral = abs(macro_delta) <= E3_CRITERIA["mechanistic_success"]["macro_dice_tolerance"]
    mechanistic = (
        pz_recall_e3 >= E3_CRITERIA["mechanistic_success"]["pz_recall_min"] and macro_neutral
    )
    similar = (
        abs(macro_delta) <= E3_CRITERIA["similar"]["macro_dice_tolerance"]
        and pz_recall_delta < E3_CRITERIA["similar"]["pz_recall_gain_below"]
    )

    checks = {
        "macro_dice": macro_e3,
        "cg_dice": cg_dice_e3,
        "pz_recall": pz_recall_e3,
        "macro_delta_vs_e1": macro_delta,
        "cg_delta_vs_e1": cg_delta,
        "pz_recall_delta_vs_e1": pz_recall_delta,
        "regression_macro_le_0.7994": regression_macro,
        "regression_cg_drop_gt_0.005": regression_cg,
        "improvement_macro_ge_0.8194": macro_e3 >= E3_CRITERIA["improvement"]["macro_dice_min"],
        "improvement_cg_ge_0.8612": cg_dice_e3 >= E3_CRITERIA["improvement"]["cg_dice_min"],
        "mechanistic_pz_recall_ge_0.7461": (
            pz_recall_e3 >= E3_CRITERIA["mechanistic_success"]["pz_recall_min"]
        ),
        "mechanistic_macro_within_0.010": macro_neutral,
        "similar_macro_within_0.010": abs(macro_delta) <= E3_CRITERIA["similar"]["macro_dice_tolerance"],
        "similar_pz_recall_gain_lt_0.030": (
            pz_recall_delta < E3_CRITERIA["similar"]["pz_recall_gain_below"]
        ),
    }

    if regression_macro or regression_cg:
        reasons = []
        if regression_macro:
            reasons.append(f"macro Dice {macro_e3:.4f} <= 0.7994")
        if regression_cg:
            reasons.append(f"CG Dice fell {-cg_delta:.4f} vs E1 (> 0.005)")
        category, reason = "regression", "; ".join(reasons)
    elif improvement:
        category = "improvement"
        reason = f"macro Dice {macro_e3:.4f} >= 0.8194 AND CG Dice {cg_dice_e3:.4f} >= 0.8612"
    elif mechanistic:
        category = "mechanistic_success"
        reason = (
            f"PZ recall {pz_recall_e3:.4f} >= 0.7461 AND macro Dice within "
            f"+/-0.010 of E1 (delta {macro_delta:+.4f})"
        )
    elif similar:
        category = "similar"
        reason = (
            f"macro Dice within +/-0.010 of E1 (delta {macro_delta:+.4f}) AND PZ recall "
            f"gain {pz_recall_delta:+.4f} < 0.030"
        )
    else:
        category = "indeterminate"
        reason = (
            f"macro Dice {macro_e3:.4f} (delta {macro_delta:+.4f}), CG Dice {cg_dice_e3:.4f}, "
            f"PZ recall {pz_recall_e3:.4f} satisfy no pre-committed category"
        )

    return {
        "category": category,
        "label": OUTCOME_LABELS[category],
        "reason": reason,
        "checks": checks,
        "criteria": E3_CRITERIA,
    }


def _fmt(value: Optional[float], places: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _fmt_delta(value: Optional[float], places: int = 4) -> str:
    return "n/a" if value is None else f"{value:+.{places}f}"


def _fmt_p(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return "< 0.0001" if value < 0.0001 else f"{value:.4f}"


# ----------------------------------------------------------------------- PLOTS


def write_plots(
    history: List[Dict[str, Any]],
    comparison: Dict[str, Any],
    paired: Dict[str, Any],
    plots_dir: str,
) -> List[str]:
    """Render the standard E3 figures. Returns the paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)
    written: List[str] = []

    if history:
        epochs = [r["epoch"] for r in history]

        # ---- 1. loss curves, with the CE and Dice terms broken out ----------
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, [r.get("train_loss_total") for r in history], label="train total")
        if any(r.get("train_loss_ce") is not None for r in history):
            ax.plot(epochs, [r.get("train_loss_ce") for r in history],
                    label="train CE (class-weighted)", linestyle="--")
        if any(r.get("train_loss_dice") is not None for r in history):
            ax.plot(epochs, [r.get("train_loss_dice") for r in history],
                    label="train SoftDice (CG, PZ)", linestyle=":")
        if any(r.get("val_loss") is not None for r in history):
            ax.plot(epochs, [r.get("val_loss") for r in history],
                    label="validation total", color="black")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("E3 (class-weighted CE + SoftDice) -- loss curves")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        path = os.path.join(plots_dir, "training_loss.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

        # ---- 2. validation Dice proxy + per-class ---------------------------
        fig, ax = plt.subplots(figsize=(9, 5))
        proxy = [r.get("val_dice_proxy") for r in history]
        ax.plot(epochs, proxy,
                label="val Dice (batch-level proxy, drives checkpointing)", color="black")
        ax.plot(epochs, [r.get("val_dice_class1_CG") for r in history],
                label="val Dice CG (class 1, slice-level)")
        ax.plot(epochs, [r.get("val_dice_class2_PZ") for r in history],
                label="val Dice PZ (class 2, slice-level)")
        valid = [(e, v) for e, v in zip(epochs, proxy) if v is not None]
        if valid:
            best_epoch, best_value = max(valid, key=lambda t: t[1])
            ax.axvline(best_epoch, color="red", linestyle=":", alpha=0.8)
            ax.annotate(
                f"best epoch {best_epoch}\nproxy={best_value:.4f}",
                xy=(best_epoch, best_value), xytext=(8, -28),
                textcoords="offset points", color="red", fontsize=9,
            )
        ax.set_xlabel("epoch")
        ax.set_ylabel("Dice")
        ax.set_title(
            "E3 -- validation Dice per epoch (slice-level proxy; NOT the comparison metric)"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        path = os.path.join(plots_dir, "validation_dice.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # ---- 3. grouped bars across every available arm -------------------------
    labels: List[str] = []
    series: Dict[str, List[Optional[float]]] = {arm: [] for arm in ARMS}
    for class_id in (CLASS_CG, CLASS_PZ):
        for metric in ("dice", "iou", "precision", "recall"):
            cell = comparison["per_class"][class_id][metric]
            if cell["e1"] is None or cell["e3"] is None:
                continue
            labels.append(f"{CLASS_LABELS[class_id].split()[0]}\n{metric}")
            for arm in ARMS:
                series[arm].append(cell[arm])
    macro = comparison["macro"]
    if macro["e1"] is not None and macro["e3"] is not None:
        labels.append("macro\nDice")
        for arm in ARMS:
            series[arm].append(macro[arm])

    if labels:
        import numpy as np

        present = [arm for arm in ARMS if any(v is not None for v in series[arm])]
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(present))
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.4), 5))
        for index, arm in enumerate(present):
            values = [0.0 if v is None else v for v in series[arm]]
            offset = (index - (len(present) - 1) / 2) * width
            ax.bar(x + offset, values, width, label=ARM_LABELS[arm])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("score")
        ax.set_ylim(0, 1)
        ax.set_title("Validation (20 cases, per-case volume-level, original voxel space)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        path = os.path.join(plots_dir, "validation_metrics.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # ---- 4. paired per-case E3 - E1 -----------------------------------------
    rows = [r for r in paired.get("rows", []) if r.get("delta_macro_dice") is not None]
    if rows:
        import numpy as np

        rows = sorted(rows, key=lambda r: r["delta_macro_dice"])
        cases = [r["patient_id"] for r in rows]
        deltas = [r["delta_macro_dice"] for r in rows]
        pz_recall_deltas = [r.get("delta_pz_recall") for r in rows]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colors = ["tab:green" if d > 0 else ("tab:red" if d < 0 else "tab:gray") for d in deltas]
        axes[0].bar(np.arange(len(cases)), deltas, color=colors)
        axes[0].axhline(0, color="black", linewidth=0.8)
        axes[0].set_xticks(np.arange(len(cases)))
        axes[0].set_xticklabels(cases, rotation=90, fontsize=7)
        axes[0].set_ylabel("E3 - E1 macro Dice")
        axes[0].set_title("Per-case paired difference, macro Dice (E3 - E1)")
        axes[0].grid(alpha=0.3, axis="y")

        pairs = [(c, d) for c, d in zip(cases, pz_recall_deltas) if d is not None]
        if pairs:
            pz_cases = [p[0] for p in pairs]
            pz_deltas = [p[1] for p in pairs]
            pz_colors = [
                "tab:green" if d > 0 else ("tab:red" if d < 0 else "tab:gray") for d in pz_deltas
            ]
            axes[1].bar(np.arange(len(pz_cases)), pz_deltas, color=pz_colors)
            axes[1].axhline(0, color="black", linewidth=0.8)
            axes[1].set_xticks(np.arange(len(pz_cases)))
            axes[1].set_xticklabels(pz_cases, rotation=90, fontsize=7)
            axes[1].set_ylabel("E3 - E1 PZ recall")
            axes[1].set_title("Per-case paired difference, PZ recall (the E3 mechanism)")
            axes[1].grid(alpha=0.3, axis="y")
        fig.suptitle("E3 vs E1 -- paired per-case differences on the 20 validation cases")
        path = os.path.join(plots_dir, "e3_vs_e1_paired.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written


# ---------------------------------------------------------------------- REPORT


def render_report(
    comparison: Dict[str, Any],
    paired: Dict[str, Any],
    outcome: Dict[str, Any],
    history: List[Dict[str, Any]],
    summaries: Dict[str, Optional[Dict[str, Any]]],
    fingerprint: Optional[Dict[str, Any]],
    training_history: Optional[Dict[str, Any]],
    checkpoints: Dict[str, str],
    commit: Optional[str],
    best_epochs: Dict[str, Any],
) -> str:
    e3_summary = summaries["e3"] or {}
    e3_meta = e3_summary.get("metadata", {})
    macro = comparison["macro"]
    loss_spec = (fingerprint or {}).get("loss_spec") or {}
    class_weights = loss_spec.get("ce_class_weights")

    epochs_completed = max((r["epoch"] for r in history), default=None)
    proxy = [(r["epoch"], r.get("val_dice_proxy")) for r in history
             if r.get("val_dice_proxy") is not None]
    best_proxy_epoch, best_proxy_value = max(proxy, key=lambda t: t[1]) if proxy else (None, None)
    recorded_best_epoch = (training_history or {}).get("best_epoch")

    lines: List[str] = []
    lines.append("# Experiment E3 -- Class-Weighted Cross Entropy + Soft Dice (RQ-1)")
    lines.append("")
    lines.append("Generated by `scripts/finalize_e3_report.py` from on-disk artifacts only.")
    lines.append("This report does **not** select a final model and does **not** evaluate the")
    lines.append("official 19-case test set.")
    lines.append("")
    lines.append("> **E3 = E1 + median-frequency class-weighted CE, and nothing else.** The Dice")
    lines.append("> term, architecture, preprocessing, sampling, optimizer, learning rate, batch")
    lines.append("> size, seed, epoch budget and precision are all byte-for-byte E1's, and")
    lines.append("> augmentation stays OFF. `tests/test_e3_config.py` asserts that mechanically.")
    lines.append("")

    # ---- 1. identity --------------------------------------------------------
    lines.append("## 1. Experiment identity")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append("| Experiment ID | `exp_e3_dice_ce_pzweight` (E3) |")
    lines.append("| Independent variable | CE class weights (background, CG, PZ) |")
    lines.append(f"| CE class weights | `{class_weights if class_weights else 'n/a'}` |")
    lines.append("| CE normalization | PyTorch weighted mean `sum w_y l / sum w_y` "
                 "(**no** manual 1/N rescaling) |")
    lines.append(f"| Git commit | `{commit or 'unknown'}` |")
    lines.append("| Dataset | Prostate158 (official train archive) |")
    lines.append("| Target | `t2_anatomy_reader1.nii.gz` (0 = background, 1 = CG, 2 = PZ) |")
    lines.append("| Train / validation split | official 119 train / 20 validation cases |")
    lines.append("| Test set | **not accessed** |")
    lines.append("| External validation | **not accessed** |")
    lines.append("| Architecture | 2D U-Net (`ProstateUNet2D`), unchanged from Baseline / E1 |")
    if fingerprint:
        lines.append(f"| Loss | {fingerprint.get('loss')} |")
        lines.append(f"| Augmentation | {fingerprint.get('augmentation')} |")
        lines.append(f"| Optimizer | {fingerprint.get('optimizer')} |")
        lines.append(f"| Learning rate | {fingerprint.get('learning_rate')} |")
        lines.append(f"| Weight decay | {fingerprint.get('weight_decay')} |")
        lines.append(f"| Batch size | {fingerprint.get('batch_size')} |")
        lines.append(f"| Max epochs | {fingerprint.get('num_epochs')} |")
        lines.append(f"| Seed | {fingerprint.get('seed')} |")
        lines.append(f"| AMP | {'ON' if fingerprint.get('amp') else 'OFF'} |")
        lines.append(f"| Precision | {fingerprint.get('precision')} |")
        lines.append(f"| Scheduler | {fingerprint.get('scheduler')} |")
        lines.append(f"| Weighted slice sampling | "
                     f"{'ON' if fingerprint.get('use_weighted_sampling') else 'OFF'} |")
        lines.append(f"| Preprocessing | {fingerprint.get('preprocessing')} |")
    lines.append(f"| Epochs completed | {epochs_completed if epochs_completed is not None else 'n/a'} |")
    best_epoch_display = recorded_best_epoch if recorded_best_epoch is not None else best_proxy_epoch
    lines.append(f"| Best checkpoint epoch | {best_epoch_display if best_epoch_display is not None else 'n/a'} |")
    lines.append(f"| Best within-run val Dice proxy | {_fmt(best_proxy_value, 6)} |")
    lines.append(f"| Best checkpoint | `{checkpoints.get('best', 'n/a')}` |")
    lines.append(f"| Latest checkpoint | `{checkpoints.get('latest', 'n/a')}` |")
    lines.append("")
    lines.append("> The best checkpoint is **not** assumed to be the last epoch. Selection uses")
    lines.append("> the unchanged within-run batch-level foreground-macro validation Dice proxy,")
    lines.append("> identical to Baseline / E1 / E2, which is what makes the arms comparable.")
    lines.append("")

    # ---- 2. why this factor -------------------------------------------------
    lines.append("## 2. Why class weighting, and why these weights")
    lines.append("")
    lines.append("Voxel census over the 119 official training cases (this repository's own")
    lines.append("measurement): background 94.20% | CG 4.30% | PZ 1.51%. Median class frequency")
    lines.append("= 0.0430, so the median-frequency weights are 0.0430/f_c:")
    lines.append("")
    lines.append("| Class | Voxel frequency | Weight |")
    lines.append("|---|---|---|")
    lines.append("| background (0) | 94.20% | 0.0456 |")
    lines.append("| CG (1) | 4.30% | 1.0000 |")
    lines.append("| PZ (2) | 1.51% | 2.8477 |")
    lines.append("")
    reference = E3_CRITERIA["e1_reference"]
    lines.append(f"E1 still shows PZ precision {reference['pz_precision']:.4f} >> PZ recall")
    lines.append(f"{reference['pz_recall']:.4f} on the 20 validation cases, i.e. PZ remains")
    lines.append("systematically **under-segmented** even with the foreground soft-Dice term in")
    lines.append("place. E1's Dice term is region-based but balanced only across the two")
    lines.append("foreground zones; its CE term is still purely pixel-wise and unweighted, so")
    lines.append("94.2% of the pixels it scores are background. E3 targets exactly that term.")
    lines.append("")

    # ---- 3. run continuity --------------------------------------------------
    lines.append("## 3. Run continuity")
    lines.append("")
    lines.append(f"* Per-epoch records on disk: **{len(history)}**")
    if epochs_completed:
        gaps = sorted(set(range(1, epochs_completed + 1)) - {r["epoch"] for r in history})
        if gaps:
            lines.append(
                f"* **Missing epoch record(s): {gaps}** -- training itself was continuous, but "
                "these rows were never written."
            )
        else:
            lines.append(
                f"* Complete, contiguous history for epochs 1..{epochs_completed} -- no epoch "
                "was lost across any Colab interruption."
            )
    lines.append(
        "* History is keyed by epoch and written atomically after every epoch, so a resumed "
        "session appends to the existing record instead of replacing it."
    )
    lines.append("")

    # ---- 4. primary result --------------------------------------------------
    lines.append("## 4. Primary result -- validation macro Dice (OBSERVED FACT)")
    lines.append("")
    lines.append(
        "Per-case, volume-level, original-voxel-space Dice over the 20 official "
        "validation cases. Macro Dice = (mean CG Dice + mean PZ Dice) / 2."
    )
    lines.append("")
    lines.append("| Metric | Baseline | E1 | E2 | **E3** | E3 - E1 | E3 - Baseline |")
    lines.append("|---|---|---|---|---|---|---|")
    for class_id in (CLASS_CG, CLASS_PZ):
        cell = comparison["per_class"][class_id]["dice"]
        lines.append(
            f"| {CLASS_LABELS[class_id]} Dice | {_fmt(cell['baseline'])} | {_fmt(cell['e1'])} | "
            f"{_fmt(cell['e2'])} | **{_fmt(cell['e3'])}** | "
            f"{_fmt_delta(cell['delta_vs_e1'])} | {_fmt_delta(cell['delta_vs_baseline'])} |"
        )
    lines.append(
        f"| **Macro Dice (CG+PZ)/2** | {_fmt(macro['baseline'])} | {_fmt(macro['e1'])} | "
        f"{_fmt(macro['e2'])} | **{_fmt(macro['e3'])}** | "
        f"{_fmt_delta(macro['delta_vs_e1'])} | {_fmt_delta(macro['delta_vs_baseline'])} |"
    )
    lines.append("")
    macro_stats = (paired.get("metrics") or {}).get("macro_dice")
    if macro_stats:
        lines.append(
            f"Per-case macro Dice, E3: mean {_fmt(macro_stats['e3_mean'])} +/- "
            f"{_fmt(macro_stats['e3_sd'])} SD (n = {macro_stats['n']}); "
            f"E1: {_fmt(macro_stats['e1_mean'])} +/- {_fmt(macro_stats['e1_sd'])} SD."
        )
        lines.append("")

    # ---- 5. pre-committed verdict ------------------------------------------
    lines.append("## 5. Pre-committed success criteria (INTERPRETATION)")
    lines.append("")
    lines.append("These thresholds were fixed **before** the run and are hard-coded in")
    lines.append("`scripts/finalize_e3_report.py::E3_CRITERIA`. They were not adjusted after")
    lines.append("seeing the numbers. Categories are applied in a fixed precedence:")
    lines.append("regression -> improvement -> mechanistic success -> similar -> indeterminate.")
    lines.append("")
    lines.append("| Category | Condition | Satisfied? |")
    lines.append("|---|---|---|")
    checks = outcome.get("checks", {})
    lines.append(
        "| Improvement | macro Dice >= 0.8194 AND CG Dice >= 0.8612 | "
        f"{checks.get('improvement_macro_ge_0.8194')} / {checks.get('improvement_cg_ge_0.8612')} |"
    )
    lines.append(
        "| Mechanistic success | PZ recall >= 0.7461 AND macro Dice within +/-0.010 of E1 | "
        f"{checks.get('mechanistic_pz_recall_ge_0.7461')} / "
        f"{checks.get('mechanistic_macro_within_0.010')} |"
    )
    lines.append(
        "| Similar | macro Dice within +/-0.010 of E1 AND PZ recall gain < 0.030 | "
        f"{checks.get('similar_macro_within_0.010')} / "
        f"{checks.get('similar_pz_recall_gain_lt_0.030')} |"
    )
    lines.append(
        "| Regression | macro Dice <= 0.7994 OR CG Dice drops > 0.005 vs E1 | "
        f"{checks.get('regression_macro_le_0.7994')} / {checks.get('regression_cg_drop_gt_0.005')} |"
    )
    lines.append("")
    lines.append(f"**E3 outcome category: {outcome['label']}.**")
    lines.append("")
    lines.append(f"Reason: {outcome['reason']}")
    lines.append("")
    lines.append("A single metric moving in the right direction is **not** a success. The")
    lines.append("category above is the only verdict this report asserts.")
    lines.append("")

    # ---- 6. the mechanism ---------------------------------------------------
    lines.append("## 6. Did the mechanism act? PZ recall / precision (OBSERVED FACT)")
    lines.append("")
    lines.append("Class weighting is expected to trade PZ precision for PZ recall. Both are")
    lines.append("reported so the trade is visible rather than implied.")
    lines.append("")
    lines.append("| Class | Metric | Baseline | E1 | E2 | E3 | E3 - E1 |")
    lines.append("|---|---|---|---|---|---|---|")
    for class_id in (CLASS_CG, CLASS_PZ):
        for metric in ("recall", "precision"):
            cell = comparison["per_class"][class_id][metric]
            lines.append(
                f"| {CLASS_LABELS[class_id]} | {metric} | {_fmt(cell['baseline'])} | "
                f"{_fmt(cell['e1'])} | {_fmt(cell['e2'])} | {_fmt(cell['e3'])} | "
                f"{_fmt_delta(cell['delta_vs_e1'])} |"
            )
    lines.append("")

    # ---- 7. secondary metrics ----------------------------------------------
    lines.append("## 7. All secondary metrics (validation, 20 cases)")
    lines.append("")
    lines.append("| Class | Metric | Baseline | E1 | E2 | E3 | E3 SD | E3 - E1 | Direction |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for class_id in (CLASS_CG, CLASS_PZ):
        for metric in REPORT_METRICS:
            cell = comparison["per_class"][class_id][metric]
            if cell["e3"] is None and cell["e1"] is None:
                continue
            direction = "lower is better" if cell["lower_is_better"] else "higher is better"
            lines.append(
                f"| {CLASS_LABELS[class_id]} | {metric} | {_fmt(cell['baseline'])} | "
                f"{_fmt(cell['e1'])} | {_fmt(cell['e2'])} | {_fmt(cell['e3'])} | "
                f"{_fmt(cell['e3_sd'])} | {_fmt_delta(cell['delta_vs_e1'])} | {direction} |"
            )
    lines.append("")
    lines.append(
        "Every metric produced by the evaluation is listed, including those where E3 is worse, "
        "so no metric can be cherry-picked after the fact."
    )
    lines.append("")

    # ---- 8. full four-arm comparison table ---------------------------------
    lines.append("## 8. Baseline vs E1 vs E2 vs E3 (OBSERVED FACT)")
    lines.append("")
    header = (
        "| Experiment | Main intervention | Best epoch | CG Dice | PZ Dice | Macro Dice | "
        "CG prec | CG rec | PZ prec | PZ rec | CG HD95 | PZ HD95 | CG ASD | PZ ASD |"
    )
    lines.append(header)
    lines.append("|" + "---|" * 14)
    for arm in ARMS:
        def value(class_id: str, metric: str, _arm: str = arm) -> Optional[float]:
            return comparison["per_class"][class_id][metric][_arm]

        lines.append(
            f"| {ARM_LABELS[arm]} | {ARM_INTERVENTIONS[arm]} | "
            f"{best_epochs.get(arm, 'n/a')} | "
            f"{_fmt(value(CLASS_CG, 'dice'))} | {_fmt(value(CLASS_PZ, 'dice'))} | "
            f"{_fmt(comparison['macro'][arm])} | "
            f"{_fmt(value(CLASS_CG, 'precision'))} | {_fmt(value(CLASS_CG, 'recall'))} | "
            f"{_fmt(value(CLASS_PZ, 'precision'))} | {_fmt(value(CLASS_PZ, 'recall'))} | "
            f"{_fmt(value(CLASS_CG, 'hd95_mm'), 3)} | {_fmt(value(CLASS_PZ, 'hd95_mm'), 3)} | "
            f"{_fmt(value(CLASS_CG, 'asd_mm'), 3)} | {_fmt(value(CLASS_PZ, 'asd_mm'), 3)} |"
        )
    lines.append("")
    lines.append("### E1 -> E3 differences (the nested comparison)")
    lines.append("")
    lines.append("| Difference | Value |")
    lines.append("|---|---|")
    lines.append(f"| macro Dice | {_fmt_delta(macro['delta_vs_e1'])} |")
    lines.append(f"| PZ Dice | {_fmt_delta(comparison['per_class'][CLASS_PZ]['dice']['delta_vs_e1'])} |")
    lines.append(f"| PZ recall | {_fmt_delta(comparison['per_class'][CLASS_PZ]['recall']['delta_vs_e1'])} |")
    lines.append(f"| CG Dice | {_fmt_delta(comparison['per_class'][CLASS_CG]['dice']['delta_vs_e1'])} |")
    lines.append("")
    lines.append("> E1 and E2 vary *different* factors, so an E2-E3 comparison is descriptive")
    lines.append("> context only. E3 is nested in E1, so **E1 is the reference arm**.")
    lines.append("")

    # ---- 9. paired statistics ----------------------------------------------
    lines.append("## 9. Paired comparison against E1 (STATISTICAL RESULT)")
    lines.append("")
    if not paired.get("available"):
        lines.append(f"Not available: {paired.get('reason')}")
        lines.append("")
    else:
        lines.append(
            f"Same {paired['n_paired_cases']} validation cases evaluated under both arms, so the "
            "comparison is paired. Bootstrap CIs use the project's existing percentile bootstrap "
            "(`src.metrics.bootstrap_ci`, 2000 resamples, seed 42). The Wilcoxon signed-rank "
            "p-values are two-sided and **descriptive on n = 20** -- they do not gate the "
            "pre-committed criteria in section 5."
        )
        lines.append("")
        lines.append(
            "| Quantity | E1 mean | E3 mean | mean diff | 95% CI of diff | won | lost | tied | "
            "Wilcoxon W | p |"
        )
        lines.append("|" + "---|" * 10)
        interesting = [
            ("macro_dice", "macro Dice"),
            (f"class{CLASS_PZ}_dice", "PZ Dice"),
            (f"class{CLASS_PZ}_recall", "PZ recall"),
            (f"class{CLASS_PZ}_precision", "PZ precision"),
            (f"class{CLASS_CG}_dice", "CG Dice"),
            (f"class{CLASS_CG}_recall", "CG recall"),
            (f"class{CLASS_CG}_precision", "CG precision"),
            (f"class{CLASS_PZ}_hd95_mm", "PZ HD95 (mm)"),
            (f"class{CLASS_CG}_hd95_mm", "CG HD95 (mm)"),
            (f"class{CLASS_PZ}_asd_mm", "PZ ASD (mm)"),
            (f"class{CLASS_CG}_asd_mm", "CG ASD (mm)"),
        ]
        for key, label in interesting:
            cell = paired["metrics"].get(key)
            if not cell:
                continue
            ci = cell["difference_bootstrap_ci95"]
            ci_text = (
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci and ci[0] is not None else "n/a"
            )
            wil = cell["wilcoxon"]
            suffix = " (lower is better)" if cell["lower_is_better"] else ""
            lines.append(
                f"| {label}{suffix} | {_fmt(cell['e1_mean'])} | {_fmt(cell['e3_mean'])} | "
                f"{_fmt_delta(cell['mean_difference'])} | {ci_text} | {cell['cases_won']} | "
                f"{cell['cases_lost']} | {cell['cases_tied']} | "
                f"{_fmt(wil['statistic'], 1) if wil['statistic'] is not None else 'n/a'} | "
                f"{_fmt_p(wil['p_value'])} |"
            )
        lines.append("")
        macro_cell = paired["metrics"].get("macro_dice")
        if macro_cell:
            lines.append(
                f"**Validation cases won/lost against E1 on macro Dice: "
                f"{macro_cell['cases_won']} won / {macro_cell['cases_lost']} lost / "
                f"{macro_cell['cases_tied']} tied (n = {macro_cell['n']}).**"
            )
            wil = macro_cell["wilcoxon"]
            if wil.get("note"):
                lines.append("")
                lines.append(f"Wilcoxon note: {wil['note']}.")
            lines.append("")

        # ---- per-case table ----
        lines.append("### Per-case macro Dice (E1 vs E3)")
        lines.append("")
        lines.append("| Case | E1 macro | E3 macro | delta | E1 PZ recall | E3 PZ recall | delta |")
        lines.append("|---|---|---|---|---|---|---|")
        for row in sorted(paired["rows"], key=lambda r: r["patient_id"]):
            lines.append(
                f"| {row['patient_id']} | {_fmt(row.get('e1_macro_dice'))} | "
                f"{_fmt(row.get('e3_macro_dice'))} | {_fmt_delta(row.get('delta_macro_dice'))} | "
                f"{_fmt(row.get('e1_pz_recall'))} | {_fmt(row.get('e3_pz_recall'))} | "
                f"{_fmt_delta(row.get('delta_pz_recall'))} |"
            )
        lines.append("")

    # ---- 10. provenance / limitations --------------------------------------
    lines.append("## 10. Provenance, integrity and limitations")
    lines.append("")
    lines.append(f"* Validation cases evaluated: {e3_meta.get('n_cases', 'n/a')}")
    lines.append(f"* Evaluation split: `{e3_meta.get('split', 'val')}`")
    lines.append(
        f"* Checkpoint evaluated: `{e3_meta.get('checkpoint_path', checkpoints.get('best', 'n/a'))}`"
    )
    lines.append("* The official 19-case test set was NOT used for training, checkpoint")
    lines.append("  selection, or any comparison in this report.")
    lines.append("* E3 was trained from a fresh initialization; it was not warm-started from the")
    lines.append("  Baseline, E1 or E2 checkpoint (enforced by")
    lines.append("  `src/train.py::verify_checkpoint_compatibility`, which now also compares")
    lines.append("  `loss.ce_class_weights`).")
    lines.append("* **Label mapping limitation (carried forward, unchanged):** class 1 is treated")
    lines.append("  as CG and class 2 as PZ, matching Baseline / E1 / E2. That identification is")
    lines.append("  still formally UNVERIFIED -- see")
    lines.append("  `results/label_mapping/label_mapping_report.md`. Every \"PZ\" statement in")
    lines.append("  this report is therefore a statement about class 2 under that hypothesis. The")
    lines.append("  mapping was deliberately NOT changed for E3, so the arms stay comparable.")
    lines.append("* n = 20 validation cases. The Wilcoxon p-values and bootstrap CIs are")
    lines.append("  descriptive at that sample size and are not treated as confirmatory.")
    lines.append("")

    # ---- 11. scope ---------------------------------------------------------
    lines.append("## 11. Not concluded here")
    lines.append("")
    lines.append("* No final model is selected.")
    lines.append("* No test-set or external-validation evaluation is performed.")
    lines.append("* No post-processing, test-time augmentation or architectural change was")
    lines.append("  introduced; E3 remains a single-factor experiment.")
    lines.append("")

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ MAIN


def _best_epoch_of(arm_dir: Optional[str]) -> Any:
    """Best checkpoint epoch for an arm, read from its own artifacts."""
    if not arm_dir:
        return "n/a"
    summary = _load_json(os.path.join(arm_dir, "summary.json")) or {}
    if summary.get("best_checkpoint_epoch") is not None:
        return summary["best_checkpoint_epoch"]
    history = _load_json(os.path.join(arm_dir, "training_history.json")) or {}
    if history.get("best_epoch") is not None:
        return history["best_epoch"]
    evaluation = _load_json(os.path.join(arm_dir, "eval_val_summary.json")) or {}
    return (evaluation.get("metadata") or {}).get("checkpoint_epoch", "n/a")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--e3-dir", default="results/exp_e3_dice_ce_pzweight",
                        help="Directory holding the E3 run artifacts.")
    parser.add_argument("--e1-dir", default="results/exp_e1_dice_ce",
                        help="Directory holding the E1 run artifacts (the reference arm).")
    parser.add_argument("--baseline-dir", default="results/exp01_baseline",
                        help="Directory holding the baseline run artifacts (context anchor).")
    parser.add_argument("--e2-dir", default="results/exp_e2_ce_aug",
                        help="Directory holding the E2 run artifacts (optional context anchor).")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    args = parser.parse_args(argv)

    e3_dir = os.path.abspath(args.e3_dir)
    e1_dir = os.path.abspath(args.e1_dir) if args.e1_dir else None
    baseline_dir = os.path.abspath(args.baseline_dir) if args.baseline_dir else None
    e2_dir = os.path.abspath(args.e2_dir) if args.e2_dir else None

    summaries: Dict[str, Optional[Dict[str, Any]]] = {
        "e3": _load_summary(os.path.join(e3_dir, "eval_val_summary.json"), "E3", required=True),
        "e1": _load_summary(
            os.path.join(e1_dir, "eval_val_summary.json"), "E1", required=True
        ) if e1_dir else None,
        "baseline": _load_summary(
            os.path.join(baseline_dir, "eval_val_summary.json"), "Baseline", required=False
        ) if baseline_dir else None,
        "e2": _load_summary(
            os.path.join(e2_dir, "eval_val_summary.json"), "E2", required=False
        ) if e2_dir else None,
    }
    if summaries["e1"] is None:
        raise FileNotFoundError(
            "E1 is the reference arm for E3; its validation summary is required. "
            f"Expected: {os.path.join(e1_dir or '<--e1-dir>', 'eval_val_summary.json')}"
        )
    for arm in ("baseline", "e2"):
        if summaries[arm] is None:
            print(f"NOTE: no {arm} validation summary found -- it will show as n/a in the tables.")

    history = read_metrics_csv(os.path.join(e3_dir, "metrics.csv"))
    fingerprint = _load_json(os.path.join(e3_dir, "config_fingerprint.json"))
    training_history = _load_json(os.path.join(e3_dir, "training_history.json"))
    comparison = build_comparison(summaries)
    commit = git_commit()

    e3_per_case = read_per_case_metrics(os.path.join(e3_dir, "eval_val_per_case_metrics.csv"))
    e1_per_case = read_per_case_metrics(
        os.path.join(e1_dir, "eval_val_per_case_metrics.csv")
    ) if e1_dir else {}
    paired = paired_comparison(e3_per_case, e1_per_case)
    paired_csv = write_paired_csv(paired.get("rows", []), os.path.join(e3_dir, "e3_vs_e1_paired_cases.csv"))

    outcome = classify_outcome(
        macro_e3=comparison["macro"]["e3"],
        cg_dice_e3=comparison["per_class"][CLASS_CG]["dice"]["e3"],
        pz_recall_e3=comparison["per_class"][CLASS_PZ]["recall"]["e3"],
    )

    checkpoints = {
        "best": os.path.join(e3_dir, "best_model.pt"),
        "latest": os.path.join(e3_dir, "latest_checkpoint.pt"),
    }
    best_epochs = {
        "baseline": _best_epoch_of(baseline_dir),
        "e1": _best_epoch_of(e1_dir),
        "e2": _best_epoch_of(e2_dir),
        "e3": _best_epoch_of(e3_dir),
    }

    plots: List[str] = []
    if not args.no_plots:
        try:
            plots = write_plots(history, comparison, paired, os.path.join(e3_dir, "plots"))
        except ImportError:
            print("WARNING: matplotlib is unavailable -- skipping figures.")

    epochs_completed = max((r["epoch"] for r in history), default=None)
    proxy = [(r["epoch"], r.get("val_dice_proxy")) for r in history
             if r.get("val_dice_proxy") is not None]
    best_proxy_epoch, best_proxy_value = max(proxy, key=lambda t: t[1]) if proxy else (None, None)
    macro = comparison["macro"]

    summary_payload = {
        "experiment": "exp_e3_dice_ce_pzweight",
        "experiment_label": "E3 (Dice + class-weighted Cross Entropy)",
        "independent_variable": {
            "field": "loss.ce_class_weights",
            "value": (fingerprint or {}).get("ce_class_weights"),
            "class_order": ["background", "CG", "PZ"],
            "normalization": "PyTorch weighted mean (sum w_y l / sum w_y); no manual 1/N",
            "reference_arm": "exp_e1_dice_ce",
            "single_factor": True,
        },
        "design_cell": {"loss": "dice_ce", "ce_class_weighted": True, "augmentation": False},
        "git_commit": commit,
        "dataset": "Prostate158 (official train archive)",
        "target_mask": "t2_anatomy_reader1.nii.gz",
        "split": {"train_cases": 119, "val_cases": 20, "test_accessed": False},
        "fingerprint": fingerprint,
        "epochs_completed": epochs_completed,
        "epoch_records_on_disk": len(history),
        "best_checkpoint_epoch": (training_history or {}).get("best_epoch", best_proxy_epoch),
        "best_val_dice_proxy": best_proxy_value,
        "best_epochs_all_arms": best_epochs,
        "checkpoints": checkpoints,
        "per_class": {
            class_id: {
                metric: comparison["per_class"][class_id][metric] for metric in REPORT_METRICS
            }
            for class_id in (CLASS_CG, CLASS_PZ)
        },
        "macro_dice": macro,
        "paired_vs_e1": paired,
        "paired_cases_csv": paired_csv,
        "outcome": outcome,
        "verdict": outcome["label"],
        "plots": plots,
        "label_mapping_status": (
            "UNVERIFIED -- class 1 assumed CG, class 2 assumed PZ, identical to "
            "Baseline / E1 / E2. See results/label_mapping/label_mapping_report.md."
        ),
        "scope_note": (
            "E3 is reported against E1 (its reference arm), with the Baseline and E2 as "
            "context. No final model selection and no test-set evaluation are performed here."
        ),
    }

    summary_path = os.path.join(e3_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    report_path = os.path.join(e3_dir, "training_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_report(
            comparison, paired, outcome, history, summaries, fingerprint,
            training_history, checkpoints, commit, best_epochs,
        ))

    # ------------------------------------------------------------------ stdout
    print("=" * 82)
    print("E3 REPORT WRITTEN")
    print("=" * 82)
    print(f"  summary.json      : {summary_path}")
    print(f"  training_report.md: {report_path}")
    if paired_csv:
        print(f"  paired cases CSV  : {paired_csv}")
    for path in plots:
        print(f"  plot              : {path}")
    print("-" * 82)
    n_cases = ((summaries["e3"] or {}).get("metadata") or {}).get("n_cases", "?")
    print(f"  Validation macro Dice ({n_cases} cases, per-case, original voxel space)")
    for arm in ARMS:
        print(f"    {ARM_LABELS[arm]:<34}: {_fmt(macro[arm])}")
    print(f"    E3 - E1  : {_fmt_delta(macro['delta_vs_e1'])}")
    print(f"    E3 - Base: {_fmt_delta(macro['delta_vs_baseline'])}")
    print("-" * 82)
    for class_id in (CLASS_CG, CLASS_PZ):
        for metric in ("dice", "recall", "precision"):
            cell = comparison["per_class"][class_id][metric]
            print(f"  {CLASS_LABELS[class_id]} {metric:<10}: E1 {_fmt(cell['e1'])} | "
                  f"E3 {_fmt(cell['e3'])} ({_fmt_delta(cell['delta_vs_e1'])})")
    print("-" * 82)
    macro_cell = (paired.get("metrics") or {}).get("macro_dice")
    if macro_cell:
        wil = macro_cell["wilcoxon"]
        print(f"  Paired vs E1 (macro Dice): {macro_cell['cases_won']} won / "
              f"{macro_cell['cases_lost']} lost / {macro_cell['cases_tied']} tied "
              f"(n={macro_cell['n']})")
        print(f"  Wilcoxon signed-rank     : W={_fmt(wil['statistic'], 1) if wil['statistic'] is not None else 'n/a'}, "
              f"p={_fmt_p(wil['p_value'])}")
    print("-" * 82)
    print(f"  PRE-COMMITTED OUTCOME: {outcome['label']}")
    print(f"    {outcome['reason']}")
    print("=" * 82)
    print("No final model selected. Test set untouched. Label mapping remains UNVERIFIED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
