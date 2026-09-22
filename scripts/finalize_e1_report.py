"""Assemble the Experiment E1 deliverables: summary.json, training_report.md, plots.

Reads only artifacts that already exist on disk -- it NEVER trains, never runs
inference, and never touches the official 19-case test set. Specifically it
consumes:

  * <e1_dir>/metrics.csv           -- per-epoch history written by run_training
  * <e1_dir>/eval_val_summary.json -- 20-case volume-level validation evaluation
                                      written by src/evaluate.py::run_evaluation
  * <baseline_dir>/eval_val_summary.json -- the same, for the Experiment-1 baseline

and writes:

  * <e1_dir>/summary.json          -- machine-readable E1 result + baseline delta
  * <e1_dir>/training_report.md    -- human-readable report for the thesis
  * <e1_dir>/plots/training_loss.png
  * <e1_dir>/plots/validation_dice.png
  * <e1_dir>/plots/validation_metrics.png

Comparison metric (pre-committed in Outputs/rq1_experiment_specification.md):

    validation macro Dice = (mean CG Dice + mean PZ Dice) / 2

computed per case, at volume level, in original voxel space, over the 20
official validation cases. CG and PZ are ALSO always reported separately, and
every available secondary metric (IoU, HD95, ASD, precision, recall) is carried
into the comparison table so no metric can be cherry-picked.

Usage
-----
    python scripts/finalize_e1_report.py \
        --e1-dir results/exp_e1_dice_ce \
        --baseline-dir results/exp01_baseline
"""

import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

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


def _load_summary(path: str, what: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{what} evaluation summary not found: {path}\n"
            "Run src/evaluate.py::run_evaluation(..., split='val') for that arm first."
        )
    with open(path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    metadata = summary.get("metadata", {})
    # Hard integrity guard: this script must never be fed a test-set evaluation.
    if metadata.get("is_official_held_out_test") or metadata.get("split") == "test":
        raise ValueError(
            f"REFUSING to build the E1 comparison from a TEST-set evaluation ({path}). "
            "E1 model selection and comparison must use the 20-case validation split "
            "only; the official 19-case test set is reserved for the single final "
            "evaluation after the RQ-1 configuration is chosen."
        )
    return summary


def _metric(summary: Dict[str, Any], class_id: str, metric: str) -> Optional[float]:
    try:
        return summary["per_class"][class_id]["metrics"][metric]["mean"]
    except (KeyError, TypeError):
        return None


def _macro_dice(summary: Dict[str, Any]) -> Optional[float]:
    cg = _metric(summary, CLASS_CG, "dice")
    pz = _metric(summary, CLASS_PZ, "dice")
    if cg is None or pz is None:
        return None
    return (cg + pz) / 2.0


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
    return rows


def build_comparison(baseline: Dict[str, Any], e1: Dict[str, Any]) -> Dict[str, Any]:
    """Per-class, per-metric baseline-vs-E1 table plus the macro-Dice decision."""
    comparison: Dict[str, Any] = {"per_class": {}, "macro": {}}

    for class_id in (CLASS_CG, CLASS_PZ):
        entry = {}
        for metric in REPORT_METRICS:
            base_value = _metric(baseline, class_id, metric)
            e1_value = _metric(e1, class_id, metric)
            delta = None if (base_value is None or e1_value is None) else e1_value - base_value
            entry[metric] = {
                "baseline": base_value,
                "e1": e1_value,
                "delta": delta,
                "lower_is_better": metric in LOWER_IS_BETTER,
            }
        comparison["per_class"][class_id] = entry

    base_macro = _macro_dice(baseline)
    e1_macro = _macro_dice(e1)
    macro_delta = None if (base_macro is None or e1_macro is None) else e1_macro - base_macro
    comparison["macro"] = {
        "metric": "macro_dice",
        "definition": "(mean CG Dice + mean PZ Dice) / 2, per-case volume-level, original voxel space",
        "baseline": base_macro,
        "e1": e1_macro,
        "delta": macro_delta,
    }
    comparison["verdict"] = interpret(macro_delta)
    return comparison


def interpret(macro_delta: Optional[float], tolerance: float = 0.005) -> str:
    """Classify the primary outcome. Deliberately conservative and symmetric.

    `tolerance` is a "practically unchanged" band around zero; it is NOT a
    success threshold and it is applied identically in both directions.
    """
    if macro_delta is None:
        return "undetermined (missing metrics)"
    if macro_delta > tolerance:
        return "improved"
    if macro_delta < -tolerance:
        return "regressed"
    return "similar (within +/-{:.3f} macro Dice of baseline)".format(tolerance)


def _fmt(value: Optional[float], places: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


def _fmt_delta(value: Optional[float], places: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.{places}f}"


def write_plots(history: List[Dict[str, Any]], comparison: Dict[str, Any], plots_dir: str) -> List[str]:
    """Render the three standard E1 figures. Returns the paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)
    written: List[str] = []

    # ---- 1. training / validation loss curves -------------------------------
    if history:
        epochs = [r["epoch"] for r in history]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, [r.get("train_loss_total") for r in history], label="train total (CE + Dice)")
        ax.plot(epochs, [r.get("train_loss_ce") for r in history], label="train CE term", linestyle="--")
        ax.plot(epochs, [r.get("train_loss_dice") for r in history], label="train Dice term", linestyle="--")
        if any(r.get("val_loss") is not None for r in history):
            ax.plot(epochs, [r.get("val_loss") for r in history], label="validation total", color="black")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("E1 (Dice + CE) -- loss curves")
        ax.legend()
        ax.grid(alpha=0.3)
        path = os.path.join(plots_dir, "training_loss.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

        # ---- 2. validation Dice proxy + per-class ---------------------------
        fig, ax = plt.subplots(figsize=(9, 5))
        proxy = [r.get("val_dice_proxy") for r in history]
        ax.plot(epochs, proxy, label="val Dice (batch-level proxy, drives checkpointing)", color="black")
        ax.plot(epochs, [r.get("val_dice_class1_CG") for r in history], label="val Dice CG (class 1, slice-level)")
        ax.plot(epochs, [r.get("val_dice_class2_PZ") for r in history], label="val Dice PZ (class 2, slice-level)")
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
        ax.set_title("E1 -- validation Dice per epoch (slice-level proxy; NOT the comparison metric)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        path = os.path.join(plots_dir, "validation_dice.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # ---- 3. baseline vs E1, volume-level validation metrics -----------------
    labels, baseline_values, e1_values = [], [], []
    for class_id in (CLASS_CG, CLASS_PZ):
        for metric in ("dice", "iou", "precision", "recall"):
            cell = comparison["per_class"][class_id][metric]
            if cell["baseline"] is None or cell["e1"] is None:
                continue
            labels.append(f"{CLASS_LABELS[class_id].split()[0]}\n{metric}")
            baseline_values.append(cell["baseline"])
            e1_values.append(cell["e1"])
    macro = comparison["macro"]
    if macro["baseline"] is not None and macro["e1"] is not None:
        labels.append("macro\nDice")
        baseline_values.append(macro["baseline"])
        e1_values.append(macro["e1"])

    if labels:
        import numpy as np

        x = np.arange(len(labels))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.2), 5))
        ax.bar(x - width / 2, baseline_values, width, label="Baseline (CE)")
        ax.bar(x + width / 2, e1_values, width, label="E1 (Dice + CE)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel("score")
        ax.set_ylim(0, 1)
        ax.set_title("Validation (20 cases, per-case volume-level, original voxel space)")
        ax.legend()
        ax.grid(alpha=0.3, axis="y")
        path = os.path.join(plots_dir, "validation_metrics.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written


def render_report(
    comparison: Dict[str, Any],
    history: List[Dict[str, Any]],
    e1_summary: Dict[str, Any],
    baseline_summary: Dict[str, Any],
    fingerprint: Optional[Dict[str, Any]],
) -> str:
    e1_meta = e1_summary.get("metadata", {})
    base_meta = baseline_summary.get("metadata", {})
    macro = comparison["macro"]

    lines: List[str] = []
    lines.append("# Experiment E1 -- Dice + Cross Entropy (RQ-1)")
    lines.append("")
    lines.append("Generated by `scripts/finalize_e1_report.py` from on-disk artifacts only.")
    lines.append("")

    lines.append("## 1. Experiment identity")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append("| Experiment | E1_DiceCE (`exp_e1_dice_ce`) |")
    lines.append("| Architecture | 2D U-Net (`ProstateUNet2D`), unchanged from baseline |")
    lines.append("| Loss | 1.0 x Cross Entropy + 1.0 x Soft Dice |")
    lines.append("| Dice classes | CG (1) + PZ (2); background excluded |")
    lines.append("| Dice epsilon | 1e-6 |")
    if fingerprint:
        lines.append(f"| Max epochs | {fingerprint.get('num_epochs')} |")
        lines.append(f"| Seed | {fingerprint.get('seed')} |")
        lines.append(f"| AMP | {'ON' if fingerprint.get('amp') else 'OFF'} |")
        lines.append(f"| Precision | {fingerprint.get('precision')} |")
        lines.append(f"| Optimizer | {fingerprint.get('optimizer')} |")
        lines.append(f"| Learning rate | {fingerprint.get('learning_rate')} |")
        lines.append(f"| Weight decay | {fingerprint.get('weight_decay')} |")
        lines.append(f"| Batch size | {fingerprint.get('batch_size')} |")
        lines.append(f"| Augmentation | {fingerprint.get('augmentation')} |")
        lines.append(f"| Scheduler | {fingerprint.get('scheduler')} |")
    lines.append(f"| Validation cases | {e1_meta.get('n_cases', 'n/a')} |")
    lines.append("| Official test cases used | 0 |")
    lines.append(f"| Best checkpoint epoch | {e1_meta.get('checkpoint_epoch', 'n/a')} |")
    lines.append("")

    lines.append("## 2. Checkpoint selection vs experiment comparison")
    lines.append("")
    lines.append("These are two DIFFERENT metrics and must not be conflated:")
    lines.append("")
    lines.append("- **Within-run checkpoint selection** (unchanged from the baseline): the")
    lines.append("  batch-level foreground-macro validation Dice proxy computed in")
    lines.append("  `src/train.py::validate`. Baseline and E1 use the identical rule, which is")
    lines.append("  what makes the two runs comparable.")
    proxy_value = e1_meta.get("training_loop_val_metric_slice_level")
    if proxy_value is not None:
        lines.append(f"  E1 best proxy value: `{proxy_value:.6f}`.")
    lines.append("- **Across-experiment comparison** (the scientific decision): per-case,")
    lines.append("  volume-level, original-voxel-space **macro Dice = (CG + PZ) / 2** over the")
    lines.append("  20 official validation cases. This is the number the verdict below uses.")
    lines.append("")

    lines.append("## 3. Primary result -- validation macro Dice (20 cases)")
    lines.append("")
    lines.append("| Metric | Baseline (CE) | E1 (Dice + CE) | Difference |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| **Macro Dice (CG+PZ)/2** | {_fmt(macro['baseline'])} | {_fmt(macro['e1'])} | "
        f"{_fmt_delta(macro['delta'])} |"
    )
    for class_id in (CLASS_CG, CLASS_PZ):
        cell = comparison["per_class"][class_id]["dice"]
        lines.append(
            f"| {CLASS_LABELS[class_id]} Dice | {_fmt(cell['baseline'])} | {_fmt(cell['e1'])} | "
            f"{_fmt_delta(cell['delta'])} |"
        )
    lines.append("")
    lines.append(f"**Verdict (macro Dice): E1 {comparison['verdict'].upper()} relative to the baseline.**")
    lines.append("")

    lines.append("## 4. Full metric comparison (no cherry-picking)")
    lines.append("")
    lines.append("All metrics are per-case, volume-level, original voxel space, mean across the")
    lines.append("20 validation cases. For HD95 and ASD a NEGATIVE difference is an improvement")
    lines.append("(lower distance is better); for all other metrics a positive difference is.")
    lines.append("")
    lines.append("| Class | Metric | Baseline | E1 | Difference | Better when |")
    lines.append("|---|---|---|---|---|---|")
    for class_id in (CLASS_CG, CLASS_PZ):
        for metric in REPORT_METRICS:
            cell = comparison["per_class"][class_id][metric]
            direction = "lower" if cell["lower_is_better"] else "higher"
            lines.append(
                f"| {CLASS_LABELS[class_id]} | {metric} | {_fmt(cell['baseline'])} | "
                f"{_fmt(cell['e1'])} | {_fmt_delta(cell['delta'])} | {direction} |"
            )
    lines.append("")

    if history:
        completed = history[-1]["epoch"]
        lines.append("## 5. Training history")
        lines.append("")
        lines.append(f"- Epochs completed: **{completed}**")
        proxy_pairs = [
            (r["epoch"], r.get("val_dice_proxy")) for r in history if r.get("val_dice_proxy") is not None
        ]
        if proxy_pairs:
            best_epoch, best_proxy = max(proxy_pairs, key=lambda t: t[1])
            lines.append(f"- Best proxy epoch: **{best_epoch}** (val Dice proxy {best_proxy:.6f})")
        first, last = history[0], history[-1]
        lines.append(
            f"- Train total loss: {_fmt(first.get('train_loss_total'), 6)} (epoch {first['epoch']}) "
            f"-> {_fmt(last.get('train_loss_total'), 6)} (epoch {last['epoch']})"
        )
        lines.append(
            f"- Train CE term: {_fmt(first.get('train_loss_ce'), 6)} -> {_fmt(last.get('train_loss_ce'), 6)}"
        )
        lines.append(
            f"- Train Dice term: {_fmt(first.get('train_loss_dice'), 6)} -> {_fmt(last.get('train_loss_dice'), 6)}"
        )
        total_seconds = sum(r.get("epoch_seconds") or 0.0 for r in history)
        lines.append(f"- Total training time: {total_seconds / 3600.0:.2f} h")
        lines.append("")
        lines.append("Figures: `plots/training_loss.png`, `plots/validation_dice.png`, "
                     "`plots/validation_metrics.png`.")
        lines.append("")

    lines.append("## 6. Integrity statement")
    lines.append("")
    lines.append("- The official 19-case test set was **not** used during E1 training, checkpoint")
    lines.append("  selection, or experiment comparison.")
    lines.append(f"- E1 evaluation split: `{e1_meta.get('split', 'n/a')}` "
                 f"({e1_meta.get('n_cases', 'n/a')} cases).")
    lines.append(f"- Baseline evaluation split: `{base_meta.get('split', 'n/a')}` "
                 f"({base_meta.get('n_cases', 'n/a')} cases).")
    lines.append("- Only the loss function differs between the two arms; every other controlled")
    lines.append("  setting is asserted identical by `tests/test_e1_config.py`.")
    lines.append("- Label identity (1 = CG, 2 = PZ) remains the documented, still-unverified")
    lines.append("  hypothesis; see `results/label_mapping/label_mapping_report.md`.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the E1 summary, report, and plots.")
    parser.add_argument("--e1-dir", default="results/exp_e1_dice_ce",
                        help="E1 output directory (local or Drive).")
    parser.add_argument("--baseline-dir", default="results/exp01_baseline",
                        help="Baseline output directory holding eval_val_summary.json.")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    args = parser.parse_args(argv)

    e1_dir = os.path.abspath(args.e1_dir)
    baseline_dir = os.path.abspath(args.baseline_dir)

    e1_summary = _load_summary(os.path.join(e1_dir, "eval_val_summary.json"), "E1")
    baseline_summary = _load_summary(os.path.join(baseline_dir, "eval_val_summary.json"), "Baseline")

    history = read_metrics_csv(os.path.join(e1_dir, "metrics.csv"))

    fingerprint = None
    fingerprint_path = os.path.join(e1_dir, "config_fingerprint.json")
    if os.path.isfile(fingerprint_path):
        with open(fingerprint_path, "r", encoding="utf-8") as f:
            fingerprint = json.load(f)

    comparison = build_comparison(baseline_summary, e1_summary)

    summary_payload = {
        "experiment": "exp_e1_dice_ce",
        "description": "RQ-1 E1: 1.0 x CrossEntropy + 1.0 x SoftDice (foreground CG+PZ) vs baseline CrossEntropy",
        "comparison_metric": comparison["macro"]["definition"],
        "verdict": comparison["verdict"],
        "macro_dice": comparison["macro"],
        "per_class": comparison["per_class"],
        "epochs_completed": history[-1]["epoch"] if history else None,
        "best_checkpoint_epoch": e1_summary.get("metadata", {}).get("checkpoint_epoch"),
        "within_run_selection_metric": (
            "batch-level foreground-macro validation Dice proxy (src/train.py::validate) "
            "-- identical rule to the baseline"
        ),
        "official_test_cases_used": 0,
        "fingerprint": fingerprint,
    }

    os.makedirs(e1_dir, exist_ok=True)
    summary_path = os.path.join(e1_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    report_path = os.path.join(e1_dir, "training_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_report(comparison, history, e1_summary, baseline_summary, fingerprint) + "\n")

    written = [summary_path, report_path]
    if not args.no_plots:
        try:
            written.extend(write_plots(history, comparison, os.path.join(e1_dir, "plots")))
        except Exception as exc:  # pragma: no cover - plotting must never fail the report
            print(f"WARNING: plot generation failed ({type(exc).__name__}: {exc}).")

    macro = comparison["macro"]
    print("=" * 78)
    print("E1 vs BASELINE -- validation macro Dice (20 cases, per-case volume-level)")
    print("=" * 78)
    print(f"  Baseline : {_fmt(macro['baseline'])}")
    print(f"  E1       : {_fmt(macro['e1'])}")
    print(f"  Delta    : {_fmt_delta(macro['delta'])}")
    print(f"  Verdict  : E1 {comparison['verdict'].upper()}")
    print()
    for class_id in (CLASS_CG, CLASS_PZ):
        cell = comparison["per_class"][class_id]["dice"]
        print(f"  {CLASS_LABELS[class_id]} Dice: baseline {_fmt(cell['baseline'])} -> "
              f"E1 {_fmt(cell['e1'])} ({_fmt_delta(cell['delta'])})")
    print()
    print("Wrote:")
    for path in written:
        print(f"  {path}")
    print()
    print("The official 19-case test set was NOT used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
