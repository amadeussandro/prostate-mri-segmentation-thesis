"""Assemble the Experiment E2 deliverables: summary.json, training_report.md, plots.

Reads only artifacts that already exist on disk -- it NEVER trains, never runs
inference, and never touches the official 19-case test set. Specifically it
consumes:

  * <e2_dir>/metrics.csv                 -- per-epoch history from run_training
  * <e2_dir>/training_history.json       -- same, richer (optional)
  * <e2_dir>/config_fingerprint.json     -- the run's resolved settings (optional)
  * <e2_dir>/eval_val_summary.json       -- 20-case volume-level validation eval
  * <baseline_dir>/eval_val_summary.json -- the same, for the Experiment-1 baseline
  * <e1_dir>/eval_val_summary.json       -- the same, for E1 (optional)

and writes:

  * <e2_dir>/summary.json                -- machine-readable E2 result + deltas
  * <e2_dir>/training_report.md          -- human-readable report for the thesis
  * <e2_dir>/plots/training_loss.png
  * <e2_dir>/plots/validation_dice.png
  * <e2_dir>/plots/validation_metrics.png

Comparison metric (pre-committed in Outputs/rq1_experiment_specification.md):

    validation macro Dice = (mean CG Dice + mean PZ Dice) / 2

computed per case, at volume level, in original voxel space, over the 20
official validation cases. CG and PZ are ALSO always reported separately, and
every available secondary metric (IoU, HD95, ASD, precision, recall) is carried
into the comparison table so no metric can be cherry-picked.

SCOPE LIMIT -- deliberately enforced here
-----------------------------------------
E2 is reported ALONGSIDE the baseline and E1. This script does NOT select a
final model, does NOT run the 2x2 factorial analysis, and does NOT evaluate the
E1+E2 combination arm; those happen only after E1+E2 has been trained. The
report says so explicitly rather than leaving the reader to infer it.

Usage
-----
    python scripts/finalize_e2_report.py \
        --e2-dir results/exp_e2_ce_aug \
        --baseline-dir results/exp01_baseline \
        --e1-dir results/exp_e1_dice_ce
"""

import argparse
import csv
import json
import os
import subprocess
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

# Arms, in report order. "e2" is always required; the others are anchors.
ARMS = ("baseline", "e1", "e2")
ARM_LABELS = {
    "baseline": "Baseline (CE, no aug)",
    "e1": "E1 (Dice+CE, no aug)",
    "e2": "E2 (CE, aug)",
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
            f"REFUSING to build the E2 comparison from a TEST-set evaluation ({path}). "
            "E2 model selection and comparison must use the 20-case validation split "
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

    Deltas are always computed against the BASELINE, which is the pre-committed
    anchor. A delta against E1 is reported separately for macro Dice only, as
    descriptive context -- it is NOT a decision rule, because E1 and E2 vary
    different factors and are not nested.
    """
    comparison: Dict[str, Any] = {"per_class": {}, "macro": {}}

    for class_id in (CLASS_CG, CLASS_PZ):
        comparison["per_class"][class_id] = {}
        for metric in REPORT_METRICS:
            cell: Dict[str, Any] = {
                arm: _metric(summaries.get(arm), class_id, metric) for arm in ARMS
            }
            cell["e2_sd"] = _metric_sd(summaries.get("e2"), class_id, metric)
            cell["delta_vs_baseline"] = (
                cell["e2"] - cell["baseline"]
                if cell["e2"] is not None and cell["baseline"] is not None
                else None
            )
            cell["lower_is_better"] = metric in LOWER_IS_BETTER
            comparison["per_class"][class_id][metric] = cell

    macro = {arm: _macro_dice(summaries.get(arm)) for arm in ARMS}
    macro["delta_vs_baseline"] = (
        macro["e2"] - macro["baseline"]
        if macro["e2"] is not None and macro["baseline"] is not None
        else None
    )
    macro["delta_vs_e1"] = (
        macro["e2"] - macro["e1"]
        if macro["e2"] is not None and macro["e1"] is not None
        else None
    )
    comparison["macro"] = macro
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
    return "n/a" if value is None else f"{value:+.{places}f}"


# ----------------------------------------------------------------------- PLOTS


def write_plots(
    history: List[Dict[str, Any]], comparison: Dict[str, Any], plots_dir: str
) -> List[str]:
    """Render the three standard E2 figures. Returns the paths written."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plots_dir, exist_ok=True)
    written: List[str] = []

    if history:
        epochs = [r["epoch"] for r in history]

        # ---- 1. loss curves -------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(epochs, [r.get("train_loss_total") for r in history],
                label="train Cross Entropy")
        if any(r.get("val_loss") is not None for r in history):
            ax.plot(epochs, [r.get("val_loss") for r in history],
                    label="validation Cross Entropy", color="black")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("E2 (CE + augmentation) -- loss curves")
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
            "E2 -- validation Dice per epoch (slice-level proxy; NOT the comparison metric)"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        path = os.path.join(plots_dir, "validation_dice.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # ---- 3. grouped bars: baseline vs E1 vs E2 ------------------------------
    labels: List[str] = []
    series: Dict[str, List[Optional[float]]] = {arm: [] for arm in ARMS}
    for class_id in (CLASS_CG, CLASS_PZ):
        for metric in ("dice", "iou", "precision", "recall"):
            cell = comparison["per_class"][class_id][metric]
            if cell["baseline"] is None or cell["e2"] is None:
                continue
            labels.append(f"{CLASS_LABELS[class_id].split()[0]}\n{metric}")
            for arm in ARMS:
                series[arm].append(cell[arm])
    macro = comparison["macro"]
    if macro["baseline"] is not None and macro["e2"] is not None:
        labels.append("macro\nDice")
        for arm in ARMS:
            series[arm].append(macro[arm])

    if labels:
        import numpy as np

        present = [arm for arm in ARMS if any(v is not None for v in series[arm])]
        x = np.arange(len(labels))
        width = 0.8 / max(1, len(present))
        fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.3), 5))
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

    return written


# ---------------------------------------------------------------------- REPORT


def _augmentation_rows(augmentation: Optional[Dict[str, Any]]) -> List[str]:
    if not augmentation:
        return ["| Augmentation | (not recorded in this run's artifacts) |"]
    if not augmentation.get("enabled"):
        return ["| Augmentation | OFF |"]
    return [
        "| Augmentation | **ON -- training slices only** |",
        f"| - random rotation | +/- {augmentation.get('rotation_degrees')} deg, "
        f"p = {augmentation.get('rotation_p')} |",
        f"| - random scale | {augmentation.get('scale_min')} - {augmentation.get('scale_max')}, "
        f"p = {augmentation.get('scale_p')} |",
        f"| - horizontal flip | p = {augmentation.get('hflip_p')} |",
        f"| - random gamma (image only) | {augmentation.get('gamma_min')} - "
        f"{augmentation.get('gamma_max')}, p = {augmentation.get('gamma_p')} |",
        f"| - image interpolation | order {augmentation.get('image_interpolation_order')} (bilinear) |",
        f"| - mask interpolation | order {augmentation.get('mask_interpolation_order')} (nearest-neighbour) |",
        f"| - augmentation seed | {augmentation.get('seed')} |",
        "| - validation augmented? | **NO -- validation is never augmented** |",
    ]


def render_report(
    comparison: Dict[str, Any],
    history: List[Dict[str, Any]],
    summaries: Dict[str, Optional[Dict[str, Any]]],
    fingerprint: Optional[Dict[str, Any]],
    training_history: Optional[Dict[str, Any]],
    checkpoints: Dict[str, str],
    commit: Optional[str],
) -> str:
    e2_summary = summaries["e2"] or {}
    e2_meta = e2_summary.get("metadata", {})
    macro = comparison["macro"]
    augmentation = (fingerprint or {}).get("augmentation_spec") or (
        (training_history or {}).get("augmentation")
    )

    epochs_completed = max((r["epoch"] for r in history), default=None)
    proxy = [(r["epoch"], r.get("val_dice_proxy")) for r in history
             if r.get("val_dice_proxy") is not None]
    best_proxy_epoch, best_proxy_value = max(proxy, key=lambda t: t[1]) if proxy else (None, None)
    recorded_best_epoch = (training_history or {}).get("best_epoch")

    lines: List[str] = []
    lines.append("# Experiment E2 -- Cross Entropy + Data Augmentation (RQ-1)")
    lines.append("")
    lines.append("Generated by `scripts/finalize_e2_report.py` from on-disk artifacts only.")
    lines.append("This report does **not** select a final model and does **not** run the 2x2")
    lines.append("factorial analysis -- both wait until the E1+E2 arm has been trained.")
    lines.append("")

    # ---- 1. identity --------------------------------------------------------
    lines.append("## 1. Experiment identity")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append("| Experiment ID | `exp_e2_ce_aug` (E2) |")
    lines.append("| Position in the 2x2 design | CE loss x augmentation ON |")
    lines.append(f"| Git commit | `{commit or 'unknown'}` |")
    lines.append("| Dataset | Prostate158 (official train archive) |")
    lines.append("| Target | `t2_anatomy_reader1.nii.gz` (0 = background, 1 = CG, 2 = PZ) |")
    lines.append("| Train / validation split | official 119 train / 20 validation cases |")
    lines.append("| Test set | **not accessed** |")
    lines.append("| External validation | **not accessed** |")
    lines.append("| Architecture | 2D U-Net (`ProstateUNet2D`), unchanged from baseline |")
    lines.append("| Loss | Cross Entropy (baseline loss, unchanged) |")
    if fingerprint:
        lines.append(f"| Optimizer | {fingerprint.get('optimizer')} |")
        lines.append(f"| Learning rate | {fingerprint.get('learning_rate')} |")
        lines.append(f"| Weight decay | {fingerprint.get('weight_decay')} |")
        lines.append(f"| Batch size | {fingerprint.get('batch_size')} |")
        lines.append(f"| Max epochs | {fingerprint.get('num_epochs')} |")
        lines.append(f"| Seed | {fingerprint.get('seed')} |")
        lines.append(f"| AMP | {'ON' if fingerprint.get('amp') else 'OFF'} |")
        lines.append(f"| Precision | {fingerprint.get('precision')} |")
        lines.append(f"| Scheduler | {fingerprint.get('scheduler')} |")
        lines.append(f"| DataLoader workers | {fingerprint.get('num_workers')} |")
        lines.append(f"| Preprocessing | {fingerprint.get('preprocessing')} |")
    lines.extend(_augmentation_rows(augmentation))
    lines.append(f"| Epochs completed | {epochs_completed if epochs_completed is not None else 'n/a'} |")
    best_epoch_display = recorded_best_epoch if recorded_best_epoch is not None else best_proxy_epoch
    lines.append(f"| Best checkpoint epoch | {best_epoch_display if best_epoch_display is not None else 'n/a'} |")
    lines.append(f"| Best within-run val Dice proxy | {_fmt(best_proxy_value)} |")
    lines.append(f"| Best checkpoint | `{checkpoints.get('best', 'n/a')}` |")
    lines.append(f"| Latest checkpoint | `{checkpoints.get('latest', 'n/a')}` |")
    lines.append("")
    lines.append("> The best checkpoint is **not** assumed to be the last epoch. Selection uses")
    lines.append("> the unchanged within-run batch-level foreground-macro validation Dice proxy,")
    lines.append("> identical to the Baseline and E1 arms, which is what makes the arms comparable.")
    lines.append("")

    # ---- 2. resume information ---------------------------------------------
    lines.append("## 2. Run continuity")
    lines.append("")
    recorded = len(history)
    lines.append(f"* Per-epoch records on disk: **{recorded}**")
    if epochs_completed:
        gaps = sorted(set(range(1, epochs_completed + 1)) - {r["epoch"] for r in history})
        if gaps:
            lines.append(
                f"* **Missing epoch record(s): {gaps}** -- training itself was continuous, but "
                "these rows were never written (or were lost before the history-preservation fix)."
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

    # ---- 3. primary result --------------------------------------------------
    verdict = interpret(macro["delta_vs_baseline"])
    lines.append("## 3. Primary result -- validation macro Dice")
    lines.append("")
    lines.append(
        "Per-case, volume-level, original-voxel-space Dice over the 20 official "
        "validation cases. Macro Dice = (mean CG Dice + mean PZ Dice) / 2."
    )
    lines.append("")
    lines.append("| Metric | Baseline (CE) | E1 (Dice+CE) | **E2 (CE + aug)** | E2 - Baseline |")
    lines.append("|---|---|---|---|---|")
    for class_id in (CLASS_CG, CLASS_PZ):
        cell = comparison["per_class"][class_id]["dice"]
        lines.append(
            f"| {CLASS_LABELS[class_id]} Dice | {_fmt(cell['baseline'])} | {_fmt(cell['e1'])} | "
            f"**{_fmt(cell['e2'])}** | {_fmt_delta(cell['delta_vs_baseline'])} |"
        )
    lines.append(
        f"| **Macro Dice (CG+PZ)/2** | {_fmt(macro['baseline'])} | {_fmt(macro['e1'])} | "
        f"**{_fmt(macro['e2'])}** | {_fmt_delta(macro['delta_vs_baseline'])} |"
    )
    lines.append("")
    lines.append(f"**Outcome vs baseline: E2 {verdict}.**")
    if macro["delta_vs_e1"] is not None:
        lines.append("")
        lines.append(
            f"For context only, E2 - E1 macro Dice = {_fmt_delta(macro['delta_vs_e1'])}. "
            "E1 and E2 vary *different* factors, so this difference is descriptive; it is not "
            "a decision rule and it does not rank the two arms."
        )
    lines.append("")

    # ---- 4. secondary metrics ----------------------------------------------
    lines.append("## 4. Secondary metrics (validation, 20 cases)")
    lines.append("")
    lines.append("| Class | Metric | Baseline | E1 | E2 | E2 - Baseline | Direction |")
    lines.append("|---|---|---|---|---|---|---|")
    for class_id in (CLASS_CG, CLASS_PZ):
        for metric in REPORT_METRICS:
            cell = comparison["per_class"][class_id][metric]
            if cell["e2"] is None and cell["baseline"] is None:
                continue
            direction = "lower is better" if cell["lower_is_better"] else "higher is better"
            lines.append(
                f"| {CLASS_LABELS[class_id]} | {metric} | {_fmt(cell['baseline'])} | "
                f"{_fmt(cell['e1'])} | {_fmt(cell['e2'])} | "
                f"{_fmt_delta(cell['delta_vs_baseline'])} | {direction} |"
            )
    lines.append("")
    lines.append(
        "Every metric produced by the evaluation is listed, including those where E2 is worse, "
        "so no metric can be cherry-picked after the fact."
    )
    lines.append("")

    # ---- 5. provenance ------------------------------------------------------
    lines.append("## 5. Provenance and integrity")
    lines.append("")
    lines.append(f"* Validation cases evaluated: {e2_meta.get('n_cases', 'n/a')}")
    lines.append(f"* Evaluation split: `{e2_meta.get('split', 'val')}`")
    lines.append(f"* Checkpoint evaluated: `{e2_meta.get('checkpoint_path', checkpoints.get('best', 'n/a'))}`")
    lines.append("* The official 19-case test set was NOT used for training, checkpoint")
    lines.append("  selection, or any comparison in this report.")
    lines.append("* Validation data was never augmented.")
    lines.append("* E2 was trained from a fresh initialization; it was not warm-started from")
    lines.append("  the Baseline or E1 checkpoint (enforced by")
    lines.append("  `src/train.py::verify_checkpoint_compatibility`).")
    lines.append("")

    # ---- 6. what happens next ----------------------------------------------
    lines.append("## 6. Not concluded here")
    lines.append("")
    lines.append("* No final model is selected.")
    lines.append("* The 2x2 factorial analysis (loss x augmentation) is **not** run; it requires")
    lines.append("  the fourth cell, E1+E2.")
    lines.append("* No test-set or external-validation evaluation is performed.")
    lines.append("")

    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ MAIN


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--e2-dir", default="results/exp_e2_ce_aug",
                        help="Directory holding the E2 run artifacts.")
    parser.add_argument("--baseline-dir", default="results/exp01_baseline",
                        help="Directory holding the baseline run artifacts.")
    parser.add_argument("--e1-dir", default="results/exp_e1_dice_ce",
                        help="Directory holding the E1 run artifacts (optional anchor).")
    parser.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    args = parser.parse_args(argv)

    e2_dir = os.path.abspath(args.e2_dir)
    baseline_dir = os.path.abspath(args.baseline_dir)
    e1_dir = os.path.abspath(args.e1_dir) if args.e1_dir else None

    summaries: Dict[str, Optional[Dict[str, Any]]] = {
        "e2": _load_summary(os.path.join(e2_dir, "eval_val_summary.json"), "E2", required=True),
        "baseline": _load_summary(
            os.path.join(baseline_dir, "eval_val_summary.json"), "Baseline", required=True
        ),
        "e1": _load_summary(
            os.path.join(e1_dir, "eval_val_summary.json"), "E1", required=False
        ) if e1_dir else None,
    }
    if summaries["e1"] is None:
        print("NOTE: no E1 validation summary found -- the report will show E2 vs baseline only.")

    history = read_metrics_csv(os.path.join(e2_dir, "metrics.csv"))
    fingerprint = _load_json(os.path.join(e2_dir, "config_fingerprint.json"))
    training_history = _load_json(os.path.join(e2_dir, "training_history.json"))
    comparison = build_comparison(summaries)
    commit = git_commit()

    checkpoints = {
        "best": os.path.join(e2_dir, "best_model.pt"),
        "latest": os.path.join(e2_dir, "latest_checkpoint.pt"),
    }

    plots: List[str] = []
    if not args.no_plots:
        try:
            plots = write_plots(history, comparison, os.path.join(e2_dir, "plots"))
        except ImportError:
            print("WARNING: matplotlib is unavailable -- skipping figures.")

    epochs_completed = max((r["epoch"] for r in history), default=None)
    proxy = [(r["epoch"], r.get("val_dice_proxy")) for r in history
             if r.get("val_dice_proxy") is not None]
    best_proxy_epoch, best_proxy_value = max(proxy, key=lambda t: t[1]) if proxy else (None, None)
    macro = comparison["macro"]

    summary_payload = {
        "experiment": "exp_e2_ce_aug",
        "experiment_label": "E2 (Cross Entropy + augmentation)",
        "design_cell": {"loss": "cross_entropy", "augmentation": True},
        "git_commit": commit,
        "dataset": "Prostate158 (official train archive)",
        "target_mask": "t2_anatomy_reader1.nii.gz",
        "split": {"train_cases": 119, "val_cases": 20, "test_accessed": False},
        "augmentation": (fingerprint or {}).get("augmentation_spec")
                        or (training_history or {}).get("augmentation"),
        "fingerprint": fingerprint,
        "epochs_completed": epochs_completed,
        "epoch_records_on_disk": len(history),
        "best_checkpoint_epoch": (training_history or {}).get("best_epoch", best_proxy_epoch),
        "best_val_dice_proxy": best_proxy_value,
        "checkpoints": checkpoints,
        "per_class": {
            class_id: {
                metric: comparison["per_class"][class_id][metric] for metric in REPORT_METRICS
            }
            for class_id in (CLASS_CG, CLASS_PZ)
        },
        "macro_dice": macro,
        "verdict": interpret(macro["delta_vs_baseline"]),
        "plots": plots,
        "scope_note": (
            "E2 is reported alongside the baseline and E1. No final model selection and no "
            "2x2 factorial analysis are performed here; both require the E1+E2 arm."
        ),
    }

    summary_path = os.path.join(e2_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    report_path = os.path.join(e2_dir, "training_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(render_report(
            comparison, history, summaries, fingerprint, training_history, checkpoints, commit
        ))

    print("=" * 78)
    print("E2 REPORT WRITTEN")
    print("=" * 78)
    print(f"  summary.json      : {summary_path}")
    print(f"  training_report.md: {report_path}")
    for path in plots:
        print(f"  plot              : {path}")
    print("-" * 78)
    print("  Validation macro Dice")
    print(f"    Baseline : {_fmt(macro['baseline'])}")
    print(f"    E1       : {_fmt(macro['e1'])}")
    print(f"    E2       : {_fmt(macro['e2'])}")
    print(f"    E2 - Baseline: {_fmt_delta(macro['delta_vs_baseline'])}  -> E2 {summary_payload['verdict']}")
    print("-" * 78)
    for class_id in (CLASS_CG, CLASS_PZ):
        cell = comparison["per_class"][class_id]["dice"]
        print(f"  {CLASS_LABELS[class_id]} Dice: baseline {_fmt(cell['baseline'])} | "
              f"E1 {_fmt(cell['e1'])} | E2 {_fmt(cell['e2'])} ({_fmt_delta(cell['delta_vs_baseline'])})")
    print("=" * 78)
    print("No final model selected. No 2x2 factorial analysis run. Test set untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
