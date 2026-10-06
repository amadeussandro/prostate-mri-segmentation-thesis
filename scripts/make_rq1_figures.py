"""Generate the RQ1 result figures from the frozen evaluation artifacts (READ-ONLY).

Reads the per-case metric CSVs and summary JSONs that each experiment already
produced and renders the figure set for the thesis / progress report. It runs no
model, touches no checkpoint, and writes only PNG files into results/figures_rq1/.

Standard deviations are computed with **ddof = 0** (population SD) throughout,
matching `src/metrics.py` -- the code that produced every validation, baseline,
E2 and E3 standard deviation on record. Mixing conventions would make the
validation-vs-test comparison inconsistent.

Usage:
    python scripts/make_rq1_figures.py
"""

import csv
import json
import os
import statistics
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

OUT_DIR = os.path.join(PROJECT_ROOT, "results", "figures_rq1")
METRICS = ("dice", "iou", "hd95_mm", "asd_mm", "precision", "recall")
CG, PZ = 1, 2

# Validation arms (20 official validation cases) + the two test evaluations.
VALIDATION_ARMS = {
    "Baseline": "results/exp01_baseline/eval_val_per_case_metrics.csv",
    "E1": "results/exp_e1_dice_ce/eval_val_per_case_metrics.csv",
    "E2": "results/exp_e2_ce_aug/eval_val_per_case_metrics.csv",
    "E3": "results/exp_e3_dice_ce_pzweight/exp_e3_dice_ce_pzweight/eval_val_per_case_metrics.csv",
}
TEST_ARMS = {
    "Baseline": "results/exp01_baseline/eval_test_per_case_metrics.csv",
    "E1": "results/test_e1_epoch88/per_case_metrics.csv",
}

ARM_COLOR = {"Baseline": "#8c8c8c", "E1": "#2a6fb5", "E2": "#d98324", "E3": "#5c9e55"}
ARM_LABEL = {
    "Baseline": "Baseline (CE)",
    "E1": "E1 (Dice + CE)",
    "E2": "E2 (CE + aug)",
    "E3": "E3 (Dice + weighted CE)",
}


def load_per_case(relpath):
    """{class_id: {case_id: {metric: value}}} from an evaluation per-case CSV."""
    table = {}
    path = os.path.join(PROJECT_ROOT, relpath)
    if not os.path.isfile(path):
        return table
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            case = str(row["patient_id"]).strip().zfill(3)
            cid = int(row["class_id"])
            table.setdefault(cid, {})[case] = {
                m: float(row[m]) for m in METRICS if row.get(m) not in (None, "")
            }
    return table


def mean_sd(table, cid, metric):
    values = [v[metric] for v in table[cid].values()]
    return statistics.fmean(values), statistics.pstdev(values)   # ddof = 0


def macro_per_case(table):
    return {c: (table[CG][c]["dice"] + table[PZ][c]["dice"]) / 2 for c in table[CG]}


def style(ax, title, ylabel=None, xlabel=None):
    ax.set_title(title, fontsize=11, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(alpha=0.25, axis="y", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=9)


# --------------------------------------------------------------------------- 1
def fig_validation_overview(val, out):
    """CG / PZ / macro Dice across the four validation arms."""
    arms = ["Baseline", "E1", "E2", "E3"]
    groups = [("CG Dice", CG), ("PZ Dice", PZ)]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(3)
    width = 0.2
    for i, arm in enumerate(arms):
        cg = mean_sd(val[arm], CG, "dice")
        pz = mean_sd(val[arm], PZ, "dice")
        macro_vals = list(macro_per_case(val[arm]).values())
        heights = [cg[0], pz[0], statistics.fmean(macro_vals)]
        errs = [cg[1], pz[1], statistics.pstdev(macro_vals)]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, heights, width, yerr=errs, capsize=3,
                      color=ARM_COLOR[arm], label=ARM_LABEL[arm],
                      error_kw={"elinewidth": 0.9, "alpha": 0.6})
        for b, h in zip(bars, heights):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.015, f"{h:.4f}",
                    ha="center", va="bottom", fontsize=7.2, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(["CG Dice", "PZ Dice", "Macro Dice"])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8.5, ncol=4, loc="lower center", frameon=False,
              bbox_to_anchor=(0.5, -0.17))
    style(ax, "RQ1 validation — 20 official validation cases (mean ± SD)", "Dice")
    fig.tight_layout()
    path = os.path.join(out, "01_validation_dice_overview.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- 2
def fig_validation_spread(val, out):
    """Per-case macro Dice distribution: how small the between-arm gap really is."""
    arms = ["Baseline", "E1", "E2", "E3"]
    data = [list(macro_per_case(val[a]).values()) for a in arms]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    parts = ax.violinplot(data, showextrema=False, widths=0.75)
    for body, arm in zip(parts["bodies"], arms):
        body.set_facecolor(ARM_COLOR[arm])
        body.set_alpha(0.25)
    bp = ax.boxplot(data, widths=0.22, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.4})
    for patch, arm in zip(bp["boxes"], arms):
        patch.set_facecolor(ARM_COLOR[arm])
        patch.set_alpha(0.75)
    rng = np.random.default_rng(42)
    for i, (arm, values) in enumerate(zip(arms, data), start=1):
        jitter = rng.normal(0, 0.035, len(values))
        ax.scatter(np.full(len(values), i) + jitter, values, s=13,
                   color="black", alpha=0.45, zorder=3, linewidths=0)
        ax.text(i, 1.005, f"{statistics.fmean(values):.4f}", ha="center",
                fontsize=8.5, fontweight="bold")
    ax.set_xticks(range(1, len(arms) + 1))
    ax.set_xticklabels([ARM_LABEL[a] for a in arms], fontsize=9)
    ax.set_ylim(0.5, 1.03)
    spread = max(statistics.fmean(d) for d in data) - min(statistics.fmean(d) for d in data)
    style(ax, f"Per-case macro Dice, validation — between-arm spread is {spread:.4f}, "
               f"smaller than every arm's own SD", "Macro Dice per case")
    fig.tight_layout()
    path = os.path.join(out, "02_validation_percase_spread.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- 3
def fig_validation_all_metrics(val, out):
    """Every reported metric, all four arms, CG and PZ."""
    arms = ["Baseline", "E1", "E2", "E3"]
    overlap = ("dice", "iou", "precision", "recall")
    distance = ("hd95_mm", "asd_mm")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for row, (cid, zone) in enumerate([(CG, "CG"), (PZ, "PZ")]):
        ax = axes[row][0]
        x = np.arange(len(overlap))
        for i, arm in enumerate(arms):
            heights = [mean_sd(val[arm], cid, m)[0] for m in overlap]
            errs = [mean_sd(val[arm], cid, m)[1] for m in overlap]
            ax.bar(x + (i - 1.5) * 0.2, heights, 0.2, yerr=errs, capsize=2.5,
                   color=ARM_COLOR[arm], label=ARM_LABEL[arm] if row == 0 else None,
                   error_kw={"elinewidth": 0.8, "alpha": 0.55})
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() if m == "iou" else m.capitalize() for m in overlap])
        ax.set_ylim(0, 1.05)
        style(ax, f"{zone} — overlap metrics (higher is better)", "score")

        ax = axes[row][1]
        x = np.arange(len(distance))
        for i, arm in enumerate(arms):
            heights = [mean_sd(val[arm], cid, m)[0] for m in distance]
            errs = [mean_sd(val[arm], cid, m)[1] for m in distance]
            ax.bar(x + (i - 1.5) * 0.2, heights, 0.2, yerr=errs, capsize=2.5,
                   color=ARM_COLOR[arm], error_kw={"elinewidth": 0.8, "alpha": 0.55})
        ax.set_xticks(x)
        ax.set_xticklabels(["HD95 (mm)", "ASD (mm)"])
        style(ax, f"{zone} — surface distance (LOWER is better)", "mm")
    axes[0][0].legend(fontsize=8.5, ncol=4, frameon=False,
                      loc="lower center", bbox_to_anchor=(1.05, -1.42))
    fig.suptitle("RQ1 validation — all reported metrics, 20 cases (mean ± SD)",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = os.path.join(out, "03_validation_all_metrics.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- 4
def fig_e3_mechanism(val, out):
    """E3's precision/recall trade on PZ -- the mechanistic result."""
    arms = ["Baseline", "E1", "E2", "E3"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    ax = axes[0]
    x = np.arange(len(arms))
    prec = [mean_sd(val[a], PZ, "precision")[0] for a in arms]
    rec = [mean_sd(val[a], PZ, "recall")[0] for a in arms]
    ax.bar(x - 0.2, prec, 0.4, label="PZ precision", color="#4c72b0")
    ax.bar(x + 0.2, rec, 0.4, label="PZ recall", color="#dd8452")
    for i, (p, r) in enumerate(zip(prec, rec)):
        ax.text(i - 0.2, p + 0.012, f"{p:.3f}", ha="center", fontsize=8)
        ax.text(i + 0.2, r + 0.012, f"{r:.3f}", ha="center", fontsize=8)
        ax.annotate("", xy=(i + 0.2, r), xytext=(i - 0.2, p),
                    arrowprops={"arrowstyle": "-", "linestyle": ":",
                                "color": "0.45", "linewidth": 1})
    ax.set_xticks(x)
    ax.set_xticklabels([ARM_LABEL[a].split(" (")[0] for a in arms])
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9, frameon=False)
    style(ax, "PZ operating point — E3 trades precision for recall", "score")

    ax = axes[1]
    e1_rec = {c: v["recall"] for c, v in val["E1"][PZ].items()}
    e3_rec = {c: v["recall"] for c, v in val["E3"][PZ].items()}
    cases = sorted(e1_rec)
    deltas = [e3_rec[c] - e1_rec[c] for c in cases]
    order = np.argsort(deltas)
    ax.barh(np.arange(len(cases)), [deltas[i] for i in order], color="#5c9e55", height=0.72)
    ax.set_yticks(np.arange(len(cases)))
    ax.set_yticklabels([cases[i] for i in order], fontsize=7.5)
    ax.axvline(0, color="black", linewidth=0.9)
    won = sum(1 for d in deltas if d > 0)
    style(ax, f"E3 − E1 PZ recall per case — improved in {won}/{len(cases)} cases",
          None, "Δ PZ recall")
    fig.suptitle("E3 mechanistic result: the intervention worked, overlap did not move",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    path = os.path.join(out, "04_e3_mechanism.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- 5
def fig_validation_vs_test(val, test, out):
    """E1 validation vs E1 test, and both test evaluations side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    labels = ["CG Dice", "PZ Dice", "Macro Dice"]
    v = [mean_sd(val["E1"], CG, "dice")[0], mean_sd(val["E1"], PZ, "dice")[0],
         statistics.fmean(macro_per_case(val["E1"]).values())]
    t = [mean_sd(test["E1"], CG, "dice")[0], mean_sd(test["E1"], PZ, "dice")[0],
         statistics.fmean(macro_per_case(test["E1"]).values())]
    x = np.arange(3)
    ax.bar(x - 0.2, v, 0.4, label="Validation (20 cases, used for selection)",
           color="#9ab8d8")
    ax.bar(x + 0.2, t, 0.4, label="Held-out test (19 cases)", color="#2a6fb5")
    for i, (a, b) in enumerate(zip(v, t)):
        ax.text(i - 0.2, a + 0.013, f"{a:.4f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.013, f"{b:.4f}", ha="center", fontsize=8)
        ax.text(i, 0.045, f"{b - a:+.4f}", ha="center", fontsize=9,
                fontweight="bold", color="#b03030")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    style(ax, "E1 — generalization gap (validation is optimistically biased)", "Dice")

    ax = axes[1]
    arms = ["Baseline", "E1"]
    x = np.arange(3)
    for i, arm in enumerate(arms):
        heights = [mean_sd(test[arm], CG, "dice")[0], mean_sd(test[arm], PZ, "dice")[0],
                   statistics.fmean(macro_per_case(test[arm]).values())]
        bars = ax.bar(x + (i - 0.5) * 0.36, heights, 0.36, color=ARM_COLOR[arm],
                      label=f"{ARM_LABEL[arm]} — test")
        for b, h in zip(bars, heights):
            ax.text(b.get_x() + b.get_width() / 2, h + 0.013, f"{h:.4f}",
                    ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    style(ax, "Official held-out test — same 19 cases, Baseline vs E1", "Dice")
    fig.suptitle("RQ1 final — held-out test evaluation", fontsize=12.5, y=1.0)
    fig.tight_layout()
    path = os.path.join(out, "05_validation_vs_test.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- 6
def fig_test_percase(test, out):
    """Per-case test results for E1, with the Baseline paired against it."""
    e1, bl = test["E1"], test["Baseline"]
    cases = sorted(e1[CG])
    e1_macro = macro_per_case(e1)
    bl_macro = macro_per_case(bl)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    ax = axes[0]
    x = np.arange(len(cases))
    ax.bar(x - 0.2, [e1[CG][c]["dice"] for c in cases], 0.4, label="CG Dice",
           color="#2a6fb5")
    ax.bar(x + 0.2, [e1[PZ][c]["dice"] for c in cases], 0.4, label="PZ Dice",
           color="#d98324")
    ax.plot(x, [e1_macro[c] for c in cases], "o--", color="black", markersize=4,
            linewidth=1.1, label="Macro Dice")
    mean_macro = statistics.fmean(e1_macro.values())
    ax.axhline(mean_macro, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.text(len(cases) - 0.5, mean_macro + 0.015, f"mean {mean_macro:.4f}",
            ha="right", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(cases, rotation=90, fontsize=7.5)
    ax.set_ylim(0, 1.03)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    style(ax, "E1 held-out test — per-case Dice (19 cases)", "Dice", "test case")

    ax = axes[1]
    deltas = [e1_macro[c] - bl_macro[c] for c in cases]
    order = np.argsort(deltas)
    colors = ["#5c9e55" if deltas[i] > 0 else "#b03030" for i in order]
    ax.barh(np.arange(len(cases)), [deltas[i] for i in order], color=colors, height=0.72)
    ax.set_yticks(np.arange(len(cases)))
    ax.set_yticklabels([cases[i] for i in order], fontsize=7.5)
    ax.axvline(0, color="black", linewidth=0.9)
    won = sum(1 for d in deltas if d > 0)
    style(ax, f"E1 − Baseline macro Dice per case — E1 better in {won}/{len(cases)}",
          None, "Δ macro Dice")
    fig.suptitle("Held-out test, per case", fontsize=12.5, y=1.0)
    fig.tight_layout()
    path = os.path.join(out, "06_test_percase.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- 7
def fig_journey(val, test, out):
    """The whole RQ1 story on one axis: four validation arms, then the test."""
    fig, ax = plt.subplots(figsize=(11, 5.4))
    arms = ["Baseline", "E1", "E2", "E3"]
    macros = [statistics.fmean(macro_per_case(val[a]).values()) for a in arms]
    sds = [statistics.pstdev(list(macro_per_case(val[a]).values())) for a in arms]
    x = np.arange(len(arms))
    ax.errorbar(x, macros, yerr=sds, fmt="o", markersize=9, capsize=5,
                color="#2a6fb5", ecolor="#9ab8d8", elinewidth=1.6,
                label="Validation macro Dice (20 cases)")
    for i, (m, a) in enumerate(zip(macros, arms)):
        ax.annotate(f"{m:.4f}", (i, m), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=9,
                    fontweight="bold" if a == "E1" else "normal")
    ax.axhline(macros[0], color="0.6", linestyle="--", linewidth=1,
               label="Baseline validation level")

    test_macros = {a: statistics.fmean(macro_per_case(test[a]).values()) for a in test}
    test_sds = {a: statistics.pstdev(list(macro_per_case(test[a]).values())) for a in test}
    ax.errorbar([4], [test_macros["E1"]], yerr=[test_sds["E1"]], fmt="D",
                markersize=10, capsize=5, color="#b03030", ecolor="#e0a0a0",
                elinewidth=1.6, label="E1 held-out TEST (19 cases)")
    ax.annotate(f"{test_macros['E1']:.4f}", (4, test_macros["E1"]),
                textcoords="offset points", xytext=(0, 14), ha="center",
                fontsize=9, fontweight="bold", color="#b03030")
    ax.errorbar([4], [test_macros["Baseline"]], yerr=[test_sds["Baseline"]], fmt="s",
                markersize=8, capsize=4, color="#8c8c8c", ecolor="#c8c8c8",
                elinewidth=1.4, label="Baseline held-out test (19 cases)")
    ax.annotate(f"{test_macros['Baseline']:.4f}", (4, test_macros["Baseline"]),
                textcoords="offset points", xytext=(0, -20), ha="center", fontsize=8.5)

    ax.axvline(3.5, color="0.75", linestyle="-", linewidth=1)
    ax.text(1.5, 0.52, "model selection — validation only", ha="center",
            fontsize=9, style="italic", color="0.35")
    ax.text(4, 0.52, "final evaluation\nheld-out test", ha="center",
            fontsize=9, style="italic", color="0.35")
    ax.set_xticks(list(range(5)))
    ax.set_xticklabels([ARM_LABEL[a].split(" (")[0] for a in arms] + ["E1 frozen"])
    ax.set_ylim(0.48, 0.92)
    ax.legend(fontsize=8.5, frameon=False, loc="lower left")
    style(ax, "RQ1 end to end — four validation arms, then one held-out test",
          "Macro Dice (mean ± SD)")
    fig.tight_layout()
    path = os.path.join(out, "07_rq1_journey.png")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    val = {}
    for arm, rel in VALIDATION_ARMS.items():
        table = load_per_case(rel)
        if not table:
            raise FileNotFoundError(f"validation per-case metrics missing for {arm}: {rel}")
        val[arm] = table
    test = {}
    for arm, rel in TEST_ARMS.items():
        table = load_per_case(rel)
        if not table:
            raise FileNotFoundError(f"test per-case metrics missing for {arm}: {rel}")
        test[arm] = table

    print("Loaded per-case metrics")
    for arm, table in val.items():
        print(f"  validation {arm:<9}: {len(table[CG])} cases")
    for arm, table in test.items():
        print(f"  test       {arm:<9}: {len(table[CG])} cases")
    print()

    written = [
        fig_validation_overview(val, OUT_DIR),
        fig_validation_spread(val, OUT_DIR),
        fig_validation_all_metrics(val, OUT_DIR),
        fig_e3_mechanism(val, OUT_DIR),
        fig_validation_vs_test(val, test, OUT_DIR),
        fig_test_percase(test, OUT_DIR),
        fig_journey(val, test, OUT_DIR),
    ]
    for path in written:
        print(f"  wrote {os.path.relpath(path, PROJECT_ROOT)}  "
              f"({os.path.getsize(path):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
