"""Figure 1 of the JMIR revision: study workflow with case counts at every step.

Counts are read from the same numbers module the manuscript uses, so the figure
cannot drift from the text. Writes results/figures_jmir/F1_workflow.png.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))
from manuscript_numbers_jmir import jmir_numbers  # noqa: E402

INK, MUTED, EDGE = "#1f2933", "#52606d", "#3e4c59"
FILL = {"data": "#dceaf7", "step": "#f5f7fa", "out": "#e3f1e6", "ext": "#fdf0dc"}


def box(ax, x, y, w, h, title, body, kind="step"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=FILL[kind], ec=EDGE, lw=1.0))
    ax.text(x + w / 2, y + h - 0.016, title, ha="center", va="top", fontsize=8.6,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.040, body, ha="center", va="top", fontsize=7.4,
            color=MUTED, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.0, shrinkA=0, shrinkB=0))


def main() -> None:
    N = jmir_numbers()
    nc, mf, e = N["nci"], N["mf"], N["external"]
    n_px, n_nci, n_nci_all = e["n_cases"], nc["n"], nc["n_total"]
    n_msd = mf["n_cases"]["msd_task05"]

    fig = plt.figure(figsize=(7.2, 7.6), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # --- left column: development
    L, W = 0.04, 0.44
    b1 = box(ax, L, 0.885, W, 0.095, "Prostate158 (development)",
             "158 axial T2-weighted examinations, 1 institution\n"
             "Official patient-level split: 119 train / 20 validation / 19 test", "data")
    b2 = box(ax, L, 0.735, W, 0.125, "4 single-factor training configurations",
             "Baseline: cross-entropy\nE1: baseline + soft Dice (CG, PZ)\n"
             "E2: baseline + augmentation\nE3: E1 + class-weighted cross-entropy")
    b3 = box(ax, L, 0.615, W, 0.095, "Model selection (validation, n=20)",
             "Within run: best slice-level validation Dice\n"
             "Between runs: highest case-level macro Dice → E1")
    b4 = box(ax, L, 0.505, W, 0.085, "Frozen model",
             "E1, epoch 88, identified by SHA-256 of its weights\n"
             "No further training or tuning", "out")
    b5 = box(ax, L, 0.355, W, 0.125, "Internal testing (held-out test, n=19)",
             "Accuracy vs reader 1 (reference standard)\n"
             "Reader 2 vs reader 1 (inter-reader benchmark)\n"
             "Zone volumes, Bland-Altman, PSA-density\nsensitivity analysis")
    b6 = box(ax, L, 0.205, W, 0.125, "3D reconstruction and verification",
             "Slices written to their stored axial index\n"
             "Inverse preprocessing, source affine restored\n"
             "Geometry check (header) + round-trip test (voxels)\n"
             "Anatomical checks (no reference needed)")
    for a, b in ((b1, b2), (b2, b3), (b3, b4), (b4, b5), (b5, b6)):
        arrow(ax, a[0] + a[2] / 2, a[1], b[0] + b[2] / 2, b[1] + b[3])

    # --- right column: external testing and fault injection
    R, WR = 0.53, 0.43
    c1 = box(ax, R, 0.775, WR, 0.110, "PROSTATEx (external, secondary)",
             f"{n_px} examinations, 3 T, 1 institution\n"
             "Inference only; analyzed before the rule existed,\n"
             "so post-processing is reported post hoc only", "ext")
    c2 = box(ax, R, 0.565, WR, 0.180, "NCI-ISBI 2013 (external, confirmatory)",
             f"{n_nci_all} examinations with CG and PZ labels\n"
             f"Excluded {n_nci_all - n_nci}: mask on a different series (1);\n"
             f"single whole-gland label (1)\n"
             f"Analyzed: n={n_nci} (1.5 T endorectal and 3 T)\n"
             "Connected-component rule written and hashed\nbefore inference", "ext")
    c3 = box(ax, R, 0.205, WR, 0.330, "Fault injection (no model)",
             f"{mf['v0_n']} reference segmentations, 4 cohorts\n"
             f"Prostate158 {mf['n_cases']['prostate158']} (LPS)  |  "
             f"PROSTATEx {mf['n_cases']['prostatex']} (LAS)\n"
             f"MSD {n_msd} of 32 (RAS)  |  NCI-ISBI {mf['n_cases']['nci_isbi_2013']} (LPS)\n\n"
             "Correct pipeline must return every\nreference exactly (gate)\n\n"
             "6 faults, alone and combined:\nreorientation not inverted, center crop,\n"
             "identity affine, spacing-only affine,\nloader-order and shuffled stacking\n\n"
             "Outcome: which check flags which fault", "data")
    # frozen model -> external cohorts
    ax.plot([L + W, 0.505], [b4[1] + b4[3] / 2] * 2, color=EDGE, lw=1.0)
    ax.plot([0.505, 0.505], [b4[1] + b4[3] / 2, c1[1] + c1[3] / 2], color=EDGE, lw=1.0)
    arrow(ax, 0.505, c1[1] + c1[3] / 2, R, c1[1] + c1[3] / 2)
    arrow(ax, 0.505, c2[1] + c2[3] / 2, R, c2[1] + c2[3] / 2)
    # reconstruction code -> fault injection
    arrow(ax, L + W, b6[1] + b6[3] / 2, R, b6[1] + b6[3] / 2)

    # --- bottom: outcome
    o = box(ax, L, 0.075, 0.92, 0.09, "Outcomes",
            "Accuracy against reader 1 and against inter-reader agreement  |  "
            "equivalence of training configurations (TOST, ±0.02 Dice)\n"
            "External accuracy and the prespecified rule's effect  |  "
            "detection matrix: geometry check, round-trip test, anatomical checks", "out")
    arrow(ax, L + W / 2, b6[1], L + W / 2, o[1] + o[3])
    arrow(ax, R + WR / 2, c3[1], R + WR / 2, o[1] + o[3])

    out = os.path.join(PROJECT, "results/figures_jmir/F1_workflow.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
