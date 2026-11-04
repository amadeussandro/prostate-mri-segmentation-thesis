"""Draw the preprocessing stage: what each step does, why, and what it records.

Asked for in supervision: a figure for preprocessing, and a step-by-step account
of what each step is for. Those are the same thing, so the figure carries the
purpose rather than only the operation.

The column on the right is the part that matters for RQ2. Every geometric step
writes down what it would take to undo it, per case. The reconstruction later
replays that column in reverse; a step that transformed the image without
recording its inverse would be unrecoverable, and the ablation shows what that
costs.

Usage:
    python scripts/make_preprocessing_figure.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT, "scripts"))

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
C_BLUE, C_ORANGE, C_AQUA = "#2a5caa", "#d4681f", "#1f8a70"
BLUE_F, BLUE_E = "#e8f0fb", C_BLUE
AQUA_F, AQUA_E = "#e6f6f0", C_AQUA
GREY_F, GREY_E = "#f2f2ef", MUTED

OUT = os.path.join(PROJECT, "results", "figures_rq2", "F6_preprocessing.png")

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

# (step, purpose, what is written down so the step can be undone)
STEPS = [
    ("Source volume",
     "Axial T2-weighted NIfTI in its original voxel space",
     "shape, affine, spacing, orientation codes"),
    ("1  Orientation standardisation",
     "Put every case on the same anatomical axes, so the network\n"
     "never has to learn left from right",
     "the original orientation codes"),
    ("2  Intensity normalisation",
     "Per-volume z-score, so intensity ranges are comparable\n"
     "across scanners. Applied to the image only, never the labels",
     "nothing geometric — no voxel moves"),
    ("3  In-plane crop / pad to 442 × 442",
     "A fixed input size without resampling, so no interpolation\n"
     "is ever applied to a label map",
     "the per-axis crop and pad amounts"),
    ("4  No through-plane resampling",
     "Deliberate omission: slice thickness is the sparsest axis,\n"
     "and interpolating it would invent anatomy between slices",
     "nothing — the axis is left untouched"),
    ("5  Axial slice extraction",
     "One training example per slice, which is what makes a 2D\n"
     "network viable on an anisotropic volume",
     "the axial index of every slice"),
]


def draw(out: str) -> str:
    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    n = len(STEPS)
    gap = 0.016
    h = (0.95 - gap * (n - 1)) / n
    left_w, right_w = 0.60, 0.33

    for i, (title, purpose, retained) in enumerate(STEPS):
        y = 0.96 - i * (h + gap)
        face, edge = (GREY_F, GREY_E) if i == 0 else (BLUE_F, BLUE_E)

        ax.add_patch(plt.Rectangle((0.02, y - h), left_w, h, transform=ax.transAxes,
                                   facecolor=face, edgecolor=edge, lw=1.3, zorder=2))
        ax.text(0.04, y - h * 0.32, title, transform=ax.transAxes, ha="left",
                va="center", fontsize=9.2, color=INK, fontweight="bold", zorder=3)
        ax.text(0.04, y - h * 0.70, purpose, transform=ax.transAxes, ha="left",
                va="center", fontsize=8.0, color=INK_2, zorder=3)

        ax.add_patch(plt.Rectangle((0.02 + left_w + 0.03, y - h), right_w, h,
                                   transform=ax.transAxes, facecolor=AQUA_F,
                                   edgecolor=AQUA_E, lw=1.1, zorder=2))
        ax.text(0.02 + left_w + 0.045, y - h / 2, retained, transform=ax.transAxes,
                ha="left", va="center", fontsize=8.0, color=INK_2, zorder=3)

        if i < n - 1:
            ax.annotate("", xy=(0.02 + left_w / 2, y - h - gap),
                        xytext=(0.02 + left_w / 2, y - h),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))

    ax.text(0.02, 0.995, "Preprocessing", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=11, fontweight="bold", color=INK)
    ax.text(0.02 + left_w + 0.03, 0.995, "Retained per case, to invert later",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=9,
            fontweight="bold", color=AQUA_E)
    ax.text(0.5, -0.055,
            "Every geometric step records what it would take to undo it. The reconstruction "
            "replays this column in reverse order,\nwhich is why no step may transform the "
            "volume without writing its inverse down first.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.2, color=INK_2)
    ax.axis("off")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("wrote", draw(OUT))
