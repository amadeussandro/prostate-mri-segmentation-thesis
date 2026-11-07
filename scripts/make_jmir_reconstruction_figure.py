"""Figure 3 of the JMIR revision: reconstruction path and the 3 verification levels.

Also writes a title-free crop of the U-Net diagram (Figure 2), since JMIR puts
titles in captions. Outputs go to results/figures_jmir/.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from PIL import Image

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "results/figures_jmir")
INK, MUTED, EDGE = "#1f2933", "#52606d", "#3e4c59"
FILL = {"data": "#dceaf7", "step": "#f5f7fa", "chk": "#e3f1e6", "l3": "#fdf0dc"}


def box(ax, x, y, w, h, title, body, kind="step"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=FILL[kind], ec=EDGE, lw=1.0))
    ax.text(x + w / 2, y + h - 0.02, title, ha="center", va="top", fontsize=8.8,
            fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.052, body, ha="center", va="top", fontsize=7.5,
            color=MUTED, linespacing=1.4)
    return (x, y, w, h)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=EDGE, lw=1.0, shrinkA=0, shrinkB=0))


def reconstruction() -> str:
    fig = plt.figure(figsize=(7.2, 5.6), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    L, W = 0.04, 0.44
    ys = [(0.86, 0.11), (0.69, 0.14), (0.55, 0.11), (0.39, 0.13), (0.06, 0.30)]
    left = [
        ("Source volume", "Axial T2-weighted NIfTI in its stored orientation\n"
         "Affine, spacing, and orientation codes kept", "data"),
        ("Forward preprocessing", "Reorient to RAS (signed axis permutation)\n"
         "Symmetric pad or center crop to 442×442\n"
         "Inverse parameters stored per case", "step"),
        ("Slicing and inference", "Each axial slice tagged with its index\n"
         "Frozen 2D U-Net, arg-max over 3 classes", "step"),
        ("Reassembly", "Slice written to its stored axial index\n"
         "Missing or duplicated index raises an error", "step"),
        ("Inverse transform and write", "Pad or crop undone with stored offsets\n"
         "Reorientation inverted\n"
         "Nearest-neighbor only; no resampling\n"
         "Written with the source affine and spacing\n(never an identity affine)\n\n"
         "Output: segmentation in the original\nvoxel space, zone volumes in mL", "step"),
    ]
    bl = [box(ax, L, y, W, h, t, b, k) for (y, h), (t, b, k) in zip(ys, left)]
    for a, b in zip(bl, bl[1:]):
        arrow(ax, L + W / 2, a[1], L + W / 2, b[1] + b[3])

    R, WR = 0.53, 0.43
    c1 = box(ax, R, 0.70, WR, 0.27, "Level 1: geometry check",
             "Reads the written file's own header\n"
             "Array shape, affine, voxel spacing,\n"
             "orientation codes, and label set\nmust match the source\n\n"
             "Catches: wrong affine or spacing", "chk")
    c2 = box(ax, R, 0.39, WR, 0.27, "Level 2: round-trip test",
             "No model involved: the reference mask\n"
             "goes forward and back through\nthe same code\n"
             "Must be recovered voxel for voxel\n(pass or fail, never an accuracy)\n\n"
             "Catches: wrong orientation or slice order", "chk")
    c3 = box(ax, R, 0.06, WR, 0.29, "Level 3: anatomical checks",
             "Need no reference, so they can run\nat inference time\n"
             "PZ centroid posterior to CG centroid\n"
             "Gland centroid moves ≤15 mm between\nadjacent slices\n\n"
             "A screen only: misses left-right flips", "l3")
    xm = 0.505
    y_out = bl[-1][1] + bl[-1][3] / 2
    ax.plot([L + W, xm], [y_out, y_out], color=EDGE, lw=1.0)
    ax.plot([xm, xm], [c3[1] + c3[3] / 2, c1[1] + c1[3] / 2], color=EDGE, lw=1.0)
    for c in (c1, c2, c3):
        arrow(ax, xm, c[1] + c[3] / 2, R, c[1] + c[3] / 2)
    out = os.path.join(OUT, "F3_reconstruction_verification.png")
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def unet_without_title() -> str:
    src = os.path.join(PROJECT, "results/figures_supervisor/fig_arsitektur_unet.png")
    im = Image.open(src)
    w, h = im.size
    top = int(round(h * 0.115))  # the title band, including the damaged subtitle
    out = os.path.join(OUT, "F2_unet.png")
    im.crop((0, top, w, h)).save(out)
    return out


def strip_embedded_text() -> list:
    """Remove titles and footnotes drawn into result figures; JMIR puts them in captions.

    Fractions of the image height kept (top, bottom), chosen by inspection so that
    axis labels and column headers stay inside the crop.
    """
    jobs = {"F4_mirrored_volume_passes.png": (0.075, 0.935),
            "F2_zone_volume_agreement.png": (0.0, 0.905),
            "F3_external_validation.png": (0.0, 0.905)}
    done = []
    for name, (t, b) in jobs.items():
        im = Image.open(os.path.join(PROJECT, "results/figures_rq2", name))
        w, h = im.size
        out = os.path.join(OUT, "jmir_" + name)
        im.crop((0, int(h * t), w, int(h * b))).save(out)
        done.append(out)
    return done


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    print("wrote", reconstruction())
    print("wrote", unet_without_title())
    for p in strip_embedded_text():
        print("wrote", p)
