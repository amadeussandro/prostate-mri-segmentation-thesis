"""Generate the CG/PZ label-mapping verification figures (READ-ONLY on the dataset).

Renders, for each selected case, the T2 image with the two anatomy labels
overlaid in **three orthogonal planes** (axial / coronal / sagittal), so the
integer -> zone assignment can be confirmed visually against prostate anatomy:

  * the **peripheral zone (PZ)** is a thin crescent / horseshoe on the
    POSTERIOR aspect of the gland, hugging the rectal interface;
  * the **central gland (CG = CZ + TZ)** is the bulky, often nodular mass
    occupying the centre and anterior of the gland.

The script does NOT relabel, modify, resample or write anything into the
dataset. It reads `t2.nii.gz` + `t2_anatomy_reader1.nii.gz` and writes PNG
figures into `results/label_mapping/`.

Volumes are reoriented to canonical RAS **for display only** (via
`nibabel.as_closest_canonical`) so "posterior" is genuinely posterior in the
rendered image regardless of each case's stored orientation. Panels are drawn
with the correct physical aspect ratio from the voxel spacing, so the thin PZ
rim is not distorted.

Usage:
    python scripts/make_label_mapping_figures.py
    python scripts/make_label_mapping_figures.py --cases 020 059 099 139
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_CASES = ("020", "059", "099", "139")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "prostate158_train", "train")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "label_mapping")

# Deliberately colour-blind-safe and unambiguous.
COLOR_L1 = (0.85, 0.20, 0.20)   # label 1 -- red
COLOR_L2 = (0.15, 0.55, 0.90)   # label 2 -- blue
ALPHA = 0.45


def load_canonical(path):
    """Load a NIfTI and reorient to closest canonical RAS (display only)."""
    img = nib.as_closest_canonical(nib.load(path))
    return np.asarray(img.dataobj), img.header.get_zooms()[:3]


def window(image):
    """Robust intensity window so the overlay stays readable across cases."""
    lo, hi = np.percentile(image[np.isfinite(image)], (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(np.min(image)), float(np.max(image)) or 1.0
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def overlay(ax, img2d, msk2d, aspect, title):
    """Draw one greyscale panel with the two labels overlaid."""
    ax.imshow(img2d.T, cmap="gray", origin="lower", aspect=aspect, vmin=0, vmax=1)
    rgba = np.zeros(msk2d.shape + (4,), dtype=float)
    rgba[msk2d == 1] = (*COLOR_L1, ALPHA)
    rgba[msk2d == 2] = (*COLOR_L2, ALPHA)
    ax.imshow(np.transpose(rgba, (1, 0, 2)), origin="lower", aspect=aspect)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def crop_bbox(mask, pad=28):
    """Bounding box of the whole gland, padded, so the gland fills the panel."""
    idx = np.argwhere(mask > 0)
    if idx.size == 0:
        return None
    lo = np.maximum(idx.min(axis=0) - pad, 0)
    hi = np.minimum(idx.max(axis=0) + pad + 1, mask.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def render_case(case_id, dataset_root, output_dir):
    t2_path = os.path.join(dataset_root, case_id, "t2.nii.gz")
    mask_path = os.path.join(dataset_root, case_id, "t2_anatomy_reader1.nii.gz")
    for p in (t2_path, mask_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    image, zooms = load_canonical(t2_path)
    mask, _ = load_canonical(mask_path)
    mask = np.rint(mask).astype(np.int16)
    if image.shape != mask.shape:
        raise ValueError(f"{case_id}: image {image.shape} != mask {mask.shape}")

    image = window(image.astype(np.float32))
    box = crop_bbox(mask)
    if box is None:
        raise ValueError(f"{case_id}: mask is empty")
    image, mask = image[box], mask[box]
    sx, sy, sz = zooms

    # Centroid of the whole gland -> the slice shown in each plane.
    cx, cy, cz = (int(round(v)) for v in np.argwhere(mask > 0).mean(axis=0))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    # RAS: x = left->Right, y = posterior->Anterior, z = inferior->Superior.
    overlay(axes[0], image[:, :, cz], mask[:, :, cz], aspect=sy / sx,
            title=f"AXIAL (z={cz})\nup = anterior, down = posterior")
    overlay(axes[1], image[:, cy, :], mask[:, cy, :], aspect=sz / sx,
            title=f"CORONAL (y={cy})\nup = superior (base), down = inferior (apex)")
    overlay(axes[2], image[cx, :, :], mask[cx, :, :], aspect=sz / sy,
            title=f"SAGITTAL (x={cx})\nright = anterior, left = posterior")

    n1 = int((mask == 1).sum())
    n2 = int((mask == 2).sum())
    handles = [
        mpatches.Patch(color=COLOR_L1, alpha=0.75,
                       label=f"label 1 — Central Gland (CG = CZ + TZ)   {n1:,} voxels"),
        mpatches.Patch(color=COLOR_L2, alpha=0.75,
                       label=f"label 2 — Peripheral Zone (PZ)   {n2:,} voxels"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle(
        f"Prostate158 case {case_id} — CG/PZ label-mapping verification "
        f"(t2_anatomy_reader1)\n"
        f"label 1 occupies the centre; label 2 forms the thin POSTERIOR crescent "
        f"→ 1 = CG, 2 = PZ    (volume ratio 1:2 = {n1 / max(n2, 1):.2f})",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.90))

    out = os.path.join(output_dir, f"label_verification_{case_id}.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out, n1, n2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"{'case':<8}{'label 1 (CG) voxels':>22}{'label 2 (PZ) voxels':>22}{'ratio 1:2':>12}")
    print("-" * 64)
    for case_id in args.cases:
        out, n1, n2 = render_case(case_id, args.dataset_root, args.output_dir)
        print(f"{case_id:<8}{n1:>22,}{n2:>22,}{n1 / max(n2, 1):>12.2f}")
        print(f"        -> {out}")
    print("\nNothing in dataset/ was read-modified; figures only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
