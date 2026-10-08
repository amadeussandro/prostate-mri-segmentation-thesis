"""Figures for the RQ2 / clinical / external-validation results.

The RQ1 figures already exist (`scripts/make_rq1_figures.py`,
`results/figures_rq1/`). This covers what the later work added and had no
figure at all: the reconstruction ablation, the zone-volume agreement, and the
external cohort.

Every number is read from a result artifact. Nothing is typed in by hand, so a
figure cannot drift away from the table it illustrates.

Palette: validated categorical slots (blue #2a78d6, orange #eb6834,
aqua #1baf7a, yellow #eda100, red #e34948). CG is orange and PZ is blue
throughout, matching the qualitative overlays, so a reader never has to relearn
the colour code between figures.

Usage:
    python scripts/make_rq2_figures.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# --- validated categorical slots -------------------------------------------
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
CG_COLOR, PZ_COLOR = C_ORANGE, C_BLUE
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#8a8a85"
GRID = "#e3e3df"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


def _rows(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _style(ax, ylabel=None, xlabel=None, title=None):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, loc="left", fontweight="bold")


# ---------------------------------------------------------------------------
# F1 -- what each omitted reconstruction step costs
# ---------------------------------------------------------------------------

def fig_ablation(ablation_csv: str, out: str) -> str:
    rows = _rows(ablation_csv)
    variants, seen = [], set()
    for r in rows:
        if r["variant"] not in seen:
            seen.add(r["variant"])
            variants.append(r["variant"])

    label = {
        "V1_no_reorientation": "Orientation\ninversion skipped",
        "V2_naive_centre_crop": "Centre crop\ninstead of offsets",
        "V3_identity_affine": "Affine\nnot restored",
        "V4_append_order": "Slices stacked\nin loader order",
        "V5_all_naive": "All shortcuts\ntogether",
    }

    def val(variant, cls, key):
        return float(next(r[key] for r in rows
                          if r["variant"] == variant and r["class_name"].startswith(cls)))

    ref_cg = val(variants[0], "CG", "dice_vs_gt_correct_mean")
    ref_pz = val(variants[0], "PZ", "dice_vs_gt_correct_mean")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3),
                                   gridspec_kw={"width_ratios": [1.05, 1]})
    x = np.arange(len(variants))
    w = 0.38

    cg = [val(v, "CG", "dice_vs_gt_variant_mean") for v in variants]
    pz = [val(v, "PZ", "dice_vs_gt_variant_mean") for v in variants]
    ax1.bar(x - w / 2, cg, w, color=CG_COLOR, label="CG", zorder=3)
    ax1.bar(x + w / 2, pz, w, color=PZ_COLOR, label="PZ", zorder=3)
    ax1.axhline(ref_cg, color=CG_COLOR, ls="--", lw=1.2, zorder=2)
    ax1.axhline(ref_pz, color=PZ_COLOR, ls="--", lw=1.2, zorder=2)
    ax1.text(len(variants) - 0.42, ref_cg + .018, f"correct pipeline {ref_cg:.3f}",
             color=CG_COLOR, fontsize=7.5, ha="right")
    ax1.text(len(variants) - 0.42, ref_pz + .018, f"correct pipeline {ref_pz:.3f}",
             color=PZ_COLOR, fontsize=7.5, ha="right")
    for xi, (a, b) in enumerate(zip(cg, pz)):
        ax1.text(xi - w / 2, a + .015, f"{a:.2f}", ha="center", fontsize=7.5, color=INK_2)
        ax1.text(xi + w / 2, b + .015, f"{b:.2f}", ha="center", fontsize=7.5, color=INK_2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([label.get(v, v) for v in variants], fontsize=7.2)
    ax1.set_ylim(0, 1.0)
    _style(ax1, ylabel="Dice against expert annotation",
           title="A  Reported accuracy under each shortcut")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)

    # Panel B: which check sees which failure. A single measure per axis -- the
    # point is the pattern of detection, not two magnitudes on one plot.
    def dice_drop(v):
        return (val(v, "CG", "dice_vs_gt_correct_mean") - val(v, "CG", "dice_vs_gt_variant_mean"),
                val(v, "PZ", "dice_vs_gt_correct_mean") - val(v, "PZ", "dice_vs_gt_variant_mean"))

    def caught_frac(v):
        r = next(r for r in rows if r["variant"] == v and r["class_name"].startswith("CG"))
        return int(r["n_caught_by_geometry_check"]), int(r["n_cases"])

    shown = [v for v in variants if v != "V5_all_naive"]
    cells, rlabels = [], []
    for v in shown:
        dcg, dpz = dice_drop(v)
        nc, nt = caught_frac(v)
        detect_overlap = max(dcg, dpz) > 0.01
        detect_geom = nc > 0
        shift_mm = val(v, "CG", "centroid_shift_mm_mean")
        volpct = val(v, "CG", "volume_pct_error_mean")
        note = []
        if detect_overlap:
            note.append(f"Dice -{dcg:.2f} CG / -{dpz:.2f} PZ")
        if shift_mm > 1:
            note.append(f"{shift_mm:.0f} mm shift")
        if volpct > 1:
            note.append(f"{volpct:.0f}% volume")
        cells.append((detect_overlap, detect_geom, " · ".join(note) or "no measurable effect"))
        rlabels.append(label.get(v, v).replace("\n", " "))

    ax2.set_xlim(0, 10)
    ax2.set_ylim(-1.15, len(shown) - 0.4)
    ax2.invert_yaxis()
    ax2.axis("off")
    X_LBL, X_OV, X_GE, X_NOTE = 3.5, 4.55, 6.15, 7.05
    ax2.text(X_OV, -0.95, "Overlap\nmetrics", ha="center", va="center",
             fontsize=8, color=INK, fontweight="bold")
    ax2.text(X_GE, -0.95, "Geometry\nchecklist", ha="center", va="center",
             fontsize=8, color=INK, fontweight="bold")
    for i, ((d_ov, d_ge, note), rl) in enumerate(zip(cells, rlabels)):
        ax2.text(X_LBL, i, rl, ha="right", va="center", fontsize=8, color=INK)
        for xpos, flag in ((X_OV, d_ov), (X_GE, d_ge)):
            ax2.text(xpos, i, "detected" if flag else "missed",
                     ha="center", va="center", fontsize=6.8, zorder=4,
                     color="white" if flag else MUTED,
                     bbox=dict(boxstyle="round,pad=0.42",
                               facecolor=C_RED if flag else GRID,
                               edgecolor="none"))
        ax2.text(X_NOTE, i, note, ha="left", va="center", fontsize=7.2, color=INK_2)
    ax2.set_title("B  Which check sees which failure", loc="left", fontweight="bold", y=1.06)

    fig.text(0.5, -0.04,
             "All variants are rebuilt from the same 2D predictions, so only the reconstruction "
             "differs. The two failure modes are complementary: overlap metrics catch the "
             "mirrored volume but not the lost affine,\nwhile the geometry checklist catches the "
             "lost affine but not the mirrored volume. Neither check alone finds both.",
             ha="center", fontsize=7.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# F2 -- zone volumes: agreement, and what it does to PSA density
# ---------------------------------------------------------------------------

def fig_zone_volumes(per_case_csv: str, out: str) -> str:
    rows = _rows(per_case_csv)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

    for ax, (key, name, col) in zip(axes[:2] , [
            ("cg_ml", "Central gland", CG_COLOR), ("pz_ml", "Peripheral zone", PZ_COLOR)]):
        gt = np.array([float(r[f"gt_{key}"]) for r in rows])
        pr = np.array([float(r[f"pred_{key}"]) for r in rows])
        mean, diff = (gt + pr) / 2, pr - gt
        bias, sd = diff.mean(), diff.std(ddof=1)
        ax.axhline(0, color=MUTED, lw=1)
        ax.axhline(bias, color=col, lw=1.6)
        ax.axhline(bias + 1.96 * sd, color=col, lw=1, ls="--")
        ax.axhline(bias - 1.96 * sd, color=col, lw=1, ls="--")
        ax.scatter(mean, diff, s=34, color=col, alpha=.85, zorder=3,
                   edgecolor="white", linewidth=.8)
        ax.text(ax.get_xlim()[1], bias, f" bias {bias:+.2f}", va="center",
                fontsize=7.5, color=col)
        ax.text(ax.get_xlim()[1], bias + 1.96 * sd, f" +1.96 SD", va="center",
                fontsize=7, color=MUTED)
        ax.text(ax.get_xlim()[1], bias - 1.96 * sd, f" -1.96 SD", va="center",
                fontsize=7, color=MUTED)
        _style(ax, ylabel="Predicted - reference (mL)", xlabel="Mean of the two (mL)",
               title=f"{name}")

    ax = axes[2]
    gt = np.array([float(r["gt_whole_gland_ml"]) for r in rows])
    pr = np.array([float(r["pred_whole_gland_ml"]) for r in rows])
    ratio = gt / pr
    order = np.argsort(ratio)
    colors = [C_RED if v > 1 else C_AQUA for v in ratio[order]]
    ax.barh(np.arange(len(ratio)), (ratio[order] - 1) * 100, color=colors, zorder=3)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline((ratio.mean() - 1) * 100, color=INK, lw=1.2, ls="--", zorder=4)
    ax.set_ylim(len(ratio) + 1.6, -1.4)
    ax.text((ratio.mean() - 1) * 100, -1.1,
            f"mean +{(ratio.mean()-1)*100:.1f}%", fontsize=7.5, color=INK,
            va="bottom", ha="center")
    ax.set_yticks([])
    _style(ax, xlabel="Error in PSA density (%)", ylabel="Cases (sorted)",
           title="Consequence for PSA density")
    ax.text(0.98, 0.04,
            f"over-estimated in {int((ratio>1).sum())}/{len(ratio)} cases",
            transform=ax.transAxes, fontsize=7.5, color=INK_2, ha="right")

    fig.text(0.5, -0.06,
             "Bland-Altman agreement for the two zones and the resulting error in PSA density. "
             "Because PSA density divides by gland volume, under-segmentation inflates it, and "
             "the bias runs\ntoward more biopsies rather than fewer. The limits of agreement, "
             "not the bias, determine whether a volume is usable for an individual patient.",
             ha="center", fontsize=7.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# F3 -- internal versus external cohort
# ---------------------------------------------------------------------------

def fig_external(internal_csv: str, external_csv: str, geometry_csv: str, out: str) -> str:
    def per_class(path, cls_id, metric):
        return np.array([float(r[metric]) for r in _rows(path)
                         if r["class_id"] == str(cls_id)])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8),
                             gridspec_kw={"width_ratios": [1, 1, 1.15]})

    ax = axes[0]
    labels, x = ["CG", "PZ"], np.arange(2)
    w = 0.36
    for off, (path, name, alpha) in [(-w / 2, (internal_csv, "Prostate158 (n=19)", 1.0)),
                                     (w / 2, (external_csv, "PROSTATEx (n=204)", 0.55))]:
        vals = [per_class(path, c, "dice").mean() for c in (1, 2)]
        errs = [per_class(path, c, "dice").std() for c in (1, 2)]
        cols = [CG_COLOR, PZ_COLOR]
        ax.bar(x + off, vals, w, yerr=errs, capsize=3, zorder=3,
               color=cols, alpha=alpha, label=name,
               error_kw={"ecolor": MUTED, "lw": 1})
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + .035, f"{v:.3f}", ha="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    _style(ax, ylabel="Dice", title="A  Internal vs external")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    for cls, col, name in [(1, CG_COLOR, "CG"), (2, PZ_COLOR, "PZ")]:
        ext = per_class(external_csv, cls, "hd95_mm")
        ax.hist(ext, bins=28, color=col, alpha=.7, zorder=3, label=f"{name} (PROSTATEx)")
    med = np.median(per_class(external_csv, 1, "hd95_mm"))
    mean = per_class(external_csv, 1, "hd95_mm").mean()
    ax.axvline(med, color=INK, lw=1.2, ls="--")
    ax.text(med, ax.get_ylim()[1] * .95, f" median {med:.0f} mm", fontsize=7.5, color=INK)
    ax.axvline(mean, color=C_RED, lw=1.2)
    ax.text(mean, ax.get_ylim()[1] * .82, f" mean {mean:.0f} mm", fontsize=7.5, color=C_RED)
    _style(ax, xlabel="HD95 (mm)", ylabel="Cases",
           title="B  A heavy tail, not a uniform shift")
    ax.legend(fontsize=8)

    ax = axes[2]
    geo = _rows(geometry_csv)
    scale = np.array([float(g["scale_vs_prostate158"]) for g in geo])
    hd = per_class(external_csv, 1, "hd95_mm")
    prec = per_class(external_csv, 1, "precision")
    sc = ax.scatter(prec, hd, c=scale, cmap="BuPu", s=34, zorder=3,
                    edgecolor="white", linewidth=.6)
    cb = fig.colorbar(sc, ax=ax, pad=.02)
    cb.set_label("in-plane scale vs Prostate158", fontsize=8)
    cb.outline.set_visible(False)
    r = np.corrcoef(prec, hd)[0, 1]
    ax.text(0.03, 0.95, f"r = {r:+.2f}", transform=ax.transAxes,
            fontsize=9, color=INK, va="top", fontweight="bold")
    _style(ax, xlabel="CG precision", ylabel="CG HD95 (mm)",
           title="C  Boundary error tracks false positives")

    fig.text(0.5, -0.06,
             "Overlap transfers to the external cohort with a small drop, but the boundary "
             "metric does not: its median stays near the internal value while the mean is "
             "pulled out by a tail of\ncases. HD95 rises as precision falls, which is the "
             "signature of small distant false positives rather than a uniformly displaced "
             "boundary.",
             ha="center", fontsize=7.5, color=INK_2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# F4 -- the error a header-only checklist cannot see
# ---------------------------------------------------------------------------

def fig_mirrored(rq2_dir: str, gt_root: str, case: str, out: str) -> str:
    """One case, reconstructed correctly and with the orientation inversion
    skipped, with each one's verdict from the geometry checklist printed on it.

    The figure exists because the verdicts are identical. A reader can see that
    the lower row is anatomically mirrored while the checklist calls it valid,
    which is the whole argument for a fidelity test that looks at voxels rather
    than headers.
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    import nibabel as nib
    from run_reconstruction_ablation import build_variant, caught_by_geometry_check, dice
    from src.geometry import read_geometry

    meta = json.load(open(os.path.join(rq2_dir, "metadata", f"meta_{case}.json"), encoding="utf-8"))
    z = np.load(os.path.join(rq2_dir, "predictions_2d", f"pred_{case}.npz"))
    pred, idx = z["pred_preprocessed"], z["slice_indices"]
    t2 = np.asarray(nib.load(os.path.join(gt_root, case, "t2.nii.gz")).get_fdata())
    gt = np.round(nib.load(os.path.join(gt_root, case,
                                        "t2_anatomy_reader1.nii.gz")).get_fdata()).astype(np.int64)

    ref, ref_aff = build_variant("V0_correct", meta, pred, idx)
    mir, mir_aff = build_variant("V1_no_reorientation", meta, pred, idx)
    src_geom = read_geometry(nib.Nifti1Image(ref, ref_aff))

    zooms = meta["spacing_mm"]
    k = int(np.argmax([(gt[:, :, s] > 0).sum() for s in range(gt.shape[2])]))

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.4))
    rows = [
        (ref, ref_aff, "Correct reconstruction", ref),
        (mir, mir_aff, "Orientation inversion skipped", mir),
    ]
    for r, (vol, aff, title, _v) in enumerate(rows):
        caught = caught_by_geometry_check(vol, aff, src_geom)
        dcg, dpz = dice(vol, gt, 1), dice(vol, gt, 2)
        for c, (data, name) in enumerate([(gt, "Expert annotation"),
                                          (vol, "Prediction"),
                                          (None, "Agreement")]):
            ax = axes[r][c]
            ax.imshow(t2[:, :, k].T, cmap="gray", origin="lower", aspect=zooms[1] / zooms[0])
            if data is not None:
                ov = np.zeros((*data[:, :, k].shape, 4))
                for cid, col in ((1, CG_COLOR), (2, PZ_COLOR)):
                    m = data[:, :, k] == cid
                    rgb = matplotlib.colors.to_rgb(col)
                    ov[m, 0], ov[m, 1], ov[m, 2], ov[m, 3] = rgb[0], rgb[1], rgb[2], .62
                ax.imshow(np.transpose(ov, (1, 0, 2)), origin="lower",
                          aspect=zooms[1] / zooms[0])
            else:
                agree = (vol[:, :, k] == gt[:, :, k]) & (gt[:, :, k] > 0)
                wrong = (vol[:, :, k] != gt[:, :, k]) & ((gt[:, :, k] > 0) | (vol[:, :, k] > 0))
                ov = np.zeros((*agree.shape, 4))
                ov[agree] = (*matplotlib.colors.to_rgb(C_AQUA), .65)
                ov[wrong] = (*matplotlib.colors.to_rgb(C_RED), .65)
                ax.imshow(np.transpose(ov, (1, 0, 2)), origin="lower",
                          aspect=zooms[1] / zooms[0])
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(name, fontsize=9, color=INK)
        axes[r][0].set_ylabel(title, fontsize=9.5, fontweight="bold", color=INK)
        verdict = "FLAGGED" if caught else "PASSES"
        axes[r][2].text(
            1.04, 0.5,
            f"Geometry checklist\n{verdict}\n\nCG Dice {dcg:.3f}\nPZ Dice {dpz:.3f}",
            transform=axes[r][2].transAxes, fontsize=8.5, va="center", ha="left",
            color=INK, bbox=dict(boxstyle="round,pad=0.55",
                                 facecolor="#fdecea" if not caught and r == 1 else "#f3f3f0",
                                 edgecolor=C_RED if not caught and r == 1 else GRID))

    fig.suptitle(f"Case {case}, axial slice {k} — the same checklist verdict, "
                 "two very different volumes",
                 fontsize=10.5, fontweight="bold", color=INK, y=0.985)
    fig.text(0.5, 0.015,
             "Both rows pass every geometric check: shape, affine, voxel spacing, orientation "
             "codes and label set are correct in each, because the header is correct and only "
             "the voxel data differs.\nOnly the round-trip fidelity test, which pushes the "
             "annotation through the same transform and requires it back unchanged, "
             "distinguishes them.",
             ha="center", fontsize=7.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.045, 0.88, 0.96))
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Figures for RQ2, clinical and external results.")
    ap.add_argument("--out-dir", default="results/figures_rq2")
    ap.add_argument("--ablation", default="results/rq2_ablation_naive/ablation_summary.csv")
    ap.add_argument("--volumes", default="results/rq2_zone_volumes/zone_volumes_per_case.csv")
    ap.add_argument("--internal", default="results/rq2_e1_epoch88/metrics_per_case.csv")
    ap.add_argument("--external", default="results/external_prostatex_e1_v2/metrics_per_case.csv")
    ap.add_argument("--geometry", default="results/external_prostatex_e1_v2/cohort_geometry.csv")
    ap.add_argument("--rq2-dir", default="results/rq2_e1_epoch88")
    ap.add_argument("--gt-root",
                    default="G:/My Drive/THESIS_PROSTATE158/dataset/test/extracted/prostate158_test/test")
    ap.add_argument("--mirror-case", default="002")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    made = []
    if os.path.exists(args.ablation):
        made.append(fig_ablation(args.ablation,
                                 os.path.join(args.out_dir, "F1_reconstruction_ablation.png")))
    if os.path.exists(args.volumes):
        made.append(fig_zone_volumes(args.volumes,
                                     os.path.join(args.out_dir, "F2_zone_volume_agreement.png")))
    if os.path.exists(args.external) and os.path.exists(args.geometry):
        made.append(fig_external(args.internal, args.external, args.geometry,
                                 os.path.join(args.out_dir, "F3_external_validation.png")))
    if os.path.isdir(args.gt_root):
        made.append(fig_mirrored(args.rq2_dir, args.gt_root, args.mirror_case,
                                 os.path.join(args.out_dir, "F4_mirrored_volume_passes.png")))
    for p in made:
        print("wrote", p)
    if not made:
        print("nothing written -- check the input paths")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Workflow diagrams
# ---------------------------------------------------------------------------

def _flow(ax, boxes, x=0.5, w=0.86, top=0.97, gap=0.015, fs=8.4):
    """Stack labelled boxes with arrows between them."""
    n = len(boxes)
    h = (top - 0.03 - gap * (n - 1)) / n
    y = top
    for text, face, edge in boxes:
        ax.add_patch(plt.Rectangle((x - w / 2, y - h), w, h, transform=ax.transAxes,
                                   facecolor=face, edgecolor=edge, linewidth=1.3, zorder=2))
        ax.text(x, y - h / 2, text, transform=ax.transAxes, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3)
        if y - h - gap > 0.02:
            ax.annotate("", xy=(x, y - h - gap), xytext=(x, y - h),
                        xycoords=ax.transAxes, textcoords=ax.transAxes,
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.3))
        y -= h + gap
    ax.axis("off")


BLUE_F, BLUE_E = "#e8f0fb", C_BLUE
ORA_F, ORA_E = "#fdeee7", C_ORANGE
AQUA_F, AQUA_E = "#e6f6f0", C_AQUA
GREY_F, GREY_E = "#f2f2ef", MUTED


def fig_study_pipeline(out: str) -> str:
    fig, ax = plt.subplots(figsize=(6.6, 8.4))
    _flow(ax, [
        ("Prostate158 — 158 examinations, axial T2-weighted\n"
         "official patient-level split 119 / 20 / 19", BLUE_F, BLUE_E),
        ("Preprocessing: RAS orientation, per-volume z-score,\n"
         "crop/pad to 442×442, no through-plane resampling\n"
         "(inverse parameters retained per case)", BLUE_F, BLUE_E),
        ("Axial 2D slice extraction, each tagged with its axial index", BLUE_F, BLUE_E),
        ("Four single-factor training configurations\n"
         "baseline CE · + soft Dice · + augmentation · + class-weighted CE", ORA_F, ORA_E),
        ("Model selection on the 20 validation cases\n"
         "rule fixed in advance: highest macro Dice", ORA_F, ORA_E),
        ("Selected model frozen — identified by SHA-256 of its checkpoint", AQUA_F, AQUA_E),
        ("Single evaluation on the 19 held-out test cases\n"
         "per case, volume level, original voxel space", AQUA_F, AQUA_E),
        ("2D → 3D reconstruction with verification  (see Figure 3)", GREY_F, GREY_E),
        ("Reconstruction ablation · zone volumes in mL ·\n"
         "external validation on 204 PROSTATEx examinations", GREY_F, GREY_E),
    ], fs=8.0)
    ax.set_title("Study workflow", fontsize=11, fontweight="bold", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def fig_reconstruction_workflow(out: str) -> str:
    fig, ax = plt.subplots(figsize=(6.8, 8.6))
    _flow(ax, [
        ("3D T2-weighted volume, original voxel space", BLUE_F, BLUE_E),
        ("Preprocessing + retained transform metadata", BLUE_F, BLUE_E),
        ("Axial 2D slice extraction", BLUE_F, BLUE_E),
        ("Frozen 2D U-Net — per-slice inference, arg-max\n"
         "(selected configuration, epoch 88)", AQUA_F, AQUA_E),
        ("Reassembly by EXPLICIT axial index\n"
         "a missing or duplicated index raises", ORA_F, ORA_E),
        ("Inverse preprocessing in reverse order\n"
         "nearest-neighbour throughout, labels preserved exactly", ORA_F, ORA_E),
        ("Source affine, spacing and orientation restored\n"
         "never a default identity affine", ORA_F, ORA_E),
        ("Write NIfTI in original voxel space", ORA_F, ORA_E),
        ("Check 1 — per-case geometric checklist, read back\n"
         "from the output header: shape, affine, spacing,\n"
         "orientation codes, label set", AQUA_F, AQUA_E),
        ("Check 2 — model-free round-trip fidelity:\n"
         "annotation forward and back must return unchanged\n"
         "(reported as pass/fail, never as an accuracy)", AQUA_F, AQUA_E),
        ("Evaluation per case against the untouched annotation,\n"
         "plus zone volumes in millilitres", GREY_F, GREY_E),
    ], fs=7.8)
    ax.set_title("2D-to-3D reconstruction and verification", fontsize=11,
                 fontweight="bold", color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out
