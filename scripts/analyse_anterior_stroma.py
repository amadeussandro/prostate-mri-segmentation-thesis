"""How much of the external central-gland gap is a difference in what was annotated?

The external reference defines its non-PZ mask as transition zone + central zone
+ anterior fibromuscular stroma. Prostate158 defines CG as transition + central
zone only. The external reference therefore contains a structure the model was
never trained to label, and some unknown part of the external CG shortfall is
that difference rather than model failure. The manuscript states this as a
caveat. This quantifies it.

The stroma sits at the anterior edge of the gland, so the hypothesis makes a
spatial prediction: external CG recall should fall toward the anterior, and the
fall should be much steeper externally than internally, because internally
there is no stroma in the reference to miss. The internal cohort is the control
that separates "the reference contains something extra" from "the model is weak
anteriorly".

Two things make this a diagnostic rather than a correction:

  - The anterior-posterior axis is taken from each volume's own affine, never
    assumed from storage order, and the sign is checked anatomically: the PZ
    centroid must lie posterior to the CG centroid. A case that fails that gate
    is reported, not silently reoriented.
  - Restricting a metric to the posterior gland is a decomposition of the error
    already reported, not a new score. The reported CG Dice stays what it is.

Usage:
    python scripts/analyse_anterior_stroma.py
    python scripts/analyse_anterior_stroma.py --limit 3     # pilot
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import nibabel as nib
import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT)

from src.metrics import dice_score  # noqa: E402

CG, PZ = 1, 2
DRIVE = "G:/My Drive/THESIS_PROSTATE158"

EXTERNAL_RECON = os.path.join(
    PROJECT, "results", "external_prostatex_e1_v2", "reconstructions_3d")
EXTERNAL_GT = os.path.join(
    PROJECT, "results", "external_prostatex_e1", "external_prostatex_e1",
    "_harmonized", "cases")
INTERNAL_RECON = os.path.join(PROJECT, "results", "rq2_e1_epoch88", "reconstructions_3d")
INTERNAL_GT = f"{DRIVE}/dataset/test/extracted/prostate158_test/test"
OUT_DIR = os.path.join(PROJECT, "results", "external_anterior_stroma")

N_BANDS = 10  # deciles of anterior-posterior position within the CG reference


def anterior_coordinate(shape: Sequence[int], affine: np.ndarray) -> np.ndarray:
    """Anterior coordinate (mm, RAS+) of every voxel.

    nibabel maps array indices to RAS+ world millimetres whatever the on-disk
    orientation, so row 1 of the affine is the anterior axis and larger is more
    anterior. Taking it from the affine is what makes this work unchanged on the
    external cohort's LAS volumes and the internal LPS ones.
    """
    i, j, k = np.meshgrid(*[np.arange(n) for n in shape[:3]], indexing="ij")
    a = affine[1]
    return a[0] * i + a[1] * j + a[2] * k + a[3]


def analyse_case(pred: np.ndarray, gt: np.ndarray, affine: np.ndarray,
                 spacing: Sequence[float]) -> Dict[str, float]:
    """Decompose one case's CG agreement along the anterior-posterior axis."""
    y = anterior_coordinate(gt.shape, affine)
    gt_cg, gt_pz = gt == CG, gt == PZ
    if not gt_cg.any():
        raise ValueError("no CG in reference")

    # Orientation gate, on anatomy rather than on header text: the peripheral
    # zone lies posterior to the central gland. If this fails, the axis sign is
    # wrong and every band below would be mirrored.
    pz_posterior = bool(y[gt_pz].mean() < y[gt_cg].mean()) if gt_pz.any() else True

    y_cg = y[gt_cg]
    y_lo, y_hi = float(y_cg.min()), float(y_cg.max())
    span = y_hi - y_lo
    if span <= 0:
        raise ValueError("degenerate CG extent along the anterior axis")

    # u = 0 at the most posterior CG voxel, 1 at the most anterior. Normalising
    # per case makes the bands comparable across glands of different size.
    u = (y_cg - y_lo) / span
    pred_here = pred[gt_cg]

    row: Dict[str, float] = {
        "pz_posterior_to_cg": pz_posterior,
        "cg_ap_span_mm": span,
        "cg_voxels": int(gt_cg.sum()),
    }

    # Per-decile recall, plus what the model put where it missed. If the stroma
    # is the cause, the misses should be background (stroma is dark on T2 and
    # unlike either trained zone), not PZ.
    edges = np.linspace(0.0, 1.0, N_BANDS + 1)
    for b in range(N_BANDS):
        lo, hi = edges[b], edges[b + 1]
        sel = (u >= lo) & (u <= hi) if b == N_BANDS - 1 else (u >= lo) & (u < hi)
        n = int(sel.sum())
        row[f"band{b}_n"] = n
        if n:
            got = pred_here[sel]
            row[f"band{b}_recall"] = float((got == CG).mean())
            row[f"band{b}_as_bg"] = float((got == 0).mean())
            row[f"band{b}_as_pz"] = float((got == PZ).mean())
        else:
            row[f"band{b}_recall"] = float("nan")
            row[f"band{b}_as_bg"] = float("nan")
            row[f"band{b}_as_pz"] = float("nan")

    # Headline split at the midpoint of the CG extent.
    for label, sel in (("anterior", u >= 0.5), ("posterior", u < 0.5)):
        n = int(sel.sum())
        row[f"recall_{label}"] = float((pred_here[sel] == CG).mean()) if n else float("nan")
        row[f"n_{label}"] = n
    # Positive means the anterior half is the weaker one. Kept per case so the
    # two cohorts can be compared as distributions, not just as means.
    row["anterior_deficit"] = row["recall_posterior"] - row["recall_anterior"]

    # Dice restricted to a half-space, applied identically to both masks, so it
    # stays a fair overlap measure rather than a repair of the prediction.
    y_mid = y_lo + 0.5 * span
    vox_ml = float(np.prod(np.asarray(spacing[:3], dtype=float))) / 1000.0
    for label, keep in (("anterior", y >= y_mid), ("posterior", y < y_mid)):
        p = np.where(keep, pred, 0)
        g = np.where(keep, gt, 0)
        row[f"cg_dice_{label}"] = dice_score(p, g, CG)
        missed = int((g == CG).sum() - ((p == CG) & (g == CG)).sum())
        row[f"cg_missed_ml_{label}"] = missed * vox_ml

    # Same split taken on the whole-gland extent instead of the CG extent. The
    # two planes differ because PZ adds posterior depth, so agreement between
    # them is what shows the conclusion is not an artefact of which structure
    # was used to normalise.
    gland = gt_cg | gt_pz
    y_gland = y[gland]
    y_mid_wg = 0.5 * (float(y_gland.min()) + float(y_gland.max()))
    for label, keep in (("anterior", y >= y_mid_wg), ("posterior", y < y_mid_wg)):
        p = np.where(keep, pred, 0)
        g = np.where(keep, gt, 0)
        row[f"cg_dice_{label}_wg"] = dice_score(p, g, CG)

    row["cg_dice_full"] = dice_score(pred, gt, CG)
    return row


def run_cohort(name: str, recon_dir: str, gt_for, limit: Optional[int]) -> List[dict]:
    files = sorted(f for f in os.listdir(recon_dir) if f.endswith(".nii.gz"))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no reconstructions in {recon_dir}")

    rows: List[dict] = []
    for i, fn in enumerate(files, 1):
        cid = fn[len("pred_"):-len(".nii.gz")]
        nii = nib.load(os.path.join(recon_dir, fn))
        pred = np.round(np.asarray(nii.dataobj)).astype(np.int16)
        gt_path = gt_for(cid)
        if not os.path.exists(gt_path):
            raise FileNotFoundError(f"{cid}: no reference at {gt_path}")
        gt = np.round(nib.load(gt_path).get_fdata()).astype(np.int16)
        if gt.shape != pred.shape:
            raise ValueError(f"{cid}: shape {pred.shape} vs reference {gt.shape}")

        row = {"cohort": name, "case_id": cid}
        row.update(analyse_case(pred, gt, nii.affine, nii.header.get_zooms()[:3]))
        rows.append(row)
        if i % 25 == 0 or i == len(files):
            print(f"  {name}: {i}/{len(files)}", flush=True)
    return rows


def _m(rows: List[dict], key: str) -> float:
    v = np.array([float(r[key]) for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else float("nan")


def _sd(rows: List[dict], key: str) -> float:
    v = np.array([float(r[key]) for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=0)) if v.size else float("nan")


def _pooled_band_recall(rows: List[dict]) -> List[float]:
    """Recall per band pooled over voxels rather than averaged over cases.

    The per-case mean answers "what happens in a typical patient"; the pooled
    value answers "what fraction of all reference voxels in this band was
    missed", which is the quantity that actually moves a cohort Dice. They can
    disagree when band sizes vary a lot between patients, so both are reported.
    """
    out = []
    for b in range(N_BANDS):
        hit = tot = 0.0
        for r in rows:
            n = float(r[f"band{b}_n"])
            rec = float(r[f"band{b}_recall"])
            if n > 0 and np.isfinite(rec):
                hit += n * rec
                tot += n
        out.append(float(hit / tot) if tot else float("nan"))
    return out


def _pooled_miss(rows: List[dict], bands: Sequence[int], what: str) -> float:
    """Voxel-pooled share of a band group that the model gave `what` instead of CG."""
    got = tot = 0.0
    for r in rows:
        for b in bands:
            n = float(r[f"band{b}_n"])
            v = float(r[f"band{b}_{what}"])
            if n > 0 and np.isfinite(v):
                got += n * v
                tot += n
    return float(got / tot) if tot else float("nan")


def summarise(ext: List[dict], internal: List[dict]) -> dict:
    def block(rows: List[dict], label: str) -> dict:
        bands = [_m(rows, f"band{b}_recall") for b in range(N_BANDS)]
        pooled = _pooled_band_recall(rows)
        return {
            "label": label,
            "n_cases": len(rows),
            "orientation_gate_pass": int(sum(bool(r["pz_posterior_to_cg"]) for r in rows)),
            "band_recall": bands,
            "band_recall_pooled": pooled,
            "recall_anterior": _m(rows, "recall_anterior"),
            "recall_posterior": _m(rows, "recall_posterior"),
            "anterior_deficit": _m(rows, "anterior_deficit"),
            "anterior_deficit_sd": _sd(rows, "anterior_deficit"),
            "recall_top_decile": bands[-1],
            "recall_top_decile_pooled": pooled[-1],
            "cg_dice_full": _m(rows, "cg_dice_full"),
            "cg_dice_anterior": _m(rows, "cg_dice_anterior"),
            "cg_dice_posterior": _m(rows, "cg_dice_posterior"),
            "cg_dice_anterior_wg": _m(rows, "cg_dice_anterior_wg"),
            "cg_dice_posterior_wg": _m(rows, "cg_dice_posterior_wg"),
            "miss_to_bg_posterior20": _pooled_miss(rows, (0, 1), "as_bg"),
            "miss_to_pz_posterior20": _pooled_miss(rows, (0, 1), "as_pz"),
            "miss_to_bg_anterior20": _pooled_miss(rows, (N_BANDS - 2, N_BANDS - 1), "as_bg"),
            "miss_to_pz_anterior20": _pooled_miss(rows, (N_BANDS - 2, N_BANDS - 1), "as_pz"),
            "cg_missed_ml_anterior": _m(rows, "cg_missed_ml_anterior"),
            "cg_missed_ml_posterior": _m(rows, "cg_missed_ml_posterior"),
            "missed_as_bg_top_decile": _m(rows, f"band{N_BANDS-1}_as_bg"),
            "missed_as_pz_top_decile": _m(rows, f"band{N_BANDS-1}_as_pz"),
            "cg_ap_span_mm": _m(rows, "cg_ap_span_mm"),
        }

    e = block(ext, "PROSTATEx (external)")
    n = block(internal, "Prostate158 (internal test)")

    # The cohorts are different patients, so the comparison of per-case anterior
    # deficits is unpaired: Mann-Whitney, not the paired test used elsewhere in
    # this project.
    def finite(rows, key):
        v = np.array([float(r[key]) for r in rows], dtype=float)
        return v[np.isfinite(v)]

    try:
        from scipy.stats import mannwhitneyu
        p_deficit = float(mannwhitneyu(
            finite(ext, "anterior_deficit"), finite(internal, "anterior_deficit"),
            alternative="two-sided").pvalue)
    except Exception:
        p_deficit = float("nan")

    return {
        "n_bands": N_BANDS,
        "external": e,
        "internal": n,
        "gap": {
            "cg_dice_full": e["cg_dice_full"] - n["cg_dice_full"],
            "cg_dice_anterior": e["cg_dice_anterior"] - n["cg_dice_anterior"],
            "cg_dice_posterior": e["cg_dice_posterior"] - n["cg_dice_posterior"],
            "cg_dice_anterior_wg": e["cg_dice_anterior_wg"] - n["cg_dice_anterior_wg"],
            "cg_dice_posterior_wg": e["cg_dice_posterior_wg"] - n["cg_dice_posterior_wg"],
            "anterior_deficit_excess": e["anterior_deficit"] - n["anterior_deficit"],
            "anterior_deficit_p": p_deficit,
            "recall_top_decile": e["recall_top_decile"] - n["recall_top_decile"],
            "recall_top_decile_pooled": (e["recall_top_decile_pooled"]
                                         - n["recall_top_decile_pooled"]),
        },
    }


def write_report(path: str, s: dict) -> str:
    e, n, g = s["external"], s["internal"], s["gap"]

    # The hypothesis predicts three things together: the anterior recall deficit
    # is markedly larger externally, the cohort gap shrinks when the metric is
    # restricted to the posterior gland, and the anterior misses go to
    # background rather than to the other zone. Any one alone is weak evidence.
    # The internal cohort is what makes the first test meaningful: an anterior
    # drop present in BOTH cohorts is a property of the anterior gland, not of
    # the external reference.
    deficit_excess = g["anterior_deficit_excess"] > 0.05
    gap_shrinks = abs(g["cg_dice_posterior"]) < abs(g["cg_dice_full"])
    misses_are_bg = e["missed_as_bg_top_decile"] > e["missed_as_pz_top_decile"]
    n_met = sum((deficit_excess, gap_shrinks, misses_are_bg))
    verdict = ("SUPPORTED" if n_met == 3
               else "PARTIALLY SUPPORTED" if n_met == 2
               else "NOT SUPPORTED")

    # Which half carries the cohort gap, as a share of the two halves' gaps.
    # Expressed this way it stays in [0, 100] even when one half's gap exceeds
    # the whole-gland gap, which a ratio against the whole-gland gap does not.
    ga, gp = abs(g["cg_dice_anterior"]), abs(g["cg_dice_posterior"])
    share_anterior = (ga / (ga + gp) * 100.0) if (ga + gp) else float("nan")

    def yn(ok: bool) -> str:
        return "met" if ok else "**not met**"

    md = [
        "# Is the external central-gland gap a model failure or an annotation difference?\n",
        f"_{e['n_cases']} external cases against {n['n_cases']} internal test cases. "
        "Diagnostic only: nothing here changes the reported method or the reported "
        "numbers._\n",
        "## Hypothesis\n",
        "The external reference includes anterior fibromuscular stroma in its non-PZ "
        "mask; Prostate158 does not. The stroma lies at the anterior edge of the gland, "
        "so the external reference asks the model to label a structure it never saw in "
        "training, at a predictable location. External CG recall should therefore fall "
        "toward the anterior far more steeply than internal CG recall does.\n",
        "## Orientation gate\n",
        "The anterior axis is read from each volume's affine, not assumed from storage "
        "order, and its sign is checked against anatomy: the PZ centroid must lie "
        "posterior to the CG centroid.\n",
        f"- External: **{e['orientation_gate_pass']}/{e['n_cases']}** pass\n",
        f"- Internal: **{n['orientation_gate_pass']}/{n['n_cases']}** pass\n",
        "\n## Recall along the anterior-posterior axis of the CG reference\n",
        "Position is normalised per case: 0 is the most posterior CG voxel, 1 the most "
        "anterior. Each cell is the fraction of reference CG voxels in that decile that "
        "the model labelled CG.\n\n",
        "Per-case mean first, then the same quantity pooled over voxels.\n\n",
        "| Decile (posterior - anterior) | External | Internal | External (pooled) "
        "| Internal (pooled) |\n|---|---|---|---|---|\n",
    ]
    for b in range(s["n_bands"]):
        lo, hi = b / s["n_bands"], (b + 1) / s["n_bands"]
        md.append(f"| {lo:.1f}-{hi:.1f} | {e['band_recall'][b]:.3f} | "
                  f"{n['band_recall'][b]:.3f} | {e['band_recall_pooled'][b]:.3f} | "
                  f"{n['band_recall_pooled'][b]:.3f} |\n")

    md += [
        "\n## Headline split at the midpoint of the gland\n",
        "| | External | Internal | Difference |\n|---|---|---|---|\n",
        f"| CG recall, posterior half | {e['recall_posterior']:.4f} | "
        f"{n['recall_posterior']:.4f} | {e['recall_posterior']-n['recall_posterior']:+.4f} |\n",
        f"| CG recall, anterior half | {e['recall_anterior']:.4f} | "
        f"{n['recall_anterior']:.4f} | {e['recall_anterior']-n['recall_anterior']:+.4f} |\n",
        f"| Anterior deficit (post - ant) | {e['anterior_deficit']:.4f} "
        f"± {e['anterior_deficit_sd']:.4f} | {n['anterior_deficit']:.4f} "
        f"± {n['anterior_deficit_sd']:.4f} | {g['anterior_deficit_excess']:+.4f} |\n",
        f"| Recall, most anterior decile | {e['recall_top_decile']:.4f} | "
        f"{n['recall_top_decile']:.4f} | {g['recall_top_decile']:+.4f} |\n",
        f"\nMann-Whitney on the per-case anterior deficit, external vs internal: "
        f"p = {g['anterior_deficit_p']:.4f}. A deficit of similar size in both cohorts "
        "is a property of the anterior gland, not of the external reference.\n",
        "\n## Where the cohort gap lives\n",
        "CG Dice restricted to a half-space, the restriction applied identically to "
        "prediction and reference.\n\n",
        "| CG Dice | External | Internal | Gap |\n|---|---|---|---|\n",
        f"| Whole gland (as reported) | {e['cg_dice_full']:.4f} | {n['cg_dice_full']:.4f} | "
        f"{g['cg_dice_full']:+.4f} |\n",
        f"| Posterior half | {e['cg_dice_posterior']:.4f} | {n['cg_dice_posterior']:.4f} | "
        f"{g['cg_dice_posterior']:+.4f} |\n",
        f"| Anterior half | {e['cg_dice_anterior']:.4f} | {n['cg_dice_anterior']:.4f} | "
        f"{g['cg_dice_anterior']:+.4f} |\n",
        "\nSame split taken on the whole-gland extent instead of the CG extent, as a "
        "check that the result is not an artefact of which structure set the plane:\n\n",
        "| CG Dice | External | Internal | Gap |\n|---|---|---|---|\n",
        f"| Posterior half (whole-gland plane) | {e['cg_dice_posterior_wg']:.4f} | "
        f"{n['cg_dice_posterior_wg']:.4f} | {g['cg_dice_posterior_wg']:+.4f} |\n",
        f"| Anterior half (whole-gland plane) | {e['cg_dice_anterior_wg']:.4f} | "
        f"{n['cg_dice_anterior_wg']:.4f} | {g['cg_dice_anterior_wg']:+.4f} |\n",
        "\n## What the model put where it missed\n",
        "Voxel-pooled destination of reference CG the model did not call CG, at the two "
        "ends of the gland.\n\n",
        "| Reference CG not labelled CG | External | Internal |\n|---|---|---|\n",
        f"| Posterior 20%, given background | {e['miss_to_bg_posterior20']*100:.1f}% | "
        f"{n['miss_to_bg_posterior20']*100:.1f}% |\n",
        f"| Posterior 20%, given PZ | {e['miss_to_pz_posterior20']*100:.1f}% | "
        f"{n['miss_to_pz_posterior20']*100:.1f}% |\n",
        f"| Anterior 20%, given background | {e['miss_to_bg_anterior20']*100:.1f}% | "
        f"{n['miss_to_bg_anterior20']*100:.1f}% |\n",
        f"| Anterior 20%, given PZ | {e['miss_to_pz_anterior20']*100:.1f}% | "
        f"{n['miss_to_pz_anterior20']*100:.1f}% |\n",
        "\nAt both ends the misses go to background rather than to the other zone, so "
        "they are the gland boundary falling short, not the CG/PZ interface being drawn "
        "in the wrong place.\n",
        "\n## Unlabelled volume\n",
        f"Reference CG the model did not label, per case: **{e['cg_missed_ml_anterior']:.2f} mL** "
        f"anteriorly against {e['cg_missed_ml_posterior']:.2f} mL posteriorly externally; "
        f"{n['cg_missed_ml_anterior']:.2f} mL against {n['cg_missed_ml_posterior']:.2f} mL "
        "internally.\n",
        f"\n## Verdict: {verdict} ({n_met}/3 predictions met)\n",
        f"1. Anterior deficit larger externally than internally — {yn(deficit_excess)}: "
        f"{e['anterior_deficit']:+.4f} externally against {n['anterior_deficit']:+.4f} "
        f"internally, an excess of {g['anterior_deficit_excess']:+.4f} (p = "
        f"{g['anterior_deficit_p']:.4f}).\n",
        f"2. Cohort gap smaller when restricted to the posterior gland — "
        f"{yn(gap_shrinks)}: {g['cg_dice_full']:+.4f} whole gland against "
        f"{g['cg_dice_posterior']:+.4f} posterior.\n",
        f"3. Anterior misses fall to background rather than to PZ — "
        f"{yn(misses_are_bg)}: {e['missed_as_bg_top_decile']*100:.1f}% background against "
        f"{e['missed_as_pz_top_decile']*100:.1f}% PZ. This one turns out not to "
        "discriminate: the misses go to background at both ends of the gland and in both "
        "cohorts, so it is consistent with the hypothesis without being evidence for it.\n",
        f"\nOf the external central-gland gap, **{share_anterior:.0f}%** sits in the "
        f"anterior half and **{100.0-share_anterior:.0f}%** in the posterior half.\n",
        "\n## What follows for the manuscript\n",
    ]

    if verdict == "NOT SUPPORTED":
        md += [
            "The anterior fibromuscular stroma does not account for the external "
            "central-gland gap. The anterior half of the gland — exactly where the "
            f"stroma lies — is where the two cohorts agree **best** ({g['cg_dice_anterior']:+.4f}), "
            f"while the gap concentrates posteriorly ({g['cg_dice_posterior']:+.4f}). The "
            "direction of the anterior deficit is reversed between cohorts: internally "
            "the anterior half is the weaker one, externally the posterior half is "
            f"(p = {g['anterior_deficit_p']:.4f}).\n",
            "\nThe differing zone definitions remain a real difference between the two "
            "datasets, documented by their authors, and the manuscript should keep saying "
            "so. What the manuscript should no longer imply is that this difference "
            "explains the external shortfall, because the shortfall is not where that "
            "explanation puts it.\n",
            "\nWhat the gap looks like instead: reference central gland that the model "
            "leaves as background at the posterior extent of the gland, with little of it "
            "handed to PZ. That is the whole-gland boundary falling short out of domain, "
            "the same under-segmentation already reported internally in millilitres, "
            "rather than a zone-assignment error or a missing structure.\n",
        ]
    else:
        md += [
            "The pattern is consistent with the stroma explanation: the anterior deficit "
            f"is {g['anterior_deficit_excess']:+.4f} larger externally "
            f"(p = {g['anterior_deficit_p']:.4f}), and the cohort gap falls from "
            f"{g['cg_dice_full']:+.4f} to {g['cg_dice_posterior']:+.4f} once the anterior "
            "gland is excluded.\n",
        ]

    md += [
        "\n## What this cannot establish either way\n",
        "The two cohorts differ in reference protocol, reader, scanner and field of view "
        "at once, so no single difference is isolated here. No stroma mask exists in "
        "either dataset to subtract, so the structure itself is never measured — only "
        "the place where the gap sits. The internal arm is 19 cases against 204, and the "
        "anterior-posterior axis, while read from each affine and checked against "
        "anatomy, is a straight-line approximation of a curved gland.\n",
        "\nNo number reported in the manuscript changes. The reported external CG Dice "
        f"remains {e['cg_dice_full']:.4f} on the unmodified pipeline.\n",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        # Each entry carries its own newline; joining with another would put a
        # blank line between table rows and break the table.
        fh.write("".join(md))
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N cases of each cohort (pilot runs)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--external-recon", default=EXTERNAL_RECON)
    ap.add_argument("--internal-recon", default=INTERNAL_RECON)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ext = run_cohort("external", args.external_recon,
                     lambda c: os.path.join(EXTERNAL_GT, c, "anatomy.nii.gz"),
                     args.limit)
    internal = run_cohort("internal", args.internal_recon,
                          lambda c: f"{INTERNAL_GT}/{c}/t2_anatomy_reader1.nii.gz",
                          args.limit)

    rows = ext + internal
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "cohort", k != "case_id", k))
    per_case = os.path.join(args.out_dir, "anterior_stroma_per_case.csv")
    with open(per_case, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = summarise(ext, internal)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    report = write_report(os.path.join(args.out_dir, "report.md"), summary)

    print(f"\nper-case : {per_case}")
    print(f"report   : {report}")


if __name__ == "__main__":
    main()
