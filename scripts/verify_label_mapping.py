"""Gate 2 — Empirical (programmatic) verification of the CG/PZ label-value mapping.

Prostate158's `t2_anatomy_reader1.nii.gz` uses integer labels {0, 1, 2}, but
secondary sources disagree on which of {1, 2} is Central Gland (CG = CZ+TZ)
and which is Peripheral Zone (PZ):
    - MONAI model card implies: 1 = CG, 2 = PZ
    - Nakic et al. 2026 text implies: 1 = PZ, 2 = CG

This script does NOT relabel anything. It only measures, across a sample of
real cases, morphological properties that distinguish "central" from
"peripheral" anatomy, and reports them plainly:

  1. Unique label values and per-label voxel counts, per case.
  2. Axial slice range covered by each label.
  3. Exterior-contact fraction: of a label's own voxels, what fraction sit on
     the outer boundary of the whole gland (label1 | label2 mask), i.e. are
     adjacent to background. A thin peripheral shell (PZ) should show a much
     higher exterior-contact fraction than a bulky central mass (CG).
  4. Normalized centroid distance: mean distance of each label's voxels to
     the whole-gland centroid, normalized by the gland's effective radius.
     A peripheral label should sit farther from the centroid on average.
  5. Volume ratio label1:label2 across the cohort sample.

Output:
  - results/label_mapping/label_mapping_report.md  (quantitative findings)
  - results/label_mapping/label_mapping_overview_<pid>.png (per-case overlay,
    for quick visual sanity; NOT a substitute for 3D Slicer verification)
  - results/label_mapping/label_mapping_hypothesis.yaml (machine-readable,
    marked unverified)

This script explicitly does NOT declare the mapping resolved. See the
"VERDICT" section printed at the end and written into the report.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy import ndimage

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "prostate158_train", "train")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "label_mapping")

# Spread across the full ID range rather than a contiguous block, so the
# sample is not biased toward one part of the cohort.
SAMPLE_PATIENT_IDS = [
    "020", "029", "039", "049", "059", "069", "079", "089",
    "099", "109", "119", "129", "139", "149", "158",
]


def exterior_contact_fraction(label_mask: np.ndarray, gland_mask: np.ndarray) -> float:
    """Fraction of label_mask voxels that are adjacent to background (outside gland_mask)."""
    if label_mask.sum() == 0:
        return float("nan")
    eroded_gland = ndimage.binary_erosion(gland_mask, iterations=1, border_value=0)
    exterior_shell = gland_mask & ~eroded_gland
    contact_voxels = np.logical_and(label_mask, exterior_shell).sum()
    return float(contact_voxels) / float(label_mask.sum())


def normalized_centroid_distance(label_mask: np.ndarray, gland_mask: np.ndarray, zooms) -> float:
    """Mean physical distance (mm) of label voxels to the whole-gland centroid,
    normalized by the gland's effective radius (mm) so it's comparable across cases."""
    if label_mask.sum() == 0:
        return float("nan")
    zooms = np.array(zooms[:3])
    gland_coords = np.array(np.nonzero(gland_mask)).T * zooms
    centroid = gland_coords.mean(axis=0)
    gland_volume_mm3 = gland_mask.sum() * np.prod(zooms)
    effective_radius = (3.0 * gland_volume_mm3 / (4.0 * np.pi)) ** (1.0 / 3.0)

    label_coords = np.array(np.nonzero(label_mask)).T * zooms
    dists = np.linalg.norm(label_coords - centroid, axis=1)
    return float(dists.mean() / effective_radius)


def inspect_case(pid: str) -> dict:
    patient_dir = os.path.join(DATASET_ROOT, pid)
    t2_path = os.path.join(patient_dir, "t2.nii.gz")
    mask_path = os.path.join(patient_dir, "t2_anatomy_reader1.nii.gz")

    if not os.path.exists(mask_path):
        return {"patient_id": pid, "error": f"missing {mask_path}"}

    t2_nii = nib.load(t2_path)
    mask_nii = nib.load(mask_path)
    mask = np.round(mask_nii.get_fdata()).astype(np.int64)
    zooms = mask_nii.header.get_zooms()

    unique_vals = sorted(int(v) for v in np.unique(mask))

    label1 = mask == 1
    label2 = mask == 2
    gland = label1 | label2

    slices_1 = np.where(label1.any(axis=(0, 1)))[0]
    slices_2 = np.where(label2.any(axis=(0, 1)))[0]

    result = {
        "patient_id": pid,
        "unique_values": unique_vals,
        "voxel_count_1": int(label1.sum()),
        "voxel_count_2": int(label2.sum()),
        "slice_range_1": (int(slices_1.min()), int(slices_1.max())) if slices_1.size else None,
        "slice_range_2": (int(slices_2.min()), int(slices_2.max())) if slices_2.size else None,
        "exterior_contact_1": exterior_contact_fraction(label1, gland),
        "exterior_contact_2": exterior_contact_fraction(label2, gland),
        "centroid_dist_1": normalized_centroid_distance(label1, gland, zooms),
        "centroid_dist_2": normalized_centroid_distance(label2, gland, zooms),
        "volume_ratio_1_to_2": (
            float(label1.sum()) / float(label2.sum()) if label2.sum() > 0 else float("inf")
        ),
    }

    # Quick visual overlay at the mid-gland slice (for our own sanity check only;
    # NOT a replacement for opening t2 + mask together in 3D Slicer).
    t2_data = t2_nii.get_fdata()
    if slices_1.size or slices_2.size:
        all_slices = np.union1d(slices_1, slices_2)
        mid_slice = int(np.median(all_slices))
    else:
        mid_slice = mask.shape[2] // 2

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(t2_data[:, :, mid_slice], cmap="gray")
    overlay = np.zeros((*mask.shape[:2], 4))
    overlay[label1[:, :, mid_slice]] = [1, 0, 0, 0.45]  # red = label 1
    overlay[label2[:, :, mid_slice]] = [0, 0.6, 1, 0.45]  # blue = label 2
    ax.imshow(overlay)
    ax.set_title(f"Patient {pid} — slice {mid_slice}\nred=label1  blue=label2")
    ax.axis("off")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(
        os.path.join(OUTPUT_DIR, f"label_mapping_overview_{pid}.png"),
        dpi=130,
        bbox_inches="tight",
    )
    plt.close(fig)

    return result


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    for pid in SAMPLE_PATIENT_IDS:
        r = inspect_case(pid)
        results.append(r)
        print(pid, r)

    valid = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]

    mean_ext_1 = np.nanmean([r["exterior_contact_1"] for r in valid])
    mean_ext_2 = np.nanmean([r["exterior_contact_2"] for r in valid])
    mean_cd_1 = np.nanmean([r["centroid_dist_1"] for r in valid])
    mean_cd_2 = np.nanmean([r["centroid_dist_2"] for r in valid])
    mean_vol_ratio = np.nanmean([r["volume_ratio_1_to_2"] for r in valid])

    all_unique = set()
    for r in valid:
        all_unique.update(r["unique_values"])

    lines = []
    lines.append("# Gate 2 — Label Mapping Verification (Programmatic)\n")
    lines.append(f"Sample size: {len(valid)} cases (errors: {len(errors)})\n")
    lines.append(f"Sampled patient IDs: {SAMPLE_PATIENT_IDS}\n")
    lines.append(f"\nUnion of unique mask values across sample: {sorted(all_unique)}\n")

    lines.append("\n## Per-case results\n")
    lines.append(
        "| pid | uniq | vox(1) | vox(2) | slices(1) | slices(2) | ext%(1) | ext%(2) | "
        "cdist(1) | cdist(2) | vol1:vol2 |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in valid:
        lines.append(
            f"| {r['patient_id']} | {r['unique_values']} | {r['voxel_count_1']} | "
            f"{r['voxel_count_2']} | {r['slice_range_1']} | {r['slice_range_2']} | "
            f"{r['exterior_contact_1']:.3f} | {r['exterior_contact_2']:.3f} | "
            f"{r['centroid_dist_1']:.3f} | {r['centroid_dist_2']:.3f} | "
            f"{r['volume_ratio_1_to_2']:.2f} |\n"
        )
    for r in errors:
        lines.append(f"| {r['patient_id']} | ERROR: {r['error']} | | | | | | | | | |\n")

    lines.append("\n## Aggregate (mean across sample)\n")
    lines.append(f"- Mean exterior-contact fraction, label 1: **{mean_ext_1:.3f}**\n")
    lines.append(f"- Mean exterior-contact fraction, label 2: **{mean_ext_2:.3f}**\n")
    lines.append(f"- Mean normalized centroid distance, label 1: **{mean_cd_1:.3f}**\n")
    lines.append(f"- Mean normalized centroid distance, label 2: **{mean_cd_2:.3f}**\n")
    lines.append(f"- Mean volume ratio label1:label2: **{mean_vol_ratio:.2f}**\n")

    peripheral_label = 1 if mean_ext_1 > mean_ext_2 else 2
    central_label = 2 if peripheral_label == 1 else 1
    agree_centroid = (
        (peripheral_label == 1 and mean_cd_1 > mean_cd_2)
        or (peripheral_label == 2 and mean_cd_2 > mean_cd_1)
    )

    # Per-case consistency: in how many of the sampled cases does each test
    # individually point to the same peripheral label as the aggregate?
    ext_agree_count = sum(
        1 for r in valid
        if (r["exterior_contact_2"] > r["exterior_contact_1"]) == (peripheral_label == 2)
    )
    cd_agree_count = sum(
        1 for r in valid
        if (r["centroid_dist_2"] > r["centroid_dist_1"]) == (peripheral_label == 2)
    )

    lines.append("\n## Morphological hypothesis (NOT a confirmed ground truth)\n")
    lines.append(
        f"- Exterior-contact fraction suggests label **{peripheral_label}** is the more "
        f"peripheral/shell-like structure (higher fraction of its voxels touch the outer "
        f"gland boundary) -> hypothesis: label {peripheral_label} = **PZ**, "
        f"label {central_label} = **CG/TZ**.\n"
    )
    lines.append(
        f"- Centroid-distance test {'AGREES' if agree_centroid else 'DISAGREES'} with the "
        f"exterior-contact test (on the aggregate mean).\n"
    )
    lines.append(
        f"- Per-case consistency: exterior-contact test agrees with the aggregate "
        f"hypothesis in **{ext_agree_count}/{len(valid)}** sampled cases; "
        f"centroid-distance test agrees in **{cd_agree_count}/{len(valid)}** sampled cases.\n"
    )
    lines.append(
        "- This is quantitative morphological evidence from real data, computed across "
        f"{len(valid)} cases spread across the full patient-ID range. It is consistent "
        "with the MONAI convention (CG=1, PZ=2) "
        + ("if label 2 = PZ" if peripheral_label == 2 else "only if label 1 = PZ, "
           "which would CONTRADICT the MONAI convention and instead match Nakic et al.")
        + ".\n"
    )

    lines.append("\n## VERDICT\n")
    lines.append(
        "**LABEL MAPPING REQUIRES HUMAN / 3D SLICER VERIFICATION.**\n\n"
        "Rationale: morphological heuristics (exterior contact, centroid distance) are "
        "strong supporting evidence but are inference, not ground truth — they cannot "
        "distinguish a genuinely ambiguous or atypical case, and published sources on "
        "this exact dataset already disagree with each other. Per the research protocol, "
        "this gate does not resolve the mapping by assumption. The pipeline continues to "
        "read and report labels as neutral **'label 1'** / **'label 2'** (or 'class 1' / "
        "'class 2') until a human confirms the anatomical identity by opening `t2.nii.gz` "
        "and `t2_anatomy_reader1.nii.gz` together in 3D Slicer for a sample of cases "
        "(suggested: patients 020, 059, 099, 139) and visually identifying which integer "
        "is the thin outer rim (PZ) vs the central mass (CG/TZ).\n"
    )
    lines.append(
        f"\nWorking (unverified) hypothesis for reference during development: "
        f"label {peripheral_label} = PZ, label {central_label} = CG/TZ. "
        "Do not report final thesis per-class results under these names until confirmed.\n"
    )

    report_path = os.path.join(OUTPUT_DIR, "label_mapping_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    hypothesis_yaml = os.path.join(OUTPUT_DIR, "label_mapping_hypothesis.yaml")
    with open(hypothesis_yaml, "w", encoding="utf-8") as f:
        f.write("# UNVERIFIED — see label_mapping_report.md. Do not treat as ground truth.\n")
        f.write("label_mapping_verified: false\n")
        f.write("label_mapping_hypothesis:\n")
        f.write(f"  1: \"{'PZ' if peripheral_label == 1 else 'CG_TZ'}\"\n")
        f.write(f"  2: \"{'PZ' if peripheral_label == 2 else 'CG_TZ'}\"\n")
        f.write("verification_method_required: \"3D Slicer visual inspection, sampled cases\"\n")

    print(f"\nWrote: {report_path}")
    print(f"Wrote: {hypothesis_yaml}")
    print("\nVERDICT: LABEL MAPPING REQUIRES HUMAN / 3D SLICER VERIFICATION")


if __name__ == "__main__":
    main()
