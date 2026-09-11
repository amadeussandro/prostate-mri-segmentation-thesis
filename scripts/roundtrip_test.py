"""Gate 8 -- Round-trip fidelity test (the flagship spatial-verification gate).

GROUND-TRUTH 3D MASK -> preprocessing/slicing -> reconstruction ->
inverse transformation -> original space.

Expected: Dice = 1.0 exactly, per class and overall. NO MODEL is involved --
this isolates geometry error from model error. If this does not hit exactly
1.0, the gate is failed and no model result should be trusted until the
geometry bug is found and fixed.

Two suites are run, both required to pass:

  1. IDENTITY suite -- target_size == each case's own native (H, W), i.e. a
     no-op crop/pad. Isolates orientation standardization + explicit-index
     reconstruction + inverse-transform correctness from any cropping/padding
     concern.

  2. BASELINE-CONFIG suite -- target_size = (442, 442), exactly matching
     configs/config_baseline.yaml (the real Experiment-1 config). Since every
     sampled case's native shape is 270x270 or 232x232 (< 442), this suite
     genuinely exercises the PAD branch of src/transforms.py's crop/pad logic
     on real data -- the identity suite alone does not, because
     target_size == native_shape makes `_compute_axis_pad_crop` return
     op="none" for every axis (verified: this was flagged in the Phase 3
     pre-commit audit as a coverage gap in this exact script). Both suites
     are kept; this one does not replace the identity suite.

Usage:
    python scripts/roundtrip_test.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import nibabel as nib
import numpy as np

from src.dataset import ProstateZonal2DDataset
from src.reconstruction import dice_per_class, reconstruct_case_to_original_space
from src.transforms import PreprocessingConfig

DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset", "prostate158_train", "train")
TEST_PATIENT_IDS = ["020", "029", "059", "099", "139"]
BASELINE_TARGET_SIZE = (442, 442)  # must match configs/config_baseline.yaml preprocessing.target_size


def run_roundtrip_for_case(pid: str, target_size=None) -> dict:
    """Run the round-trip test for one case.

    Parameters
    ----------
    target_size : Optional[Tuple[int, int]]
        If None (default), uses this case's own native (H, W) -- an identity
        crop/pad, isolating orientation/reconstruction correctness. If given
        explicitly (e.g. BASELINE_TARGET_SIZE), exercises the real crop/pad
        branch for cases whose native shape differs from it.
    """
    mask_path = os.path.join(DATASET_ROOT, pid, "t2_anatomy_reader1.nii.gz")
    original_mask_nii = nib.load(mask_path)
    native_shape = original_mask_nii.shape[:2]  # (H, W) as loaded, pre-reorientation

    effective_target = tuple(target_size) if target_size is not None else native_shape

    config = PreprocessingConfig(
        normalization="per_volume",
        z_resample=False,
        spatial_mode="crop_pad",
        target_size=effective_target,
    )

    dataset = ProstateZonal2DDataset(
        dataset_root=DATASET_ROOT,
        patient_ids=[pid],
        slice_sampling="all",
        preprocessing_config=config,
    )

    geometry, transform_meta = dataset.get_case_metadata(pid)

    slices = []
    slice_indices = []
    for i in range(len(dataset)):
        sample = dataset[i]
        slices.append(sample["mask"])
        slice_indices.append(sample["slice_idx"])

    reconstructed_nii = reconstruct_case_to_original_space(
        slices=slices,
        slice_indices=slice_indices,
        transform_meta=transform_meta,
        source_geometry=geometry,
    )

    reconstructed = np.asarray(reconstructed_nii.dataobj)
    original = np.round(original_mask_nii.get_fdata()).astype(np.uint8)

    dice_scores = dice_per_class(reconstructed, original, class_ids=[0, 1, 2])
    shape_match = reconstructed.shape == original.shape
    affine_match = np.allclose(reconstructed_nii.affine, original_mask_nii.affine, atol=1e-4)
    exact_match = bool(np.array_equal(reconstructed, original))

    return {
        "patient_id": pid,
        "target_size": effective_target,
        "native_shape": native_shape,
        "pad_crop_op_h": transform_meta.pad_crop_h.op,
        "pad_crop_op_w": transform_meta.pad_crop_w.op,
        "shape_match": shape_match,
        "affine_match": affine_match,
        "exact_voxel_match": exact_match,
        "dice_per_class": dice_scores,
    }


def _case_passed(result: dict) -> bool:
    return (
        result["shape_match"]
        and result["affine_match"]
        and all(abs(d - 1.0) < 1e-9 for d in result["dice_per_class"].values())
    )


def _run_suite(label: str, patient_ids, target_size, require_pad_exercised: bool) -> bool:
    print(f"\n== {label} (target_size={target_size or 'native per case'}) ==")
    print(f"{'pid':>5} | {'pad_op(h,w)':>13} | {'shape_ok':>8} | {'affine_ok':>9} | {'exact':>6} | dice(0,1,2)")
    suite_passed = True
    for pid in patient_ids:
        result = run_roundtrip_for_case(pid, target_size=target_size)
        dice_str = ", ".join(f"{c}:{d:.6f}" for c, d in result["dice_per_class"].items())
        pad_op_str = f"{result['pad_crop_op_h']},{result['pad_crop_op_w']}"
        print(
            f"{result['patient_id']:>5} | {pad_op_str:>13} | {str(result['shape_match']):>8} | "
            f"{str(result['affine_match']):>9} | {str(result['exact_voxel_match']):>6} | {dice_str}"
        )
        if not _case_passed(result):
            suite_passed = False
        if require_pad_exercised and "pad" not in (result["pad_crop_op_h"], result["pad_crop_op_w"]):
            print(f"  WARNING: expected the pad branch to be exercised for {pid} but got "
                  f"op=({result['pad_crop_op_h']}, {result['pad_crop_op_w']}) -- native shape "
                  f"{result['native_shape']} may already equal target_size {target_size}.")
    return suite_passed


def main() -> bool:
    identity_passed = _run_suite(
        "IDENTITY suite (orientation + reconstruction correctness)",
        TEST_PATIENT_IDS,
        target_size=None,
        require_pad_exercised=False,
    )
    baseline_passed = _run_suite(
        "BASELINE-CONFIG suite (real crop/pad branch, matches config_baseline.yaml)",
        TEST_PATIENT_IDS,
        target_size=BASELINE_TARGET_SIZE,
        require_pad_exercised=True,
    )

    all_passed = identity_passed and baseline_passed

    print()
    if all_passed:
        print("ROUND-TRIP TEST PASSED: Dice = 1.0 exactly for all classes, all sampled cases, both suites.")
    else:
        print("ROUND-TRIP TEST FAILED. STOP -- diagnose the geometry bug before trusting any model result.")
    return all_passed


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
