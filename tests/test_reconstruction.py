"""Gate 7/8 tests: explicit-index reconstruction and the round-trip fidelity
gate (ground-truth mask -> preprocess/slice -> reconstruct -> inverse
transform -> original space must be Dice = 1.0 exactly, no model involved).
"""

import os
import sys
import unittest

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
scripts_dir = os.path.join(project_root, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from src.geometry import assert_complete_slice_indices
from src.reconstruction import dice_per_class, reconstruct_volume_from_slices


class TestReconstructionUnit(unittest.TestCase):
    def test_reconstruct_volume_from_slices_explicit_index(self):
        # slices[i] carries value i; indices says which depth position it belongs to.
        # Reconstruction must place by explicit index, not by list order.
        slices = [np.full((4, 4), i, dtype=np.int64) for i in range(3)]
        indices = [2, 0, 1]  # slices[0]->depth2, slices[1]->depth0, slices[2]->depth1
        vol = reconstruct_volume_from_slices(slices, indices, expected_depth=3)
        expected_value_at_depth = {2: 0, 0: 1, 1: 2}
        for depth, expected_value in expected_value_at_depth.items():
            self.assertTrue(np.all(vol[:, :, depth] == expected_value))

    def test_reconstruct_missing_slice_raises(self):
        slices = [np.zeros((4, 4), dtype=np.int64) for _ in range(2)]
        with self.assertRaises(ValueError):
            reconstruct_volume_from_slices(slices, [0, 2], expected_depth=3)  # missing index 1

    def test_dice_per_class_identical_volumes_is_one(self):
        vol = np.random.default_rng(0).integers(0, 3, size=(10, 10, 4))
        scores = dice_per_class(vol, vol, class_ids=[0, 1, 2])
        for c, d in scores.items():
            self.assertAlmostEqual(d, 1.0, places=9)

    def test_dice_per_class_disjoint_volumes_is_zero(self):
        a = np.zeros((5, 5, 2), dtype=np.int64)
        a[0, 0, 0] = 1
        b = np.zeros((5, 5, 2), dtype=np.int64)
        b[1, 1, 0] = 1
        scores = dice_per_class(a, b, class_ids=[1])
        self.assertAlmostEqual(scores[1], 0.0, places=9)


class TestRoundTripFidelityGate(unittest.TestCase):
    """Gate 8: exact Dice = 1.0 on real ground-truth data, no model."""

    def test_roundtrip_dice_is_exactly_one_identity_target(self):
        """IDENTITY suite: target_size == native shape (isolates orientation +
        explicit-index reconstruction correctness from crop/pad)."""
        dataset_root = os.path.join(project_root, "dataset", "prostate158_train", "train")
        if not os.path.isdir(os.path.join(dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        from roundtrip_test import run_roundtrip_for_case

        for pid in ["020", "029"]:
            result = run_roundtrip_for_case(pid)
            self.assertTrue(result["shape_match"], f"Shape mismatch for {pid}")
            self.assertTrue(result["affine_match"], f"Affine mismatch for {pid}")
            for class_id, dice in result["dice_per_class"].items():
                self.assertAlmostEqual(
                    dice, 1.0, places=9,
                    msg=f"Round-trip Dice != 1.0 for patient {pid}, class {class_id}: {dice}",
                )

    def test_roundtrip_dice_is_exactly_one_with_real_baseline_padding(self):
        """BASELINE-CONFIG suite: target_size=(442,442), exactly matching
        configs/config_baseline.yaml. Patients 020/029 are natively 270x270,
        so this genuinely exercises the PAD branch of the crop/pad logic on
        real data -- the identity-target test above does not (audit finding:
        target_size == native_shape makes AxisPadCrop.op == "none" for every
        axis, so the pad/crop inversion path was previously untested here)."""
        dataset_root = os.path.join(project_root, "dataset", "prostate158_train", "train")
        if not os.path.isdir(os.path.join(dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        from roundtrip_test import BASELINE_TARGET_SIZE, run_roundtrip_for_case

        for pid in ["020", "029"]:
            result = run_roundtrip_for_case(pid, target_size=BASELINE_TARGET_SIZE)

            # Precondition: this must actually be a real pad, not an
            # accidental no-op -- otherwise the test would silently degrade
            # back into the identity case it's meant to complement.
            self.assertIn("pad", (result["pad_crop_op_h"], result["pad_crop_op_w"]),
                           f"Expected the pad branch for {pid} (native {result['native_shape']} "
                           f"vs target {BASELINE_TARGET_SIZE}); got op="
                           f"({result['pad_crop_op_h']}, {result['pad_crop_op_w']})")

            self.assertTrue(result["shape_match"], f"Shape mismatch for {pid}")
            self.assertTrue(result["affine_match"], f"Affine mismatch for {pid}")
            self.assertTrue(result["exact_voxel_match"], f"Exact voxel mismatch for {pid}")
            for class_id, dice in result["dice_per_class"].items():
                self.assertAlmostEqual(
                    dice, 1.0, places=9,
                    msg=f"Round-trip Dice != 1.0 (baseline padding) for patient {pid}, class {class_id}: {dice}",
                )


if __name__ == "__main__":
    unittest.main()
