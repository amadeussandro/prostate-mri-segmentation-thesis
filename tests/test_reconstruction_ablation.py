"""RQ2 reconstruction-ablation tests -- torch-free, no dataset or checkpoint.

The ablation exists to turn the manuscript's "naive reconstruction is not
trivial" assertion into a measurement. These tests pin the properties that make
that measurement trustworthy:

  * the correct variant is a genuine inverse (rebuild == original),
  * each shortcut damages exactly what it is supposed to damage,
  * and -- the finding the ablation exists for -- the geometry checklist is
    blind to a mirrored volume whose header is correct, while it does catch a
    wrong affine.

If the last property ever flips, the argument for keeping the round-trip
fidelity test has changed and the report's Section 4 must be rewritten.
"""

import os
import sys
import unittest

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import nibabel as nib

from run_reconstruction_ablation import (
    assemble_by_append,
    assemble_by_index,
    build_variant,
    caught_by_geometry_check,
    centroid_mm,
    dice,
    forward_ornt,
    undo_pad_centre,
    undo_pad_recorded,
    volume_ml,
)
from src.geometry import read_geometry


def _lps_affine(sx=0.5, sy=0.5, sz=3.0):
    """An LPS affine like Prostate158's: negative x/y, anisotropic z."""
    a = np.diag([-sx, -sy, sz, 1.0]).astype(np.float64)
    a[:3, 3] = [40.0, 10.0, -50.0]
    return a


def _meta(shape=(12, 12, 4), target=20, affine=None):
    affine = _lps_affine() if affine is None else affine
    h, w, d = shape
    start = (target - h) // 2
    return {
        "original_shape": list(shape),
        "affine": affine.tolist(),
        "transform_meta": {
            "post_resample_depth": d,
            "pad_crop_h": {"op": "pad", "native_size": h, "target_size": target, "start": start},
            "pad_crop_w": {"op": "pad", "native_size": w, "target_size": target, "start": start},
        },
    }


def _synthetic_preprocessed(meta, seed=0):
    """A label volume in preprocessed space, built by the forward transform so
    the correct inverse must recover it exactly."""
    tm = meta["transform_meta"]
    th = tm["pad_crop_h"]["target_size"]
    d = tm["post_resample_depth"]
    h, w = tm["pad_crop_h"]["native_size"], tm["pad_crop_w"]["native_size"]
    rng = np.random.default_rng(seed)
    native = rng.integers(0, 3, size=(h, w, d)).astype(np.uint8)
    oriented = nib.orientations.apply_orientation(
        native, forward_ornt(np.array(meta["affine"])))
    sh = tm["pad_crop_h"]["start"]
    sw = tm["pad_crop_w"]["start"]
    pre = np.zeros((th, th, d), dtype=np.uint8)
    pre[sh:sh + oriented.shape[0], sw:sw + oriented.shape[1], :] = oriented
    return native, pre


class TestAssembly(unittest.TestCase):
    def test_explicit_index_restores_shuffled_slices(self):
        pred = np.stack([np.full((3, 3), k, dtype=np.uint8) for k in (2, 0, 1)], axis=-1)
        out = assemble_by_index(pred, [2, 0, 1], 3)
        for k in range(3):
            self.assertTrue((out[:, :, k] == k).all())

    def test_append_order_scrambles_shuffled_slices(self):
        pred = np.stack([np.full((3, 3), k, dtype=np.uint8) for k in (2, 0, 1)], axis=-1)
        out = assemble_by_append(pred, [2, 0, 1], 3)
        self.assertFalse((out[:, :, 0] == 0).all())

    def test_append_order_is_harmless_when_already_sorted(self):
        pred = np.stack([np.full((3, 3), k, dtype=np.uint8) for k in (0, 1, 2)], axis=-1)
        a = assemble_by_index(pred, [0, 1, 2], 3)
        b = assemble_by_append(pred, [0, 1, 2], 3)
        self.assertTrue((a == b).all())


class TestPadInversion(unittest.TestCase):
    def test_centre_crop_matches_recorded_when_padding_is_symmetric(self):
        pc = {"op": "pad", "native_size": 10, "target_size": 20, "start": 5}
        vol = np.random.default_rng(3).integers(0, 3, (20, 20, 2)).astype(np.uint8)
        self.assertTrue((undo_pad_recorded(vol, pc, pc) == undo_pad_centre(vol, pc, pc)).all())

    def test_centre_crop_always_matches_for_this_pipelines_pad_convention(self):
        # _compute_axis_pad_crop always places the pad at (target - native) // 2,
        # so a centre-crop can never diverge from the recorded offset for a PAD.
        # V2 is therefore structurally harmless here -- a property of the
        # convention, not of the cohort.
        for native, target in [(10, 20), (10, 21), (232, 442), (3, 8)]:
            pc = {"op": "pad", "native_size": native, "target_size": target,
                  "start": (target - native) // 2}
            vol = np.random.default_rng(native).integers(0, 3, (target, target, 2)).astype(np.uint8)
            self.assertTrue(
                (undo_pad_recorded(vol, pc, pc) == undo_pad_centre(vol, pc, pc)).all(),
                msg=f"diverged for native={native}, target={target}")

    def test_centre_crop_diverges_when_the_offset_is_not_centred(self):
        # A different implementation (or a future asymmetric convention) would
        # record an off-centre start; the naive crop ignores it and takes the
        # middle regardless.
        pc = {"op": "pad", "native_size": 10, "target_size": 21, "start": 2}
        vol = np.random.default_rng(4).integers(1, 3, (21, 21, 2)).astype(np.uint8)
        self.assertFalse((undo_pad_recorded(vol, pc, pc) == undo_pad_centre(vol, pc, pc)).all())

    def test_centre_crop_cannot_undo_a_forward_crop(self):
        # Forward CROP (native > target) must be inverted by padding zeros back.
        # The naive path has nothing to crop and returns the wrong shape.
        pc = {"op": "crop", "native_size": 20, "target_size": 10, "start": 5}
        vol = np.random.default_rng(5).integers(1, 3, (10, 10, 2)).astype(np.uint8)
        self.assertEqual(undo_pad_recorded(vol, pc, pc).shape[:2], (20, 20))
        self.assertEqual(undo_pad_centre(vol, pc, pc).shape[:2], (10, 10))


class TestPhysicalQuantities(unittest.TestCase):
    def test_volume_scales_with_voxel_size(self):
        vol = np.zeros((4, 4, 4), dtype=np.uint8)
        vol[:2, :2, :2] = 1  # 8 voxels
        correct = volume_ml(vol, _lps_affine(), 1)
        identity = volume_ml(vol, np.eye(4), 1)
        self.assertAlmostEqual(correct, 8 * 0.5 * 0.5 * 3.0 / 1000.0, places=9)
        self.assertAlmostEqual(identity, 8 / 1000.0, places=9)
        self.assertNotAlmostEqual(correct, identity, places=6)

    def test_dice_is_nan_when_shapes_differ(self):
        a = np.ones((4, 4, 4), dtype=np.uint8)
        b = np.ones((4, 4, 3), dtype=np.uint8)
        self.assertTrue(np.isnan(dice(a, b, 1)))

    def test_centroid_is_none_for_absent_class(self):
        self.assertIsNone(centroid_mm(np.zeros((3, 3, 3), dtype=np.uint8), np.eye(4), 1))


class TestVariants(unittest.TestCase):
    def setUp(self):
        self.meta = _meta()
        self.native, self.pre = _synthetic_preprocessed(self.meta)
        self.idx = list(range(self.pre.shape[2]))

    def test_correct_variant_is_an_exact_inverse(self):
        vol, aff = build_variant("V0_correct", self.meta, self.pre, self.idx)
        self.assertEqual(vol.shape, self.native.shape)
        self.assertTrue((vol == self.native).all())
        self.assertTrue(np.allclose(aff, np.array(self.meta["affine"])))

    def test_skipping_reorientation_changes_the_voxels(self):
        correct, _ = build_variant("V0_correct", self.meta, self.pre, self.idx)
        naive, _ = build_variant("V1_no_reorientation", self.meta, self.pre, self.idx)
        self.assertEqual(correct.shape, naive.shape)
        self.assertFalse((correct == naive).all())

    def test_identity_affine_keeps_voxels_but_loses_geometry(self):
        correct, aff_c = build_variant("V0_correct", self.meta, self.pre, self.idx)
        naive, aff_n = build_variant("V3_identity_affine", self.meta, self.pre, self.idx)
        self.assertTrue((correct == naive).all())          # voxels untouched
        self.assertFalse(np.allclose(aff_c, aff_n))        # geometry destroyed


class TestWhatTheGeometryChecklistCanSee(unittest.TestCase):
    """The ablation's headline result, pinned as a test."""

    def setUp(self):
        self.meta = _meta()
        _, self.pre = _synthetic_preprocessed(self.meta)
        self.idx = list(range(self.pre.shape[2]))
        ref, ref_aff = build_variant("V0_correct", self.meta, self.pre, self.idx)
        self.source = read_geometry(nib.Nifti1Image(ref, ref_aff))

    def _caught(self, variant):
        vol, aff = build_variant(variant, self.meta, self.pre, self.idx)
        return caught_by_geometry_check(vol, aff, self.source)

    def test_correct_variant_is_not_flagged(self):
        self.assertFalse(self._caught("V0_correct"))

    def test_wrong_affine_is_caught(self):
        self.assertTrue(self._caught("V3_identity_affine"))

    def test_mirrored_volume_is_NOT_caught(self):
        # The whole argument for the round-trip fidelity test: a header-only
        # checklist cannot see that the voxel data has been mirrored.
        self.assertFalse(self._caught("V1_no_reorientation"))


if __name__ == "__main__":
    unittest.main()
