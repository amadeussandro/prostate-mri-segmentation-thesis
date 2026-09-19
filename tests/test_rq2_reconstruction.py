"""RQ2 reconstruction tests -- torch-free, no real dataset or checkpoint needed.

Covers the pieces added for RQ2 (3D reconstruction of the frozen 2D U-Net's
predictions) that can be validated without the deep-learning stack:

  * validate_reconstruction_geometry (Step 5): shape/affine/spacing/orientation
    /label-validity table on a reconstructed NIfTI.
  * synthetic round-trip fidelity (Step 6): GT mask -> preprocess -> slice ->
    reconstruct -> inverse -> original space is Dice = 1.0 exactly, shape and
    affine exact, NO model involved (exercises the real crop/pad PAD branch).
  * objective representative-case selection (Step 8).
  * NIfTI save/reload preserves geometry (Step 4).
  * 3D-projection rendering smoke test (Step 8).

The model-dependent parts (real inference on the 19 test cases) run on
Colab/GPU via scripts/run_rq2_reconstruction.py and are not covered here.
"""

import os
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import nibabel as nib

from src.reconstruction import (
    dice_per_class,
    reconstruct_case_to_original_space,
    validate_reconstruction_geometry,
)
from src.geometry import read_geometry
from src.metrics import CaseMetrics
from src.transforms import PreprocessingConfig, preprocess_case


def _synthetic_case(h=40, w=44, d=6, target=(48, 48)):
    """Build a synthetic T2 + zonal mask (labels 0/1/2) on an RAS affine, run the
    baseline crop/pad preprocessing, and return everything needed to round-trip.
    target > native forces the PAD branch of the crop/pad inversion."""
    rng = np.random.default_rng(0)
    affine = np.diag([0.5, 0.5, 3.0, 1.0]).astype(np.float64)
    affine[:3, 3] = [1.0, 2.0, 3.0]

    mask = np.zeros((h, w, d), dtype=np.int64)
    # Central gland (label 1): a central block. Peripheral zone (label 2): a rim.
    mask[14:26, 16:28, 1:5] = 1
    mask[10:30, 12:32, 1:5][mask[10:30, 12:32, 1:5] == 0] = 2
    t2 = rng.random((h, w, d)).astype(np.float32)

    t2_nii = nib.Nifti1Image(t2, affine)
    mask_nii = nib.Nifti1Image(mask.astype(np.float32), affine)
    cfg = PreprocessingConfig(
        normalization="per_volume", z_resample=False,
        spatial_mode="crop_pad", target_size=target,
    )
    _t2p, maskp, meta = preprocess_case(t2_nii, mask_nii, cfg)
    return t2_nii, mask_nii, maskp, meta


class TestSyntheticRoundTripFidelity(unittest.TestCase):
    """Step 6: reconstruction transform is loss-free (Dice = 1.0), no model."""

    def test_roundtrip_dice_is_exactly_one(self):
        t2_nii, mask_nii, maskp, meta = _synthetic_case()

        # PAD branch must actually be exercised (target > native), else the test
        # silently degrades to the identity case.
        self.assertIn("pad", (meta.pad_crop_h.op, meta.pad_crop_w.op))

        slices = [maskp[:, :, k] for k in range(maskp.shape[2])]
        indices = list(range(maskp.shape[2]))
        recon_nii = reconstruct_case_to_original_space(
            slices, indices, meta, meta.original_geometry)
        recon = np.asarray(recon_nii.dataobj)
        original = np.round(mask_nii.get_fdata()).astype(np.int64)

        self.assertEqual(recon.shape, original.shape)
        self.assertTrue(np.allclose(recon_nii.affine, mask_nii.affine, atol=1e-4))
        self.assertTrue(np.array_equal(recon, original), "reconstruction is not voxel-exact")
        for c, dice in dice_per_class(recon, original, [0, 1, 2]).items():
            self.assertAlmostEqual(dice, 1.0, places=9, msg=f"class {c} Dice != 1.0")


class TestValidateReconstructionGeometry(unittest.TestCase):
    """Step 5: per-case geometry-validation record."""

    def setUp(self):
        _t2, _mask, maskp, meta = _synthetic_case()
        self.meta = meta
        self.source_geom = meta.original_geometry
        slices = [maskp[:, :, k] for k in range(maskp.shape[2])]
        indices = list(range(maskp.shape[2]))
        self.recon_nii = reconstruct_case_to_original_space(
            slices, indices, meta, meta.original_geometry)

    def test_matching_reconstruction_passes_all(self):
        v = validate_reconstruction_geometry(self.recon_nii, self.source_geom, valid_labels=(0, 1, 2))
        self.assertTrue(v["all_ok"])
        self.assertTrue(v["shape_match"])
        self.assertTrue(v["affine_match"])
        self.assertTrue(v["spacing_match"])
        self.assertTrue(v["orientation_match"])
        self.assertTrue(v["depth_match"])
        self.assertTrue(v["labels_valid"])
        self.assertEqual(v["labels_found"], [0, 1, 2])

    def test_shape_mismatch_flagged(self):
        wrong = nib.Nifti1Image(
            np.zeros((self.source_geom.shape[0] + 2, *self.source_geom.shape[1:]), dtype=np.uint8),
            affine=self.source_geom.affine)
        v = validate_reconstruction_geometry(wrong, self.source_geom)
        self.assertFalse(v["shape_match"])
        self.assertFalse(v["all_ok"])

    def test_invalid_label_flagged(self):
        arr = np.asarray(self.recon_nii.dataobj).copy()
        arr[0, 0, 0] = 3  # illegal label
        bad = nib.Nifti1Image(arr.astype(np.uint8), affine=self.source_geom.affine)
        bad.header.set_zooms(self.source_geom.zooms)
        v = validate_reconstruction_geometry(bad, self.source_geom, valid_labels=(0, 1, 2))
        self.assertFalse(v["labels_valid"])
        self.assertIn(3, v["labels_found"])
        self.assertFalse(v["all_ok"])

    def test_affine_mismatch_flagged(self):
        bad_affine = np.asarray(self.source_geom.affine).copy()
        bad_affine[0, 3] += 5.0
        moved = nib.Nifti1Image(np.asarray(self.recon_nii.dataobj), affine=bad_affine)
        v = validate_reconstruction_geometry(moved, self.source_geom)
        self.assertFalse(v["affine_match"])
        self.assertGreater(v["affine_max_abs_diff"], 1.0)


class TestNiftiSaveReloadPreservesGeometry(unittest.TestCase):
    """Step 4: the saved reconstruction reloads with identical shape/affine."""

    def test_save_reload(self):
        _t2, _mask, maskp, meta = _synthetic_case()
        slices = [maskp[:, :, k] for k in range(maskp.shape[2])]
        recon_nii = reconstruct_case_to_original_space(
            slices, list(range(maskp.shape[2])), meta, meta.original_geometry)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "pred_synthetic.nii.gz")
            nib.save(recon_nii, path)
            self.assertTrue(os.path.exists(path))
            reloaded = read_geometry(nib.load(path))
            self.assertEqual(reloaded.shape, meta.original_geometry.shape)
            self.assertTrue(np.allclose(reloaded.affine, meta.original_geometry.affine, atol=1e-4))


class TestRepresentativeSelection(unittest.TestCase):
    """Step 8: objective good/median/challenging selection."""

    @staticmethod
    def _case(pid, cg, pz):
        return CaseMetrics(patient_id=pid, per_class={
            0: {"dice": 1.0}, 1: {"dice": cg}, 2: {"dice": pz},
        })

    def test_good_median_challenging(self):
        from run_rq2_reconstruction import mean_foreground_dice, select_representative_cases

        cases = [
            self._case("001", 0.90, 0.80),  # mean 0.85 -> good
            self._case("002", 0.70, 0.60),  # mean 0.65 -> median
            self._case("003", 0.40, 0.30),  # mean 0.35 -> challenging
        ]
        self.assertAlmostEqual(mean_foreground_dice(cases[0]), 0.85, places=6)
        reps = select_representative_cases(cases)
        self.assertEqual(reps["good"], "001")
        self.assertEqual(reps["median"], "002")
        self.assertEqual(reps["challenging"], "003")

    def test_deterministic_tie_break(self):
        from run_rq2_reconstruction import select_representative_cases

        cases = [self._case("020", 0.5, 0.5), self._case("010", 0.5, 0.5)]
        reps = select_representative_cases(cases)
        # Lowest rank tie-broken by id -> '010' is challenging, '020' is good.
        self.assertEqual(reps["challenging"], "010")
        self.assertEqual(reps["good"], "020")


class TestReconstructionValidationCsv(unittest.TestCase):
    def test_writes_expected_columns(self):
        from run_rq2_reconstruction import write_reconstruction_validation_csv

        rows = [{
            "patient_id": "001", "all_ok": True, "shape_match": True,
            "recon_shape": (40, 44, 6), "source_shape": (40, 44, 6), "depth_match": True,
            "affine_match": True, "affine_max_abs_diff": 0.0, "spacing_match": True,
            "recon_spacing_mm": (0.5, 0.5, 3.0), "source_spacing_mm": (0.5, 0.5, 3.0),
            "orientation_match": True, "recon_orientation": "RAS", "source_orientation": "RAS",
            "labels_found": [0, 1, 2], "labels_valid": True,
        }]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reconstruction_validation.csv")
            write_reconstruction_validation_csv(path, rows)
            import csv as _csv
            with open(path) as f:
                read = list(_csv.DictReader(f))
            self.assertEqual(len(read), 1)
            self.assertEqual(read[0]["patient_id"], "001")
            self.assertEqual(read[0]["all_ok"], "True")


class TestRender3DProjectionsSmoke(unittest.TestCase):
    def test_writes_png(self):
        from run_rq2_reconstruction import render_3d_projections

        h, w, d = 32, 36, 5
        t2 = np.random.default_rng(1).random((h, w, d)).astype(np.float32)
        gt = np.zeros((h, w, d), dtype=np.int64)
        gt[12:20, 14:22, 1:4] = 1
        gt[8:24, 10:26, 1:4][gt[8:24, 10:26, 1:4] == 0] = 2
        pred = gt.copy()
        pred[12:16, 14:18, 1:3] = 2  # small disagreement
        with tempfile.TemporaryDirectory() as dd:
            out = os.path.join(dd, "proj.png")
            render_3d_projections(t2, gt, pred, "001", out)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)


if __name__ == "__main__":
    unittest.main()
