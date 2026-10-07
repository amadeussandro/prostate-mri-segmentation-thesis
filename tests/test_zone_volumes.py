"""RQ2 zone-volume tests -- torch-free, no dataset or checkpoint needed.

Volumes in mL are the clinical output of RQ2, so the arithmetic that produces
them has to be pinned: a millilitre figure is a voxel count multiplied by the
affine determinant, and both halves can go wrong silently.
"""

import json
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

from run_zone_volumes import run, volume_ml, zone_volumes


def _affine(sx=0.5, sy=0.5, sz=3.0):
    a = np.diag([-sx, -sy, sz, 1.0]).astype(np.float64)
    a[:3, 3] = [40.0, 10.0, -50.0]
    return a


def _case(shape=(8, 8, 4), n_cg=16, n_pz=8):
    vol = np.zeros(shape, dtype=np.uint8)
    flat = vol.reshape(-1)
    flat[:n_cg] = 1
    flat[n_cg:n_cg + n_pz] = 2
    return flat.reshape(shape)


class TestVolumeArithmetic(unittest.TestCase):
    def test_volume_is_count_times_voxel_volume(self):
        vol = _case(n_cg=16, n_pz=8)
        self.assertAlmostEqual(volume_ml(vol, _affine(), 1), 16 * 0.5 * 0.5 * 3.0 / 1000.0, places=9)
        self.assertAlmostEqual(volume_ml(vol, _affine(), 2), 8 * 0.5 * 0.5 * 3.0 / 1000.0, places=9)

    def test_identity_affine_inflates_the_volume(self):
        vol = _case()
        self.assertGreater(volume_ml(vol, np.eye(4), 1), volume_ml(vol, _affine(), 1))

    def test_rotation_preserves_volume_but_zooms_would_not(self):
        # A rotated affine has the same determinant, so the volume is unchanged;
        # reading the header zooms instead would be wrong here.
        th = np.pi / 6
        rot = np.eye(4)
        rot[:3, :3] = np.array([[np.cos(th), -np.sin(th), 0],
                                [np.sin(th), np.cos(th), 0],
                                [0, 0, 1.0]]) @ _affine()[:3, :3]
        vol = _case()
        self.assertAlmostEqual(volume_ml(vol, rot, 1), volume_ml(vol, _affine(), 1), places=9)

    def test_absent_class_is_zero(self):
        self.assertEqual(volume_ml(np.zeros((4, 4, 4), dtype=np.uint8), _affine(), 1), 0.0)

    def test_whole_gland_and_fraction(self):
        z = zone_volumes(_case(n_cg=16, n_pz=8), _affine())
        self.assertAlmostEqual(z["whole_gland_ml"], z["cg_ml"] + z["pz_ml"], places=12)
        self.assertAlmostEqual(z["pz_fraction"], 8 / 24, places=9)

    def test_fraction_is_nan_for_an_empty_gland(self):
        z = zone_volumes(np.zeros((4, 4, 4), dtype=np.uint8), _affine())
        self.assertTrue(np.isnan(z["pz_fraction"]))


class TestRunModes(unittest.TestCase):
    def _write_case(self, d, pid, vol, affine, name="pred"):
        os.makedirs(d, exist_ok=True)
        nib.save(nib.Nifti1Image(vol, affine), os.path.join(d, f"{name}_{pid}.nii.gz"))

    def test_prediction_only_mode_reports_no_error_columns(self):
        with tempfile.TemporaryDirectory() as td:
            recon = os.path.join(td, "recon")
            for pid in ("001", "002"):
                self._write_case(recon, pid, _case(), _affine())
            res = run(recon, os.path.join(td, "out"))
            self.assertEqual(res["n_cases"], 2)
            self.assertEqual(res["payload"]["mode"], "prediction_only")
            self.assertEqual(res["payload"]["agreement"], {})
            head = open(res["per_case"], encoding="utf-8").readline()
            self.assertNotIn("error_", head)

    def test_ground_truth_mode_reports_bias_and_limits(self):
        with tempfile.TemporaryDirectory() as td:
            recon, gt_root = os.path.join(td, "recon"), os.path.join(td, "gt")
            for pid in ("001", "002", "003"):
                self._write_case(recon, pid, _case(n_cg=20, n_pz=8), _affine())
                os.makedirs(os.path.join(gt_root, pid), exist_ok=True)
                nib.save(nib.Nifti1Image(_case(n_cg=16, n_pz=8), _affine()),
                         os.path.join(gt_root, pid, "t2_anatomy_reader1.nii.gz"))
            res = run(recon, os.path.join(td, "out"), gt_root=gt_root)
            self.assertEqual(res["payload"]["mode"], "prediction_vs_ground_truth")
            bias = res["payload"]["agreement"]["cg_ml"]["bias_ml"]
            # 4 extra CG voxels per case, consistently over-estimated.
            self.assertAlmostEqual(bias, 4 * 0.5 * 0.5 * 3.0 / 1000.0, places=9)
            self.assertGreater(bias, 0.0)

    def test_missing_ground_truth_refuses_a_partial_cohort(self):
        with tempfile.TemporaryDirectory() as td:
            recon, gt_root = os.path.join(td, "recon"), os.path.join(td, "gt")
            for pid in ("001", "002"):
                self._write_case(recon, pid, _case(), _affine())
            os.makedirs(os.path.join(gt_root, "001"), exist_ok=True)
            nib.save(nib.Nifti1Image(_case(), _affine()),
                     os.path.join(gt_root, "001", "t2_anatomy_reader1.nii.gz"))
            with self.assertRaises(FileNotFoundError):
                run(recon, os.path.join(td, "out"), gt_root=gt_root)

    def test_shape_mismatch_against_ground_truth_raises(self):
        with tempfile.TemporaryDirectory() as td:
            recon, gt_root = os.path.join(td, "recon"), os.path.join(td, "gt")
            self._write_case(recon, "001", _case((8, 8, 4)), _affine())
            os.makedirs(os.path.join(gt_root, "001"), exist_ok=True)
            nib.save(nib.Nifti1Image(_case((8, 8, 5)), _affine()),
                     os.path.join(gt_root, "001", "t2_anatomy_reader1.nii.gz"))
            with self.assertRaises(ValueError):
                run(recon, os.path.join(td, "out"), gt_root=gt_root)

    def test_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "recon"))
            with self.assertRaises(FileNotFoundError):
                run(os.path.join(td, "recon"), os.path.join(td, "out"))


if __name__ == "__main__":
    unittest.main()
