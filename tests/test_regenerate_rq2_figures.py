"""Tests for redrawing the RQ2 figures without re-running inference.

The point of this script is that regenerating three pictures should not require
loading a checkpoint. These tests pin that it reads the saved predictions, that
it picks the same representative cases the report names, and that it refuses to
draw an overlay on volumes that do not share a grid.
"""

import csv
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

from regenerate_rq2_figures import regenerate_case, representative_cases, run

METRIC_FIELDS = ["patient_id", "class_id", "class_name", "dice", "iou",
                 "hd95_mm", "asd_mm", "precision", "recall"]


def _write_metrics(path, per_case_dice):
    """per_case_dice: {pid: (cg_dice, pz_dice)}"""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=METRIC_FIELDS)
        w.writeheader()
        for pid, (cg, pz) in per_case_dice.items():
            for cid, d in ((0, 0.99), (1, cg), (2, pz)):
                w.writerow({"patient_id": pid, "class_id": cid,
                            "class_name": f"c{cid}", "dice": d, "iou": d,
                            "hd95_mm": 1.0, "asd_mm": 1.0,
                            "precision": d, "recall": d})


def _affine(sz=3.0):
    a = np.diag([-0.46875, -0.46875, sz, 1.0]).astype(np.float64)
    a[:3, 3] = [50.0, 8.0, -52.0]
    return a


class TestRepresentativeSelection(unittest.TestCase):
    def test_picks_highest_median_and_lowest_mean_foreground_dice(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "m.csv")
            _write_metrics(p, {"001": (0.90, 0.90),   # mean 0.90 -> good
                               "002": (0.70, 0.70),   # mean 0.70 -> median
                               "003": (0.20, 0.20)})  # mean 0.20 -> challenging
            self.assertEqual(representative_cases(p),
                             {"challenging": "003", "median": "002", "good": "001"})

    def test_background_class_is_excluded_from_the_ranking(self):
        # Background Dice is ~0.99 for every case; if it leaked into the mean the
        # ranking would compress and could reorder.
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "m.csv")
            _write_metrics(p, {"001": (0.50, 0.50), "002": (0.80, 0.80), "003": (0.60, 0.60)})
            self.assertEqual(representative_cases(p)["challenging"], "001")
            self.assertEqual(representative_cases(p)["good"], "002")

    def test_empty_metrics_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "m.csv")
            _write_metrics(p, {})
            with self.assertRaises(ValueError):
                representative_cases(p)


class _Fixture:
    """A minimal rq2-dir plus a matching dataset root."""

    def __init__(self, td, shape=(10, 10, 4), gt_shape=None):
        self.rq2 = os.path.join(td, "rq2")
        self.gt_root = os.path.join(td, "data")
        os.makedirs(os.path.join(self.rq2, "reconstructions_3d"), exist_ok=True)
        _write_metrics(os.path.join(self.rq2, "metrics_per_case.csv"),
                       {"001": (0.9, 0.9), "002": (0.7, 0.7), "003": (0.2, 0.2)})
        for pid in ("001", "002", "003"):
            pred = np.zeros(shape, dtype=np.uint8)
            pred[2:5, 2:5, 1:3] = 1
            pred[5:7, 5:7, 1:3] = 2
            nib.save(nib.Nifti1Image(pred, _affine()),
                     os.path.join(self.rq2, "reconstructions_3d", f"pred_{pid}.nii.gz"))
            d = os.path.join(self.gt_root, pid)
            os.makedirs(d, exist_ok=True)
            gshape = gt_shape or shape
            nib.save(nib.Nifti1Image(np.random.default_rng(0).random(gshape).astype(np.float32),
                                     _affine()), os.path.join(d, "t2.nii.gz"))
            gt = np.zeros(gshape, dtype=np.uint8)
            gt[2:5, 2:5, 1:3] = 1
            gt[5:7, 5:7, 1:3] = 2
            nib.save(nib.Nifti1Image(gt, _affine()), os.path.join(d, "anat.nii.gz"))


class TestRegeneration(unittest.TestCase):
    def test_writes_figures_without_any_model(self):
        with tempfile.TemporaryDirectory() as td:
            f = _Fixture(td)
            info = regenerate_case("001", f.rq2, f.gt_root, "anat.nii.gz", n_slices=2)
            self.assertTrue(os.path.exists(info["projection_figure"]))
            self.assertGreater(len(info["axial_figures"]), 0)
            for p in info["axial_figures"]:
                self.assertGreater(os.path.getsize(p), 0)

    def test_spacing_is_read_from_the_saved_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            f = _Fixture(td)
            info = regenerate_case("001", f.rq2, f.gt_root, "anat.nii.gz", n_slices=1)
            self.assertAlmostEqual(info["spacing_mm"][0], 0.46875, places=5)
            self.assertAlmostEqual(info["spacing_mm"][2], 3.0, places=5)

    def test_mismatched_grids_are_refused(self):
        with tempfile.TemporaryDirectory() as td:
            f = _Fixture(td, shape=(10, 10, 4), gt_shape=(10, 10, 5))
            with self.assertRaises(ValueError) as ctx:
                regenerate_case("001", f.rq2, f.gt_root, "anat.nii.gz", n_slices=1)
            self.assertIn("mismatch", str(ctx.exception).lower())

    def test_missing_input_is_named(self):
        with tempfile.TemporaryDirectory() as td:
            f = _Fixture(td)
            os.remove(os.path.join(f.gt_root, "001", "t2.nii.gz"))
            with self.assertRaises(FileNotFoundError):
                regenerate_case("001", f.rq2, f.gt_root, "anat.nii.gz", n_slices=1)

    def test_run_defaults_to_the_reports_representatives(self):
        with tempfile.TemporaryDirectory() as td:
            f = _Fixture(td)
            res = run(f.rq2, f.gt_root, "anat.nii.gz", n_slices=1)
            self.assertEqual(res["representatives"],
                             {"challenging": "003", "median": "002", "good": "001"})
            self.assertEqual(len(res["written"]), 3)

    def test_explicit_cases_override_the_rule(self):
        with tempfile.TemporaryDirectory() as td:
            f = _Fixture(td)
            res = run(f.rq2, f.gt_root, "anat.nii.gz", cases=["002"], n_slices=1)
            self.assertEqual(list(res["representatives"].values()), ["002"])


if __name__ == "__main__":
    unittest.main()
