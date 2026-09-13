"""Unit tests for scripts/audit_test_archive.py helpers, on synthetic NIfTI
cases (no real archive / no network / no model). Validates the audit tool
before it is used in Colab against the real Drive archive.
"""

import os
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
scripts_dir = os.path.join(project_root, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import audit_test_archive as audit


def _write_nii(path, shape=(8, 8, 3), labels=False):
    affine = np.diag([0.5, 0.5, 3.0, 1.0])
    if labels:
        data = np.zeros(shape, dtype=np.float32)
        data[2:5, 2:5, :] = 1
        data[5:7, 5:7, :] = 2
    else:
        data = np.random.default_rng(0).uniform(0, 500, size=shape).astype(np.float32)
    nib.save(nib.Nifti1Image(data, affine), path)


def _make_case(dir_path, with_gt=True):
    os.makedirs(dir_path, exist_ok=True)
    _write_nii(os.path.join(dir_path, audit.T2_NAME))
    if with_gt:
        _write_nii(os.path.join(dir_path, audit.GT_NAME), labels=True)


class TestNormalizeId(unittest.TestCase):
    def test_zero_pads_numeric(self):
        self.assertEqual(audit._normalize_id("5"), "005")
        self.assertEqual(audit._normalize_id("160"), "160")


class TestFindCaseRoot(unittest.TestCase):
    def test_direct_case_root(self):
        with tempfile.TemporaryDirectory() as d:
            _make_case(os.path.join(d, "160"))
            _make_case(os.path.join(d, "161"))
            self.assertEqual(audit.find_case_root(d), d)

    def test_descends_single_wrapper(self):
        with tempfile.TemporaryDirectory() as d:
            wrapper = os.path.join(d, "prostate158_test")
            _make_case(os.path.join(wrapper, "160"))
            _make_case(os.path.join(wrapper, "161"))
            self.assertEqual(audit.find_case_root(d), wrapper)


class TestInspectCase(unittest.TestCase):
    def test_case_with_t2_and_gt(self):
        with tempfile.TemporaryDirectory() as d:
            case = os.path.join(d, "160")
            _make_case(case, with_gt=True)
            info = audit.inspect_case(case)
            self.assertTrue(info["has_t2"])
            self.assertTrue(info["has_gt"])
            self.assertEqual(info["gt_unique_labels"], [0, 1, 2])
            self.assertTrue(info["gt_matches_t2_geometry"])
            self.assertEqual(info["t2_shape"], (8, 8, 3))

    def test_case_with_t2_only(self):
        with tempfile.TemporaryDirectory() as d:
            case = os.path.join(d, "160")
            _make_case(case, with_gt=False)
            info = audit.inspect_case(case)
            self.assertTrue(info["has_t2"])
            self.assertFalse(info["has_gt"])
            self.assertNotIn("gt_unique_labels", info)


class TestExtractSkip(unittest.TestCase):
    def test_no_zip_and_existing_dir_is_ok(self):
        with tempfile.TemporaryDirectory() as d:
            audit.extract_if_needed(None, d)  # should not raise

    def test_no_zip_missing_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            audit.extract_if_needed(None, "/no/such/extract/dir")


if __name__ == "__main__":
    unittest.main()
