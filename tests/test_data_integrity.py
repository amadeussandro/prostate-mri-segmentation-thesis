"""Tests for file-level dataset/cache integrity validation (src/data_integrity).

The incident these guard against: a local cache with the right NUMBER of case
directories but missing/empty NIfTI files. Directory counting is insufficient;
these check the actual required files exist and are non-empty.
"""

import os
import sys
import tempfile
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_integrity import (  # noqa: E402
    REQUIRED_CASE_FILES,
    verify_case_files,
    verify_dataset_cache,
)


def _make_case(root, pid, files, size=10):
    d = os.path.join(root, pid)
    os.makedirs(d, exist_ok=True)
    for fname, content_size in files.items():
        with open(os.path.join(d, fname), "wb") as f:
            f.write(b"x" * content_size)


class TestVerifyCaseFiles(unittest.TestCase):
    def test_all_present_is_ok(self):
        with tempfile.TemporaryDirectory() as d:
            for pid in ("020", "021"):
                _make_case(d, pid, {f: 10 for f in REQUIRED_CASE_FILES})
            rep = verify_case_files(d, ["020", "021"])
            self.assertTrue(rep["ok"])
            self.assertEqual(rep["n_ok"], 2)
            self.assertEqual(rep["missing"], {})

    def test_missing_file_detected(self):
        with tempfile.TemporaryDirectory() as d:
            _make_case(d, "020", {"t2.nii.gz": 10})  # anatomy mask missing
            rep = verify_case_files(d, ["020"])
            self.assertFalse(rep["ok"])
            self.assertIn("t2_anatomy_reader1.nii.gz", rep["missing"]["020"])

    def test_empty_file_detected(self):
        """A present-but-empty file (partial copy) must be flagged."""
        with tempfile.TemporaryDirectory() as d:
            _make_case(d, "020", {"t2.nii.gz": 10, "t2_anatomy_reader1.nii.gz": 0})
            rep = verify_case_files(d, ["020"])
            self.assertFalse(rep["ok"])
            self.assertIn("t2_anatomy_reader1.nii.gz", rep["missing"]["020"])

    def test_missing_case_dir_detected(self):
        with tempfile.TemporaryDirectory() as d:
            rep = verify_case_files(d, ["999"])
            self.assertFalse(rep["ok"])
            self.assertIn("999", rep["missing"])


class TestVerifyDatasetCache(unittest.TestCase):
    def _write_split_csv(self, path, ids):
        with open(path, "w", encoding="utf-8") as f:
            f.write("ID\n")
            for i in ids:
                f.write(f"{int(i)}\n")

    def test_complete_cache_ok(self):
        with tempfile.TemporaryDirectory() as root:
            ds = os.path.join(root, "train")
            os.makedirs(ds)
            for pid in ("020", "021", "022"):
                _make_case(ds, pid, {f: 10 for f in REQUIRED_CASE_FILES})
            train_csv = os.path.join(root, "train.csv")
            valid_csv = os.path.join(root, "valid.csv")
            self._write_split_csv(train_csv, ["20", "21"])
            self._write_split_csv(valid_csv, ["22"])
            rep = verify_dataset_cache(ds, train_csv, valid_csv)
            self.assertTrue(rep["ok"], rep["problems"])

    def test_incomplete_cache_reports_missing(self):
        with tempfile.TemporaryDirectory() as root:
            ds = os.path.join(root, "train")
            os.makedirs(ds)
            _make_case(ds, "020", {f: 10 for f in REQUIRED_CASE_FILES})
            _make_case(ds, "021", {"t2.nii.gz": 10})  # anatomy missing
            train_csv = os.path.join(root, "train.csv")
            valid_csv = os.path.join(root, "valid.csv")
            self._write_split_csv(train_csv, ["20"])
            self._write_split_csv(valid_csv, ["21"])
            rep = verify_dataset_cache(ds, train_csv, valid_csv)
            self.assertFalse(rep["ok"])
            self.assertFalse(rep["val"]["ok"])
            self.assertIn("021", rep["val"]["missing"])

    def test_missing_split_csv(self):
        with tempfile.TemporaryDirectory() as root:
            rep = verify_dataset_cache(os.path.join(root, "train"),
                                       os.path.join(root, "train.csv"),
                                       os.path.join(root, "valid.csv"))
            self.assertFalse(rep["ok"])
            self.assertFalse(rep["split_files_present"])


class TestRealDatasetCacheIfPresent(unittest.TestCase):
    def test_real_train_val_cache_is_complete(self):
        root = os.path.join(project_root, "dataset", "prostate158_train")
        ds = os.path.join(root, "train")
        train_csv = os.path.join(root, "train.csv")
        valid_csv = os.path.join(root, "valid.csv")
        if not (os.path.isdir(ds) and os.path.exists(train_csv) and os.path.exists(valid_csv)):
            self.skipTest("Local dataset not present; skipping real-cache check.")
        rep = verify_dataset_cache(ds, train_csv, valid_csv)
        # If the local dataset is present it should be complete for train+val.
        self.assertTrue(rep["ok"], rep["problems"])


if __name__ == "__main__":
    unittest.main()
