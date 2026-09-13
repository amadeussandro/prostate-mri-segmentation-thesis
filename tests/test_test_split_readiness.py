"""Readiness tests for the official held-out TEST split.

Covers the minimal code change that lets the test split use a SEPARATE,
isolated dataset root (the extracted test archive) and can never silently
fall back to the train/validation data. No model, no training, no metrics --
pure loader/split-resolution logic on synthetic directories + the real
train.csv/valid.csv.
"""

import os
import sys
import tempfile
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.evaluate import _resolve_split_dataset_root
from src.splits import load_official_split

DATASET_DIR = os.path.join(project_root, "dataset", "prostate158_train")
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VALID_CSV = os.path.join(DATASET_DIR, "valid.csv")


class TestSplitDatasetRootResolution(unittest.TestCase):
    def test_train_and_val_use_dataset_root(self):
        data_cfg = {"dataset_root": "/data/train", "test_dir": "/data/test"}
        self.assertEqual(_resolve_split_dataset_root(data_cfg, "train"), "/data/train")
        self.assertEqual(_resolve_split_dataset_root(data_cfg, "val"), "/data/train")
        self.assertEqual(_resolve_split_dataset_root(data_cfg, "validation"), "/data/train")

    def test_test_uses_test_dir(self):
        data_cfg = {"dataset_root": "/data/train", "test_dir": "/data/test/extracted"}
        self.assertEqual(_resolve_split_dataset_root(data_cfg, "test"), "/data/test/extracted")

    def test_test_without_test_dir_raises(self):
        data_cfg = {"dataset_root": "/data/train"}  # no test_dir
        with self.assertRaises(ValueError):
            _resolve_split_dataset_root(data_cfg, "test")

    def test_test_never_returns_train_root(self):
        # Even if someone only sets dataset_root, the test split must NOT resolve
        # to it -- it raises instead, so train/val data can't be read as test.
        data_cfg = {"dataset_root": "/data/train"}
        try:
            root = _resolve_split_dataset_root(data_cfg, "test")
            self.fail(f"expected ValueError, got root={root}")
        except ValueError:
            pass


class TestOfficialSplitWithTestDir(unittest.TestCase):
    def setUp(self):
        if not (os.path.exists(TRAIN_CSV) and os.path.exists(VALID_CSV)):
            self.skipTest("Official split CSVs not found.")

    def _make_test_dir(self, case_ids):
        tmp = tempfile.mkdtemp()
        for cid in case_ids:
            os.makedirs(os.path.join(tmp, cid))
        return tmp

    def test_resolves_19_test_ids_from_dir_without_leakage(self):
        # 19 synthetic case ids that do NOT overlap train/val (020..158).
        test_ids = [str(900 + i) for i in range(19)]
        test_dir = self._make_test_dir(test_ids)
        try:
            split = load_official_split(TRAIN_CSV, VALID_CSV, test_dir=test_dir)
            self.assertEqual(len(split.test_ids), 19)
            self.assertEqual(set(split.test_ids) & set(split.train_ids), set())
            self.assertEqual(set(split.test_ids) & set(split.val_ids), set())
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_overlapping_test_case_is_rejected(self):
        # Include a case id that IS in the training set -> leakage -> must raise.
        import pandas as pd
        train_first = str(int(pd.read_csv(TRAIN_CSV)["ID"].iloc[0])).zfill(3)
        test_ids = [train_first] + [str(900 + i) for i in range(18)]
        test_dir = self._make_test_dir(test_ids)
        try:
            with self.assertRaises(ValueError):
                load_official_split(TRAIN_CSV, VALID_CSV, test_dir=test_dir)
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_wrong_case_count_is_rejected(self):
        test_dir = self._make_test_dir([str(900 + i) for i in range(18)])  # 18, not 19
        try:
            with self.assertRaises(ValueError):
                load_official_split(TRAIN_CSV, VALID_CSV, test_dir=test_dir)
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
