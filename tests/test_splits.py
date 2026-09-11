"""Unit tests for src/splits.py -- the official Prostate158 case-level split.

Gate 6: 119 train / 20 validation, case-level, no leakage, split before slicing.
The 19-case test archive is expected to be ABSENT in this environment; that
must be reported as a clear warning, not silently ignored or fabricated.
"""

import os
import sys
import unittest
import warnings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.splits import assert_no_leakage, load_official_split

DATASET_DIR = os.path.join(project_root, "dataset", "prostate158_train")
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VALID_CSV = os.path.join(DATASET_DIR, "valid.csv")


class TestOfficialSplit(unittest.TestCase):
    def setUp(self):
        if not (os.path.exists(TRAIN_CSV) and os.path.exists(VALID_CSV)):
            self.skipTest("Official split CSVs not found; skipping integration test.")

    def test_official_counts(self):
        split = load_official_split(TRAIN_CSV, VALID_CSV)
        self.assertEqual(len(split.train_ids), 119)
        self.assertEqual(len(split.val_ids), 20)

    def test_no_leakage_between_train_and_val(self):
        split = load_official_split(TRAIN_CSV, VALID_CSV)
        self.assertEqual(set(split.train_ids) & set(split.val_ids), set())

    def test_test_set_absent_warns_clearly(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            split = load_official_split(TRAIN_CSV, VALID_CSV)
            self.assertIsNone(split.test_ids)
            self.assertTrue(
                any("19-case" in str(w.message) for w in caught),
                "Expected a clear warning about the missing 19-case test archive.",
            )

    def test_ids_are_zero_padded_and_match_dataset_folders(self):
        split = load_official_split(TRAIN_CSV, VALID_CSV)
        for pid in split.train_ids[:5]:
            self.assertEqual(len(pid), 3)
            self.assertTrue(pid.isdigit())

    def test_assert_no_leakage_raises_on_overlap(self):
        with self.assertRaises(ValueError):
            assert_no_leakage(["020", "021"], ["021", "022"])

    def test_assert_no_leakage_raises_on_test_overlap(self):
        with self.assertRaises(ValueError):
            assert_no_leakage(["020", "021"], ["022"], ["021", "023"])

    def test_split_ids_correspond_to_real_directories(self):
        split = load_official_split(TRAIN_CSV, VALID_CSV)
        train_dir = os.path.join(DATASET_DIR, "train")
        for pid in split.train_ids + split.val_ids:
            self.assertTrue(
                os.path.isdir(os.path.join(train_dir, pid)),
                f"Split references case {pid} with no matching directory.",
            )


if __name__ == "__main__":
    unittest.main()
