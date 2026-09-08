"""Unit tests for dataset scaffolding and slice indexing."""

import os
import sys
import unittest

# Ensure project root is in sys.path for robust imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code.dataset import Prostate2DDataset


class TestDataset(unittest.TestCase):
    """Test suite for Prostate2DDataset interface."""

    def setUp(self):
        # Resolve real dataset location
        self.project_root = project_root
        self.dataset_root = os.path.join(
            self.project_root, "dataset", "prostate158_train", "train"
        )

    def test_dataset_initialization_patient_020(self):
        """Test dataset initialization and slice indexing on Patient 020."""
        if not os.path.exists(os.path.join(self.dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        dataset = Prostate2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            modalities=["t2", "adc", "dwi"],
            mask_name="t2_tumor_reader1.nii.gz",
            slice_sampling="all",
        )

        # Patient 020 has 24 slices
        self.assertEqual(len(dataset), 24)

    def test_dataset_getitem_structure(self):
        """Test item structure returned by __getitem__."""
        if not os.path.exists(os.path.join(self.dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        dataset = Prostate2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            modalities=["t2", "adc", "dwi"],
            mask_name="t2_tumor_reader1.nii.gz",
        )

        sample = dataset[7]
        self.assertIn("image", sample)
        self.assertIn("mask", sample)
        self.assertIn("patient_id", sample)
        self.assertIn("slice_idx", sample)

        self.assertEqual(sample["patient_id"], "020")
        self.assertEqual(sample["slice_idx"], 7)
        self.assertEqual(sample["image"].shape, (3, 270, 270))
        self.assertEqual(sample["mask"].shape, (1, 270, 270))

    def test_dataset_out_of_bounds_index(self):
        """Test that indexing beyond dataset size raises IndexError."""
        dataset = Prostate2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
        )
        with self.assertRaises(IndexError):
            _ = dataset[999]


if __name__ == "__main__":
    unittest.main()

