"""****QUARANTINED / RETIRED**** -- tests for the superseded lesion/multimodal
dataset (Prostate2DDataset), from the retired research direction.

These are kept only so the prior implementation state stays documented; they
do NOT exercise the current thesis pipeline. Current thesis tests for
ProstateZonal2DDataset live in tests/test_dataset.py.
"""

import os
import sys
import unittest
import warnings
import numpy as np

# Ensure project root is in sys.path for robust imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import Prostate2DDataset


class TestDataset(unittest.TestCase):
    """Test suite for Prostate2DDataset interface and preprocessing integration."""

    def setUp(self):
        # Resolve real dataset location
        self.project_root = project_root
        self.dataset_root = os.path.join(
            self.project_root, "dataset", "prostate158_train", "train"
        )
        self.has_patient_020 = os.path.exists(os.path.join(self.dataset_root, "020"))

    def test_dataset_initialization_patient_020_all(self):
        """Test dataset initialization and slice indexing on Patient 020 with slice_sampling='all'."""
        if not self.has_patient_020:
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        dataset = Prostate2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            modalities=["t2", "adc", "dwi"],
            mask_name="t2_tumor_reader1.nii.gz",
            slice_sampling="all",
        )

        # Patient 020 has 24 slices dynamically retrieved from volume.shape[2]
        self.assertEqual(len(dataset), 24)

        # Check sample 0 properties
        sample = dataset[0]
        self.assertIn("image", sample)
        self.assertIn("mask", sample)
        self.assertIn("patient_id", sample)
        self.assertIn("slice_idx", sample)

        self.assertEqual(sample["patient_id"], "020")
        self.assertEqual(sample["slice_idx"], 0)
        self.assertEqual(sample["image"].shape, (3, 270, 270))
        self.assertEqual(sample["mask"].shape, (1, 270, 270))
        self.assertEqual(sample["image"].dtype, np.float32)
        self.assertEqual(sample["mask"].dtype, np.float32)

    def test_dataset_tumor_slices_patient_020(self):
        """Verify tumor slice indices for Patient 020 (expected [5, 6, 7])."""
        if not self.has_patient_020:
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        dataset = Prostate2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            slice_sampling="all",
        )

        tumor_slices = [
            i for i in range(len(dataset))
            if np.any(dataset[i]["mask"] > 0)
        ]
        self.assertEqual(tumor_slices, [5, 6, 7])

        # Verify binary values in mask
        unique_vals_slice7 = np.unique(dataset[7]["mask"])
        self.assertTrue(np.array_equal(unique_vals_slice7, [0.0, 1.0]))

        unique_vals_slice0 = np.unique(dataset[0]["mask"])
        self.assertTrue(np.array_equal(unique_vals_slice0, [0.0]))

    def test_dataset_slice_sampling_tumor_only(self):
        """Test slice_sampling='tumor_only' extracts only positive tumor slices."""
        if not self.has_patient_020:
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        dataset = Prostate2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            slice_sampling="tumor_only",
        )

        # Patient 020 has 3 positive tumor slices: 5, 6, 7
        self.assertEqual(len(dataset), 3)
        extracted_slice_indices = [dataset[i]["slice_idx"] for i in range(len(dataset))]
        self.assertEqual(extracted_slice_indices, [5, 6, 7])

    def test_dataset_slice_sampling_prostate_only(self):
        """Test slice_sampling='prostate_only' uses t2_anatomy_reader1."""
        if not self.has_patient_020:
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        dataset = Prostate2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            slice_sampling="prostate_only",
        )

        # Verified earlier: prostate anatomy covers slices 1 through 19 (19 slices)
        self.assertEqual(len(dataset), 19)

    def test_dataset_invalid_sampling_raises_error(self):
        """Test that invalid slice_sampling raises ValueError."""
        with self.assertRaises(ValueError):
            Prostate2DDataset(
                dataset_root=self.dataset_root,
                patient_ids=["020"],
                slice_sampling="invalid_mode",
            )

    def test_dataset_missing_folder_raises_error(self):
        """Test that missing patient folder raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            Prostate2DDataset(
                dataset_root=self.dataset_root,
                patient_ids=["nonexistent_patient_999"],
            )

    def test_dataset_out_of_bounds_index(self):
        """Test that indexing beyond dataset size raises IndexError."""
        if not self.has_patient_020:
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        dataset = Prostate2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
        )
        with self.assertRaises(IndexError):
            _ = dataset[999]

    def test_dataloader_compatibility(self):
        """Test compatibility with PyTorch DataLoader if torch is available."""
        if not self.has_patient_020:
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        try:
            from torch.utils.data import DataLoader
        except ImportError:
            self.skipTest("PyTorch not installed in this environment; skipping DataLoader test.")

        dataset = Prostate2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            slice_sampling="all",
        )

        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        images = batch["image"]
        masks = batch["mask"]

        # Expected batch shapes: (B=2, C=3, H=270, W=270) and (B=2, C=1, H=270, W=270)
        self.assertEqual(tuple(images.shape), (2, 3, 270, 270))
        self.assertEqual(tuple(masks.shape), (2, 1, 270, 270))


if __name__ == "__main__":
    unittest.main()
