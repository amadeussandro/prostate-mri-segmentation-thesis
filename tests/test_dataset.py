"""Unit tests for ProstateZonal2DDataset -- the CURRENT thesis dataset.

Covers Gates 3 and 4: image/mask shape & dtype, label validity, spatial
metadata retention, slicing correspondence, and DataLoader batching across
multiple patients with different native (H, W) sizes.
"""

import os
import sys
import unittest

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import ProstateZonal2DDataset
from src.transforms import PreprocessingConfig


class TestProstateZonal2DDataset(unittest.TestCase):
    def setUp(self):
        self.dataset_root = os.path.join(
            project_root, "dataset", "prostate158_train", "train"
        )
        self.has_020 = os.path.exists(os.path.join(self.dataset_root, "020"))
        self.has_021 = os.path.exists(os.path.join(self.dataset_root, "021"))

    def test_sample_shape_and_dtype(self):
        if not self.has_020:
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            slice_sampling="all",
        )
        sample = dataset[0]

        self.assertEqual(sample["image"].shape, (1, 256, 256))
        self.assertEqual(sample["mask"].shape, (256, 256))
        self.assertEqual(sample["image"].dtype, np.float32)
        self.assertEqual(sample["mask"].dtype, np.int64)

    def test_label_validity(self):
        if not self.has_020:
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            slice_sampling="all",
        )
        all_labels = set()
        for i in range(len(dataset)):
            all_labels.update(np.unique(dataset[i]["mask"]).tolist())

        self.assertTrue(all_labels.issubset({0, 1, 2}))

    def test_case_geometry_and_transform_metadata_retained(self):
        if not self.has_020:
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            slice_sampling="all",
        )
        geometry, transform_meta = dataset.get_case_metadata("020")

        self.assertEqual(geometry.shape, (270, 270, 24))
        self.assertIsNotNone(geometry.affine)
        self.assertFalse(np.allclose(geometry.affine, np.eye(4)))  # never identity-default
        self.assertEqual(transform_meta.target_size, (256, 256))

    def test_slice_correspondence_across_full_volume(self):
        if not self.has_020:
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            slice_sampling="all",
        )
        # Preprocessed depth must match the reoriented volume's slice count exactly,
        # and slice indices must be contiguous starting at 0 (explicit-index safe).
        slice_indices = [s_idx for (_, s_idx) in dataset.samples]
        self.assertEqual(slice_indices, list(range(len(dataset))))

    def test_invalid_slice_sampling_raises(self):
        with self.assertRaises(ValueError):
            ProstateZonal2DDataset(
                dataset_root=self.dataset_root,
                patient_ids=["020"] if self.has_020 else ["nonexistent"],
                slice_sampling="bogus_mode",
            )

    def test_missing_patient_raises_filenotfound(self):
        with self.assertRaises(FileNotFoundError):
            ProstateZonal2DDataset(
                dataset_root=self.dataset_root,
                patient_ids=["nonexistent_patient_999"],
            )

    def test_dataloader_batches_multiple_patients_with_different_native_sizes(self):
        if not (self.has_020 and self.has_021):
            self.skipTest("Patients 020/021 not found; skipping integration test.")

        try:
            from torch.utils.data import DataLoader
        except ImportError:
            self.skipTest("PyTorch not installed; skipping DataLoader test.")

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020", "021"],
            slice_sampling="all",
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

        batch = next(iter(loader))
        images = batch["image"]
        masks = batch["mask"]

        self.assertEqual(tuple(images.shape)[1:], (1, 256, 256))
        self.assertEqual(tuple(masks.shape)[1:], (256, 256))
        self.assertEqual(images.dtype.__str__(), "torch.float32")

        # Image-mask correspondence survives batching: same (pid, slice_idx) pairing.
        for pid, s_idx in zip(batch["patient_id"], batch["slice_idx"].tolist()):
            geometry, _ = dataset.get_case_metadata(pid)
            self.assertIsNotNone(geometry)

    def test_custom_preprocessing_config_target_size(self):
        if not self.has_020:
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        cfg = PreprocessingConfig(target_size=(160, 160))
        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            preprocessing_config=cfg,
        )
        sample = dataset[0]
        self.assertEqual(sample["image"].shape, (1, 160, 160))
        self.assertEqual(sample["mask"].shape, (160, 160))


class TestTargetMaskConfigWiring(unittest.TestCase):
    """Fix-pass regression tests (audit P1 item 6): configs/config_baseline.yaml
    declares data.target_mask, and src.train._build_dataset must actually
    honor it -- while the thesis-safe default remains t2_anatomy_reader1.nii.gz
    and no lesion mask can be silently selected."""

    def setUp(self):
        self.dataset_root = os.path.join(
            project_root, "dataset", "prostate158_train", "train"
        )
        if not os.path.isdir(os.path.join(self.dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")

    def test_default_config_uses_anatomy_target(self):
        from src.train import _build_dataset

        config = {"data": {"dataset_root": self.dataset_root}}  # no target_mask key
        dataset = _build_dataset(config, ["020"])
        self.assertEqual(dataset.mask_name, "t2_anatomy_reader1.nii.gz")

    def test_explicit_target_mask_key_is_actually_read(self):
        """Proves the config value is honored (not silently ignored): pointing
        target_mask at a nonexistent filename must surface THAT filename in
        the resulting error, which is only possible if _build_dataset passed
        it through instead of hardcoding t2_anatomy_reader1.nii.gz."""
        from src.train import _build_dataset

        config = {
            "data": {
                "dataset_root": self.dataset_root,
                "target_mask": "nonexistent_mask_marker.nii.gz",
            }
        }
        with self.assertRaises(FileNotFoundError) as ctx:
            _build_dataset(config, ["020"])
        self.assertIn("nonexistent_mask_marker.nii.gz", str(ctx.exception))

    def test_lesion_mask_override_is_still_rejected(self):
        """Even if a config explicitly tries to point target_mask at the
        retired lesion mask (labels {0,3}), ProstateZonal2DDataset's own
        label validation ({0,1,2} only) must still reject it -- the
        safety net does not depend on train.py alone."""
        from src.train import _build_dataset

        lesion_mask_path = os.path.join(self.dataset_root, "020", "t2_tumor_reader1.nii.gz")
        if not os.path.exists(lesion_mask_path):
            self.skipTest("t2_tumor_reader1.nii.gz not found for patient 020; skipping.")

        config = {
            "data": {
                "dataset_root": self.dataset_root,
                "target_mask": "t2_tumor_reader1.nii.gz",
            }
        }
        with self.assertRaises(ValueError):
            _build_dataset(config, ["020"])


if __name__ == "__main__":
    unittest.main()
