"""Regression tests for the Experiment-2 evaluation fairness fix.

The primary (original-voxel-space) evaluation protocol must score every
preprocessing arm's prediction against the IDENTICAL, untouched on-disk ground
truth -- never against the preprocessed mask inverse-transformed back. If the
GT were round-tripped through preprocessing, the lossy Exp02-C `resize` arm
would degrade prediction and GT identically, cancelling preprocessing error
and inflating that arm's Dice -- an unfair advantage in the RQ-1 ablation.

These tests assert:
  1. crop_pad (Exp01/A/B baseline): the preprocessed mask round-trips EXACTLY
     to the on-disk mask -- so switching to untouched GT leaves Experiment-1
     numbers unchanged. (Also the reconstruction invariant the round-trip gate
     depends on.)
  2. resize (Exp02-C): the preprocessed mask does NOT round-trip exactly -- so
     round-tripped GT would have been optimistic; the fix matters here.
  3. predict_case_original_space returns, as GT, the untouched on-disk mask
     bit-for-bit under BOTH crop_pad and resize -- i.e. the evaluated GT is
     independent of the preprocessing arm.
"""

import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


DATASET_ROOT = os.path.join(project_root, "dataset", "prostate158_train", "train")
PID = "020"


def _has_case() -> bool:
    return os.path.isdir(os.path.join(DATASET_ROOT, PID))


class TestOriginalSpaceGroundTruth(unittest.TestCase):
    def setUp(self):
        if not _has_case():
            self.skipTest(f"Case {PID} folder not found; skipping integration test.")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping integration test.")

    def _untouched_gt(self):
        import nibabel as nib
        import numpy as np

        mask_path = os.path.join(DATASET_ROOT, PID, "t2_anatomy_reader1.nii.gz")
        return np.round(nib.load(mask_path).get_fdata()).astype(np.int64)

    def _roundtrip_gt(self, spatial_mode, target_size):
        """Preprocess the GT and invert it straight back (no model)."""
        import nibabel as nib
        import numpy as np

        from src.transforms import (
            PreprocessingConfig,
            invert_preprocessing_mask,
            preprocess_case,
        )

        t2 = nib.load(os.path.join(DATASET_ROOT, PID, "t2.nii.gz"))
        mk = nib.load(os.path.join(DATASET_ROOT, PID, "t2_anatomy_reader1.nii.gz"))
        cfg = PreprocessingConfig(
            normalization="per_volume",
            z_resample=False,
            spatial_mode=spatial_mode,
            target_size=target_size,
        )
        _, proc_mask, meta = preprocess_case(t2, mk, cfg)
        return invert_preprocessing_mask(proc_mask, meta)

    def test_crop_pad_roundtrip_gt_is_bit_identical(self):
        """crop_pad is lossless: round-tripped GT == untouched GT exactly, so the
        untouched-GT fix does not change Experiment-1 (crop_pad) results."""
        import numpy as np

        untouched = self._untouched_gt()
        roundtripped = self._roundtrip_gt("crop_pad", (442, 442))
        self.assertTrue(
            np.array_equal(roundtripped, untouched),
            "crop_pad GT round-trip must be bit-identical to the on-disk mask.",
        )

    def test_resize_roundtrip_gt_is_lossy(self):
        """resize is deliberately lossy: round-tripped GT differs from the
        untouched GT -- this is exactly why the GT must NOT be round-tripped for
        the resize arm (it would otherwise be scored optimistically)."""
        import numpy as np

        untouched = self._untouched_gt()
        roundtripped = self._roundtrip_gt("resize", (256, 256))
        self.assertFalse(
            np.array_equal(roundtripped, untouched),
            "resize GT round-trip is expected to be lossy on this cohort.",
        )

    def test_predicted_case_gt_is_untouched_under_both_arms(self):
        """predict_case_original_space must return the untouched on-disk mask as
        GT regardless of the preprocessing arm -- so all arms are compared to the
        same ground truth."""
        import numpy as np

        from src.dataset import ProstateZonal2DDataset
        from src.evaluate import predict_case_original_space
        from src.model import ProstateUNet2D
        from src.transforms import PreprocessingConfig

        untouched = self._untouched_gt()
        model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=8)

        for spatial_mode, target_size in (("crop_pad", (442, 442)), ("resize", (256, 256))):
            with self.subTest(spatial_mode=spatial_mode):
                dataset = ProstateZonal2DDataset(
                    dataset_root=DATASET_ROOT,
                    patient_ids=[PID],
                    slice_sampling="all",
                    preprocessing_config=PreprocessingConfig(
                        normalization="per_volume",
                        z_resample=False,
                        spatial_mode=spatial_mode,
                        target_size=target_size,
                    ),
                )
                _, gt_original, geometry = predict_case_original_space(
                    PID, model, dataset, device="cpu"
                )
                self.assertEqual(tuple(gt_original.shape), tuple(geometry.shape))
                self.assertTrue(
                    np.array_equal(gt_original.astype(np.int64), untouched),
                    f"GT for arm '{spatial_mode}' must equal the untouched on-disk mask.",
                )


if __name__ == "__main__":
    unittest.main()
