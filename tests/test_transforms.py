"""Unit tests for src/transforms.py: preprocessing must be exactly invertible
for masks under the default (no-resample, crop/pad) configuration -- this is
what Gate 8's round-trip Dice=1.0 test depends on.
"""

import os
import sys
import unittest

import nibabel as nib
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.transforms import (
    PreprocessingConfig,
    invert_preprocessing_mask,
    preprocess_case,
)


def _synthetic_case(shape=(37, 41, 6), affine=None, seed=0):
    """A synthetic T2-like volume + a categorical mask with labels {0,1,2}."""
    rng = np.random.default_rng(seed)
    if affine is None:
        # Deliberately non-trivial (LPS-like, mirrors the real on-disk data).
        affine = np.array(
            [
                [-0.4, 0.0, 0.0, 50.0],
                [0.0, -0.4, 0.0, 60.0],
                [0.0, 0.0, 3.0, -20.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
    t2_data = rng.uniform(0, 1000, size=shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.float32)
    h, w, d = shape
    mask[h // 4 : h // 2, w // 4 : w // 2, :] = 1
    mask[h // 2 : 3 * h // 4, w // 2 : 3 * w // 4, :] = 2
    t2_nii = nib.Nifti1Image(t2_data, affine)
    mask_nii = nib.Nifti1Image(mask, affine)
    return t2_nii, mask_nii


class TestTransformsRoundTrip(unittest.TestCase):
    def test_crop_pad_mask_roundtrip_exact(self):
        """Default config (crop/pad, no z-resample) must invert a mask EXACTLY."""
        t2_nii, mask_nii = _synthetic_case(shape=(37, 41, 6))
        original_mask = np.round(mask_nii.get_fdata()).astype(np.int64)

        config = PreprocessingConfig(spatial_mode="crop_pad", target_size=(64, 64), z_resample=False)
        _, proc_mask, meta = preprocess_case(t2_nii, mask_nii, config)

        recovered_mask = invert_preprocessing_mask(proc_mask, meta)

        self.assertEqual(recovered_mask.shape, original_mask.shape)
        np.testing.assert_array_equal(recovered_mask, original_mask)

    def test_crop_pad_mask_roundtrip_when_native_larger_than_target(self):
        """Native size > target triggers cropping; must still invert exactly
        for the retained (uncropped-away) region, and the cropped-away
        border must have been all-background so exact equality still holds."""
        t2_nii, mask_nii = _synthetic_case(shape=(80, 90, 4))
        original_mask = np.round(mask_nii.get_fdata()).astype(np.int64)

        # target smaller than native -> crop; foreground stays within the crop window
        config = PreprocessingConfig(spatial_mode="crop_pad", target_size=(64, 64), z_resample=False)
        _, proc_mask, meta = preprocess_case(t2_nii, mask_nii, config)
        recovered_mask = invert_preprocessing_mask(proc_mask, meta)

        self.assertEqual(recovered_mask.shape, original_mask.shape)
        np.testing.assert_array_equal(recovered_mask, original_mask)

    def test_mask_labels_preserved_after_forward_pass(self):
        t2_nii, mask_nii = _synthetic_case(shape=(50, 50, 5))
        config = PreprocessingConfig(spatial_mode="crop_pad", target_size=(64, 64))
        _, proc_mask, _ = preprocess_case(t2_nii, mask_nii, config)
        self.assertTrue(set(np.unique(proc_mask).tolist()).issubset({0, 1, 2}))

    def test_image_dtype_and_mask_dtype(self):
        t2_nii, mask_nii = _synthetic_case(shape=(50, 50, 5))
        config = PreprocessingConfig(spatial_mode="crop_pad", target_size=(64, 64))
        proc_t2, proc_mask, _ = preprocess_case(t2_nii, mask_nii, config)
        self.assertEqual(proc_t2.dtype, np.float32)
        self.assertEqual(proc_mask.dtype, np.int64)


if __name__ == "__main__":
    unittest.main()
