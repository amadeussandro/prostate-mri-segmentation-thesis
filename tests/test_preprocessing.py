"""Unit tests for preprocessing scaffolding functions."""

import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path for robust imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from code.preprocessing import (
    binarize_mask,
    crop_prostate_roi,
    normalize_intensity,
    stack_modalities,
)


class TestPreprocessing(unittest.TestCase):
    """Test suite for preprocessing operations."""

    def test_binarize_mask_standard(self):
        """Test binary mask thresholding."""
        raw = np.array([[0, 1], [0, 1]], dtype=np.float32)
        binary = binarize_mask(raw)
        self.assertEqual(binary.dtype, np.uint8)
        self.assertTrue(np.array_equal(binary, [[0, 1], [0, 1]]))

    def test_binarize_mask_special_label_3(self):
        """Verify handling of label 3.0 (verified in t2_tumor_reader1)."""
        raw = np.array([[0.0, 3.0], [0.0, 3.0]], dtype=np.float32)
        binary = binarize_mask(raw)
        self.assertEqual(binary.dtype, np.uint8)
        self.assertTrue(np.array_equal(binary, [[0, 1], [0, 1]]))

    def test_normalize_intensity_shape_and_type(self):
        """Verify normalization preserves array shape and produces float32."""
        dummy = np.random.uniform(0, 1000, size=(270, 270)).astype(np.float32)
        normed = normalize_intensity(dummy, method="minmax")
        self.assertEqual(normed.shape, (270, 270))
        self.assertEqual(normed.dtype, np.float32)
        self.assertAlmostEqual(float(np.min(normed)), 0.0, places=4)
        self.assertAlmostEqual(float(np.max(normed)), 1.0, places=4)

    def test_stack_modalities(self):
        """Verify stacking three 2D slices into a 3-channel tensor."""
        t2 = np.ones((270, 270), dtype=np.float32)
        adc = np.ones((270, 270), dtype=np.float32) * 2
        dwi = np.ones((270, 270), dtype=np.float32) * 3

        stacked = stack_modalities(t2, adc, dwi)
        self.assertEqual(stacked.shape, (3, 270, 270))
        self.assertEqual(stacked.dtype, np.float32)
        self.assertEqual(float(stacked[0, 0, 0]), 1.0)
        self.assertEqual(float(stacked[1, 0, 0]), 2.0)
        self.assertEqual(float(stacked[2, 0, 0]), 3.0)

    def test_crop_prostate_roi(self):
        """Verify central cropping to target dimensions."""
        img = np.zeros((270, 270), dtype=np.float32)
        cropped = crop_prostate_roi(img, target_shape=(160, 160))
        self.assertEqual(cropped.shape, (160, 160))


if __name__ == "__main__":
    unittest.main()

