"""Unit tests for the RQ-3 inter-slice consistency analysis (src/metrics.py).
Kept separate from primary Dice/IoU/HD95/ASD tests per thesis protocol --
this is a supporting analysis, not a claimed novel metric.
"""

import os
import sys
import unittest

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.metrics import inter_slice_dice_inconsistency


class TestInterSliceConsistency(unittest.TestCase):
    def test_perfect_prediction_is_zero(self):
        vol = np.zeros((10, 10, 6), dtype=np.int64)
        vol[2:5, 2:5, :] = 1
        score = inter_slice_dice_inconsistency(vol, vol, class_id=1)
        self.assertAlmostEqual(score, 0.0)

    def test_alternating_errors_score_higher_than_uniform_errors(self):
        gt = np.zeros((10, 10, 4), dtype=np.int64)
        gt[2:6, 2:6, :] = 1

        # Uniform: every slice equally degraded -> low inter-slice variance.
        uniform_pred = np.zeros((10, 10, 4), dtype=np.int64)
        uniform_pred[2:6, 2:5, :] = 1  # same partial overlap every slice

        # Alternating: perfect on even slices, empty on odd slices -> high variance.
        alternating_pred = np.zeros((10, 10, 4), dtype=np.int64)
        alternating_pred[2:6, 2:6, 0] = 1
        alternating_pred[2:6, 2:6, 2] = 1
        # slices 1, 3 left empty (prediction misses entirely)

        uniform_score = inter_slice_dice_inconsistency(uniform_pred, gt, class_id=1)
        alternating_score = inter_slice_dice_inconsistency(alternating_pred, gt, class_id=1)

        self.assertGreater(alternating_score, uniform_score)

    def test_single_slice_volume_is_zero(self):
        vol = np.zeros((5, 5, 1), dtype=np.int64)
        self.assertEqual(inter_slice_dice_inconsistency(vol, vol, class_id=1), 0.0)


if __name__ == "__main__":
    unittest.main()
