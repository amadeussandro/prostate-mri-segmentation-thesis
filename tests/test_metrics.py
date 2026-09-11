"""Unit tests for src/metrics.py -- volume-level Dice/IoU/HD95/ASD/precision/recall."""

import os
import sys
import unittest

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.metrics import (
    aggregate_case_results,
    average_surface_distance,
    dice_score,
    evaluate_case_volume,
    hausdorff_distance_95,
    iou_score,
    precision_recall,
)


class TestBasicMetrics(unittest.TestCase):
    def test_dice_identical_is_one(self):
        vol = np.zeros((10, 10, 5), dtype=np.int64)
        vol[2:5, 2:5, :] = 1
        self.assertAlmostEqual(dice_score(vol, vol, 1), 1.0)

    def test_dice_disjoint_is_zero(self):
        a = np.zeros((10, 10, 5), dtype=np.int64)
        a[0:2, 0:2, :] = 1
        b = np.zeros((10, 10, 5), dtype=np.int64)
        b[8:10, 8:10, :] = 1
        self.assertAlmostEqual(dice_score(a, b, 1), 0.0)

    def test_dice_both_empty_is_one(self):
        a = np.zeros((5, 5, 2), dtype=np.int64)
        b = np.zeros((5, 5, 2), dtype=np.int64)
        self.assertAlmostEqual(dice_score(a, b, 1), 1.0)

    def test_iou_identical_is_one(self):
        vol = np.zeros((10, 10, 5), dtype=np.int64)
        vol[2:5, 2:5, :] = 1
        self.assertAlmostEqual(iou_score(vol, vol, 1), 1.0)

    def test_precision_recall_perfect(self):
        vol = np.zeros((10, 10, 5), dtype=np.int64)
        vol[2:5, 2:5, :] = 1
        p, r = precision_recall(vol, vol, 1)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 1.0)

    def test_precision_recall_partial(self):
        gt = np.zeros((10, 10, 1), dtype=np.int64)
        gt[0:4, 0:4, 0] = 1  # 16 voxels
        pred = np.zeros((10, 10, 1), dtype=np.int64)
        pred[0:2, 0:4, 0] = 1  # 8 voxels, all within gt -> precision=1, recall=0.5
        p, r = precision_recall(pred, gt, 1)
        self.assertAlmostEqual(p, 1.0)
        self.assertAlmostEqual(r, 0.5)


class TestSurfaceDistanceMetrics(unittest.TestCase):
    def test_hd95_identical_cubes_is_zero(self):
        vol = np.zeros((20, 20, 20), dtype=np.int64)
        vol[5:15, 5:15, 5:15] = 1
        hd95 = hausdorff_distance_95(vol, vol, spacing=(1.0, 1.0, 1.0), class_id=1)
        self.assertAlmostEqual(hd95, 0.0, places=6)

    def test_hd95_offset_cubes_matches_expected_shift(self):
        spacing = (1.0, 1.0, 1.0)
        a = np.zeros((30, 30, 30), dtype=np.int64)
        a[5:15, 5:15, 5:15] = 1
        b = np.zeros((30, 30, 30), dtype=np.int64)
        b[10:20, 5:15, 5:15] = 1  # shifted by 5 voxels along axis 0
        hd95 = hausdorff_distance_95(a, b, spacing, class_id=1)
        # Max face-to-face separation along the shift axis is 5mm; HD95 should
        # be on that order (bounded above by the corner-to-corner distance).
        self.assertGreater(hd95, 0.0)
        self.assertLessEqual(hd95, 10.0)

    def test_hd95_scales_with_spacing(self):
        a = np.zeros((30, 30, 30), dtype=np.int64)
        a[5:15, 5:15, 5:15] = 1
        b = np.zeros((30, 30, 30), dtype=np.int64)
        b[10:20, 5:15, 5:15] = 1

        hd_iso = hausdorff_distance_95(a, b, spacing=(1.0, 1.0, 1.0), class_id=1)
        hd_scaled = hausdorff_distance_95(a, b, spacing=(2.0, 1.0, 1.0), class_id=1)
        self.assertGreater(hd_scaled, hd_iso)

    def test_hd95_both_empty_is_zero(self):
        vol = np.zeros((10, 10, 10), dtype=np.int64)
        self.assertEqual(hausdorff_distance_95(vol, vol, (1, 1, 1), class_id=1), 0.0)

    def test_hd95_one_empty_is_nan(self):
        a = np.zeros((10, 10, 10), dtype=np.int64)
        b = np.zeros((10, 10, 10), dtype=np.int64)
        b[2:4, 2:4, 2:4] = 1
        result = hausdorff_distance_95(a, b, (1, 1, 1), class_id=1)
        self.assertTrue(np.isnan(result))

    def test_asd_identical_is_zero(self):
        vol = np.zeros((15, 15, 15), dtype=np.int64)
        vol[3:10, 3:10, 3:10] = 1
        asd = average_surface_distance(vol, vol, (1, 1, 1), class_id=1)
        self.assertAlmostEqual(asd, 0.0, places=6)


class TestCaseAndAggregation(unittest.TestCase):
    def test_evaluate_case_volume_returns_all_classes(self):
        pred = np.zeros((10, 10, 5), dtype=np.int64)
        gt = np.zeros((10, 10, 5), dtype=np.int64)
        pred[2:5, 2:5, :] = 1
        gt[2:5, 2:5, :] = 1
        result = evaluate_case_volume(pred, gt, spacing=(1, 1, 1), class_ids=[0, 1, 2], patient_id="test")
        self.assertEqual(set(result.per_class.keys()), {0, 1, 2})
        self.assertAlmostEqual(result.per_class[1]["dice"], 1.0)

    def test_aggregate_excludes_nan_and_reports_count(self):
        pred_a = np.zeros((10, 10, 5), dtype=np.int64)
        gt_a = np.zeros((10, 10, 5), dtype=np.int64)
        gt_a[2:4, 2:4, :] = 2  # class 2 present in gt but absent in pred -> HD95 NaN

        case_a = evaluate_case_volume(pred_a, gt_a, (1, 1, 1), [0, 1, 2], "a")
        case_b = evaluate_case_volume(gt_a, gt_a, (1, 1, 1), [0, 1, 2], "b")  # perfect match

        summary = aggregate_case_results([case_a, case_b], class_ids=[0, 1, 2])
        self.assertEqual(summary[2]["hd95_mm"]["n_excluded_undefined"], 1)
        self.assertEqual(summary[2]["hd95_mm"]["n_cases"], 1)
        self.assertAlmostEqual(summary[2]["dice"]["mean"], (0.0 + 1.0) / 2.0)


if __name__ == "__main__":
    unittest.main()
