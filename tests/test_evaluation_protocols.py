"""Gate 12 (Experiment 3 readiness) tests: scoring the SAME model predictions
under different evaluation protocols (per-case/original-voxel vs
resampled-space, plus per-slice pooling), with no retraining.
"""

import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestEvaluationProtocols(unittest.TestCase):
    def setUp(self):
        self.dataset_root = os.path.join(project_root, "dataset", "prostate158_train", "train")
        if not os.path.isdir(os.path.join(self.dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping integration test.")

    def test_protocol_comparison_runs_model_once_and_scores_both_ways(self):
        from src.dataset import ProstateZonal2DDataset
        from src.evaluate import evaluate_patient_protocols
        from src.metrics import aggregate_per_slice_pooled
        from src.model import ProstateUNet2D

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            slice_sampling="all",
        )
        model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=8)

        result = evaluate_patient_protocols("020", model, dataset, device="cpu", class_ids=(0, 1, 2))

        self.assertIn("original_voxel_per_case", result)
        self.assertIn("resampled_per_case", result)
        self.assertIn("per_slice_dice", result)

        # Per-slice list length must equal the number of axial slices for this case.
        geometry, _ = dataset.get_case_metadata("020")
        self.assertEqual(len(result["per_slice_dice"][1]), geometry.shape[2])

        pooled = aggregate_per_slice_pooled([result["per_slice_dice"]], class_ids=[0, 1, 2])
        self.assertEqual(pooled[1]["n_slices"], geometry.shape[2])

    def test_resampled_per_case_schema_matches_documentation(self):
        """Fix-pass regression test: resampled_per_case must contain exactly
        the metrics its own docstring/config comments claim (Dice, IoU,
        precision, recall -- no HD95/ASD, since those are spacing-dependent
        and this protocol scores in the un-inverted, possibly-resized grid)."""
        from src.dataset import ProstateZonal2DDataset
        from src.evaluate import evaluate_patient_protocols
        from src.model import ProstateUNet2D

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root, patient_ids=["020"], slice_sampling="all",
        )
        model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=8)

        result = evaluate_patient_protocols("020", model, dataset, device="cpu", class_ids=(0, 1, 2))

        expected_keys = {"dice", "iou", "precision", "recall"}
        for class_id, metrics in result["resampled_per_case"].items():
            self.assertEqual(
                set(metrics.keys()), expected_keys,
                f"resampled_per_case[{class_id}] keys {set(metrics.keys())} "
                f"don't match documented schema {expected_keys}",
            )
            for value in metrics.values():
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
