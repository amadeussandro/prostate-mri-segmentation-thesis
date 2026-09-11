"""Gate 9 tests: volume-level, per-case, original-voxel-space evaluation
pipeline plumbing (src/evaluate.py: evaluate_patient / evaluate_cases).

Uses an untrained (randomly initialized) model -- this test is about proving
the pipeline is structurally correct (shapes, reconstruction, aggregation),
NOT about model accuracy. Model quality is only meaningful after training
(Experiment 1), which this phase deliberately does not run.
"""

import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestVolumeLevelEvaluation(unittest.TestCase):
    def setUp(self):
        self.dataset_root = os.path.join(
            project_root, "dataset", "prostate158_train", "train"
        )
        if not os.path.isdir(os.path.join(self.dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping integration test.")

    def test_evaluate_patient_returns_valid_metrics(self):
        from src.dataset import ProstateZonal2DDataset
        from src.evaluate import evaluate_patient
        from src.model import ProstateUNet2D

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020"],
            slice_sampling="all",
        )
        model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=8)

        result = evaluate_patient("020", model, dataset, device="cpu", class_ids=(0, 1, 2))

        self.assertEqual(result.patient_id, "020")
        self.assertEqual(set(result.per_class.keys()), {0, 1, 2})
        for class_id, metrics in result.per_class.items():
            self.assertIn("dice", metrics)
            self.assertIn("hd95_mm", metrics)
            self.assertGreaterEqual(metrics["dice"], 0.0)
            self.assertLessEqual(metrics["dice"], 1.0)

    def test_evaluate_cases_aggregates_across_cohort(self):
        if not os.path.isdir(os.path.join(self.dataset_root, "021")):
            self.skipTest("Patient 021 not found; skipping multi-case aggregation test.")

        from src.dataset import ProstateZonal2DDataset
        from src.evaluate import evaluate_cases
        from src.model import ProstateUNet2D

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root,
            patient_ids=["020", "021"],
            slice_sampling="all",
        )
        model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=8)

        case_results, summary = evaluate_cases(model, dataset, device="cpu", class_ids=(0, 1, 2))

        self.assertEqual(len(case_results), 2)
        for class_id in (0, 1, 2):
            self.assertIn("mean", summary[class_id]["dice"])
            self.assertIn("ci95_low", summary[class_id]["dice"])
            self.assertEqual(summary[class_id]["dice"]["n_cases"], 2)


if __name__ == "__main__":
    unittest.main()
