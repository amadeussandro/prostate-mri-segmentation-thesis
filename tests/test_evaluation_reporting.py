"""Tests for the evaluation result serializers (src/metrics.py reporting
helpers) and the run_evaluation orchestrator (src/evaluate.py).

The serializer tests are pure numpy + stdlib (no torch / nibabel), so they run
anywhere the metrics module imports. They lock down the machine-readable result
files the baseline evaluation must produce: per-case CSV, cohort summary CSV,
and cohort summary JSON -- including NaN handling and the explicit "unverified"
label-mapping status carried into every output.

The run_evaluation integration test skips cleanly when torch / the dataset are
unavailable (repo convention), so it is exercised in Colab via the pytest cell.
"""

import csv
import json
import os
import sys
import tempfile
import unittest

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.metrics import (
    CaseMetrics,
    DEFAULT_CLASS_NAMES,
    REPORT_METRIC_NAMES,
    aggregate_case_results,
    case_results_to_records,
    write_case_metrics_csv,
    write_summary_csv,
    write_summary_json,
)


def _make_case(patient_id, class_ids, hd95_for_class2=5.0):
    """Build a CaseMetrics with all six reported metrics per class. class 2 gets
    a NaN HD95/ASD to exercise the undefined-metric exclusion path."""
    per_class = {}
    for c in class_ids:
        per_class[c] = {
            "dice": 0.8 if c != 0 else 0.99,
            "iou": 0.7 if c != 0 else 0.98,
            "hd95_mm": (float("nan") if c == 2 else hd95_for_class2),
            "asd_mm": (float("nan") if c == 2 else 1.2),
            "precision": 0.85,
            "recall": 0.75,
        }
    return CaseMetrics(patient_id=patient_id, per_class=per_class)


class TestEvaluationReporting(unittest.TestCase):
    def setUp(self):
        self.class_ids = (0, 1, 2)
        self.cases = [_make_case(pid, self.class_ids) for pid in ("020", "021", "022")]
        self.summary = aggregate_case_results(self.cases, class_ids=self.class_ids)

    def test_records_long_form_shape_and_names(self):
        rows = case_results_to_records(self.cases, self.class_ids)
        # one row per (case, class)
        self.assertEqual(len(rows), len(self.cases) * len(self.class_ids))
        for row in rows:
            for metric in REPORT_METRIC_NAMES:
                self.assertIn(metric, row)
        # label-mapping status must be carried in the class names, not asserted as fact
        c1_names = {r["class_name"] for r in rows if r["class_id"] == 1}
        c2_names = {r["class_name"] for r in rows if r["class_id"] == 2}
        self.assertTrue(all("unverified" in n for n in c1_names))
        self.assertTrue(all("unverified" in n for n in c2_names))

    def test_per_case_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_case_metrics_csv(os.path.join(d, "pc.csv"), self.cases, self.class_ids)
            with open(path) as f:
                reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), len(self.cases) * len(self.class_ids))
            header = reader[0].keys()
            for col in ("patient_id", "class_id", "class_name", *REPORT_METRIC_NAMES):
                self.assertIn(col, header)
            # NaN hd95 for class 2 is serialized as an empty/"" -> "nan" string, still parseable
            c2 = [r for r in reader if r["class_id"] == "2"]
            self.assertTrue(all(r["hd95_mm"] in ("nan", "") or r["hd95_mm"].lower() == "nan" for r in c2))

    def test_summary_csv_has_every_class_metric_row(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_summary_csv(os.path.join(d, "sum.csv"), self.summary, self.class_ids)
            with open(path) as f:
                reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), len(self.class_ids) * len(REPORT_METRIC_NAMES))
            # class 2 HD95 was NaN for every case -> excluded, n_cases 0, n_excluded 3
            c2_hd95 = [r for r in reader if r["class_id"] == "2" and r["metric"] == "hd95_mm"][0]
            self.assertEqual(int(c2_hd95["n_excluded_undefined"]), len(self.cases))
            self.assertEqual(int(c2_hd95["n_cases"]), 0)

    def test_summary_json_structure_and_metadata(self):
        metadata = {
            "checkpoint_epoch": 77,
            "split": "val",
            "is_official_held_out_test": False,
            "label_mapping_status": "unverified",
        }
        with tempfile.TemporaryDirectory() as d:
            path = write_summary_json(os.path.join(d, "sum.json"), self.summary, metadata)
            with open(path) as f:
                payload = json.load(f)
            self.assertIn("metadata", payload)
            self.assertIn("per_class", payload)
            self.assertEqual(payload["metadata"]["checkpoint_epoch"], 77)
            self.assertFalse(payload["metadata"]["is_official_held_out_test"])
            # JSON keys are stringified class ids; dice mean present for class 1
            self.assertIn("1", payload["per_class"])
            self.assertIn("metrics", payload["per_class"]["1"])
            self.assertIn("dice", payload["per_class"]["1"]["metrics"])
            self.assertIn("unverified", payload["per_class"]["1"]["class_name"])

    def test_nan_not_written_as_bare_token_breaking_strict_json(self):
        """json.dump would emit NaN (invalid JSON) if a NaN leaked into metadata;
        the summary numbers themselves are already float() from aggregate. This
        guards that the file we write is loadable by a strict parser."""
        with tempfile.TemporaryDirectory() as d:
            path = write_summary_json(os.path.join(d, "s.json"), self.summary, {"x": 1})
            with open(path) as f:
                text = f.read()
            # Must be STRICT JSON: no bare NaN/Infinity tokens (would break
            # pandas.read_json / JS JSON.parse). class-2 HD95 was undefined for
            # every case -> its mean must serialize as null, never NaN.
            self.assertNotIn("NaN", text)
            self.assertNotIn("Infinity", text)
            payload = json.loads(text)  # would raise if NaN leaked and we set allow_nan=False upstream
            self.assertIsNone(payload["per_class"]["2"]["metrics"]["hd95_mm"]["mean"])
            self.assertTrue(np.isfinite(payload["per_class"]["1"]["metrics"]["dice"]["mean"]))


class TestRunEvaluationIntegration(unittest.TestCase):
    """End-to-end orchestration test. Skips unless torch + the dataset are
    present (so it runs in Colab via the pytest cell, not in dependency-free
    environments)."""

    def setUp(self):
        self.dataset_root = os.path.join(project_root, "dataset", "prostate158_train", "train")
        if not os.path.isdir(os.path.join(self.dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping integration test.")

    def test_run_evaluation_writes_outputs_for_a_tiny_split(self):
        import torch

        from src.dataset import ProstateZonal2DDataset  # noqa: F401
        from src.evaluate import run_evaluation
        from src.model import ProstateUNet2D
        from src.transforms import PreprocessingConfig

        # Build and save a tiny throwaway checkpoint (random weights) in the
        # exact format run_training writes, so no real training is needed.
        model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=8)
        cfg = {
            "data": {
                "dataset_root": self.dataset_root,
                "patient_ids": ["020"],
                "target_mask": "t2_anatomy_reader1.nii.gz",
            },
            "validation": {"val_patient_ids": ["020"]},
            "model": {"in_channels": 1, "out_channels": 3, "init_features": 8, "dropout_rate": 0.2},
            "preprocessing": {"normalization": "per_volume", "z_resample": False,
                              "spatial_mode": "crop_pad", "target_size": [270, 270]},
            "experiment": {"output_dir": "."},
        }
        with tempfile.TemporaryDirectory() as d:
            ckpt_path = os.path.join(d, "tiny.pt")
            torch.save(
                {"epoch": 1, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": {}, "best_metric": 0.123, "config": cfg},
                ckpt_path,
            )
            result = run_evaluation(
                config_path=None if False else _write_tmp_config(cfg, d),
                checkpoint_path=ckpt_path,
                output_dir=d,
                split="val",
                device="cpu",
            )
            self.assertTrue(os.path.exists(result["per_case_csv"]))
            self.assertTrue(os.path.exists(result["summary_csv"]))
            self.assertTrue(os.path.exists(result["summary_json"]))
            with open(result["summary_json"]) as f:
                payload = json.load(f)
            self.assertFalse(payload["metadata"]["is_official_held_out_test"])
            self.assertEqual(payload["metadata"]["checkpoint_epoch"], 1)


def _write_tmp_config(cfg, d):
    import yaml
    p = os.path.join(d, "cfg.yaml")
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f)
    return p


if __name__ == "__main__":
    unittest.main()
