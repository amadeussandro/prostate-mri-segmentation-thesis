"""Tests for scripts/visualize_baseline_cases.py.

Selection and slice-picking are pure numpy/stdlib logic and are tested without
torch or the dataset. Rendering is tested on tiny synthetic volumes. A full
run() integration test uses a random-weight throwaway checkpoint on real
patient-020 data and skips cleanly when torch/data are unavailable.
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
scripts_dir = os.path.join(project_root, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

import visualize_baseline_cases as viz


def _write_metrics_csv(path, rows):
    """rows: list of (patient_id, class_id, dice)."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "class_id", "class_name", "dice",
                                          "iou", "hd95_mm", "asd_mm", "precision", "recall"])
        w.writeheader()
        for pid, cid, dice in rows:
            w.writerow({"patient_id": pid, "class_id": cid, "class_name": f"class{cid}",
                        "dice": dice, "iou": dice * 0.9, "hd95_mm": 5.0, "asd_mm": 1.0,
                        "precision": dice, "recall": dice})


class TestPatientIdNormalization(unittest.TestCase):
    def test_pads_numeric(self):
        self.assertEqual(viz.normalize_patient_id("60"), "060")
        self.assertEqual(viz.normalize_patient_id("060"), "060")
        self.assertEqual(viz.normalize_patient_id(87), "087")

    def test_nonnumeric_stripped(self):
        self.assertEqual(viz.normalize_patient_id("  caseA  "), "caseA")


class TestCsvSelection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.csv = os.path.join(self.tmp.name, "m.csv")
        # 5 patients, mean foreground (class 1&2) dice: 060=0.30, 077=0.40,
        # 087=0.90, 090=0.60, 100=0.75  (background class 0 must be ignored)
        rows = []
        for pid, d in [("060", 0.30), ("077", 0.40), ("087", 0.90), ("090", 0.60), ("100", 0.75)]:
            rows.append((pid, 0, 0.99))  # background -- must not affect ranking
            rows.append((pid, 1, d))
            rows.append((pid, 2, d))
        _write_metrics_csv(self.csv, rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_mean_dice_ignores_background(self):
        means = viz.load_per_case_dice(self.csv)
        self.assertAlmostEqual(means["087"], 0.90)
        self.assertAlmostEqual(means["060"], 0.30)

    def test_worst_selection(self):
        self.assertEqual(viz.select_patients_by_mode(self.csv, "worst", 3), ["060", "077", "090"])

    def test_best_selection(self):
        self.assertEqual(viz.select_patients_by_mode(self.csv, "best", 3), ["087", "100", "090"])

    def test_median_selection_centered(self):
        # ascending: 060,077,090,100,087 -> median index 2 (090); n=1 -> [090]
        self.assertEqual(viz.select_patients_by_mode(self.csv, "median", 1), ["090"])

    def test_n_capped_to_available(self):
        self.assertEqual(len(viz.select_patients_by_mode(self.csv, "worst", 99)), 5)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            viz.select_patients_by_mode(self.csv, "sideways", 3)

    def test_missing_column_raises(self):
        bad = os.path.join(self.tmp.name, "bad.csv")
        with open(bad, "w", newline="") as f:
            f.write("patient_id,foo\n060,1\n")
        with self.assertRaises(ValueError):
            viz.load_per_case_dice(bad)


class TestSliceSelection(unittest.TestCase):
    def test_picks_max_foreground_and_disagreement(self):
        gt = np.zeros((10, 10, 6), dtype=np.int64)
        pred = np.zeros((10, 10, 6), dtype=np.int64)
        # slice 2: large GT foreground, perfectly predicted
        gt[2:8, 2:8, 2] = 1
        pred[2:8, 2:8, 2] = 1
        # slice 4: GT present, prediction totally wrong (max disagreement)
        gt[3:6, 3:6, 4] = 2
        pred[0:2, 0:2, 4] = 1
        idx = viz.select_slice_indices(gt, pred, n=3)
        self.assertIn(2, idx)   # representative (max GT foreground)
        self.assertIn(4, idx)   # max disagreement
        self.assertEqual(idx, sorted(idx))  # ascending

    def test_empty_gt_falls_back_to_middle(self):
        gt = np.zeros((8, 8, 5), dtype=np.int64)
        pred = np.zeros((8, 8, 5), dtype=np.int64)
        idx = viz.select_slice_indices(gt, pred, n=3)
        self.assertEqual(idx, [2])  # middle of depth 5

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            viz.select_slice_indices(np.zeros((4, 4, 3)), np.zeros((4, 4, 2)))


class TestRendering(unittest.TestCase):
    def test_render_creates_pngs_and_dir(self):
        h, w, d = 16, 16, 4
        t2 = np.random.default_rng(0).uniform(0, 500, size=(h, w, d)).astype(np.float32)
        gt = np.zeros((h, w, d), dtype=np.int64)
        pred = np.zeros((h, w, d), dtype=np.int64)
        gt[4:10, 4:10, 1] = 1
        pred[4:9, 4:10, 1] = 1
        gt[5:8, 5:8, 2] = 2
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = os.path.join(tmp, "patient_060")
            written = viz.render_case_slices(t2, gt, pred, [1, 2], "060", out_dir)
            self.assertTrue(os.path.isdir(out_dir))
            self.assertEqual(len(written), 2)
            for p in written:
                self.assertTrue(os.path.exists(p) and os.path.getsize(p) > 0)


class TestRunIntegration(unittest.TestCase):
    def setUp(self):
        self.dataset_root = os.path.join(project_root, "dataset", "prostate158_train", "train")
        if not os.path.isdir(os.path.join(self.dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping integration test.")

    def test_run_explicit_patient_end_to_end(self):
        import torch
        import yaml

        from src.model import ProstateUNet2D

        with tempfile.TemporaryDirectory() as tmp:
            # Throwaway random-weight checkpoint in the run_training format.
            model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=8)
            ckpt_cfg = {
                "model": {"in_channels": 1, "out_channels": 3, "init_features": 8, "dropout_rate": 0.2},
                "preprocessing": {"normalization": "per_volume", "z_resample": False,
                                  "spatial_mode": "crop_pad", "target_size": [270, 270]},
            }
            ckpt_path = os.path.join(tmp, "best_model.pt")
            torch.save({"epoch": 1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": {}, "best_metric": 0.5, "config": ckpt_cfg}, ckpt_path)

            cfg = {
                "data": {"dataset_root": self.dataset_root, "patient_ids": ["020"],
                         "target_mask": "t2_anatomy_reader1.nii.gz"},
                "validation": {"val_patient_ids": ["020"]},
            }
            cfg_path = os.path.join(tmp, "cfg.yaml")
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f)

            out_dir = os.path.join(tmp, "visualizations")
            summary = viz.run(
                config_path=cfg_path, checkpoint_path=ckpt_path, output_dir=out_dir,
                split="val", patients=["20"], n_slices=2, device="cpu",
            )
            self.assertEqual(len(summary["cases"]), 1)
            case = summary["cases"][0]
            self.assertEqual(case["patient_id"], "020")
            self.assertEqual(case["geometry_shape"], [270, 270, 24])  # geometry preserved
            self.assertTrue(len(case["figures"]) >= 1)
            for p in case["figures"]:
                self.assertTrue(os.path.exists(p))
            self.assertTrue(os.path.exists(summary["summary_path"]))
            with open(summary["summary_path"]) as f:
                payload = json.load(f)
            self.assertFalse(payload["is_official_held_out_test"])
            self.assertIn("unverified", payload["label_mapping_status"])

    def test_run_invalid_patient_is_reported(self):
        import torch
        import yaml

        from src.model import ProstateUNet2D

        with tempfile.TemporaryDirectory() as tmp:
            model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=8)
            ckpt_path = os.path.join(tmp, "best_model.pt")
            torch.save({"epoch": 1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": {}, "best_metric": 0.5,
                        "config": {"model": {"in_channels": 1, "out_channels": 3, "init_features": 8},
                                   "preprocessing": {"target_size": [270, 270]}}}, ckpt_path)
            cfg = {"data": {"dataset_root": self.dataset_root, "patient_ids": ["020"]},
                   "validation": {"val_patient_ids": ["020"]}}
            cfg_path = os.path.join(tmp, "cfg.yaml")
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f)
            # All-invalid selection must raise a clear error.
            with self.assertRaises(ValueError):
                viz.run(config_path=cfg_path, checkpoint_path=ckpt_path,
                        output_dir=os.path.join(tmp, "v"), split="val",
                        patients=["99999"], device="cpu")


if __name__ == "__main__":
    unittest.main()
