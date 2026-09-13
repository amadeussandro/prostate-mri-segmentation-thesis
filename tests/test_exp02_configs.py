"""Scientific-control tests for the Experiment 2 preprocessing ablation configs.

These assert the ONLY difference between each Exp02 config and the Experiment-1
baseline is the intended preprocessing factor -- everything that must stay
constant (model, optimizer, lr, seed, epochs, batch size, split, target mask,
loss, slice sampling) is verified identical. Also verifies: no test split is
referenced, seeds are deterministic and equal, and every Exp02 output dir is
isolated from Experiment 1.

Plus unit tests for the leakage-safe global-stats auto-computation used by
Exp02-B (train-split-only; no-op for the baseline's per-volume path).
"""

import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_config

CONFIG_DIR = os.path.join(project_root, "configs")
BASELINE = os.path.join(CONFIG_DIR, "config_baseline.yaml")
EXP02 = {
    "A_zresample_on": os.path.join(CONFIG_DIR, "experiments", "exp02_a_zresample_on.yaml"),
    "B_norm_global": os.path.join(CONFIG_DIR, "experiments", "exp02_b_normalization_global.yaml"),
    "C_spatial_resize": os.path.join(CONFIG_DIR, "experiments", "exp02_c_spatial_resize.yaml"),
}

# Keys that MUST be identical between the baseline and every Exp02 arm.
CONTROLLED_MODEL_KEYS = ("architecture", "in_channels", "out_channels", "init_features", "dropout_rate")
CONTROLLED_TRAINING_KEYS = ("batch_size", "learning_rate", "num_epochs", "seed",
                            "optimizer", "weight_decay", "loss_function")
CONTROLLED_DATA_KEYS = ("dataset_root", "train_csv", "valid_csv", "target_mask", "slice_sampling")


class TestExp02ScientificControl(unittest.TestCase):
    def setUp(self):
        self.baseline = load_config(BASELINE)
        self.configs = {name: load_config(path) for name, path in EXP02.items()}

    def test_all_exp02_configs_exist_and_parse(self):
        for name, cfg in self.configs.items():
            self.assertIn("preprocessing", cfg, f"{name} missing preprocessing block")
            self.assertIn("model", cfg)
            self.assertIn("training", cfg)

    def test_model_identical_to_baseline(self):
        for name, cfg in self.configs.items():
            for k in CONTROLLED_MODEL_KEYS:
                self.assertEqual(
                    cfg["model"][k], self.baseline["model"][k],
                    f"{name}: model.{k} differs from baseline (must be fixed in Exp02)",
                )

    def test_training_hyperparameters_identical_to_baseline(self):
        for name, cfg in self.configs.items():
            for k in CONTROLLED_TRAINING_KEYS:
                self.assertEqual(
                    cfg["training"][k], self.baseline["training"][k],
                    f"{name}: training.{k} differs from baseline (must be fixed in Exp02)",
                )

    def test_data_split_and_target_identical_to_baseline(self):
        for name, cfg in self.configs.items():
            for k in CONTROLLED_DATA_KEYS:
                self.assertEqual(
                    cfg["data"][k], self.baseline["data"][k],
                    f"{name}: data.{k} differs from baseline (split/target must be fixed)",
                )

    def test_seed_is_deterministic_and_equal(self):
        for name, cfg in self.configs.items():
            self.assertEqual(cfg["training"]["seed"], 42)

    def test_no_test_split_referenced(self):
        for name, cfg in self.configs.items():
            data = cfg.get("data", {})
            self.assertNotIn("test_csv", data, f"{name} must NOT reference a test split")
            self.assertNotIn("test_dir", data, f"{name} must NOT reference a test split")

    def test_output_dirs_isolated_from_exp01(self):
        baseline_out = self.baseline["experiment"]["output_dir"].replace("\\", "/").rstrip("/")
        seen = set()
        for name, cfg in self.configs.items():
            out = cfg["experiment"]["output_dir"].replace("\\", "/").rstrip("/")
            self.assertTrue(
                out.startswith("./results/exp02_preprocessing") or out.startswith("results/exp02_preprocessing"),
                f"{name}: output_dir '{out}' must live under results/exp02_preprocessing/",
            )
            self.assertNotEqual(out, baseline_out, f"{name}: output_dir collides with Exp01")
            self.assertNotIn("exp01", out, f"{name}: output_dir must not touch Exp01 tree")
            self.assertNotIn(out, seen, f"{name}: duplicate output_dir across Exp02 arms")
            seen.add(out)

    # ---- independent-variable isolation, per factor ----

    def test_factor_A_only_zresample_changes(self):
        base_p = self.baseline["preprocessing"]
        p = self.configs["A_zresample_on"]["preprocessing"]
        self.assertTrue(p["z_resample"])                       # the IV: ON
        self.assertFalse(base_p["z_resample"])                 # baseline: OFF
        # everything else in preprocessing stays at baseline values
        self.assertEqual(p["normalization"], base_p["normalization"])
        self.assertEqual(p["spatial_mode"], base_p["spatial_mode"])
        self.assertEqual(list(p["target_size"]), list(base_p["target_size"]))

    def test_factor_B_only_normalization_changes(self):
        base_p = self.baseline["preprocessing"]
        p = self.configs["B_norm_global"]["preprocessing"]
        self.assertEqual(p["normalization"], "global")         # the IV
        self.assertEqual(base_p["normalization"], "per_volume")
        # global_stats must NOT be hardcoded (auto-derived from train split)
        self.assertIsNone(p.get("global_stats"))
        # everything else stays at baseline values
        self.assertEqual(p["z_resample"], base_p["z_resample"])
        self.assertEqual(p["spatial_mode"], base_p["spatial_mode"])
        self.assertEqual(list(p["target_size"]), list(base_p["target_size"]))

    def test_factor_C_only_spatial_handling_changes(self):
        base_p = self.baseline["preprocessing"]
        p = self.configs["C_spatial_resize"]["preprocessing"]
        # Factor C's IV is spatial handling = (spatial_mode + its target_size).
        self.assertEqual(p["spatial_mode"], "resize")
        self.assertEqual(base_p["spatial_mode"], "crop_pad")
        # the OTHER preprocessing factors stay at baseline values
        self.assertEqual(p["normalization"], base_p["normalization"])
        self.assertEqual(p["z_resample"], base_p["z_resample"])


class TestGlobalStatsLeakageGuard(unittest.TestCase):
    """src/train.py::_ensure_global_stats must (1) no-op for per_volume,
    (2) respect explicit stats, (3) compute from the TRAIN split only."""

    def test_noop_for_per_volume(self):
        from src.train import _ensure_global_stats
        cfg = {"preprocessing": {"normalization": "per_volume"},
               "data": {"dataset_root": "x"}}
        changed = _ensure_global_stats(cfg, ["020", "021"])
        self.assertFalse(changed)
        self.assertNotIn("global_stats", cfg["preprocessing"])

    def test_respects_explicit_stats(self):
        from src.train import _ensure_global_stats
        cfg = {"preprocessing": {"normalization": "global",
                                 "global_stats": {"p_low": 1.0, "p_high": 9.0, "mean": 5.0, "std": 2.0}},
               "data": {"dataset_root": "x"}}
        changed = _ensure_global_stats(cfg, ["020"])
        self.assertFalse(changed)
        self.assertEqual(cfg["preprocessing"]["global_stats"]["mean"], 5.0)

    def test_computes_from_train_ids_only(self):
        import src.transforms as transforms_mod
        from src.train import _ensure_global_stats

        captured = {}

        def fake_compute(dataset_root, case_ids, image_name="t2.nii.gz"):
            captured["case_ids"] = list(case_ids)
            return {"p_low": 0.0, "p_high": 100.0, "mean": 50.0, "std": 10.0}

        original = transforms_mod.compute_global_intensity_stats
        transforms_mod.compute_global_intensity_stats = fake_compute
        try:
            cfg = {"preprocessing": {"normalization": "global"},
                   "data": {"dataset_root": "x"}}
            train_ids = ["020", "021", "022"]  # NB: no validation/test ids here
            changed = _ensure_global_stats(cfg, train_ids)
        finally:
            transforms_mod.compute_global_intensity_stats = original

        self.assertTrue(changed)
        # Leakage guard: stats computed over exactly the train ids, nothing else.
        self.assertEqual(captured["case_ids"], train_ids)
        self.assertEqual(cfg["preprocessing"]["global_stats"]["mean"], 50.0)


if __name__ == "__main__":
    unittest.main()
