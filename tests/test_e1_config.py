"""Scientific-control tests for the Experiment E1 (Dice+CE loss) config.

E1's validity rests on ONE claim: the loss function is the only thing that
changed. These tests enforce that claim mechanically, so config drift cannot
silently turn E1 into a multi-variable experiment.

Verified here:
  * every controlled key (model, optimizer, LR, weight decay, batch size,
    epochs, seed, preprocessing, split, target mask, slice sampling) is
    IDENTICAL to configs/config_baseline.yaml;
  * the loss really is the independent variable (baseline CE vs E1 dice_ce);
  * AMP is OFF in both arms (FP32 comparison);
  * E1 references no test split and writes to an isolated output_dir;
  * E1 does not accidentally carry a conflicting `training.loss_function`.

Also covers the checkpoint-compatibility guard that stops an E1 run from
resuming out of the baseline's checkpoint.
"""

import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_config

CONFIG_DIR = os.path.join(project_root, "configs")
BASELINE_PATH = os.path.join(CONFIG_DIR, "config_baseline.yaml")
E1_PATH = os.path.join(CONFIG_DIR, "experiments", "exp_e1_dice_ce.yaml")

CONTROLLED_MODEL_KEYS = ("architecture", "in_channels", "out_channels", "init_features", "dropout_rate")
CONTROLLED_TRAINING_KEYS = ("batch_size", "learning_rate", "num_epochs", "seed", "optimizer", "weight_decay")
CONTROLLED_DATA_KEYS = ("dataset_root", "train_csv", "valid_csv", "target_mask", "slice_sampling")
CONTROLLED_PREPROCESSING_KEYS = ("normalization", "z_resample", "spatial_mode", "target_size")


class TestE1ScientificControl(unittest.TestCase):
    def setUp(self):
        self.baseline = load_config(BASELINE_PATH)
        self.e1 = load_config(E1_PATH)

    def test_e1_config_exists_and_parses(self):
        self.assertTrue(os.path.isfile(E1_PATH), "E1 config file is missing")
        for block in ("experiment", "data", "preprocessing", "model", "training", "loss"):
            self.assertIn(block, self.e1, f"E1 config missing `{block}` block")

    def test_model_identical_to_baseline(self):
        for key in CONTROLLED_MODEL_KEYS:
            self.assertEqual(
                self.e1["model"][key], self.baseline["model"][key],
                f"model.{key} differs from baseline -- architecture must be fixed in E1",
            )

    def test_training_hyperparameters_identical_to_baseline(self):
        for key in CONTROLLED_TRAINING_KEYS:
            self.assertEqual(
                self.e1["training"][key], self.baseline["training"][key],
                f"training.{key} differs from baseline -- must be fixed in E1",
            )

    def test_preprocessing_identical_to_baseline(self):
        for key in CONTROLLED_PREPROCESSING_KEYS:
            self.assertEqual(
                self.e1["preprocessing"][key], self.baseline["preprocessing"][key],
                f"preprocessing.{key} differs from baseline -- must be fixed in E1",
            )

    def test_data_split_and_target_identical_to_baseline(self):
        for key in CONTROLLED_DATA_KEYS:
            self.assertEqual(
                self.e1["data"][key], self.baseline["data"][key],
                f"data.{key} differs from baseline -- split/target must be fixed in E1",
            )

    def test_seed_is_42_in_both_arms(self):
        self.assertEqual(self.e1["training"]["seed"], 42)
        self.assertEqual(self.baseline["training"]["seed"], 42)

    def test_max_epochs_is_100(self):
        self.assertEqual(self.e1["training"]["num_epochs"], 100)

    def test_amp_off_in_both_arms(self):
        self.assertFalse(self.e1["training"].get("amp", False), "E1 must train in FP32 (amp: false)")
        self.assertFalse(
            self.baseline["training"].get("amp", False),
            "baseline is FP32; E1 must match it for a controlled comparison",
        )

    def test_no_scheduler_or_augmentation_introduced(self):
        """E2/E3 are not approved yet -- E1 must not smuggle them in."""
        self.assertNotIn("augmentation", self.e1)
        self.assertNotIn("lr_scheduler", self.e1.get("training", {}))
        self.assertNotIn("scheduler", self.e1.get("training", {}))
        self.assertNotIn("early_stopping", self.e1.get("training", {}))

    def test_no_test_split_referenced(self):
        data = self.e1.get("data", {})
        self.assertNotIn("test_csv", data, "E1 must NOT reference the official test split")
        self.assertNotIn("test_dir", data, "E1 must NOT reference the official test split")

    def test_output_dir_isolated_from_baseline(self):
        baseline_out = self.baseline["experiment"]["output_dir"].replace("\\", "/").rstrip("/")
        e1_out = self.e1["experiment"]["output_dir"].replace("\\", "/").rstrip("/")
        self.assertNotEqual(e1_out, baseline_out, "E1 output_dir collides with the baseline")
        self.assertNotIn("exp01", e1_out, "E1 must not write into the baseline result tree")
        self.assertIn("exp_e1", e1_out)

    def test_experiment_name_is_distinct(self):
        self.assertEqual(self.e1["experiment"]["name"], "exp_e1_dice_ce")
        self.assertNotEqual(
            self.e1["experiment"]["name"], self.baseline["experiment"]["name"]
        )

    # ---- the independent variable itself ----

    def test_loss_is_the_only_independent_variable(self):
        from src.losses import LossConfig

        baseline_loss = LossConfig.from_config(self.baseline)
        e1_loss = LossConfig.from_config(self.e1)

        self.assertTrue(baseline_loss.is_baseline_cross_entropy, "baseline must be plain CE")
        self.assertEqual(e1_loss.name, "dice_ce", "E1 must use the compound Dice+CE loss")

    def test_e1_loss_parameters_match_the_approved_specification(self):
        from src.losses import LossConfig

        cfg = LossConfig.from_config(self.e1)
        self.assertEqual(cfg.ce_weight, 1.0)
        self.assertEqual(cfg.dice_weight, 1.0)
        self.assertEqual(cfg.dice_classes, (1, 2), "Dice must cover foreground CG(1) and PZ(2)")
        self.assertNotIn(0, cfg.dice_classes, "background must be excluded from the Dice term")
        self.assertAlmostEqual(cfg.epsilon, 1e-6)

    def test_e1_has_no_conflicting_legacy_loss_key(self):
        self.assertNotIn(
            "loss_function", self.e1["training"],
            "E1 must declare its loss ONLY in the top-level `loss:` block "
            "(two sources of truth are rejected at runtime)",
        )

    def test_baseline_config_still_resolves_to_cross_entropy(self):
        """Regression guard: the E1 work must not have altered the baseline."""
        from src.losses import LossConfig

        self.assertTrue(LossConfig.from_config(self.baseline).is_baseline_cross_entropy)
        self.assertEqual(self.baseline["training"]["loss_function"], "cross_entropy")


class TestCheckpointCompatibilityGuard(unittest.TestCase):
    """An E1 run must refuse to continue from a baseline checkpoint."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")
        self.baseline = load_config(BASELINE_PATH)
        self.e1 = load_config(E1_PATH)

    def test_e1_refuses_baseline_checkpoint(self):
        from src.train import verify_checkpoint_compatibility

        with self.assertRaises(ValueError) as ctx:
            verify_checkpoint_compatibility(self.baseline, self.e1, checkpoint_path="baseline/best_model.pt")
        message = str(ctx.exception)
        self.assertIn("DIFFERENT experiment", message)
        self.assertIn("experiment.name", message)
        self.assertIn("loss", message)

    def test_e1_accepts_its_own_checkpoint(self):
        from src.train import verify_checkpoint_compatibility

        result = verify_checkpoint_compatibility(self.e1, self.e1, checkpoint_path="e1/latest_checkpoint.pt")
        self.assertTrue(result["verified"])
        self.assertEqual(result["mismatches"], [])

    def test_legacy_checkpoint_without_experiment_block_is_skipped(self):
        """Older checkpoints stay loadable; verification degrades gracefully."""
        from src.train import verify_checkpoint_compatibility

        result = verify_checkpoint_compatibility(
            {"note": "epoch-60-format", "model": {"init_features": 4}}, self.e1
        )
        self.assertFalse(result["verified"])
        self.assertIsNotNone(result["skipped_reason"])

    def test_differing_loss_alone_is_rejected(self):
        """Same experiment name but a different loss must still be refused."""
        import copy

        from src.train import verify_checkpoint_compatibility

        tampered = copy.deepcopy(self.e1)
        tampered["loss"]["dice_weight"] = 0.5

        with self.assertRaises(ValueError) as ctx:
            verify_checkpoint_compatibility(tampered, self.e1)
        self.assertIn("loss", str(ctx.exception))

    def test_epoch_budget_change_is_allowed(self):
        """Extending an interrupted run's epoch budget is legitimate recovery."""
        import copy

        from src.train import verify_checkpoint_compatibility

        extended = copy.deepcopy(self.e1)
        extended["training"]["num_epochs"] = 50  # checkpoint was made with a smaller budget

        result = verify_checkpoint_compatibility(extended, self.e1)
        self.assertTrue(result["verified"])


class TestConfigFingerprint(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")

    def test_fingerprint_reports_e1_facts(self):
        from src.utils import build_config_fingerprint, format_config_fingerprint

        fingerprint = build_config_fingerprint(load_config(E1_PATH), device="cuda")
        self.assertEqual(fingerprint["experiment"], "exp_e1_dice_ce")
        self.assertEqual(fingerprint["precision"], "FP32")
        self.assertFalse(fingerprint["amp"])
        self.assertEqual(fingerprint["seed"], 42)
        self.assertEqual(fingerprint["num_epochs"], 100)
        self.assertEqual(fingerprint["dice_classes"], [1, 2])
        self.assertFalse(fingerprint["references_test_split"])

        rendered = format_config_fingerprint(fingerprint)
        for expected in ("EXPERIMENT CONFIGURATION FINGERPRINT", "exp_e1_dice_ce",
                         "FP32", "SoftDice", "Augmentation", "Scheduler"):
            self.assertIn(expected, rendered)

    def test_fingerprint_reports_baseline_as_cross_entropy(self):
        from src.utils import build_config_fingerprint

        fingerprint = build_config_fingerprint(load_config(BASELINE_PATH))
        self.assertIn("Cross Entropy", fingerprint["loss"])
        self.assertIsNone(fingerprint["dice_classes"])
        self.assertEqual(fingerprint["precision"], "FP32")


if __name__ == "__main__":
    unittest.main()
