"""Scientific-control tests for the Experiment E2 (CE + augmentation) config.

E2's validity rests on ONE claim: augmentation is the only thing that changed
relative to the baseline. These tests enforce that claim mechanically, so
config drift cannot silently turn E2 into a multi-variable experiment.

Verified here:
  * every controlled key (model, optimizer, LR, weight decay, batch size,
    epochs, seed, preprocessing, split, target mask, slice sampling) is
    IDENTICAL to configs/config_baseline.yaml;
  * the LOSS is unchanged (plain Cross Entropy, same as baseline) -- E2 must
    NOT inherit E1's Dice+CE, or it would confound the two factors;
  * augmentation really is the independent variable (baseline OFF, E2 ON) and
    its parameters match the pre-committed specification in
    Outputs/rq1_experiment_specification.md section 6;
  * AMP is OFF and precision is FP32 in both arms;
  * E2 references no test split and writes to an isolated output_dir;
  * E2 pins num_workers to 0, which reproducible augmentation requires.

Also covers the checkpoint-compatibility guard that stops an E2 run from
resuming out of the Baseline's or E1's checkpoint (E2 must train FRESH).
"""

import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.augment import AugmentationConfig  # noqa: E402
from src.losses import CROSS_ENTROPY, LossConfig  # noqa: E402
from src.utils import build_config_fingerprint, load_config  # noqa: E402

CONFIG_DIR = os.path.join(project_root, "configs")
BASELINE_PATH = os.path.join(CONFIG_DIR, "config_baseline.yaml")
E1_PATH = os.path.join(CONFIG_DIR, "experiments", "exp_e1_dice_ce.yaml")
E2_PATH = os.path.join(CONFIG_DIR, "experiments", "exp_e2_ce_aug.yaml")

CONTROLLED_MODEL_KEYS = ("architecture", "in_channels", "out_channels", "init_features", "dropout_rate")
CONTROLLED_TRAINING_KEYS = ("batch_size", "learning_rate", "num_epochs", "seed", "optimizer", "weight_decay")
CONTROLLED_DATA_KEYS = ("dataset_root", "train_csv", "valid_csv", "target_mask", "slice_sampling")
CONTROLLED_PREPROCESSING_KEYS = ("normalization", "z_resample", "spatial_mode", "target_size")

# The pre-committed E2 augmentation specification, restated here independently
# of src/augment.py's defaults so a change to either one is caught.
SPEC = {
    "rotation_degrees": 10.0,
    "rotation_p": 0.5,
    "scale_min": 0.9,
    "scale_max": 1.1,
    "scale_p": 0.3,
    "hflip_p": 0.5,
    "gamma_min": 0.7,
    "gamma_max": 1.5,
    "gamma_p": 0.3,
}


class TestE2ScientificControl(unittest.TestCase):
    def setUp(self):
        self.baseline = load_config(BASELINE_PATH)
        self.e2 = load_config(E2_PATH)

    def test_e2_config_exists_and_parses(self):
        self.assertTrue(os.path.isfile(E2_PATH), "E2 config file is missing")
        for block in ("experiment", "data", "preprocessing", "model", "training", "augmentation"):
            self.assertIn(block, self.e2, f"E2 config missing `{block}` block")

    def test_model_identical_to_baseline(self):
        for key in CONTROLLED_MODEL_KEYS:
            self.assertEqual(
                self.e2["model"][key], self.baseline["model"][key],
                f"model.{key} differs from baseline -- architecture must be fixed in E2",
            )

    def test_training_hyperparameters_identical_to_baseline(self):
        for key in CONTROLLED_TRAINING_KEYS:
            self.assertEqual(
                self.e2["training"][key], self.baseline["training"][key],
                f"training.{key} differs from baseline -- must be fixed in E2",
            )

    def test_data_split_identical_to_baseline(self):
        for key in CONTROLLED_DATA_KEYS:
            self.assertEqual(
                self.e2["data"][key], self.baseline["data"][key],
                f"data.{key} differs from baseline -- the split must be fixed in E2",
            )

    def test_preprocessing_identical_to_baseline(self):
        for key in CONTROLLED_PREPROCESSING_KEYS:
            self.assertEqual(
                self.e2["preprocessing"][key], self.baseline["preprocessing"][key],
                f"preprocessing.{key} differs from baseline -- must be fixed in E2",
            )

    def test_loss_is_unchanged_baseline_cross_entropy(self):
        """E2 is the augmentation arm. Inheriting E1's Dice+CE would confound
        the two factors and make the 2x2 design unreadable."""
        baseline_loss = LossConfig.from_config(self.baseline)
        e2_loss = LossConfig.from_config(self.e2)
        self.assertEqual(e2_loss.name, CROSS_ENTROPY)
        self.assertTrue(e2_loss.is_baseline_cross_entropy)
        self.assertEqual(e2_loss.to_dict(), baseline_loss.to_dict())

    def test_e2_carries_no_top_level_loss_block(self):
        self.assertNotIn(
            "loss", self.e2,
            "E2 must not declare a `loss:` block -- it uses the baseline CE loss",
        )

    def test_no_scheduler_introduced(self):
        self.assertNotIn("lr_scheduler", self.e2.get("training", {}))
        self.assertNotIn("scheduler", self.e2)

    def test_amp_off_in_both_arms(self):
        self.assertFalse(self.baseline["training"].get("amp", False))
        self.assertFalse(self.e2["training"].get("amp", False))

    def test_isolated_output_dir(self):
        self.assertNotEqual(
            self.e2["experiment"]["output_dir"], self.baseline["experiment"]["output_dir"]
        )
        self.assertEqual(self.e2["experiment"]["name"], "exp_e2_ce_aug")

    def test_no_test_split_referenced(self):
        self.assertNotIn("test_csv", self.e2["data"])
        self.assertNotIn("test_dir", self.e2["data"])


class TestE2AugmentationIsTheIndependentVariable(unittest.TestCase):
    def setUp(self):
        self.baseline = load_config(BASELINE_PATH)
        self.e1 = load_config(E1_PATH)
        self.e2 = load_config(E2_PATH)

    def test_baseline_and_e1_have_no_augmentation(self):
        self.assertNotIn("augmentation", self.baseline)
        self.assertNotIn("augmentation", self.e1)
        self.assertFalse(AugmentationConfig.from_config(self.baseline).enabled)
        self.assertFalse(AugmentationConfig.from_config(self.e1).enabled)

    def test_e2_augmentation_is_enabled(self):
        self.assertTrue(AugmentationConfig.from_config(self.e2).enabled)

    def test_e2_augmentation_matches_the_precommitted_specification(self):
        cfg = AugmentationConfig.from_config(self.e2)
        for key, expected in SPEC.items():
            self.assertAlmostEqual(
                getattr(cfg, key), expected, places=9,
                msg=f"augmentation.{key} drifted from the pre-committed E2 specification",
            )

    def test_e2_augmentation_seed_is_42(self):
        self.assertEqual(AugmentationConfig.from_config(self.e2).seed, 42)
        self.assertEqual(self.e2["training"]["seed"], 42)

    def test_mask_interpolation_is_nearest_neighbour(self):
        spec = AugmentationConfig.from_config(self.e2).to_dict()
        self.assertEqual(spec["mask_interpolation_order"], 0)
        self.assertEqual(spec["image_interpolation_order"], 1)
        self.assertEqual(spec["applied_to"], "train")

    def test_num_workers_is_zero_for_reproducible_augmentation(self):
        self.assertEqual(
            self.e2["training"].get("num_workers", 0), 0,
            "augmented arms must run with num_workers=0 or the augmentation RNG "
            "is forked into workers and reproducibility is lost",
        )

    def test_fingerprint_reports_augmentation_on_for_e2_and_off_for_baseline(self):
        baseline_fp = build_config_fingerprint(self.baseline, device="cpu")
        e2_fp = build_config_fingerprint(self.e2, device="cpu")
        self.assertFalse(baseline_fp["augmentation_enabled"])
        self.assertTrue(e2_fp["augmentation_enabled"])
        self.assertEqual(e2_fp["precision"], "FP32")
        self.assertFalse(e2_fp["references_test_split"])

    def test_e2_differs_from_baseline_only_in_name_output_dir_and_augmentation(self):
        """Whole-config diff. The ONLY permitted differences are the isolation
        fields, the augmentation block, and explicit restatements of values the
        baseline leaves at their default."""
        permitted_top_level = {"experiment", "augmentation", "training"}
        differing = {
            key for key in set(self.baseline) | set(self.e2)
            if self.baseline.get(key) != self.e2.get(key)
        }
        self.assertTrue(
            differing.issubset(permitted_top_level),
            f"E2 differs from baseline in unexpected block(s): {sorted(differing - permitted_top_level)}",
        )

        # Within `training`, only defaults may be restated -- never changed.
        baseline_training = self.baseline["training"]
        e2_training = self.e2["training"]
        default_restatements = {"amp": False, "num_workers": 0}
        for key in set(baseline_training) | set(e2_training):
            if key in baseline_training and key in e2_training:
                self.assertEqual(
                    baseline_training[key], e2_training[key],
                    f"training.{key} differs between baseline and E2",
                )
            elif key in e2_training:
                self.assertIn(
                    key, default_restatements,
                    f"E2 adds training.{key}, which the baseline does not set",
                )
                self.assertEqual(e2_training[key], default_restatements[key])
            else:
                self.fail(f"E2 dropped training.{key}, which the baseline sets")


class TestE2CannotResumeFromAnotherArm(unittest.TestCase):
    """E2 must train from a FRESH initialization -- never from Baseline or E1."""

    def setUp(self):
        self.baseline = load_config(BASELINE_PATH)
        self.e1 = load_config(E1_PATH)
        self.e2 = load_config(E2_PATH)

    def test_refuses_baseline_checkpoint(self):
        from src.train import verify_checkpoint_compatibility

        with self.assertRaises(ValueError) as ctx:
            verify_checkpoint_compatibility(self.baseline, self.e2, "baseline/best_model.pt")
        message = str(ctx.exception)
        self.assertIn("experiment.name", message)
        self.assertIn("augmentation", message)

    def test_refuses_e1_checkpoint(self):
        from src.train import verify_checkpoint_compatibility

        with self.assertRaises(ValueError) as ctx:
            verify_checkpoint_compatibility(self.e1, self.e2, "e1/latest_checkpoint.pt")
        message = str(ctx.exception)
        self.assertIn("experiment.name", message)
        self.assertIn("loss", message)
        self.assertIn("augmentation", message)

    def test_accepts_its_own_checkpoint(self):
        from src.train import verify_checkpoint_compatibility

        result = verify_checkpoint_compatibility(self.e2, self.e2, "e2/latest_checkpoint.pt")
        self.assertTrue(result["verified"])
        self.assertEqual(result["mismatches"], [])

    def test_refuses_a_checkpoint_whose_augmentation_was_turned_off(self):
        """The exact failure mode this guard exists for: someone flips
        `enabled: false` and resumes, producing a half-augmented run."""
        import copy

        from src.train import verify_checkpoint_compatibility

        tampered = copy.deepcopy(self.e2)
        tampered["augmentation"]["enabled"] = False
        with self.assertRaises(ValueError) as ctx:
            verify_checkpoint_compatibility(tampered, self.e2, "e2/latest_checkpoint.pt")
        self.assertIn("augmentation", str(ctx.exception))

    def test_refuses_a_checkpoint_with_changed_augmentation_parameters(self):
        import copy

        from src.train import verify_checkpoint_compatibility

        tampered = copy.deepcopy(self.e2)
        tampered["augmentation"]["rotation_degrees"] = 25.0
        with self.assertRaises(ValueError) as ctx:
            verify_checkpoint_compatibility(tampered, self.e2, "e2/latest_checkpoint.pt")
        self.assertIn("augmentation", str(ctx.exception))

    def test_refuses_a_checkpoint_with_changed_preprocessing(self):
        import copy

        from src.train import verify_checkpoint_compatibility

        tampered = copy.deepcopy(self.e2)
        tampered["preprocessing"]["normalization"] = "global"
        with self.assertRaises(ValueError) as ctx:
            verify_checkpoint_compatibility(tampered, self.e2, "e2/latest_checkpoint.pt")
        self.assertIn("preprocessing.normalization", str(ctx.exception))

    def test_refuses_a_checkpoint_with_a_changed_split(self):
        import copy

        from src.train import verify_checkpoint_compatibility

        tampered = copy.deepcopy(self.e2)
        tampered["data"]["valid_csv"] = "dataset/prostate158_train/other_valid.csv"
        with self.assertRaises(ValueError) as ctx:
            verify_checkpoint_compatibility(tampered, self.e2, "e2/latest_checkpoint.pt")
        self.assertIn("data.valid_csv", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
