"""Scientific-control tests for the Experiment E3 (class-weighted Dice+CE) config.

E3's validity rests on ONE claim: **E3 is E1 plus median-frequency class weights
on the Cross Entropy term, and nothing else changed.** These tests enforce that
claim mechanically, so config drift cannot silently turn E3 into a multi-variable
experiment.

Verified here:
  * E3 is key-for-key identical to configs/experiments/exp_e1_dice_ce.yaml except
    `experiment.{name,description,output_dir}` and `loss.ce_class_weights`;
  * the class weights are exactly the approved [0.0456, 1.0, 2.8477] (bg, CG, PZ);
  * the Dice component is untouched (same weight, same classes, same epsilon);
  * augmentation stays OFF, no scheduler, no early stopping, FP32, seed 42;
  * E3 references no test split and writes to an isolated output_dir;
  * the CE weights actually reach `nn.CrossEntropyLoss(weight=...)` and PyTorch's
    DEFAULT weighted-mean normalization is used (no manual 1/N rescaling);
  * `ce_class_weights: None` leaves Baseline / E1 / E2 bit-for-bit unchanged;
  * the checkpoint-compatibility guard refuses to resume E3 from Baseline / E1 /
    E2 -- and refuses an E3 checkpoint whose class weights differ.
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
E2_PATH = os.path.join(CONFIG_DIR, "experiments", "exp_e2_ce_aug.yaml")
E3_PATH = os.path.join(CONFIG_DIR, "experiments", "exp_e3_dice_ce_pzweight.yaml")

# The one and only approved E3 weight vector (bg, CG, PZ).
APPROVED_CE_CLASS_WEIGHTS = (0.0456, 1.0000, 2.8477)

CONTROLLED_MODEL_KEYS = ("architecture", "in_channels", "out_channels", "init_features", "dropout_rate")
CONTROLLED_TRAINING_KEYS = ("batch_size", "learning_rate", "num_epochs", "seed", "optimizer", "weight_decay")
CONTROLLED_DATA_KEYS = ("dataset_root", "train_csv", "valid_csv", "target_mask", "slice_sampling")
CONTROLLED_PREPROCESSING_KEYS = ("normalization", "z_resample", "spatial_mode", "target_size")


def _flatten(mapping, prefix=""):
    """Flatten a nested config into {dotted.key: scalar} for exhaustive diffing."""
    flat = {}
    for key, value in mapping.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, dotted + "."))
        else:
            flat[dotted] = value
    return flat


class TestE3ScientificControl(unittest.TestCase):
    def setUp(self):
        self.baseline = load_config(BASELINE_PATH)
        self.e1 = load_config(E1_PATH)
        self.e3 = load_config(E3_PATH)

    def test_e3_config_exists_and_parses(self):
        self.assertTrue(os.path.isfile(E3_PATH), "E3 config file is missing")
        for block in ("experiment", "data", "preprocessing", "model", "training", "loss"):
            self.assertIn(block, self.e3, f"E3 config missing `{block}` block")

    def test_e3_differs_from_e1_in_exactly_the_approved_keys(self):
        """The decisive single-factor test: an exhaustive key-by-key diff vs E1."""
        allowed = {
            "experiment.name",
            "experiment.description",
            "experiment.output_dir",
            "loss.ce_class_weights",
        }
        flat_e1, flat_e3 = _flatten(self.e1), _flatten(self.e3)

        differing = {
            key for key in set(flat_e1) | set(flat_e3)
            if flat_e1.get(key, "<absent>") != flat_e3.get(key, "<absent>")
        }
        unexpected = differing - allowed
        self.assertEqual(
            unexpected, set(),
            "E3 deviates from E1 in key(s) that are NOT the approved independent "
            f"variable: {sorted(unexpected)}. E3 must be E1 + ce_class_weights only.",
        )
        self.assertIn(
            "loss.ce_class_weights", differing,
            "E3 does not actually differ from E1 -- the independent variable is missing.",
        )

    def test_model_identical_to_e1_and_baseline(self):
        for key in CONTROLLED_MODEL_KEYS:
            self.assertEqual(self.e3["model"][key], self.e1["model"][key], f"model.{key} differs from E1")
            self.assertEqual(self.e3["model"][key], self.baseline["model"][key], f"model.{key} differs from baseline")

    def test_training_hyperparameters_identical_to_e1_and_baseline(self):
        for key in CONTROLLED_TRAINING_KEYS:
            self.assertEqual(self.e3["training"][key], self.e1["training"][key], f"training.{key} differs from E1")
            self.assertEqual(
                self.e3["training"][key], self.baseline["training"][key],
                f"training.{key} differs from baseline",
            )

    def test_preprocessing_identical_to_e1(self):
        for key in CONTROLLED_PREPROCESSING_KEYS:
            self.assertEqual(self.e3["preprocessing"][key], self.e1["preprocessing"][key],
                             f"preprocessing.{key} differs from E1")

    def test_data_split_and_target_identical_to_e1(self):
        for key in CONTROLLED_DATA_KEYS:
            self.assertEqual(self.e3["data"][key], self.e1["data"][key], f"data.{key} differs from E1")

    def test_seed_is_42(self):
        self.assertEqual(self.e3["training"]["seed"], 42)

    def test_max_epochs_is_100(self):
        self.assertEqual(self.e3["training"]["num_epochs"], 100)

    def test_amp_off_fp32(self):
        self.assertFalse(self.e3["training"].get("amp", False), "E3 must train in FP32 (amp: false)")

    def test_augmentation_is_off(self):
        """E3 isolates class weighting; E2's augmentation factor must NOT be on."""
        self.assertNotIn("augmentation", self.e3, "E3 must not enable augmentation (that is E2's factor)")
        from src.augment import AugmentationConfig

        self.assertFalse(AugmentationConfig.from_config(self.e3).enabled)

    def test_no_scheduler_post_processing_or_early_stopping(self):
        training = self.e3.get("training", {})
        for forbidden in ("lr_scheduler", "scheduler", "early_stopping"):
            self.assertNotIn(forbidden, training, f"E3 must not introduce `{forbidden}`")
        for forbidden in ("post_processing", "postprocessing", "tta"):
            self.assertNotIn(forbidden, self.e3, f"E3 must not introduce `{forbidden}`")

    def test_no_test_split_referenced(self):
        data = self.e3.get("data", {})
        self.assertNotIn("test_csv", data, "E3 must NOT reference the official test split")
        self.assertNotIn("test_dir", data, "E3 must NOT reference the official test split")

    def test_output_dir_isolated(self):
        e3_out = self.e3["experiment"]["output_dir"].replace("\\", "/").rstrip("/")
        for other in (self.baseline, self.e1, load_config(E2_PATH)):
            other_out = other["experiment"]["output_dir"].replace("\\", "/").rstrip("/")
            self.assertNotEqual(e3_out, other_out, f"E3 output_dir collides with {other_out}")
        self.assertIn("exp_e3", e3_out)

    def test_experiment_name_is_distinct(self):
        self.assertEqual(self.e3["experiment"]["name"], "exp_e3_dice_ce_pzweight")

    def test_e3_has_no_conflicting_legacy_loss_key(self):
        self.assertNotIn(
            "loss_function", self.e3["training"],
            "E3 must declare its loss ONLY in the top-level `loss:` block",
        )

    # ---- the independent variable itself ----

    def test_class_weights_are_exactly_the_approved_vector(self):
        from src.losses import LossConfig

        cfg = LossConfig.from_config(self.e3)
        self.assertIsNotNone(cfg.ce_class_weights, "E3 must declare `loss.ce_class_weights`")
        self.assertEqual(len(cfg.ce_class_weights), 3, "one weight per class (bg, CG, PZ)")
        for index, expected in enumerate(APPROVED_CE_CLASS_WEIGHTS):
            self.assertAlmostEqual(
                cfg.ce_class_weights[index], expected, places=6,
                msg=f"ce_class_weights[{index}] drifted from the pre-committed value",
            )
        # Direction of the intervention: PZ up-weighted, background down-weighted.
        self.assertGreater(cfg.ce_class_weights[2], cfg.ce_class_weights[1], "PZ must outweigh CG")
        self.assertLess(cfg.ce_class_weights[0], cfg.ce_class_weights[1], "background must be down-weighted")

    def test_dice_component_is_byte_for_byte_e1(self):
        from src.losses import LossConfig

        e1_cfg = LossConfig.from_config(self.e1)
        e3_cfg = LossConfig.from_config(self.e3)
        self.assertEqual(e3_cfg.name, "dice_ce")
        self.assertEqual(e3_cfg.ce_weight, e1_cfg.ce_weight)
        self.assertEqual(e3_cfg.dice_weight, e1_cfg.dice_weight)
        self.assertEqual(e3_cfg.dice_classes, e1_cfg.dice_classes)
        self.assertEqual(e3_cfg.epsilon, e1_cfg.epsilon)
        self.assertEqual(e3_cfg.dice_classes, (1, 2))

    def test_e1_and_baseline_remain_unweighted(self):
        """Regression guard: the E3 work must not have altered the earlier arms."""
        from src.losses import LossConfig

        for name, config in (("baseline", self.baseline), ("E1", self.e1), ("E2", load_config(E2_PATH))):
            cfg = LossConfig.from_config(config)
            self.assertIsNone(cfg.ce_class_weights, f"{name} must stay an UNWEIGHTED CE")
            self.assertFalse(cfg.has_ce_class_weights)
        self.assertTrue(LossConfig.from_config(self.baseline).is_baseline_cross_entropy)


class TestE3LossNumerics(unittest.TestCase):
    """The weights must actually reach CrossEntropyLoss, with PyTorch's own mean."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")
        self.e1 = load_config(E1_PATH)
        self.e3 = load_config(E3_PATH)

    def _probe(self):
        """Deliberately class-imbalanced synthetic slices, like the real data."""
        import torch

        torch.manual_seed(1234)
        logits = torch.randn(4, 3, 16, 16, dtype=torch.float32)
        target = torch.zeros(4, 16, 16, dtype=torch.long)
        target[:, 2:9, 2:9] = 1
        target[:, 10:13, 10:13] = 2
        return logits, target

    def test_e3_ce_equals_weighted_cross_entropy_loss(self):
        import torch
        from torch import nn

        from src.losses import LossConfig, compute_loss_components, soft_dice_loss

        logits, target = self._probe()
        cfg = LossConfig.from_config(self.e3)
        weight = torch.tensor(APPROVED_CE_CLASS_WEIGHTS, dtype=torch.float32)

        expected_ce = nn.CrossEntropyLoss(weight=weight, reduction="mean")(logits, target)
        expected_dice = soft_dice_loss(logits, target, class_ids=(1, 2), epsilon=1e-6)
        expected_total = 1.0 * expected_ce + 1.0 * expected_dice

        total, components = compute_loss_components(logits, target, cfg)
        self.assertTrue(torch.equal(total, expected_total), "E3 total loss is not CE_weighted + SoftDice")
        self.assertAlmostEqual(components["ce"], float(expected_ce), places=6)
        self.assertAlmostEqual(components["dice"], float(expected_dice), places=6)

    def test_pytorch_default_weighted_mean_normalization_is_used(self):
        """sum_i w_{y_i} l_i / sum_i w_{y_i} -- NOT a manual 1/N rescaling."""
        import torch
        from torch import nn

        from src.losses import LossConfig, compute_loss_components

        logits, target = self._probe()
        cfg = LossConfig.from_config(self.e3)
        weight = torch.tensor(APPROVED_CE_CLASS_WEIGHTS, dtype=torch.float32)

        per_pixel = nn.CrossEntropyLoss(reduction="none")(logits, target)
        weight_map = weight[target]
        weighted_mean = (weight_map * per_pixel).sum() / weight_map.sum()
        plain_1_over_n = (weight_map * per_pixel).mean()

        _, components = compute_loss_components(logits, target, cfg)
        self.assertAlmostEqual(components["ce"], float(weighted_mean), places=6)
        self.assertNotAlmostEqual(
            components["ce"], float(plain_1_over_n), places=4,
            msg="the CE term looks 1/N-normalized; that would change the CE:Dice ratio",
        )

    def test_weighting_changes_ce_but_not_the_dice_term(self):
        from src.losses import LossConfig, compute_loss_components

        logits, target = self._probe()
        _, e1_components = compute_loss_components(logits, target, LossConfig.from_config(self.e1))
        _, e3_components = compute_loss_components(logits, target, LossConfig.from_config(self.e3))

        self.assertNotAlmostEqual(
            e1_components["ce"], e3_components["ce"], places=5,
            msg="the class weights had no effect on the CE term",
        )
        self.assertAlmostEqual(
            e1_components["dice"], e3_components["dice"], places=12,
            msg="E3 must not perturb the Dice term -- that would be a second factor",
        )

    def test_unweighted_path_is_bit_identical_to_the_pre_e3_behaviour(self):
        import torch
        from torch import nn

        from src.losses import LossConfig, compute_loss_components, soft_dice_loss

        logits, target = self._probe()

        baseline_total, _ = compute_loss_components(
            logits, target, LossConfig.from_config(load_config(BASELINE_PATH))
        )
        self.assertTrue(
            torch.equal(baseline_total, nn.CrossEntropyLoss()(logits, target)),
            "the baseline CE path is no longer bit-identical to nn.CrossEntropyLoss()",
        )

        e1_total, _ = compute_loss_components(logits, target, LossConfig.from_config(self.e1))
        expected_e1 = nn.CrossEntropyLoss()(logits, target) + soft_dice_loss(
            logits, target, class_ids=(1, 2), epsilon=1e-6
        )
        self.assertTrue(torch.equal(e1_total, expected_e1), "the E1 loss changed -- E1 is no longer reproducible")

    def test_gradients_are_finite_and_nonzero(self):
        import torch

        from src.losses import LossConfig, compute_loss_components

        logits, target = self._probe()
        logits = logits.clone().requires_grad_(True)
        total, _ = compute_loss_components(logits, target, LossConfig.from_config(self.e3))
        total.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_weight_tensor_matches_prediction_device(self):
        import torch

        from src.losses import build_ce_class_weight_tensor

        cpu_weight = build_ce_class_weight_tensor(APPROVED_CE_CLASS_WEIGHTS, 3, device=torch.device("cpu"))
        self.assertEqual(cpu_weight.dtype, torch.float32)
        for index, expected in enumerate(APPROVED_CE_CLASS_WEIGHTS):
            # float32 storage, so compare to float32 precision rather than exactly.
            self.assertAlmostEqual(cpu_weight[index].item(), expected, places=6)
        self.assertIsNone(build_ce_class_weight_tensor(None, 3))

        if torch.cuda.is_available():
            logits, target = self._probe()
            from src.losses import LossConfig, compute_loss_components

            cfg = LossConfig.from_config(self.e3)
            cpu_total, _ = compute_loss_components(logits, target, cfg)
            cuda_total, _ = compute_loss_components(logits.cuda(), target.cuda(), cfg)
            self.assertAlmostEqual(float(cpu_total), float(cuda_total), places=4)

    def test_class_count_mismatch_is_rejected(self):
        from src.losses import LossConfig, compute_loss_components

        logits, target = self._probe()
        with self.assertRaises(ValueError) as ctx:
            compute_loss_components(
                logits, target, LossConfig(name="dice_ce", ce_class_weights=[1.0, 2.0])
            )
        self.assertIn("ce_class_weights", str(ctx.exception))


class TestCeClassWeightValidation(unittest.TestCase):
    def test_rejects_negative_nan_and_all_zero_weights(self):
        from src.losses import LossConfig

        for bad in ([1.0, -1.0, 2.0], [0.0, 0.0, 0.0], [float("nan"), 1.0, 1.0],
                    [float("inf"), 1.0, 1.0], [], "1,2,3"):
            with self.assertRaises(ValueError, msg=f"accepted invalid weights {bad!r}"):
                LossConfig(name="dice_ce", ce_class_weights=bad)

    def test_accepts_the_approved_vector_and_normalizes_to_floats(self):
        from src.losses import LossConfig

        cfg = LossConfig(name="dice_ce", ce_class_weights=[0.0456, 1, 2.8477])
        self.assertIsInstance(cfg.ce_class_weights, tuple)
        self.assertTrue(all(isinstance(w, float) for w in cfg.ce_class_weights))

    def test_none_is_the_default(self):
        from src.losses import LossConfig

        self.assertIsNone(LossConfig().ce_class_weights)
        self.assertTrue(LossConfig().is_baseline_cross_entropy)

    def test_weighted_cross_entropy_is_not_the_baseline_path(self):
        from src.losses import LossConfig

        cfg = LossConfig(name="cross_entropy", ce_class_weights=[0.0456, 1.0, 2.8477])
        self.assertFalse(cfg.is_baseline_cross_entropy, "a weighted CE is a different objective")
        self.assertTrue(cfg.has_ce_class_weights)
        self.assertIn("ce_class_weights", cfg.to_dict())

    def test_to_dict_round_trips_through_from_config(self):
        from src.losses import LossConfig

        original = LossConfig.from_config(load_config(E3_PATH))
        restored = LossConfig.from_config({"loss": original.to_dict()})
        self.assertEqual(original.to_dict(), restored.to_dict())


class TestCheckpointCompatibilityGuard(unittest.TestCase):
    """An E3 run must refuse to continue from Baseline / E1 / E2 checkpoints."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")
        self.baseline = load_config(BASELINE_PATH)
        self.e1 = load_config(E1_PATH)
        self.e2 = load_config(E2_PATH)
        self.e3 = load_config(E3_PATH)

    def test_e3_refuses_foreign_checkpoints(self):
        from src.train import verify_checkpoint_compatibility

        for name, foreign in (("baseline", self.baseline), ("E1", self.e1), ("E2", self.e2)):
            with self.assertRaises(ValueError, msg=f"E3 accepted the {name} checkpoint") as ctx:
                verify_checkpoint_compatibility(foreign, self.e3, checkpoint_path=f"{name}/best_model.pt")
            message = str(ctx.exception)
            self.assertIn("DIFFERENT experiment", message)
            self.assertIn("experiment.name", message)

    def test_e3_refuses_an_e1_checkpoint_on_the_loss_spec_alone(self):
        """Even with the name forced to match, differing class weights must block."""
        import copy

        from src.train import verify_checkpoint_compatibility

        disguised = copy.deepcopy(self.e1)
        disguised["experiment"]["name"] = self.e3["experiment"]["name"]

        with self.assertRaises(ValueError) as ctx:
            verify_checkpoint_compatibility(disguised, self.e3)
        self.assertIn("loss", str(ctx.exception))
        self.assertIn("ce_class_weights", str(ctx.exception))

    def test_e3_accepts_its_own_checkpoint(self):
        from src.train import verify_checkpoint_compatibility

        result = verify_checkpoint_compatibility(self.e3, self.e3, checkpoint_path="e3/latest_checkpoint.pt")
        self.assertTrue(result["verified"])
        self.assertEqual(result["mismatches"], [])

    def test_changed_class_weights_alone_are_rejected(self):
        import copy

        from src.train import verify_checkpoint_compatibility

        tampered = copy.deepcopy(self.e3)
        tampered["loss"]["ce_class_weights"] = [0.0456, 1.0, 5.0]

        with self.assertRaises(ValueError) as ctx:
            verify_checkpoint_compatibility(tampered, self.e3)
        self.assertIn("ce_class_weights", str(ctx.exception))

    def test_epoch_budget_change_is_allowed(self):
        import copy

        from src.train import verify_checkpoint_compatibility

        extended = copy.deepcopy(self.e3)
        extended["training"]["num_epochs"] = 50

        self.assertTrue(verify_checkpoint_compatibility(extended, self.e3)["verified"])


class TestConfigFingerprint(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")

    def test_fingerprint_reports_the_e3_factor(self):
        from src.utils import build_config_fingerprint, format_config_fingerprint

        fingerprint = build_config_fingerprint(load_config(E3_PATH), device="cuda")
        self.assertEqual(fingerprint["experiment"], "exp_e3_dice_ce_pzweight")
        self.assertEqual(fingerprint["precision"], "FP32")
        self.assertFalse(fingerprint["amp"])
        self.assertEqual(fingerprint["seed"], 42)
        self.assertEqual(fingerprint["num_epochs"], 100)
        self.assertEqual(fingerprint["dice_classes"], [1, 2])
        self.assertFalse(fingerprint["augmentation_enabled"])
        self.assertFalse(fingerprint["references_test_split"])
        self.assertEqual(len(fingerprint["ce_class_weights"]), 3)
        for index, expected in enumerate(APPROVED_CE_CLASS_WEIGHTS):
            self.assertAlmostEqual(fingerprint["ce_class_weights"][index], expected, places=6)

        rendered = format_config_fingerprint(fingerprint)
        for expected in ("exp_e3_dice_ce_pzweight", "CE class weights", "2.8477", "FP32", "SoftDice"):
            self.assertIn(expected, rendered)

    def test_earlier_arms_report_unweighted_ce(self):
        from src.utils import build_config_fingerprint, format_config_fingerprint

        for path in (BASELINE_PATH, E1_PATH, E2_PATH):
            fingerprint = build_config_fingerprint(load_config(path))
            self.assertIsNone(fingerprint["ce_class_weights"], f"{path} should be unweighted")
            self.assertIn("unweighted", format_config_fingerprint(fingerprint))


if __name__ == "__main__":
    unittest.main()
