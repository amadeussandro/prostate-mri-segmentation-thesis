"""Unit tests for the configurable loss layer (src/losses.py, Experiment E1).

Two obligations are tested here:

  1. BASELINE PRESERVATION -- with the default / "cross_entropy" configuration
     the returned loss must be EXACTLY nn.CrossEntropyLoss()(logits, target),
     bit-for-bit, so the already-trained Experiment-1 baseline stays valid and
     reproducible. This is the most important test in this file.

  2. E1 CORRECTNESS -- the soft Dice term must behave like a Dice coefficient
     (perfect prediction -> 0 loss), must exclude background, must be finite and
     differentiable, and the compound loss must equal ce_weight*CE + dice_weight*Dice.
"""

import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestLossConfigResolution(unittest.TestCase):
    def test_default_is_cross_entropy(self):
        from src.losses import LossConfig

        cfg = LossConfig()
        self.assertEqual(cfg.name, "cross_entropy")
        self.assertTrue(cfg.is_baseline_cross_entropy)

    def test_empty_config_resolves_to_cross_entropy(self):
        from src.losses import LossConfig

        self.assertTrue(LossConfig.from_config({}).is_baseline_cross_entropy)
        self.assertTrue(LossConfig.from_config(None).is_baseline_cross_entropy)

    def test_legacy_training_loss_function_still_honored(self):
        """Every pre-existing config (baseline, Exp02, pilots) uses this spelling."""
        from src.losses import LossConfig

        cfg = LossConfig.from_config({"training": {"loss_function": "cross_entropy"}})
        self.assertTrue(cfg.is_baseline_cross_entropy)

    def test_top_level_loss_block_selects_dice_ce(self):
        from src.losses import LossConfig

        cfg = LossConfig.from_config({
            "loss": {
                "name": "dice_ce",
                "ce_weight": 1.0,
                "dice_weight": 1.0,
                "dice_classes": [1, 2],
                "epsilon": 1e-6,
            }
        })
        self.assertEqual(cfg.name, "dice_ce")
        self.assertEqual(cfg.dice_classes, (1, 2))
        self.assertEqual(cfg.ce_weight, 1.0)
        self.assertEqual(cfg.dice_weight, 1.0)
        self.assertAlmostEqual(cfg.epsilon, 1e-6)
        self.assertFalse(cfg.is_baseline_cross_entropy)

    def test_conflicting_sources_raise(self):
        """Two disagreeing sources of truth must be a hard error, not precedence."""
        from src.losses import LossConfig

        with self.assertRaises(ValueError) as ctx:
            LossConfig.from_config({
                "loss": {"name": "dice_ce"},
                "training": {"loss_function": "cross_entropy"},
            })
        self.assertIn("Conflicting loss configuration", str(ctx.exception))

    def test_agreeing_sources_are_allowed(self):
        from src.losses import LossConfig

        cfg = LossConfig.from_config({
            "loss": {"name": "ce"},
            "training": {"loss_function": "cross_entropy"},
        })
        self.assertTrue(cfg.is_baseline_cross_entropy)

    def test_unsupported_loss_raises(self):
        from src.losses import LossConfig

        with self.assertRaises(ValueError) as ctx:
            LossConfig(name="focal_tversky")
        self.assertIn("Unsupported loss_type", str(ctx.exception))

    def test_invalid_dice_settings_raise(self):
        from src.losses import LossConfig

        with self.assertRaises(ValueError):
            LossConfig(name="dice_ce", dice_classes=())
        with self.assertRaises(ValueError):
            LossConfig(name="dice_ce", dice_classes=(1, 1))
        with self.assertRaises(ValueError):
            LossConfig(name="dice_ce", ce_weight=0.0, dice_weight=0.0)
        with self.assertRaises(ValueError):
            LossConfig(name="dice_ce", epsilon=0.0)

    def test_to_dict_is_minimal_for_baseline(self):
        """A CE checkpoint's loss spec must not carry irrelevant dice fields."""
        from src.losses import LossConfig

        self.assertEqual(LossConfig().to_dict(), {"name": "cross_entropy"})


class TestBaselinePreservation(unittest.TestCase):
    """The default path must be numerically identical to the original baseline."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")

    def _fixture(self):
        import torch

        torch.manual_seed(0)
        logits = torch.randn(2, 3, 16, 16)
        targets = torch.randint(0, 3, (2, 16, 16))
        return logits, targets

    def test_cross_entropy_path_matches_nn_crossentropyloss_exactly(self):
        import torch
        from torch import nn

        from src.losses import LossConfig, compute_loss_components

        logits, targets = self._fixture()
        expected = nn.CrossEntropyLoss()(logits, targets.long())
        actual, components = compute_loss_components(logits, targets, LossConfig())

        self.assertTrue(torch.equal(actual, expected), "baseline CE path is not bit-identical")
        self.assertAlmostEqual(components["ce"], float(expected), places=12)
        self.assertEqual(components["dice"], 0.0)
        self.assertAlmostEqual(components["total"], float(expected), places=12)

    def test_train_compute_loss_default_matches_baseline(self):
        """src.train.compute_loss keeps its original contract for old callers."""
        import torch
        from torch import nn

        from src.train import compute_loss

        logits, targets = self._fixture()
        expected = nn.CrossEntropyLoss()(logits, targets.long())

        self.assertTrue(torch.equal(compute_loss(logits, targets), expected))
        self.assertTrue(
            torch.equal(compute_loss(logits, targets, loss_type="cross_entropy"), expected)
        )

    def test_train_compute_loss_rejects_unknown_loss(self):
        from src.train import compute_loss

        logits, targets = self._fixture()
        with self.assertRaises(ValueError):
            compute_loss(logits, targets, loss_type="not_a_loss")


class TestSoftDiceLoss(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")

    def test_perfect_prediction_gives_near_zero_loss(self):
        import torch

        from src.losses import soft_dice_loss

        targets = torch.zeros(1, 8, 8, dtype=torch.long)
        targets[0, :4, :] = 1
        targets[0, 4:, :] = 2

        # Very confident, perfectly correct logits.
        logits = torch.full((1, 3, 8, 8), -20.0)
        for c in range(3):
            logits[0, c][targets[0] == c] = 20.0

        loss = soft_dice_loss(logits, targets, class_ids=(1, 2), epsilon=1e-6)
        self.assertLess(float(loss), 1e-4)

    def test_completely_wrong_prediction_gives_near_one_loss(self):
        import torch

        from src.losses import soft_dice_loss

        targets = torch.ones(1, 8, 8, dtype=torch.long)  # all CG
        logits = torch.full((1, 3, 8, 8), -20.0)
        logits[0, 2] = 20.0  # predict all PZ

        loss = soft_dice_loss(logits, targets, class_ids=(1, 2), epsilon=1e-6)
        self.assertGreater(float(loss), 0.9)

    def test_background_is_excluded(self):
        """An all-background case must be dominated by the foreground terms only.

        With class_ids=(1,2) and neither class present in GT or prediction, both
        per-class Dice values collapse to eps/eps = 1, so the loss is ~0 -- the
        94.2%-background majority contributes nothing.
        """
        import torch

        from src.losses import soft_dice_loss

        targets = torch.zeros(1, 8, 8, dtype=torch.long)
        logits = torch.full((1, 3, 8, 8), -20.0)
        logits[0, 0] = 20.0  # confidently all background

        loss = soft_dice_loss(logits, targets, class_ids=(1, 2), epsilon=1e-6)
        self.assertLess(float(loss), 1e-3)

    def test_loss_is_finite_and_differentiable(self):
        import torch

        from src.losses import soft_dice_loss

        logits = torch.randn(2, 3, 8, 8, requires_grad=True)
        targets = torch.randint(0, 3, (2, 8, 8))

        loss = soft_dice_loss(logits, targets)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bounded_in_unit_interval(self):
        import torch

        from src.losses import soft_dice_loss

        for seed in range(5):
            torch.manual_seed(seed)
            logits = torch.randn(2, 3, 8, 8)
            targets = torch.randint(0, 3, (2, 8, 8))
            loss = float(soft_dice_loss(logits, targets))
            self.assertGreaterEqual(loss, -1e-6)
            self.assertLessEqual(loss, 1.0 + 1e-6)

    def test_out_of_range_class_id_raises(self):
        import torch

        from src.losses import soft_dice_loss

        logits = torch.randn(1, 3, 4, 4)
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        with self.assertRaises(ValueError):
            soft_dice_loss(logits, targets, class_ids=(3,))

    def test_rejects_non_4d_logits(self):
        import torch

        from src.losses import soft_dice_loss

        with self.assertRaises(ValueError):
            soft_dice_loss(torch.randn(3, 4, 4), torch.zeros(4, 4, dtype=torch.long))


class TestDiceCECompound(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")

    def test_total_equals_weighted_sum_of_terms(self):
        import torch
        from torch import nn

        from src.losses import LossConfig, compute_loss_components, soft_dice_loss

        torch.manual_seed(7)
        logits = torch.randn(2, 3, 16, 16)
        targets = torch.randint(0, 3, (2, 16, 16))

        cfg = LossConfig(name="dice_ce", ce_weight=1.0, dice_weight=1.0,
                         dice_classes=(1, 2), epsilon=1e-6)
        total, components = compute_loss_components(logits, targets, cfg)

        expected_ce = nn.CrossEntropyLoss()(logits, targets.long())
        expected_dice = soft_dice_loss(logits, targets, (1, 2), 1e-6)

        self.assertAlmostEqual(components["ce"], float(expected_ce), places=6)
        self.assertAlmostEqual(components["dice"], float(expected_dice), places=6)
        self.assertAlmostEqual(
            float(total), float(expected_ce) + float(expected_dice), places=6
        )

    def test_weights_are_applied(self):
        import torch

        from src.losses import LossConfig, compute_loss_components

        torch.manual_seed(7)
        logits = torch.randn(2, 3, 8, 8)
        targets = torch.randint(0, 3, (2, 8, 8))

        _, equal = compute_loss_components(
            logits, targets, LossConfig(name="dice_ce", ce_weight=1.0, dice_weight=1.0)
        )
        total_ce_only, _ = compute_loss_components(
            logits, targets, LossConfig(name="dice_ce", ce_weight=1.0, dice_weight=0.0)
        )
        self.assertAlmostEqual(float(total_ce_only), equal["ce"], places=6)

    def test_dice_ce_differs_from_plain_ce(self):
        """Sanity: E1 must actually change the objective (else it is not an experiment)."""
        import torch

        from src.losses import LossConfig, compute_loss_components

        torch.manual_seed(11)
        logits = torch.randn(2, 3, 16, 16)
        targets = torch.randint(0, 3, (2, 16, 16))

        ce_total, _ = compute_loss_components(logits, targets, LossConfig())
        dice_ce_total, _ = compute_loss_components(
            logits, targets, LossConfig(name="dice_ce")
        )
        self.assertNotAlmostEqual(float(ce_total), float(dice_ce_total), places=4)

    def test_backward_pass_updates_weights(self):
        import torch

        from src.losses import LossConfig, compute_loss_components
        from src.model import ProstateUNet2D

        torch.manual_seed(0)
        model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        images = torch.randn(2, 1, 32, 32)
        targets = torch.randint(0, 3, (2, 32, 32))

        before = model.final_conv.weight.detach().clone()
        loss, components = compute_loss_components(
            model(images), targets, LossConfig(name="dice_ce")
        )
        self.assertTrue(torch.isfinite(loss))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        self.assertFalse(torch.equal(before, model.final_conv.weight.detach()))
        self.assertGreater(components["dice"], 0.0)


if __name__ == "__main__":
    unittest.main()
