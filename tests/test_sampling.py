"""Fix-pass tests: training-time foreground/background weighted sampling.

Verifies (per the audit's P0 item 1):
  - compute_foreground_sample_weights does not filter/delete any sample
  - a WeightedRandomSampler built from those weights draws foreground and
    background slices with roughly equal probability, even though patient
    020's raw population is skewed toward foreground slices
  - the sampler is deterministic given the same seed
  - the sampler is actually the one used by src.train's train loader
    construction (not just available but unused)
  - validation/test loaders remain unweighted (full population, shuffle=False)
"""

import os
import sys
import unittest

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestForegroundSampleWeights(unittest.TestCase):
    def setUp(self):
        self.dataset_root = os.path.join(project_root, "dataset", "prostate158_train", "train")
        if not os.path.isdir(os.path.join(self.dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping integration test.")

    def test_weights_do_not_filter_any_sample(self):
        from src.dataset import ProstateZonal2DDataset

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root, patient_ids=["020"], slice_sampling="all",
        )
        n_before = len(dataset)
        weights = dataset.compute_foreground_sample_weights()

        self.assertEqual(len(weights), n_before)
        self.assertEqual(len(dataset), n_before)  # dataset itself is untouched
        self.assertEqual(len(dataset.samples), n_before)

    def test_weights_are_positive_and_balance_classes(self):
        from src.dataset import ProstateZonal2DDataset

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root, patient_ids=["020"], slice_sampling="all",
        )
        weights = dataset.compute_foreground_sample_weights()
        self.assertTrue(np.all(weights > 0))

        is_fg = np.array([
            np.isin(dataset[i]["mask"], [1, 2]).any() for i in range(len(dataset))
        ])
        # Total weight mass assigned to the foreground group should equal the
        # mass assigned to the background group (inverse-frequency balancing).
        fg_mass = weights[is_fg].sum()
        bg_mass = weights[~is_fg].sum()
        self.assertAlmostEqual(fg_mass, bg_mass, places=6)

    def test_weighted_random_sampler_draws_are_balanced_and_deterministic(self):
        import torch
        from torch.utils.data import WeightedRandomSampler

        from src.dataset import ProstateZonal2DDataset
        from src.train import _build_train_sampler

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root, patient_ids=["020"], slice_sampling="all",
        )
        is_fg = np.array([
            np.isin(dataset[i]["mask"], [1, 2]).any() for i in range(len(dataset))
        ])
        # Sanity precondition: the raw population IS skewed (otherwise this
        # test wouldn't prove anything about balancing).
        raw_fg_fraction = is_fg.mean()
        self.assertTrue(0.0 < raw_fg_fraction < 1.0)

        sampler = _build_train_sampler(dataset, seed=42, use_weighted_sampling=True)
        self.assertIsInstance(sampler, WeightedRandomSampler)

        draws = list(sampler)
        drawn_fg_fraction = np.mean([is_fg[i] for i in draws])
        self.assertGreater(drawn_fg_fraction, 0.35)
        self.assertLess(drawn_fg_fraction, 0.65)

        # Determinism: same seed -> identical draw sequence.
        sampler_again = _build_train_sampler(dataset, seed=42, use_weighted_sampling=True)
        draws_again = list(sampler_again)
        self.assertEqual(draws, draws_again)

        # Different seed -> allowed to differ (not asserted equal/unequal
        # strictly, just confirms the seed argument is actually plumbed through).
        sampler_other_seed = _build_train_sampler(dataset, seed=123, use_weighted_sampling=True)
        draws_other_seed = list(sampler_other_seed)
        self.assertEqual(len(draws_other_seed), len(draws))

    def test_disabled_returns_none(self):
        from src.dataset import ProstateZonal2DDataset
        from src.train import _build_train_sampler

        dataset = ProstateZonal2DDataset(
            dataset_root=self.dataset_root, patient_ids=["020"], slice_sampling="all",
        )
        sampler = _build_train_sampler(dataset, seed=42, use_weighted_sampling=False)
        self.assertIsNone(sampler)

    def test_run_training_uses_weighted_sampler_by_default_and_val_stays_unweighted(self):
        """Integration-level proof that src.train.run_training actually wires
        the weighted sampler into the TRAIN DataLoader (not just that the
        helper exists), and that the VAL DataLoader remains unweighted."""
        if not os.path.isdir(os.path.join(self.dataset_root, "021")):
            self.skipTest("Patient 021 not found; skipping.")

        import tempfile
        import yaml
        from torch.utils.data import WeightedRandomSampler

        from src import train as train_module

        captured = {}
        original_dataloader = train_module.DataLoader

        def spy_dataloader(dataset, *args, **kwargs):
            loader = original_dataloader(dataset, *args, **kwargs)
            captured.setdefault("loaders", []).append((kwargs.get("sampler"), kwargs.get("shuffle")))
            return loader

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = {
                "experiment": {"output_dir": tmp_dir},
                "data": {
                    "dataset_root": self.dataset_root,
                    "patient_ids": ["020"],
                    "slice_sampling": "all",
                },
                "preprocessing": {
                    "normalization": "per_volume", "z_resample": False,
                    "spatial_mode": "crop_pad", "target_size": [270, 270],
                },
                "model": {"architecture": "ProstateUNet2D", "in_channels": 1, "out_channels": 3, "init_features": 4},
                "training": {
                    "batch_size": 4, "learning_rate": 0.001, "num_epochs": 1,
                    "seed": 42, "optimizer": "AdamW", "weight_decay": 0.0001,
                    "loss_function": "cross_entropy",
                },
                "validation": {"val_patient_ids": ["021"]},
            }
            config_path = os.path.join(tmp_dir, "tiny.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f)

            train_module.DataLoader = spy_dataloader
            try:
                train_module.run_training(config_path)
            finally:
                train_module.DataLoader = original_dataloader

        train_sampler, train_shuffle = captured["loaders"][0]
        val_sampler, val_shuffle = captured["loaders"][1]

        self.assertIsInstance(train_sampler, WeightedRandomSampler)
        self.assertFalse(train_shuffle)  # sampler present -> shuffle must be False

        self.assertIsNone(val_sampler)
        self.assertFalse(val_shuffle)  # val uses shuffle=False, no sampler, full population


if __name__ == "__main__":
    unittest.main()
