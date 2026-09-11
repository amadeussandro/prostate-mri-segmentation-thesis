"""Fix-pass integration test (audit P1 item 8): run src.train.run_training
end-to-end for 1 epoch on a minimal real-data subset, CPU only.

This does not test model accuracy -- it proves the training loop itself is
wired correctly after the Phase-3 fix pass (official-split resolution,
target_mask config wiring, weighted sampler construction, checkpointing) by
actually running it, rather than only unit-testing its pieces in isolation.
"""

import os
import sys
import tempfile
import unittest

import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _tiny_config(dataset_root, output_dir, patient_ids, val_patient_ids=None, num_epochs=1):
    config = {
        "experiment": {"name": "integration_test", "output_dir": output_dir},
        "data": {
            "dataset_root": dataset_root,
            "patient_ids": patient_ids,
            "slice_sampling": "all",
        },
        "preprocessing": {
            "normalization": "per_volume",
            "z_resample": False,
            "spatial_mode": "crop_pad",
            "target_size": [270, 270],
        },
        "model": {
            "architecture": "ProstateUNet2D",
            "in_channels": 1,
            "out_channels": 3,
            "init_features": 4,  # tiny, CPU-fast
            "dropout_rate": 0.2,
        },
        "training": {
            "batch_size": 4,
            "learning_rate": 0.001,
            "num_epochs": num_epochs,
            "seed": 42,
            "optimizer": "AdamW",
            "weight_decay": 0.0001,
            "loss_function": "cross_entropy",
        },
        "validation": {"val_patient_ids": val_patient_ids or []},
    }
    return config


class TestRunTrainingIntegration(unittest.TestCase):
    def setUp(self):
        self.dataset_root = os.path.join(project_root, "dataset", "prostate158_train", "train")
        if not os.path.isdir(os.path.join(self.dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping integration test.")

    def test_run_training_without_validation_produces_checkpoint(self):
        import torch
        from src.train import run_training

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _tiny_config(self.dataset_root, tmp_dir, patient_ids=["020"])
            config_path = os.path.join(tmp_dir, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f)

            run_training(config_path)  # must not raise; CPU only (no CUDA required)

            checkpoint_path = os.path.join(tmp_dir, "best_model.pt")
            self.assertTrue(os.path.isfile(checkpoint_path), "Checkpoint was not created.")

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            for key in ("epoch", "model_state_dict", "optimizer_state_dict", "best_metric", "config"):
                self.assertIn(key, checkpoint)
            self.assertEqual(checkpoint["epoch"], 1)

    def test_run_training_with_validation_runs_validation_and_checkpoints(self):
        import torch
        from src.train import run_training

        if not os.path.isdir(os.path.join(self.dataset_root, "021")):
            self.skipTest("Patient 021 not found; skipping validation-path integration test.")

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = _tiny_config(
                self.dataset_root, tmp_dir, patient_ids=["020"], val_patient_ids=["021"], num_epochs=2,
            )
            config_path = os.path.join(tmp_dir, "config.yaml")
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f)

            run_training(config_path)

            checkpoint_path = os.path.join(tmp_dir, "best_model.pt")
            self.assertTrue(os.path.isfile(checkpoint_path), "Checkpoint was not created.")

            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertIn("best_metric", checkpoint)
            self.assertTrue(1 <= checkpoint["epoch"] <= 2)
            # Best-metric selection with a validation set uses val_dice, which
            # is bounded in [0, 1] (unlike the no-val fallback, -train_loss).
            self.assertGreaterEqual(checkpoint["best_metric"], 0.0)
            self.assertLessEqual(checkpoint["best_metric"], 1.0)


if __name__ == "__main__":
    unittest.main()
