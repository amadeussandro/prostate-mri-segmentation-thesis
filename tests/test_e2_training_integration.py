"""End-to-end integration test for the E2 wiring: augmentation + resume safety.

Runs `src.train.run_training` for real on a minimal CPU-sized subset (tiny
model, two cases, 270x270), NOT to measure accuracy but to prove the two
things E2 depends on actually hold when the whole loop runs:

  1. AUGMENTATION IS WIRED TRAIN-ONLY -- the training dataset carries the
     seeded augmentor, the validation dataset carries nothing.
  2. HISTORY SURVIVES A RESUME -- a run stopped after epoch 2 and resumed to
     epoch 4 ends with epochs 1..4 in both `metrics.csv` and
     `training_history.json`, not 3..4.

Skipped automatically when the Prostate158 train archive is not present
locally, matching tests/test_train_integration.py.
"""

import csv
import json
import os
import sys
import tempfile
import unittest

import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.augment import SliceAugmentor  # noqa: E402
from src.train import _build_dataset, run_training  # noqa: E402

AUGMENTATION_BLOCK = {
    "enabled": True,
    "seed": 42,
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


def tiny_e2_config(dataset_root, output_dir, train_ids, val_ids, num_epochs):
    """A CPU-sized stand-in for configs/experiments/exp_e2_ce_aug.yaml: same
    loss, same optimizer family, same augmentation block; only the model width,
    the case count, and the epoch budget are shrunk so this runs in seconds."""
    return {
        "experiment": {"name": "e2_integration_test", "output_dir": output_dir},
        "data": {
            "dataset_root": dataset_root,
            "patient_ids": train_ids,
            "target_mask": "t2_anatomy_reader1.nii.gz",
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
            "init_features": 4,
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
            "amp": False,
            "num_workers": 0,
        },
        "augmentation": dict(AUGMENTATION_BLOCK),
        "validation": {"val_patient_ids": val_ids},
    }


class TestE2TrainingIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_root = os.path.join(project_root, "dataset", "prostate158_train", "train")
        cls.train_ids = ["020"]
        cls.val_ids = ["021"]
        for pid in cls.train_ids + cls.val_ids:
            if not os.path.isdir(os.path.join(cls.dataset_root, pid)):
                raise unittest.SkipTest(
                    f"Prostate158 case '{pid}' not available locally; skipping E2 integration test."
                )

    def _write_config(self, directory, num_epochs):
        config = tiny_e2_config(
            self.dataset_root,
            os.path.join(directory, "out"),
            self.train_ids,
            self.val_ids,
            num_epochs,
        )
        path = os.path.join(directory, "e2_integration.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        return path, config

    def test_augmentation_is_attached_to_train_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, config = self._write_config(tmp, num_epochs=1)

            train_dataset = _build_dataset(config, self.train_ids, augment=True)
            val_dataset = _build_dataset(config, self.val_ids, augment=False)

            self.assertIsInstance(
                train_dataset.transform, SliceAugmentor,
                "the training dataset must carry the seeded augmentor",
            )
            self.assertIsNone(
                val_dataset.transform,
                "the validation dataset must NEVER be augmented",
            )
            self.assertTrue(train_dataset.transform.config.enabled)
            self.assertEqual(train_dataset.transform.config.seed, 42)

    def test_augmentation_is_absent_without_the_config_block(self):
        """Baseline / E1 preservation, checked through the real builder."""
        with tempfile.TemporaryDirectory() as tmp:
            _, config = self._write_config(tmp, num_epochs=1)
            config.pop("augmentation")
            dataset = _build_dataset(config, self.train_ids, augment=True)
            self.assertIsNone(dataset.transform)

    def test_augmented_training_samples_differ_from_unaugmented_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, config = self._write_config(tmp, num_epochs=1)
            plain = _build_dataset(config, self.train_ids, augment=False)
            augmented = _build_dataset(config, self.train_ids, augment=True)

            index = len(plain) // 2
            reference = plain[index]
            differs = False
            for _ in range(10):
                candidate = augmented[index]
                self.assertEqual(candidate["image"].shape, reference["image"].shape)
                self.assertEqual(candidate["mask"].shape, reference["mask"].shape)
                self.assertTrue(set(candidate["mask"].ravel().tolist()).issubset({0, 1, 2}))
                if not (candidate["image"] == reference["image"]).all():
                    differs = True
            self.assertTrue(differs, "augmentation never changed a single training sample")

    def test_run_training_refuses_num_workers_above_zero_with_augmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config = self._write_config(tmp, num_epochs=1)
            config["training"]["num_workers"] = 2
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            with self.assertRaises(ValueError) as ctx:
                run_training(path)
            self.assertIn("num_workers", str(ctx.exception))

    def test_resume_preserves_the_full_history(self):
        """The headline regression test: stop at epoch 2, resume to epoch 4,
        and require epochs 1..4 in both artifacts."""
        with tempfile.TemporaryDirectory() as tmp:
            path, config = self._write_config(tmp, num_epochs=2)
            output_dir = config["experiment"]["output_dir"]
            history_path = os.path.join(output_dir, "training_history.json")
            metrics_path = os.path.join(output_dir, "metrics.csv")

            first = run_training(path, history_path=history_path)
            self.assertEqual(first["epochs_recorded"], 2)
            self.assertFalse(first["resumed"])
            self.assertTrue(first["augmentation"]["enabled"])
            self.assertIn(first["best_epoch"], (1, 2))

            # Extend the budget and resume from latest_checkpoint.pt, exactly as
            # the Colab notebook does after a disconnect.
            config["training"]["num_epochs"] = 4
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            second = run_training(
                path,
                resume_from=os.path.join(output_dir, "latest_checkpoint.pt"),
                history_path=history_path,
            )
            self.assertTrue(second["resumed"])
            self.assertEqual(second["start_epoch"], 3)
            self.assertEqual(
                second["epochs_recorded"], 4,
                "the resumed run must end with 4 epoch records, not just the 2 it ran",
            )

            with open(metrics_path, newline="", encoding="utf-8") as f:
                epochs = [int(row["epoch"]) for row in csv.DictReader(f)]
            self.assertEqual(epochs, [1, 2, 3, 4], "metrics.csv lost earlier epochs on resume")

            with open(history_path, encoding="utf-8") as f:
                payload = json.load(f)
            self.assertEqual([r["epoch"] for r in payload["epochs"]], [1, 2, 3, 4])
            self.assertTrue(payload["augmentation"]["enabled"])
            self.assertIsNotNone(payload["best_epoch"])

    def test_fresh_run_refuses_to_overwrite_an_existing_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config = self._write_config(tmp, num_epochs=1)
            history_path = os.path.join(config["experiment"]["output_dir"], "training_history.json")

            run_training(path, history_path=history_path)
            with self.assertRaises(ValueError) as ctx:
                run_training(path, history_path=history_path)
            self.assertIn("already contains training history", str(ctx.exception))

    def test_resume_refuses_a_checkpoint_with_augmentation_switched_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, config = self._write_config(tmp, num_epochs=1)
            output_dir = config["experiment"]["output_dir"]
            history_path = os.path.join(output_dir, "training_history.json")
            run_training(path, history_path=history_path)

            config["training"]["num_epochs"] = 3
            config["augmentation"]["enabled"] = False
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            with self.assertRaises(ValueError) as ctx:
                run_training(
                    path,
                    resume_from=os.path.join(output_dir, "latest_checkpoint.pt"),
                    history_path=history_path,
                )
            self.assertIn("augmentation", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
