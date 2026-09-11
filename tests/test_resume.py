"""Tests for resume-from-checkpoint recovery in src/train.py.

Recovery-only feature: continue an interrupted Experiment-1 run from a saved
checkpoint's epoch + 1 without restarting at epoch 1 and without resetting
best_metric. The experiment design is unchanged.

Two groups:
  - Unit tests of `_load_resume_checkpoint` -- no dataset needed, fast.
  - Integration tests of `run_training(resume_from=...)` on a tiny single-patient
    config (patient 020, init_features=4, 1-2 epochs) -- skipped if data/torch
    are unavailable. These never run a full 100-epoch / 119-case run.
"""

import os
import sys
import tempfile
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _make_epoch60_style_checkpoint(path, epoch=60, best_metric=0.6496429678992481, init_features=4):
    """Write a checkpoint in the exact format run_training saves (the same
    5-key format as the real Experiment-1 epoch-60 best_model.pt)."""
    import torch

    from src.model import ProstateUNet2D

    model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=init_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    # One real optimizer step so optimizer_state_dict carries non-empty state.
    x = torch.randn(1, 1, 32, 32)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": {"note": "epoch-60-format", "model": {"init_features": init_features}},
        },
        path,
    )
    return model, optimizer


class TestLoadResumeCheckpointUnit(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")

    def test_start_epoch_and_best_metric_restored(self):
        import torch
        from src.model import ProstateUNet2D
        from src.train import _load_resume_checkpoint

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            _make_epoch60_style_checkpoint(ckpt, epoch=60, best_metric=0.6496429678992481)

            model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)

            start_epoch, best_metric, checkpoint_epoch = _load_resume_checkpoint(
                ckpt, model, optimizer, device="cpu", num_epochs=100
            )
            self.assertEqual(checkpoint_epoch, 60)
            self.assertEqual(start_epoch, 61)  # continues at 61, NOT 1
            self.assertAlmostEqual(best_metric, 0.6496429678992481, places=12)

    def test_model_weights_actually_restored(self):
        import torch
        from src.model import ProstateUNet2D
        from src.train import _load_resume_checkpoint

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            saved_model, _ = _make_epoch60_style_checkpoint(ckpt, epoch=60)
            saved_state = {k: v.clone() for k, v in saved_model.state_dict().items()}

            fresh_model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
            optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=0.001)
            _load_resume_checkpoint(ckpt, fresh_model, optimizer, device="cpu", num_epochs=100)

            for key, restored in fresh_model.state_dict().items():
                self.assertTrue(
                    torch.equal(restored, saved_state[key]),
                    f"Parameter '{key}' was not restored from the checkpoint.",
                )

    def test_optimizer_state_restored(self):
        import torch
        from src.model import ProstateUNet2D
        from src.train import _load_resume_checkpoint

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            _make_epoch60_style_checkpoint(ckpt, epoch=60)

            model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            self.assertEqual(len(optimizer.state_dict()["state"]), 0)  # empty before

            _load_resume_checkpoint(ckpt, model, optimizer, device="cpu", num_epochs=100)
            self.assertGreater(
                len(optimizer.state_dict()["state"]), 0,
                "Optimizer momentum/state was not restored from the checkpoint.",
            )

    def test_fail_clearly_when_at_or_past_total_epochs(self):
        import torch
        from src.model import ProstateUNet2D
        from src.train import _load_resume_checkpoint

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            _make_epoch60_style_checkpoint(ckpt, epoch=100)  # already at total

            model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            with self.assertRaises(ValueError):
                _load_resume_checkpoint(ckpt, model, optimizer, device="cpu", num_epochs=100)

    def test_missing_checkpoint_raises(self):
        import torch
        from src.model import ProstateUNet2D
        from src.train import _load_resume_checkpoint

        model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        with self.assertRaises(FileNotFoundError):
            _load_resume_checkpoint(
                "/no/such/checkpoint.pt", model, optimizer, device="cpu", num_epochs=100
            )

    def test_epoch60_format_remains_loadable(self):
        """The exact 5-key format used by the real epoch-60 checkpoint must
        still load cleanly (backward compatibility guarantee)."""
        import torch
        from src.model import ProstateUNet2D
        from src.train import _load_resume_checkpoint

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            _make_epoch60_style_checkpoint(ckpt, epoch=60, best_metric=0.6496429678992481)

            loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
            self.assertEqual(
                set(loaded.keys()),
                {"epoch", "model_state_dict", "optimizer_state_dict", "best_metric", "config"},
            )

            model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            start_epoch, best_metric, _ = _load_resume_checkpoint(
                ckpt, model, optimizer, device="cpu", num_epochs=100
            )
            self.assertEqual(start_epoch, 61)
            self.assertAlmostEqual(best_metric, 0.6496429678992481, places=12)


def _tiny_config(dataset_root, output_dir, num_epochs, init_features=4):
    return {
        "experiment": {"name": "resume_test", "output_dir": output_dir},
        "data": {
            "dataset_root": dataset_root,
            "patient_ids": ["020"],
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
            "init_features": init_features,
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
        "validation": {"val_patient_ids": []},
    }


class TestRunTrainingResumeIntegration(unittest.TestCase):
    def setUp(self):
        self.dataset_root = os.path.join(project_root, "dataset", "prostate158_train", "train")
        if not os.path.isdir(os.path.join(self.dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping integration test.")

    def _write_config(self, tmp_dir, num_epochs):
        import yaml

        config = _tiny_config(self.dataset_root, tmp_dir, num_epochs)
        path = os.path.join(tmp_dir, "config.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)
        return path

    def test_fresh_run_starts_at_epoch_1(self):
        from src.train import run_training

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = self._write_config(tmp_dir, num_epochs=1)
            summary = run_training(config_path)  # no resume_from -> fresh
            self.assertFalse(summary["resumed"])
            self.assertEqual(summary["start_epoch"], 1)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "best_model.pt")))

    def test_resume_continues_at_checkpoint_epoch_plus_one(self):
        import torch
        from src.train import run_training

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Fresh 1-epoch run produces best_model.pt at epoch 1.
            config_path = self._write_config(tmp_dir, num_epochs=1)
            run_training(config_path)
            ckpt_path = os.path.join(tmp_dir, "best_model.pt")
            self.assertEqual(
                torch.load(ckpt_path, map_location="cpu", weights_only=False)["epoch"], 1
            )

            # Resume with a higher epoch target -> must continue at epoch 2.
            config_path2 = self._write_config(tmp_dir, num_epochs=2)
            summary = run_training(config_path2, resume_from=ckpt_path)
            self.assertTrue(summary["resumed"])
            self.assertEqual(summary["start_epoch"], 2)  # NOT 1

    def test_best_metric_not_reset_on_resume(self):
        """Resume from a checkpoint whose best_metric is artificially high and
        unbeatable in one tiny epoch: best_model.pt must NOT be overwritten,
        proving best_metric was restored (not reset to -inf)."""
        import torch
        from src.train import run_training

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = os.path.join(tmp_dir, "best_model.pt")
            # Craft an epoch-1 checkpoint with an unbeatable best_metric.
            _make_epoch60_style_checkpoint(ckpt_path, epoch=1, best_metric=0.99, init_features=4)

            config_path = self._write_config(tmp_dir, num_epochs=2)
            summary = run_training(config_path, resume_from=ckpt_path)
            self.assertEqual(summary["start_epoch"], 2)

            # best_model.pt must still be the original epoch-1 / 0.99 checkpoint,
            # because the single resumed epoch cannot beat best_metric = 0.99.
            after = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.assertEqual(after["epoch"], 1)
            self.assertAlmostEqual(after["best_metric"], 0.99, places=6)

    def test_resumed_checkpoint_remains_loadable(self):
        import torch
        from src.train import run_training

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Craft an epoch-1 checkpoint with a low best_metric so the resumed
            # epoch WILL improve on it and overwrite best_model.pt.
            ckpt_path = os.path.join(tmp_dir, "best_model.pt")
            _make_epoch60_style_checkpoint(ckpt_path, epoch=1, best_metric=-999.0, init_features=4)

            config_path = self._write_config(tmp_dir, num_epochs=2)
            run_training(config_path, resume_from=ckpt_path)

            reloaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            for key in ("epoch", "model_state_dict", "optimizer_state_dict", "best_metric", "config"):
                self.assertIn(key, reloaded)
            self.assertEqual(reloaded["epoch"], 2)  # improved epoch was saved


if __name__ == "__main__":
    unittest.main()
