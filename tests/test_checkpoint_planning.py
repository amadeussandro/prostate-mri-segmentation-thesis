"""Tests for the resume/checkpoint ORCHESTRATION helpers in src/train.py:
`inspect_checkpoint` and `plan_training_run`.

These back the Experiment-2 Colab resume-safety workflow (each arm auto-detects
its Drive checkpoint and resumes instead of restarting). They are read-only:
they must never train, mutate, or delete a checkpoint. The actual resume
mechanics are tested separately in tests/test_resume.py.

Covered (per the Exp02 resume-hardening requirements):
  - no checkpoint            -> fresh start at epoch 1 (no accidental resume)
  - checkpoint epoch 19      -> resume at epoch 20
  - checkpoint epoch 99      -> resume at epoch 100
  - checkpoint epoch 100     -> training skipped (already complete)
  - corrupted checkpoint     -> clear failure, file NOT deleted, no fresh start
  - incompatible checkpoint  -> clear failure (missing required keys)
  - inspection never mutates the checkpoint file
  - A/B/C use isolated checkpoint paths
  - scheduler/RNG state reported (expected absent by design)
"""

import os
import sys
import tempfile
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _write_checkpoint(path, epoch, best_metric=0.5, init_features=4):
    """Write a checkpoint in the exact 5-key format run_training saves."""
    import torch

    from src.model import ProstateUNet2D

    model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=init_features)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    x = torch.randn(1, 1, 32, 32)
    model(x).sum().backward()
    optimizer.step()
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "config": {"model": {"init_features": init_features}},
        },
        path,
    )


class TestInspectCheckpoint(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")

    def test_missing_checkpoint(self):
        from src.train import inspect_checkpoint

        info = inspect_checkpoint("/no/such/best_model.pt")
        self.assertFalse(info["exists"])
        self.assertFalse(info["resumable"])
        self.assertIsNone(info["epoch"])

    def test_good_checkpoint_fields(self):
        from src.train import inspect_checkpoint

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            _write_checkpoint(ckpt, epoch=19, best_metric=0.593479)

            info = inspect_checkpoint(ckpt)
            self.assertTrue(info["exists"])
            self.assertTrue(info["readable"])
            self.assertTrue(info["resumable"])
            self.assertEqual(info["epoch"], 19)
            self.assertAlmostEqual(info["best_metric"], 0.593479, places=6)
            self.assertTrue(info["has_model_state"])
            self.assertTrue(info["has_optimizer_state"])
            # No scheduler / RNG state in this project's format (by design).
            self.assertFalse(info["has_scheduler_state"])
            self.assertFalse(info["has_rng_state"])
            self.assertGreater(info["size_bytes"], 0)

    def test_inspection_does_not_mutate_file(self):
        from src.train import inspect_checkpoint

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            _write_checkpoint(ckpt, epoch=19)
            before_size = os.path.getsize(ckpt)
            before_mtime = os.path.getmtime(ckpt)

            inspect_checkpoint(ckpt)

            self.assertTrue(os.path.exists(ckpt))
            self.assertEqual(os.path.getsize(ckpt), before_size)
            self.assertEqual(os.path.getmtime(ckpt), before_mtime)

    def test_corrupted_checkpoint_reported_not_raised(self):
        from src.train import inspect_checkpoint

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            with open(ckpt, "wb") as f:
                f.write(b"this is not a valid torch checkpoint")

            info = inspect_checkpoint(ckpt)
            self.assertTrue(info["exists"])
            self.assertFalse(info["readable"])
            self.assertFalse(info["resumable"])
            self.assertIsNotNone(info["error"])
            self.assertTrue(os.path.exists(ckpt))  # not deleted

    def test_incompatible_checkpoint_missing_keys(self):
        import torch
        from src.train import inspect_checkpoint

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            torch.save({"epoch": 5}, ckpt)  # missing model/optimizer/best_metric

            info = inspect_checkpoint(ckpt)
            self.assertTrue(info["readable"])
            self.assertFalse(info["resumable"])
            self.assertIn("Missing required key", info["error"])


class TestPlanTrainingRun(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")

    def test_no_checkpoint_starts_fresh(self):
        from src.train import plan_training_run

        plan = plan_training_run("/no/such/best_model.pt", num_epochs=100)
        self.assertEqual(plan["action"], "fresh")
        self.assertIsNone(plan["resume_from"])
        self.assertEqual(plan["start_epoch"], 1)
        self.assertIn("fresh", plan["message"].lower())

    def test_epoch19_resumes_at_20(self):
        from src.train import plan_training_run

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            _write_checkpoint(ckpt, epoch=19, best_metric=0.593479)

            plan = plan_training_run(ckpt, num_epochs=100)
            self.assertEqual(plan["action"], "resume")
            self.assertEqual(plan["resume_from"], ckpt)
            self.assertEqual(plan["checkpoint_epoch"], 19)
            self.assertEqual(plan["start_epoch"], 20)
            self.assertIn("Continuing from epoch 20/100", plan["message"])

    def test_epoch99_resumes_at_100(self):
        from src.train import plan_training_run

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            _write_checkpoint(ckpt, epoch=99)

            plan = plan_training_run(ckpt, num_epochs=100)
            self.assertEqual(plan["action"], "resume")
            self.assertEqual(plan["start_epoch"], 100)

    def test_epoch100_is_already_complete(self):
        from src.train import plan_training_run

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            _write_checkpoint(ckpt, epoch=100)

            plan = plan_training_run(ckpt, num_epochs=100)
            self.assertEqual(plan["action"], "already_complete")
            self.assertIsNone(plan["resume_from"])
            self.assertIsNone(plan["start_epoch"])
            self.assertIn("already complete", plan["message"].lower())

    def test_corrupted_checkpoint_fails_clearly_without_deleting(self):
        from src.train import plan_training_run

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            with open(ckpt, "wb") as f:
                f.write(b"garbage-not-a-checkpoint")

            plan = plan_training_run(ckpt, num_epochs=100)
            self.assertEqual(plan["action"], "corrupt")
            self.assertIsNone(plan["resume_from"])  # NOT a fresh start
            self.assertIn("NOT been modified or deleted", plan["message"])
            self.assertTrue(os.path.exists(ckpt))  # still there

    def test_arms_use_isolated_checkpoint_paths(self):
        """A/B/C must resolve to three distinct best_model.pt paths, all under
        results/exp02_preprocessing/ -- so resuming one arm can never touch
        another arm's checkpoint."""
        from src.utils import load_config

        configs = {
            "A": "configs/experiments/exp02_a_zresample_on.yaml",
            "B": "configs/experiments/exp02_b_normalization_global.yaml",
            "C": "configs/experiments/exp02_c_spatial_resize.yaml",
        }
        ckpt_paths = {}
        for arm, cfg_path in configs.items():
            out = load_config(os.path.join(project_root, cfg_path))["experiment"]["output_dir"]
            out = out.replace("\\", "/").rstrip("/")
            self.assertIn("exp02_preprocessing", out, f"{arm}: output_dir not under exp02_preprocessing")
            ckpt_paths[arm] = out + "/best_model.pt"

        self.assertEqual(len(set(ckpt_paths.values())), 3, f"arm checkpoint paths collide: {ckpt_paths}")


if __name__ == "__main__":
    unittest.main()
