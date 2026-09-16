"""Tests for the isolated AMP pilot infrastructure.

Scope (CPU-safe; no real CUDA required):
  - amp=false preserves the FP32 path (no scaler, no scaler_state_dict in ckpt).
  - pilot configs parse and differ from each other ONLY by training.amp (+ the
    isolated output_dir/name), and match the Exp01 baseline controls otherwise.
  - an AMP training step executes and a GradScaler is handled correctly.
  - AMP checkpoint save/load (scaler_state_dict) round-trips.
  - existing FP32 checkpoints (no scaler key) still load, with or without a
    scaler passed in (backward compatible).
  - evaluation/inference path takes no AMP argument (stays FP32).
  - pilot output isolation guardrail accepts results/pilot_amp/* and rejects
    official experiment trees.

Integration bits that need the dataset/torch skip cleanly when unavailable.
"""

import inspect
import os
import sys
import tempfile
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_config  # noqa: E402

CONFIG_DIR = os.path.join(project_root, "configs")
BASELINE = os.path.join(CONFIG_DIR, "config_baseline.yaml")
PILOT_FP32 = os.path.join(CONFIG_DIR, "experiments", "pilot_amp_fp32.yaml")
PILOT_AMP = os.path.join(CONFIG_DIR, "experiments", "pilot_amp_amp.yaml")


def _tiny_config(dataset_root, output_dir, num_epochs, amp, init_features=4):
    return {
        "experiment": {"name": "amp_pilot_test", "output_dir": output_dir},
        "data": {"dataset_root": dataset_root, "patient_ids": ["020"], "slice_sampling": "all"},
        "preprocessing": {"normalization": "per_volume", "z_resample": False,
                          "spatial_mode": "crop_pad", "target_size": [270, 270]},
        "model": {"architecture": "ProstateUNet2D", "in_channels": 1, "out_channels": 3,
                 "init_features": init_features, "dropout_rate": 0.2},
        "training": {"batch_size": 4, "learning_rate": 0.001, "num_epochs": num_epochs,
                    "seed": 42, "optimizer": "AdamW", "weight_decay": 0.0001,
                    "loss_function": "cross_entropy", "amp": amp},
        "validation": {"val_patient_ids": []},
    }


class TestPilotConfigs(unittest.TestCase):
    def test_configs_parse_and_amp_flag(self):
        fp32 = load_config(PILOT_FP32)
        amp = load_config(PILOT_AMP)
        self.assertFalse(fp32["training"]["amp"])
        self.assertTrue(amp["training"]["amp"])
        # Pilot is a short 10-epoch screening run.
        self.assertEqual(fp32["training"]["num_epochs"], 10)
        self.assertEqual(amp["training"]["num_epochs"], 10)

    def test_official_experiments_are_100_epochs(self):
        base = load_config(BASELINE)
        self.assertEqual(base["training"]["num_epochs"], 100)
        for name in ("exp02_a_zresample_on", "exp02_b_normalization_global", "exp02_c_spatial_resize"):
            cfg = load_config(os.path.join(CONFIG_DIR, "experiments", f"{name}.yaml"))
            self.assertEqual(cfg["training"]["num_epochs"], 100, f"{name} must stay at 100 epochs")

    def test_fp32_and_amp_differ_only_by_amp_and_output(self):
        fp32 = load_config(PILOT_FP32)
        amp = load_config(PILOT_AMP)
        # Model + data + preprocessing identical.
        self.assertEqual(fp32["model"], amp["model"])
        self.assertEqual(fp32["data"], amp["data"])
        self.assertEqual(fp32["preprocessing"], amp["preprocessing"])
        # Training identical except the amp flag.
        t1, t2 = dict(fp32["training"]), dict(amp["training"])
        self.assertEqual(t1.pop("amp"), False)
        self.assertEqual(t2.pop("amp"), True)
        self.assertEqual(t1, t2)
        # Output dirs isolated + distinct (clean v2 pilot dirs).
        self.assertIn("pilot_amp_v2/fp32", fp32["experiment"]["output_dir"].replace("\\", "/"))
        self.assertIn("pilot_amp_v2/amp", amp["experiment"]["output_dir"].replace("\\", "/"))

    def test_pilot_matches_baseline_scientific_controls(self):
        base = load_config(BASELINE)
        for cfg in (load_config(PILOT_FP32), load_config(PILOT_AMP)):
            for k in ("architecture", "in_channels", "out_channels", "init_features", "dropout_rate"):
                self.assertEqual(cfg["model"][k], base["model"][k])
            for k in ("batch_size", "learning_rate", "seed", "optimizer", "weight_decay", "loss_function"):
                self.assertEqual(cfg["training"][k], base["training"][k])
            for k in ("dataset_root", "train_csv", "valid_csv", "target_mask", "slice_sampling"):
                self.assertEqual(cfg["data"][k], base["data"][k])
            self.assertEqual(cfg["preprocessing"], base["preprocessing"])

    def test_pilot_output_dirs_never_touch_official_trees(self):
        for cfg in (load_config(PILOT_FP32), load_config(PILOT_AMP)):
            out = cfg["experiment"]["output_dir"].replace("\\", "/")
            self.assertIn("results/pilot_amp", out)
            self.assertNotIn("exp01", out)
            self.assertNotIn("exp02", out)


class TestOutputIsolationGuard(unittest.TestCase):
    def _guard(self):
        from scripts.run_amp_pilot import _assert_isolated_output_dir
        return _assert_isolated_output_dir

    def test_accepts_pilot_dirs(self):
        guard = self._guard()
        self.assertIn("pilot_amp", guard("./results/pilot_amp/fp32"))
        self.assertIn("pilot_amp", guard("results/pilot_amp/amp"))
        # v2 (clean) pilot dirs must also be accepted.
        self.assertIn("pilot_amp", guard("./results/pilot_amp_v2/fp32"))
        self.assertIn("pilot_amp", guard(
            "/content/drive/MyDrive/THESIS_PROSTATE158/results/pilot_amp_v2/amp"))

    def test_rejects_official_dirs(self):
        guard = self._guard()
        for bad in ("./results/exp01_baseline",
                    "./results/exp02_preprocessing/a_zresample_on",
                    "./results/something_else"):
            with self.assertRaises(SystemExit):
                guard(bad)


class TestResolvePilotRun(unittest.TestCase):
    """The incident fix: the runner must never silently overwrite a checkpoint."""

    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")

    def _write_ckpt(self, path, epoch, best=0.4, init_features=4):
        import torch
        from src.model import ProstateUNet2D
        m = ProstateUNet2D(in_channels=1, out_channels=3, init_features=init_features)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        torch.save({"epoch": epoch, "model_state_dict": m.state_dict(),
                    "optimizer_state_dict": opt.state_dict(), "best_metric": best,
                    "config": {"model": {"init_features": init_features}}}, path)

    def _resolve(self):
        from scripts.run_amp_pilot import resolve_pilot_run
        return resolve_pilot_run

    def test_no_checkpoint_is_fresh(self):
        resolve = self._resolve()
        with tempfile.TemporaryDirectory() as d:
            r = resolve(os.path.join(d, "best_model.pt"), num_epochs=10)
            self.assertEqual(r["mode"], "fresh")
            self.assertIsNone(r["resume_from"])

    def test_existing_checkpoint_resumes_at_next_epoch(self):
        resolve = self._resolve()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            self._write_ckpt(ckpt, epoch=6)
            r = resolve(ckpt, num_epochs=10)
            self.assertEqual(r["mode"], "resume")
            self.assertEqual(r["resume_from"], ckpt)
            self.assertEqual(r["plan"]["start_epoch"], 7)  # N+1

    def test_completed_checkpoint_is_complete(self):
        resolve = self._resolve()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            self._write_ckpt(ckpt, epoch=10)
            r = resolve(ckpt, num_epochs=10)
            self.assertEqual(r["mode"], "complete")
            self.assertIsNone(r["resume_from"])

    def test_fresh_refuses_when_checkpoint_exists(self):
        resolve = self._resolve()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            self._write_ckpt(ckpt, epoch=6)
            with self.assertRaises(SystemExit):  # must NOT silently overwrite
                resolve(ckpt, num_epochs=10, fresh=True)

    def test_fresh_allowed_when_no_checkpoint(self):
        resolve = self._resolve()
        with tempfile.TemporaryDirectory() as d:
            r = resolve(os.path.join(d, "best_model.pt"), num_epochs=10, fresh=True)
            self.assertEqual(r["mode"], "fresh")

    def test_force_fresh_overrides_existing_checkpoint(self):
        resolve = self._resolve()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            self._write_ckpt(ckpt, epoch=6)
            r = resolve(ckpt, num_epochs=10, force_fresh=True)
            self.assertEqual(r["mode"], "fresh_forced")
            self.assertIsNone(r["resume_from"])

    def test_corrupt_checkpoint_fails_clearly(self):
        resolve = self._resolve()
        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            with open(ckpt, "wb") as f:
                f.write(b"not a checkpoint")
            with self.assertRaises(SystemExit):
                resolve(ckpt, num_epochs=10)


class TestEvaluationStaysFP32(unittest.TestCase):
    def test_validate_has_no_amp_parameter(self):
        from src.train import validate
        params = inspect.signature(validate).parameters
        self.assertNotIn("use_amp", params)
        self.assertNotIn("scaler", params)

    def test_run_evaluation_has_no_amp_parameter(self):
        from src.evaluate import run_evaluation
        params = inspect.signature(run_evaluation).parameters
        self.assertNotIn("use_amp", params)
        self.assertNotIn("amp", params)


class TestAmpStepAndCheckpoint(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")

    def test_amp_train_step_executes_cpu(self):
        import torch
        from src.model import ProstateUNet2D
        from src.train import train_one_epoch

        model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler(enabled=False)  # CPU: scaler disabled, path still exercised

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return 4
            def __getitem__(self, i):
                return {"image": torch.zeros(1, 16, 16), "mask": torch.zeros(16, 16, dtype=torch.long)}

        loader = torch.utils.data.DataLoader(_DS(), batch_size=2)
        loss = train_one_epoch(model, loader, optimizer, "cpu", use_amp=True, scaler=scaler)
        self.assertIsInstance(loss, float)

    def test_amp_checkpoint_saves_and_reloads_scaler(self):
        import torch
        from src.model import ProstateUNet2D
        from src.train import _load_resume_checkpoint, inspect_checkpoint

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler(enabled=False)
            torch.save({
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": 0.1,
                "config": {"training": {"amp": True}},
                "scaler_state_dict": scaler.state_dict(),
            }, ckpt)

            info = inspect_checkpoint(ckpt)
            self.assertTrue(info["has_scaler_state"])
            self.assertTrue(info["resumable"])

            # Resume WITH a scaler -> scaler state restored, continues at epoch 2.
            model2 = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
            opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
            scaler2 = torch.amp.GradScaler(enabled=False)
            start_epoch, best_metric, ckpt_epoch = _load_resume_checkpoint(
                ckpt, model2, opt2, device="cpu", num_epochs=15, scaler=scaler2
            )
            self.assertEqual(start_epoch, 2)
            self.assertEqual(ckpt_epoch, 1)

    def test_fp32_checkpoint_loads_with_and_without_scaler(self):
        """A legacy FP32 checkpoint (no scaler_state_dict) must still load, both
        when no scaler is passed and when a scaler IS passed (left fresh)."""
        import torch
        from src.model import ProstateUNet2D
        from src.train import _load_resume_checkpoint, inspect_checkpoint

        with tempfile.TemporaryDirectory() as d:
            ckpt = os.path.join(d, "best_model.pt")
            model = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            torch.save({  # exact 5-key FP32 format, no scaler
                "epoch": 3,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_metric": 0.5,
                "config": {"training": {}},
            }, ckpt)

            info = inspect_checkpoint(ckpt)
            self.assertFalse(info["has_scaler_state"])
            self.assertTrue(info["resumable"])

            m1 = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
            o1 = torch.optim.AdamW(m1.parameters(), lr=1e-3)
            se1, _, _ = _load_resume_checkpoint(ckpt, m1, o1, device="cpu", num_epochs=15)
            self.assertEqual(se1, 4)

            m2 = ProstateUNet2D(in_channels=1, out_channels=3, init_features=4)
            o2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler(enabled=False)
            se2, _, _ = _load_resume_checkpoint(ckpt, m2, o2, device="cpu", num_epochs=15, scaler=scaler)
            self.assertEqual(se2, 4)  # loads fine; scaler simply left fresh


class TestAmpRunTrainingIntegration(unittest.TestCase):
    def setUp(self):
        self.dataset_root = os.path.join(project_root, "dataset", "prostate158_train", "train")
        if not os.path.isdir(os.path.join(self.dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping integration test.")

    def _write(self, tmp, num_epochs, amp):
        import yaml
        cfg = _tiny_config(self.dataset_root, tmp, num_epochs, amp)
        p = os.path.join(tmp, "config.yaml")
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
        return p

    def test_fp32_run_checkpoint_has_no_scaler_and_reports_amp_false(self):
        import torch
        from src.train import run_training

        with tempfile.TemporaryDirectory() as tmp:
            summary = run_training(self._write(tmp, 1, amp=False))
            self.assertFalse(summary["amp"])
            ck = torch.load(os.path.join(tmp, "best_model.pt"), map_location="cpu", weights_only=False)
            self.assertNotIn("scaler_state_dict", ck)  # FP32 format unchanged

    def test_amp_run_writes_scaler_and_history(self):
        import json
        import torch
        from src.train import run_training

        with tempfile.TemporaryDirectory() as tmp:
            hist = os.path.join(tmp, "pilot_history.json")
            summary = run_training(self._write(tmp, 1, amp=True), history_path=hist)
            self.assertTrue(summary["amp"])
            ck = torch.load(os.path.join(tmp, "best_model.pt"), map_location="cpu", weights_only=False)
            self.assertIn("scaler_state_dict", ck)
            self.assertTrue(os.path.exists(hist))
            with open(hist) as f:
                payload = json.load(f)
            self.assertTrue(payload["amp"])
            self.assertEqual(len(payload["epochs"]), 1)
            self.assertIn("epoch_seconds", payload["epochs"][0])


if __name__ == "__main__":
    unittest.main()
