"""Tests for the performance-only DataLoader knobs added for T4 throughput.

These knobs (num_workers/pin_memory/persistent_workers/prefetch_factor, plus
non_blocking transfers) must be PURE performance changes: with an unchanged
config they must reproduce the exact historical DataLoader behavior
(num_workers=0, pin_memory only on CUDA), so the scientific run is untouched.
They never change which samples are drawn, their order, or their values.
"""

import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class TestDataLoaderPerfKwargs(unittest.TestCase):
    def setUp(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("PyTorch not installed; skipping.")

    def test_default_cpu_matches_historical_behavior(self):
        from src.train import _dataloader_perf_kwargs

        kw = _dataloader_perf_kwargs({}, "cpu")
        # Historical behavior: num_workers=0, pin_memory=(device=='cuda') -> False on CPU.
        self.assertEqual(kw, {"num_workers": 0, "pin_memory": False})

    def test_default_cuda_matches_historical_behavior(self):
        from src.train import _dataloader_perf_kwargs

        kw = _dataloader_perf_kwargs({}, "cuda")
        self.assertEqual(kw, {"num_workers": 0, "pin_memory": True})

    def test_workers_enable_persistent_and_prefetch(self):
        from src.train import _dataloader_perf_kwargs

        kw = _dataloader_perf_kwargs({"num_workers": 4}, "cuda")
        self.assertEqual(kw["num_workers"], 4)
        self.assertTrue(kw["persistent_workers"])
        self.assertEqual(kw["prefetch_factor"], 4)

    def test_no_persistent_or_prefetch_when_single_process(self):
        from src.train import _dataloader_perf_kwargs

        kw = _dataloader_perf_kwargs({"num_workers": 0}, "cuda")
        self.assertNotIn("persistent_workers", kw)
        self.assertNotIn("prefetch_factor", kw)

    def test_explicit_overrides_respected(self):
        from src.train import _dataloader_perf_kwargs

        kw = _dataloader_perf_kwargs(
            {"num_workers": 2, "pin_memory": False, "persistent_workers": False, "prefetch_factor": 8},
            "cuda",
        )
        self.assertEqual(kw["num_workers"], 2)
        self.assertFalse(kw["pin_memory"])
        self.assertFalse(kw["persistent_workers"])
        self.assertEqual(kw["prefetch_factor"], 8)

    def test_move_batch_to_device_cpu_is_correct(self):
        import numpy as np
        import torch

        from src.train import _move_batch_to_device

        batch = {
            "image": torch.from_numpy(np.zeros((2, 1, 8, 8), dtype=np.float32)),
            "mask": torch.from_numpy(np.zeros((2, 8, 8), dtype=np.int64)),
        }
        images, masks = _move_batch_to_device(batch, "cpu")
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(masks.dtype, torch.int64)
        self.assertEqual(tuple(images.shape), (2, 1, 8, 8))
        self.assertEqual(tuple(masks.shape), (2, 8, 8))


if __name__ == "__main__":
    unittest.main()
