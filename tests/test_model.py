"""Unit tests for ProstateUNet2D model architecture and forward pass."""

import os
import sys
import unittest

# Ensure project root is in sys.path for robust imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import ProstateUNet2D, build_model


class TestProstateUNet2D(unittest.TestCase):
    """Test suite for 2D U-Net model."""

    def setUp(self):
        try:
            import torch
            self.torch = torch
            self.has_torch = True
        except ImportError:
            self.has_torch = False

    def test_model_forward_270x270(self):
        """Verify forward pass on Patient 020 resolution [B=2, C=3, H=270, W=270]."""
        if not self.has_torch:
            self.skipTest("PyTorch not installed in this environment.")

        model = ProstateUNet2D(in_channels=3, out_channels=1, init_features=32, dropout_rate=0.2)
        model.eval()

        x = self.torch.randn(2, 3, 270, 270)
        with self.torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(x.shape), (2, 3, 270, 270))
        self.assertEqual(tuple(y.shape), (2, 1, 270, 270))

    def test_model_forward_even_dimension(self):
        """Verify forward pass on standard even dimension [B=1, C=3, H=160, W=160]."""
        if not self.has_torch:
            self.skipTest("PyTorch not installed in this environment.")

        model = ProstateUNet2D(in_channels=3, out_channels=1, init_features=16)
        model.eval()

        x = self.torch.randn(1, 3, 160, 160)
        with self.torch.no_grad():
            y = model(x)

        self.assertEqual(tuple(y.shape), (1, 1, 160, 160))

    def test_build_model_factory(self):
        """Verify build_model factory instantiates ProstateUNet2D from config."""
        if not self.has_torch:
            self.skipTest("PyTorch not installed in this environment.")

        cfg = {
            "model": {
                "in_channels": 3,
                "out_channels": 1,
                "init_features": 32,
                "dropout_rate": 0.2,
            }
        }
        model = build_model(cfg)
        self.assertIsInstance(model, ProstateUNet2D)
        self.assertEqual(model.in_channels, 3)
        self.assertEqual(model.out_channels, 1)
        self.assertEqual(model.init_features, 32)
        self.assertEqual(model.dropout_rate, 0.2)

    def test_cuda_device_transfer_if_available(self):
        """Verify model device transfer on CUDA when available."""
        if not self.has_torch:
            self.skipTest("PyTorch not installed in this environment.")

        if not self.torch.cuda.is_available():
            self.skipTest("CUDA is not available on this host environment.")

        model = ProstateUNet2D(in_channels=3, out_channels=1).cuda()
        x = self.torch.randn(2, 3, 270, 270, device="cuda")
        y = model(x)
        self.assertTrue(y.is_cuda)
        self.assertEqual(tuple(y.shape), (2, 1, 270, 270))


if __name__ == "__main__":
    unittest.main()

