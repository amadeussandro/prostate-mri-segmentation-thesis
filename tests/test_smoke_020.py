"""Gate 5 regression test -- wraps scripts/smoke_test_020.py so the forward-only
Case-020 smoke test is checked automatically by the test suite too."""

import os
import sys
import unittest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
scripts_dir = os.path.join(project_root, "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)


class TestSmokeTest020(unittest.TestCase):
    def test_smoke_test_020_passes(self):
        dataset_root = os.path.join(project_root, "dataset", "prostate158_train", "train", "020")
        if not os.path.exists(dataset_root):
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        from smoke_test_020 import run_smoke_test

        self.assertTrue(run_smoke_test("configs/config_020.yaml"))


if __name__ == "__main__":
    unittest.main()
