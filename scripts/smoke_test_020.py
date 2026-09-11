"""Gate 5 -- Case-020 forward-only smoke test.

Exercises the full current-thesis chain with NO TRAINING:

    Case 020 -> load T2 + anatomy mask -> spatial verification -> preprocessing
    -> axial 3D->2D slicing -> Dataset -> DataLoader -> 2D U-Net -> forward pass

Verifies:
  - no exception raised anywhere in the chain
  - input tensor shape (B, 1, H, W), output tensor shape (B, 3, H, W)
  - correct dtypes (image float32, mask int64, logits float32)
  - valid mask labels (subset of {0, 1, 2})
  - all output tensors are finite (no NaN/Inf)
  - image-mask correspondence (same patient_id + slice_idx pairing survives batching)

Usage:
    python scripts/smoke_test_020.py
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from torch.utils.data import DataLoader

from src.geometry import verify_image_mask_geometry
from src.model import build_model
from src.train import _build_dataset
from src.utils import load_config


def run_smoke_test(config_path: str = "configs/config_020.yaml") -> bool:
    print(f"[1/6] Loading config: {config_path}")
    config = load_config(os.path.join(PROJECT_ROOT, config_path))
    patient_ids = config["data"]["patient_ids"]
    assert patient_ids == ["020"], "This smoke test is scoped to patient 020 only."

    print("[2/6] Building ProstateZonal2DDataset (load + spatial verification + preprocessing + slicing)...")
    dataset = _build_dataset(config, patient_ids)
    assert len(dataset) > 0, "Dataset produced zero slices."
    print(f"      Slices for patient 020: {len(dataset)}")

    print("[3/6] Verifying spatial metadata was retained (never discarded)...")
    geometry, transform_meta = dataset.get_case_metadata("020")
    print(f"      Original shape: {geometry.shape}, spacing: {geometry.zooms}, axcodes: {geometry.axcodes}")
    assert geometry.shape == (270, 270, 24), f"Unexpected geometry shape: {geometry.shape}"

    print("[4/6] Building DataLoader...")
    batch_size = config.get("training", {}).get("batch_size", 4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    images = batch["image"]
    masks = batch["mask"]
    patient_ids_batch = batch["patient_id"]
    slice_idx_batch = batch["slice_idx"]

    print(f"      image batch shape: {tuple(images.shape)}, dtype: {images.dtype}")
    print(f"      mask  batch shape: {tuple(masks.shape)}, dtype: {masks.dtype}")

    assert images.ndim == 4 and images.shape[1] == 1, f"Expected (B,1,H,W), got {tuple(images.shape)}"
    assert masks.ndim == 3, f"Expected (B,H,W), got {tuple(masks.shape)}"
    assert images.shape[0] == masks.shape[0] == len(patient_ids_batch)
    assert images.dtype == torch.float32
    assert masks.dtype == torch.int64

    unique_labels = set(torch.unique(masks).tolist())
    assert unique_labels.issubset({0, 1, 2}), f"Invalid labels found: {unique_labels}"

    print("[5/6] Verifying image-mask correspondence for every item in the batch...")
    for i, (pid, s_idx) in enumerate(zip(patient_ids_batch, slice_idx_batch.tolist())):
        # Same (pid, slice_idx) pair recorded in dataset.samples at this position.
        expected_pid, expected_s_idx = dataset.samples[i]
        assert pid == expected_pid and s_idx == expected_s_idx, (
            f"Correspondence broken at batch index {i}: "
            f"got ({pid}, {s_idx}), expected ({expected_pid}, {expected_s_idx})"
        )

    print("[6/6] Building 2D U-Net and running a single forward pass (no_grad, no training)...")
    model = build_model(config)
    model.eval()
    with torch.no_grad():
        logits = model(images)

    print(f"      logits shape: {tuple(logits.shape)}, dtype: {logits.dtype}")
    expected_out_channels = config["model"]["out_channels"]
    assert logits.shape == (images.shape[0], expected_out_channels, images.shape[2], images.shape[3]), (
        f"Expected output shape (B, {expected_out_channels}, H, W), got {tuple(logits.shape)}"
    )
    assert torch.isfinite(logits).all(), "Model produced non-finite (NaN/Inf) outputs."

    print("\nSMOKE TEST PASSED for patient 020.")
    print(f"  input  shape: {tuple(images.shape)} (B, 1, H, W)")
    print(f"  output shape: {tuple(logits.shape)} (B, 3, H, W)")
    return True


if __name__ == "__main__":
    ok = run_smoke_test()
    sys.exit(0 if ok else 1)
