"""Gate 10 readiness check for configs/config_baseline.yaml -- NOT a training
run. Verifies:
  - the official split resolves to 119/20 with no leakage (test set flagged absent)
  - the dataset builds correctly for a small subset of official-split patients
  - batching across those patients produces the configured fixed shape (442, 442)
  - the model builds with the configured channel counts

Deliberately does NOT load all 119+20 volumes (too slow for a smoke check) and
does NOT run any training steps.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from torch.utils.data import DataLoader

from src.model import build_model
from src.train import _build_dataset, _resolve_split_ids
from src.utils import load_config


def main():
    config = load_config(os.path.join(PROJECT_ROOT, "configs", "config_baseline.yaml"))

    train_ids, val_ids, test_ids = _resolve_split_ids(config)
    print(f"Official split resolved: train={len(train_ids)}, val={len(val_ids)}, "
          f"test={'MISSING (' + str(test_ids) + ')' if test_ids is None else len(test_ids)}")
    assert len(train_ids) == 119
    assert len(val_ids) == 20
    assert set(train_ids) & set(val_ids) == set()

    subset = train_ids[:2] + val_ids[:1]
    print(f"Building dataset for a small subset (not the full 119+20): {subset}")
    dataset = _build_dataset(config, subset)
    print(f"  slices: {len(dataset)}")

    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    batch = next(iter(loader))
    print(f"  batch image shape: {tuple(batch['image'].shape)}")
    assert tuple(batch["image"].shape[1:]) == (1, 442, 442)

    model = build_model(config)
    print(f"  model in_channels={model.in_channels}, out_channels={model.out_channels}")
    assert model.in_channels == 1 and model.out_channels == 3

    print("\nGate 10 config check PASSED (structural readiness only; no training run).")


if __name__ == "__main__":
    main()
