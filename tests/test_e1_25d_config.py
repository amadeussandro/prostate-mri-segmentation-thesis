"""E1-2.5D must differ from E1 only in the input context, and 2.5D slicing must be right."""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils import load_config  # noqa: E402

E1 = os.path.join(ROOT, "configs", "experiments", "exp_e1_dice_ce.yaml")
E25 = os.path.join(ROOT, "configs", "experiments", "exp_e1_25d.yaml")


def test_only_context_and_channels_differ():
    a, b = load_config(E1), load_config(E25)
    assert set(a) == set(b)
    for sec in a:
        if sec == "experiment":
            continue
        da, db = dict(a[sec]), dict(b[sec])
        if sec == "data":
            assert db.pop("context_slices") == 1 and "context_slices" not in da
        if sec == "model":
            assert da.pop("in_channels") == 1 and db.pop("in_channels") == 3
        assert da == db, sec
    assert a["experiment"]["output_dir"] != b["experiment"]["output_dir"]
    assert "test_csv" not in b["data"] and b["training"]["amp"] is False


def _ds(k, depth=5):
    from src.dataset import ProstateZonal2DDataset
    ds = ProstateZonal2DDataset.__new__(ProstateZonal2DDataset)
    vol = np.stack([np.full((4, 4), z, np.float32) for z in range(depth)], axis=2)
    mask = np.zeros((4, 4, depth), np.int64)
    ds.context_slices, ds.transform, ds.samples = k, None, [("p", z) for z in range(depth)]
    ds._get_patient_volumes = lambda pid: (vol, mask)
    return ds


def test_2d_path_unchanged():
    s = _ds(0)[2]
    assert s["image"].shape == (1, 4, 4) and float(s["image"][0, 0, 0]) == 2


def test_25d_neighbours_and_edges():
    ds = _ds(1)
    assert [float(c[0, 0]) for c in ds[2]["image"]] == [1, 2, 3]
    assert [float(c[0, 0]) for c in ds[0]["image"]] == [0, 0, 1]
    assert [float(c[0, 0]) for c in ds[4]["image"]] == [3, 4, 4]
    assert ds[2]["mask"].shape == (4, 4)


def test_channel_mismatch_rejected():
    from src.train import _context_slices
    cfg = load_config(E25)
    assert _context_slices(cfg) == 1
    cfg["model"]["in_channels"] = 1
    try:
        _context_slices(cfg)
    except ValueError:
        return
    raise AssertionError("mismatch not rejected")
