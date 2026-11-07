"""The standalone verifier must pass a correct volume and catch each fault class."""
import os
import sys

import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import verify_reconstruction as vr  # noqa: E402

AFF = np.diag([-0.5, -0.5, 3.0, 1.0])  # LPS-stored, like Prostate158
AFF[:3, 3] = [60.0, 60.0, -40.0]


def _mask():
    m = np.zeros((64, 64, 12), np.uint8)
    m[20:44, 18:40, 3:9] = 1  # CG
    m[20:44, 40:48, 3:9] = 2  # PZ: larger j index, i.e. more posterior with -0.5 y spacing
    return m


def test_correct_volume_passes_everything():
    m = _mask()
    img = nib.Nifti1Image(m, AFF)
    assert vr.geometry(img, img)["passed"]
    assert vr.anatomy(img)["passed"]
    assert vr.roundtrip(m, lambda x: np.flip(x, 0), lambda x: np.flip(x, 0))["passed"]


def test_identity_affine_caught_by_geometry_only():
    m = _mask()
    bad = nib.Nifti1Image(m, np.eye(4))
    assert not vr.geometry(nib.Nifti1Image(m, AFF), bad)["passed"]


def test_missing_inverse_caught_by_roundtrip():
    m = _mask()
    assert not vr.roundtrip(m, lambda x: np.flip(x, 1), lambda x: x)["passed"]


def test_anterior_posterior_flip_caught_by_anatomy():
    m = np.flip(_mask(), 1).copy()
    assert not vr.anatomy(nib.Nifti1Image(m, AFF))["pz_posterior_to_cg"]


def test_shuffled_slices_caught_by_continuity():
    m = _mask()
    m[:, :, 5] = 0  # an empty slice inside the gland
    assert not vr.anatomy(nib.Nifti1Image(m, AFF))["slice_continuity"]
