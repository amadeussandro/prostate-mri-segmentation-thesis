"""Unit tests for the statistics and helpers in scripts/run_picai_psad.py (no PI-CAI data needed)."""
import os
import sys

import nibabel as nib
import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import run_picai_psad as rp  # noqa: E402


def test_auc_perfect_inverse_and_ties():
    y = np.array([0, 0, 1, 1])
    assert rp.auc(np.array([1, 2, 3, 4.0]), y) == 1.0
    assert rp.auc(np.array([4, 3, 2, 1.0]), y) == 0.0
    assert rp.auc(np.array([1, 1, 1, 1.0]), y) == 0.5


def test_auc_matches_pairwise_definition():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.normal(size=200) + y, 1)  # rounding creates ties
    pos, neg = s[y == 1], s[y == 0]
    ref = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    assert rp.auc(s, y) == pytest.approx(ref)


def test_psad_empty_volume_ranks_highest():
    d = rp.psad(np.array([5.0, 5.0]), np.array([50.0, 0.0]))
    assert d[0] == pytest.approx(0.1) and np.isinf(d[1])
    y = np.array([0, 1])
    assert rp.auc(d, y) == 1.0  # +inf is ranked above every finite value


def test_compare_identical_scores_give_zero_difference():
    rng = np.random.default_rng(2)
    y = np.r_[np.zeros(60, int), np.ones(40, int)]
    a = rng.normal(size=100) + y
    c = rp.compare(a, a.copy(), y, n=200)
    assert c["diff"] == 0 and c["diff_ci95"] == [0.0, 0.0]
    assert c["noninferior"] and not c["superior"]


def test_compare_detects_a_clearly_worse_score():
    rng = np.random.default_rng(3)
    y = np.r_[np.zeros(150, int), np.ones(150, int)]
    good = y * 3.0 + rng.normal(size=300)
    noise = rng.normal(size=300)
    c = rp.compare(noise, good, y, n=500)
    assert c["diff_ci95"][1] < 0 and not c["noninferior"]


def test_compare_is_reproducible_with_the_fixed_seed():
    rng = np.random.default_rng(4)
    y = np.r_[np.zeros(50, int), np.ones(50, int)]
    a, b = rng.normal(size=100) + y, rng.normal(size=100) + y
    assert rp.compare(a, b, y, n=300) == rp.compare(a, b, y, n=300)


def test_agreement_bias_and_limits():
    ref = np.array([40.0, 50, 60, 70])
    x = ref + np.array([1.0, 3, 1, 3])
    a = rp.agreement(x, ref)
    assert a["bias"] == pytest.approx(2.0)
    assert a["loa95"][0] == pytest.approx(2 - 1.96 * np.std([1, 3, 1, 3], ddof=1))


def test_at_threshold():
    y = np.array([1, 1, 0, 0])
    r = rp.at_threshold(np.array([0.2, 0.1, 0.16, 0.05]), y, t=0.15)
    assert r["sensitivity"] == 0.5 and r["specificity"] == 0.5 and r["n_positive"] == 2


def test_canonical_shape_matches_as_closest_canonical():
    aff = np.diag([-0.5, -0.5, 3.0, 1.0])[:, [1, 0, 2, 3]]  # swapped and flipped in-plane axes
    img = nib.Nifti1Image(np.zeros((30, 40, 5), np.float32), aff)
    assert rp.canonical_shape(img) == nib.as_closest_canonical(img).shape


def test_lock_detects_a_changed_prespecification(tmp_path, monkeypatch):
    pre = tmp_path / "pre.md"
    pre.write_bytes(b"line one\r\nline two\n")
    lock = tmp_path / "lock.json"
    lock.write_text('{"sha256_lf": "%s"}' % rp.sha_lf(str(pre)))
    monkeypatch.setattr(rp, "PRESPEC", str(pre))
    monkeypatch.setattr(rp, "LOCK", str(lock))
    rp.check_lock()
    pre.write_bytes(b"line one\nline two changed\n")
    with pytest.raises(RuntimeError):
        rp.check_lock()


def test_the_real_lock_matches():
    rp.check_lock()
