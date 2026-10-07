"""PROSTATEx external-validation tests -- torch-free, no dataset or checkpoint.

External validation is a one-shot measurement on an independent cohort, so the
failure modes that matter are the silent ones: a mask combined with the wrong
comparison, a case resolved by guessing a filename, a partially uploaded cohort
scored as if it were complete. Each of those is pinned here.
"""

import csv
import json
import os
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import nibabel as nib

from run_prostatex_external_validation import (
    EXPECTED_E1_SHA256,
    crop_loss_report,
    harmonize_case,
    prepare,
    read_manifest,
    sha256_of,
)
from src.transforms import AxisPadCrop, PreprocessingConfig, preprocess_case


def _affine():
    a = np.diag([-0.5, 0.5, 3.0, 1.0]).astype(np.float64)  # LAS-like, as PROSTATEx is
    a[:3, 3] = [40.0, -10.0, -50.0]
    return a


class _Cohort:
    """A miniature on-disk PROSTATEx mirror: raw masks plus a manifest."""

    def __init__(self, root, n=2, shape=(8, 8, 3), stray_value=False):
        self.root = root
        self.shape = shape
        os.makedirs(os.path.join(root, "raw", "images", "t2"), exist_ok=True)
        os.makedirs(os.path.join(root, "raw", "masks", "pz"), exist_ok=True)
        os.makedirs(os.path.join(root, "raw", "masks", "rest"), exist_ok=True)
        os.makedirs(os.path.join(root, "metadata"), exist_ok=True)
        self.rows = []
        for i in range(n):
            cid = f"ProstateX-{i:04d}"
            # Deliberately non-canonical mask stem, as 15 real cases have.
            stem = f"ProstateX-{i:03d}" if i == 1 else cid
            t2 = np.random.default_rng(i).random(shape).astype(np.float32)
            pz = np.zeros(shape, dtype=np.uint8)
            rest = np.zeros(shape, dtype=np.uint8)
            pz[0:2, 0:2, :] = 1
            rest[4:6, 4:6, :] = 1
            if stray_value and i == 0:
                pz[3, 3, 0] = 2  # the ProstateX-0086 artifact
            paths = {
                "t2_path": f"raw/images/t2/{cid}_t2_tse_tra_4.nii.gz",
                "pz_mask_path": f"raw/masks/pz/{stem}_pz.nii.gz",
                "rest_mask_path": f"raw/masks/rest/{stem}_tz.nii.gz",
            }
            nib.save(nib.Nifti1Image(t2, _affine()), os.path.join(root, paths["t2_path"]))
            nib.save(nib.Nifti1Image(pz, _affine()), os.path.join(root, paths["pz_mask_path"]))
            nib.save(nib.Nifti1Image(rest, _affine()), os.path.join(root, paths["rest_mask_path"]))
            self.rows.append({"case_id": cid, "raw_mask_stem": stem, "paired_ok": "True", **paths})
        with open(os.path.join(root, "metadata", "prostatex_manifest.csv"),
                  "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)


class TestManifest(unittest.TestCase):
    def test_missing_manifest_fails_loudly(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                read_manifest(td)

    def test_manifest_is_the_only_way_masks_are_resolved(self):
        # Case 0001's masks are stored under a 3-digit stem; harmonization must
        # still find them, which is only possible via the manifest paths.
        with tempfile.TemporaryDirectory() as td:
            c = _Cohort(os.path.join(td, "px"))
            rec = harmonize_case(c.rows[1], c.root, os.path.join(td, "out"))
            self.assertEqual(rec["case_id"], "ProstateX-0001")
            self.assertGreater(rec["cg_voxels"], 0)


class TestHarmonization(unittest.TestCase):
    def _harmonized(self, root, out, row):
        harmonize_case(row, root, out)
        p = os.path.join(out, row["case_id"], "anatomy.nii.gz")
        return np.asarray(nib.load(p).dataobj)

    def test_combines_rest_to_1_and_pz_to_2(self):
        with tempfile.TemporaryDirectory() as td:
            c = _Cohort(os.path.join(td, "px"))
            comb = self._harmonized(c.root, os.path.join(td, "out"), c.rows[0])
            self.assertEqual(sorted(int(v) for v in np.unique(comb)), [0, 1, 2])
            self.assertEqual(int((comb == 2).sum()), 2 * 2 * c.shape[2])  # pz block
            self.assertEqual(int((comb == 1).sum()), 2 * 2 * c.shape[2])  # rest block

    def test_stray_label_value_is_kept_by_the_greater_than_zero_test(self):
        # ProstateX-0086 carries one voxel of value 2 in its PZ mask. `mask == 1`
        # would silently drop it; `mask > 0` keeps it.
        with tempfile.TemporaryDirectory() as td:
            c = _Cohort(os.path.join(td, "px"), stray_value=True)
            comb = self._harmonized(c.root, os.path.join(td, "out"), c.rows[0])
            expected = 2 * 2 * c.shape[2] + 1
            self.assertEqual(int((comb == 2).sum()), expected)

    def test_overlapping_zones_refuse_to_harmonize(self):
        with tempfile.TemporaryDirectory() as td:
            c = _Cohort(os.path.join(td, "px"))
            overlapping = np.zeros(c.shape, dtype=np.uint8)
            overlapping[0:2, 0:2, :] = 1  # same block as pz
            nib.save(nib.Nifti1Image(overlapping, _affine()),
                     os.path.join(c.root, c.rows[0]["rest_mask_path"]))
            with self.assertRaises(ValueError):
                harmonize_case(c.rows[0], c.root, os.path.join(td, "out"))

    def test_geometry_mismatch_refuses_to_harmonize(self):
        with tempfile.TemporaryDirectory() as td:
            c = _Cohort(os.path.join(td, "px"))
            nib.save(nib.Nifti1Image(np.zeros((8, 8, 4), dtype=np.uint8), _affine()),
                     os.path.join(c.root, c.rows[0]["pz_mask_path"]))
            with self.assertRaises(ValueError):
                harmonize_case(c.rows[0], c.root, os.path.join(td, "out"))

    def test_records_the_scale_shift_against_prostate158(self):
        with tempfile.TemporaryDirectory() as td:
            c = _Cohort(os.path.join(td, "px"))
            rec = harmonize_case(c.rows[0], c.root, os.path.join(td, "out"))
            self.assertAlmostEqual(rec["inplane_mm"], 0.5, places=6)
            self.assertAlmostEqual(rec["scale_vs_prostate158"], 0.46875 / 0.5, places=6)


class TestCohortCompleteness(unittest.TestCase):
    def test_a_short_cohort_is_refused(self):
        # The realistic failure: a Drive upload that quietly dropped files.
        with tempfile.TemporaryDirectory() as td:
            c = _Cohort(os.path.join(td, "px"), n=2)
            with self.assertRaises(ValueError) as ctx:
                prepare(c.root, os.path.join(td, "work"), expected_cases=204)
            self.assertIn("incomplete", str(ctx.exception).lower())

    def test_a_complete_cohort_is_accepted(self):
        with tempfile.TemporaryDirectory() as td:
            c = _Cohort(os.path.join(td, "px"), n=3)
            recs = prepare(c.root, os.path.join(td, "work"), expected_cases=3)
            self.assertEqual(len(recs), 3)
            for r in recs:
                self.assertEqual(r["pz_rest_overlap"], 0)


class TestCropLoss(unittest.TestCase):
    """Prostate158 is always padded; PROSTATEx can be cropped, which can cut
    anatomy off before the model ever sees it."""

    def _meta_and_gt(self, native, target):
        gt = np.zeros((native, native, 2), dtype=np.int64)
        gt[:, :, :] = 0
        gt[native // 2 - 1:native // 2 + 1, native // 2 - 1:native // 2 + 1, :] = 1
        t2 = np.random.default_rng(0).random((native, native, 2)).astype(np.float32)
        cfg = PreprocessingConfig(spatial_mode="crop_pad", target_size=(target, target))
        _, _, meta = preprocess_case(nib.Nifti1Image(t2, _affine()),
                                     nib.Nifti1Image(gt.astype(np.float32), _affine()), cfg)
        return gt, meta

    def test_padding_loses_nothing(self):
        gt, meta = self._meta_and_gt(native=8, target=16)
        loss = crop_loss_report(gt, meta)
        self.assertEqual(loss["class1_lost"], 0)

    def test_a_crop_that_misses_the_anatomy_loses_nothing(self):
        gt, meta = self._meta_and_gt(native=16, target=8)  # centred anatomy survives
        loss = crop_loss_report(gt, meta)
        self.assertEqual(loss["class1_lost"], 0)

    def test_a_crop_that_cuts_the_anatomy_is_detected(self):
        native, target = 16, 8
        gt = np.zeros((native, native, 2), dtype=np.int64)
        gt[0:3, 0:3, :] = 1  # anatomy pushed into the corner, outside the centre crop
        t2 = np.random.default_rng(1).random((native, native, 2)).astype(np.float32)
        cfg = PreprocessingConfig(spatial_mode="crop_pad", target_size=(target, target))
        _, _, meta = preprocess_case(nib.Nifti1Image(t2, _affine()),
                                     nib.Nifti1Image(gt.astype(np.float32), _affine()), cfg)
        loss = crop_loss_report(gt, meta)
        self.assertGreater(loss["class1_lost"], 0)
        self.assertLess(loss["class1_after"], loss["class1_before"])


class TestCheckpointGate(unittest.TestCase):
    def test_hash_detects_a_changed_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "ckpt.pt")
            open(p, "wb").write(b"weights-v1")
            first = sha256_of(p)
            open(p, "wb").write(b"weights-v2")
            self.assertNotEqual(first, sha256_of(p))

    def test_expected_sha_is_the_frozen_e1_constant(self):
        self.assertEqual(len(EXPECTED_E1_SHA256), 64)
        self.assertEqual(
            EXPECTED_E1_SHA256,
            "2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2")


if __name__ == "__main__":
    unittest.main()
