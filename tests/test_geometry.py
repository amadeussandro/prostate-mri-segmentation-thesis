"""Unit tests for src/geometry.py -- the spatial-verification authority."""

import os
import sys
import unittest

import nibabel as nib
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.geometry import (
    VolumeGeometry,
    assert_complete_slice_indices,
    read_geometry,
    verify_affine_match,
    verify_image_mask_geometry,
    verify_orientation_match,
    verify_original_space_compatibility,
    verify_reconstruction_compatibility,
    verify_shape_match,
    verify_spacing_match,
)


def _make_nii(shape=(10, 12, 5), affine=None):
    if affine is None:
        affine = np.diag([0.4, 0.4, 3.0, 1.0])
    data = np.zeros(shape, dtype=np.float32)
    return nib.Nifti1Image(data, affine)


class TestGeometry(unittest.TestCase):
    def test_read_geometry_never_defaults_to_identity(self):
        affine = np.diag([0.4, 0.4, 3.0, 1.0])
        img = _make_nii(affine=affine)
        geom = read_geometry(img)
        self.assertFalse(np.allclose(geom.affine, np.eye(4)))
        self.assertTrue(np.allclose(geom.affine, affine))

    def test_shape_mismatch_raises(self):
        a = read_geometry(_make_nii(shape=(10, 10, 5)))
        b = read_geometry(_make_nii(shape=(10, 12, 5)))
        with self.assertRaises(ValueError):
            verify_shape_match(a, b)

    def test_affine_mismatch_raises(self):
        a = read_geometry(_make_nii(affine=np.diag([0.4, 0.4, 3.0, 1.0])))
        b = read_geometry(_make_nii(affine=np.diag([0.5, 0.5, 3.0, 1.0])))
        with self.assertRaises(ValueError):
            verify_affine_match(a, b)

    def test_spacing_mismatch_raises(self):
        a = read_geometry(_make_nii(affine=np.diag([0.4, 0.4, 3.0, 1.0])))
        b = read_geometry(_make_nii(affine=np.diag([0.4, 0.4, 5.0, 1.0])))
        with self.assertRaises(ValueError):
            verify_spacing_match(a, b)

    def test_orientation_mismatch_raises(self):
        ras_affine = np.diag([0.4, 0.4, 3.0, 1.0])
        lps_affine = np.diag([-0.4, -0.4, 3.0, 1.0])
        a = read_geometry(_make_nii(affine=ras_affine))
        b = read_geometry(_make_nii(affine=lps_affine))
        with self.assertRaises(ValueError):
            verify_orientation_match(a, b)

    def test_verify_image_mask_geometry_passes_for_identical(self):
        affine = np.diag([0.4, 0.4, 3.0, 1.0])
        a = read_geometry(_make_nii(affine=affine))
        b = read_geometry(_make_nii(affine=affine))
        verify_image_mask_geometry(a, b)  # should not raise

    def test_reconstruction_compatibility(self):
        geom = read_geometry(_make_nii(shape=(10, 12, 5)))
        verify_reconstruction_compatibility((10, 12, 5), geom)  # should not raise
        with self.assertRaises(ValueError):
            verify_reconstruction_compatibility((10, 12, 4), geom)

    def test_verify_original_space_compatibility_passes_for_identical_geometry(self):
        geom = read_geometry(_make_nii(affine=np.diag([0.4, 0.4, 3.0, 1.0])))
        verify_original_space_compatibility(geom, geom)  # should not raise

    def test_verify_original_space_compatibility_fails_on_altered_affine(self):
        source = read_geometry(_make_nii(affine=np.diag([0.4, 0.4, 3.0, 1.0])))
        altered = read_geometry(_make_nii(affine=np.diag([0.5, 0.5, 3.0, 1.0])))
        with self.assertRaises(ValueError):
            verify_original_space_compatibility(altered, source)

    def test_verify_original_space_compatibility_fails_on_altered_shape(self):
        source = read_geometry(_make_nii(shape=(10, 12, 5)))
        altered = read_geometry(_make_nii(shape=(10, 14, 5)))
        with self.assertRaises(ValueError):
            verify_original_space_compatibility(altered, source)

    def test_verify_original_space_compatibility_fails_on_altered_spacing(self):
        source = read_geometry(_make_nii(affine=np.diag([0.4, 0.4, 3.0, 1.0])))
        altered = read_geometry(_make_nii(affine=np.diag([0.4, 0.4, 5.0, 1.0])))
        with self.assertRaises(ValueError):
            verify_original_space_compatibility(altered, source)

    def test_verify_original_space_compatibility_fails_on_altered_orientation(self):
        source = read_geometry(_make_nii(affine=np.diag([0.4, 0.4, 3.0, 1.0])))
        altered = read_geometry(_make_nii(affine=np.diag([-0.4, -0.4, 3.0, 1.0])))
        with self.assertRaises(ValueError):
            verify_original_space_compatibility(altered, source)

    def test_reconstruction_wires_original_space_verification(self):
        """Integration: reconstruct_case_to_original_space must actually call
        verify_original_space_compatibility (checklist item 9) and propagate
        its failure. Because the function builds its output NIfTI directly
        FROM source_geometry, simply passing a corrupted source_geometry is
        self-consistent (not a mismatch) -- so to prove the check is wired in
        for real (not called-and-ignored), we monkeypatch
        src.reconstruction.read_geometry to return a deliberately mismatched
        VolumeGeometry for the post-construction read-back, simulating the
        exact failure this check exists to catch (the NIfTI object actually
        produced doesn't match what was intended)."""
        from dataclasses import replace as _dc_replace

        import src.reconstruction as reconstruction_module
        from src.dataset import ProstateZonal2DDataset
        from src.reconstruction import reconstruct_case_to_original_space
        from src.transforms import PreprocessingConfig

        dataset_root = os.path.join(project_root, "dataset", "prostate158_train", "train")
        if not os.path.isdir(os.path.join(dataset_root, "020")):
            self.skipTest("Patient 020 folder not found; skipping integration test.")

        config = PreprocessingConfig(spatial_mode="crop_pad", target_size=(270, 270), z_resample=False)
        dataset = ProstateZonal2DDataset(
            dataset_root=dataset_root, patient_ids=["020"], slice_sampling="all", preprocessing_config=config,
        )
        geometry, transform_meta = dataset.get_case_metadata("020")

        slices = [dataset[i]["mask"] for i in range(len(dataset))]
        slice_indices = [dataset[i]["slice_idx"] for i in range(len(dataset))]

        # Correct path: must succeed (proves the normal path still works with
        # the new verification wired in, i.e. no false positives).
        reconstruct_case_to_original_space(slices, slice_indices, transform_meta, geometry)

        # Simulate a reconstruction bug: the NIfTI actually produced has a
        # spacing that doesn't match the intended source geometry.
        mismatched_affine = geometry.affine.copy()
        mismatched_affine[0, 0] *= 2.0
        mismatched_geometry = _dc_replace(geometry, affine=mismatched_affine)

        original_read_geometry = reconstruction_module.read_geometry
        reconstruction_module.read_geometry = lambda nii_img: mismatched_geometry
        try:
            with self.assertRaises(ValueError):
                reconstruct_case_to_original_space(slices, slice_indices, transform_meta, geometry)
        finally:
            reconstruction_module.read_geometry = original_read_geometry

    def test_assert_complete_slice_indices(self):
        assert_complete_slice_indices([0, 1, 2, 3], 4)  # should not raise
        with self.assertRaises(ValueError):
            assert_complete_slice_indices([0, 1, 3], 4)  # missing index 2


if __name__ == "__main__":
    unittest.main()
