"""Spatial-verification utilities: a single geometry authority for the pipeline.

NiBabel (RAS convention) is used as the geometry authority throughout, per the
thesis proposal's recommendation (the baseline architecture reference is
MONAI, which is built on top of NiBabel/RAS). Every geometric fact needed to
go from raw NIfTI -> 2D slices -> reconstructed 3D volume -> original space is
captured here explicitly; nothing is silently discarded or defaulted to an
identity affine.

This module implements the thesis's 9-point spatial-verification checklist:
  1. shape
  2. affine
  3. voxel spacing
  4. orientation
  5. image-mask correspondence (checked at the case level here; slice-level
     correspondence is structural in the dataset, see src/dataset.py)
  6. slice axis/index (axial = last axis, by convention enforced here)
  7. transform consistency (affine equality within atol)
  8. reconstruction compatibility (shape/affine of a candidate reconstruction
     against a source geometry)
  9. original-space compatibility (post round-trip / de-transform check)
"""

from dataclasses import dataclass
from typing import Tuple

import nibabel as nib
import numpy as np


AXIAL_AXIS = 2  # last array axis = axial (superior-inferior), by convention


@dataclass(frozen=True)
class VolumeGeometry:
    """Immutable snapshot of a NIfTI volume's spatial identity."""

    shape: Tuple[int, int, int]
    affine: np.ndarray
    zooms: Tuple[float, float, float]
    axcodes: Tuple[str, str, str]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, VolumeGeometry):
            return NotImplemented
        return (
            self.shape == other.shape
            and np.allclose(self.affine, other.affine, atol=1e-4)
            and np.allclose(self.zooms, other.zooms, atol=1e-3)
            and self.axcodes == other.axcodes
        )


def read_geometry(nii_img: nib.Nifti1Image) -> VolumeGeometry:
    """Extract the full spatial identity of a loaded NIfTI image. Never defaults
    to an identity affine -- if nibabel could load it, the affine is real."""
    shape = tuple(int(s) for s in nii_img.shape[:3])
    affine = np.array(nii_img.affine, dtype=np.float64, copy=True)
    zooms = tuple(float(z) for z in nii_img.header.get_zooms()[:3])
    axcodes = tuple(nib.aff2axcodes(affine))
    return VolumeGeometry(shape=shape, affine=affine, zooms=zooms, axcodes=axcodes)


def verify_shape_match(a: VolumeGeometry, b: VolumeGeometry) -> None:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")


def verify_affine_match(a: VolumeGeometry, b: VolumeGeometry, atol: float = 1e-4) -> None:
    if not np.allclose(a.affine, b.affine, atol=atol):
        raise ValueError(f"Affine mismatch (atol={atol}):\n{a.affine}\nvs\n{b.affine}")


def verify_spacing_match(a: VolumeGeometry, b: VolumeGeometry, atol: float = 1e-3) -> None:
    if not np.allclose(a.zooms, b.zooms, atol=atol):
        raise ValueError(f"Voxel spacing mismatch: {a.zooms} vs {b.zooms}")


def verify_orientation_match(a: VolumeGeometry, b: VolumeGeometry) -> None:
    if a.axcodes != b.axcodes:
        raise ValueError(f"Orientation mismatch: {a.axcodes} vs {b.axcodes}")


def verify_image_mask_geometry(
    image_geom: VolumeGeometry,
    mask_geom: VolumeGeometry,
    atol: float = 1e-4,
) -> None:
    """Checklist items 1-4: shape, affine, spacing, orientation must all agree
    between an image volume and its corresponding mask volume."""
    verify_shape_match(image_geom, mask_geom)
    verify_affine_match(image_geom, mask_geom, atol=atol)
    verify_spacing_match(image_geom, mask_geom, atol=atol)
    verify_orientation_match(image_geom, mask_geom)


def verify_reconstruction_compatibility(
    reconstructed_shape: Tuple[int, int, int],
    source_geom: VolumeGeometry,
) -> None:
    """Checklist item 8: a reconstructed volume must match the source geometry's
    shape before it can be written out with the source affine/header."""
    if tuple(reconstructed_shape) != source_geom.shape:
        raise ValueError(
            f"Reconstruction shape {reconstructed_shape} does not match "
            f"source geometry shape {source_geom.shape}."
        )


def verify_original_space_compatibility(
    result_geom: VolumeGeometry,
    source_geom: VolumeGeometry,
    atol: float = 1e-4,
) -> None:
    """Checklist item 9: after inverse-transforming back to original space, the
    result's full geometry (shape, affine, spacing, orientation) must match the
    untouched source geometry exactly (within numerical tolerance)."""
    verify_shape_match(result_geom, source_geom)
    verify_affine_match(result_geom, source_geom, atol=atol)
    verify_spacing_match(result_geom, source_geom, atol=atol)
    verify_orientation_match(result_geom, source_geom)


def assert_complete_slice_indices(present_indices, expected_depth: int) -> None:
    """Checklist item 6/7 companion: reconstruction must use explicit slice
    indices with no gaps, never rely on append order."""
    present = set(int(i) for i in present_indices)
    expected = set(range(expected_depth))
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        raise ValueError(
            f"Incomplete/incorrect slice indices for reconstruction. "
            f"Missing: {missing}. Unexpected: {extra}."
        )
