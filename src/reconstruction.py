"""2D -> 3D reconstruction with explicit slice indices and inverse geometric
transforms, restoring predictions (or ground-truth round-trips) to original
voxel space with the source affine/header preserved.

This is the most error-prone step in the pipeline (PDF section 7.4 item 7)
and the flagship spatial-verification gate (round-trip Dice must equal 1.0
on ground truth, with no model involved) lives in scripts/roundtrip_test.py,
built directly on top of this module.
"""

from typing import Dict, Sequence, Tuple

import nibabel as nib
import numpy as np

from src.geometry import (
    VolumeGeometry,
    assert_complete_slice_indices,
    read_geometry,
    verify_original_space_compatibility,
    verify_reconstruction_compatibility,
)
from src.transforms import CaseTransformMeta, invert_preprocessing_mask


def reconstruct_volume_from_slices(
    slices: Sequence[np.ndarray],
    slice_indices: Sequence[int],
    expected_depth: int,
) -> np.ndarray:
    """Assemble 2D label-map slices into a (H, W, D) volume by EXPLICIT index,
    never by list/append order. Raises if any slice index is missing or
    duplicated (completeness assertion).

    Parameters
    ----------
    slices : Sequence[np.ndarray]
        2D arrays of identical (H, W) shape (in the *preprocessed* space).
    slice_indices : Sequence[int]
        The axial index each slice belongs to, same order as `slices`.
    expected_depth : int
        The full depth (D) of the preprocessed volume this case was sliced
        from (i.e. `len(dataset)` restricted to this patient).
    """
    assert_complete_slice_indices(slice_indices, expected_depth)

    h, w = slices[0].shape
    dtype = slices[0].dtype
    volume = np.zeros((h, w, expected_depth), dtype=dtype)
    for arr, k in zip(slices, slice_indices):
        if arr.shape != (h, w):
            raise ValueError(f"Slice at index {k} has shape {arr.shape}, expected {(h, w)}")
        volume[:, :, k] = arr

    return volume


def reconstruct_case_to_original_space(
    slices: Sequence[np.ndarray],
    slice_indices: Sequence[int],
    transform_meta: CaseTransformMeta,
    source_geometry: VolumeGeometry,
) -> nib.Nifti1Image:
    """Full reconstruction pipeline for one case: explicit-index assembly in
    the preprocessed space -> invert preprocessing geometric transforms in
    reverse order (nearest-neighbour, since this is a label map) -> attach
    the ORIGINAL source affine/header -> return a ready-to-save NIfTI image.

    This function is used both by the round-trip fidelity test (ground truth
    in, ground truth back out, must be Dice = 1.0) and by real evaluation
    (model predictions in, original-space prediction volume out).
    """
    preprocessed_depth = transform_meta.post_resample_depth

    volume_preprocessed_space = reconstruct_volume_from_slices(
        slices=slices,
        slice_indices=slice_indices,
        expected_depth=preprocessed_depth,
    )

    volume_original_space = invert_preprocessing_mask(volume_preprocessed_space, transform_meta)

    verify_reconstruction_compatibility(volume_original_space.shape, source_geometry)

    # Preserve source affine/header exactly -- never default to np.eye(4).
    out_img = nib.Nifti1Image(
        volume_original_space.astype(np.uint8),
        affine=source_geometry.affine,
    )
    out_img.header.set_zooms(source_geometry.zooms)

    # Checklist item 9: verify the geometry actually attached to the output
    # NIfTI object -- read back from its own header, not from the Python
    # variables used to build it -- is fully compatible with the untouched
    # source geometry (shape, affine, spacing, orientation). This is a real
    # post-construction check, not a tautology: NIfTI headers store the
    # affine in float32 sform/qform fields, so this also catches any
    # precision loss introduced by nibabel's own on-disk representation.
    reconstructed_geometry = read_geometry(out_img)
    verify_original_space_compatibility(reconstructed_geometry, source_geometry)

    return out_img


def dice_per_class(pred: np.ndarray, gt: np.ndarray, class_ids: Sequence[int]) -> Dict[int, float]:
    """Volume-level Dice per class, computed on integer label volumes."""
    scores = {}
    for c in class_ids:
        p = pred == c
        t = gt == c
        intersection = np.logical_and(p, t).sum()
        denom = p.sum() + t.sum()
        scores[c] = 1.0 if denom == 0 else float(2.0 * intersection / denom)
    return scores
