"""Configuration-driven preprocessing pipeline for the zonal segmentation thesis.

Every geometric operation here is applied identically to the image and the
mask (same crop/pad window, same orientation transform, same z-resample
factor). The only per-modality difference is interpolation order:
  - image: smooth interpolation (order=1, i.e. trilinear/bilinear)
  - mask:  ALWAYS nearest-neighbour (order=0) -- never linear on a label map.

`preprocess_case` returns a `CaseTransformMeta` object that records everything
needed to invert the geometric steps later (src/reconstruction.py), in
reverse order: pad/crop -> z-resample -> orientation.

Ablation knobs (RQ-1 / Experiment 2), all independently controllable via
PreprocessingConfig:
  - orientation:      standardize to RAS (always on; this is not an ablation
                       axis in the thesis, it is a correctness requirement)
  - normalization:     "per_volume" | "global"
  - z_resample:        True | False
  - spatial_mode:      "crop_pad" | "resize"
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

from src.geometry import VolumeGeometry, read_geometry
from src.preprocessing import normalize_intensity


@dataclass
class PreprocessingConfig:
    normalization: str = "per_volume"  # "per_volume" | "global"
    z_resample: bool = False
    target_spacing_z: float = 3.0
    spatial_mode: str = "crop_pad"  # "crop_pad" | "resize"
    target_size: Tuple[int, int] = (256, 256)
    global_stats: Optional[Dict[str, float]] = None  # {"mean":..., "std":..., "p_low":..., "p_high":...}

    @staticmethod
    def from_dict(d: dict) -> "PreprocessingConfig":
        d = d or {}
        return PreprocessingConfig(
            normalization=d.get("normalization", "per_volume"),
            z_resample=d.get("z_resample", False),
            target_spacing_z=d.get("target_spacing_z", 3.0),
            spatial_mode=d.get("spatial_mode", "crop_pad"),
            target_size=tuple(d.get("target_size", [256, 256])),
            global_stats=d.get("global_stats"),
        )


@dataclass
class AxisPadCrop:
    op: str  # "pad" or "crop" or "none"
    native_size: int
    target_size: int
    start: int  # for "pad": amount of padding before; for "crop": crop start index


@dataclass
class CaseTransformMeta:
    original_geometry: VolumeGeometry
    orientation_fwd: np.ndarray  # ornt array: original -> RAS
    reoriented_shape: Tuple[int, int, int]
    pad_crop_h: AxisPadCrop
    pad_crop_w: AxisPadCrop
    spatial_mode: str
    target_size: Tuple[int, int]
    z_resample_applied: bool
    z_resample_factor: float
    pre_resample_depth: int
    post_resample_depth: int


def _invert_ornt(ornt: np.ndarray) -> np.ndarray:
    """Invert a nibabel orientation-transform array (signed permutation)."""
    ornt = np.asarray(ornt)
    inv = np.zeros_like(ornt)
    for i in range(ornt.shape[0]):
        j = int(ornt[i, 0])
        s = ornt[i, 1]
        inv[j, 0] = i
        inv[j, 1] = s
    return inv


def _standardize_orientation(data: np.ndarray, affine: np.ndarray, order: int):
    """Reorient a volume to RAS. `order` is unused for orientation (pure
    permutation/flip, no interpolation) but kept for interface symmetry."""
    orig_ornt = nib.io_orientation(affine)
    ras_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
    reoriented = nib.orientations.apply_orientation(data, transform)
    return reoriented, transform


def _unstandardize_orientation(data: np.ndarray, orientation_fwd: np.ndarray) -> np.ndarray:
    inv_transform = _invert_ornt(orientation_fwd)
    return nib.orientations.apply_orientation(data, inv_transform)


def _compute_axis_pad_crop(native_size: int, target_size: int) -> AxisPadCrop:
    if native_size == target_size:
        return AxisPadCrop(op="none", native_size=native_size, target_size=target_size, start=0)
    if native_size < target_size:
        pad_total = target_size - native_size
        pad_before = pad_total // 2
        return AxisPadCrop(op="pad", native_size=native_size, target_size=target_size, start=pad_before)
    crop_total = native_size - target_size
    crop_start = crop_total // 2
    return AxisPadCrop(op="crop", native_size=native_size, target_size=target_size, start=crop_start)


def _apply_pad_crop_2d(volume_hwd: np.ndarray, pc_h: AxisPadCrop, pc_w: AxisPadCrop, pad_value: float) -> np.ndarray:
    """Apply an identical in-plane crop/pad window to a (H, W, D) volume."""
    vol = volume_hwd

    if pc_h.op == "crop":
        vol = vol[pc_h.start:pc_h.start + pc_h.target_size, :, :]
    if pc_w.op == "crop":
        vol = vol[:, pc_w.start:pc_w.start + pc_w.target_size, :]

    pad_before_h = pc_h.start if pc_h.op == "pad" else 0
    pad_after_h = (pc_h.target_size - pc_h.native_size - pad_before_h) if pc_h.op == "pad" else 0
    pad_before_w = pc_w.start if pc_w.op == "pad" else 0
    pad_after_w = (pc_w.target_size - pc_w.native_size - pad_before_w) if pc_w.op == "pad" else 0

    if pad_before_h or pad_after_h or pad_before_w or pad_after_w:
        vol = np.pad(
            vol,
            ((pad_before_h, pad_after_h), (pad_before_w, pad_after_w), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
    return vol


def _invert_pad_crop_2d(volume_hwd: np.ndarray, pc_h: AxisPadCrop, pc_w: AxisPadCrop) -> np.ndarray:
    """Invert `_apply_pad_crop_2d` exactly, recovering the native in-plane size."""
    vol = volume_hwd

    # Undo padding first (padding was applied after cropping in forward pass).
    if pc_h.op == "pad":
        vol = vol[pc_h.start:pc_h.start + pc_h.native_size, :, :]
    if pc_w.op == "pad":
        vol = vol[:, pc_w.start:pc_w.start + pc_w.native_size, :]

    # Undo cropping by padding zeros back into the cropped-away region.
    if pc_h.op == "crop":
        pad_before = pc_h.start
        pad_after = pc_h.native_size - pc_h.target_size - pad_before
        vol = np.pad(vol, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant")
    if pc_w.op == "crop":
        pad_before = pc_w.start
        pad_after = pc_w.native_size - pc_w.target_size - pad_before
        vol = np.pad(vol, ((0, 0), (pad_before, pad_after), (0, 0)), mode="constant")

    return vol


def _resize_2d(volume_hwd: np.ndarray, target_size: Tuple[int, int], order: int) -> np.ndarray:
    h, w, d = volume_hwd.shape
    th, tw = target_size
    zoom_factors = (th / h, tw / w, 1.0)
    out = ndimage.zoom(volume_hwd, zoom_factors, order=order, mode="nearest" if order == 0 else "constant")
    # Guard against off-by-one rounding from ndimage.zoom.
    return out[:th, :tw, :]


def _resize_2d_back(volume_hwd: np.ndarray, native_size: Tuple[int, int], order: int) -> np.ndarray:
    h, w, d = volume_hwd.shape
    nh, nw = native_size
    zoom_factors = (nh / h, nw / w, 1.0)
    out = ndimage.zoom(volume_hwd, zoom_factors, order=order, mode="nearest" if order == 0 else "constant")
    return out[:nh, :nw, :]


def preprocess_case(
    t2_nii: nib.Nifti1Image,
    mask_nii: nib.Nifti1Image,
    config: PreprocessingConfig,
) -> Tuple[np.ndarray, np.ndarray, CaseTransformMeta]:
    """Run the full preprocessing pipeline on one case's T2 volume + anatomy mask.

    Returns
    -------
    (image_volume, mask_volume, transform_meta)
        image_volume : float32 (H, W, D) normalized T2W volume
        mask_volume  : int64 (H, W, D) label volume in {0, 1, 2}
        transform_meta : everything needed to invert back to original space
    """
    original_geometry = read_geometry(t2_nii)
    t2_data = t2_nii.get_fdata()
    mask_data = np.round(mask_nii.get_fdata()).astype(np.int64)

    # 1. Orientation standardization -> RAS, identical transform for image and mask.
    t2_ras, orientation_fwd = _standardize_orientation(t2_data, t2_nii.affine, order=1)
    mask_ras, _ = _standardize_orientation(mask_data, mask_nii.affine, order=0)
    reoriented_shape = t2_ras.shape

    # 2. Intensity normalization (image only; mask untouched).
    if config.normalization == "per_volume":
        t2_norm = normalize_intensity(t2_ras)
    elif config.normalization == "global":
        if not config.global_stats:
            raise ValueError("normalization='global' requires config.global_stats to be set.")
        p_low = config.global_stats["p_low"]
        p_high = config.global_stats["p_high"]
        mean = config.global_stats["mean"]
        std = config.global_stats["std"]
        clipped = np.clip(t2_ras.astype(np.float32), p_low, p_high)
        t2_norm = (clipped - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown normalization strategy: {config.normalization}")

    # 3. Optional z-axis resampling (image: linear; mask: nearest-neighbour).
    pre_resample_depth = reoriented_shape[2]
    if config.z_resample:
        current_spacing_z = original_geometry.zooms[2]
        factor = current_spacing_z / config.target_spacing_z
        new_depth = max(1, int(round(pre_resample_depth * factor)))
        t2_norm = ndimage.zoom(t2_norm, (1.0, 1.0, new_depth / pre_resample_depth), order=1, mode="constant")
        mask_ras = ndimage.zoom(mask_ras, (1.0, 1.0, new_depth / pre_resample_depth), order=0, mode="nearest")
        post_resample_depth = t2_norm.shape[2]
        z_resample_factor = new_depth / pre_resample_depth
    else:
        post_resample_depth = pre_resample_depth
        z_resample_factor = 1.0

    # 4. In-plane spatial standardization (fixes cross-patient batching).
    h, w = t2_norm.shape[0], t2_norm.shape[1]
    th, tw = config.target_size

    if config.spatial_mode == "crop_pad":
        pc_h = _compute_axis_pad_crop(h, th)
        pc_w = _compute_axis_pad_crop(w, tw)
        t2_final = _apply_pad_crop_2d(t2_norm, pc_h, pc_w, pad_value=float(t2_norm.min()))
        mask_final = _apply_pad_crop_2d(mask_ras, pc_h, pc_w, pad_value=0)
    elif config.spatial_mode == "resize":
        pc_h = AxisPadCrop(op="resize", native_size=h, target_size=th, start=0)
        pc_w = AxisPadCrop(op="resize", native_size=w, target_size=tw, start=0)
        t2_final = _resize_2d(t2_norm, (th, tw), order=1)
        mask_final = _resize_2d(mask_ras, (th, tw), order=0)
    else:
        raise ValueError(f"Unknown spatial_mode: {config.spatial_mode}")

    mask_final = np.round(mask_final).astype(np.int64)
    t2_final = t2_final.astype(np.float32)

    transform_meta = CaseTransformMeta(
        original_geometry=original_geometry,
        orientation_fwd=orientation_fwd,
        reoriented_shape=reoriented_shape,
        pad_crop_h=pc_h,
        pad_crop_w=pc_w,
        spatial_mode=config.spatial_mode,
        target_size=config.target_size,
        z_resample_applied=config.z_resample,
        z_resample_factor=z_resample_factor,
        pre_resample_depth=pre_resample_depth,
        post_resample_depth=post_resample_depth,
    )

    return t2_final, mask_final, transform_meta


def invert_preprocessing_mask(volume_hwd: np.ndarray, meta: CaseTransformMeta) -> np.ndarray:
    """Invert the geometric preprocessing steps for a MASK/label volume, in
    reverse order (pad/crop -> z-resample -> orientation), always using
    nearest-neighbour interpolation. Returns an int64 volume in the original
    on-disk orientation and shape.
    """
    vol = np.round(volume_hwd).astype(np.int64)

    # Undo in-plane crop/pad or resize.
    if meta.spatial_mode == "crop_pad":
        vol = _invert_pad_crop_2d(vol, meta.pad_crop_h, meta.pad_crop_w)
    elif meta.spatial_mode == "resize":
        native_hw = (meta.pad_crop_h.native_size, meta.pad_crop_w.native_size)
        vol = _resize_2d_back(vol, native_hw, order=0)
    else:
        raise ValueError(f"Unknown spatial_mode in transform_meta: {meta.spatial_mode}")
    vol = np.round(vol).astype(np.int64)

    # Undo z-resampling (nearest-neighbour to preserve integer labels).
    if meta.z_resample_applied and meta.post_resample_depth != meta.pre_resample_depth:
        vol = ndimage.zoom(
            vol.astype(np.float64),
            (1.0, 1.0, meta.pre_resample_depth / vol.shape[2]),
            order=0,
            mode="nearest",
        )
        vol = np.round(vol).astype(np.int64)

    # Crop/pad along z back to the exact reoriented depth if rounding drifted.
    if vol.shape[2] != meta.reoriented_shape[2]:
        d = vol.shape[2]
        target_d = meta.reoriented_shape[2]
        if d > target_d:
            vol = vol[:, :, :target_d]
        else:
            vol = np.pad(vol, ((0, 0), (0, 0), (0, target_d - d)), mode="constant")

    # Undo orientation standardization -> back to original on-disk orientation.
    vol = _unstandardize_orientation(vol, meta.orientation_fwd)
    vol = np.round(vol).astype(np.int64)

    return vol


def compute_global_intensity_stats(
    dataset_root: str,
    case_ids,
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    image_name: str = "t2.nii.gz",
) -> Dict[str, float]:
    """Compute pooled intensity statistics from a set of TRAINING cases only.

    Never call this with validation/test case IDs -- that would leak
    evaluation-set information into normalization statistics.
    """
    import os

    all_values = []
    for pid in case_ids:
        path = os.path.join(dataset_root, pid, image_name)
        data = nib.load(path).get_fdata().astype(np.float32).ravel()
        # Subsample large volumes to keep memory bounded; still representative.
        if data.size > 200_000:
            rng = np.random.default_rng(42)
            data = rng.choice(data, size=200_000, replace=False)
        all_values.append(data)

    pooled = np.concatenate(all_values)
    p_low = float(np.percentile(pooled, lower_percentile))
    p_high = float(np.percentile(pooled, upper_percentile))
    clipped = np.clip(pooled, p_low, p_high)
    return {
        "p_low": p_low,
        "p_high": p_high,
        "mean": float(np.mean(clipped)),
        "std": float(np.std(clipped)),
    }
