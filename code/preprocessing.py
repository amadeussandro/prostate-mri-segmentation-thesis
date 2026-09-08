"""Preprocessing utilities for 2D Prostate MRI Segmentation.

This module provides preprocessing routines for MRI slice extraction, intensity
normalization, mask binarization (specifically handling non-standard lesion labels
such as value 3.0 found in t2_tumor_reader1), multi-modal channel stacking, and
region of interest (ROI) cropping.
"""

from typing import Optional, Tuple
import numpy as np


def binarize_mask(mask: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Convert raw mask array to a clean binary label array (0 or 1).

    Handles non-standard integer or floating-point label values, such as the
    verified value 3.0 present in Prostate158 't2_tumor_reader1.nii.gz'.

    Parameters
    ----------
    mask : np.ndarray
        Input segmentation mask array.
    threshold : float, default=0.0
        Values strictly greater than this threshold are marked as foreground (1).

    Returns
    -------
    np.ndarray
        Binary mask array of type np.uint8 with values in {0, 1}.
    """
    binary = (mask > threshold).astype(np.uint8)
    # TODO: Add optional multi-class mapping if distinguishing PI-RADS grades or anatomical zones
    return binary


def normalize_intensity(
    image: np.ndarray,
    method: str = "percentile_zscore",
    lower_percentile: float = 0.5,
    upper_percentile: float = 99.5,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalize intensity values of a 2D MRI slice.

    Parameters
    ----------
    image : np.ndarray
        Input 2D MRI slice or multi-channel slice.
    method : str, default="percentile_zscore"
        Normalization strategy:
        - 'percentile_zscore': Clip to [lower_percentile, upper_percentile], then z-score normalize.
        - 'minmax': Scale intensities linearly to [0, 1].
        - 'zscore': Standard zero-mean, unit-variance normalization.
    lower_percentile : float, default=0.5
        Lower percentile cutoff to eliminate extreme scanner low-intensity artifacts.
    upper_percentile : float, default=99.5
        Upper percentile cutoff to eliminate extreme hyperintense artifacts.
    eps : float, default=1e-8
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalized image array of type np.float32.
    """
    img = image.astype(np.float32)

    # Robust clipping against intensity outliers
    p_low = np.percentile(img, lower_percentile)
    p_high = np.percentile(img, upper_percentile)
    img_clipped = np.clip(img, p_low, p_high)

    if method == "percentile_zscore":
        mean = np.mean(img_clipped)
        std = np.std(img_clipped)
        return (img_clipped - mean) / (std + eps)
    elif method == "minmax":
        min_v = np.min(img_clipped)
        max_v = np.max(img_clipped)
        return (img_clipped - min_v) / (max_v - min_v + eps)
    elif method == "zscore":
        mean = np.mean(img)
        std = np.std(img)
        return (img - mean) / (std + eps)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # TODO: Evaluate Nyul histogram standardization for multi-patient cross-site harmonization


def stack_modalities(
    t2_slice: np.ndarray, adc_slice: np.ndarray, dwi_slice: np.ndarray
) -> np.ndarray:
    """Stack co-registered T2, ADC, and DWI 2D slices into a 3-channel tensor.

    Parameters
    ----------
    t2_slice : np.ndarray
        2D slice from T2 volume, shape (H, W).
    adc_slice : np.ndarray
        2D slice from ADC volume, shape (H, W).
    dwi_slice : np.ndarray
        2D slice from DWI volume, shape (H, W).

    Returns
    -------
    np.ndarray
        Stacked array of shape (3, H, W).

    Raises
    ------
    ValueError
        If slice dimensions do not match.
    """
    if not (t2_slice.shape == adc_slice.shape == dwi_slice.shape):
        raise ValueError(
            f"Shape mismatch across modalities: T2 {t2_slice.shape}, "
            f"ADC {adc_slice.shape}, DWI {dwi_slice.shape}"
        )

    stacked = np.stack([t2_slice, adc_slice, dwi_slice], axis=0).astype(np.float32)
    # TODO: Add support for configurable modality subsets (e.g. T2+ADC bpMRI only)
    return stacked


def crop_prostate_roi(
    image: np.ndarray,
    target_shape: Tuple[int, int] = (160, 160),
    center: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Crop an in-plane region of interest (ROI) centered at prostate gland or image center.

    Parameters
    ----------
    image : np.ndarray
        Input 2D array of shape (H, W) or multi-channel (C, H, W).
    target_shape : Tuple[int, int], default=(160, 160)
        Desired output height and width.
    center : Optional[Tuple[int, int]], default=None
        Center coordinate (cy, cx). If None, uses geometric center of the array.

    Returns
    -------
    np.ndarray
        Cropped array of target_shape.
    """
    is_channel_first = image.ndim == 3
    h, w = image.shape[1:] if is_channel_first else image.shape[:2]
    th, tw = target_shape

    if center is None:
        cy, cx = h // 2, w // 2
    else:
        cy, cx = center

    y1 = max(0, cy - th // 2)
    y2 = min(h, y1 + th)
    x1 = max(0, cx - tw // 2)
    x2 = min(w, x1 + tw)

    # Adjust bounding box if clipping boundary is reached
    if (y2 - y1) < th:
        y1 = max(0, y2 - th)
    if (x2 - x1) < tw:
        x1 = max(0, x2 - tw)

    if is_channel_first:
        return image[:, y1:y2, x1:x2]
    return image[y1:y2, x1:x2]

    # TODO: Implement dynamic center-of-mass cropping guided by t2_anatomy_reader1 mask

