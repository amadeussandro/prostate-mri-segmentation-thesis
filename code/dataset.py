"""Dataset abstraction for Prostate MRI 2D slice extraction and batch loading.

This module provides two dataset classes for Prostate MRI research:
1. Prostate2DDataset: Legacy / exploratory multi-channel (T2, ADC, DWI) binary
   tumor segmentation dataset.
2. ProstateZonal2DDataset: Core thesis dataset for multi-class prostate zonal
   anatomy segmentation (Central Gland CG/TZ + Peripheral Zone PZ) using
   T2-weighted MRI alone.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import nibabel as nib
import numpy as np

from code.preprocessing import binarize_mask, normalize_intensity, stack_modalities

# Safe PyTorch import: allow module to function as standalone or torch.utils.data.Dataset
try:
    from torch.utils.data import Dataset as BaseDataset  # type: ignore
except ImportError:
    BaseDataset = object


class Prostate2DDataset(BaseDataset):
    """Dataset of 2D axial slices extracted from 3D Prostate MRI volumes (Tumor Segmentation).

    Parameters
    ----------
    dataset_root : str
        Path to the dataset directory containing patient folders (e.g. prostate158_train/train).
    patient_ids : Sequence[str]
        List of patient identifier strings (e.g. ['020', '024']).
    modalities : Sequence[str], default=('t2', 'adc', 'dwi')
        MRI sequences to load and stack along channels.
    mask_name : str, default='t2_tumor_reader1.nii.gz'
        Target segmentation mask filename.
    anatomy_name : str, default='t2_anatomy_reader1.nii.gz'
        Anatomy segmentation mask filename (used when slice_sampling='prostate_only').
    slice_sampling : str, default='all'
        Slice filtering policy:
        - 'all': Include all slices from the 3D volume.
        - 'tumor_only': Only include slices with positive tumor labels (mask > 0).
        - 'prostate_only': Only include slices with positive prostate anatomy (anatomy > 0).
    transform : Optional[Any], default=None
        Optional transformation / augmentation callable.
    cache_data : bool, default=True
        Whether to cache normalized 3D volumes in memory to prevent repetitive disk I/O.
    """

    def __init__(
        self,
        dataset_root: str,
        patient_ids: Sequence[str],
        modalities: Sequence[str] = ("t2", "adc", "dwi"),
        mask_name: str = "t2_tumor_reader1.nii.gz",
        anatomy_name: str = "t2_anatomy_reader1.nii.gz",
        slice_sampling: str = "all",
        transform: Optional[Any] = None,
        cache_data: bool = True,
    ) -> None:
        self.dataset_root = dataset_root
        self.patient_ids = list(patient_ids)
        self.modalities = list(modalities)
        self.mask_name = mask_name
        self.anatomy_name = anatomy_name
        self.transform = transform
        self.cache_data = cache_data

        valid_samplings = ("all", "tumor_only", "prostate_only")
        if slice_sampling not in valid_samplings:
            raise ValueError(
                f"Invalid slice_sampling: '{slice_sampling}'. "
                f"Must be one of: {valid_samplings}"
            )
        self.slice_sampling = slice_sampling

        # Internal in-memory cache: pid -> (normalized_modality_dict, binarized_mask)
        self._patient_cache: Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray]] = {}

        # List of (patient_id, slice_index) tuples
        self.samples: List[Tuple[str, int]] = []
        self._build_index()

    def _load_and_validate_patient(
        self, pid: str
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray]]:
        """Load raw 3D volumes, validate shape consistency, normalize, and binarize.

        Parameters
        ----------
        pid : str
            Patient identifier string.

        Returns
        -------
        Tuple[Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray]]
            - Normalized 3D modalities dict: {mod_name: 3D float32 array}
            - Binarized 3D tumor mask: 3D uint8 array with values in {0, 1}
            - Optional 3D anatomy mask (if slice_sampling == 'prostate_only')

        Raises
        ------
        FileNotFoundError
            If patient directory or any required NIfTI file is missing.
        ValueError
            If spatial shapes differ across modalities or masks.
        """
        patient_dir = os.path.join(self.dataset_root, pid)
        if not os.path.isdir(patient_dir):
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        # 1. Load required MRI modalities
        raw_modalities: Dict[str, np.ndarray] = {}
        for mod in self.modalities:
            fpath = os.path.join(patient_dir, f"{mod}.nii.gz")
            if not os.path.exists(fpath):
                raise FileNotFoundError(
                    f"Required modality file not found for patient '{pid}': {fpath}"
                )
            img = nib.load(fpath)
            raw_modalities[mod] = img.get_fdata()

        # 2. Load required tumor mask
        mask_path = os.path.join(patient_dir, self.mask_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Required tumor mask file not found for patient '{pid}': {mask_path}"
            )
        raw_mask = nib.load(mask_path).get_fdata()

        # 3. Validate shape consistency across modalities and mask
        shapes = {mod: raw_modalities[mod].shape for mod in self.modalities}
        shapes["mask"] = raw_mask.shape
        first_shape = shapes[self.modalities[0]]

        for name, shape in shapes.items():
            if shape != first_shape:
                raise ValueError(
                    f"Shape mismatch in patient '{pid}': "
                    f"{', '.join(f'{k}={s}' for k, s in shapes.items())}"
                )

        # 4. Normalize full 3D volumes using preprocessing.normalize_intensity
        normalized_modalities: Dict[str, np.ndarray] = {
            mod: normalize_intensity(raw_modalities[mod])
            for mod in self.modalities
        }

        # 5. Binarize tumor mask (handles label 3.0 in t2_tumor_reader1)
        binarized_tumor_mask = binarize_mask(raw_mask)

        # 6. Load anatomy mask if needed for 'prostate_only' slice filtering
        anatomy_vol: Optional[np.ndarray] = None
        if self.slice_sampling == "prostate_only":
            anatomy_path = os.path.join(patient_dir, self.anatomy_name)
            if not os.path.exists(anatomy_path):
                raise FileNotFoundError(
                    f"Required anatomy mask not found for 'prostate_only' sampling: {anatomy_path}"
                )
            anatomy_vol = nib.load(anatomy_path).get_fdata()
            if anatomy_vol.shape != first_shape:
                raise ValueError(
                    f"Anatomy mask shape {anatomy_vol.shape} mismatch with "
                    f"MRI shape {first_shape} for patient '{pid}'"
                )

        return normalized_modalities, binarized_tumor_mask, anatomy_vol

    def _get_patient_volumes(
        self, pid: str
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Retrieve processed 3D volumes for a patient from cache or disk."""
        if pid in self._patient_cache:
            return self._patient_cache[pid]

        norm_mods, bin_mask, _ = self._load_and_validate_patient(pid)
        if self.cache_data:
            self._patient_cache[pid] = (norm_mods, bin_mask)
        return norm_mods, bin_mask

    def _build_index(self) -> None:
        """Scan patient directories, validate volumes, and build dynamic slice index."""
        for pid in self.patient_ids:
            norm_mods, bin_mask, anat_vol = self._load_and_validate_patient(pid)

            if self.cache_data:
                self._patient_cache[pid] = (norm_mods, bin_mask)

            # Dynamic slice count based on actual 3D volume shape (z-dimension)
            num_slices = norm_mods[self.modalities[0]].shape[2]

            for s_idx in range(num_slices):
                if self.slice_sampling == "all":
                    self.samples.append((pid, s_idx))
                elif self.slice_sampling == "tumor_only":
                    # Keep only slices with positive tumor mask
                    if np.any(bin_mask[:, :, s_idx] > 0):
                        self.samples.append((pid, s_idx))
                elif self.slice_sampling == "prostate_only":
                    # Keep only slices with positive prostate anatomy mask
                    if anat_vol is not None and np.any(anat_vol[:, :, s_idx] > 0):
                        self.samples.append((pid, s_idx))

    def __len__(self) -> int:
        """Return total number of 2D slice samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and return a single 2D multi-channel slice and corresponding mask.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'image': 3-channel slice array of shape (3, H, W), float32
            - 'mask': Binary mask array of shape (1, H, W), float32
            - 'patient_id': Patient ID string
            - 'slice_idx': Integer slice index
        """
        if idx >= len(self.samples) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")

        pid, s_idx = self.samples[idx]
        norm_mods, bin_mask = self._get_patient_volumes(pid)

        # Extract 2D axial slice for each modality
        modality_slices = [norm_mods[mod][:, :, s_idx] for mod in self.modalities]

        # Stack modalities along channel dimension (3, H, W)
        if len(self.modalities) == 3:
            stacked_image = stack_modalities(
                modality_slices[0], modality_slices[1], modality_slices[2]
            )
        else:
            stacked_image = np.stack(modality_slices, axis=0).astype(np.float32)

        # Extract 2D mask slice and add channel dimension (1, H, W) as float32
        mask_slice = (bin_mask[:, :, s_idx][np.newaxis, ...] > 0).astype(np.float32)

        sample = {
            "image": stacked_image,
            "mask": mask_slice,
            "patient_id": pid,
            "slice_idx": s_idx,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class ProstateZonal2DDataset(BaseDataset):
    """Dataset of 2D axial slices for multi-class prostate zonal anatomy segmentation.

    Tailored directly to the core thesis research task: segmenting the Central Gland
    (CG/TZ) and Peripheral Zone (PZ) from T2-weighted MRI volumes alone.

    Target labels:
    - 0: Background
    - 1: Anatomy Foreground Class 1
    - 2: Anatomy Foreground Class 2

    Parameters
    ----------
    dataset_root : str
        Path to the dataset directory containing patient folders (e.g. prostate158_train/train).
    patient_ids : Sequence[str]
        List of patient identifier strings (e.g. ['020', '024']).
    image_name : str, default='t2.nii.gz'
        T2W MRI volume filename to load (T2W ONLY).
    mask_name : str, default='t2_anatomy_reader1.nii.gz'
        Multi-class anatomical segmentation mask filename (values in {0, 1, 2}).
    slice_sampling : str, default='all'
        Slice filtering policy:
        - 'all': Include all axial slices from the 3D volume.
        - 'prostate_only': Only include slices with positive anatomy mask (mask > 0).
    transform : Optional[Any], default=None
        Optional transformation / augmentation callable.
    cache_data : bool, default=True
        Whether to cache normalized 3D volumes in memory to prevent repetitive disk I/O.
    """

    def __init__(
        self,
        dataset_root: str,
        patient_ids: Sequence[str],
        image_name: str = "t2.nii.gz",
        mask_name: str = "t2_anatomy_reader1.nii.gz",
        slice_sampling: str = "all",
        transform: Optional[Any] = None,
        cache_data: bool = True,
    ) -> None:
        self.dataset_root = dataset_root
        self.patient_ids = list(patient_ids)
        self.image_name = image_name
        self.mask_name = mask_name
        self.transform = transform
        self.cache_data = cache_data

        valid_samplings = ("all", "prostate_only")
        if slice_sampling not in valid_samplings:
            raise ValueError(
                f"Invalid slice_sampling: '{slice_sampling}'. "
                f"Must be one of: {valid_samplings}"
            )
        self.slice_sampling = slice_sampling

        # Internal in-memory cache: pid -> (normalized_t2, raw_anatomy_mask_int64)
        self._patient_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # List of (patient_id, slice_index) tuples
        self.samples: List[Tuple[str, int]] = []
        self._build_index()

    def _load_and_validate_patient(
        self, pid: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load raw T2W volume and anatomy mask, validate geometry and labels, normalize T2.

        Parameters
        ----------
        pid : str
            Patient identifier string.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            - Normalized 3D T2W volume: float32 array of shape (H, W, D)
            - Multi-class 3D anatomy mask: int64 array of shape (H, W, D) with labels in {0, 1, 2}

        Raises
        ------
        FileNotFoundError
            If patient directory, T2W image file, or anatomy mask file is missing.
        ValueError
            If dimensions are not 3D, shapes/affines mismatch, or unexpected labels are present.
        """
        patient_dir = os.path.join(self.dataset_root, pid)
        if not os.path.isdir(patient_dir):
            raise FileNotFoundError(f"Patient directory not found: {patient_dir}")

        # 1. Load T2W MRI volume (T2W ONLY)
        img_path = os.path.join(patient_dir, self.image_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Required T2W image file not found for patient '{pid}': {img_path}"
            )
        img_nii = nib.load(img_path)
        raw_img = img_nii.get_fdata()

        # 2. Load multi-class anatomy mask
        mask_path = os.path.join(patient_dir, self.mask_name)
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f"Required anatomy mask file not found for patient '{pid}': {mask_path}"
            )
        mask_nii = nib.load(mask_path)
        raw_mask = mask_nii.get_fdata()

        # 3. Validate 3D dimensionality
        if raw_img.ndim != 3:
            raise ValueError(
                f"T2W image for patient '{pid}' must be a 3D volume, got {raw_img.ndim}D with shape {raw_img.shape}"
            )
        if raw_mask.ndim != 3:
            raise ValueError(
                f"Anatomy mask for patient '{pid}' must be a 3D volume, got {raw_mask.ndim}D with shape {raw_mask.shape}"
            )

        # 4. Validate shape consistency (H, W, D)
        if raw_img.shape != raw_mask.shape:
            raise ValueError(
                f"Shape mismatch for patient '{pid}': T2W shape {raw_img.shape} vs mask shape {raw_mask.shape}"
            )

        # 5. Validate affine spatial alignment (Gerbang 1: geometri harus identik)
        if not np.allclose(img_nii.affine, mask_nii.affine, atol=1e-4):
            raise ValueError(
                f"Affine mismatch for patient '{pid}': T2W affine does not match anatomy mask affine."
            )

        # 6. Validate multi-class label integrity (expected subset of {0, 1, 2})
        unique_labels = np.unique(raw_mask)
        if not np.all(np.isclose(unique_labels, np.round(unique_labels))):
            raise ValueError(
                f"Anatomy mask for patient '{pid}' contains non-integer values: {unique_labels.tolist()}"
            )
        int_labels = set(int(round(v)) for v in unique_labels)
        valid_labels = {0, 1, 2}
        if not int_labels.issubset(valid_labels):
            invalid_found = int_labels - valid_labels
            raise ValueError(
                f"Invalid label(s) {sorted(list(invalid_found))} found in anatomy mask for patient '{pid}'. "
                f"Expected labels to be a subset of [0, 1, 2], found: {sorted(list(int_labels))}."
            )

        # 7. Normalize full 3D T2W volume using existing normalize_intensity (percentile clipping + z-score)
        norm_t2 = normalize_intensity(raw_img)

        # Convert mask to int64 without binarization or alteration
        anatomy_mask_int64 = np.round(raw_mask).astype(np.int64)

        return norm_t2, anatomy_mask_int64

    def _get_patient_volumes(
        self, pid: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieve processed 3D T2 volume and anatomy mask from cache or disk."""
        if pid in self._patient_cache:
            return self._patient_cache[pid]

        norm_t2, anatomy_mask = self._load_and_validate_patient(pid)
        if self.cache_data:
            self._patient_cache[pid] = (norm_t2, anatomy_mask)
        return norm_t2, anatomy_mask

    def _build_index(self) -> None:
        """Scan patient directories, validate volumes, and build dynamic slice index."""
        for pid in self.patient_ids:
            norm_t2, anatomy_mask = self._load_and_validate_patient(pid)

            if self.cache_data:
                self._patient_cache[pid] = (norm_t2, anatomy_mask)

            # Dynamic slice count from actual volume shape (z-dimension)
            num_slices = norm_t2.shape[2]

            for s_idx in range(num_slices):
                if self.slice_sampling == "all":
                    self.samples.append((pid, s_idx))
                elif self.slice_sampling == "prostate_only":
                    # Only include slices where prostate anatomy is present (> 0)
                    if np.any(anatomy_mask[:, :, s_idx] > 0):
                        self.samples.append((pid, s_idx))

    def __len__(self) -> int:
        """Return total number of 2D slice samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load and return a single 2D T2 slice and corresponding multi-class mask.

        Parameters
        ----------
        idx : int
            Index of the sample.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - 'image': 1-channel T2 slice of shape (1, H, W), float32
            - 'mask': Multi-class mask array of shape (H, W), int64 with labels in [0, 1, 2]
            - 'patient_id': Patient ID string
            - 'slice_idx': Integer slice index
        """
        if idx >= len(self.samples) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")

        pid, s_idx = self.samples[idx]
        norm_t2, anatomy_mask = self._get_patient_volumes(pid)

        # Extract 2D axial slice: shape (1, H, W), float32
        image_slice = norm_t2[:, :, s_idx][np.newaxis, ...].astype(np.float32)

        # Extract 2D mask slice: shape (H, W), int64 (for torch.nn.CrossEntropyLoss)
        mask_slice = anatomy_mask[:, :, s_idx].astype(np.int64)

        sample = {
            "image": image_slice,
            "mask": mask_slice,
            "patient_id": pid,
            "slice_idx": s_idx,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def load_patient_volume(patient_dir: str, file_name: str) -> np.ndarray:
    """Load a 3D NIfTI volume array from disk.

    Parameters
    ----------
    patient_dir : str
        Directory containing the patient NIfTI files.
    file_name : str
        Name of the .nii.gz file to load.

    Returns
    -------
    np.ndarray
        3D NumPy array of voxel intensities.
    """
    fpath = os.path.join(patient_dir, file_name)
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"NIfTI file not found: {fpath}")
    return nib.load(fpath).get_fdata()
