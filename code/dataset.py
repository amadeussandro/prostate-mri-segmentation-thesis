"""Dataset abstraction for Prostate MRI 2D slice extraction and batch loading.

This module indexes co-registered 3D NIfTI volumes (T2, ADC, DWI) and segmentation
masks across patient directories, extracting 2D axial slices for training and evaluation.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import nibabel as nib
import numpy as np

# Safe PyTorch import: allow module to function as standalone or torch.utils.data.Dataset
try:
    from torch.utils.data import Dataset as BaseDataset
except ImportError:
    BaseDataset = object  # type: ignore


class Prostate2DDataset(BaseDataset):
    """Dataset of 2D axial slices extracted from 3D Prostate MRI volumes.

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
    slice_sampling : str, default='all'
        Slice filtering policy:
        - 'all': Include all slices (e.g. 24 slices per volume).
        - 'prostate_only': Filter slices using t2_anatomy_reader1 (slices containing prostate).
        - 'tumor_only': Only include slices with positive tumor labels.
    transform : Optional[Any], default=None
        Optional transformation / augmentation callable.
    """

    def __init__(
        self,
        dataset_root: str,
        patient_ids: Sequence[str],
        modalities: Sequence[str] = ("t2", "adc", "dwi"),
        mask_name: str = "t2_tumor_reader1.nii.gz",
        slice_sampling: str = "all",
        transform: Optional[Any] = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.patient_ids = list(patient_ids)
        self.modalities = list(modalities)
        self.mask_name = mask_name
        self.slice_sampling = slice_sampling
        self.transform = transform

        # List of (patient_id, slice_index) tuples
        self.samples: List[Tuple[str, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        """Scan patient directories and construct the slice index."""
        # Scaffolding placeholder: construct slice index based on verified file dimensions (24 slices)
        for pid in self.patient_ids:
            patient_dir = os.path.join(self.dataset_root, pid)
            if not os.path.isdir(patient_dir):
                continue

            # In the verified Prostate158 dataset, volumes contain 24 axial slices
            # TODO: Inspect actual z-dimension dynamically from NIfTI header per volume
            num_slices = 24
            for s_idx in range(num_slices):
                # TODO: Implement slice_sampling filtering ('prostate_only' / 'tumor_only')
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
            - 'image': 3-channel slice array of shape (3, H, W)
            - 'mask': Binary mask array of shape (1, H, W)
            - 'patient_id': Patient ID string
            - 'slice_idx': Integer slice index
        """
        if idx >= len(self.samples) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")

        pid, s_idx = self.samples[idx]
        patient_dir = os.path.join(self.dataset_root, pid)

        # Scaffolding placeholder for volume reading
        # TODO: Implement cached NIfTI volume loading to prevent repetitive disk I/O per slice
        modality_slices = []
        for mod in self.modalities:
            fpath = os.path.join(patient_dir, f"{mod}.nii.gz")
            if os.path.exists(fpath):
                img = nib.load(fpath)
                data = img.get_fdata()
                modality_slices.append(data[:, :, s_idx])
            else:
                # Fallback zero-array for missing modality during testing/scaffolding
                modality_slices.append(np.zeros((270, 270), dtype=np.float32))

        stacked_image = np.stack(modality_slices, axis=0).astype(np.float32)

        # Load target mask
        mask_path = os.path.join(patient_dir, self.mask_name)
        if os.path.exists(mask_path):
            mask_data = nib.load(mask_path).get_fdata()
            raw_mask_slice = mask_data[:, :, s_idx]
            # Binarize (handles label 3.0 in t2_tumor_reader1)
            binary_mask = (raw_mask_slice > 0).astype(np.uint8)[np.newaxis, ...]
        else:
            binary_mask = np.zeros((1, 270, 270), dtype=np.uint8)

        sample = {
            "image": stacked_image,
            "mask": binary_mask,
            "patient_id": pid,
            "slice_idx": s_idx,
        }

        if self.transform is not None:
            # TODO: Apply augmentation pipeline (e.g. Albumentations / MONAI / torchvision)
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

