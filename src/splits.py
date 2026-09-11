"""Official Prostate158 case-level data split (119 train / 20 validation / 19 test).

The split MUST be determined at the patient/case level, before any 3D->2D
slicing happens -- slicing turns one patient into dozens of highly-correlated
2D samples, so a slice-level split would leak the same patient's anatomy
across sets. This module only ever returns lists of case IDs; slicing is a
downstream concern (src/dataset.py), applied independently per set.
"""

import os
import warnings
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class OfficialSplit:
    train_ids: List[str]
    val_ids: List[str]
    test_ids: Optional[List[str]]  # None if the 19-case test archive is not available locally


def _ids_from_csv(csv_path: str, id_column: str = "ID", zero_pad: int = 3) -> List[str]:
    df = pd.read_csv(csv_path)
    if id_column not in df.columns:
        raise ValueError(f"Expected column '{id_column}' in {csv_path}, found: {list(df.columns)}")
    return [str(int(v)).zfill(zero_pad) for v in df[id_column].tolist()]


def load_official_split(
    train_csv: str,
    valid_csv: str,
    test_csv: Optional[str] = None,
    test_dir: Optional[str] = None,
) -> OfficialSplit:
    """Load the official case-level Prostate158 split from the dataset's own CSVs.

    Parameters
    ----------
    train_csv, valid_csv : str
        Paths to the official train.csv / valid.csv files distributed with the
        139-case training archive (DOI 10.5281/zenodo.6481141).
    test_csv : Optional[str]
        Path to a CSV listing the 19 held-out test case IDs, if the separate
        test archive (DOI 10.5281/zenodo.6592345) has been obtained. If not
        provided (or the file doesn't exist), test_ids is returned as None
        with a clear warning -- this is NOT silently treated as "no test set
        needed", it is a flagged, visible gap.
    test_dir : Optional[str]
        If test_csv is not available but the test case directories are present
        on disk, pass the directory to derive test IDs from folder names instead.

    Raises
    ------
    ValueError
        If train/val/test sets are not pairwise disjoint (leakage), or if the
        official case counts (119/20) don't match what was loaded.
    """
    train_ids = _ids_from_csv(train_csv)
    val_ids = _ids_from_csv(valid_csv)

    if len(train_ids) != 119:
        raise ValueError(
            f"Expected 119 official training cases, found {len(train_ids)} in {train_csv}."
        )
    if len(val_ids) != 20:
        raise ValueError(
            f"Expected 20 official validation cases, found {len(val_ids)} in {valid_csv}."
        )

    test_ids: Optional[List[str]] = None
    if test_csv and os.path.exists(test_csv):
        test_ids = _ids_from_csv(test_csv)
    elif test_dir and os.path.isdir(test_dir):
        test_ids = sorted(
            d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))
        )

    if test_ids is None:
        warnings.warn(
            "The official 19-case Prostate158 TEST archive (DOI 10.5281/zenodo.6592345) "
            "was not found locally. Proceeding with train/validation only. Final RQ-4 "
            "baseline comparison against the official test set cannot be completed until "
            "this archive is obtained.",
            UserWarning,
            stacklevel=2,
        )
    elif len(test_ids) != 19:
        raise ValueError(f"Expected 19 official test cases, found {len(test_ids)}.")

    assert_no_leakage(train_ids, val_ids, test_ids)

    return OfficialSplit(train_ids=train_ids, val_ids=val_ids, test_ids=test_ids)


def assert_no_leakage(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: Optional[List[str]] = None,
) -> None:
    """Assert train/val/test are pairwise disjoint at the case level."""
    train_set, val_set = set(train_ids), set(val_ids)

    overlap_train_val = train_set & val_set
    if overlap_train_val:
        raise ValueError(f"Train/validation case-ID overlap (leakage): {sorted(overlap_train_val)}")

    if test_ids:
        test_set = set(test_ids)
        overlap_train_test = train_set & test_set
        overlap_val_test = val_set & test_set
        if overlap_train_test:
            raise ValueError(f"Train/test case-ID overlap (leakage): {sorted(overlap_train_test)}")
        if overlap_val_test:
            raise ValueError(f"Validation/test case-ID overlap (leakage): {sorted(overlap_val_test)}")
