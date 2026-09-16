"""File-level dataset / local-cache integrity validation.

Motivation (incident): a local Colab cache could have the correct NUMBER of case
directories while still being incomplete (missing or empty NIfTI files). Counting
directories is therefore NOT a sufficient integrity check. This module verifies
that every required per-case file actually exists and is non-empty, plus the
split/config files, so training is only allowed to start on a complete cache.

Pure stdlib + the split CSVs -- no torch/nibabel import, so it is cheap and safe
to run as an early gate.
"""

import os
from typing import Dict, List, Optional, Sequence

# Files every case must have for the current thesis pipeline (T2W-only zonal
# anatomy segmentation): the T2 image and the reader-1 anatomy mask.
REQUIRED_CASE_FILES = ("t2.nii.gz", "t2_anatomy_reader1.nii.gz")


def _ids_from_csv(csv_path: str, id_column: str = "ID", zero_pad: int = 3) -> List[str]:
    import csv

    ids: List[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if id_column not in (reader.fieldnames or []):
            raise ValueError(f"Column '{id_column}' not in {csv_path}: {reader.fieldnames}")
        for row in reader:
            ids.append(str(int(row[id_column])).zfill(zero_pad))
    return ids


def verify_case_files(
    case_root: str,
    case_ids: Sequence[str],
    required_files: Sequence[str] = REQUIRED_CASE_FILES,
    min_bytes: int = 1,
) -> Dict[str, object]:
    """Verify each case dir under `case_root` contains every required file and
    that each file is at least `min_bytes` bytes (catches empty/partial copies).

    Returns {"ok": bool, "n_cases": int, "n_ok": int, "missing": {pid: [files]}}
    where `missing` lists, per case, the required files that are absent, empty,
    or whose case directory does not exist.
    """
    missing: Dict[str, List[str]] = {}
    n_ok = 0
    for pid in case_ids:
        case_dir = os.path.join(case_root, pid)
        if not os.path.isdir(case_dir):
            missing[pid] = ["<case directory missing>"]
            continue
        bad: List[str] = []
        for fname in required_files:
            fpath = os.path.join(case_dir, fname)
            try:
                if not os.path.exists(fpath) or os.path.getsize(fpath) < min_bytes:
                    bad.append(fname)
            except OSError:
                bad.append(fname)
        if bad:
            missing[pid] = bad
        else:
            n_ok += 1
    return {
        "ok": len(missing) == 0,
        "n_cases": len(case_ids),
        "n_ok": n_ok,
        "missing": missing,
    }


def verify_dataset_cache(
    dataset_root: str,
    train_csv: str,
    valid_csv: str,
    required_files: Sequence[str] = REQUIRED_CASE_FILES,
    test_dir: Optional[str] = None,
    test_ids: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Validate a dataset root (or local cache) at FILE level for the official
    train+val split (and optionally the test cases).

    `dataset_root` is the directory that directly contains the case folders
    (e.g. .../prostate158_train/train). `train_csv`/`valid_csv` are the official
    split CSVs. Returns a report dict:
      {"ok", "train": <verify_case_files>, "val": <...>, "test": <...|None>,
       "split_files_present": bool, "problems": [str, ...]}

    `ok` is True only when the split CSVs exist AND every train + val (+ test,
    when requested) case has all required, non-empty files.
    """
    problems: List[str] = []

    split_files_present = os.path.exists(train_csv) and os.path.exists(valid_csv)
    if not split_files_present:
        problems.append(f"Missing split CSV(s): train={train_csv} exists={os.path.exists(train_csv)}, "
                        f"valid={valid_csv} exists={os.path.exists(valid_csv)}")
        return {"ok": False, "train": None, "val": None, "test": None,
                "split_files_present": False, "problems": problems}

    train_ids = _ids_from_csv(train_csv)
    val_ids = _ids_from_csv(valid_csv)

    train_report = verify_case_files(dataset_root, train_ids, required_files)
    val_report = verify_case_files(dataset_root, val_ids, required_files)

    test_report = None
    if test_ids is not None or test_dir is not None:
        t_root = test_dir or dataset_root
        t_ids = list(test_ids) if test_ids is not None else sorted(
            d for d in os.listdir(t_root) if os.path.isdir(os.path.join(t_root, d))
        )
        test_report = verify_case_files(t_root, t_ids, required_files)

    if not train_report["ok"]:
        problems.append(f"TRAIN incomplete: {len(train_report['missing'])} case(s) missing required files.")
    if not val_report["ok"]:
        problems.append(f"VAL incomplete: {len(val_report['missing'])} case(s) missing required files.")
    if test_report is not None and not test_report["ok"]:
        problems.append(f"TEST incomplete: {len(test_report['missing'])} case(s) missing required files.")

    ok = train_report["ok"] and val_report["ok"] and (test_report is None or test_report["ok"])
    return {
        "ok": ok,
        "train": train_report,
        "val": val_report,
        "test": test_report,
        "split_files_present": True,
        "problems": problems,
    }


def format_cache_report(report: Dict[str, object], max_list: int = 10) -> str:
    """Human-readable one-block summary of a verify_dataset_cache() report."""
    lines = []
    ok = report.get("ok")
    lines.append("DATASET CACHE INTEGRITY: " + ("OK" if ok else "INCOMPLETE"))
    for split in ("train", "val", "test"):
        rep = report.get(split)
        if rep is None:
            continue
        lines.append(f"  {split}: {rep['n_ok']}/{rep['n_cases']} cases complete")
        miss = rep.get("missing", {})
        if miss:
            shown = list(miss.items())[:max_list]
            for pid, files in shown:
                lines.append(f"    - {pid}: missing {files}")
            if len(miss) > max_list:
                lines.append(f"    ... and {len(miss) - max_list} more")
    for p in report.get("problems", []):
        lines.append("  ! " + p)
    return "\n".join(lines)
