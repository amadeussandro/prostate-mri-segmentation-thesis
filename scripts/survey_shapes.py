"""One-off utility: survey native in-plane (H, W) shapes and z-spacing across
the official training cohort, to pick a sensible default crop/pad target_size
for the baseline research config. Not part of any gate; informational only.
"""
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import nibabel as nib
import numpy as np
import pandas as pd

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "prostate158_train")
TRAIN_CSV = os.path.join(DATASET_DIR, "train.csv")
VALID_CSV = os.path.join(DATASET_DIR, "valid.csv")


def main():
    ids = []
    for csv_path in (TRAIN_CSV, VALID_CSV):
        df = pd.read_csv(csv_path)
        ids.extend(str(int(v)).zfill(3) for v in df["ID"].tolist())

    shapes = []
    zspacings = []
    for pid in ids:
        path = os.path.join(DATASET_DIR, "train", pid, "t2.nii.gz")
        img = nib.load(path)
        shapes.append(img.shape[:2])
        zspacings.append(img.header.get_zooms()[2])

    hs = [s[0] for s in shapes]
    ws = [s[1] for s in shapes]
    print(f"n cases: {len(ids)}")
    print(f"H: min={min(hs)} max={max(hs)} median={np.median(hs)}")
    print(f"W: min={min(ws)} max={max(ws)} median={np.median(ws)}")
    print(f"unique shapes: {sorted(set(shapes))}")
    print(f"z spacing: min={min(zspacings):.3f} max={max(zspacings):.3f} median={np.median(zspacings):.3f}")


if __name__ == "__main__":
    main()
