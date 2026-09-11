"""One-off generator for notebooks/prostate158_experiment_01_colab.ipynb.

Not part of any pipeline gate. Run once locally to (re)build the notebook
file as valid nbformat4 JSON, so cell content doesn't have to be hand-escaped.
Safe to delete after the notebook exists; kept only for reproducible edits.
"""
import json
import os

def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines]}

def code(*lines):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }

cells = []

cells.append(md(
    "# Prostate158 Zonal Segmentation -- Experiment 1 Colab Orchestration",
    "",
    "**This notebook is orchestration only.** It does not contain any pipeline logic --",
    "every substantive step calls the existing `src/` modules or `scripts/*.py` entry",
    "points from the repository, which remains the single source of truth on GitHub.",
    "",
    "- GitHub: source of truth for code (this repo).",
    "- Google Drive: stores the Prostate158 dataset, checkpoints, predictions, and results.",
    "  **The dataset is never stored in GitHub.**",
    "- This notebook does **not** start the full 119-case Experiment-1 training run.",
    "  The last section prepares it (Drive-backed output paths) but leaves the actual",
    "  `run_training(...)` call commented out.",
    "",
    "Run the cells in order, top to bottom.",
))

cells.append(md("## 1. Check GPU"))
cells.append(code(
    "import subprocess",
    "import torch",
    "",
    "print(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)",
    "print('torch:', torch.__version__)",
    "print('CUDA available:', torch.cuda.is_available())",
    "if torch.cuda.is_available():",
    "    print('GPU:', torch.cuda.get_device_name(0))",
))

cells.append(md("## 2. Mount Google Drive"))
cells.append(code(
    "from google.colab import drive",
    "",
    "drive.mount('/content/drive')",
))

cells.append(md(
    "## 3. Define paths",
    "",
    "Adjust `DRIVE_ROOT` only if your dataset lives somewhere other than",
    "`/content/drive/MyDrive/THESIS_PROSTATE158`.",
))
cells.append(code(
    "import os",
    "",
    "REPO_URL = 'https://github.com/amadeussandro/prostate-mri-segmentation-thesis.git'",
    "REPO_DIR = '/content/thesis-project'",
    "",
    "DRIVE_ROOT = '/content/drive/MyDrive/THESIS_PROSTATE158'",
    "DRIVE_DATASET_DIR = os.path.join(DRIVE_ROOT, 'dataset')",
    "DRIVE_RESULTS_DIR = os.path.join(DRIVE_ROOT, 'results')",
    "",
    "print('Repo dir:        ', REPO_DIR)",
    "print('Drive dataset dir:', DRIVE_DATASET_DIR)",
    "print('Drive results dir:', DRIVE_RESULTS_DIR)",
))

cells.append(md("## 4. Clone or pull the GitHub repository"))
cells.append(code(
    "import subprocess",
    "",
    "if not os.path.isdir(REPO_DIR):",
    "    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)",
    "",
    "os.chdir(REPO_DIR)",
    "subprocess.run(['git', 'checkout', 'main'], check=True)",
    "subprocess.run(['git', 'pull', 'origin', 'main'], check=True)",
    "subprocess.run(['git', 'log', '-1', '--oneline'], check=True)",
))

cells.append(md("## 5. Install requirements"))
cells.append(code(
    "!pip install -q -r requirements.txt",
))

cells.append(md(
    "## 6. Link the Drive dataset into the repository",
    "",
    "**Why this step exists (the actual Colab-compatibility fix):** every config",
    "in this repo (`configs/config_020.yaml`, `configs/config_baseline.yaml`, ...)",
    "uses a path like `dataset_root: \"dataset/prostate158_train/train\"`. That path is",
    "**relative to the current working directory**, and `dataset/` is git-ignored --",
    "it is never present in a fresh clone. Rather than editing any config or any",
    "line of pipeline code, we create one symlink so the existing relative paths",
    "resolve to the real data on Drive. This is the only \"fix\" this Colab setup",
    "needed; no source file was changed.",
))
cells.append(code(
    "assert os.path.isdir(DRIVE_DATASET_DIR), f'Dataset not found on Drive at {DRIVE_DATASET_DIR}'",
    "",
    "link_path = os.path.join(REPO_DIR, 'dataset')",
    "if os.path.islink(link_path):",
    "    os.remove(link_path)",
    "elif os.path.exists(link_path):",
    "    raise RuntimeError(f'{link_path} exists and is not a symlink -- refusing to overwrite.')",
    "",
    "os.symlink(DRIVE_DATASET_DIR, link_path)",
    "print('Linked:', link_path, '->', os.readlink(link_path))",
    "",
    "# Sanity check against the exact relative path configs/config_020.yaml uses.",
    "expected_t2 = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train', '020', 't2.nii.gz')",
    "expected_mask = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train', '020', 't2_anatomy_reader1.nii.gz')",
    "print('t2.nii.gz found:               ', os.path.exists(expected_t2))",
    "print('t2_anatomy_reader1.nii.gz found:', os.path.exists(expected_mask))",
))

cells.append(md(
    "## 7. Inventory check: how much of the official split is on Drive",
    "",
    "The official split is 119 train / 20 validation cases (see `src/splits.py`).",
    "`scripts/check_baseline_config.py` (section 10 below) needs specific cases from",
    "that split to be present, not just case 020. This cell reports coverage up",
    "front instead of failing later with a confusing `FileNotFoundError`.",
))
cells.append(code(
    "import pandas as pd",
    "",
    "train_csv = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train.csv')",
    "valid_csv = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'valid.csv')",
    "train_root = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train')",
    "",
    "if os.path.exists(train_csv) and os.path.exists(valid_csv):",
    "    train_ids = [str(int(v)).zfill(3) for v in pd.read_csv(train_csv)['ID']]",
    "    val_ids = [str(int(v)).zfill(3) for v in pd.read_csv(valid_csv)['ID']]",
    "    all_ids = train_ids + val_ids",
    "    present = [pid for pid in all_ids if os.path.isdir(os.path.join(train_root, pid))]",
    "    missing = [pid for pid in all_ids if pid not in present]",
    "    print(f'Official split: {len(train_ids)} train + {len(val_ids)} val = {len(all_ids)} cases')",
    "    print(f'Present on Drive: {len(present)}/{len(all_ids)}')",
    "    if missing:",
    "        preview = missing[:10]",
    "        print(f'Missing ({len(missing)}): {preview}{\"...\" if len(missing) > 10 else \"\"}')",
    "else:",
    "    print('train.csv/valid.csv not found under the Drive dataset -- cannot check official split coverage.')",
    "    print('Case 020 present:', os.path.isdir(os.path.join(train_root, '020')))",
))

cells.append(md(
    "## 8. Run the repository test suite",
    "",
    "Every test in `tests/` skips cleanly (via `unittest.skipTest`) when a patient",
    "folder it needs isn't present, so this is safe to run regardless of how much",
    "of the dataset has been synced to Drive so far -- it will just skip more if",
    "fewer cases are available. Takes roughly 1-3 minutes.",
))
cells.append(code(
    "!python -m pytest tests/ -q",
))

cells.append(md(
    "## 9. Load Case 020 through the REAL dataset API",
    "",
    "This directly addresses the manual-testing error: `ProstateZonal2DDataset` takes",
    "`dataset_root` (not `data_root`), and it has **no** `split` argument at all --",
    "it only ever takes an explicit `patient_ids` list. Train/val/test split",
    "resolution is a separate concern, handled by `src/splits.py` +",
    "`src/train.py::_resolve_split_ids` (exercised in section 10 below), which",
    "reads `train.csv`/`valid.csv` and hands `ProstateZonal2DDataset` a plain list",
    "of case-ID strings.",
))
cells.append(code(
    "import sys",
    "sys.path.insert(0, REPO_DIR)",
    "",
    "from src.dataset import ProstateZonal2DDataset",
    "",
    "dataset_root = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train')",
    "dataset = ProstateZonal2DDataset(",
    "    dataset_root=dataset_root,   # NOT data_root",
    "    patient_ids=['020'],         # NOT split='train' -- explicit case-ID list only",
    "    slice_sampling='all',",
    ")",
    "",
    "print('Slices for patient 020:', len(dataset))",
    "sample = dataset[0]",
    "print('image shape/dtype:', sample['image'].shape, sample['image'].dtype)",
    "print('mask  shape/dtype:', sample['mask'].shape, sample['mask'].dtype)",
    "",
    "geometry, transform_meta = dataset.get_case_metadata('020')",
    "print('original geometry shape:', geometry.shape)",
    "print('original spacing (mm):  ', geometry.zooms)",
    "print('original orientation:   ', geometry.axcodes)",
))

cells.append(md(
    "## 10. Verify image/mask shapes and labels via the tested, config-driven path",
    "",
    "This is the same chain `scripts/smoke_test_020.py` was built to prove end to end:",
    "load -> spatial verification -> preprocessing -> slicing -> Dataset -> DataLoader",
    "-> 2D U-Net forward pass. No training. Uses `configs/config_020.yaml`.",
))
cells.append(code(
    "!python scripts/smoke_test_020.py",
))

cells.append(md(
    "## 11. Round-trip fidelity gate (no model, ground truth only)",
    "",
    "Requires patients 020, 029, 059, 099, 139 to be present under the Drive dataset.",
    "Must report Dice = 1.0 exactly for every class, on both the identity-shape suite",
    "and the suite matching `configs/config_baseline.yaml`'s real (442, 442) crop/pad.",
))
cells.append(code(
    "!python scripts/roundtrip_test.py",
))

cells.append(md(
    "## 12. Baseline (Experiment 1) structural readiness check",
    "",
    "Resolves the official 119/20 split from `train.csv`/`valid.csv`, builds the",
    "dataset for a small subset, and checks batching/model shapes -- **no training**.",
    "Needs the specific cases in that small subset (see section 7's inventory) to",
    "actually be present on Drive; if section 7 reported missing cases, this may",
    "raise `FileNotFoundError` for a case that simply hasn't been synced yet -- that",
    "is a data-availability issue, not a code bug.",
))
cells.append(code(
    "!python scripts/check_baseline_config.py",
))

cells.append(md(
    "## 13. Confirm no lesion mask can be accidentally selected",
    "",
    "`configs/config_baseline.yaml` declares `data.target_mask`, and `src/train.py`",
    "honors it -- but `ProstateZonal2DDataset` independently rejects any mask whose",
    "labels aren't a subset of `{0,1,2}`, so even an explicit attempt to point at",
    "`t2_tumor_reader1.nii.gz` (labels `{0,3}`) fails loudly. These are the regression",
    "tests proving both directions.",
))
cells.append(code(
    "!python -m pytest tests/test_dataset.py::TestTargetMaskConfigWiring -v",
))

cells.append(md(
    "## 14. Prepare Experiment 1 baseline execution (does NOT start training)",
    "",
    "Redirects this run's checkpoints/predictions/results to Google Drive so they",
    "survive a Colab session reset, **without editing the committed",
    "`configs/config_baseline.yaml`** or any pipeline code -- the override is written",
    "to a throwaway config file outside the repo (`/content/config_baseline_colab.yaml`),",
    "which is never committed to GitHub.",
    "",
    "**The actual training call is left commented out.** This is the final checkpoint",
    "before Experiment 1 -- everything above (sections 6-13) must be green first.",
))
cells.append(code(
    "import yaml",
    "from src.utils import load_config",
    "",
    "config = load_config('configs/config_baseline.yaml')",
    "",
    "drive_output_dir = os.path.join(DRIVE_RESULTS_DIR, 'exp01_baseline')",
    "os.makedirs(drive_output_dir, exist_ok=True)",
    "config['experiment']['output_dir'] = drive_output_dir",
    "",
    "colab_config_path = '/content/config_baseline_colab.yaml'  # outside the repo -- never committed",
    "with open(colab_config_path, 'w') as f:",
    "    yaml.safe_dump(config, f)",
    "",
    "print('Colab-specific config written to:', colab_config_path)",
    "print('experiment.output_dir ->', config['experiment']['output_dir'])",
    "print()",
    "print('NOT starting training yet. When sections 6-13 above are all green and')",
    "print('the full official split is confirmed present on Drive, run:')",
    "print()",
    "print(\"    from src.train import run_training\")",
    "print(f\"    run_training('{colab_config_path}')\")",
    "",
    "# from src.train import run_training",
    "# run_training(colab_config_path)   # <-- intentionally commented out",
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"name": "prostate158_experiment_01_colab.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "prostate158_experiment_01_colab.ipynb")
out_path = os.path.abspath(out_path)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
    f.write("\n")

print("Wrote", out_path)
