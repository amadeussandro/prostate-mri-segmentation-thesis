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

cells.append(md(
    "## 1. Check GPU",
    "",
    "`!nvidia-smi` is the standard Colab-compatible invocation -- it runs through the",
    "notebook's shell (the same one `!nvidia-smi` on its own uses), so it doesn't",
    "depend on `nvidia-smi` being resolvable via the Jupyter kernel process's own",
    "`PATH` the way a plain Python `subprocess.run([...])` call would. If this ever",
    "runs on a CPU-only runtime, `!nvidia-smi` just prints an error line below --",
    "it won't raise a Python exception or stop the next cells from running.",
))
cells.append(code(
    "!nvidia-smi",
))
cells.append(code(
    "import torch",
    "",
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

cells.append(md(
    "## 15. Resume Experiment 1 (recover after a Colab disconnect)",
    "",
    "> ### SKIP FOR CURRENT WORKFLOW",
    ">",
    "> **Do not run Section 15.** Experiment 1 training is already finished/stopped,",
    "> and the workflow has moved on to **evaluation (Section 16)**. This section is",
    "> retained only as documentation of the resume/recovery mechanism for future",
    "> reference. Running it would *resume training*, which is not what we want now.",
    "> Go straight to **Section 16. Evaluate Experiment 1 baseline**.",
    "",
    "Use this section **only** to continue an Experiment-1 run that was interrupted",
    "(e.g. the GPU backend was reclaimed on a usage limit). It does **not** change",
    "the experiment: same official 119/20 split, T2W-only, `t2_anatomy_reader1`,",
    "`ProstateUNet2D` (1->3, init_features=32, dropout 0.2), 442x442 crop/pad,",
    "per-volume normalization, AdamW lr=0.001 wd=0.0001, no scheduler, seed=42,",
    "total epochs=100.",
    "",
    "`run_training(config_path, resume_from=CKPT)` restores the checkpoint's model,",
    "optimizer, and `best_metric`, then continues from the checkpoint's stored",
    "epoch + 1 through epoch 100 -- it never restarts at epoch 1, and `best_model.pt`",
    "is still only overwritten when validation Dice beats the restored best.",
    "",
    "**Honest caveats for the thesis writeup:**",
    "- The surviving checkpoint is `best_model.pt`, whose stored epoch is the last",
    "  epoch that *improved* validation Dice (e.g. 60), which may be earlier than the",
    "  last epoch actually reached before the disconnect (e.g. 64). Resuming therefore",
    "  re-runs the few non-improving epochs after the best one. This is correct and",
    "  safe, just not bit-identical to an uninterrupted 1..100 run.",
    "- RNG state (sampler generator, dropout) is not stored in the checkpoint format,",
    "  so the resumed segment is not bitwise-reproducible against an uninterrupted run.",
    "  The experiment *design* (data, model, hyperparameters, seed) is unchanged.",
))
cells.append(code(
    "import os",
    "import torch",
    "",
    "RESUME_CKPT = '/content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt'",
    "",
    "assert os.path.exists(RESUME_CKPT), f'Checkpoint not found: {RESUME_CKPT}'",
    "_ckpt = torch.load(RESUME_CKPT, map_location='cpu', weights_only=False)",
    "print('Checkpoint keys:      ', sorted(_ckpt.keys()))",
    "print('Checkpoint epoch:     ', _ckpt['epoch'])",
    "print('Restored best_metric: ', _ckpt['best_metric'])",
    "print(f\"Will resume at epoch {int(_ckpt['epoch']) + 1} and train through 100.\")",
))
cells.append(md(
    "Prepare the same Drive-backed config used in Section 14 (idempotent -- safe to",
    "run even if Section 14 was not run in this session), then resume. The training",
    "call below **is active** because resuming is the intended recovery action; it",
    "will train epochs (checkpoint_epoch + 1)..100 on the GPU.",
))
cells.append(code(
    "import yaml",
    "from src.utils import load_config",
    "from src.train import run_training",
    "",
    "config = load_config('configs/config_baseline.yaml')",
    "drive_output_dir = os.path.join(DRIVE_RESULTS_DIR, 'exp01_baseline')",
    "os.makedirs(drive_output_dir, exist_ok=True)",
    "config['experiment']['output_dir'] = drive_output_dir",
    "",
    "colab_config_path = '/content/config_baseline_colab.yaml'  # outside the repo -- never committed",
    "with open(colab_config_path, 'w') as f:",
    "    yaml.safe_dump(config, f)",
    "",
    "# Resumes from RESUME_CKPT (defined in the cell above); continues to epoch 100.",
    "summary = run_training(colab_config_path, resume_from=RESUME_CKPT)",
    "print(summary)",
))

cells.append(md(
    "## 16. Evaluate Experiment 1 baseline",
    "",
    "**This section is inference/evaluation ONLY -- it does not train, resume, or",
    "modify the checkpoint.** It runs the existing evaluation entry point",
    "`scripts/run_evaluation.py` (finalized in commit 94b2a8e) against the best",
    "checkpoint from the stopped Experiment-1 run. All evaluation logic lives in",
    "`src/evaluate.py` / `src/metrics.py`; this notebook only orchestrates the call.",
    "",
    "It evaluates the **official 20-case VALIDATION split** (`--split val`). The",
    "official 19-case held-out test archive is not available locally, so we do NOT",
    "substitute it -- run_evaluation would refuse a held-out-test request anyway,",
    "and these validation numbers are explicitly not final test results.",
    "",
    "Evaluation is volume-level, per-case, in original voxel space (the thesis",
    "protocol): checkpoint -> 2D inference -> 2D->3D reconstruction -> inverse",
    "preprocessing -> per-case Dice / IoU / HD95 / ASD / precision / recall, with",
    "mean +/- SD and bootstrap CI.",
    "",
    "Expected output files written to the Drive results directory:",
    "- `eval_val_per_case_metrics.csv`",
    "- `eval_val_summary.csv`",
    "- `eval_val_summary.json`",
))
cells.append(code(
    "import os",
    "",
    "EVAL_CONFIG = 'configs/config_baseline.yaml'",
    "EVAL_CHECKPOINT = '/content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt'",
    "EVAL_OUTPUT_DIR = '/content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline'",
    "EVAL_SPLIT = 'val'  # 20-case validation split; NOT the (unavailable) held-out test set",
    "",
    "print('Experiment 1 baseline evaluation (inference only -- no training):')",
    "print('  config     :', EVAL_CONFIG)",
    "print('  checkpoint :', EVAL_CHECKPOINT)",
    "print('  output dir :', EVAL_OUTPUT_DIR)",
    "print('  split      :', EVAL_SPLIT)",
    "assert os.path.exists(EVAL_CHECKPOINT), f'Checkpoint not found: {EVAL_CHECKPOINT}'",
))
cells.append(code(
    "# Single-line shell command (literal paths) -- matches Sections 8/10-13 and",
    "# avoids any dependency on multi-line `!` handling or `{var}` expansion.",
    "!python scripts/run_evaluation.py --config configs/config_baseline.yaml --checkpoint /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt --output-dir /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline --split val",
))
cells.append(code(
    "# Confirm the expected result files exist after evaluation.",
    "for _fname in ('eval_val_per_case_metrics.csv', 'eval_val_summary.csv', 'eval_val_summary.json'):",
    "    _fpath = os.path.join(EVAL_OUTPUT_DIR, _fname)",
    "    print(('OK  ' if os.path.exists(_fpath) else 'MISSING '), _fpath)",
    "print('\\nExperiment 1 baseline evaluation complete (validation split).')",
))

cells.append(md(
    "## 17. Visualize Experiment 1 baseline (qualitative, inference only)",
    "",
    "**Inference/visualization ONLY -- no training, no checkpoint changes.** Runs",
    "`scripts/visualize_baseline_cases.py`, which reuses the SAME production path as",
    "Section 16's evaluation (shared `src.evaluate.predict_case_original_space`:",
    "checkpoint -> preprocessing -> per-slice inference -> 2D->3D reconstruction ->",
    "original voxel space). All logic lives in `scripts/` + `src/`; this notebook",
    "only orchestrates.",
    "",
    "It renders axial panels (T2 | Ground Truth | Prediction | GT/Pred overlay |",
    "error map) for representative, central, and max-disagreement slices. Labels are",
    "strictly **Class 1 / Class 2** -- the integer->anatomy mapping is UNVERIFIED, so",
    "no CG/TZ/PZ names are used. This is the **validation** split, not the held-out",
    "test set.",
    "",
    "Two selection modes below (run whichever you want):",
    "- **worst/best by metrics CSV** -- data-driven, reads eval_val_per_case_metrics.csv.",
    "- **explicit patients** -- e.g. 60 77 87 (verify against the CSV first).",
    "",
    "Figures + `summary/visualization_summary.json` are written under the Drive",
    "results dir and are intentionally NOT committed to Git (see .gitignore).",
))
cells.append(code(
    "# Data-driven selection: worst 3 and best 3 validation cases by mean foreground Dice.",
    "!python scripts/visualize_baseline_cases.py --config configs/config_baseline.yaml --checkpoint /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt --metrics-csv /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/eval_val_per_case_metrics.csv --output-dir /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/visualizations --split val --mode worst --n 3",
    "!python scripts/visualize_baseline_cases.py --config configs/config_baseline.yaml --checkpoint /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt --metrics-csv /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/eval_val_per_case_metrics.csv --output-dir /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/visualizations --split val --mode best --n 3",
))
cells.append(code(
    "# Explicit patients (optional). Verify these ids exist in the metrics CSV first.",
    "!python scripts/visualize_baseline_cases.py --config configs/config_baseline.yaml --checkpoint /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt --metrics-csv /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/eval_val_per_case_metrics.csv --output-dir /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/visualizations --split val --patients 60 77 87",
))
cells.append(code(
    "# Peek at the generated visualization summary.",
    "import json, os",
    "_summary = '/content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/visualizations/summary/visualization_summary.json'",
    "if os.path.exists(_summary):",
    "    with open(_summary) as f:",
    "        _payload = json.load(f)",
    "    print('split:', _payload['split'], '| held-out test?', _payload['is_official_held_out_test'])",
    "    print('label mapping:', _payload['label_mapping_status'])",
    "    for _c in _payload['cases']:",
    "        print(f\"  patient {_c['patient_id']}: slices {_c['selected_slice_indices']} \"",
    "              f\"-> {len(_c['figures'])} figure(s)\")",
    "else:",
    "    print('No summary yet -- run a visualization cell above first.')",
))

cells.append(md(
    "## 18. Audit official held-out TEST archive (READ-ONLY, no evaluation)",
    "",
    "**Audit/readiness check only -- no model, no inference, no metrics, no**",
    "**training.** Runs `scripts/audit_test_archive.py`, which extracts the official",
    "test ZIP into an isolated directory (the ZIP is left untouched), then reports:",
    "case count + ids, per-case presence of `t2.nii.gz` and `t2_anatomy_reader1.nii.gz`,",
    "NIfTI shape/spacing/orientation/label values, overlap against the 119 train +",
    "20 validation cases, and whether `load_official_split` resolves exactly 19 test",
    "ids from the extracted directory.",
    "",
    "This is what tells us whether the final Experiment-1 held-out test evaluation",
    "can compute Dice/IoU/HD95/ASD (it needs `t2_anatomy_reader1` ground truth to",
    "exist). It does NOT run the evaluation and does NOT touch the checkpoint.",
    "",
    "The extracted dataset and audit JSON stay on Drive and are never committed to Git.",
))
cells.append(code(
    "!python scripts/audit_test_archive.py --zip /content/drive/MyDrive/THESIS_PROSTATE158/dataset/test/prostate158_test.zip --extract-dir /content/drive/MyDrive/THESIS_PROSTATE158/dataset/test/extracted --train-csv dataset/prostate158_train/train.csv --valid-csv dataset/prostate158_train/valid.csv",
))
cells.append(md(
    "**Only if the audit above reports READY** (19 cases, all with T2 + ground",
    "truth, no train/val overlap, loader resolves 19 test ids): the final",
    "Experiment-1 held-out test evaluation is run by setting `data.test_dir` in the",
    "config to the audit's reported case-root and running `run_evaluation` on the",
    "held-out test split. That evaluation step is Section 19 below.",
))

cells.append(md(
    "## 19. FINAL Experiment 1 held-out TEST evaluation (inference only, one-time)",
    "",
    "**Authorized one-time final evaluation of the already-trained Experiment-1",
    "baseline on the official 19-case held-out test split.** Inference only -- no",
    "training, no resume, no tuning, no checkpoint modification. It calls the exact",
    "production path used for validation (`src.evaluate.run_evaluation`), just with",
    "`split='test'` and `data.test_dir` pointed at the audited, isolated test",
    "case-root. Metrics are volume-level, per-case, in original voxel space:",
    "Dice / IoU / HD95 / ASD / precision / recall, with mean +/- SD and bootstrap CI.",
    "",
    "Safety built into this cell: it verifies the checkpoint is epoch 77 and",
    "READ-ONLY, refuses to overwrite any existing `eval_test_*` file (stops instead),",
    "and writes results as `eval_test_*` -- never touching the `eval_val_*` files,",
    "the checkpoint, or any other Experiment-1 artifact. Labels stay Background /",
    "Class 1 / Class 2 (mapping still UNVERIFIED). The extracted test data and the",
    "result CSV/JSON live on Drive and are NOT committed to Git.",
))
cells.append(code(
    "import os, yaml, torch",
    "from src.utils import load_config",
    "from src.evaluate import run_evaluation",
    "",
    "TEST_CASE_ROOT = '/content/drive/MyDrive/THESIS_PROSTATE158/dataset/test/extracted/prostate158_test/test'",
    "CHECKPOINT = '/content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt'",
    "OUTPUT_DIR = '/content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline'",
    "",
    "# 1) checkpoint exists, readable, epoch 77, weights untouched (read-only load).",
    "assert os.path.exists(CHECKPOINT), f'Checkpoint not found: {CHECKPOINT}'",
    "_ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)",
    "print('checkpoint epoch      :', _ckpt.get('epoch'))",
    "print('checkpoint best metric:', _ckpt.get('best_metric'))",
    "assert int(_ckpt.get('epoch', -1)) == 77, 'Expected epoch-77 checkpoint.'",
    "del _ckpt  # do not hold/modify weights",
    "",
    "# 2) refuse to overwrite existing final test results -- STOP if present.",
    "_existing = [f for f in ('eval_test_per_case_metrics.csv', 'eval_test_summary.csv', 'eval_test_summary.json')",
    "             if os.path.exists(os.path.join(OUTPUT_DIR, f))]",
    "if _existing:",
    "    raise SystemExit(f'STOP: test result file(s) already exist, refusing to overwrite: {_existing}. '",
    "                     'Move/rename them first if a re-run is truly intended.')",
    "",
    "# 3) build a Colab config: baseline config + isolated test_dir + Drive output.",
    "#    (test_dir is environment-specific, so it is injected here, never committed.)",
    "assert os.path.isdir(TEST_CASE_ROOT), f'Test case-root not found: {TEST_CASE_ROOT}'",
    "cfg = load_config('configs/config_baseline.yaml')",
    "cfg['data']['test_dir'] = TEST_CASE_ROOT",
    "cfg['experiment']['output_dir'] = OUTPUT_DIR",
    "colab_cfg_path = '/content/config_baseline_test_colab.yaml'",
    "with open(colab_cfg_path, 'w') as f:",
    "    yaml.safe_dump(cfg, f)",
    "",
    "# 4) run the PRODUCTION evaluation on the held-out test split (inference only).",
    "result = run_evaluation(config_path=colab_cfg_path, checkpoint_path=CHECKPOINT,",
    "                        output_dir=OUTPUT_DIR, split='test')",
    "print('\\nFinal test outputs:')",
    "print('  per-case CSV:', result['per_case_csv'])",
    "print('  summary  CSV:', result['summary_csv'])",
    "print('  summary JSON:', result['summary_json'])",
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
