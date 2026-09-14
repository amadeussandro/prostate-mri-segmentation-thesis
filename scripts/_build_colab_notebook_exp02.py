"""One-off generator for notebooks/prostate158_experiment_02_colab.ipynb.

Not part of any pipeline gate. Run once locally to (re)build the Experiment 2
(preprocessing ablation, RQ-1) orchestration notebook as valid nbformat4 JSON,
so cell content doesn't have to be hand-escaped. Mirrors the structure and
conventions of scripts/_build_colab_notebook.py (Experiment 1). Safe to delete
after the notebook exists; kept only for reproducible edits.

The notebook is ORCHESTRATION ONLY: every substantive step calls existing
src/ modules or scripts/*.py entry points. Only preprocessing varies across the
three arms; model, optimizer, loss, seed, official split, target mask, slicing,
and the evaluation definition are held exactly as audited (Exp01 baseline).
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
    "# Prostate158 Zonal Segmentation -- Experiment 2 (Preprocessing Ablation, RQ-1) Colab Orchestration",
    "",
    "**This notebook is orchestration only.** It contains no pipeline logic --",
    "every substantive step calls the existing `src/` modules or `scripts/*.py`",
    "entry points from the repository, which remains the single source of truth on",
    "GitHub.",
    "",
    "**Research question (RQ-1):** what is the effect of preprocessing decisions on",
    "volume-level segmentation performance when model architecture and",
    "hyperparameters are kept constant?",
    "",
    "**Controlled ablation -- ONLY preprocessing varies.** Held fixed at the Exp01",
    "baseline values (see `configs/config_baseline.yaml`): ProstateUNet2D",
    "(1->3, init_features=32, dropout 0.2), cross_entropy, AdamW lr=0.001 wd=1e-4,",
    "batch_size=8, num_epochs=100, seed=42, official 119/20 split, target mask",
    "`t2_anatomy_reader1`, axial slicing, and the evaluation definition (volume-level,",
    "per-case, original voxel space). `tests/test_exp02_configs.py` asserts this.",
    "",
    "| Arm | Config | Ablated preprocessing factor |",
    "|---|---|---|",
    "| **A** | `configs/experiments/exp02_a_zresample_on.yaml` | z-axis resampling ON (baseline: OFF) |",
    "| **B** | `configs/experiments/exp02_b_normalization_global.yaml` | global train-only normalization (baseline: per-volume) |",
    "| **C** | `configs/experiments/exp02_c_spatial_resize.yaml` | resize 256x256 (baseline: crop/pad 442x442) |",
    "",
    "> **Note on Arm A.** The Prostate158 cohort already has a uniform ~3.0 mm",
    "> z-spacing, so z-resampling to 3.0 mm is expected to be a *near no-op* on THIS",
    "> dataset. This is preserved deliberately: it documents the ablation axis defined",
    "> in the thesis proposal, and per-case spacing could in principle differ. The",
    "> z-resampling target is intentionally NOT redesigned around this cohort.",
    "",
    "- GitHub: source of truth for code (this repo).",
    "- Google Drive: stores the dataset, checkpoints, and results. **Never in GitHub.**",
    "- Each arm's checkpoint and results go under `results/exp02_preprocessing/<arm>/`",
    "  (mirrored on Drive).",
    "- **Resume-safe for T4 timeouts.** Each arm's training cell auto-detects its",
    "  Drive checkpoint: if none exists it starts fresh at epoch 1; if one exists it",
    "  resumes from `epoch+1` (never restarting at epoch 1) and skips training",
    "  entirely once the checkpoint reaches 100 epochs. Just rerun the same cell",
    "  after a disconnect. Decision logic lives in `src/train.py` (orchestration-only",
    "  notebook).",
    "- **The official 19-case held-out TEST set is untouched during variant selection.**",
    "  This notebook evaluates the VALIDATION split only; held-out test is deferred",
    "  until after a variant is chosen (a separate, one-time step, as in Exp01).",
    "",
    "Run the readiness cells (1-10b) in order. The per-arm training/validation cells",
    "(11 onward) are clearly separated -- run them only after readiness is green.",
))

# ---- 1. GPU ----
cells.append(md(
    "## 1. Check GPU",
    "",
    "`!nvidia-smi` runs through the notebook shell; on a CPU-only runtime it just",
    "prints an error line and does not stop later cells.",
))
cells.append(code("!nvidia-smi"))
cells.append(code(
    "import torch",
    "",
    "print('torch:', torch.__version__)",
    "print('CUDA available:', torch.cuda.is_available())",
    "if torch.cuda.is_available():",
    "    print('GPU:', torch.cuda.get_device_name(0))",
))

# ---- 2. Drive ----
cells.append(md("## 2. Mount Google Drive"))
cells.append(code(
    "from google.colab import drive",
    "",
    "drive.mount('/content/drive')",
))

# ---- 3. Paths ----
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
    "# Experiment 2 results live under results/exp02_preprocessing/<arm>/ (on Drive).",
    "DRIVE_EXP02_DIR = os.path.join(DRIVE_RESULTS_DIR, 'exp02_preprocessing')",
    "EXP02_ARMS = {",
    "    'A': {'name': 'a_zresample_on',        'config': 'configs/experiments/exp02_a_zresample_on.yaml'},",
    "    'B': {'name': 'b_normalization_global','config': 'configs/experiments/exp02_b_normalization_global.yaml'},",
    "    'C': {'name': 'c_spatial_resize',      'config': 'configs/experiments/exp02_c_spatial_resize.yaml'},",
    "}",
    "",
    "print('Repo dir:         ', REPO_DIR)",
    "print('Drive dataset dir:', DRIVE_DATASET_DIR)",
    "print('Drive results dir:', DRIVE_RESULTS_DIR)",
    "print('Drive exp02 dir:  ', DRIVE_EXP02_DIR)",
))

# ---- 4. Clone/pull ----
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

# ---- 5. Requirements ----
cells.append(md("## 5. Install requirements"))
cells.append(code("!pip install -q -r requirements.txt"))

# ---- 6. Link dataset (with optional local runtime cache) ----
cells.append(md(
    "## 6. Link the dataset into the repository (optional local runtime cache)",
    "",
    "Every config uses a path relative to the repo root",
    "(`dataset_root: \"dataset/prostate158_train/train\"`), and `dataset/` is",
    "git-ignored -- never present in a fresh clone. One symlink makes the existing",
    "relative paths resolve to the real data; no config or source file is edited.",
    "",
    "**Optional runtime speedup (`USE_LOCAL_DATASET_CACHE`).** Reading NIfTI",
    "straight from the Drive FUSE mount is slow and flaky, and the dataset is fully",
    "loaded + preprocessed into RAM once at the start of every `run_training`",
    "(including every resume). Copying the dataset ONCE to fast local Colab storage",
    "(`/content/local_dataset`) and pointing the symlink there makes that one-time",
    "startup read fast and reliable. This is a pure runtime optimization:",
    "",
    "- The dataset **content is not modified** (a plain recursive copy; Drive stays",
    "  the persistent source, read-only here).",
    "- The **official train/validation split is unchanged** -- the same `train.csv`/",
    "  `valid.csv` and case folders are copied verbatim.",
    "- **Checkpoints/results still go to Google Drive** (Section 3's",
    "  `results/exp02_preprocessing/<arm>/`), never to `/content` -- only the",
    "  read-only dataset is cached locally. `/content` is wiped on a timeout, which",
    "  is fine for a re-copyable dataset but would be fatal for checkpoints.",
    "",
    "Set `USE_LOCAL_DATASET_CACHE = False` to read directly from Drive (the prior",
    "behavior).",
))
cells.append(code(
    "import shutil, subprocess",
    "",
    "USE_LOCAL_DATASET_CACHE = True  # set False to read NIfTI directly from Drive",
    "LOCAL_DATASET_DIR = '/content/local_dataset'",
    "",
    "assert os.path.isdir(DRIVE_DATASET_DIR), f'Dataset not found on Drive at {DRIVE_DATASET_DIR}'",
    "",
    "def _count_case_dirs(root):",
    "    train_dir = os.path.join(root, 'prostate158_train', 'train')",
    "    return len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]) if os.path.isdir(train_dir) else 0",
    "",
    "if USE_LOCAL_DATASET_CACHE:",
    "    drive_n = _count_case_dirs(DRIVE_DATASET_DIR)",
    "    local_n = _count_case_dirs(LOCAL_DATASET_DIR)",
    "    if local_n >= drive_n and drive_n > 0:",
    "        print(f'Local dataset cache already present ({local_n} case dirs) -- skipping copy.')",
    "    else:",
    "        print(f'Copying dataset Drive -> {LOCAL_DATASET_DIR} (one-time; {drive_n} case dirs)...')",
    "        os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)",
    "        # rsync is fast + resumable on Colab; fall back to shutil.copytree.",
    "        rc = subprocess.run(['rsync', '-a', DRIVE_DATASET_DIR.rstrip('/') + '/', LOCAL_DATASET_DIR.rstrip('/') + '/']).returncode",
    "        if rc != 0:",
    "            shutil.copytree(DRIVE_DATASET_DIR, LOCAL_DATASET_DIR, dirs_exist_ok=True)",
    "        print(f'Local cache ready: {_count_case_dirs(LOCAL_DATASET_DIR)} case dirs.')",
    "    DATASET_SRC = LOCAL_DATASET_DIR",
    "else:",
    "    DATASET_SRC = DRIVE_DATASET_DIR",
    "",
    "link_path = os.path.join(REPO_DIR, 'dataset')",
    "if os.path.islink(link_path):",
    "    os.remove(link_path)",
    "elif os.path.exists(link_path):",
    "    raise RuntimeError(f'{link_path} exists and is not a symlink -- refusing to overwrite.')",
    "",
    "os.symlink(DATASET_SRC, link_path)",
    "print('Linked:', link_path, '->', os.readlink(link_path))",
    "",
    "expected_t2 = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train', '020', 't2.nii.gz')",
    "expected_mask = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train', '020', 't2_anatomy_reader1.nii.gz')",
    "print('t2.nii.gz found:               ', os.path.exists(expected_t2))",
    "print('t2_anatomy_reader1.nii.gz found:', os.path.exists(expected_mask))",
))

# ---- 7. Inventory ----
cells.append(md(
    "## 7. Inventory check: how much of the official split is on Drive",
    "",
    "The official split is 119 train / 20 validation cases (`src/splits.py`). This",
    "reports coverage up front instead of failing later with a confusing",
    "`FileNotFoundError`. Experiment 2 trains on the SAME split as Exp01.",
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
    "    print('train.csv/valid.csv not found under the Drive dataset -- cannot check coverage.')",
))

# ---- 8. Verify configs + matrix ----
cells.append(md(
    "## 8. Verify all three Exp02 configs (scientific-control gate)",
    "",
    "`tests/test_exp02_configs.py` asserts that each arm differs from the Exp01",
    "baseline in EXACTLY its intended preprocessing factor and nothing else (model,",
    "optimizer, lr, seed, epochs, batch size, split, target mask, loss, slice",
    "sampling all identical), that no test split is referenced, that seeds equal 42,",
    "that each arm's output dir is isolated under `results/exp02_preprocessing/`, and",
    "that Arm B's global stats are auto-derived from the TRAIN split only (no leakage).",
))
cells.append(code("!python -m pytest tests/test_exp02_configs.py -v"))
cells.append(md(
    "Print the ablation matrix directly from the configs so the single varying factor",
    "per arm is visible at a glance (baseline column included for reference).",
))
cells.append(code(
    "import sys",
    "sys.path.insert(0, REPO_DIR)",
    "from src.utils import load_config",
    "",
    "def _preproc(path):",
    "    p = load_config(path).get('preprocessing', {})",
    "    return {",
    "        'normalization': p.get('normalization'),",
    "        'z_resample': p.get('z_resample'),",
    "        'spatial_mode': p.get('spatial_mode'),",
    "        'target_size': p.get('target_size'),",
    "    }",
    "",
    "rows = {'baseline (Exp01)': _preproc('configs/config_baseline.yaml')}",
    "for key, arm in EXP02_ARMS.items():",
    "    rows[f'Exp02-{key} ({arm[\"name\"]})'] = _preproc(arm['config'])",
    "",
    "matrix = pd.DataFrame(rows).T[['normalization', 'z_resample', 'spatial_mode', 'target_size']]",
    "print(matrix.to_string())",
))

# ---- 9. Geometry round-trip gate ----
cells.append(md(
    "## 9. Round-trip fidelity gate (geometry, no model)",
    "",
    "`scripts/roundtrip_test.py` must report Dice = 1.0 exactly for every class on the",
    "crop/pad suites -- this is the geometry-correctness gate the crop/pad arms",
    "(baseline, A, B) depend on. The `resize` arm (C) is deliberately NOT pixel-exact,",
    "which is why evaluation compares against the untouched on-disk ground truth",
    "(next cell), not a round-tripped mask.",
))
cells.append(code("!python scripts/roundtrip_test.py"))

# ---- 10. Evaluation-fairness guard ----
cells.append(md(
    "## 10. Evaluation-fairness guard (untouched original-space GT)",
    "",
    "`tests/test_original_space_gt.py` proves the primary protocol scores every arm's",
    "prediction against the IDENTICAL, untouched on-disk mask in original voxel space:",
    "crop/pad round-trips bit-for-bit (so Exp01 numbers are unchanged), the `resize`",
    "round-trip is lossy, and the evaluated GT is the untouched mask under BOTH arms.",
    "This prevents the lossy resize arm from being scored optimistically -- essential",
    "for a fair RQ-1 comparison.",
))
cells.append(code("!python -m pytest tests/test_original_space_gt.py -v"))

# ---- 10b. Checkpoint / safepoint inspection (inspection only) ----
cells.append(md(
    "## 10b. Checkpoint / safepoint inspection (inspection only -- does NOT train)",
    "",
    "T4 sessions can time out mid-run, so each arm trains to a Drive-backed",
    "`best_model.pt` and is **resume-safe**: rerunning an arm's training cell",
    "auto-detects its checkpoint and continues from `epoch+1` instead of",
    "restarting at epoch 1. This cell reports, per arm, whether a checkpoint",
    "exists and whether it is resumable -- using the read-only",
    "`src.train.inspect_checkpoint` / `plan_training_run` helpers. It starts no",
    "training and never modifies a checkpoint.",
    "",
    "Checkpoint format (verified): `epoch`, `model_state_dict`,",
    "`optimizer_state_dict`, `best_metric`, `config`. There is **no LR scheduler**",
    "in this experiment (nothing to restore), and RNG state is intentionally not",
    "persisted -- the fixed `seed=42` preserves the experiment design; a resumed",
    "run is not bit-identical to an uninterrupted one but is scientifically",
    "equivalent for this controlled ablation.",
))
cells.append(code(
    "import os",
    "from src.train import inspect_checkpoint, plan_training_run",
    "",
    "print('EXP02 CHECKPOINT STATUS')",
    "print('-' * 62)",
    "for _key, _arm in EXP02_ARMS.items():",
    "    _ckpt = os.path.join(DRIVE_EXP02_DIR, _arm['name'], 'best_model.pt')",
    "    _info = inspect_checkpoint(_ckpt)",
    "    _plan = plan_training_run(_ckpt, num_epochs=100)",
    "    if not _info['exists']:",
    "        print(f'{_key} | NONE  | -            | -                | READY FOR FRESH RUN')",
    "        continue",
    "    _size_mb = (_info['size_bytes'] or 0) / 1e6",
    "    if _plan['action'] == 'corrupt':",
    "        print(f'{_key} | FOUND | UNUSABLE -- {_info[\"error\"]}')",
    "        print(f'    path {_ckpt} ({_size_mb:.1f} MB) -- NOT modified/deleted')",
    "        continue",
    "    _status = {'resume': 'RESUMABLE', 'already_complete': 'ALREADY COMPLETE'}.get(_plan['action'], _plan['action'].upper())",
    "    _best = _info['best_metric']",
    "    _best_str = f'{_best:.6f}' if isinstance(_best, (int, float)) else str(_best)",
    "    print(f'{_key} | FOUND | epoch {_info[\"epoch\"]}/100 | best Dice {_best_str} | {_status}')",
    "    print(f'    path {_ckpt} ({_size_mb:.1f} MB)')",
    "    print(f'    model={_info[\"has_model_state\"]} optim={_info[\"has_optimizer_state\"]} '",
    "          f'scheduler={_info[\"has_scheduler_state\"]} rng={_info[\"has_rng_state\"]}')",
    "print('-' * 62)",
    "print('Inspection only -- no training was started; no checkpoint was modified.')",
))

# ---- 10c. Throughput benchmark (optional, measurement only) ----
cells.append(md(
    "## 10c. Throughput benchmark (optional -- measurement only, no full training)",
    "",
    "Profiles where per-step wall-clock goes on THIS T4, using the real pipeline",
    "(`scripts/benchmark_training.py`): dataset build (one-time), pure data loading,",
    "per-stage train step (data / host->device / forward / backward / optimizer),",
    "validation, checkpoint save, and an extrapolated per-epoch estimate. It also",
    "sweeps `num_workers` and (measurement only) times AMP vs FP32 and",
    "`cudnn.benchmark` on/off so any speed-up is evidence-based.",
    "",
    "It trains nothing to convergence, writes only to a temp dir, and does not touch",
    "any real checkpoint. AMP and `cudnn.benchmark` are reported but NOT enabled --",
    "they change numerics vs the FP32 Exp01 baseline, so switching them on is a",
    "supervisor-level decision, not a silent default.",
))
cells.append(code(
    "!python scripts/benchmark_training.py --config configs/experiments/exp02_a_zresample_on.yaml --train-batches 30 --val-batches 20 --num-workers-sweep 0,2,4 --measure-amp --measure-cudnn-benchmark",
))

# ---- STOP divider ----
cells.append(md(
    "---",
    "# ⛔ TRAINING BELOW -- run only after cells 1-10 are green",
    "",
    "Each arm has its own **training** cell and **validation-evaluation** cell, run",
    "independently. Do NOT \"Run all\" from here: run one arm's training cell, let it",
    "finish (100 epochs), run that arm's evaluation cell, then move to the next arm.",
    "",
    "**If a T4 session times out mid-training, just rerun that same arm's training",
    "cell** -- it auto-detects the Drive checkpoint and resumes from `epoch+1` (it",
    "prints `RESUMING` and the epoch it continues from). Cell 10b shows each arm's",
    "current checkpoint status first.",
    "",
    "Every training cell redirects `experiment.output_dir` to Drive",
    "(`results/exp02_preprocessing/<arm>/`) via a throwaway config written OUTSIDE the",
    "repo (never committed) -- the committed configs are unchanged. Training uses the",
    "arm's config verbatim, so all controlled factors stay fixed; only preprocessing",
    "differs. Checkpoints survive a Colab reset because they land on Drive.",
    "",
    "**The official 19-case held-out TEST set is NOT evaluated here.** Variant",
    "selection uses the validation split only; held-out test is a separate one-time",
    "step performed after a variant is chosen.",
))


def _training_cell(arm_key):
    arm = EXP02_ARMS_PY[arm_key]
    name = arm["name"]
    cfg = arm["config"]
    colab_cfg = f"/content/config_exp02_{arm_key.lower()}_colab.yaml"
    return code(
        "import os, yaml",
        "from src.utils import load_config",
        "from src.train import run_training, plan_training_run",
        "",
        f"ARM = '{arm_key}'  # {name}",
        f"ARM_CONFIG = '{cfg}'",
        f"ARM_OUTPUT_DIR = os.path.join(DRIVE_EXP02_DIR, '{name}')",
        "ARM_CHECKPOINT = os.path.join(ARM_OUTPUT_DIR, 'best_model.pt')",
        f"COLAB_CONFIG_PATH = '{colab_cfg}'  # outside the repo -- never committed",
        "",
        "# Requirement: checkpoints/results MUST live on Google Drive, never on",
        "# ephemeral Colab storage (which is wiped on a timeout/cooldown).",
        "assert ARM_OUTPUT_DIR.startswith('/content/drive/'), f'Output dir must be on Drive: {ARM_OUTPUT_DIR}'",
        "os.makedirs(ARM_OUTPUT_DIR, exist_ok=True)",
        "",
        "# Build the Drive-backed throwaway config (committed configs stay untouched).",
        "config = load_config(ARM_CONFIG)",
        "config['experiment']['output_dir'] = ARM_OUTPUT_DIR  # redirect checkpoints/results to Drive",
        "NUM_EPOCHS = int(config['training']['num_epochs'])  # 100 (unchanged)",
        "with open(COLAB_CONFIG_PATH, 'w') as f:",
        "    yaml.safe_dump(config, f)",
        "",
        "# Decide fresh vs resume vs already-complete vs corrupt. All decision logic",
        "# lives in src/train.py::plan_training_run; the resume mechanics live in",
        "# run_training/_load_resume_checkpoint. This cell only orchestrates.",
        "plan = plan_training_run(ARM_CHECKPOINT, num_epochs=NUM_EPOCHS)",
        "banner = {'resume': 'RESUMING', 'fresh': 'FRESH START'}.get(plan['action'], plan['action'].upper())",
        "print('=' * 62)",
        "print(f'EXP02-{ARM} training  |  {banner}')",
        "print('=' * 62)",
        "print(plan['message'])",
        "print('  output_dir ->', ARM_OUTPUT_DIR)",
        "print('  preprocessing ->', config.get('preprocessing'))",
        "print('-' * 62)",
        "",
        "if plan['action'] == 'fresh':",
        "    summary = run_training(COLAB_CONFIG_PATH)",
        "    print(summary)",
        "elif plan['action'] == 'resume':",
        "    # resume_from restores model+optimizer+best_metric and continues at epoch+1.",
        "    summary = run_training(COLAB_CONFIG_PATH, resume_from=ARM_CHECKPOINT)",
        "    print(summary)",
        "elif plan['action'] == 'already_complete':",
        "    print(f'Nothing to train -- skip to the Exp02-{ARM} validation-evaluation cell.')",
        "else:  # 'corrupt' -- fail clearly; do NOT start fresh, do NOT delete the checkpoint.",
        "    raise RuntimeError(plan['message'])",
    )


def _eval_cell(arm_key):
    arm = EXP02_ARMS_PY[arm_key]
    name = arm["name"]
    cfg = arm["config"]
    return code(
        "import os",
        f"ARM_OUTPUT_DIR = os.path.join(DRIVE_EXP02_DIR, '{name}')".replace("{name}", name),
        "ARM_CHECKPOINT = os.path.join(ARM_OUTPUT_DIR, 'best_model.pt')",
        "assert os.path.exists(ARM_CHECKPOINT), f'Checkpoint not found (train this arm first): {ARM_CHECKPOINT}'",
        "print('Evaluating Exp02 arm on the VALIDATION split (inference only, no training):')",
        "print('  checkpoint:', ARM_CHECKPOINT)",
        "print('  output dir:', ARM_OUTPUT_DIR)",
        "",
        "# run_evaluation reads the preprocessing from the checkpoint's embedded config,",
        "# so evaluation matches exactly how this arm was trained. --split val keeps the",
        "# official 19-case held-out TEST set untouched. {ARM_*} expand to the Python",
        "# variables above (Colab shell-magic variable substitution).",
        f"!python scripts/run_evaluation.py --config {cfg} --checkpoint {{ARM_CHECKPOINT}} --output-dir {{ARM_OUTPUT_DIR}} --split val",
    )


# EXP02_ARMS is only defined in the notebook runtime (cell 3); mirror it here for the builder.
EXP02_ARMS_PY = {
    "A": {"name": "a_zresample_on", "config": "configs/experiments/exp02_a_zresample_on.yaml"},
    "B": {"name": "b_normalization_global", "config": "configs/experiments/exp02_b_normalization_global.yaml"},
    "C": {"name": "c_spatial_resize", "config": "configs/experiments/exp02_c_spatial_resize.yaml"},
}

# ---- 11. Arm A ----
cells.append(md(
    "## 11. Exp02-A -- z-axis resampling ON",
    "",
    "**Only change vs baseline:** `z_resample: true` (target 3.0 mm; image linear,",
    "mask nearest-neighbour). Expected to be a *near no-op* on this ~3.0 mm cohort --",
    "kept deliberately to exercise the ablation axis (see the note in the header).",
    "Everything else is the baseline.",
))
cells.append(_training_cell("A"))
cells.append(md("### 11b. Exp02-A validation evaluation (inference only)"))
cells.append(_eval_cell("A"))

# ---- 12. Arm B ----
cells.append(md(
    "## 12. Exp02-B -- global (train-only) normalization",
    "",
    "**Only change vs baseline:** `normalization: global`. Stats are auto-computed",
    "from the TRAINING split ONLY (`src/train.py::_ensure_global_stats`, leakage-safe)",
    "and embedded in the checkpoint so evaluation reuses the identical train-derived",
    "stats. Validation/test never contribute to the statistics.",
))
cells.append(_training_cell("B"))
cells.append(md("### 12b. Exp02-B validation evaluation (inference only)"))
cells.append(_eval_cell("B"))

# ---- 13. Arm C ----
cells.append(md(
    "## 13. Exp02-C -- spatial resize (256x256)",
    "",
    "**Only change vs baseline:** `spatial_mode: resize`, `target_size: [256, 256]`",
    "(image bilinear, mask nearest-neighbour). This resize is lossy and not",
    "pixel-exact, so predictions are still inverse-transformed to original voxel space",
    "and scored against the UNTOUCHED on-disk ground truth (guarded in cell 10) -- no",
    "optimistic self-comparison. Everything else is the baseline.",
))
cells.append(_training_cell("C"))
cells.append(md("### 13b. Exp02-C validation evaluation (inference only)"))
cells.append(_eval_cell("C"))

# ---- 14. Comparison summary ----
cells.append(md(
    "## 14. Result collection -- validation comparison across arms",
    "",
    "Reads each arm's `eval_val_summary.csv` (written by the evaluation cells) and",
    "builds a per-class Dice comparison table (mean, SD, 95% CI). The Exp01 baseline",
    "validation summary is included for reference when present. All numbers are",
    "volume-level, per-case, original voxel space, VALIDATION split -- **not** the",
    "held-out test set. Use this table to select a preprocessing variant; the",
    "held-out test evaluation of the chosen variant is a separate, later step.",
))
cells.append(code(
    "import os",
    "import pandas as pd",
    "",
    "# (label, results_dir) for every arm we want in the table, baseline first.",
    "sources = [('baseline (Exp01)', os.path.join(DRIVE_RESULTS_DIR, 'exp01_baseline'))]",
    "for key, arm in EXP02_ARMS.items():",
    "    sources.append((f'Exp02-{key} ({arm[\"name\"]})', os.path.join(DRIVE_EXP02_DIR, arm['name'])))",
    "",
    "records = []",
    "for label, results_dir in sources:",
    "    summary_csv = os.path.join(results_dir, 'eval_val_summary.csv')",
    "    if not os.path.exists(summary_csv):",
    "        print(f'(skip) {label}: no eval_val_summary.csv at {summary_csv}')",
    "        continue",
    "    df = pd.read_csv(summary_csv)",
    "    dice = df[df['metric'] == 'dice']",
    "    for _, r in dice.iterrows():",
    "        if int(r['class_id']) == 0:",
    "            continue  # background is not a reported foreground zone",
    "        records.append({",
    "            'arm': label,",
    "            'class_id': int(r['class_id']),",
    "            'class_name': r['class_name'],",
    "            'dice_mean': round(float(r['mean']), 4),",
    "            'dice_sd': round(float(r['std']), 4),",
    "            'ci95_low': round(float(r['ci95_low']), 4),",
    "            'ci95_high': round(float(r['ci95_high']), 4),",
    "            'n_cases': int(r['n_cases']),",
    "        })",
    "",
    "if records:",
    "    table = pd.DataFrame.from_records(records)",
    "    pivot = table.pivot(index='arm', columns='class_id', values='dice_mean')",
    "    print('Validation Dice (mean) by arm and class:')",
    "    print(pivot.to_string())",
    "    print()",
    "    print('Full per-class detail:')",
    "    print(table.to_string(index=False))",
    "else:",
    "    print('No arm results found yet. Train + evaluate at least one arm above first.')",
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"name": "prostate158_experiment_02_colab.ipynb", "provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(os.path.dirname(__file__), "..", "notebooks", "prostate158_experiment_02_colab.ipynb")
out_path = os.path.abspath(out_path)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
    f.write("\n")

print("Wrote", out_path)
