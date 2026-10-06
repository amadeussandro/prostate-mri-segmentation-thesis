"""Generator for notebooks/prostate158_e1_final_test_colab.ipynb.

Builds the ONE-TIME official held-out test evaluation of the frozen RQ1
candidate (E1, epoch 88) as valid nbformat4 JSON. Orchestration only -- every
substantive step calls an existing `scripts/*.py` entry point or `src/` module.

THIS NOTEBOOK IS DIFFERENT FROM THE TRAINING NOTEBOOKS. It trains nothing,
selects nothing, and tunes nothing. It runs inference once on the official
19-case test archive and records the result. The model was already chosen on
validation data; these numbers are reported, never used to revisit that choice.

Safety interlocks, in execution order:
   1  SHA-256 of the frozen E1 checkpoint must match the recorded value
   2  the checkpoint must say epoch 88 and carry the E1 loss spec
   3  the test archive is audited read-only (19 cases, GT present, no leakage)
   4  an existing eval_test_* for E1 HARD-STOPS the run (no silent re-run)
   5  ONE_TIME_RUN_CONFIRMED must be set to True by hand
   6  the evaluation itself is a single run_evaluation(split="test") call

Sections:
   1  What this notebook is / is not
   2  One-time run confirmation  (the only switch)
   3  Colab environment / GPU + dependencies
   4  Google Drive mount
   5  Repository setup
   6  Dataset + test archive paths
   7  Test archive audit (READ-ONLY, no inference)
   8  Freeze verification: E1 checkpoint SHA-256 and identity
   9  Re-run guard
  10  Test configuration
  11  THE ONE-TIME TEST EVALUATION
  12  Per-case test metrics
  13  Aggregate test metrics
  14  Validation vs test (generalization gap)
  15  Literature comparison
  16  Artifact verification
  17  Reproducibility record
  18  Final RQ1 summary
"""
import json
import os

_cell_counter = 0


def _next_id(kind):
    global _cell_counter
    _cell_counter += 1
    return f"e1test-{kind}-{_cell_counter:03d}"


def md(*lines):
    return {
        "cell_type": "markdown",
        "id": _next_id("md"),
        "metadata": {},
        "source": [l + "\n" for l in lines],
    }


def code(*lines):
    return {
        "cell_type": "code",
        "id": _next_id("code"),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines],
    }


cells = []

# ===================================================================== HEADER
cells.append(md(
    "# RQ1 FINAL — One-Time Official Test Evaluation (E1, epoch 88)",
    "",
    "**Thesis:** Segmentasi Zona Anatomi pada Citra MRI Pasien Kanker Prostat",
    "menggunakan 2D U-Net dan Rekonstruksi 3D.",
    "",
    "**RQ1:** *Berapa Hasil Evaluasi dari Segmentasi Citra MRI pada Pasien Kanker",
    "Prostat menggunakan 2D U-Net?*",
    "",
    "---",
    "",
    "## 1. What this notebook is — and is not",
    "",
    "| | |",
    "|---|---|",
    "| **Is** | a single inference pass of the **already-frozen** E1 checkpoint over the official 19-case held-out test set |",
    "| **Is not** | training, fine-tuning, model selection, threshold tuning, post-processing, or test-time augmentation |",
    "",
    "### The model was already chosen. This notebook only measures it.",
    "",
    "E1 was selected on **validation data only** (20 cases), by the pre-committed rule",
    "*highest validation macro Dice*, before the test set was ever considered for it:",
    "",
    "| Arm | Intervention | Best epoch | CG Dice | PZ Dice | Macro Dice |",
    "|---|---|---|---|---|---|",
    "| Baseline | CE | 77 | 0.8642 | 0.7375 | 0.8008 |",
    "| **E1 ← selected** | **+ SoftDice(CG,PZ)** | **88** | **0.8662** | **0.7527** | **0.8094** |",
    "| E2 | + augmentation | 70 | 0.8636 | 0.7327 | 0.7982 |",
    "| E3 | + CE class weights | 99 | 0.8652 | 0.7510 | 0.8081 |",
    "",
    "> All four configurations performed within a relatively narrow validation range",
    "> (macro Dice spread 0.0112, smaller than any single arm's per-case SD of ≈0.05–0.07).",
    "> E1 was selected because it achieved the highest validation macro Dice, CG Dice and",
    "> PZ Dice, while also showing the strongest PZ ASD and competitive PZ HD95.",
    "> **E1 is not claimed to be statistically superior** — the one available paired test",
    "> (E3 vs E1, n=20) gives p = 0.330 with a bootstrap CI of [−0.0091, +0.0073].",
    "",
    "### The one rule that must not be broken",
    "",
    "**Whatever number comes out of section 11, it does not change the selected model.**",
    "If E1 scores lower on test than on validation, that is the finding and it is",
    "reported as such. Re-running this notebook with a different arm after seeing these",
    "numbers would turn the held-out set into a second validation set and invalidate it.",
    "",
    "### Test-set usage history (disclose in the thesis)",
    "",
    "The official test set is **not** being touched for the first time. Prior uses, both",
    "with the **Baseline epoch-77** checkpoint and both **before** the E1/E2/E3 tuning",
    "arms were designed:",
    "",
    "| Date | What | Checkpoint |",
    "|---|---|---|",
    "| 2026-09-13 | baseline test evaluation (`results/exp01_baseline/eval_test_*`) | Baseline ep. 77 |",
    "| 2026-09-19 | RQ2 reconstruction geometry study (`results/exp04_reconstruction/`) | Baseline ep. 77 |",
    "| **today** | **this run — first and only test evaluation of E1** | **E1 ep. 88** |",
    "",
    "RQ1 model selection was made on validation only: every arm's selection artifact",
    "records `split: val`, `n_cases: 20`, `is_official_held_out_test: false`, and no arm's",
    "config declares `test_csv` / `test_dir`.",
    "",
    "### Label mapping",
    "",
    "`0 = background, 1 = CG (CZ + TZ), 2 = PZ` — **verified 2026-10-05** by multi-planar",
    "inspection of cases 020/059/099/139; see `results/label_mapping/label_mapping_verified.md`.",
    "Per-class results below are reported under those anatomical names.",
    "",
    "---",
    "",
    "## How to run",
    "",
    "1. `Runtime > Change runtime type > T4 GPU`",
    "2. Set `ONE_TIME_RUN_CONFIRMED = True` in section 2",
    "3. `Runtime > Run all`, approve the Drive mount",
    "",
    "Inference over 19 cases takes a few minutes. There is no resume logic because there",
    "is nothing long-running to resume — if the session drops, simply re-run.",
    "**Re-running after a completed evaluation is blocked by section 9 on purpose.**",
))

# ========================================== 2. ONE-TIME RUN CONFIRMATION
cells.append(md(
    "## 2. One-time run confirmation",
    "",
    "This is the only switch. It is `False` by default so that an accidental **Run all**",
    "cannot spend the held-out test set.",
    "",
    "Set it to `True` only when all of the following are true:",
    "",
    "- [ ] the label mapping is verified (`results/label_mapping/label_mapping_verified.md`)",
    "- [ ] the official 19-case archive is on Drive",
    "- [ ] E1 is the agreed final RQ1 candidate and will not be revisited",
    "- [ ] you accept that the resulting numbers are final, whatever they are",
))

cells.append(code(
    "# ====================== THE ONLY SWITCH — READ SECTION 2 ======================",
    "ONE_TIME_RUN_CONFIRMED = False   # set to True to permit the test evaluation",
    "# =============================================================================",
    "",
    "# ---- Locations (same convention as the Baseline / E1 / E2 / E3 notebooks) ----",
    "DRIVE_ROOT      = '/content/drive/MyDrive/THESIS_PROSTATE158'",
    "REPO_URL        = 'https://github.com/amadeussandro/prostate-mri-segmentation-thesis.git'",
    "REPO_DIR        = '/content/thesis-project'",
    "REPO_BRANCH     = 'main'",
    "EXPERIMENT_NAME = 'exp_e1_dice_ce'",
    "E1_CONFIG_RELPATH = 'configs/experiments/exp_e1_dice_ce.yaml'",
    "",
    "# Where the official 19-case archive lives on Drive. Adjust ONLY these two if your",
    "# layout differs; everything else is derived.",
    "TEST_ZIP         = f'{DRIVE_ROOT}/dataset/test/prostate158_test.zip'   # optional if already extracted",
    "TEST_EXTRACT_DIR = f'{DRIVE_ROOT}/dataset/test/extracted'",
    "",
    "# ---- The frozen candidate. These are assertions, not settings. ----",
    "FROZEN_SHA256 = '2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2'",
    "FROZEN_EPOCH  = 88",
    "FROZEN_SIZE_BYTES = 93_272_507",
    "",
    "# ---- Validation anchors, for the generalization-gap table in section 14 ----",
    "E1_VALIDATION = {",
    "    'cg_dice': 0.8662, 'pz_dice': 0.7527, 'macro_dice': 0.8094,",
    "    'cg_precision': 0.8823, 'cg_recall': 0.8604,",
    "    'pz_precision': 0.8073, 'pz_recall': 0.7161,",
    "    'cg_hd95': 5.2467, 'pz_hd95': 5.4061,",
    "    'cg_asd': 1.1508, 'pz_asd': 0.9453,",
    "}",
    "",
    "# ---- Published Prostate158 reference (Adams et al. 2022 abstract) ----",
    "ADAMS_REFERENCE = {'cg_dice': 0.877, 'pz_dice': 0.754}",
    "",
    "print(f'ONE_TIME_RUN_CONFIRMED : {ONE_TIME_RUN_CONFIRMED}')",
    "print(f'Frozen candidate       : {EXPERIMENT_NAME} epoch {FROZEN_EPOCH}')",
    "print(f'Expected SHA-256       : {FROZEN_SHA256}')",
    "print(f'Test archive (Drive)   : {TEST_EXTRACT_DIR}')",
    "if not ONE_TIME_RUN_CONFIRMED:",
    "    print()",
    "    print('NOTE: the evaluation in section 11 will be SKIPPED until you set')",
    "    print('      ONE_TIME_RUN_CONFIRMED = True. Everything before it still runs,')",
    "    print('      so you can complete the audit and the freeze check first.')",
))

# ============================================ 3. ENVIRONMENT / DEPENDENCIES
cells.append(md(
    "## 3. Colab environment / GPU verification and dependencies",
    "",
    "A GPU is not strictly required for 19 inference passes, but it keeps the run to",
    "minutes rather than tens of minutes. Evaluation is FP32 regardless.",
))

cells.append(code(
    "import subprocess",
    "subprocess.run(['nvidia-smi'])",
))

cells.append(code(
    "import torch",
    "",
    "CUDA_AVAILABLE = torch.cuda.is_available()",
    "GPU_NAME = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else None",
    "DEVICE = 'cuda' if CUDA_AVAILABLE else 'cpu'",
    "",
    "print(f'CUDA available : {CUDA_AVAILABLE}')",
    "print(f'GPU            : {GPU_NAME}')",
    "print(f'Device         : {DEVICE}')",
    "print(f'torch          : {torch.__version__}')",
    "if not CUDA_AVAILABLE:",
    "    print()",
    "    print('NOTE: running on CPU. Inference over 19 volumes will be slow but the')",
    "    print('      numbers are identical -- evaluation is FP32 on every device.')",
))

cells.append(code(
    "import subprocess, sys",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',",
    "                'nibabel', 'scipy', 'pandas', 'pyyaml', 'matplotlib'], check=True)",
    "import nibabel, scipy, pandas, yaml, matplotlib",
    "print('nibabel', nibabel.__version__, '| scipy', scipy.__version__,",
    "      '| pandas', pandas.__version__, '| matplotlib', matplotlib.__version__)",
))

# =================================================== 4. GOOGLE DRIVE MOUNT
cells.append(md(
    "## 4. Mount Google Drive",
    "",
    "Both the frozen checkpoint and the test archive live on Drive. Results are written",
    "back next to the E1 artifacts.",
))

cells.append(code(
    "from google.colab import drive",
    "drive.mount('/content/drive')",
))

# ===================================================== 5. REPOSITORY SETUP
cells.append(md(
    "## 5. Repository setup",
))

cells.append(code(
    "import os, subprocess, sys",
    "",
    "if not os.path.isdir(REPO_DIR):",
    "    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)",
    "os.chdir(REPO_DIR)",
    "subprocess.run(['git', 'checkout', REPO_BRANCH], check=True)",
    "subprocess.run(['git', 'pull', 'origin', REPO_BRANCH], check=True)",
    "",
    "GIT_COMMIT = subprocess.run(['git', '-C', REPO_DIR, 'rev-parse', 'HEAD'],",
    "                            capture_output=True, text=True).stdout.strip()",
    "print('Repo   :', REPO_DIR)",
    "print('Branch :', REPO_BRANCH)",
    "print('HEAD   :', GIT_COMMIT)",
    "subprocess.run(['git', '-C', REPO_DIR, 'log', '-1', '--oneline'], check=True)",
    "",
    "if REPO_DIR not in sys.path:",
    "    sys.path.insert(0, REPO_DIR)",
    "",
    "E1_CONFIG_REPO = os.path.join(REPO_DIR, E1_CONFIG_RELPATH)",
    "assert os.path.isfile(E1_CONFIG_REPO), f'Committed E1 config not found: {E1_CONFIG_REPO}'",
    "print('\\nE1 config:', E1_CONFIG_REPO)",
))

# ========================================== 6. DATASET + TEST ARCHIVE PATHS
cells.append(md(
    "## 6. Dataset and test-archive paths",
    "",
    "The train/validation archive is needed for the leakage check in section 7 (the 19",
    "test cases must not appear among the 119 train or 20 validation cases). It is **not**",
    "evaluated here.",
))

cells.append(code(
    "import os, shutil, subprocess",
    "",
    "DRIVE_DATASET_DIR = os.path.join(DRIVE_ROOT, 'dataset')",
    "DRIVE_RESULTS_DIR = os.path.join(DRIVE_ROOT, 'results')",
    "assert os.path.isdir(DRIVE_DATASET_DIR), f'Dataset not found on Drive: {DRIVE_DATASET_DIR}'",
    "",
    "# Link the Drive dataset into the repo so the committed relative paths resolve.",
    "link_path = os.path.join(REPO_DIR, 'dataset')",
    "if os.path.islink(link_path):",
    "    os.remove(link_path)",
    "elif os.path.exists(link_path):",
    "    raise RuntimeError(f'{link_path} exists and is not a symlink -- refusing to overwrite.')",
    "os.symlink(DRIVE_DATASET_DIR, link_path)",
    "print('Linked', link_path, '->', os.readlink(link_path))",
    "",
    "TRAIN_CSV = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train.csv')",
    "VALID_CSV = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'valid.csv')",
    "DS_ROOT   = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train')",
    "for p in (DS_ROOT, TRAIN_CSV, VALID_CSV):",
    "    assert os.path.exists(p), f'Required path missing: {p}'",
    "",
    "print('\\nTrain/validation archive:')",
    "print('  root  :', DS_ROOT)",
    "print('  train :', TRAIN_CSV)",
    "print('  valid :', VALID_CSV)",
    "",
    "print('\\nTest archive:')",
    "print('  zip        :', TEST_ZIP, '->', 'FOUND' if os.path.isfile(TEST_ZIP) else 'not present')",
    "print('  extract-dir:', TEST_EXTRACT_DIR, '->',",
    "      'FOUND' if os.path.isdir(TEST_EXTRACT_DIR) else 'not present')",
    "if not os.path.isfile(TEST_ZIP) and not os.path.isdir(TEST_EXTRACT_DIR):",
    "    raise FileNotFoundError(",
    "        'Neither the test ZIP nor an extracted test directory was found.\\n'",
    "        f'  expected ZIP     : {TEST_ZIP}\\n'",
    "        f'  expected extract : {TEST_EXTRACT_DIR}\\n'",
    "        'Obtain the official Prostate158 test archive (DOI 10.5281/zenodo.6592345), '",
    "        'place it under <DRIVE_ROOT>/dataset/test/, then re-run. Adjust TEST_ZIP / '",
    "        'TEST_EXTRACT_DIR in section 2 if your layout differs.')",
))

# ================================================== 7. TEST ARCHIVE AUDIT
cells.append(md(
    "## 7. Test archive audit — READ-ONLY, no inference",
    "",
    "`scripts/audit_test_archive.py` loads no model, runs no inference and computes no",
    "segmentation metric. It reports case count and IDs, per-case presence of `t2.nii.gz`",
    "and `t2_anatomy_reader1.nii.gz`, NIfTI geometry and label values, overlap against the",
    "119 train + 20 validation cases, and whether `src.splits.load_official_split` resolves",
    "exactly 19 test IDs.",
    "",
    "**Looking at the ground-truth masks' existence and label values is not \"using\" the",
    "test set** — no prediction is made and no metric is computed. This is the standard",
    "data-integrity gate that must precede any held-out evaluation.",
))

cells.append(code(
    "import subprocess, sys, json",
    "",
    "AUDIT_JSON = '/content/test_archive_audit.json'",
    "audit_cmd = [sys.executable, 'scripts/audit_test_archive.py',",
    "             '--extract-dir', TEST_EXTRACT_DIR,",
    "             '--train-csv', TRAIN_CSV,",
    "             '--valid-csv', VALID_CSV,",
    "             '--audit-json', AUDIT_JSON]",
    "if os.path.isfile(TEST_ZIP) and not os.path.isdir(TEST_EXTRACT_DIR):",
    "    audit_cmd += ['--zip', TEST_ZIP]",
    "",
    "rc = subprocess.run(audit_cmd, cwd=REPO_DIR).returncode",
    "assert rc == 0, 'Test archive audit FAILED -- do not evaluate until resolved.'",
))

cells.append(code(
    "# Hard gates on the audit result.",
    "with open(AUDIT_JSON, encoding='utf-8') as f:",
    "    audit = json.load(f)",
    "",
    "TEST_CASE_ROOT = audit.get('case_root') or TEST_EXTRACT_DIR",
    "test_case_ids = audit.get('case_ids') or []",
    "",
    "gates = []",
    "def gate(label, ok, detail=''):",
    "    gates.append((label, bool(ok), detail))",
    "    print(f\"[{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + str(detail)) if detail else ''}\")",
    "",
    "# Key names below MUST match the payload written by scripts/audit_test_archive.py.",
    "# A typo here would make a gate pass on a missing key, so each is read explicitly",
    "# and a missing key is treated as a FAILURE, never as 'nothing to report'.",
    "REQUIRED_AUDIT_KEYS = ('case_ids', 'num_with_t2', 'num_with_gt', 'overlap_with_train',",
    "                       'overlap_with_val', 'loader_resolves_19_test_ids',",
    "                       'unique_gt_label_sets')",
    "absent = [k for k in REQUIRED_AUDIT_KEYS if k not in audit]",
    "assert not absent, (f'Audit JSON is missing {absent}. The audit script schema changed; '",
    "                    'fix the gates below before trusting them.')",
    "",
    "gate('T1. exactly 19 test cases', len(test_case_ids) == 19, len(test_case_ids))",
    "gate('T2. no overlap with the 119 training cases',",
    "     audit['overlap_with_train'] == [], audit['overlap_with_train'] or 'NONE')",
    "gate('T3. no overlap with the 20 validation cases',",
    "     audit['overlap_with_val'] == [], audit['overlap_with_val'] or 'NONE')",
    "gate('T4. ground truth present for every case',",
    "     audit['num_with_gt'] == len(test_case_ids),",
    "     f\"{audit['num_with_gt']}/{len(test_case_ids)}\")",
    "gate('T5. T2 image present for every case',",
    "     audit['num_with_t2'] == len(test_case_ids),",
    "     f\"{audit['num_with_t2']}/{len(test_case_ids)}\")",
    "gate('T6. loader resolves exactly 19 test IDs',",
    "     bool(audit['loader_resolves_19_test_ids']), audit.get('loader_message'))",
    "label_sets = audit['unique_gt_label_sets']",
    "observed_labels = sorted({int(v) for group in label_sets for v in group})",
    "gate('T7. GT labels are a subset of {0,1,2}',",
    "     bool(observed_labels) and set(observed_labels).issubset({0, 1, 2}), observed_labels)",
    "",
    "failed = [l for l, ok, _ in gates if not ok]",
    "assert not failed, f'TEST ARCHIVE GATES FAILED: {failed}'",
    "print(f'\\nTest archive: {len(gates)}/{len(gates)} gates passed.')",
    "print('Case root  :', TEST_CASE_ROOT)",
    "print('Case IDs   :', ', '.join(test_case_ids))",
))

# ============================================== 8. FREEZE VERIFICATION
cells.append(md(
    "## 8. Freeze verification — the E1 checkpoint",
    "",
    "The checkpoint evaluated here must be bit-identical to the one audited on",
    "2026-10-05. A SHA-256 mismatch means a different file, and the run stops.",
))

cells.append(code(
    "import hashlib",
    "",
    "E1_DRIVE_DIR = os.path.join(DRIVE_RESULTS_DIR, EXPERIMENT_NAME)",
    "BEST_CKPT = os.path.join(E1_DRIVE_DIR, 'best_model.pt')",
    "assert os.path.isfile(BEST_CKPT), f'Frozen E1 checkpoint not found: {BEST_CKPT}'",
    "",
    "h = hashlib.sha256()",
    "with open(BEST_CKPT, 'rb') as f:",
    "    for blk in iter(lambda: f.read(1 << 20), b''):",
    "        h.update(blk)",
    "actual_sha = h.hexdigest()",
    "actual_size = os.path.getsize(BEST_CKPT)",
    "",
    "print('Checkpoint :', BEST_CKPT)",
    "print('size       :', f'{actual_size:,} bytes  (expected {FROZEN_SIZE_BYTES:,})')",
    "print('SHA-256    :', actual_sha)",
    "print('expected   :', FROZEN_SHA256)",
    "assert actual_sha == FROZEN_SHA256, (",
    "    'SHA-256 MISMATCH -- this is not the audited frozen E1 checkpoint.\\n'",
    "    f'  expected {FROZEN_SHA256}\\n  got      {actual_sha}\\n'",
    "    'Do NOT proceed. Restore the correct file, or re-run the pre-test audit if the '",
    "    'checkpoint was legitimately replaced.')",
    "print('\\nSHA-256 VERIFIED -- bit-identical to the audited frozen candidate.')",
))

cells.append(code(
    "# Identity and loadability of the frozen checkpoint under the current code.",
    "from src.losses import LossConfig",
    "from src.model import build_model",
    "from src.train import verify_checkpoint_compatibility",
    "from src.utils import load_config",
    "",
    "ckpt = torch.load(BEST_CKPT, map_location='cpu', weights_only=False)",
    "ckpt_cfg = ckpt.get('config') or {}",
    "committed = load_config(E1_CONFIG_REPO)",
    "",
    "checks = []",
    "def check(label, ok, detail=''):",
    "    checks.append((label, bool(ok), detail))",
    "    print(f\"[{'PASS' if ok else 'FAIL'}] {label}{(' -- ' + str(detail)) if detail else ''}\")",
    "",
    "check('F1. checkpoint epoch is 88', ckpt.get('epoch') == FROZEN_EPOCH, ckpt.get('epoch'))",
    "check('F2. experiment name is exp_e1_dice_ce',",
    "      ckpt_cfg.get('experiment', {}).get('name') == EXPERIMENT_NAME,",
    "      ckpt_cfg.get('experiment', {}).get('name'))",
    "loss_spec = LossConfig.from_config(ckpt_cfg)",
    "check('F3. loss is Dice+CE with UNWEIGHTED CE (the E1 arm)',",
    "      loss_spec.name == 'dice_ce' and loss_spec.ce_class_weights is None,",
    "      loss_spec.to_dict())",
    "check('F4. dice classes are (1, 2)', loss_spec.dice_classes == (1, 2))",
    "check('F5. checkpoint carries no test-split reference',",
    "      'test_csv' not in ckpt_cfg.get('data', {}) and 'test_dir' not in ckpt_cfg.get('data', {}))",
    "check('F6. model state is finite',",
    "      all(torch.isfinite(v).all() for v in ckpt['model_state_dict'].values()",
    "          if v.is_floating_point()))",
    "",
    "model = build_model(ckpt_cfg)",
    "model.load_state_dict(ckpt['model_state_dict'], strict=True)",
    "model.eval()",
    "with torch.no_grad():",
    "    probe = model(torch.zeros(1, 1, 442, 442))",
    "check('F7. loads strict=True and forwards to (1,3,442,442)',",
    "      tuple(probe.shape) == (1, 3, 442, 442) and bool(torch.isfinite(probe).all()))",
    "# Architecture identity without a magic constant: the model rebuilt from the",
    "# CHECKPOINT's embedded config must have exactly the same parameter count as one",
    "# rebuilt from the COMMITTED config. (Note two different quantities exist here --",
    "# trainable parameters, and the state_dict total which also counts BatchNorm",
    "# buffers; they are reported separately so neither is mistaken for the other.)",
    "n_params = sum(p.numel() for p in model.parameters())",
    "n_state = sum(v.numel() for v in ckpt['model_state_dict'].values())",
    "reference = build_model(committed)",
    "n_reference = sum(p.numel() for p in reference.parameters())",
    "check('F8. architecture identical to the committed E1 config',",
    "      n_params == n_reference, f'{n_params:,} parameters (state_dict total {n_state:,})')",
    "del reference",
    "",
    "compat = verify_checkpoint_compatibility(ckpt_cfg, committed, BEST_CKPT)",
    "check('F9. compatible with the committed E1 config',",
    "      compat['verified'] or compat['skipped_reason'] is not None, compat['mismatches'])",
    "",
    "del model, probe",
    "failed = [l for l, ok, _ in checks if not ok]",
    "assert not failed, f'FREEZE VERIFICATION FAILED: {failed}'",
    "print(f'\\nFreeze verification: {len(checks)}/{len(checks)} checks passed.')",
))

# ===================================================== 9. RE-RUN GUARD
cells.append(md(
    "## 9. Re-run guard",
    "",
    "The held-out set is spent once. If an E1 test evaluation already exists on Drive,",
    "this notebook stops rather than quietly overwriting it — re-running and keeping the",
    "better of two numbers is exactly what a held-out set must prevent.",
    "",
    "If you genuinely need to re-run (e.g. the first attempt crashed mid-write), move the",
    "existing `eval_test_*` files aside **by hand** and record why.",
))

cells.append(code(
    "existing = [f for f in os.listdir(E1_DRIVE_DIR) if f.startswith('eval_test')] \\",
    "           if os.path.isdir(E1_DRIVE_DIR) else []",
    "",
    "print('E1 results dir :', E1_DRIVE_DIR)",
    "print('existing eval_test_* artifacts:', existing or 'NONE (correct for a first run)')",
    "",
    "if existing:",
    "    raise RuntimeError(",
    "        'An E1 test evaluation ALREADY EXISTS:\\n'",
    "        + ''.join(f'  - {f}\\n' for f in existing)",
    "        + 'Refusing to overwrite it. The official test set is evaluated once per '",
    "          'frozen model.\\nIf the previous attempt was incomplete, move those files '",
    "          'aside manually and note the reason in the thesis reproducibility section.')",
    "",
    "ALREADY_EVALUATED = bool(existing)",
    "print('\\nRe-run guard: clear.')",
))

# ================================================ 10. TEST CONFIGURATION
cells.append(md(
    "## 10. Test configuration",
    "",
    "The committed E1 config is used unchanged except for two additions that are",
    "*required* to reach the test split and were deliberately absent during training:",
    "`data.test_dir` and `experiment.output_dir`. Nothing scientific is altered — the",
    "architecture and preprocessing actually used for inference come from the",
    "**checkpoint's own embedded config**, so evaluation always matches how the weights",
    "were trained.",
))

cells.append(code(
    "import copy, yaml",
    "",
    "cfg = copy.deepcopy(committed)",
    "cfg['experiment']['output_dir'] = E1_DRIVE_DIR",
    "cfg['data']['test_dir'] = TEST_CASE_ROOT",
    "",
    "# Guard: only the two expected keys may differ from the committed config.",
    "def flatten(m, prefix=''):",
    "    out = {}",
    "    for k, v in m.items():",
    "        key = f'{prefix}{k}'",
    "        if isinstance(v, dict):",
    "            out.update(flatten(v, key + '.'))",
    "        else:",
    "            out[key] = v",
    "    return out",
    "",
    "ALLOWED_DEVIATIONS = {'experiment.output_dir', 'data.test_dir'}",
    "fa, fb = flatten(committed), flatten(cfg)",
    "diff = {k for k in set(fa) | set(fb) if fa.get(k, '<absent>') != fb.get(k, '<absent>')}",
    "print('Keys differing from the committed E1 config:')",
    "for k in sorted(diff):",
    "    print(f'  {k}: {fa.get(k, \"<absent>\")!r} -> {fb.get(k, \"<absent>\")!r}')",
    "unexpected = diff - ALLOWED_DEVIATIONS",
    "assert not unexpected, f'Unexpected config deviation: {sorted(unexpected)}'",
    "assert diff == ALLOWED_DEVIATIONS, (",
    "    f'Expected exactly {sorted(ALLOWED_DEVIATIONS)} to differ, got {sorted(diff)}')",
    "",
    "TEST_CONFIG_COLAB = '/content/config_e1_final_test_colab.yaml'",
    "with open(TEST_CONFIG_COLAB, 'w') as f:",
    "    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)",
    "print('\\nTest config written:', TEST_CONFIG_COLAB)",
    "",
    "from src.utils import build_config_fingerprint, format_config_fingerprint",
    "fingerprint = build_config_fingerprint(cfg, device=DEVICE)",
    "print()",
    "print(format_config_fingerprint(fingerprint))",
    "assert fingerprint['references_test_split'] is True, \\",
    "    'the test config must reference the test split -- that is its whole purpose'",
))

# =========================================== 11. THE ONE-TIME EVALUATION
cells.append(md(
    "## 11. THE ONE-TIME TEST EVALUATION",
    "",
    "Inference only. `model.eval()` under `torch.no_grad()`, FP32, argmax over 3 classes,",
    "per-case, volume-level, in original voxel space, against the untouched on-disk ground",
    "truth. No augmentation, no post-processing, no test-time augmentation, no",
    "threshold tuning.",
    "",
    "**This cell writes `eval_test_per_case_metrics.csv`, `eval_test_summary.csv` and",
    "`eval_test_summary.json` into the E1 results directory on Drive. Run it once.**",
))

cells.append(code(
    "from src.evaluate import run_evaluation",
    "import time",
    "",
    "if not ONE_TIME_RUN_CONFIRMED:",
    "    raise RuntimeError(",
    "        'ONE_TIME_RUN_CONFIRMED is False -- the test evaluation was NOT run.\\n'",
    "        'Everything before this point (archive audit, freeze verification, re-run '",
    "        'guard) has completed successfully. Review those outputs, then set '",
    "        'ONE_TIME_RUN_CONFIRMED = True in section 2 and Run all again.')",
    "",
    "print('=' * 78)",
    "print('OFFICIAL HELD-OUT TEST EVALUATION -- E1, epoch 88')",
    "print('=' * 78)",
    "print(f'  checkpoint : {BEST_CKPT}')",
    "print(f'  SHA-256    : {FROZEN_SHA256}')",
    "print(f'  test root  : {TEST_CASE_ROOT}')",
    "print(f'  cases      : {len(test_case_ids)}')",
    "print(f'  device     : {DEVICE}')",
    "print('=' * 78)",
    "",
    "t0 = time.time()",
    "test_result = run_evaluation(",
    "    config_path=TEST_CONFIG_COLAB,",
    "    checkpoint_path=BEST_CKPT,",
    "    output_dir=E1_DRIVE_DIR,",
    "    split='test',",
    "    class_ids=(0, 1, 2),",
    ")",
    "print(f'\\nElapsed: {time.time() - t0:.1f} s')",
    "",
    "meta = test_result['metadata']",
    "assert meta['n_cases'] == 19, f\"expected 19 test cases, evaluated {meta['n_cases']}\"",
    "assert meta['is_official_held_out_test'] is True",
    "assert meta['checkpoint_epoch'] == FROZEN_EPOCH",
    "print('\\nTEST EVALUATION COMPLETE. These numbers are final.')",
))

# ============================================== 12. PER-CASE METRICS
cells.append(md(
    "## 12. Per-case test metrics (19 cases)",
))

cells.append(code(
    "import pandas as pd",
    "",
    "per_case = pd.read_csv(os.path.join(E1_DRIVE_DIR, 'eval_test_per_case_metrics.csv'))",
    "print(f'Rows: {len(per_case)}  |  cases: {per_case[\"patient_id\"].nunique()}')",
    "print()",
    "print(per_case.to_string(index=False))",
    "",
    "fg = per_case[per_case['class_id'].isin([1, 2])]",
    "macro_pc = fg.pivot(index='patient_id', columns='class_id', values='dice')",
    "macro_pc['macro_dice'] = macro_pc[[1, 2]].mean(axis=1)",
    "print('\\nPer-case macro Dice:')",
    "print(macro_pc.to_string())",
    "print(f\"\\nmacro Dice: mean {macro_pc['macro_dice'].mean():.4f} \"",
    "      f\"+/- {macro_pc['macro_dice'].std(ddof=1):.4f} SD | \"",
    "      f\"median {macro_pc['macro_dice'].median():.4f} | \"",
    "      f\"min {macro_pc['macro_dice'].min():.4f} | max {macro_pc['macro_dice'].max():.4f}\")",
))

# ============================================== 13. AGGREGATE METRICS
cells.append(md(
    "## 13. Aggregate test metrics — **the final RQ1 result**",
))

cells.append(code(
    "summary = test_result['summary']",
    "CG, PZ = 1, 2",
    "",
    "test_cg_dice = summary[CG]['dice']['mean']",
    "test_pz_dice = summary[PZ]['dice']['mean']",
    "test_macro = (test_cg_dice + test_pz_dice) / 2",
    "",
    "print('=' * 82)",
    "print('RQ1 FINAL RESULT -- official held-out test set, 19 cases')",
    "print('per-case, volume-level, original voxel space')",
    "print('=' * 82)",
    "print(f\"  {'class':<5}{'metric':<12}{'mean':>10}{'SD':>10}{'CI95 low':>11}{'CI95 high':>11}\")",
    "print('-' * 82)",
    "for cid, name in ((0, 'bg'), (CG, 'CG'), (PZ, 'PZ')):",
    "    for metric in ('dice', 'iou', 'hd95_mm', 'asd_mm', 'precision', 'recall'):",
    "        c = summary[cid].get(metric)",
    "        if c is None:",
    "            continue",
    "        print(f\"  {name:<5}{metric:<12}{c['mean']:>10.4f}{c.get('std', float('nan')):>10.4f}\"",
    "              f\"{c.get('ci95_low', float('nan')):>11.4f}{c.get('ci95_high', float('nan')):>11.4f}\")",
    "print('-' * 82)",
    "print(f'  MACRO Dice (CG+PZ)/2 : {test_macro:.4f}')",
    "print('=' * 82)",
))

# ========================================= 14. VALIDATION VS TEST
cells.append(md(
    "## 14. Validation vs test — the generalization gap",
    "",
    "A drop from validation to test is normal and expected: the model was selected on the",
    "validation cases, so validation performance is optimistically biased. The test numbers",
    "are the unbiased estimate and are what the thesis reports as the RQ1 result.",
))

cells.append(code(
    "rows = [",
    "    ('CG Dice',      E1_VALIDATION['cg_dice'],      summary[CG]['dice']['mean'],      True),",
    "    ('PZ Dice',      E1_VALIDATION['pz_dice'],      summary[PZ]['dice']['mean'],      True),",
    "    ('Macro Dice',   E1_VALIDATION['macro_dice'],   test_macro,                       True),",
    "    ('CG precision', E1_VALIDATION['cg_precision'], summary[CG]['precision']['mean'], True),",
    "    ('CG recall',    E1_VALIDATION['cg_recall'],    summary[CG]['recall']['mean'],    True),",
    "    ('PZ precision', E1_VALIDATION['pz_precision'], summary[PZ]['precision']['mean'], True),",
    "    ('PZ recall',    E1_VALIDATION['pz_recall'],    summary[PZ]['recall']['mean'],    True),",
    "    ('CG HD95 (mm)', E1_VALIDATION['cg_hd95'],      summary[CG]['hd95_mm']['mean'],   False),",
    "    ('PZ HD95 (mm)', E1_VALIDATION['pz_hd95'],      summary[PZ]['hd95_mm']['mean'],   False),",
    "    ('CG ASD (mm)',  E1_VALIDATION['cg_asd'],       summary[CG]['asd_mm']['mean'],    False),",
    "    ('PZ ASD (mm)',  E1_VALIDATION['pz_asd'],       summary[PZ]['asd_mm']['mean'],    False),",
    "]",
    "",
    "print('=' * 78)",
    "print('E1 -- VALIDATION (20 cases, selection) vs TEST (19 cases, final)')",
    "print('=' * 78)",
    "print(f\"  {'metric':<15}{'validation':>12}{'test':>12}{'delta':>12}  direction\")",
    "print('-' * 78)",
    "for name, val, tst, higher_better in rows:",
    "    arrow = 'higher is better' if higher_better else 'lower is better'",
    "    print(f'  {name:<15}{val:>12.4f}{tst:>12.4f}{tst - val:>+12.4f}  {arrow}')",
    "print('=' * 78)",
    "print()",
    "print('Reading this honestly: the validation column was used to SELECT this model,')",
    "print('so it is optimistically biased. The test column is the unbiased estimate and')",
    "print('is what the thesis reports. A gap is expected, not a defect.')",
))

# ========================================= 15. LITERATURE COMPARISON
cells.append(md(
    "## 15. Comparison with the published Prostate158 reference",
    "",
    "Adams et al. 2022 (the dataset paper) report CG/TZ 0.877 and PZ 0.754 against Rater 1.",
    "",
    "**Comparability caveats that must accompany any such comparison in the thesis:**",
    "",
    "* Adams used a **3D** U-Net (MONAI, residual units); this work uses a **2D** U-Net —",
    "  that is the thesis's declared scope, not an oversight.",
    "* Confirm the zone definition matches: this work reports **CG = CZ + TZ**. Papers",
    "  reporting **TZ only** are not like-for-like.",
    "* Evaluation protocol (per-case vs per-slice aggregation, evaluation space,",
    "  resampling) differs between papers and is itself a documented source of",
    "  disagreement in this literature.",
    "",
    "A lower number against a 3D reference is an expected and explainable outcome, not a",
    "failure.",
))

cells.append(code(
    "print('=' * 78)",
    "print('E1 (2D U-Net, this work) vs Adams et al. 2022 (3D U-Net, dataset paper)')",
    "print('official 19-case test set')",
    "print('=' * 78)",
    "print(f\"  {'zone':<10}{'this work':>12}{'Adams 2022':>13}{'difference':>13}\")",
    "print('-' * 78)",
    "for name, ours, theirs in (('CG/TZ', test_cg_dice, ADAMS_REFERENCE['cg_dice']),",
    "                           ('PZ',    test_pz_dice, ADAMS_REFERENCE['pz_dice'])):",
    "    print(f'  {name:<10}{ours:>12.4f}{theirs:>13.4f}{ours - theirs:>+13.4f}')",
    "print('=' * 78)",
    "print('Dimensionality differs (2D vs 3D) -- this is a scope difference, not a defect.')",
))

# ========================================== 16. ARTIFACT VERIFICATION
cells.append(md(
    "## 16. Artifact verification",
))

cells.append(code(
    "REQUIRED = [",
    "    ('per-case test metrics',  os.path.join(E1_DRIVE_DIR, 'eval_test_per_case_metrics.csv')),",
    "    ('test summary CSV',       os.path.join(E1_DRIVE_DIR, 'eval_test_summary.csv')),",
    "    ('test summary JSON',      os.path.join(E1_DRIVE_DIR, 'eval_test_summary.json')),",
    "    ('frozen checkpoint',      BEST_CKPT),",
    "    ('validation per-case',    os.path.join(E1_DRIVE_DIR, 'eval_val_per_case_metrics.csv')),",
    "    ('validation summary',     os.path.join(E1_DRIVE_DIR, 'eval_val_summary.json')),",
    "]",
    "missing = []",
    "print(f\"{'artifact':<26}{'size':>14}  path\")",
    "print('-' * 104)",
    "for label, path in REQUIRED:",
    "    ok = os.path.isfile(path)",
    "    size = os.path.getsize(path) if ok else 0",
    "    if not ok or size == 0:",
    "        missing.append(label)",
    "    print(f\"{label:<26}{(f'{size:,}' if ok else 'MISSING'):>14}  {path}\")",
    "print('-' * 104)",
    "assert not missing, f'Missing or empty artifacts: {missing}'",
    "",
    "# The validation artifacts must be untouched by this run.",
    "val_meta = json.load(open(os.path.join(E1_DRIVE_DIR, 'eval_val_summary.json'), encoding='utf-8'))['metadata']",
    "assert val_meta['split'] == 'val' and val_meta['n_cases'] == 20, 'validation artifacts were altered'",
    "print('\\nAll artifacts present. Validation artifacts untouched.')",
))

# ========================================= 17. REPRODUCIBILITY RECORD
cells.append(md(
    "## 17. Reproducibility record",
))

cells.append(code(
    "import platform, sys, time",
    "",
    "repro = {",
    "    'record_type': 'RQ1_final_official_test_evaluation',",
    "    'experiment': EXPERIMENT_NAME,",
    "    'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),",
    "    'git_commit': GIT_COMMIT,",
    "    'git_branch': REPO_BRANCH,",
    "    'repo_url': REPO_URL,",
    "    'notebook': 'notebooks/prostate158_e1_final_test_colab.ipynb',",
    "    'config_committed': E1_CONFIG_RELPATH,",
    "    'frozen_checkpoint': {",
    "        'path': BEST_CKPT,",
    "        'sha256': actual_sha,",
    "        'size_bytes': actual_size,",
    "        'epoch': int(ckpt.get('epoch')),",
    "        'selection_rule': 'highest validation macro Dice (pre-committed), validation data only',",
    "    },",
    "    'selection_evidence': {",
    "        'split_used_for_selection': 'val (20 official validation cases)',",
    "        'validation_macro_dice': E1_VALIDATION['macro_dice'],",
    "        'arms_compared': ['exp01_baseline', 'exp_e1_dice_ce', 'exp_e2_ce_aug',",
    "                          'exp_e3_dice_ce_pzweight'],",
    "        'test_metrics_used_for_selection': False,",
    "    },",
    "    'test_evaluation': {",
    "        'split': 'test',",
    "        'n_cases': int(meta['n_cases']),",
    "        'case_ids': list(meta['case_ids']),",
    "        'is_official_held_out_test': True,",
    "        'evaluation_space': meta.get('evaluation_space'),",
    "        'aggregation': meta.get('aggregation'),",
    "        'ground_truth_source': meta.get('ground_truth_source'),",
    "        'runs_performed': 1,",
    "        'post_processing': 'none',",
    "        'test_time_augmentation': 'none',",
    "    },",
    "    'results': {",
    "        'cg_dice': test_cg_dice, 'pz_dice': test_pz_dice, 'macro_dice': test_macro,",
    "        'cg_precision': summary[CG]['precision']['mean'],",
    "        'cg_recall': summary[CG]['recall']['mean'],",
    "        'pz_precision': summary[PZ]['precision']['mean'],",
    "        'pz_recall': summary[PZ]['recall']['mean'],",
    "        'cg_hd95_mm': summary[CG]['hd95_mm']['mean'],",
    "        'pz_hd95_mm': summary[PZ]['hd95_mm']['mean'],",
    "        'cg_asd_mm': summary[CG]['asd_mm']['mean'],",
    "        'pz_asd_mm': summary[PZ]['asd_mm']['mean'],",
    "    },",
    "    'test_set_usage_history': [",
    "        {'date': '2026-09-13', 'what': 'baseline test evaluation', 'checkpoint': 'exp01_baseline epoch 77'},",
    "        {'date': '2026-09-19', 'what': 'RQ2 reconstruction geometry study', 'checkpoint': 'exp01_baseline epoch 77'},",
    "        {'date': time.strftime('%Y-%m-%d'), 'what': 'E1 final test evaluation (this run)', 'checkpoint': 'exp_e1_dice_ce epoch 88'},",
    "    ],",
    "    'label_mapping': {",
    "        'status': 'verified_multiplanar_2026_10_05',",
    "        'mapping': {'0': 'background', '1': 'CG (CZ+TZ)', '2': 'PZ'},",
    "        'record': 'results/label_mapping/label_mapping_verified.md',",
    "    },",
    "    'environment': {",
    "        'python': sys.version, 'platform': platform.platform(),",
    "        'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': GPU_NAME,",
    "    },",
    "    'test_archive_audit': audit,",
    "}",
    "",
    "repro_path = os.path.join(E1_DRIVE_DIR, 'reproducibility_final_test.json')",
    "with open(repro_path, 'w', encoding='utf-8') as f:",
    "    json.dump(repro, f, indent=2, default=str)",
    "print('Reproducibility record:', repro_path)",
    "print(json.dumps({k: v for k, v in repro.items() if k != 'test_archive_audit'},",
    "                 indent=2, default=str)[:2500])",
))

# ========================================== 18. FINAL RQ1 SUMMARY
cells.append(md(
    "## 18. Final RQ1 summary",
))

cells.append(code(
    "print('=' * 82)",
    "print('RQ1 -- FINAL ANSWER')",
    "print('=' * 82)",
    "print( '  Question  : Berapa hasil evaluasi segmentasi MRI prostat dengan 2D U-Net?')",
    "print()",
    "print(f'  Model     : {EXPERIMENT_NAME} (Dice + Cross-Entropy), epoch {FROZEN_EPOCH}')",
    "print(f'  SHA-256   : {actual_sha}')",
    "print( '  Selected  : on 20 validation cases, by highest validation macro Dice')",
    "print( '  Evaluated : once, on the official 19-case held-out test set')",
    "print('-' * 82)",
    "print( '  OFFICIAL TEST RESULT (per-case, volume-level, original voxel space)')",
    "print(f\"    CG Dice     : {test_cg_dice:.4f} +/- {summary[CG]['dice']['std']:.4f}\")",
    "print(f\"    PZ Dice     : {test_pz_dice:.4f} +/- {summary[PZ]['dice']['std']:.4f}\")",
    "print(f'    MACRO Dice  : {test_macro:.4f}')",
    "print(f\"    CG HD95/ASD : {summary[CG]['hd95_mm']['mean']:.3f} / {summary[CG]['asd_mm']['mean']:.3f} mm\")",
    "print(f\"    PZ HD95/ASD : {summary[PZ]['hd95_mm']['mean']:.3f} / {summary[PZ]['asd_mm']['mean']:.3f} mm\")",
    "print('-' * 82)",
    "print( '  Label mapping: 0=background, 1=CG (CZ+TZ), 2=PZ (verified 2026-10-05)')",
    "print( '  Post-processing: none. Test-time augmentation: none. Runs: 1.')",
    "print('=' * 82)",
    "print()",
    "print('Artifacts on Drive:', E1_DRIVE_DIR)",
    "for entry in sorted(os.listdir(E1_DRIVE_DIR)):",
    "    print(f'    {entry}')",
    "print()",
    "print('NEXT STEPS:')",
    "print('  1. RQ1 is now closed. Do NOT re-run this evaluation or change the model.')",
    "print('  2. Re-point RQ2 at this frozen E1 checkpoint (currently Baseline epoch 77):')",
    "print('       configs/experiments/exp04_rq2_reconstruction.yaml  line 17')",
    "print('       notebooks/prostate158_rq2_reconstruction_colab.ipynb  CHECKPOINT_PATH')",
    "print('  3. Re-run RQ2 reconstruction + geometry validation with E1.')",
    "print('  4. Report RQ2 reconstruction fidelity SEPARATELY from segmentation accuracy:')",
    "print('       round-trip Dice = 1.0 is transform fidelity (a verification), NOT a result.')",
    "print('  5. Update the thesis/journal draft with the numbers above.')",
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"name": "prostate158_e1_final_test_colab.ipynb", "provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "notebooks", "prostate158_e1_final_test_colab.ipynb"))
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
    f.write("\n")
print("Wrote", out_path, "with", len(cells), "cells")
