"""Generator for notebooks/prostate158_exp_e1_dice_ce_colab.ipynb.

Run locally to (re)build the Experiment-E1 Colab notebook as valid nbformat4
JSON (so cell content need not be hand-escaped). The notebook is ORCHESTRATION
ONLY -- every substantive step calls existing `src/` modules or `scripts/*.py`
entry points, so the GitHub repo stays the single source of truth.

Unlike the Exp02 notebook, this one IS designed for **Runtime > Run all**:
after mounting Drive the whole experiment runs unattended, and re-running it
after a Colab timeout automatically resumes from `latest_checkpoint.pt` on Drive.

Structure:
   1  GPU / CUDA verification (hard-fails rather than silently using CPU)
   2  Google Drive mount
   3  Repository clone / pull
   4  Dependency installation
   5  Dataset setup (Drive -> local cache -> symlink)
   6  Dataset integrity + label verification
   7  Repository / commit verification
   8  Scientific-control tests (E1 differs from baseline ONLY by the loss)
   9  E1 configuration for Colab (output_dir -> Drive; nothing else changed)
  10  Configuration fingerprint
  11  Pre-flight smoke test (18 checks, no test data touched)
  12  Checkpoint detection + resume plan
  13  Training (fresh or resumed, max 100 epochs)
  14  Validation evaluation (20 official validation cases)
  15  Report + plots + baseline comparison
  16  Final summary

The official 19-case test set is never referenced by this notebook.
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

# ===================================================================== HEADER
cells.append(md(
    "# Experiment E1 -- Dice + Cross Entropy (RQ-1)",
    "",
    "**Thesis:** Segmentasi Zona Anatomi pada Citra MRI Pasien Kanker Prostat",
    "menggunakan 2D U-Net dan Rekonstruksi 3D.",
    "",
    "**Research question (E1):** *Does replacing the baseline Cross Entropy loss with",
    "Dice + Cross Entropy improve validation segmentation performance while keeping all",
    "other major training conditions unchanged?*",
    "",
    "The outcome is **not presumed**. E1 may improve, match, or regress against the",
    "baseline; the measured validation result decides.",
    "",
    "---",
    "",
    "## How to run",
    "",
    "1. `Runtime > Change runtime type > T4 GPU`",
    "2. `Runtime > Run all`",
    "3. Approve the Google Drive mount prompt when it appears.",
    "",
    "That is the only manual interaction required. If Colab disconnects mid-training,",
    "just open the notebook and **Run all** again -- training resumes from",
    "`latest_checkpoint.pt` on Drive at the next epoch, never from epoch 1.",
    "",
    "## Controlled comparison",
    "",
    "| | Baseline | E1 |",
    "|---|---|---|",
    "| Loss | Cross Entropy | **1.0 x CE + 1.0 x Soft Dice (CG, PZ)** |",
    "| Architecture | 2D U-Net (1 -> 3) | identical |",
    "| Optimizer / LR / WD | AdamW / 1e-3 / 1e-4 | identical |",
    "| Batch size / epochs / seed | 8 / 100 / 42 | identical |",
    "| Precision | FP32 (AMP off) | identical |",
    "| Augmentation / scheduler | none | none |",
    "",
    "**Selection vs comparison** (two different metrics, never conflated):",
    "",
    "* *within-run checkpoint selection* -- the batch-level foreground-macro validation",
    "  Dice proxy in `src/train.py::validate`, **unchanged** from the baseline;",
    "* *across-experiment comparison* -- per-case, volume-level, original-voxel-space",
    "  **macro Dice = (CG + PZ) / 2** over the 20 official validation cases.",
    "",
    "**The official 19-case test set is NOT used anywhere in this notebook.**",
))

# ================================================================ 1. GPU CHECK
cells.append(md("## 1. GPU / CUDA verification"))
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
    "print(f'CUDA available: {CUDA_AVAILABLE}')",
    "print(f'GPU: {GPU_NAME}')",
    "print(f'Device being used: {DEVICE}')",
    "print(f'torch: {torch.__version__}')",
    "",
    "if not CUDA_AVAILABLE:",
    "    raise RuntimeError(",
    "        'No CUDA GPU detected. A full 100-epoch E1 run must NOT fall back to CPU.\\n'",
    "        'Fix: Runtime > Change runtime type > T4 GPU, then Runtime > Run all.'",
    "    )",
    "if GPU_NAME and 'T4' not in GPU_NAME:",
    "    print(f'\\nNOTE: expected a T4; got {GPU_NAME}. Training will still run, but'",
    "          ' epoch timings will not be comparable to the T4 reference.')",
))

# ================================================================ 2. DRIVE
cells.append(md(
    "## 2. Mount Google Drive",
    "",
    "All persistent experiment state (checkpoints, metrics, report) lives on Drive so",
    "nothing is lost when the Colab VM is recycled.",
))
cells.append(code(
    "from google.colab import drive",
    "drive.mount('/content/drive')",
))

# ================================================================ 3. REPO
cells.append(md("## 3. Repository clone / pull"))
cells.append(code(
    "import os, subprocess",
    "REPO_URL = 'https://github.com/amadeussandro/prostate-mri-segmentation-thesis.git'",
    "REPO_DIR = '/content/thesis-project'",
    "REPO_BRANCH = 'main'",
    "if not os.path.isdir(REPO_DIR):",
    "    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)",
    "os.chdir(REPO_DIR)",
    "subprocess.run(['git', 'checkout', REPO_BRANCH], check=True)",
    "subprocess.run(['git', 'pull', 'origin', REPO_BRANCH], check=True)",
    "subprocess.run(['git', 'log', '-1', '--oneline'], check=True)",
))

# ================================================================ 4. DEPS
cells.append(md("## 4. Dependencies"))
cells.append(code(
    "import subprocess, sys",
    "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',",
    "                'nibabel', 'scipy', 'pandas', 'pyyaml', 'matplotlib'], check=True)",
    "import nibabel, scipy, pandas, yaml, matplotlib",
    "print('nibabel', nibabel.__version__, '| scipy', scipy.__version__,",
    "      '| pandas', pandas.__version__, '| matplotlib', matplotlib.__version__)",
))

# ================================================================ 5. DATASET
cells.append(md(
    "## 5. Dataset setup",
    "",
    "The Prostate158 train+validation archive is read from Drive and cached on the local",
    "Colab disk (much faster random access during training), then symlinked into the repo",
    "so the committed relative dataset paths resolve unchanged.",
))
cells.append(code(
    "import os, shutil, subprocess",
    "DRIVE_ROOT = '/content/drive/MyDrive/THESIS_PROSTATE158'",
    "DRIVE_DATASET_DIR = os.path.join(DRIVE_ROOT, 'dataset')",
    "DRIVE_RESULTS_DIR = os.path.join(DRIVE_ROOT, 'results')",
    "LOCAL_DATASET_DIR = '/content/local_dataset'",
    "USE_LOCAL_DATASET_CACHE = True   # set False to read NIfTI directly from Drive",
    "",
    "assert os.path.isdir(DRIVE_DATASET_DIR), f'Dataset not found on Drive: {DRIVE_DATASET_DIR}'",
    "",
    "if USE_LOCAL_DATASET_CACHE:",
    "    os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)",
    "    print('Syncing dataset Drive -> local (safe to re-run)...')",
    "    rc = subprocess.run(['rsync', '-a', DRIVE_DATASET_DIR.rstrip('/') + '/',",
    "                         LOCAL_DATASET_DIR.rstrip('/') + '/']).returncode",
    "    if rc != 0:",
    "        shutil.copytree(DRIVE_DATASET_DIR, LOCAL_DATASET_DIR, dirs_exist_ok=True)",
    "    DATASET_SRC = LOCAL_DATASET_DIR",
    "else:",
    "    DATASET_SRC = DRIVE_DATASET_DIR",
    "",
    "link_path = os.path.join(REPO_DIR, 'dataset')",
    "if os.path.islink(link_path):",
    "    os.remove(link_path)",
    "elif os.path.exists(link_path):",
    "    raise RuntimeError(f'{link_path} exists and is not a symlink -- refusing to overwrite.')",
    "os.symlink(DATASET_SRC, link_path)",
    "print('Linked', link_path, '->', os.readlink(link_path))",
))

cells.append(md("### 5b. Dataset integrity + label verification"))
cells.append(code(
    "import sys",
    "sys.path.insert(0, REPO_DIR)",
    "from src.data_integrity import verify_dataset_cache, format_cache_report",
    "",
    "DS_ROOT   = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train')",
    "TRAIN_CSV = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train.csv')",
    "VALID_CSV = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'valid.csv')",
    "",
    "for p in (DS_ROOT, TRAIN_CSV, VALID_CSV):",
    "    assert os.path.exists(p), f'Required dataset path missing: {p}'",
    "",
    "report = verify_dataset_cache(DS_ROOT, TRAIN_CSV, VALID_CSV)",
    "print(format_cache_report(report))",
    "assert bool(report['ok']), 'Dataset cache INCOMPLETE -- fix before training (see report above).'",
    "print('\\nDataset integrity: OK')",
))
cells.append(code(
    "# Label sanity: the anatomy masks must contain ONLY {0, 1, 2}.",
    "import numpy as np, nibabel as nib",
    "from src.splits import load_official_split",
    "",
    "split = load_official_split(train_csv=TRAIN_CSV, valid_csv=VALID_CSV)",
    "print(f'Official split -> train {len(split.train_ids)} | val {len(split.val_ids)} |',",
    "      f'test {len(split.test_ids) if split.test_ids else 0} (test intentionally absent)')",
    "assert len(split.train_ids) == 119 and len(split.val_ids) == 20",
    "",
    "seen = set()",
    "for pid in (split.train_ids[:3] + split.val_ids[:3]):",
    "    m = nib.load(os.path.join(DS_ROOT, pid, 't2_anatomy_reader1.nii.gz')).get_fdata()",
    "    seen |= set(int(round(v)) for v in np.unique(m))",
    "    assert os.path.exists(os.path.join(DS_ROOT, pid, 't2.nii.gz')), f'missing t2 for {pid}'",
    "print('Observed mask labels (sampled cases):', sorted(seen))",
    "assert seen.issubset({0, 1, 2}), f'Unexpected labels: {seen}'",
    "print('Label check: OK (0=background, 1=CG, 2=PZ)')",
))

# ================================================================ 7. COMMIT
cells.append(md("## 6. Repository / commit verification"))
cells.append(code(
    "import subprocess",
    "print('HEAD commit:')",
    "subprocess.run(['git', '-C', REPO_DIR, 'log', '-1', '--oneline'], check=True)",
    "print('\\nWorking tree (should be clean in a fresh clone):')",
    "subprocess.run(['git', '-C', REPO_DIR, 'status', '--short'], check=True)",
))

# ================================================================ 8. CONTROLS
cells.append(md(
    "## 7. Scientific-control tests",
    "",
    "Mechanically asserts that E1 differs from the baseline **only** by the loss",
    "function, that the baseline CE path is still bit-identical, and that E1 cannot",
    "resume from a foreign checkpoint. If any of this drifts, the run stops here.",
))
cells.append(code(
    "import subprocess, sys",
    "rc = subprocess.run([sys.executable, '-m', 'pytest', '-q',",
    "                     'tests/test_losses.py', 'tests/test_e1_config.py'],",
    "                    cwd=REPO_DIR).returncode",
    "assert rc == 0, 'Scientific-control tests FAILED -- do not train until resolved.'",
    "print('\\nScientific control: VERIFIED')",
))

# ================================================================ 9. CONFIG
cells.append(md(
    "## 8. E1 configuration for Colab",
    "",
    "The committed config `configs/experiments/exp_e1_dice_ce.yaml` is the source of",
    "truth. The only field overridden here is `experiment.output_dir`, pointed at Drive",
    "so checkpoints survive a disconnect. The cell asserts nothing else changed.",
))
cells.append(code(
    "import copy, yaml",
    "from src.utils import load_config",
    "",
    "E1_CONFIG_REPO = os.path.join(REPO_DIR, 'configs', 'experiments', 'exp_e1_dice_ce.yaml')",
    "E1_DRIVE_DIR   = os.path.join(DRIVE_RESULTS_DIR, 'exp_e1_dice_ce')",
    "BASELINE_DRIVE_DIR = os.path.join(DRIVE_RESULTS_DIR, 'exp01_baseline')",
    "os.makedirs(E1_DRIVE_DIR, exist_ok=True)",
    "",
    "committed = load_config(E1_CONFIG_REPO)",
    "cfg = copy.deepcopy(committed)",
    "cfg['experiment']['output_dir'] = E1_DRIVE_DIR",
    "",
    "# Guard: output_dir is the ONLY permitted deviation from the committed config.",
    "changed = [k for k in set(committed) | set(cfg) if committed.get(k) != cfg.get(k)]",
    "assert changed == ['experiment'], f'Unexpected config deviation: {changed}'",
    "for k in set(committed['experiment']) | set(cfg['experiment']):",
    "    if k != 'output_dir':",
    "        assert committed['experiment'].get(k) == cfg['experiment'].get(k), k",
    "assert 'test_csv' not in cfg['data'] and 'test_dir' not in cfg['data'], \\",
    "    'E1 config must not reference the official test split.'",
    "",
    "E1_CONFIG_COLAB = '/content/config_e1_dice_ce_colab.yaml'",
    "with open(E1_CONFIG_COLAB, 'w') as f:",
    "    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)",
    "",
    "NUM_EPOCHS    = cfg['training']['num_epochs']",
    "LATEST_CKPT   = os.path.join(E1_DRIVE_DIR, 'latest_checkpoint.pt')",
    "BEST_CKPT     = os.path.join(E1_DRIVE_DIR, 'best_model.pt')",
    "",
    "print('E1 config (Colab):', E1_CONFIG_COLAB)",
    "print('Drive output dir :', E1_DRIVE_DIR)",
    "print('Max epochs       :', NUM_EPOCHS)",
    "print('AMP              :', cfg['training'].get('amp', False))",
    "print('Loss             :', cfg['loss'])",
))

# ================================================================ 10. FINGERPRINT
cells.append(md(
    "## 9. Configuration fingerprint",
    "",
    "Printed here and persisted to Drive alongside the results by `run_training`.",
))
cells.append(code(
    "from src.utils import build_config_fingerprint, format_config_fingerprint",
    "",
    "fingerprint = build_config_fingerprint(cfg, device=DEVICE)",
    "print(format_config_fingerprint(fingerprint))",
    "",
    "assert fingerprint['amp'] is False, 'AMP must be OFF for E1 (FP32 baseline comparison).'",
    "assert fingerprint['precision'] == 'FP32'",
    "assert fingerprint['seed'] == 42",
    "assert fingerprint['num_epochs'] == 100",
    "assert fingerprint['dice_classes'] == [1, 2]",
    "assert fingerprint['references_test_split'] is False",
    "print('\\nAMP: OFF')",
    "print('Precision: FP32')",
))

# ================================================================ 11. SMOKE
cells.append(md(
    "## 10. Pre-flight smoke test",
    "",
    "18 checks on the real E1 configuration before committing to a 100-epoch run.",
    "Passing this proves the pipeline is *wired* correctly -- it says nothing about",
    "the scientific outcome.",
))
cells.append(code(
    "import tempfile, numpy as np, torch",
    "from src.dataset import ProstateZonal2DDataset",
    "from src.losses import LossConfig, compute_loss_components",
    "from src.model import build_model",
    "from src.train import plan_training_run",
    "from src.transforms import PreprocessingConfig",
    "",
    "checks = []",
    "def check(name, ok, detail=''):",
    "    checks.append((name, bool(ok), detail))",
    "    print(f\"[{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + str(detail)) if detail else ''}\")",
    "",
    "check('1. imports', True)",
    "",
    "smoke_ds = ProstateZonal2DDataset(",
    "    dataset_root=DS_ROOT, patient_ids=[split.train_ids[0]],",
    "    image_name='t2.nii.gz', mask_name=cfg['data']['target_mask'],",
    "    slice_sampling='all',",
    "    preprocessing_config=PreprocessingConfig.from_dict(cfg['preprocessing']),",
    "    cache_data=True)",
    "check('2. dataset loading', len(smoke_ds) > 0, f'{len(smoke_ds)} slices')",
    "",
    "sample = smoke_ds[len(smoke_ds) // 2]",
    "check('3. one training sample', isinstance(sample, dict))",
    "check('4. image shape', sample['image'].shape == (1, 442, 442), sample['image'].shape)",
    "check('5. mask shape', sample['mask'].shape == (442, 442), sample['mask'].shape)",
    "labels = set(int(v) for v in np.unique(sample['mask']))",
    "check('6. unique labels subset of {0,1,2}', labels.issubset({0, 1, 2}), sorted(labels))",
    "",
    "model = build_model(cfg).to(DEVICE)",
    "check('7. model creation', sum(p.numel() for p in model.parameters()) > 0,",
    "      f\"{sum(p.numel() for p in model.parameters()):,} params\")",
    "",
    "images = torch.from_numpy(sample['image']).unsqueeze(0).float().to(DEVICE)",
    "targets = torch.from_numpy(sample['mask']).unsqueeze(0).long().to(DEVICE)",
    "model.train()",
    "logits = model(images)",
    "check('8. forward pass', torch.isfinite(logits).all())",
    "check('9. output shape', tuple(logits.shape) == (1, 3, 442, 442), tuple(logits.shape))",
    "check('10. output classes == 3', logits.shape[1] == 3)",
    "",
    "loss_cfg = LossConfig.from_config(cfg)",
    "loss, components = compute_loss_components(logits, targets, loss_cfg)",
    "check('11. Dice + CE loss calculation', loss_cfg.name == 'dice_ce', loss_cfg.describe())",
    "check('12. finite loss', bool(torch.isfinite(loss)),",
    "      f\"total={components['total']:.6f} ce={components['ce']:.6f} dice={components['dice']:.6f}\")",
    "",
    "optimizer = torch.optim.AdamW(model.parameters(),",
    "                              lr=cfg['training']['learning_rate'],",
    "                              weight_decay=cfg['training']['weight_decay'])",
    "before = model.final_conv.weight.detach().clone()",
    "optimizer.zero_grad(set_to_none=True)",
    "loss.backward()",
    "check('13. backward pass', all(torch.isfinite(p.grad).all()",
    "                               for p in model.parameters() if p.grad is not None))",
    "optimizer.step()",
    "check('14. optimizer step', not torch.equal(before, model.final_conv.weight.detach()))",
    "",
    "with tempfile.TemporaryDirectory() as td:",
    "    tmp_ckpt = os.path.join(td, 'smoke.pt')",
    "    torch.save({'epoch': 1, 'model_state_dict': model.state_dict(),",
    "                'optimizer_state_dict': optimizer.state_dict(),",
    "                'best_metric': 0.0, 'config': cfg}, tmp_ckpt)",
    "    check('15. checkpoint write', os.path.isfile(tmp_ckpt))",
    "    reloaded = torch.load(tmp_ckpt, map_location='cpu', weights_only=False)",
    "    check('16. checkpoint reload', reloaded['epoch'] == 1 and 'model_state_dict' in reloaded)",
    "    plan = plan_training_run(tmp_ckpt, NUM_EPOCHS)",
    "    check('17. resume logic', plan['action'] == 'resume' and plan['start_epoch'] == 2,",
    "          f\"action={plan['action']} start_epoch={plan['start_epoch']}\")",
    "",
    "check('18. no test data accessed',",
    "      ('test_csv' not in cfg['data']) and ('test_dir' not in cfg['data'])",
    "      and split.test_ids is None)",
    "",
    "del model, optimizer, logits, loss, images, targets",
    "torch.cuda.empty_cache()",
    "",
    "failed = [n for n, ok, _ in checks if not ok]",
    "print(f'\\nSmoke test: {len(checks) - len(failed)}/{len(checks)} passed')",
    "assert not failed, f'Smoke test FAILED: {failed}'",
    "print('Pipeline wiring verified. NOTE: this validates wiring only, NOT the experiment.')",
))

# ================================================================ 12. PLAN
cells.append(md(
    "## 11. Checkpoint detection / resume plan",
    "",
    "`latest_checkpoint.pt` on Drive is the resume source (it is refreshed after every",
    "epoch). `best_model.pt` is the selection artefact and is never used to resume.",
))
cells.append(code(
    "from src.train import inspect_checkpoint, plan_training_run",
    "",
    "info = inspect_checkpoint(LATEST_CKPT)",
    "print('latest_checkpoint.pt ->', {k: info[k] for k in",
    "      ('exists', 'readable', 'epoch', 'best_metric', 'resumable', 'error')})",
    "",
    "plan = plan_training_run(LATEST_CKPT, NUM_EPOCHS)",
    "print()",
    "print(plan['message'])",
    "",
    "if plan['action'] == 'corrupt':",
    "    raise RuntimeError('Existing E1 checkpoint is unusable -- see message above. '",
    "                       'It has NOT been modified; investigate before re-running.')",
))

# ================================================================ 13. TRAIN
cells.append(md(
    "## 12. Training (max 100 epochs, FP32, auto-resume)",
    "",
    "Writes after **every** epoch, atomically, to Drive: `latest_checkpoint.pt`",
    "(model + optimizer + epoch + best metric + config + RNG/sampler state) and",
    "`metrics.csv`. `best_model.pt` is written whenever the unchanged within-run",
    "selection proxy improves.",
    "",
    "If this cell is interrupted, re-run the notebook: it continues at the next epoch.",
))
cells.append(code(
    "import time",
    "from src.train import run_training",
    "",
    "print('AMP: OFF')",
    "print('Precision: FP32')",
    "print(f'Device: {DEVICE} ({GPU_NAME})')",
    "print()",
    "",
    "if plan['action'] == 'already_complete':",
    "    print(f'E1 already completed {NUM_EPOCHS} epochs -- skipping training.')",
    "    train_summary = None",
    "else:",
    "    t0 = time.time()",
    "    train_summary = run_training(",
    "        E1_CONFIG_COLAB,",
    "        resume_from=plan['resume_from'],",
    "        history_path=os.path.join(E1_DRIVE_DIR, 'training_history.json'),",
    "    )",
    "    print(f'\\nElapsed: {(time.time() - t0) / 3600.0:.2f} h')",
    "    print(train_summary)",
))

# ================================================================ 14. EVAL
cells.append(md(
    "## 13. Validation evaluation (20 official validation cases)",
    "",
    "Per-case, volume-level, original-voxel-space metrics (Dice, IoU, HD95, ASD,",
    "precision, recall) for the selected checkpoint. Inference only, FP32.",
    "",
    "**The official 19-case test set is not touched.**",
))
cells.append(code(
    "from src.evaluate import run_evaluation",
    "",
    "assert os.path.isfile(BEST_CKPT), f'No best_model.pt found at {BEST_CKPT}'",
    "",
    "eval_result = run_evaluation(",
    "    config_path=E1_CONFIG_COLAB,",
    "    checkpoint_path=BEST_CKPT,",
    "    output_dir=E1_DRIVE_DIR,",
    "    split='val',            # NEVER 'test' in E1",
    "    class_ids=(0, 1, 2),",
    ")",
    "",
    "summary = eval_result['summary']",
    "cg = summary[1]['dice']['mean']",
    "pz = summary[2]['dice']['mean']",
    "print()",
    "print(f'E1 validation CG Dice   : {cg:.4f}')",
    "print(f'E1 validation PZ Dice   : {pz:.4f}')",
    "print(f'E1 validation MACRO Dice: {(cg + pz) / 2:.4f}')",
))

# ================================================================ 15. REPORT
cells.append(md(
    "## 14. Report, plots, and baseline comparison",
    "",
    "The baseline's `eval_val_summary.json` is read from Drive (it is a generated",
    "artefact and is intentionally not committed to Git).",
))
cells.append(code(
    "import subprocess, sys",
    "",
    "baseline_summary = os.path.join(BASELINE_DRIVE_DIR, 'eval_val_summary.json')",
    "if not os.path.isfile(baseline_summary):",
    "    raise FileNotFoundError(",
    "        f'Baseline validation summary not found on Drive: {baseline_summary}\\n'",
    "        'It is a generated artefact (gitignored). Re-run the baseline validation '",
    "        'evaluation, or copy the existing file to that path, then re-run this cell.'",
    "    )",
    "",
    "rc = subprocess.run([sys.executable, 'scripts/finalize_e1_report.py',",
    "                     '--e1-dir', E1_DRIVE_DIR,",
    "                     '--baseline-dir', BASELINE_DRIVE_DIR],",
    "                    cwd=REPO_DIR).returncode",
    "assert rc == 0, 'Report generation failed.'",
))
cells.append(code(
    "from IPython.display import Image, Markdown, display",
    "",
    "for name in ('training_loss.png', 'validation_dice.png', 'validation_metrics.png'):",
    "    p = os.path.join(E1_DRIVE_DIR, 'plots', name)",
    "    if os.path.isfile(p):",
    "        display(Image(filename=p))",
    "",
    "report_path = os.path.join(E1_DRIVE_DIR, 'training_report.md')",
    "if os.path.isfile(report_path):",
    "    display(Markdown(open(report_path).read()))",
))

# ================================================================ 16. SUMMARY
cells.append(md("## 15. Final summary"))
cells.append(code(
    "import json",
    "",
    "with open(os.path.join(E1_DRIVE_DIR, 'summary.json')) as f:",
    "    e1_summary = json.load(f)",
    "macro = e1_summary['macro_dice']",
    "",
    "print('=' * 78)",
    "print('EXPERIMENT E1 -- FINAL SUMMARY')",
    "print('=' * 78)",
    "print(f\"  Experiment              : E1_DiceCE ({e1_summary['experiment']})\")",
    "print( '  Architecture            : 2D U-Net (unchanged from baseline)')",
    "print( '  Loss                    : 1.0 CE + 1.0 Soft Dice')",
    "print( '  Dice classes            : CG (1) + PZ (2), background excluded')",
    "print( '  Maximum epochs          : 100')",
    "print(f\"  Epochs completed        : {e1_summary.get('epochs_completed')}\")",
    "print(f\"  Best checkpoint epoch   : {e1_summary.get('best_checkpoint_epoch')}\")",
    "print( '  Seed                    : 42')",
    "print( '  AMP                     : OFF')",
    "print( '  Precision               : FP32')",
    "print( '  Validation cases        : 20')",
    "print( '  Official test cases used: 0')",
    "print('-' * 78)",
    "for cid, label in (('1', 'CG'), ('2', 'PZ')):",
    "    d = e1_summary['per_class'][cid]['dice']",
    "    print(f\"  {label} Dice  baseline {d['baseline']:.4f} -> E1 {d['e1']:.4f}  ({d['delta']:+.4f})\")",
    "print(f\"  MACRO Dice baseline {macro['baseline']:.4f} -> E1 {macro['e1']:.4f}  ({macro['delta']:+.4f})\")",
    "print('-' * 78)",
    "print(f\"  VERDICT: E1 {e1_summary['verdict'].upper()} vs baseline (validation macro Dice)\")",
    "print('=' * 78)",
    "print()",
    "print('Artifacts on Drive:', E1_DRIVE_DIR)",
    "for f in sorted(os.listdir(E1_DRIVE_DIR)):",
    "    print('   ', f)",
    "print()",
    "print('The official 19-case test set was NOT used during E1 training,')",
    "print('model selection, or experiment comparison.')",
    "print()",
    "print('NEXT: report this result to the supervisor. Do NOT run the official test')",
    "print('evaluation until the final RQ-1 configuration has been selected.')",
))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        "colab": {"name": "prostate158_exp_e1_dice_ce_colab.ipynb", "provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = os.path.join(
    os.path.dirname(__file__), "..", "notebooks", "prostate158_exp_e1_dice_ce_colab.ipynb"
)
out_path = os.path.abspath(out_path)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
    f.write("\n")
print("Wrote", out_path, "with", len(cells), "cells")
