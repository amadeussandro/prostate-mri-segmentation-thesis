"""Generator for notebooks/prostate158_experiment_02_colab.ipynb.

Run locally to (re)build the FINAL Colab orchestration notebook as valid
nbformat4 JSON (so cell content need not be hand-escaped). The notebook is
ORCHESTRATION ONLY -- every substantive step calls existing src/ modules or
scripts/*.py entry points; the repo on GitHub stays the single source of truth.

Structure (NOT designed for "Run All" -- run cells deliberately, top to bottom;
training cells are independently executable and resume-safe from Drive):
   1  GPU / environment check
   2  Google Drive mount
   3  Repository clone / pull
   4  Dependency installation
   5  Dataset source setup (optional local runtime cache)
   6  Dataset integrity validation (FILE level, not dir count)
   7  Repository / commit verification
   8  Scientific control checks
   9  Geometry / round-trip verification
   10 Checkpoint safety inspection (pilot + official)
   11 FP32 pilot (10 epochs)
   12 AMP pilot (10 epochs)
   13 FP32 vs AMP comparison
   14 Explicit decision point (no auto-decision)
   15 Official Exp01 (locked baseline) -- status only
   16 Exp01 evaluation
   17 Exp02-A training + evaluation
   18 Exp02-B training + evaluation
   19 Exp02-C training + evaluation
   20 Final result collection

Only preprocessing varies across the official Exp02 arms; model, optimizer,
loss, seed, split, target mask, slicing, and evaluation are held exactly as
audited. Precision (FP32/AMP) is a manual decision from the pilot -- never
auto-selected here.
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


# Official Exp02 arms -- REAL committed config names / output dirs (unchanged).
EXP02_ARMS = {
    "A": {"name": "a_zresample_on", "config": "configs/experiments/exp02_a_zresample_on.yaml"},
    "B": {"name": "b_normalization_global", "config": "configs/experiments/exp02_b_normalization_global.yaml"},
    "C": {"name": "c_spatial_resize", "config": "configs/experiments/exp02_c_spatial_resize.yaml"},
}

cells = []

# ===================================================================== HEADER
cells.append(md(
    "# Prostate158 Zonal Segmentation -- Final Colab Orchestration",
    "",
    "**Orchestration only.** Every substantive step calls existing `src/` modules",
    "or `scripts/*.py`; the GitHub repo is the single source of truth. This notebook",
    "takes you through: environment + dataset verification -> a short **FP32-vs-AMP",
    "pilot** -> a precision **decision** you make from the evidence -> the official",
    "**100-epoch** experiments.",
    "",
    "### How to run this notebook",
    "- **Do NOT \"Run All\".** Run cells deliberately, top to bottom.",
    "- Training cells are **independently executable and resume-safe**: each inspects",
    "  its Drive checkpoint first and RESUMES from `epoch+1` -- it never silently",
    "  restarts at epoch 1 or overwrites an existing checkpoint. If Colab times out,",
    "  just rerun the same training cell.",
    "- **Pilot vs official outputs are fully separated.** Pilot -> `results/pilot_amp_v2/`;",
    "  official -> `results/exp01_baseline/` and `results/exp02_preprocessing/<arm>/`.",
    "- **Precision is NOT auto-selected.** The pilot only presents evidence; you choose",
    "  FP32 or AMP for the official experiments (Section 14).",
    "",
    "> Scientific pipeline (unchanged): dataset -> preprocessing -> 3D->2D axial",
    "> slicing -> 2D U-Net -> inference -> 2D->3D reconstruction -> spatial",
    "> verification -> original-space volume evaluation. Spatial verification and the",
    "> round-trip gate (Section 9) must pass before trusting any training result.",
))

# ===================================================================== 1 GPU
cells.append(md("## 1. GPU / environment check"))
cells.append(code("!nvidia-smi"))
cells.append(code(
    "import torch",
    "print('torch:', torch.__version__)",
    "print('CUDA available:', torch.cuda.is_available())",
    "if torch.cuda.is_available():",
    "    print('GPU:', torch.cuda.get_device_name(0))",
))

# ===================================================================== 2 DRIVE
cells.append(md("## 2. Mount Google Drive"))
cells.append(code("from google.colab import drive", "drive.mount('/content/drive')"))

# ===================================================================== 3 CLONE
cells.append(md(
    "## 3. Clone / pull the repository",
    "",
    "The committed repo is authoritative. A fresh clone always has the correct",
    "pipeline code (do NOT rely on any locally-edited copy).",
))
cells.append(code(
    "import os, subprocess",
    "REPO_URL = 'https://github.com/amadeussandro/prostate-mri-segmentation-thesis.git'",
    "REPO_DIR = '/content/thesis-project'",
    "if not os.path.isdir(REPO_DIR):",
    "    subprocess.run(['git', 'clone', REPO_URL, REPO_DIR], check=True)",
    "os.chdir(REPO_DIR)",
    "subprocess.run(['git', 'checkout', 'main'], check=True)",
    "subprocess.run(['git', 'pull', 'origin', 'main'], check=True)",
    "subprocess.run(['git', 'log', '-1', '--oneline'], check=True)",
))

# ===================================================================== 4 DEPS
cells.append(md("## 4. Install dependencies"))
cells.append(code("!pip install -q -r requirements.txt"))

# ===================================================================== 5 DATA SRC
cells.append(md(
    "## 5. Dataset source setup (optional local runtime cache)",
    "",
    "Configs use the repo-relative path `dataset/prostate158_train/train`. A symlink",
    "makes that resolve to the real data. `USE_LOCAL_DATASET_CACHE=True` first copies",
    "the dataset ONCE from Drive to fast local `/content` storage (Drive stays the",
    "read-only source; **checkpoints/results always go to Drive**). Integrity is",
    "validated at FILE level in Section 6 -- not by directory count.",
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
    "if USE_LOCAL_DATASET_CACHE:",
    "    os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)",
    "    print('Syncing dataset Drive -> local (rsync -a; safe to re-run)...')",
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
    "",
    "# Official Exp02 arms (real committed config names / output dirs). Used by later cells.",
    "EXP02_ARMS = {",
    "    'A': {'name': 'a_zresample_on',         'config': 'configs/experiments/exp02_a_zresample_on.yaml'},",
    "    'B': {'name': 'b_normalization_global', 'config': 'configs/experiments/exp02_b_normalization_global.yaml'},",
    "    'C': {'name': 'c_spatial_resize',       'config': 'configs/experiments/exp02_c_spatial_resize.yaml'},",
    "}",
))

# ===================================================================== 6 INTEGRITY
cells.append(md(
    "## 6. Dataset integrity validation (FILE level)",
    "",
    "A cache can have the right NUMBER of case folders yet be missing or have empty",
    "NIfTI files. This verifies every official train+val case actually has non-empty",
    "`t2.nii.gz` and `t2_anatomy_reader1.nii.gz`, plus the split CSVs",
    "(`src.data_integrity.verify_dataset_cache`). If incomplete, it reports exactly",
    "what is missing, rebuilds from the authoritative Drive dataset, and re-verifies.",
    "**Do not run training until this reports OK.**",
))
cells.append(code(
    "import sys, subprocess, shutil",
    "sys.path.insert(0, REPO_DIR)",
    "from src.data_integrity import verify_dataset_cache, format_cache_report",
    "",
    "DS_ROOT = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train')",
    "TRAIN_CSV = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'train.csv')",
    "VALID_CSV = os.path.join(REPO_DIR, 'dataset', 'prostate158_train', 'valid.csv')",
    "",
    "report = verify_dataset_cache(DS_ROOT, TRAIN_CSV, VALID_CSV)",
    "print(format_cache_report(report))",
    "",
    "if not report['ok'] and USE_LOCAL_DATASET_CACHE:",
    "    print('\\nLocal cache INCOMPLETE -> rebuilding from Drive (rsync) and re-verifying...')",
    "    rc = subprocess.run(['rsync', '-a', DRIVE_DATASET_DIR.rstrip('/') + '/',",
    "                         LOCAL_DATASET_DIR.rstrip('/') + '/']).returncode",
    "    if rc != 0:",
    "        shutil.copytree(DRIVE_DATASET_DIR, LOCAL_DATASET_DIR, dirs_exist_ok=True)",
    "    report = verify_dataset_cache(DS_ROOT, TRAIN_CSV, VALID_CSV)",
    "    print(format_cache_report(report))",
    "",
    "DATASET_OK = bool(report['ok'])",
    "assert DATASET_OK, 'Dataset cache is INCOMPLETE -- fix before training (see report above).'",
    "print('\\nDataset integrity: OK -- training cells are allowed.')",
))

# ===================================================================== 7 COMMIT
cells.append(md("## 7. Repository / commit verification"))
cells.append(code(
    "import subprocess",
    "print('HEAD commit:')",
    "subprocess.run(['git', '-C', REPO_DIR, 'log', '-1', '--oneline'], check=True)",
    "print('Status (should be clean in a fresh clone):')",
    "subprocess.run(['git', '-C', REPO_DIR, 'status', '--short'], check=True)",
))

# ===================================================================== 8 CONTROLS
cells.append(md(
    "## 8. Scientific control checks",
    "",
    "Asserts the Exp02 arms differ from the Exp01 baseline in exactly their intended",
    "preprocessing factor (nothing else), that pilot configs differ only by AMP, and",
    "that official experiments are 100 epochs / pilot is 10.",
))
cells.append(code(
    "# -k 'not Integration' keeps this verification cell training-free (it skips the",
    "# 1-epoch run_training integration test in tests/test_amp_pilot.py).",
    "!python -m pytest tests/test_exp02_configs.py tests/test_amp_pilot.py -q -k \"not Integration\"",
))
cells.append(code(
    "from src.utils import load_config",
    "import pandas as pd",
    "def _p(path):",
    "    p = load_config(path).get('preprocessing', {})",
    "    return {'normalization': p.get('normalization'), 'z_resample': p.get('z_resample'),",
    "            'spatial_mode': p.get('spatial_mode'), 'target_size': p.get('target_size')}",
    "rows = {'baseline (Exp01)': _p('configs/config_baseline.yaml')}",
    "for k, a in EXP02_ARMS.items():",
    "    rows[f'Exp02-{k}'] = _p(a['config'])",
    "print(pd.DataFrame(rows).T[['normalization','z_resample','spatial_mode','target_size']].to_string())",
))

# ===================================================================== 9 GEOMETRY
cells.append(md(
    "## 9. Geometry / round-trip verification",
    "",
    "The flagship spatial gate: crop/pad round-trips ground truth to Dice = 1.0",
    "(no model). Also runs the evaluation-fairness guard (predictions scored against",
    "the untouched on-disk GT). **Both must pass before any training result is",
    "trusted.**",
))
cells.append(code("!python scripts/roundtrip_test.py"))
cells.append(code("!python -m pytest tests/test_original_space_gt.py -q"))

# ===================================================================== 10 CKPT INSPECT
cells.append(md(
    "## 10. Checkpoint safety inspection (pilot + official)",
    "",
    "Read-only status for every pilot and official checkpoint on Drive, using",
    "`src.train.inspect_checkpoint` / `plan_training_run`. Starts no training and",
    "modifies nothing. Shows whether each arm would start FRESH, RESUME (and at which",
    "epoch), is ALREADY COMPLETE, or has an UNUSABLE checkpoint.",
))
cells.append(code(
    "from src.train import inspect_checkpoint, plan_training_run",
    "",
    "PILOT = {'PILOT-FP32 (10ep)': ('results/pilot_amp_v2/fp32', 10),",
    "         'PILOT-AMP  (10ep)': ('results/pilot_amp_v2/amp', 10)}",
    "OFFICIAL = {'EXP01 (100ep, LOCKED)': ('results/exp01_baseline', 100),",
    "            'EXP02-A (100ep)': ('results/exp02_preprocessing/a_zresample_on', 100),",
    "            'EXP02-B (100ep)': ('results/exp02_preprocessing/b_normalization_global', 100),",
    "            'EXP02-C (100ep)': ('results/exp02_preprocessing/c_spatial_resize', 100)}",
    "",
    "def _row(label, subdir, num_epochs):",
    "    ckpt = os.path.join(DRIVE_RESULTS_DIR, os.path.relpath(subdir, 'results'), 'best_model.pt')",
    "    info = inspect_checkpoint(ckpt)",
    "    if not info['exists']:",
    "        print(f'{label:26s} | NONE          | READY FOR FRESH RUN'); return",
    "    plan = plan_training_run(ckpt, num_epochs)",
    "    status = {'resume':'RESUMABLE','already_complete':'ALREADY COMPLETE',",
    "              'corrupt':'UNUSABLE','fresh':'FRESH'}.get(plan['action'], plan['action'])",
    "    nxt = f\" -> next epoch {plan['start_epoch']}\" if plan['action']=='resume' else ''",
    "    best = info['best_metric']; best = f'{best:.6f}' if isinstance(best,(int,float)) else best",
    "    print(f'{label:26s} | epoch {info[\"epoch\"]}/{num_epochs} best {best} | {status}{nxt}')",
    "",
    "print('== PILOT =='); [ _row(l, s, n) for l,(s,n) in PILOT.items() ]",
    "print('== OFFICIAL =='); [ _row(l, s, n) for l,(s,n) in OFFICIAL.items() ]",
    "print('\\nInspection only -- nothing was trained or modified.')",
))

# ===================================================================== 11 FP32 PILOT
cells.append(md(
    "## 11. FP32 pilot -- 10 epochs (`results/pilot_amp_v2/fp32`)",
    "",
    "Runs the checkpoint-safe pilot runner. It prints a status block (FRESH / RESUME)",
    "BEFORE training and can never silently overwrite an existing checkpoint. Rerun",
    "this same cell after a timeout to resume. Trains FP32, then runs the FINAL FP32",
    "volume-level validation evaluation (all 20 patients).",
    "",
    "The pilot configs already point `experiment.output_dir` at Drive",
    "(`.../pilot_amp_v2/fp32`), so no redirect is needed.",
))
cells.append(code(
    "!python scripts/run_amp_pilot.py --config configs/experiments/pilot_amp_fp32.yaml",
))

# ===================================================================== 12 AMP PILOT
cells.append(md(
    "## 12. AMP pilot -- 10 epochs (`results/pilot_amp_v2/amp`)",
    "",
    "Identical to the FP32 pilot in every scientific setting except `amp: true`",
    "(mixed-precision TRAINING only; evaluation stays FP32). Same checkpoint safety.",
))
cells.append(code(
    "!python scripts/run_amp_pilot.py --config configs/experiments/pilot_amp_amp.yaml",
))

# ===================================================================== 13 COMPARISON
cells.append(md(
    "## 13. FP32 vs AMP comparison",
    "",
    "Reads each arm's `pilot_history.json` (per-epoch loss / val Dice / per-class Dice",
    "/ runtime) and `eval_val_summary.csv` (volume-level Dice / IoU / HD95 / ASD), then",
    "builds a comparison table + plots and the FP32-AMP differences. Missing metrics",
    "are reported as **unavailable** -- nothing is fabricated. This cell does NOT pick",
    "a winner.",
))
cells.append(code(
    "import os, json",
    "import pandas as pd",
    "import matplotlib.pyplot as plt",
    "",
    "ARMS = {'FP32': os.path.join(DRIVE_RESULTS_DIR, 'pilot_amp_v2', 'fp32'),",
    "        'AMP':  os.path.join(DRIVE_RESULTS_DIR, 'pilot_amp_v2', 'amp')}",
    "",
    "def _load_history(d):",
    "    p = os.path.join(d, 'pilot_history.json')",
    "    if not os.path.exists(p): return None",
    "    with open(p) as f: return json.load(f)",
    "",
    "def _load_eval(d):",
    "    p = os.path.join(d, 'eval_val_summary.csv')",
    "    if not os.path.exists(p): return None",
    "    return pd.read_csv(p)",
    "",
    "def _na(x): return 'unavailable' if x is None else x",
    "",
    "hist = {k: _load_history(v) for k, v in ARMS.items()}",
    "ev = {k: _load_eval(v) for k, v in ARMS.items()}",
    "",
    "summary_rows = []",
    "for arm in ('FP32', 'AMP'):",
    "    h = hist[arm]; e = ev[arm]",
    "    row = {'arm': arm}",
    "    if h and h.get('epochs'):",
    "        eps = h['epochs']",
    "        dices = [x.get('val_dice') for x in eps if x.get('val_dice') is not None]",
    "        secs = [x.get('epoch_seconds') for x in eps if x.get('epoch_seconds') is not None]",
    "        row['epochs_done'] = len(eps)",
    "        row['best_val_dice'] = round(max(dices), 6) if dices else 'unavailable'",
    "        row['final_val_dice'] = round(dices[-1], 6) if dices else 'unavailable'",
    "        row['total_runtime_s'] = round(sum(secs), 1) if secs else 'unavailable'",
    "        row['runtime_per_epoch_s'] = round(sum(secs)/len(secs), 1) if secs else 'unavailable'",
    "        pcd = eps[-1].get('val_per_class_dice')",
    "        row['final_per_class_dice'] = pcd if pcd else 'unavailable'",
    "    else:",
    "        row.update({'epochs_done':'unavailable','best_val_dice':'unavailable',",
    "                    'final_val_dice':'unavailable','total_runtime_s':'unavailable',",
    "                    'runtime_per_epoch_s':'unavailable','final_per_class_dice':'unavailable'})",
    "    if e is not None:",
    "        d = e[e['metric']=='dice']",
    "        for _, r in d.iterrows():",
    "            if int(r['class_id']) in (1,2):",
    "                row[f'vol_dice_c{int(r[\"class_id\"])}'] = round(float(r['mean']), 6)",
    "        for metric in ('iou','hd95_mm','asd_mm'):",
    "            mm = e[e['metric']==metric]",
    "            for _, r in mm.iterrows():",
    "                if int(r['class_id']) in (1,2):",
    "                    row[f'{metric}_c{int(r[\"class_id\"])}'] = round(float(r['mean']), 4)",
    "    summary_rows.append(row)",
    "",
    "tbl = pd.DataFrame(summary_rows).set_index('arm')",
    "print('=== FP32 vs AMP pilot comparison ==='); print(tbl.T.to_string())",
    "",
    "# FP32-AMP numeric differences where both available.",
    "print('\\n=== FP32 - AMP differences (numeric fields) ===')",
    "for col in tbl.columns:",
    "    a, b = tbl.loc['FP32', col], tbl.loc['AMP', col]",
    "    if isinstance(a, (int, float)) and isinstance(b, (int, float)):",
    "        print(f'  {col}: {a - b:+.6f}')",
    "",
    "# Plots (skip cleanly if a history is unavailable).",
    "if hist['FP32'] and hist['AMP']:",
    "    fig, ax = plt.subplots(1, 3, figsize=(16, 4))",
    "    for arm in ('FP32','AMP'):",
    "        eps = hist[arm]['epochs']",
    "        xs = [e['epoch'] for e in eps]",
    "        ax[0].plot(xs, [e.get('train_loss') for e in eps], marker='o', label=arm)",
    "        ax[1].plot(xs, [e.get('val_dice') for e in eps], marker='o', label=arm)",
    "        ax[2].plot(xs, [e.get('epoch_seconds') for e in eps], marker='o', label=arm)",
    "    ax[0].set_title('train loss'); ax[1].set_title('val Dice'); ax[2].set_title('epoch seconds')",
    "    for a in ax: a.set_xlabel('epoch'); a.legend(); a.grid(alpha=0.3)",
    "    plt.tight_layout(); plt.show()",
    "else:",
    "    print('\\n(plots skipped -- run both pilot arms first)')",
))

# ===================================================================== 14 DECISION
cells.append(md(
    "## 14. Explicit decision point -- YOU choose the precision",
    "",
    "**This notebook does not choose for you.** Review Section 13 and decide whether",
    "AMP is acceptable, weighing:",
    "",
    "- **Consistency:** are AMP's best/final val Dice, per-class Dice, and volume-level",
    "  Dice/IoU/HD95/ASD close enough to FP32 that precision would not confound the",
    "  Exp02 preprocessing comparison?",
    "- **Benefit:** is AMP's runtime/epoch meaningfully lower (prior T4 benchmark",
    "  ~2.45x faster)?",
    "",
    "> **Methodological caution (important).** The official **Exp01 baseline is FP32**",
    "> and locked. If you choose **AMP** for the official Exp02 arms, precision differs",
    "> between Exp01 (FP32) and Exp02 (AMP) and becomes an *uncontrolled confound* for",
    "> any Exp01-vs-Exp02 comparison. To use AMP officially you must EITHER keep all",
    "> compared experiments on the same precision (e.g. re-run the FP32 baseline under",
    "> AMP, or restrict comparisons to within-Exp02) OR keep everything FP32. Decide",
    "> this with your supervisor before enabling AMP officially.",
    "",
    "Record your decision below. The official training cells (17-19) read `training.amp`",
    "from the committed Exp02 configs, which are **`amp: false`**. To run official Exp02",
    "under AMP you must set `training.amp: true` in all three `configs/experiments/",
    "exp02_*.yaml` (a deliberate, committed change) so all arms share one precision.",
))
cells.append(code(
    "CHOSEN_PRECISION = None   # <-- set to 'fp32' or 'amp' AFTER reviewing Section 13",
    "assert CHOSEN_PRECISION in (None, 'fp32', 'amp')",
    "if CHOSEN_PRECISION is None:",
    "    print('No precision chosen yet. Review Section 13, then set CHOSEN_PRECISION.')",
    "else:",
    "    print('You chose:', CHOSEN_PRECISION)",
    "    print('Official Exp02 configs currently have training.amp = false (FP32).')",
    "    if CHOSEN_PRECISION == 'amp':",
    "        print('To run official Exp02 under AMP, set training.amp: true in ALL THREE')",
    "        print('configs/experiments/exp02_*.yaml and mind the Exp01 FP32 confound above.')",
))

# ===================================================================== 15 EXP01 STATUS
cells.append(md(
    "## 15. Official Exp01 baseline -- status only (LOCKED, already trained)",
    "",
    "Exp01 is the **already-trained, officially-evaluated FP32 baseline** (best epoch",
    "77). It must remain untouched. This cell only reports its checkpoint status and",
    "**refuses to retrain** if a checkpoint exists, so the locked baseline can never be",
    "resumed/overwritten from here. (A from-scratch Exp01 reproduction would require a",
    "brand-new, empty output directory -- not this one.)",
))
cells.append(code(
    "from src.train import inspect_checkpoint",
    "EXP01_DIR = os.path.join(DRIVE_RESULTS_DIR, 'exp01_baseline')",
    "EXP01_CKPT = os.path.join(EXP01_DIR, 'best_model.pt')",
    "info = inspect_checkpoint(EXP01_CKPT)",
    "if info['exists']:",
    "    print(f\"Exp01 checkpoint FOUND: epoch {info['epoch']} best {info['best_metric']}\")",
    "    print('LOCKED baseline -- NOT retraining. Proceed to Section 16 (evaluation).')",
    "else:",
    "    print('No Exp01 checkpoint on Drive. Exp01 is the official baseline; retrain only')",
    "    print('deliberately into a clean dir with configs/config_baseline.yaml if truly needed.')",
))

# ===================================================================== 16 EXP01 EVAL
cells.append(md(
    "## 16. Exp01 evaluation (inference only, FP32)",
    "",
    "Volume-level, per-case, original-voxel-space validation evaluation of the locked",
    "Exp01 checkpoint. Inference only -- never trains or modifies the checkpoint.",
))
cells.append(code(
    "EXP01_DIR = os.path.join(DRIVE_RESULTS_DIR, 'exp01_baseline')",
    "EXP01_CKPT = os.path.join(EXP01_DIR, 'best_model.pt')",
    "assert os.path.exists(EXP01_CKPT), f'Exp01 checkpoint not found: {EXP01_CKPT}'",
    "!python scripts/run_evaluation.py --config configs/config_baseline.yaml --checkpoint \"{EXP01_CKPT}\" --output-dir \"{EXP01_DIR}\" --split val",
))

# ============================================== 17-19 OFFICIAL EXP02 ARMS
def _official_train_md(letter, name, factor):
    return md(
        f"## {{sec}}. Exp02-{letter} training + evaluation -- {factor}",
        "",
        f"Official 100-epoch arm (`results/exp02_preprocessing/{name}/`). Resume-safe:",
        "prints FRESH/RESUME status first and continues from `epoch+1` if a checkpoint",
        "exists -- never restarts at 1, never silently overwrites. Rerun the training",
        "cell after a timeout. Output is redirected to Drive; the committed config is",
        "untouched. Precision follows the committed config (`amp: false` = FP32) unless",
        "you deliberately changed it per Section 14.",
    )


def _official_train_code(letter, name, config):
    colab_cfg = f"/content/config_exp02_{letter.lower()}.yaml"
    return code(
        "import os, yaml",
        "from src.utils import load_config",
        "from src.train import run_training, plan_training_run",
        "",
        f"ARM_CONFIG = '{config}'",
        f"ARM_OUTPUT_DIR = os.path.join(DRIVE_RESULTS_DIR, 'exp02_preprocessing', '{name}')",
        "os.makedirs(ARM_OUTPUT_DIR, exist_ok=True)",
        "ARM_CKPT = os.path.join(ARM_OUTPUT_DIR, 'best_model.pt')",
        f"COLAB_CFG = '{colab_cfg}'",
        "",
        "assert ARM_OUTPUT_DIR.startswith('/content/drive/'), 'Official outputs must be on Drive.'",
        "cfg = load_config(ARM_CONFIG)",
        "cfg['experiment']['output_dir'] = ARM_OUTPUT_DIR   # redirect to Drive; committed config untouched",
        "NUM = int(cfg['training']['num_epochs'])           # 100 (unchanged)",
        "with open(COLAB_CFG, 'w') as f: yaml.safe_dump(cfg, f)",
        "",
        "plan = plan_training_run(ARM_CKPT, NUM)",
        "banner = {'resume':'RESUME','fresh':'FRESH'}.get(plan['action'], plan['action'].upper())",
        f"print('=' * 60); print(f'EXP02-{letter} (100ep, amp=' + str(cfg['training'].get('amp', False)) + ')  |  ' + banner)",
        "print('Output:', ARM_OUTPUT_DIR); print(plan['message']); print('=' * 60)",
        "",
        "if plan['action'] == 'fresh':",
        "    print(run_training(COLAB_CFG))",
        "elif plan['action'] == 'resume':",
        "    print(run_training(COLAB_CFG, resume_from=ARM_CKPT))",
        "elif plan['action'] == 'already_complete':",
        f"    print('Exp02-{letter} already complete -- skip to its evaluation cell below.')",
        "else:",
        "    raise RuntimeError(plan['message'])",
    )


def _official_eval_code(letter, name, config):
    return code(
        "import os",
        f"ARM_OUTPUT_DIR = os.path.join(DRIVE_RESULTS_DIR, 'exp02_preprocessing', '{name}')".replace("{name}", name),
        "ARM_CKPT = os.path.join(ARM_OUTPUT_DIR, 'best_model.pt')",
        "assert os.path.exists(ARM_CKPT), f'Train this arm first: {ARM_CKPT}'",
        f"!python scripts/run_evaluation.py --config {config} --checkpoint {{ARM_CKPT}} --output-dir {{ARM_OUTPUT_DIR}} --split val",
    )


_SEC = {"A": 17, "B": 18, "C": 19}
_FACTOR = {"A": "z-axis resampling ON", "B": "global train-only normalization", "C": "spatial resize 256x256"}
for letter in ("A", "B", "C"):
    arm = EXP02_ARMS[letter]
    m = _official_train_md(letter, arm["name"], _FACTOR[letter])
    m["source"] = [line.replace("{sec}", str(_SEC[letter])) for line in m["source"]]
    cells.append(m)
    cells.append(_official_train_code(letter, arm["name"], arm["config"]))
    cells.append(md(f"### {_SEC[letter]}b. Exp02-{letter} validation evaluation (inference only, FP32)"))
    cells.append(_official_eval_code(letter, arm["name"], arm["config"]))

# ===================================================================== 20 COLLECT
cells.append(md(
    "## 20. Final result collection (official validation comparison)",
    "",
    "Reads each official arm's `eval_val_summary.csv` and prints a per-class",
    "validation Dice comparison (baseline included when present). Validation split",
    "only -- the held-out 19-case test set is evaluated separately, once, AFTER a",
    "variant is selected.",
))
cells.append(code(
    "import os, pandas as pd",
    "sources = [('baseline (Exp01)', os.path.join(DRIVE_RESULTS_DIR, 'exp01_baseline'))]",
    "for k, a in EXP02_ARMS.items():",
    "    sources.append((f'Exp02-{k} ({a[\"name\"]})', os.path.join(DRIVE_RESULTS_DIR, 'exp02_preprocessing', a['name'])))",
    "records = []",
    "for label, d in sources:",
    "    csvp = os.path.join(d, 'eval_val_summary.csv')",
    "    if not os.path.exists(csvp):",
    "        print(f'(skip) {label}: no eval_val_summary.csv'); continue",
    "    df = pd.read_csv(csvp); dice = df[df['metric']=='dice']",
    "    for _, r in dice.iterrows():",
    "        if int(r['class_id']) == 0: continue",
    "        records.append({'arm': label, 'class_id': int(r['class_id']),",
    "                        'dice_mean': round(float(r['mean']), 4), 'n_cases': int(r['n_cases'])})",
    "if records:",
    "    t = pd.DataFrame.from_records(records)",
    "    print(t.pivot(index='arm', columns='class_id', values='dice_mean').to_string())",
    "else:",
    "    print('No official arm results yet.')",
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
print("Wrote", out_path, "with", len(cells), "cells")
