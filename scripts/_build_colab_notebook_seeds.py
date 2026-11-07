"""Generate notebooks/prostate158_e1_seeds_25d_colab.ipynb.

One notebook covers every run of results/seed_25d_prespecification.md: the
person running it sets ARM and SEED in the first code cell. It is written to be
run on someone else's Colab account, so it downloads Prostate158 itself from
Zenodo (md5-checked), writes only to one folder of that account's Drive, and
packs each run into a small zip to send back.

    python scripts/_build_colab_notebook_seeds.py
"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "notebooks", "prostate158_e1_seeds_25d_colab.ipynb")

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n")})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n")})


md("""
# Prostate158 E1: seed replicates and 2.5D input

**How to run**

1. `Runtime > Change runtime type > T4 GPU`
2. Set `ARM`, `SEED` and `SMOKE_TEST` in the first code cell below.
3. `Runtime > Run all`, and allow the Google Drive prompt.

Run the **smoke test first** (`SMOKE_TEST = True`, about 10-15 minutes). It trains 2 epochs,
prints the seconds per epoch and estimates the compute units of a full run. Then check
the usage rate in the Colab **Resources** panel (top right).

If Colab disconnects during a full run, open the notebook and **Run all** again with the
same `ARM` and `SEED`: training resumes from the last finished epoch on Drive.

Everything is written to `MyDrive/Prostate158_E1_seeds/<run>/`. At the end a small zip
(`<run>.zip`, about 100 MB) is written there; that zip is all that needs to be sent back.

| Run order | ARM | SEED |
|---|---|---|
| 1 | `2.5d` | 42 |
| 2 | `2d` | 43 |
| 3 | `2.5d` | 43 |
| 4 | `2d` | 44 |
| 5 | `2.5d` | 44 |

The rule for choosing between 2D and 2.5D was written before these runs:
`results/seed_25d_prespecification.md`. This notebook never reads the test split.
""")

code("""
# ===== Set these 3 values, then Runtime > Run all =====
ARM = '2.5d'        # '2d' or '2.5d'
SEED = 42           # 42, 43 or 44
SMOKE_TEST = True   # True: 2 epochs to measure speed. False: the full 100-epoch run.
# =======================================================
REPO_BRANCH = 'e1-seeds-25d'
assert ARM in ('2d', '2.5d') and SEED in (42, 43, 44)
RUN = f"e1_{ARM.replace('.', '')}_seed{SEED}" + ('_smoke' if SMOKE_TEST else '')
print('Run:', RUN)
""")

md("## 1. GPU")
code("""
import subprocess, torch
subprocess.run(['nvidia-smi'])
assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU'
GPU_NAME = torch.cuda.get_device_name(0)
print('GPU:', GPU_NAME, '| torch', torch.__version__)
if 'T4' not in GPU_NAME:
    print('WARNING: not a T4. The earlier runs used a T4; please switch if possible.')
# FP32 everywhere, as in the earlier runs (TF32 would change the arithmetic on newer GPUs).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
""")

md("## 2. Google Drive (output only)")
code("""
import os
from google.colab import drive
drive.mount('/content/drive')
OUT_ROOT = '/content/drive/MyDrive/Prostate158_E1_seeds'
OUT_DIR = os.path.join(OUT_ROOT, RUN)
os.makedirs(OUT_DIR, exist_ok=True)
print('Output folder:', OUT_DIR)
""")

md("## 3. Code (public GitHub repository)")
code("""
REPO_URL = 'https://github.com/amadeussandro/prostate-mri-segmentation-thesis.git'
REPO_DIR = '/content/thesis-project'
if not os.path.isdir(REPO_DIR):
    subprocess.run(['git', 'clone', '--branch', REPO_BRANCH, REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
subprocess.run(['git', 'fetch', 'origin', REPO_BRANCH], check=True)
subprocess.run(['git', 'checkout', REPO_BRANCH], check=True)
subprocess.run(['git', 'reset', '--hard', f'origin/{REPO_BRANCH}'], check=True)
COMMIT = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
print('Commit:', COMMIT)
import sys
sys.path.insert(0, REPO_DIR)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'nibabel', 'scipy', 'pandas',
                'pyyaml', 'matplotlib', 'SimpleITK', 'pytest'], check=True)
""")

md("""
## 4. Prostate158 training data (downloaded from Zenodo, about 2.8 GB)

Public data (record 6481141, CC BY 4.0). Downloaded to the Colab disk, not to Drive.
""")
code("""
import hashlib, zipfile, glob
ZIP_URL = 'https://zenodo.org/api/records/6481141/files/prostate158_train.zip/content'
ZIP_MD5 = 'a0d5361b7c4cbf3f17f7ccb9a2e1739c'
DATA_DIR = '/content/p158'
ZIP_PATH = '/content/prostate158_train.zip'

found = glob.glob(f'{DATA_DIR}/**/train.csv', recursive=True)
if not found:
    if not os.path.isfile(ZIP_PATH):
        subprocess.run(['wget', '-q', '--show-progress', '-O', ZIP_PATH, ZIP_URL], check=True)
    h = hashlib.md5()
    with open(ZIP_PATH, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    assert h.hexdigest() == ZIP_MD5, f'md5 mismatch: {h.hexdigest()} (download again)'
    with zipfile.ZipFile(ZIP_PATH) as z:
        z.extractall(DATA_DIR)
    os.remove(ZIP_PATH)
    found = glob.glob(f'{DATA_DIR}/**/train.csv', recursive=True)
assert len(found) == 1, found
P158_ROOT = os.path.dirname(found[0])          # holds train/, train.csv, valid.csv

# The configs expect dataset/prostate158_train/{train, train.csv, valid.csv}.
os.makedirs('/content/dslink', exist_ok=True)
if not os.path.exists('/content/dslink/prostate158_train'):
    os.symlink(P158_ROOT, '/content/dslink/prostate158_train')
if not os.path.exists(os.path.join(REPO_DIR, 'dataset')):
    os.symlink('/content/dslink', os.path.join(REPO_DIR, 'dataset'))

from src.data_integrity import verify_dataset_cache, format_cache_report
DS_ROOT = 'dataset/prostate158_train/train'
report = verify_dataset_cache(DS_ROOT, 'dataset/prostate158_train/train.csv',
                              'dataset/prostate158_train/valid.csv')
print(format_cache_report(report))
assert bool(report['ok']), 'Dataset incomplete'
""")

md("## 5. Prespecification fingerprint and control tests")
code("""
PRESPEC = os.path.join(REPO_DIR, 'results', 'seed_25d_prespecification.md')
PRESPEC_SHA = hashlib.sha256(open(PRESPEC, 'rb').read()).hexdigest()
print('Prespecification SHA-256:', PRESPEC_SHA)
rc = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests/test_e1_25d_config.py',
                     'tests/test_e1_config.py', 'tests/test_losses.py'], cwd=REPO_DIR).returncode
assert rc == 0, 'Control tests failed: do not train.'
""")

md("## 6. Run configuration")
code("""
import copy, yaml, json
from src.utils import load_config, build_config_fingerprint, format_config_fingerprint
CFG_REPO = os.path.join(REPO_DIR, 'configs', 'experiments',
                        'exp_e1_dice_ce.yaml' if ARM == '2d' else 'exp_e1_25d.yaml')
committed = load_config(CFG_REPO)
cfg = copy.deepcopy(committed)
cfg['experiment']['output_dir'] = OUT_DIR
cfg['experiment']['name'] = RUN
cfg['training']['seed'] = SEED
if SMOKE_TEST:
    cfg['training']['num_epochs'] = 2

# Only output_dir, name, seed (and the smoke-test epoch count) may differ from the repo.
for sec in committed:
    for k in set(committed[sec]) | set(cfg[sec]):
        allowed = {('experiment', 'output_dir'), ('experiment', 'name'), ('training', 'seed')}
        if SMOKE_TEST:
            allowed.add(('training', 'num_epochs'))
        if (sec, k) not in allowed:
            assert committed[sec].get(k) == cfg[sec].get(k), (sec, k)
assert 'test_csv' not in cfg['data'] and cfg['training']['amp'] is False

CFG_PATH = os.path.join(OUT_DIR, 'config.yaml')
with open(CFG_PATH, 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
fp = build_config_fingerprint(cfg, device='cuda')
print(format_config_fingerprint(fp))
assert fp['amp'] is False and fp['seed'] == SEED and fp['references_test_split'] is False
json.dump({'run': RUN, 'arm': ARM, 'seed': SEED, 'smoke_test': SMOKE_TEST, 'commit': COMMIT,
           'gpu': GPU_NAME, 'torch': torch.__version__, 'prespec_sha256': PRESPEC_SHA,
           'config_file': os.path.basename(CFG_REPO)},
          open(os.path.join(OUT_DIR, 'run_info.json'), 'w'), indent=2)
""")

md("## 7. Training (FP32, resumes automatically)")
code("""
import time
from src.train import plan_training_run, run_training
LATEST = os.path.join(OUT_DIR, 'latest_checkpoint.pt')
BEST = os.path.join(OUT_DIR, 'best_model.pt')
HIST = os.path.join(OUT_DIR, 'training_history.json')
plan = plan_training_run(LATEST, cfg['training']['num_epochs'])
print(plan['message'])
assert plan['action'] != 'corrupt', 'Checkpoint unreadable: stop and report.'
if plan['action'] != 'already_complete':
    t0 = time.time()
    print(run_training(CFG_PATH, resume_from=plan['resume_from'], history_path=HIST))
    print(f'Elapsed this session: {(time.time() - t0) / 3600:.2f} h')
""")

md("""
## 8. Speed and cost estimate

Type the rate from the Colab **Resources** panel ("... compute units per hour") into
`RATE_CU_PER_HOUR` to see the cost of one full 100-epoch run.
""")
code("""
RATE_CU_PER_HOUR = 6.0   # replace with the value shown in Resources
h = json.load(open(HIST))
secs = [e['epoch_seconds'] for e in h['epochs'] if e.get('epoch_seconds')]
per_epoch = sorted(secs)[len(secs) // 2]
full_h = per_epoch * 100 / 3600
print(f'Seconds per epoch (median of {len(secs)}): {per_epoch:.0f}')
print(f'Full 100-epoch run: about {full_h:.1f} h, about {full_h * RATE_CU_PER_HOUR:.0f} compute units')
print(f'5 runs: about {5 * full_h * RATE_CU_PER_HOUR:.0f} compute units')
""")

md("## 9. Validation evaluation (20 validation cases; the test split is not touched)")
code("""
from src.evaluate import run_evaluation
res = run_evaluation(config_path=CFG_PATH, checkpoint_path=BEST, output_dir=OUT_DIR,
                     split='val', class_ids=(0, 1, 2))
s = res['summary']
cg, pz = s[1]['dice']['mean'], s[2]['dice']['mean']
print(f'{RUN}: validation CG {cg:.4f} | PZ {pz:.4f} | macro {(cg + pz) / 2:.4f}')
""")

md("## 10. Pack the results to send back")
code("""
import shutil
best_sha = hashlib.sha256(open(BEST, 'rb').read()).hexdigest()
info = json.load(open(os.path.join(OUT_DIR, 'run_info.json')))
info.update({'best_model_sha256': best_sha, 'val_cg_dice': cg, 'val_pz_dice': pz,
             'val_macro_dice': (cg + pz) / 2, 'seconds_per_epoch': per_epoch})
json.dump(info, open(os.path.join(OUT_DIR, 'run_info.json'), 'w'), indent=2)
pack = f'/content/{RUN}'
os.makedirs(pack, exist_ok=True)
for name in os.listdir(OUT_DIR):
    if name != 'latest_checkpoint.pt' and not name.endswith('.zip'):
        src = os.path.join(OUT_DIR, name)
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(pack, name))
zip_path = shutil.make_archive(os.path.join(OUT_ROOT, RUN), 'zip', pack)
print('Send this file back:', zip_path, f'({os.path.getsize(zip_path) / 1e6:.0f} MB)')
print(json.dumps(info, indent=2))
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("wrote", OUT, len(cells), "cells")
