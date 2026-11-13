"""Build the Colab run that scores the final seed ensemble on PROSTATEx.

The final model is frozen by scripts/e1_seed_selection.py on the validation
split. PROSTATEx (204 cases) is too slow on a laptop CPU for a 3-member
ensemble, so it runs once on a Colab T4. This script writes:

  1. a bundle zip: the code (src/ + the PROSTATEx runner), the selection record,
     the 3 member checkpoints and the PROSTATEx manifest + checksums. The images
     themselves are NOT bundled: the notebook downloads them from the public
     source repo at the pinned commit and checks every file's SHA-256.
  2. the notebook, which downloads the bundle through the Drive API (the bundle
     sits in a folder shared with the person running it), runs inference only,
     and writes the results to their Drive.

Usage:  .venv/Scripts/python.exe scripts/_build_colab_prostatex_final.py <drive_folder>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_NAME = "prostatex_final_bundle.zip"
NOTEBOOK_NAME = "Run 6 - PROSTATEx external test, final ensemble.ipynb"
SELECTION = "results/e1_seed_selection/selection.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_bundle(dest):
    sel = json.load(open(os.path.join(REPO, SELECTION)))
    files = [SELECTION, "scripts/run_prostatex_external_validation.py", "requirements.txt"]
    files += [m["path"] for m in sel["final_model"]["members"]]
    for d in ("src",):
        for root, _, names in os.walk(os.path.join(REPO, d)):
            if "__pycache__" in root:
                continue
            files += [os.path.relpath(os.path.join(root, n), REPO).replace("\\", "/")
                      for n in names if n.endswith(".py")]
    meta = "dataset/external/prostatex/metadata"
    files += [f"{meta}/{n}" for n in os.listdir(os.path.join(REPO, meta))]
    files += ["dataset/external/prostatex/README.md"]
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_STORED) as z:
        for rel in sorted(set(files)):
            z.write(os.path.join(REPO, rel), rel)
    return sha256(dest), sel


CELLS = []


def md(s):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": [s]})


def code(s):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": [ln + "\n" for ln in s.strip("\n").split("\n")]})


def build_notebook(bundle_sha, sel):
    fm = sel["final_model"]
    md(f"""# Sandro - Prostate158 - PROSTATEx external test of the final model

**Inference only** (no training). Runtime > Change runtime type > **T4 GPU**, then **Runtime > Run all**,
and allow Google Drive when asked (twice: once to mount Drive, once to read the shared bundle).

* Model: {fm['arm']} softmax ensemble of seeds {', '.join(str(m['seed']) for m in fm['members'])}
  {'with' if fm['tta_flip_left_right'] else 'without'} left-right flip TTA, frozen on the validation split.
* Cohort: PROSTATEx, 204 cases, downloaded from the public source at a pinned commit, every file checked by SHA-256.
* Expected: about 30-60 min on a T4, about 1-2 compute units.
* Results go to `MyDrive/Prostate158_E1_seeds/final_prostatex/` and `final_prostatex.zip` next to it.
  If Colab disconnects, Run all again: finished cases are kept on Drive and skipped.""")
    code("""
import subprocess, torch
assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU'
print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout)
from google.colab import drive
drive.mount('/content/drive')
OUT_DIR = '/content/drive/MyDrive/Prostate158_E1_seeds/final_prostatex'
""")
    code(f"""
# Download the bundle shared with you (code + 3 frozen checkpoints + manifest).
import hashlib, io, os, zipfile
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
BUNDLE_NAME, BUNDLE_SHA256 = {BUNDLE_NAME!r}, {bundle_sha!r}
BUNDLE = '/content/' + BUNDLE_NAME
def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()
if not (os.path.exists(BUNDLE) and sha256(BUNDLE) == BUNDLE_SHA256):
    auth.authenticate_user()
    svc = build('drive', 'v3')
    hits = svc.files().list(q=f"name = '{{BUNDLE_NAME}}' and trashed = false",
                            fields='files(id, name, size)', includeItemsFromAllDrives=True,
                            supportsAllDrives=True).execute()['files']
    assert hits, 'Bundle not found: open the shared folder link once, then run again.'
    req = svc.files().get_media(fileId=hits[0]['id'])
    with io.FileIO(BUNDLE, 'wb') as fh:
        dl = MediaIoBaseDownload(fh, req, chunksize=64 << 20)
        done = False
        while not done:
            status, done = dl.next_chunk()
            print(f'  {{int(status.progress() * 100)}}%', end='')
    print()
assert sha256(BUNDLE) == BUNDLE_SHA256, 'Bundle checksum mismatch'
WORK = '/content/bundle'
zipfile.ZipFile(BUNDLE).extractall(WORK)
os.chdir(WORK)
print('bundle OK:', BUNDLE_SHA256[:16])
""")
    code("""
# PROSTATEx images + masks from the public source at the pinned commit; every file is checked.
import csv, json, urllib.request
from concurrent.futures import ThreadPoolExecutor
ROOT = 'dataset/external/prostatex'
prov = json.load(open(f'{ROOT}/metadata/acquisition_provenance.json'))
BASE = f"https://raw.githubusercontent.com/rcuocolo/PROSTATEx_masks/{prov['git_commit']}/Files/prostate"
SRC = {'raw/images/t2': 'Images', 'raw/masks/pz': 'mask_pz', 'raw/masks/rest': 'mask_tz'}
expected = {}
for line in open(f'{ROOT}/metadata/checksums.sha256'):
    digest, rel = line.split()
    expected[rel] = digest
def fetch(rel):
    path = os.path.join(ROOT, rel)
    if os.path.exists(path) and sha256(path) == expected[rel]:
        return rel, True
    folder, name = os.path.dirname(rel), os.path.basename(rel)
    os.makedirs(os.path.join(ROOT, folder), exist_ok=True)
    url = f"{BASE}/{SRC[folder]}/{name}"
    for _ in range(4):
        try:
            urllib.request.urlretrieve(url, path)
            if sha256(path) == expected[rel]:
                return rel, True
        except Exception as e:
            print('retry', name, e)
    return rel, False
with ThreadPoolExecutor(16) as ex:
    res = list(ex.map(fetch, sorted(expected)))
bad = [r for r, ok in res if not ok]
assert not bad, f'{len(bad)} files failed the checksum: {bad[:5]}'
print(f'{len(res)} PROSTATEx files downloaded and verified')
""")
    code("""
# Inference with the frozen ensemble (member hashes are checked against the selection record).
import sys
cmd = [sys.executable, '-u', 'scripts/run_prostatex_external_validation.py',
       '--prostatex-root', ROOT, '--checkpoint', 'results/e1_seed_selection/selection.json',
       '--output-dir', OUT_DIR, '--work-dir', '/content/prostatex_harmonized', '--device', 'cuda']
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    print(line, end='')
assert p.wait() == 0, 'run failed: Run all again to resume'
""")
    code("""
# Pack the results for Sandro (reconstructions included, harmonized copies of the data excluded).
import shutil
zip_path = shutil.make_archive(OUT_DIR, 'zip', OUT_DIR)
print('Done. Send this back:', zip_path, f'({os.path.getsize(zip_path) / 1e6:.0f} MB)')
s = json.load(open(os.path.join(OUT_DIR, 'summary.json')))
for c in ('1', '2'):
    d = s['summary'][c]['metrics']['dice']
    print(s['summary'][c]['class_name'], 'Dice', round(d['mean'], 4), 'n', d['n_cases'])
""")
    return {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main():
    drive_folder = sys.argv[1]
    bundle = os.path.join(drive_folder, BUNDLE_NAME)
    bundle_sha, sel = build_bundle(bundle)
    nb = build_notebook(bundle_sha, sel)
    for dest in (os.path.join(REPO, "notebooks", "prostate158_final_prostatex_colab.ipynb"),
                 os.path.join(drive_folder, NOTEBOOK_NAME)):
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"bundle {bundle} ({os.path.getsize(bundle) / 1e6:.0f} MB) sha256 {bundle_sha}")
    print("notebook written to notebooks/ and", drive_folder)


if __name__ == "__main__":
    main()
