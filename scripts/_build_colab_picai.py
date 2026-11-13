"""Build the Colab run for the PI-CAI PSA-density analysis (inference only).

The prespecification (results/picai_psad_prespecification.md) was hash-locked
before any PI-CAI file was downloaded. Inference of the 3-member ensemble with
flip averaging over 1440 studies takes about 45 h on the laptop CPU, so it runs
on a Colab T4. This script writes:

  1. a bundle zip: the code (src/ + the PI-CAI runner and the R1 rule it
     reuses), the selection record, the 3 member checkpoints, the locked
     prespecification with its lock record, and the cohort built locally from the
     marksheet. The images are NOT bundled: the notebook downloads each fold from
     Zenodo and checks its MD5 against the Zenodo record.
  2. the notebook, which works fold by fold (download, convert, infer, delete),
     keeps every finished case on the runner's Drive and so resumes after a
     disconnect, and writes the per-case results to Drive.

Overlap, round-trip and the analysis itself run locally afterwards.

Usage:  .venv/Scripts/python.exe scripts/_build_colab_picai.py <drive_folder>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import zipfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE_NAME = "picai_psad_bundle.zip"
NOTEBOOK_NAME = "Run 7 - PI-CAI PSA density, final ensemble.ipynb"
SELECTION = "results/e1_seed_selection/selection.json"
LABELS_COMMIT = "ce4a4723d7c46d882a6cbaacb40ed6c4be86282f"
FOLDS_MD5 = {  # from https://zenodo.org/api/records/6624726 (v2.0)
    0: "b0cdd8f5fb7733ca585dca3541586a34",
    1: "d7c8cd0efc0e7f264df8d5acf860662f",
    2: "5ac0caf27cfe94dfc04bb408a971b351",
    3: "6602bce3d0882d7b9c2153d880c97e96",
    4: "8941720d8a8ee66571691cfb10dbee66",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_bundle(dest):
    sel = json.load(open(os.path.join(REPO, SELECTION)))
    files = [SELECTION, "scripts/run_picai_psad.py", "scripts/run_nci_external_validation.py",
             "requirements.txt", "results/picai_psad_prespecification.md",
             "results/picai_psad/prespec_lock.json", "results/picai_psad/erratum_2026-11-12.md",
             "results/picai_psad/cohort.csv", "results/picai_psad/eligibility.json"]
    files += [m["path"].replace("\\", "/") for m in sel["final_model"]["members"]]
    for root, _, names in os.walk(os.path.join(REPO, "src")):
        if "__pycache__" in root:
            continue
        files += [os.path.relpath(os.path.join(root, n), REPO).replace("\\", "/")
                  for n in names if n.endswith(".py")]
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


def build_notebook(bundle_sha, sel, n_eligible, folds, out_rel, note=""):
    fm = sel["final_model"]
    md(f"""# Sandro - Prostate158 - PI-CAI: PSA density from the final model

**Inference only** (no training). Runtime > Change runtime type > **T4 GPU**, then **Runtime > Run all**,
and allow Google Drive when asked (twice: once to mount Drive, once to read the bundle).
Keep this tab open while it runs.

* Model: {fm['arm']} softmax ensemble of seeds {', '.join(str(m['seed']) for m in fm['members'])}
  {'with' if fm['tta_flip_left_right'] else 'without'} left-right flip TTA, frozen on the validation split.
* Cohort: PI-CAI v2.0 (Zenodo 6624726), {n_eligible} eligible first studies, axial T2W only.
  This notebook runs folds {', '.join(map(str, folds))}. Each fold (about 5.3 GB) is downloaded from
  Zenodo, checked by MD5, converted, inferred and deleted.{note}
* Expected: about {max(1, round(len(folds) * 0.6))}-{round(len(folds) * 0.8 + 0.9)} h on a T4.
* Results go to `MyDrive/{out_rel}/` (and a zip next to it). If Colab disconnects, **Run all again**:
  finished folds and finished cases are kept on Drive and skipped.""")
    code(f"""
import subprocess, torch
assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU'
print(subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout)
from google.colab import drive
drive.mount('/content/drive')
import os
OUT_DIR = '/content/drive/MyDrive/{out_rel}'
FOLDS = {list(folds)!r}
for sub in ('conversions', 'cache_infer', 'predictions'):
    os.makedirs(os.path.join(OUT_DIR, sub), exist_ok=True)
""")
    code(f"""
# Download the bundle (code + 3 frozen checkpoints + locked prespecification + cohort).
import hashlib, io, zipfile
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
    assert hits, 'Bundle not found in your Drive.'
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
    code(f"""
# Labels at the pinned commit (marksheet checked against the local cohort build), and the work folders.
import json, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'SimpleITK==2.4.1', 'nibabel>=5.4'], check=True)
DATA = '/content/picai'
LAB = DATA + '/picai_labels'
os.makedirs(DATA + '/zips', exist_ok=True)
if not os.path.isdir(LAB):
    subprocess.run(['git', 'clone', '-q', 'https://github.com/DIAGNijmegen/picai_labels.git', LAB], check=True)
subprocess.run(['git', '-C', LAB, 'checkout', '-q', {LABELS_COMMIT!r}], check=True)
elig = json.load(open('results/picai_psad/eligibility.json'))
assert sha256(LAB + '/clinical_information/marksheet.csv') == elig['marksheet_sha256'], 'marksheet differs'
for sub in ('conversions', 'cache_infer', 'predictions'):  # finished work lives on Drive
    link = os.path.join(DATA, sub)
    if not os.path.islink(link):
        os.symlink(os.path.join(OUT_DIR, sub), link)
import SimpleITK, nibabel, numpy, scipy
print('labels', {LABELS_COMMIT[:7]!r}, '| SimpleITK', SimpleITK.Version_VersionString(), '| nibabel',
      nibabel.__version__, '| torch', torch.__version__, '| eligible', elig['n_eligible'])
""")
    code(f"""
# Fold by fold: download + MD5, convert (prepare), infer on the GPU, then free the disk.
import shutil, time
MD5 = {FOLDS_MD5!r}
def md5(p):
    h = hashlib.md5()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''): h.update(b)
    return h.hexdigest()
def run(args):
    p = subprocess.Popen([sys.executable, '-u', 'scripts/run_picai_psad.py', '--data', DATA] + args,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end='')
    assert p.wait() == 0, 'step failed: Run all again to resume'
for k in FOLDS:
    marker = os.path.join(OUT_DIR, f'fold{{k}}.done')
    if os.path.exists(marker):
        print(f'fold {{k}}: done earlier, skipped'); continue
    name = f'picai_public_images_fold{{k}}.zip'
    z = os.path.join(DATA, 'zips', name)
    for attempt in range(5):
        if os.path.exists(z) and md5(z) == MD5[k]:
            break
        t = time.time()
        subprocess.run(['curl', '-sS', '-L', '--retry', '5', '-C', '-', '-o', z,
                        f'https://zenodo.org/api/records/6624726/files/{{name}}/content'])
        print(f'fold {{k}}: downloaded in {{time.time() - t:.0f}} s')
        if os.path.exists(z) and os.path.getsize(z) > 0 and md5(z) != MD5[k]:
            os.remove(z)
    assert md5(z) == MD5[k], f'fold {{k}}: MD5 mismatch'
    run(['--phase', 'prepare'])
    run(['--phase', 'infer', '--device', 'cuda'])
    os.remove(z)
    shutil.rmtree(os.path.join(DATA, 'work'), ignore_errors=True)
    open(marker, 'w').write(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    print(f'fold {{k}}: done')
""")
    code("""
# Copy the summary files next to the per-case results and check the counts.
for f in ('conversion.csv', 'per_case.csv', 'model_identity.json'):
    shutil.copy2(os.path.join('results/picai_psad', f), os.path.join(OUT_DIR, f))
import csv
conv = list(csv.DictReader(open(os.path.join(OUT_DIR, 'conversion.csv'))))
pc = list(csv.DictReader(open(os.path.join(OUT_DIR, 'per_case.csv'))))
ok = sum(r['status'] == 'ok' for r in conv)
print(f"folds {FOLDS}: converted {len(conv)} (ok {ok}); inferred {len(pc)}")
print('geometry check passed:', sum(r['geometry_ok'] == 'True' for r in pc), 'of', len(pc))
zip_path = shutil.make_archive(OUT_DIR, 'zip', OUT_DIR)
print('Done. Results are in', OUT_DIR, 'and', zip_path, f'({os.path.getsize(zip_path) / 1e6:.0f} MB); tell Sandro it finished.')
""")
    return {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def main():
    import argparse
    import shutil
    ap = argparse.ArgumentParser()
    ap.add_argument("drive_folder")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--out-rel", default="Prostate158_PICAI", help="results folder, relative to MyDrive")
    ap.add_argument("--reuse-bundle", help="copy this existing bundle instead of building a new one, "
                                           "so every fold runs the identical code")
    ap.add_argument("--notebook-name", default=NOTEBOOK_NAME)
    ap.add_argument("--repo-notebook", default="picai_psad_colab.ipynb")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    folds = [int(k) for k in a.folds.split(",")]
    os.makedirs(a.drive_folder, exist_ok=True)
    bundle = os.path.join(a.drive_folder, BUNDLE_NAME)
    sel = json.load(open(os.path.join(REPO, SELECTION)))
    if a.reuse_bundle:
        if os.path.abspath(a.reuse_bundle) != os.path.abspath(bundle):
            shutil.copy2(a.reuse_bundle, bundle)
        bundle_sha = sha256(bundle)
        if bundle_sha != sha256(a.reuse_bundle):
            raise RuntimeError("bundle copy differs from its source")
    else:
        bundle_sha, sel = build_bundle(bundle)
    elig = json.load(open(os.path.join(REPO, "results/picai_psad/eligibility.json")))
    nb = build_notebook(bundle_sha, sel, elig["n_eligible"], folds, a.out_rel, a.note)
    for dest in (os.path.join(REPO, "notebooks", a.repo_notebook),
                 os.path.join(a.drive_folder, a.notebook_name)):
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"bundle {bundle} ({os.path.getsize(bundle) / 1e6:.0f} MB) sha256 {bundle_sha}")
    print("notebook written to notebooks/ and", a.drive_folder)


if __name__ == "__main__":
    main()
