"""Build the two follow-up Colab notebooks for RQ2.

Both are generated here rather than edited by hand, matching the convention for
every other notebook in this repo: edit this builder, re-run it, commit the
regenerated .ipynb.

Two notebooks, because they need different runtimes and have different risk:

  prostate158_rq2_finalize_colab.ipynb
      Finishes task C (zone volumes against the annotation) and redraws the
      qualitative figures with the corrected voxel aspect ratio. CPU only --
      it reads the predictions already saved by the RQ2 run and never loads
      the model, so there is nothing here that can change a result.

  prostate158_prostatex_external_colab.ipynb
      Task D: the frozen E1 model on the 204-case PROSTATEx cohort, inference
      only. Needs a GPU. This one touches an independent cohort once, so its
      gates are strict: checkpoint hash, upload completeness, harmonization.

Usage:
    python scripts/_build_colab_notebook_rq2_followup.py
"""

from __future__ import annotations

import json
import os

NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")

REPO_URL = "https://github.com/amadeussandro/prostate-mri-segmentation-thesis.git"
DRIVE_BASE = "/content/drive/MyDrive/THESIS_PROSTATE158"
PROJECT_DIR = "/content/prostate-mri-segmentation-thesis"


def _lines(text: str) -> list:
    """nbformat stores `source` as a list of lines that each KEEP their newline,
    except the last. Splitting without re-adding them collapses the whole cell
    onto one line when the notebook is opened."""
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text.strip())}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": _lines(text.strip("\n"))}


def notebook(cells, accelerator="None") -> dict:
    return {
        "cells": cells,
        "metadata": {
            "accelerator": accelerator,
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


MOUNT = [
    md("## 1. Mount Google Drive"),
    code("from google.colab import drive\ndrive.mount('/content/drive')"),
]

REPO_CELLS = [
    md("## 3. Repository setup (clone or pull — no duplicates)"),
    code(f"""
import os, subprocess

def sh(*args, check=True):
    print("$", " ".join(args))
    return subprocess.run(args, check=check)

if not os.path.isdir(os.path.join(PROJECT_DIR, ".git")):
    sh("git", "clone", "--branch", REPO_BRANCH, REPO_URL, PROJECT_DIR)
else:
    sh("git", "-C", PROJECT_DIR, "fetch", "origin", REPO_BRANCH)
    sh("git", "-C", PROJECT_DIR, "checkout", REPO_BRANCH)
    sh("git", "-C", PROJECT_DIR, "pull", "--ff-only", "origin", REPO_BRANCH)

os.chdir(PROJECT_DIR)
print("\\ncwd:", os.getcwd())
sh("git", "-C", PROJECT_DIR, "log", "--oneline", "-1")
"""),
]


# ---------------------------------------------------------------------------
# Notebook A -- finalize RQ2 (CPU)
# ---------------------------------------------------------------------------

def build_finalize() -> dict:
    cells = [
        md(f"""
# Prostate158 RQ2 — finalize: zone volumes + corrected figures (Colab)

Two jobs that both need the Drive data and nothing else. **No GPU, no model,
no inference** — this notebook never loads the checkpoint, so it cannot change
any segmentation result.

1. **Zone volumes against the annotation** — finishes task C. Adds per-case
   error, bias and Bland–Altman limits of agreement to the predicted volumes
   already computed locally.
2. **Redraw the qualitative figures** — the saved 3D projections were rendered
   with square voxels, which compresses the coronal and sagittal panels by the
   spacing ratio (~6.4×) and makes a geometrically valid reconstruction look
   like a flat disc. The renderer is fixed; these figures predate the fix.

A CPU runtime is enough and is faster to get. Runtime → Change runtime type →
CPU, then Run all.
"""),
        *MOUNT,
        md("## 2. Configuration (edit here)"),
        code(f"""
REPO_URL     = "{REPO_URL}"
REPO_BRANCH  = "main"
PROJECT_DIR  = "{PROJECT_DIR}"

DRIVE_BASE   = "{DRIVE_BASE}"

# The completed RQ2 run (E1, epoch 88). Read, never overwritten except for the
# figures under visualizations/ and the volume outputs below.
RQ2_DIR      = f"{{DRIVE_BASE}}/results/rq2_e1_epoch88"

# Where the official 19-case test archive was extracted. Section 5 descends
# automatically to the folder that directly holds the case directories.
TEST_DIR     = f"{{DRIVE_BASE}}/dataset/test/extracted"

VOLUMES_OUT  = f"{{DRIVE_BASE}}/results/rq2_zone_volumes"
MASK_NAME    = "t2_anatomy_reader1.nii.gz"
EXPECTED_TEST_CASES = 19

for k in ["PROJECT_DIR", "DRIVE_BASE", "RQ2_DIR", "TEST_DIR", "VOLUMES_OUT"]:
    print(f"{{k:12s}} = {{globals()[k]}}")
"""),
        *REPO_CELLS,
        md("## 4. Install dependencies\n\nNo torch needed — nothing here runs the model."),
        code("""
%pip install -q nibabel>=5.4 scipy>=1.13 pyyaml>=6.0 pandas>=2.0
import nibabel, numpy, scipy, pandas, matplotlib
print("nibabel", nibabel.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__)
"""),
        md("## 5. Verify paths and resolve the test case-root"),
        code("""
import os, sys

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))

problems = []
if not os.path.isdir(DRIVE_BASE):
    problems.append(f"Drive base not found: {DRIVE_BASE} (is Drive mounted?)")
if not os.path.isdir(os.path.join(RQ2_DIR, "reconstructions_3d")):
    problems.append(f"No reconstructions_3d/ under {RQ2_DIR} — is the RQ2 run on Drive?")
if not os.path.isdir(TEST_DIR):
    problems.append(f"TEST_DIR not found: {TEST_DIR}")
if problems:
    raise FileNotFoundError("Environment check failed:" + "".join(
        chr(10) + "  - " + p for p in problems))

from audit_test_archive import find_case_root
TEST_CASE_ROOT = find_case_root(TEST_DIR)

n_recon = len([f for f in os.listdir(os.path.join(RQ2_DIR, "reconstructions_3d"))
               if f.endswith(".nii.gz")])
case_dirs = sorted(d for d in os.listdir(TEST_CASE_ROOT)
                   if os.path.isdir(os.path.join(TEST_CASE_ROOT, d)))

print(f"TEST_CASE_ROOT  = {TEST_CASE_ROOT}")
print(f"test cases      = {len(case_dirs)}  (expected {EXPECTED_TEST_CASES})")
print(f"reconstructions = {n_recon}  (expected {EXPECTED_TEST_CASES})")

assert len(case_dirs) == EXPECTED_TEST_CASES, (
    f"Resolved case-root holds {len(case_dirs)} folders, expected {EXPECTED_TEST_CASES}.")
assert n_recon == EXPECTED_TEST_CASES, (
    f"{RQ2_DIR}/reconstructions_3d holds {n_recon} volumes, expected {EXPECTED_TEST_CASES}. "
    "A partial Drive sync is the likely cause.")
print("\\nOK — cohort complete.")
"""),
        md("""
## 6. Zone volumes against the annotation (finishes task C)

Volume is voxel count × |det(affine)|, so these numbers are only meaningful
because the reconstruction restored the source affine exactly. Bias is
mean(prediction − ground truth); the **limits of agreement**, not the bias, are
what decide whether a volume is usable for an individual patient.
"""),
        code("""
# Colab expands {NAME} from the Python namespace inside a ! line.
!python scripts/run_zone_volumes.py \\
    --recon-dir  "{RQ2_DIR}/reconstructions_3d" \\
    --gt-root    "{TEST_CASE_ROOT}" \\
    --mask-name  "{MASK_NAME}" \\
    --output-dir "{VOLUMES_OUT}"
"""),
        md("## 7. Redraw the qualitative figures with the corrected aspect ratio"),
        code("""
!python scripts/regenerate_rq2_figures.py \\
    --rq2-dir   "{RQ2_DIR}" \\
    --gt-root   "{TEST_CASE_ROOT}" \\
    --mask-name "{MASK_NAME}"
"""),
        md("## 8. Look at the results"),
        code("""
import os
from IPython.display import Markdown, Image, display

report = os.path.join(VOLUMES_OUT, "report.md")
if os.path.exists(report):
    display(Markdown(open(report, encoding="utf-8").read()))

viz = os.path.join(RQ2_DIR, "visualizations")
for pid in sorted(os.listdir(viz)):
    proj = os.path.join(viz, pid, f"{pid.replace('patient_', 'patient_')}_3d_projections.png")
    if os.path.exists(proj):
        print(pid)
        display(Image(filename=proj, width=950))
"""),
        md("""
## 9. What changed, and what did not

- The **segmentation metrics are untouched.** This notebook never loaded the
  model. `metrics_per_case.csv`, `metrics_summary.csv`, `summary.json` and the
  reconstructed volumes are exactly as the RQ2 run left them.
- The figures under `visualizations/` were **overwritten** with correctly
  proportioned versions. The coronal and sagittal panels should now show a
  walnut, not a pancake.
- `results/rq2_zone_volumes/` now carries agreement figures. Until this ran,
  those volumes were model output with no accuracy claim attached.
"""),
    ]
    return notebook(cells, accelerator="None")


# ---------------------------------------------------------------------------
# Notebook B -- PROSTATEx external validation (GPU)
# ---------------------------------------------------------------------------

def build_prostatex() -> dict:
    cells = [
        md(f"""
# External validation on PROSTATEx — frozen E1 model (Colab)

**Task D.** The frozen E1 model (epoch 88) run once over the 204-case PROSTATEx
cohort. **Inference only**: no training, no fine-tuning, no threshold search, no
model selection. Needs a GPU runtime.

### Read this before you look at the numbers

PROSTATEx is an **independent** cohort — different scanners, different
annotators, different acquisition geometry (0.3–0.6 mm in-plane against
Prostate158's ~0.47 mm; 3.0–4.5 mm through-plane). **Expect lower Dice than the
Prostate158 test set.** That is a finding about generalization, which is exactly
what JMIR asks for when it requires independent dataset validation. It is not a
failure, and this framing is fixed here, before the numbers exist, so it cannot
be rationalized afterwards.

### Upload checklist

Upload the cohort to Drive preserving the layout:

```
{DRIVE_BASE}/dataset/external/prostatex/
├── raw/images/t2/          (204 files)
├── raw/masks/pz/           (204 files)
├── raw/masks/rest/         (204 files)
└── metadata/prostatex_manifest.csv
```

The manifest is **required**, not optional: 15 cases use non-canonical mask
filenames that cannot be derived from the case id. Section 5 refuses to continue
if any of these is short — a partial upload is the realistic failure here and it
would otherwise be scored as if the cohort were complete.
"""),
        *MOUNT,
        md("## 2. Configuration (edit here)"),
        code(f"""
REPO_URL     = "{REPO_URL}"
REPO_BRANCH  = "main"
PROJECT_DIR  = "{PROJECT_DIR}"

DRIVE_BASE   = "{DRIVE_BASE}"
PROSTATEX_ROOT = f"{{DRIVE_BASE}}/dataset/external/prostatex"

# The frozen RQ1 model — identical weights to the Prostate158 test result.
CHECKPOINT_PATH = f"{{DRIVE_BASE}}/results/exp_e1_dice_ce/best_model.pt"
OUTPUT_DIR      = f"{{DRIVE_BASE}}/results/external_prostatex_e1"

EXPECTED_SHA256 = "2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2"
EXPECTED_CASES  = 204

for k in ["PROJECT_DIR", "DRIVE_BASE", "PROSTATEX_ROOT", "CHECKPOINT_PATH", "OUTPUT_DIR"]:
    print(f"{{k:16s}} = {{globals()[k]}}")
print(f"{{'expected SHA':16s}} = {{EXPECTED_SHA256}}")
"""),
        *REPO_CELLS,
        md("## 4. Install dependencies"),
        code("""
%pip install -q nibabel>=5.4 scipy>=1.13 pyyaml>=6.0 pandas>=2.0
import torch, nibabel, numpy, scipy
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
print("nibabel", nibabel.__version__, "| numpy", numpy.__version__)
if not torch.cuda.is_available():
    print("\\nWARNING: no GPU. 204 cases on CPU will be slow — "
          "Runtime > Change runtime type > GPU.")
"""),
        md("""
## 5. Upload completeness gate

The realistic failure mode is a Drive upload that silently dropped files. 612
NIfTI volumes through the web uploader is exactly the situation where that
happens, and a short cohort scored as complete would produce wrong numbers with
no error. This cell refuses to continue unless all four pieces are present.
"""),
        code("""
import os, sys, csv

if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "scripts"))

expected = {
    "raw/images/t2":   EXPECTED_CASES,
    "raw/masks/pz":    EXPECTED_CASES,
    "raw/masks/rest":  EXPECTED_CASES,
}
problems = []
for rel, n_expected in expected.items():
    d = os.path.join(PROSTATEX_ROOT, rel)
    if not os.path.isdir(d):
        problems.append(f"missing directory: {d}")
        continue
    n = len([f for f in os.listdir(d) if f.endswith(".nii.gz")])
    status = "OK" if n == n_expected else "SHORT"
    print(f"  {rel:18s} {n:4d} / {n_expected}   {status}")
    if n != n_expected:
        problems.append(f"{rel} holds {n} files, expected {n_expected}")

manifest = os.path.join(PROSTATEX_ROOT, "metadata", "prostatex_manifest.csv")
if not os.path.isfile(manifest):
    problems.append(f"manifest missing: {manifest} (REQUIRED — 15 cases have "
                    "non-canonical filenames that cannot be derived from the case id)")
else:
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    print(f"  {'manifest':18s} {len(rows):4d} rows")
    if len(rows) != EXPECTED_CASES:
        problems.append(f"manifest lists {len(rows)} cases, expected {EXPECTED_CASES}")

if problems:
    raise FileNotFoundError(
        "Cohort incomplete — a partial Drive upload is the likely cause. "
        "Re-upload the missing pieces before running anything:" +
        "".join(chr(10) + "  - " + p for p in problems))
print("\\nOK — all 204 cases and the manifest are present.")
"""),
        md("""
## 6. Checkpoint identity

A filename proves nothing; the hash does. These must be the same weights that
produced the Prostate158 test result, or the comparison between the two cohorts
means nothing.
"""),
        code("""
import hashlib, os

digest = hashlib.sha256()
with open(CHECKPOINT_PATH, "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
        digest.update(block)
sha = digest.hexdigest()

print(f"checkpoint : {CHECKPOINT_PATH}")
print(f"size       : {os.path.getsize(CHECKPOINT_PATH):,} bytes")
print(f"SHA-256    : {sha}")
assert sha == EXPECTED_SHA256, (
    "SHA-256 MISMATCH — this is not the frozen E1 checkpoint.\\n"
    f"  expected {EXPECTED_SHA256}\\n  got      {sha}")
print("\\nOK — frozen E1 confirmed.")
"""),
        md("""
## 7. Dry run: harmonize and verify, without the model

PROSTATEx ships PZ and "rest" as two separate binary masks; the model's
convention is `0=background, 1=CG, 2=PZ`. The combination is built here and
never written back into `raw/`.

Two details this step gets right, both of which would otherwise corrupt the run
silently: masks are combined with `mask > 0` rather than `mask == 1` (case
ProstateX-0086 carries a stray voxel of value 2 in its PZ mask), and zone
disjointness is re-measured per case instead of trusted.

Nothing here needs the GPU, so a failure costs you seconds rather than a run.
"""),
        code("""
# Colab expands {NAME} from the Python namespace inside a ! line.
!python scripts/run_prostatex_external_validation.py \\
    --prostatex-root "{PROSTATEX_ROOT}" \\
    --output-dir     "{OUTPUT_DIR}" \\
    --checkpoint     "{CHECKPOINT_PATH}" \\
    --prepare-only
"""),
        md("""
## 8. Run the external validation

Inference over 204 cases through the **same** inference-and-reconstruction path
used for RQ1 and RQ2 (`predict_case_reconstructed`), so there is no second
implementation that could drift from the one the thesis reports.
"""),
        code("""
!python scripts/run_prostatex_external_validation.py \\
    --prostatex-root "{PROSTATEX_ROOT}" \\
    --checkpoint     "{CHECKPOINT_PATH}" \\
    --output-dir     "{OUTPUT_DIR}"
"""),
        md("## 9. Results"),
        code("""
import os
from IPython.display import Markdown, display
import pandas as pd

report = os.path.join(OUTPUT_DIR, "report.md")
if os.path.exists(report):
    display(Markdown(open(report, encoding="utf-8").read()))

geom = os.path.join(OUTPUT_DIR, "cohort_geometry.csv")
if os.path.exists(geom):
    df = pd.read_csv(geom)
    print("\\nCases cropped by the 442 target:",
          int(df["inplane_exceeds_target"].sum()))
    lost = df[(df["class1_lost"] > 0) | (df["class2_lost"] > 0)]
    print("Cases losing annotated foreground to the crop:", len(lost))
    if len(lost):
        display(lost[["case_id", "shape", "class1_lost", "class2_lost"]])
    print("\\nIn-plane scale vs Prostate158: "
          f"{df['scale_vs_prostate158'].min():.2f}x to {df['scale_vs_prostate158'].max():.2f}x")
"""),
        md("""
## 10. How to write this up

Do **not** present these numbers as a like-for-like contrast with the
Prostate158 test result. The annotators, scanners, geometry and zone definitions
all differ, so the comparison is not controlled. They answer a different
question — does a model trained on one cohort retain useful zonal segmentation
on an independent one — and the honest unit of that answer is the **gap**,
reported next to the measured covariate shift in Section 3 of the report that
partly explains it.

The pipeline does no in-plane resampling, so part of any drop is scale shift
rather than model failure. If you want to separate the two, that is a second
run with a resampling adapter, and it is a scientific decision to take with your
supervisor rather than a knob to turn quietly.
"""),
    ]
    return notebook(cells, accelerator="GPU")


def main() -> None:
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)
    for name, nb in [("prostate158_rq2_finalize_colab.ipynb", build_finalize()),
                     ("prostate158_prostatex_external_colab.ipynb", build_prostatex())]:
        path = os.path.join(NOTEBOOK_DIR, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(nb, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print(f"wrote {path}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
