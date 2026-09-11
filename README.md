# Prostate158 Zonal Anatomy Segmentation — 2D U-Net with Spatially-Verified 3D Reconstruction

> **Scope note (read first):** this project's research direction changed after
> the initial scaffolding stage. The **current thesis** is multi-class
> **prostate zonal anatomy segmentation** (label 1 / label 2, hypothesis:
> CG/TZ and PZ respectively — see `results/label_mapping/label_mapping_report.md`,
> **not yet human-confirmed**) from **T2-weighted MRI only**, on the
> Prostate158 dataset, using a fixed 2D U-Net applied slice-by-slice with an
> explicit, spatially-verified 2D→3D reconstruction. It is **segmentation-only**
> — no lesion/tumor segmentation, no classification, no multi-modal input, no
> architecture novelty, no SOTA or clinical claims. An earlier lesion/
> multimodal (T2+ADC+DWI) proposal was superseded; its code is retained but
> quarantined (see below), never used by the current pipeline.

## 1. Thesis Purpose

Measure how much **preprocessing decisions** and the **evaluation protocol**
— independent of model architecture — move the reported Dice/IoU/HD95 numbers
for prostate zonal segmentation on Prostate158, and provide a reproducible
spatial-correctness verification procedure (round-trip fidelity test) for the
2D→3D pipeline. The 2D U-Net and the 3D→2D→3D cycle are **methodological
choices, not the contribution** — the contribution is trustworthy,
reproducible measurement of a standard pipeline on a standard dataset.

Research questions: RQ-1 preprocessing effect (Experiment 2), RQ-2
evaluation-protocol effect + CG/PZ cross-paper contradiction (Experiment 3),
RQ-3 reconstruction integrity + inter-slice consistency (Experiments 1 & 4),
RQ-4 baseline replication (Experiment 1, not a novelty claim).

## 2. Project Structure

```
thesis-project/
├── src/                             # Core implementation (current thesis pipeline)
│   ├── dataset.py                   # ProstateZonal2DDataset (CURRENT) + Prostate2DDataset (RETIRED, quarantined)
│   ├── geometry.py                  # Spatial-verification authority (9-point checklist)
│   ├── transforms.py                # Config-driven preprocessing + inverse transforms
│   ├── reconstruction.py            # Explicit-index 2D->3D reconstruction, round-trip support
│   ├── splits.py                    # Official case-level Prostate158 split (119/20/19)
│   ├── metrics.py                   # Dice/IoU/HD95/ASD/precision/recall, bootstrap CI, inter-slice consistency
│   ├── model.py                     # ProstateUNet2D (2D U-Net) + build_model()
│   ├── train.py                     # Training loop (entry point: configs/config_baseline.yaml)
│   ├── evaluate.py                  # Volume-level per-case evaluation + evaluation-protocol ablation support
│   ├── preprocessing.py             # normalize_intensity (used) + legacy lesion-era helpers (unused by current path)
│   └── utils.py                     # config/seed/device helpers
│
├── scripts/                         # One-off verification / smoke-test entry points
│   ├── verify_label_mapping.py      # Gate 2: label-value -> anatomy mapping evidence
│   ├── smoke_test_020.py            # Gate 5: forward-only Case-020 smoke test
│   ├── roundtrip_test.py            # Gate 8: GT round-trip fidelity (Dice must = 1.0)
│   ├── check_baseline_config.py     # Gate 10: baseline config structural readiness (no training)
│   └── survey_shapes.py             # cohort shape/spacing survey (informed target_size choice)
│
├── configs/
│   ├── config_020.yaml              # Case-020 smoke-test config (NOT the research split)
│   ├── config_baseline.yaml         # CURRENT Experiment-1 research config (official split)
│   ├── experiments/                 # Experiment 2/3/4 ablation configs (config-driven, no training here)
│   └── legacy/                      # RETIRED lesion/multimodal configs -- quarantined, do not use
│
├── tests/                           # Current-thesis test suite (+ tests/legacy/ quarantine)
├── notebooks/legacy/                # experiment_020.ipynb (RETIRED -- still references Prostate2DDataset; quarantined, not the current entry point)
├── results/                         # label_mapping/, exp01_baseline/ .. exp04_reconstruction/, dataset_inspection
├── dataset/                         # Local raw Prostate158 data (git-ignored, never committed)
├── requirements.txt
└── README.md
```

## 3. Dataset

- **Source:** Prostate158 (Adams et al. 2022), NIfTI, official split **119
  train / 20 validation / 19 test** (case-level).
- **Local availability:** the 139-case train+valid archive is present under
  `dataset/prostate158_train/` and matches `train.csv` (119) / `valid.csv`
  (20) exactly. **The separate 19-case test archive
  (DOI 10.5281/zenodo.6592345) is NOT present locally** — confirmed absent by
  filesystem search. `src/splits.py` handles this gap explicitly (warns, does
  not fail, and will pick up the archive automatically once obtained).
- **Files used:** `t2.nii.gz` (input, T2W only) and `t2_anatomy_reader1.nii.gz`
  (target, labels `{0, 1, 2}`). Lesion masks and ADC/DWI volumes exist in the
  archive but are out of scope for this thesis.
- **Label mapping:** which integer is CG/TZ vs PZ is **not resolved by
  assumption**. See `results/label_mapping/label_mapping_report.md` for
  quantitative morphological evidence (exterior-contact fraction, centroid
  distance, sampled across 15 cases spanning the full ID range) — strongly
  and consistently pointing to **label 1 = CG/TZ, label 2 = PZ**, but this
  remains a documented hypothesis pending 3D Slicer human confirmation.

## 4. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CPU is sufficient for dataset verification, smoke tests, and the round-trip
test. Full Experiment 1-4 training runs on GPU via Google Colab (see §6).

## 5. Reproducing the verification gates locally

```bash
# Gate 2 -- label mapping evidence (writes results/label_mapping/)
python scripts/verify_label_mapping.py

# Gate 5 -- Case-020 forward-only smoke test (no training)
python scripts/smoke_test_020.py

# Gate 8 -- round-trip fidelity test (GT mask only, no model; must print Dice=1.0)
python scripts/roundtrip_test.py

# Gate 10 -- baseline config structural readiness (no training)
python scripts/check_baseline_config.py

# Full test suite
pytest tests/ -q
```

## 6. Development / Execution Workflow

```
Claude Code / VS Code  --push-->  GitHub  --pull-->  Google Colab (GPU)
        (edit, test,                (source of           |
         debug locally)              truth for code)      v
                                                    Google Drive
                                                 (Prostate158 dataset,
                                                  checkpoints, outputs)
```

- **VS Code / Claude Code:** all code authoring, debugging, and the local
  gates above (environment, geometry, smoke test, round-trip test) run here
  on CPU against a small subset of cases.
- **GitHub:** version control and the single source of truth for code. The
  dataset is never committed (`.gitignore` excludes `dataset/`, `*.nii*`,
  `*.pt`, `*.pth`).
- **Google Colab:** clones/pulls this repository, `pip install -r
  requirements.txt`, mounts Google Drive to locate the Prostate158 data, and
  runs the actual GPU training (`configs/config_baseline.yaml` and the
  `configs/experiments/*.yaml` ablations). Colab is never the place source
  code is edited.
- **Google Drive:** stores the Prostate158 archives and, optionally,
  checkpoints/experiment outputs too large for Git.

## 7. Status

See the Phase 3 implementation report (delivered in-session) for the current
gate-by-gate status, exact remaining blockers (official test-set archive,
final label-mapping human confirmation), and the exact commands to run the
Case-020 smoke test and, later, the Experiment 1 baseline.
