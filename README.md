# Deep Learning-Based 2D Prostate MRI Segmentation

## 1. Thesis Project Purpose

This thesis project investigates automated lesion segmentation in multi-parametric and bi-parametric prostate Magnetic Resonance Imaging (MRI) using deep convolutional neural networks (2D U-Net architectures). The research aims to segment clinically significant prostate cancer lesions from co-registered multi-modal MRI acquisitions (T2-weighted, ADC maps, and high b-value DWI) on the standardized **Prostate158** benchmark dataset.

---

## 2. Project Structure

The repository follows a clean, modular architecture separating configuration, data loading, preprocessing, modeling, training, and evaluation:

```
thesis-project/
├── code/                           # Core implementation modules
│   ├── img-generate.py             # Exploration script for quick slice visualization
│   ├── inspect_patient_020.py      # Automated NIfTI inspection for patient 020
│   ├── visualize_patient_020.py    # 8-panel multi-modal visualization generator
│   ├── dataset.py                  # 2D slice dataset loader & volume indexer
│   ├── preprocessing.py            # Normalization, binarization, ROI cropping, channel stacking
│   ├── model.py                    # 2D U-Net segmentation architecture definitions
│   ├── train.py                    # Training loop, loss computation (Dice+BCE), optimizer
│   ├── evaluate.py                 # Evaluation metrics (Dice, IoU) and prediction visualizer
│   └── utils.py                    # Config parser, random seed setter, device manager
│
├── configs/                        # YAML experiment configuration files
│   ├── config_020.yaml             # Single-patient sanity check & pipeline debugging
│   ├── config_10.yaml              # 10-patient pilot training experiment
│   └── config_full.yaml            # Full Prostate158 dataset training run
│
├── notebooks/                      # Thin experiment controllers and dashboards
│   └── experiment_020.ipynb        # Interactive runner for patient 020 validation
│
├── results/                        # Generated reports, figures, and experiment logs
│   ├── 020/                        # Patient 020 specific runs and outputs
│   ├── 10_patients/                # 10-patient pilot experiment logs
│   ├── full/                       # Full training cohort evaluation metrics
│   ├── 020_dataset_inspection.md   # Verified inspection report for patient 020
│   └── 020_patient_overview.png    # 8-panel verification figure for slice 7
│
├── tests/                          # Automated unit test suite
│   ├── test_dataset.py             # Unit tests for dataset slice indexing & loading
│   └── test_preprocessing.py       # Unit tests for mask binarization & normalization
│
├── dataset/                        # Local raw medical dataset storage (git-ignored)
│   └── prostate158_train/          # [RAW DATASET — UNTOUCHED]
│       ├── train/                  # Patients 020 through 158
│       ├── train.csv               # Official training split
│       └── valid.csv               # Official validation split
│
├── requirements.txt                # Python environment package dependencies
├── README.md                       # Master thesis project documentation
└── .gitignore                      # Git exclusion rules
```

---

## 3. Dataset Location

- **Local Path:** `dataset/prostate158_train/`
- **Contents:** 139 patient cases (`020` through `158`), containing co-registered 3D volumes:
  - `t2.nii.gz` (T2-weighted MRI)
  - `adc.nii.gz` (Apparent Diffusion Coefficient map)
  - `dwi.nii.gz` (High b-value Diffusion-Weighted MRI)
  - `t2_tumor_reader1.nii.gz` (T2 lesion annotations — note label value `3.0`)
  - `adc_tumor_reader1.nii.gz` (ADC lesion annotations — reader 1)
  - `adc_tumor_reader2.nii.gz` (ADC lesion annotations — reader 2)
  - `t2_anatomy_reader1.nii.gz` (Prostate zonal anatomy mask: labels 1 and 2)
- **Partitions:** Standardized train/validation split defined in `train.csv` and `valid.csv`.

---

## 4. Dataset Git Exclusion Policy

> [!IMPORTANT]
> The raw Prostate158 dataset is **strictly excluded from Git tracking** via `.gitignore`.
> Large binary NIfTI files (`*.nii`, `*.nii.gz`) and medical imaging directories (`dataset/`, `data/`) must never be committed to remote version control repositories due to GitHub file size limits and medical data governance principles.

---

## 5. Local Development Environment

- **Platform:** Windows 11 / PowerShell
- **Virtual Environment:** Dedicated local venv at `D:\1.AmadeusKuliah\Skripsi\thesis-project\.venv`
- **Python Executable:** `.\.venv\Scripts\python.exe`
- **Installed Packages:** `nibabel`, `numpy`, `matplotlib`, `pillow` (see `requirements.txt`).

To setup the virtual environment locally:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 6. Development Ecosystem & Tool Roles

### VS Code Role

- Primary development workstation.
- Code authoring, syntax checking, refactoring, and static analysis.
- Local debugging, unit testing (`tests/`), and single-patient sanity checks.
- Managing Git commits, branches, and documentation.

### Google Colab Role

- Cloud GPU acceleration (NVIDIA T4 / A100 / V100) for deep neural network training.
- Executing multi-patient (`config_10.yaml`) and full-cohort (`config_full.yaml`) training loops.
- Running hyperparameter search and training iterations that exceed local hardware capabilities.

### GitHub Role

- Source code version control.
- Tracking experiments, configuration changes, test suites, and documentation.
- Collaborating and syncing code cleanly between the local VS Code workspace and Google Colab.

### Google Drive Role

- Persistent cloud storage for large files not tracked by Git:
  - Compressed and extracted raw Prostate158 dataset archives.
  - Checkpoint model weights (`*.pth`, `*.pt`).
  - High-resolution evaluation logs and training artifacts.

---

## 7. Planned Experiment Progression

The research pipeline is designed in three structured, incremental stages:

```
┌────────────────────────────────────────────────────────┐
│ Stage 1: Patient 020 Sanity Check                      │
│ - Verified dataset geometry (270x270x24, LPS, 0.4x3mm) │
│ - Solved label value 3.0 handling                      │
│ - Verified 2D slice indexing and visualization         │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 2: 10-Patient Pilot Experiment (020 - 029)       │
│ - Validate training loop convergence                   │
│ - Test multi-channel 2D input (T2 + ADC + DWI)         │
│ - Evaluate Dice / BCE loss weighting & slice sampling  │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│ Stage 3: Full Training Dataset (All 139 Patients)      │
│ - Official train.csv / valid.csv splits                │
│ - Full GPU-accelerated training on Google Colab        │
│ - Quantitative benchmark (Volumetric Dice, IoU, HD95)  │
│ - Inter-reader comparison (Reader 1 vs Reader 2)       │
└────────────────────────────────────────────────────────┘
```

---

## 8. Current Project Status

> **Current Status:** **Patient 020 dataset inspection completed; segmentation pipeline not implemented yet.**
>
> - **Completed:**
>   - Comprehensive inspection and statistics verification for Patient 020 (`results/020_dataset_inspection.md`).
>   - Multi-modal slice visualization and reader comparison (`results/020_patient_overview.png`).
>   - Clean, modular repository scaffolding created across `code/`, `configs/`, `notebooks/`, `results/`, and `tests/`.
>   - Scaffolding unit tests implemented and passing.
> - **Next Phase:**
>   - Implementation of the PyTorch 2D U-Net model architecture in `code/model.py`.
>   - Implementation of the training and validation loops in `code/train.py`.
