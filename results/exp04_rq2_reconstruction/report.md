# RQ2 — 3D Reconstruction of 2D U-Net Segmentation (Technical Report)

> **Status of this file.** This is the *seeded* RQ2 report. Its numerical results
> (Section 10) are the **frozen official held-out test results** supplied for the
> epoch-77 checkpoint. The reconstructed NIfTI volumes, per-case CSVs, the
> geometry-validation table, the extra metrics (IoU/HD95/ASD/precision/recall),
> and the figures are produced by running
> `scripts/run_rq2_reconstruction.py` on Colab/Drive (this repository checkout
> contains neither the dataset nor the checkpoint — both are git-ignored and live
> on Google Drive). Running that script **regenerates this report.md in place**
> with the live per-case numbers via the same `write_report()` used here.

Thesis: *"Segmentasi Zona Anatomi pada Citra MRI Pasien Kanker Prostat menggunakan
2D U-NET dan Rekonstruksi 3D."*
RQ2: *"Bagaimana Implementasi Rekonstruksi 3D Citra MRI Pasien Kanker Prostat dari
hasil Segmentasi 2D U-Net?"*

---

## 1. Objective

Implement and validate the reconstruction of the **frozen** 2D U-Net's slice-wise
predictions into the **original 3D MRI voxel space** — preserving the original
geometry (shape, affine, spacing, orientation) — and quantify the reconstructed 3D
segmentation against the untouched ground truth.

**RQ1 (the model and its training) is frozen. No retraining, fine-tuning, or
architecture/hyperparameter/split change was performed, and
`results/exp01_baseline` was not overwritten.** RQ2 consumes only the existing
Experiment-1 predictions.

## 2. Input data

- **Dataset:** Prostate158, official case-level split (119 train / 20 val / **19 test**).
- **RQ2 evaluation split:** the official **held-out test set (19 cases)**.
- **Input volume:** `t2.nii.gz` — T2-weighted, **single channel**.
- **Ground truth:** `t2_anatomy_reader1.nii.gz`, labels `{0, 1, 2}`, read **untouched** from disk (`src/evaluate.py::load_original_space_ground_truth`); GT is never modified or round-tripped.
- **Label mapping (human-verified in 3D Slicer, cases 020/059):** `0 = Background`, `1 = Central Gland (CG)`, `2 = Peripheral Zone (PZ)`. This mapping is **not changed** anywhere in RQ2.

## 3. Existing checkpoint used

- **Path:** `results/exp01_baseline/best_model.pt` (Drive-backed).
- **Best epoch:** **77** (selected on validation slice-level Dice; the test set is never used for model selection — `src/train.py:948,972`).
- **Architecture:** `ProstateUNet2D`, `in_channels = 1` (T2W), `out_channels = 3` (0=bg, 1=CG, 2=PZ). The RQ2 script asserts these before running (`scripts/run_rq2_reconstruction.py`, Step 2).
- The checkpoint, its weights, and the training pipeline are **not modified**. Inference runs under `model.eval()` + `torch.no_grad()`.

## 4. 2D inference procedure

For each test case, every axial slice of the preprocessed T2 volume is passed
through the frozen U-Net; the per-slice prediction is `argmax` over the 3 class
logits. Preprocessing is rebuilt from the **checkpoint's own embedded config**, so
inference sees exactly the pipeline the weights were trained on. Implemented by the
single shared inference+reconstruction path `src/evaluate.py::predict_case_reconstructed`
(the existing `predict_case_original_space` now delegates to it, so evaluation and
RQ2 can never diverge).

## 5. Reconstruction procedure

Predicted 2D slices are reassembled into a `(H, W, D)` volume by **explicit slice
index**, never by list/append order; a missing or duplicated index raises
(`src/reconstruction.py::reconstruct_volume_from_slices`,
`src/geometry.py::assert_complete_slice_indices`). The raw reassembled 2D stack and
its slice order are saved per case to `predictions_2d/pred_<id>.npz`.

## 6. Inverse preprocessing

The preprocessed-space volume is inverse-transformed in **reverse order**
(undo in-plane crop/pad → undo z-resampling → undo RAS orientation
standardization), always with **nearest-neighbour** interpolation to preserve
integer labels (`src/transforms.py::invert_preprocessing_mask`). For the baseline
`crop_pad` pipeline this inverse is pixel-exact.

## 7. Original-space restoration

The reconstructed volume is wrapped in a NIfTI with the **original source affine and
voxel spacing restored exactly** (never a default identity affine); the geometry is
then re-read from the output header and checked against the source
(`src/reconstruction.py::reconstruct_case_to_original_space`; 9-point checklist in
`src/geometry.py`). Each case is saved to
`reconstructions_3d/pred_<id>.nii.gz`.

## 8. Geometry validation

Per-case, machine-readable validation is written to
`reconstruction_validation.csv` by
`src/reconstruction.py::validate_reconstruction_geometry`, which reads the geometry
back from **each output NIfTI's own header** and checks, against the untouched
source MRI: shape match, affine match, voxel-spacing preservation, orientation
(axis codes), depth/slice-order, and that the reconstructed mask contains only
labels `⊂ {0, 1, 2}`. Expected result: **all 19/19 cases pass all checks**
(populated on the Colab run).

## 9. Round-trip fidelity result

Reconstruction-**transform** fidelity is established independently of the model by
`scripts/roundtrip_test.py` (and `tests/test_reconstruction.py`,
`tests/test_rq2_reconstruction.py`): GT mask → preprocess → reconstruct → inverse →
original space yields **Dice = 1.0 exactly** for classes 0/1/2, with shape and
affine exact, on both the identity suite and the baseline-`(442,442)` suite, **with
no model involved**. Latest recorded run (cases 020, 029, 059, 099, 139, both
suites): `shape_ok=True, affine_ok=True, exact=True, Dice 0/1/2 = 1.000000`.

> ⚠️ **This Dice = 1.0 is a property of the reconstruction transform only. It is NOT
> the segmentation accuracy of the trained model, and does not imply the model is
> perfect.** The model's real accuracy is Section 10.

## 10. Real 3D quantitative results

Model predictions reconstructed to **original voxel space**, scored **per case**
against the untouched GT, then aggregated (mean ± SD across the **19** official test
cases; per-case first, never slice-pooled). Official held-out test result from the
epoch-77 checkpoint:

| Class | Dice (mean ± SD) | IoU | HD95 (mm) | ASD (mm) | Precision | Recall |
|---|---|---|---|---|---|---|
| Central Gland (CG, label 1) | **0.8423 ± 0.0466** | *(on run)* | *(on run)* | *(on run)* | *(on run)* | *(on run)* |
| Peripheral Zone (PZ, label 2) | **0.6676 ± 0.1122** | *(on run)* | *(on run)* | *(on run)* | *(on run)* | *(on run)* |

- Evaluation space: `original_voxel`; aggregation: `per_case`; n = 19.
- IoU/HD95/ASD/precision/recall are computed by `src/metrics.py::evaluate_case_volume`
  and filled into `metrics_summary.csv` / `metrics_per_case.csv` / `summary.json` on the run.
- These are **volume-level, original-voxel-space, per-case** metrics — distinct from
  any slice-level training-loop `val_dice`, and distinct from the round-trip Dice in
  Section 9.
- CG > PZ with lower PZ variance-to-mean is expected: PZ is a thin peripheral
  structure, harder to segment than the bulky central gland.

## 11. Qualitative visualization cases

Representative cases are chosen by an **objective, documented rule**
(`select_representative_cases`): rank the 19 test cases by mean foreground
(CG+PZ) Dice, then take the **highest = "good"**, **middle rank = "median"**, and
**lowest = "challenging"** (ties broken by patient id for determinism). No
cherry-picking of only successful cases. For each selected case the run writes, under
`visualizations/patient_<id>/`:
1. **Axial slice panels** (`render_case_slices`): T2 | Ground Truth | Prediction |
   GT-fill vs Pred-outline overlay | error map (correct / FP / miss / wrong-class),
   at objectively selected slices (`select_slice_indices`).
2. **3D label projections** (`render_3d_projections`): T2 maximum-intensity
   projection with GT (top) and Pred (bottom) label silhouettes overlaid along the
   axial, coronal, and sagittal planes — GT vs predicted 3D segmentation side by side.
   (A full surface render can also be produced from the saved NIfTI in 3D Slicer.)

## 12. Limitations

- **2D slice-wise model:** no explicit inter-slice (z) context, so residual axial
  discontinuities are possible (quantifiable via
  `src/metrics.py::inter_slice_dice_inconsistency`).
- **PZ difficulty:** the peripheral zone is thin and scores lower with higher
  variance than the central gland — expected for zonal segmentation.
- **HD95/ASD undefined when a class is absent** from a case (in prediction or GT);
  such cases are NaN-skipped in the aggregate (`aggregate_case_results`).
- **Small cohort (n = 19):** report mean ± SD and bootstrap 95% CI; avoid strong
  significance claims.
- **Reconstruction is exact for the baseline `crop_pad` pipeline**; the (unused here)
  Experiment-2 `resize` arm is deliberately lossy and would not round-trip to Dice = 1.0.

## 13. Reproducibility instructions

```bash
# 1) Reconstruction-transform fidelity (no model; needs the local dataset):
python scripts/roundtrip_test.py

# 2) Torch-free RQ2 unit tests (geometry validation, synthetic round-trip, selection):
python -m pytest tests/test_rq2_reconstruction.py -q

# 3) Full RQ2 run (Colab/GPU; checkpoint + extracted 19-case test archive on Drive):
python scripts/run_rq2_reconstruction.py \
    --config configs/config_baseline.yaml \
    --checkpoint /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt \
    --test-dir  <extracted 19-case test case-root> \
    --output-dir results/exp04_rq2_reconstruction \
    --split test
# (Verify the test archive first: python scripts/audit_test_archive.py --extract-dir <...>)
```

**Frozen-scope statement (explicit):**
- RQ1 model/training was **frozen**; no retraining was performed.
- No architecture, hyperparameter, dataset-split, or test-set change; `results/exp01_baseline` was not overwritten; no prior artifacts deleted.
- RQ2 uses the **existing** 2D U-Net predictions only.
- **Round-trip Dice = 1.0 is a reconstruction-fidelity result, not model segmentation accuracy.**
