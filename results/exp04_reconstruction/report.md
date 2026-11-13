# RQ2 -- 3D Reconstruction of 2D U-Net Segmentation (Technical Report)

_Generated: 2026-09-19T08:11:12.043497Z | split: test (19 cases) | checkpoint epoch: 77_

## 1. Objective

Answer RQ2: implement and validate the reconstruction of the frozen 2D U-Net's slice-wise predictions into the original 3D MRI voxel space, preserving the original geometry, and quantify the reconstructed 3D segmentation against the untouched ground truth. **RQ1 (the model and its training) is frozen; no retraining was performed.**

## 2. Input data

- Dataset: Prostate158, official held-out **test** split (19 cases: `001, 002, 003, 004, 005, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 016, 017, 018, 019`).
- Input volume: `t2.nii.gz` (T2W, single channel).
- Ground truth: `t2_anatomy_reader1.nii.gz` (labels {0,1,2}), read untouched from disk.
- Data root (this split): `/content/drive/MyDrive/THESIS_PROSTATE158/dataset/test/extracted/prostate158_test/test`.

## 3. Existing checkpoint used

- Path: `/content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt`
- Best epoch: **77** (training-loop slice-level best metric: 0.6510601027791716).
- Architecture: `ProstateUNet2D`, in_channels=1, out_channels=3 (0=background, 1=CG, 2=PZ).
- The checkpoint, its weights, and the training pipeline were not modified.

## 4. 2D inference procedure

For each case, every axial slice of the preprocessed T2 volume is passed through the frozen U-Net under `model.eval()` + `torch.no_grad()`; the per-slice prediction is `argmax` over the 3 class logits. Preprocessing is rebuilt from the checkpoint's own embedded config, so inference sees exactly the pipeline the weights were trained on (`src.evaluate.predict_case_reconstructed`).

## 5. Reconstruction procedure

Predicted 2D slices are reassembled into a (H, W, D) volume by **explicit slice index** (never append order); missing/duplicate indices raise (`src.reconstruction.reconstruct_volume_from_slices`).

## 6. Inverse preprocessing

The preprocessed-space volume is inverse-transformed in reverse order (undo crop/pad -> undo z-resample -> undo RAS orientation), always with nearest-neighbour interpolation to preserve integer labels (`src.transforms.invert_preprocessing_mask`).

## 7. Original-space restoration

The reconstructed volume is wrapped in a NIfTI with the **original source affine and voxel spacing restored exactly** (never a default identity affine); the geometry is then re-read from the output header and checked against the source (`src.reconstruction.reconstruct_case_to_original_space`, 9-point checklist in `src.geometry`). Saved to `reconstructions_3d/`.

## 8. Geometry validation

Per-case machine-readable validation (`reconstruction_validation.csv`): **19/19** cases pass all checks (shape, affine, spacing, orientation, depth/slice-order, labels subset of {0,1,2}). `validate_reconstruction_geometry` reads geometry back from each output NIfTI's own header.

## 9. Round-trip fidelity result

Reconstruction-transform fidelity is established by `scripts/roundtrip_test.py` (and `tests/test_reconstruction.py` / `tests/test_rq2_reconstruction.py`): GT mask -> preprocess -> reconstruct -> inverse -> original space yields **Dice = 1.0 exactly** for classes 0/1/2 with shape and affine exact, on both the identity and baseline-(442,442) suites, **with no model involved**.

> **This Dice = 1.0 is a property of the reconstruction transform only. It is NOT the segmentation accuracy of the model. The model's real accuracy is in Section 10.**

## 10. Real 3D quantitative results

Model predictions reconstructed to original voxel space, scored per case against the untouched GT, then aggregated (mean +/- SD across 19 cases; per-case first, never slice-pooled):


| Class | Dice | IoU | HD95 (mm) | ASD (mm) | Precision | Recall |

|---|---|---|---|---|---|---|

| CG_central_gland | 0.8423 +/- 0.0466 | 0.7303 +/- 0.0683 | 6.7329 +/- 3.5923 | 1.5781 +/- 0.5851 | 0.8524 +/- 0.0551 | 0.8376 +/- 0.0748 |

| PZ_peripheral_zone | 0.6676 +/- 0.1122 | 0.5111 +/- 0.1203 | 6.8748 +/- 3.8013 | 1.4236 +/- 0.6365 | 0.7907 +/- 0.0899 | 0.5986 +/- 0.1467 |


These are volume-level, original-voxel-space, per-case metrics -- distinct from any slice-level training-loop `val_dice`.

## 11. Qualitative visualization cases

Representative cases selected by an objective rule (rank by mean foreground Dice; pick highest = good, middle = median, lowest = challenging):


- **challenging**: patient `016` (mean foreground Dice 0.5874) -- axial panels + 3D projections under `visualizations/patient_016/`.

- **median**: patient `007` (mean foreground Dice 0.7812) -- axial panels + 3D projections under `visualizations/patient_007/`.

- **good**: patient `014` (mean foreground Dice 0.8402) -- axial panels + 3D projections under `visualizations/patient_014/`.


## 12. Limitations

- 2D slice-wise model: no explicit inter-slice (z) context; residual axial discontinuities are possible (see `src.metrics.inter_slice_dice_inconsistency`).
- PZ is a thin peripheral structure and scores lower / with higher variance than CG -- expected for zonal segmentation.
- HD95/ASD are undefined for a class absent from a case (prediction or GT) and are excluded from that case's aggregate (NaN-skipping).
- n = 19 (official test): report mean +/- SD and bootstrap CI, not strong significance claims.

## 13. Reproducibility instructions

```bash
# 1) Reconstruction-transform fidelity (no model, needs the local dataset):
python scripts/roundtrip_test.py

# 2) Torch-free RQ2 unit tests (geometry validation, synthetic round-trip):
python -m pytest tests/test_rq2_reconstruction.py -q

# 3) Full RQ2 run (Colab/GPU, checkpoint + test archive on Drive):
python scripts/run_rq2_reconstruction.py \
    --config configs/config_baseline.yaml \
    --checkpoint /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt \
    --test-dir <extracted 19-case test root> \
    --output-dir results/exp04_rq2_reconstruction \
    --split test
```

**Frozen-scope statement:** RQ1 model/training was frozen; no retraining, no architecture/hyperparameter/split change, and `results/exp01_baseline` was not overwritten. RQ2 uses the existing 2D U-Net predictions only. Round-trip Dice = 1.0 is a reconstruction-fidelity result, not model segmentation accuracy.
