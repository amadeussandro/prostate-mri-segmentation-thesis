# Patient 020 Dataset Inspection

## 1. Dataset location

- **Workspace Root:** `D:\1.AmadeusKuliah\Skripsi\thesis-project`
- **Patient 020 Directory:** `D:\1.AmadeusKuliah\Skripsi\thesis-project\dataset\prostate158_train\train\020`
- **Split Reference:** Patient `020` is indexed in `dataset\prostate158_train\valid.csv` (Row 2, `ID=20`).
- **Raw Dataset Modification Status:** The raw dataset directory and all original files were left completely untouched.

---

## 2. Files found

A total of **7 files** (all `.nii.gz`) were found in the patient 020 directory:

| Filename                    | File Size (Bytes) | Header Data Type | Modality / Role                           |
| :-------------------------- | :---------------- | :--------------- | :---------------------------------------- |
| `adc.nii.gz`                | 5,677,945         | `float32`        | Apparent Diffusion Coefficient map        |
| `adc_tumor_reader1.nii.gz`  | 7,183             | `float32`        | Tumor segmentation mask (Reader 1 on ADC) |
| `adc_tumor_reader2.nii.gz`  | 13,969            | `float64`        | Tumor segmentation mask (Reader 2 on ADC) |
| `dwi.nii.gz`                | 5,661,656         | `float32`        | Diffusion-Weighted Imaging                |
| `t2.nii.gz`                 | 2,781,417         | `float32`        | T2-weighted MRI                           |
| `t2_anatomy_reader1.nii.gz` | 10,184            | `uint16`         | Multi-class prostate anatomy segmentation |
| `t2_tumor_reader1.nii.gz`   | 7,095             | `float32`        | Tumor segmentation mask (Reader 1 on T2)  |

_(Note: In `valid.csv`, column `t2_anatomy_reader2` is empty for patient 020, matching the absence of this file on disk.)_

---

## 3. NIfTI dimensions

All 7 files share identical spatial dimensions:

- **3D Matrix Shape:** `(270, 270, 24)`
- **Number of Dimensions:** `3`
- **Total Voxels per Volume:** `270 × 270 × 24 = 1,749,600` voxels
- **Plane Dimensions:**
  - In-plane ($X \times Y$): $270 \times 270$
  - Through-plane / Slice count ($Z$): $24$ axial slices

---

## 4. Voxel spacing

All 7 files have identical voxel spacing recorded in the NIfTI header:

- **Zooms:** `(0.4017857015132904, 0.4017857015132904, 3.0)` mm
- **In-plane Resolution ($X, Y$):** $\approx 0.4018\text{ mm} \times 0.4018\text{ mm}$
- **Through-plane Spacing ($Z$):** $3.0\text{ mm}$ (slice thickness / step)
- **Anisotropy Ratio:** Through-plane resolution is $\approx 7.47\times$ coarser than the in-plane resolution.

---

## 5. Orientation and affine

- **Orientation Codes:** `('L', 'P', 'S')` (Left, Posterior, Superior — LPS coordinate system)
- **Affine Transformation Matrix (4×4):**
  Identical across all 7 volumes:
  ```
  [[-4.01785702e-01,  1.96754462e-12,  0.00000000e+00,  5.42410698e+01],
   [-1.96754462e-12, -4.01785702e-01,  0.00000000e+00,  1.86478500e+01],
   [ 0.00000000e+00,  0.00000000e+00,  3.00000000e+00, -5.68644142e+00],
   [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
  ```
- **Registration Status:** Verified that all three MRI sequences (`t2`, `adc`, `dwi`) and all four mask volumes are already mapped to a shared physical space and coordinate grid.

---

## 6. Intensity statistics

| Filename                    | Loaded Type | Min | Max       | Mean     | Non-Zero Voxels | % Non-Zero |
| :-------------------------- | :---------- | :-- | :-------- | :------- | :-------------- | :--------- |
| `adc.nii.gz`                | `float64`   | 0.0 | 2906.2351 | 761.1310 | 1,601,518       | 91.536%    |
| `dwi.nii.gz`                | `float64`   | 0.0 | 131.4472  | 43.0831  | 1,676,700       | 95.833%    |
| `t2.nii.gz`                 | `float64`   | 0.0 | 1162.0000 | 227.1079 | 1,749,095       | 99.971%    |
| `t2_anatomy_reader1.nii.gz` | `float64`   | 0.0 | 2.0000    | 0.0833   | 102,541         | 5.861%     |
| `t2_tumor_reader1.nii.gz`   | `float64`   | 0.0 | 3.0000    | 0.0008   | 485             | 0.028%     |
| `adc_tumor_reader1.nii.gz`  | `float64`   | 0.0 | 1.0000    | 0.0009   | 1,569           | 0.090%     |
| `adc_tumor_reader2.nii.gz`  | `float64`   | 0.0 | 1.0000    | 0.0004   | 703             | 0.040%     |

---

## 7. Tumor / segmentation mask investigation

Each file was evaluated on its array values to verify whether it acts as a segmentation mask or continuous image:

1. **`adc_tumor_reader1.nii.gz` (VERIFIED MASK):**
   - Unique values: `[0.0, 1.0]`
   - Distribution: `{0.0: 1,748,031 voxels, 1.0: 1,569 voxels}`
   - Active slice range: Slices `6, 7, 8, 9` (counts: slice 6: 380, slice 7: 415, slice 8: 396, slice 9: 378).
   - Classification: Binary lesion segmentation mask.

2. **`adc_tumor_reader2.nii.gz` (VERIFIED MASK):**
   - Unique values: `[0.0, 1.0]`
   - Distribution: `{0.0: 1,748,897 voxels, 1.0: 703 voxels}`
   - Active slice range: Slices `6, 7, 8` (counts: slice 6: 234, slice 7: 383, slice 8: 86).
   - Classification: Binary lesion segmentation mask from a second reader. Notably smaller volume (703 voxels vs 1,569 voxels).

3. **`t2_tumor_reader1.nii.gz` (VERIFIED MASK WITH SPECIAL LABEL):**
   - Unique values: `[0.0, 3.0]` _(Critical verification: Lesion value is `3.0`, NOT `1.0`)_
   - Distribution: `{0.0: 1,749,115 voxels, 3.0: 485 voxels}`
   - Active slice range: Slices `5, 6, 7` (counts: slice 5: 64, slice 6: 212, slice 7: 209).
   - Classification: Discrete lesion segmentation mask. The label `3.0` indicates lesion severity / PI-RADS 3 or a specific class identifier in the Prostate158 annotation protocol.

4. **`t2_anatomy_reader1.nii.gz` (VERIFIED MULTI-CLASS MASK):**
   - Unique values: `[0.0, 1.0, 2.0]`
   - Distribution: `{0.0: 1,647,059, 1.0: 59,304, 2.0: 43,237}`
   - Active slice range: Slices `1` through `19` (spans 19 of the 24 slices).
   - Classification: Multi-class anatomical segmentation mask for the prostate gland (peripheral zone vs transition zone).

---

## 8. Important findings

1. **Identical Coordinate Grid:** All volumes and masks share the exact same matrix size `(270, 270, 24)` and identical affine matrix. No reslicing or rigid registration is needed between T2, ADC, and DWI for this patient.
2. **Label Value Discrepancy in `t2_tumor_reader1`:** The tumor mask voxels have value `3.0`. Any code assuming binary labels are strictly `{0, 1}` would fail if testing `mask == 1`.
3. **Inter-Observer & Inter-Modality Discrepancy:**
   - Reader 1 (ADC): 1,569 voxels across 4 slices.
   - Reader 2 (ADC): 703 voxels across 3 slices (44.8% of Reader 1 volume).
   - Reader 1 (T2): 485 voxels across 3 slices.
   - Slices 6 and 7 represent the core consensus lesion region across all readers and modalities.
4. **Severe Class Imbalance:** Tumor volume represents only `0.028%` to `0.090%` of the total 3D volume.
5. **High Anisotropy:** In-plane resolution is $0.402\text{ mm}$, whereas through-plane slice thickness is $3.0\text{ mm}$ ($\approx 7.5\times$ thicker).

---

## 9. Potential preprocessing considerations

For designing a **2D prostate MRI segmentation pipeline**:

1. **Binarization / Label Harmonization:**
   - Apply `(mask > 0).astype(np.uint8)` or explicit mapping `{0: 0, 3: 1}` to prevent silent bugs caused by the `3.0` label in T2 tumor masks.
2. **Slice Filtering / Sampling:**
   - 20 out of 24 slices contain no tumor.
   - An anatomy-guided slice sampler should select slices within the prostate volume (slices 1–19 from `t2_anatomy_reader1`) and maintain a balanced ratio between positive tumor slices (slices 5–9) and negative prostate slices.
3. **Multi-Channel 2D Input Stacking:**
   - Because T2, ADC, and DWI have identical matrix dimensions `(270, 270)`, slices can be directly stacked along the channel axis into a 3-channel input tensor: `[C=3, H=270, W=270]`.
4. **Intensity Normalization:**
   - Dynamic ranges vary substantially: T2 ($\approx [0, 1162]$), ADC ($\approx [0, 2906]$), DWI ($\approx [0, 131]$).
   - Preprocessing should apply channel-wise normalization: 1st/99th percentile clipping to remove extreme scanner artifacts, followed by z-score standardization or $[0, 1]$ min-max scaling per sequence.
5. **Spatial Cropping / Bounding Box:**
   - The prostate gland occupies roughly $5.86\%$ of the field of view. Center-cropping or anatomy-mask-based cropping to e.g. $160 \times 160$ or $128 \times 128$ will reduce memory footprint, eliminate irrelevant peripheral anatomy, and increase tumor-to-background ratio.
6. **Ground Truth Consensus Policy:**
   - Decide how to handle Reader 1 vs Reader 2 annotations: use Reader 1 as primary target, take majority voting / intersection / union, or evaluate inter-observer agreement.

---

## 10. Uncertainties

1. **Clinical meaning of value `3.0` in `t2_tumor_reader1`:**
   - _Verified from file:_ The voxel values are exactly `0.0` and `3.0`.
   - _Uncertainty:_ Whether `3.0` corresponds to a PI-RADS 3 lesion assessment or an internal class enumeration index cannot be determined purely from the NIfTI header; it requires referencing the dataset's clinical annotation protocol.
2. **Anatomical zone labeling in `t2_anatomy_reader1`:**
   - _Verified from file:_ Label values are `1.0` (59,304 voxels) and `2.0` (43,237 voxels).
   - _Uncertainty:_ The mapping of `1` vs `2` to Peripheral Zone (PZ) vs Transition Zone (TZ) is inferred from prostate anatomy morphology, but the exact string mapping is not stored in the NIfTI header.
3. **Acquisition parameters:**
   - Scanner manufacturer, magnetic field strength (1.5T vs 3.0T), repetition/echo times (TR/TE), and specific DWI b-values are omitted from standard NIfTI headers and would require raw DICOM metadata.
