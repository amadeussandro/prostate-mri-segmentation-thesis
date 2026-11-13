# Prespecification: PSA density from the final model on PI-CAI

Status: locked on 2026-11-12, before any PI-CAI image, label or clinical value was downloaded or
opened. Its SHA-256 (LF line endings) is recorded in `results/picai_psad/prespec_lock.json`, and
the run script refuses to run if this file has changed. Corrections after locking go to a dated
erratum file next to it; this file is not edited.

## Question

In an independent cohort with measured PSA and case-level cancer labels, does PSA density computed
from the final model's whole-gland volume discriminate clinically significant prostate cancer
(csPCa) as well as PSA density computed from the volume in the radiology report?

The study's own PSA analysis used a range of PSA values because Prostate158 has no PSA. This
analysis replaces that limitation with measured values. It is a repetition on a new cohort; no
model, rule or threshold is selected on it.

## Data

- Images: PI-CAI Public Training and Development Dataset v2.0 (Zenodo record 6624726,
  CC BY-NC 4.0), all 5 folds (1500 studies from 1476 patients), axial T2-weighted series only.
  Archive checksums are checked against the Zenodo record.
- Labels and clinical values: `picai_labels` (github.com/DIAGNijmegen/picai_labels), file
  `clinical_information/marksheet.csv`, at a commit hash recorded by the run script. Columns used:
  `patient_id`, `study_id`, `mri_date`, `psa`, `prostate_volume`, `case_csPCa`, `center`.
- AI whole-gland segmentations from the same repository (`anatomical_delineations/whole_gland/AI/
  Bosma22b`), for secondary comparisons only. They are not a reference standard and are not used
  for inference.

## Eligibility

- One study per patient: the earliest `mri_date` (ties: lowest `study_id`).
- Required: a readable axial T2-weighted image, `psa` > 0, and a `case_csPCa` label.
- The primary comparison also requires a reported `prostate_volume` > 0 (complete cases).
- Every exclusion is counted by reason.

## Model and pipeline (frozen)

- The final model as identified by its 3 member weight hashes (results/e1_seed_selection/
  selection.json): a softmax ensemble of 3 seeds with left-right flip averaging.
- The reconstruction pipeline is exactly as in the main analysis.
- Each image is converted from MHA to NIfTI with SimpleITK. The conversion is checked by comparing
  the world coordinates of the 8 corner voxels before and after (tolerance 0.01 mm). A failure stops
  the run until the cause is found and reported.
- Primary: no post-processing (as for the primary outcomes elsewhere in the study).
- Sensitivity: rule R1 (largest connected CG component), which passed its internal check for this
  model.
- Whole-gland volume = CG volume + PZ volume, in mL, from voxel counts and the source spacing.
- A study with an empty model segmentation stays in the analysis. Its PSA density is set to
  +infinity (ranked highest), and such studies are counted.
- A study whose reconstruction fails the header-based geometry check stays in the analysis and is
  reported. The primary endpoint is repeated without such studies.

## Endpoints

Primary
- Difference in the area under the ROC curve for csPCa between PSA density from the model
  (PSA / model volume) and PSA density from the report (PSA / reported volume), model minus report.
- Each AUC is the empirical (Mann-Whitney) estimate. The difference is paired, with a 95% percentile
  CI from a bootstrap stratified by outcome (10,000 resamples, seed 0).
- Noninferiority margin: -0.05 AUC.
  - The model-based PSA density is called noninferior if the lower bound of the 95% CI exceeds
    -0.05. Otherwise noninferiority is not shown.
  - No superiority claim is made unless the lower bound also exceeds 0.

Secondary (descriptive)
1. Volume agreement, model vs report: bias and 95% limits of agreement (Bland-Altman), median
   relative difference, Spearman correlation. The report volume is an ellipsoid estimate, so this
   is agreement, not accuracy.
2. The same, model vs the Bosma22b AI segmentation volume (each volume from its own grid), and the
   whole-gland Dice between them where the Bosma22b mask lies on the T2-weighted grid.
3. Reclassification at PSA density 0.15 ng/mL/mL:
   - the number of patients on different sides of the threshold;
   - sensitivity and specificity for csPCa of each PSA density at that threshold.
4. AUC of PSA density from the Bosma22b volume, for reference.
5. The primary endpoint by center.
6. Reconstruction verification on this cohort: the header-based geometry check on every model
   output, and the round-trip test applied to the Bosma22b masks on their own grid.
7. Sensitivity analyses of the primary endpoint:
   - with rule R1;
   - excluding any study whose T2-weighted image duplicates one in PROSTATEx, NCI-ISBI 2013 or the
     MSD prostate task.
   Duplicates are found as in the main study, by voxel-wise image correlation: images of the same
   array shape after reorientation to RAS with a Pearson r of at least 0.99. This check runs
   before the analysis.

## Reporting

- All numbers are generated by the run script and loaded by the numbers module.
- The manuscript text is guarded so that it cannot state noninferiority unless the CI supports it.
- If the eligibility rules remove more than 20% of studies, that is stated as a limitation.

## Known limitations, stated in advance

- PI-CAI labels combine histopathology with radiological follow-up for some negatives, as defined
  by the dataset.
- Report volumes come from different radiologists and methods.
- Zone volumes cannot be checked because PI-CAI has no zonal reference segmentation.
- Part of PI-CAI comes from the same center as PROSTATEx and the MSD prostate task.
