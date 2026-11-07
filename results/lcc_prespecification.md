# Pre-specification: largest-connected-component post-processing and the NCI-ISBI 2013 external cohort

Written 2026-11-06, **before** any model inference on NCI-ISBI 2013 and before the rule was applied to the Prostate158 validation split.
`scripts/run_nci_external_validation.py` records this file's SHA-256 in its outputs, so any later edit to this file is visible.

## Why the rule exists

The PROSTATEx diagnostic (`results/external_hd95_tail/`) found that the heavy central-gland HD95 tail comes from small, distant false-positive components. That analysis was post hoc on PROSTATEx, so its HD95 improvement there cannot be reported as a result. This file fixes the rule so it can be tested on data that has not been seen yet.

## The rule (fixed)

- **R1 (primary post-processing rule): central gland only.** In the reconstructed 3D prediction (original voxel space), keep the largest 6-connected component of label 1 (CG). Set every other label-1 voxel to background. Label 2 (PZ) is not changed.
  - 6-connectivity, the same definition as the PROSTATEx diagnostic.
- **R2 (secondary): R1 plus PZ.** Apply R1, then do the same for label 2: keep the largest 6-connected PZ component.
  - R2 is secondary because the peripheral zone is a thin, curved structure that can legitimately split into separate pieces at the apex and base.
- There are no parameters: no size threshold, no distance threshold, no tuning.

## Step 1: internal check (Prostate158 validation split, 20 cases, frozen E1)

The rule is adopted for the external cohort only if, on the validation split, it does not cause harm:
- the mean Dice of the zone it changes drops by no more than 0.005, **and**
- the mean HD95 of that zone does not increase.

If R1 fails this check, it is reported as failed, and the NCI-ISBI analysis reports the unmodified pipeline only. R2 is judged the same way, independently of R1.
The validation split was already used to select the E1 epoch. This check is a safety check, not an estimate of benefit.

## Step 2: NCI-ISBI 2013 (confirmatory external cohort)

- **Model:** frozen E1, the checkpoint whose SHA-256 is 2366e3c6…d5d2. Inference only, no tuning.
- **Cohort:** the 77 cases with both zones (`dataset/external/nci_isbi_2013/nifti/manifest.json`). Exclusions are reported:
  - ProstateDx-01-0055, whose mask and image grids differ;
  - Prostate3T-01-0019, excluded from both zones (see Amendment 1).
- **Labels:** NCI 1 = PZ, 2 = CG, verified anatomically. They are remapped to the model's convention (1 = CG, 2 = PZ).
- **Primary outcome:** Dice and HD95 per zone for the **unmodified** pipeline, as mean (SD) with bootstrap 95% CI, against the single NCI reference.
- **Secondary outcome (only if the rule passed Step 1):** the paired difference between the rule and the unmodified pipeline, for CG HD95 and CG Dice (R1). The R2 difference is reported for PZ. Test: two-sided Wilcoxon signed-rank. The result counts as an improvement only if HD95 falls with P<.05 and the mean Dice change is no lower than −0.005.
- **Pre-specified subgroups (descriptive, no tests):** site, either 1.5 T BMC with endorectal coil or 3 T Radboud with surface coil.
- **Reconstruction:** each prediction passes the geometry check (`validate_reconstruction_geometry`). Any case that fails is reported, not dropped silently.

## What will not be done

- No change to the rule after seeing NCI-ISBI.
- No re-selection of the checkpoint, no threshold search, and no test-time augmentation.
- PROSTATEx is not re-analysed with the rule as a result. It stays a secondary cohort, where the rule was motivated.

## Amendment 1 (2026-11-06, written while the validation check was running and before any NCI-ISBI inference)

Prostate3T-01-0019 carries a single foreground label covering the whole gland (about 106 mL). The same image is in MSD Task05 as prostate_18, also with a single whole-gland label. It is not a "PZ-only" case, as first written above. After label remapping, it would have been scored as central gland, which is wrong. It is excluded from both zones, so the cohort is 77 cases.

This amendment changes only the exclusion list. It does not touch the rule or the decision criteria, and it was made before any model output on NCI-ISBI existed. The validation-split decision is re-recorded against this version of the file. Its per-case metrics are unchanged.
