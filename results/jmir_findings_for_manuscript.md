# New findings for the JMIR manuscript (ready to paste)

Generated 2026-11-06 from existing results, CPU only, no retraining.
Scripts: `scripts/jmir_reader2_benchmark.py`, `scripts/jmir_stats_rework.py`.
Raw outputs: `results/jmir_reader2/`, `results/jmir_stats/`.
All numbers below are already in JMIR house style (sample SD, P without leading zero).

## 1. Inter-reader benchmark (19 held-out cases, original voxel space)

| Zone | Metric | Reader 1 vs reader 2 | Model vs reader 1 | Model vs reader 2 | Model − inter-reader (95% CI); P |
|---|---|---|---|---|---|
| CG | Dice, mean (SD) | 0.873 (0.033) | 0.844 (0.051) | 0.845 (0.049) | −0.028 (−0.051 to −0.006); P=.006 |
| CG | HD95, mm, mean (SD) | 6.7 (3.2) | 6.4 (3.2) | 5.8 (2.3) | −0.3 (−1.6 to 1.0); P=.52 |
| PZ | Dice, mean (SD) | 0.747 (0.068) | 0.685 (0.132) | 0.640 (0.133) | −0.062 (−0.113 to −0.011); P=.02 |
| PZ | HD95, mm, mean (SD) | 6.7 (4.6) | 6.5 (4.2) | 8.8 (4.7) | −0.2 (−2.3 to 1.9); P=.80 |

P values: Wilcoxon signed-rank, paired by case.

Volume agreement against reader 1 (bias, 95% limits of agreement):

| Zone | Reader 2 − reader 1 | Model − reader 1 |
|---|---|---|
| CG | −1.0 mL (−5.6 to 3.6) | −1.5 mL (−10.8 to 7.8) |
| PZ | +2.7 mL (−2.5 to 8.0); +19.3% | −5.1 mL (−16.5 to 6.3); −25.6% |
| Whole gland | +1.7 mL (−4.4 to 7.9) | −6.6 mL (−20.2 to 7.0) |

Note: −25.6% is the mean of per-case percentages. The manuscript's −28.5% is the ratio of mean bias to mean reference volume. State which one is used.

Suggested text (Results):
> On the same 19 cases, the 2 expert readers agreed with a Dice of 0.873 (SD 0.033) for the central gland and 0.747 (SD 0.068) for the peripheral zone. The model's Dice against reader 1 was lower by 0.028 (95% CI 0.006-0.051; P=.006) and 0.062 (95% CI 0.011-0.113; P=.02), corresponding to 96.7% and 91.7% of inter-reader agreement. Boundary distance did not differ from inter-reader variability: the model's HD95 was 6.4 mm (central gland) and 6.5 mm (peripheral zone), against 6.7 mm for both zones between readers (P=.52 and P=.80).

Interpretation for the Discussion: the model's boundary error is within the spread between experts, while its overlap is slightly lower. The peripheral-zone volume shortfall (−5.1 mL) is larger than, and opposite in sign to, the disagreement between readers (+2.7 mL), so it is a model property rather than label noise.

## 2. Statistics rework

Held-out test, sample SD (replace population SD everywhere):
- CG Dice 0.844 (SD 0.051), 95% CI 0.821-0.864
- PZ Dice 0.685 (SD 0.132), 95% CI 0.621-0.738
- Full table: `results/jmir_stats/test_table_sample_sd.csv`

Equivalence (TOST, margin ±0.02 Dice, paired t; 90% CI must lie inside the margin):

| Comparison | Split | Macro | CG | PZ |
|---|---|---|---|---|
| Augmentation (E2) − baseline | validation | equivalent (P=.01) | equivalent (P=.009) | not shown (P=.06) |
| Class weighting (E3) − E1 | validation | equivalent (P<.001) | equivalent (P<.001) | equivalent (P=.002) |
| Soft Dice (E1) − baseline | validation | equivalent (P=.003) | equivalent (P<.001) | not equivalent (+0.015, P=.19) |
| Soft Dice (E1) − baseline | test | inconclusive (+0.010, 90% CI −0.003 to 0.022; P=.08) | equivalent (P<.001) | inconclusive (+0.018, 90% CI −0.004 to 0.039) |

Superiority on test, E1 vs baseline (Wilcoxon):
- Macro (pre-specified primary): +0.010, better in 13/19, W=47, P=.055 — not significant
- CG (post hoc): +0.002, W=75, P=.44; Holm-adjusted P=.44
- PZ (post hoc): +0.018, better in 16/19, W=37, P=.02; Holm-adjusted P=.04 (2 zone comparisons)

Suggested text (Results):
> Class weighting was equivalent to the soft Dice configuration within ±0.02 Dice in every zone (all P≤.002), and augmentation was equivalent to the baseline for macro and central-gland Dice (P=.01 and P=.009). On the held-out set, adding a soft Dice term left central-gland Dice unchanged (equivalent within ±0.02; P<.001) and improved peripheral-zone Dice by 0.018 (16 of 19 cases; Holm-adjusted P=.04, post hoc). The pre-specified macro comparison was not significant (P=.055) and was not equivalent either (90% CI −0.003 to 0.022), so the overall effect remains inconclusive.

Caveats to keep in the manuscript:
- The validation split was used for selection, so validation tests are descriptive.
- The ±0.02 margin must be fixed and justified in the Methods before results are reported (proposed: below the inter-reader SD of CG Dice, 0.033). Agree it with the supervisor.
- The zone-level tests are post hoc; label them as such even though PZ survives Holm.

## 3. Model-free reconstruction ablation (4 datasets, 3 orientation conventions)

Script: `scripts/run_modelfree_ablation.py` (NCI-ISBI prepared by `scripts/prepare_nci_isbi.py`).
Outputs: `results/modelfree_ablation/{summary_all.csv, per_case_all.csv, detection_matrix.csv, gate.json}`.
No model is involved: the reference mask is pushed through the production forward transform (reorient, pad/crop to 442 × 442, slice) and then reconstructed, correctly (V0) or with one deliberate fault.

Cohorts: Prostate158 (158 cases, LPS), PROSTATEx (204, LAS), MSD Task05 (30 of 32; 2 lack a TZ label, RAS), NCI-ISBI 2013 (78 of 80; 1 grid mismatch, 1 without CG, LPS). Total 470 cases.

Gate: the correct pipeline (V0) returned the reference voxel-for-voxel in 470 of 470 cases and passed the geometry check in all of them.

### Fault detection

| Fault | Prostate158 (LPS) | PROSTATEx (LAS) | MSD (RAS) | NCI-ISBI (LPS) | Caught by |
|---|---|---|---|---|---|
| No reorientation (V1) | Dice CG 0.47 / PZ 0.16 | 0.82 / 0.65 | **no damage** (already RAS) | 0.30 / 0.22 | Round-trip 100%; anatomy check 99-100% on LPS, 0% on LAS |
| Identity affine (V3) | volume error +140% | +37.9% | −30.5% | +49.4% | Geometry 100% |
| Spacing-only affine (V6) | position shift 119 mm | 163 mm | 158 mm | 138 mm | Geometry 100% |
| Shuffled slice order (V4b) | Dice 0.43 / 0.19 | 0.52 / 0.33 | 0.55 / 0.27 | 0.39 / 0.23 | Round-trip 100%; slice continuity 87-100% |
| Centre crop instead of stored crop (V2) | no damage | no damage | no damage | no damage | — |
| Append order (V4, sorted slices) | no damage | no damage | no damage | no damage | — |

Across all faulted cases that were damaged (n=2320), the geometry check plus the round-trip test flagged 2320 (100%), and flagged 0 of 970 faulted-but-undamaged cases. The two reference-free anatomical checks (PZ posterior to CG; slice-to-slice continuity) flagged 1378 of 2320 (59%). False alarms on the correct pipeline: 0 of 470 for the PZ-posterior check and 9 of 470 (1.9%) for slice continuity.

Suggested text (Results):
> Applied to 470 reference segmentations from 4 public cohorts in 3 orientation conventions, the correct inverse pipeline restored every mask exactly. Each of 4 damaging faults was detected in every damaged case by the geometry check or the round-trip test, with no false flags. Omitting reorientation was silent on the RAS-stored cohort, cost only a left-right flip on the LAS cohort (Dice 0.82 and 0.65), and destroyed overlap on the 2 LPS cohorts (central gland Dice 0.30-0.47). An affine that carried the correct spacing but no origin or direction preserved every volume exactly while displacing the gland by 119-163 mm.

Interpretation for the Discussion:
- The same bug has a different size, or no size at all, depending on how the data were stored. A pipeline developed and tested on RAS data (such as MSD) would pass, then fail on LPS data. This is the strongest argument for a dataset-independent round-trip test.
- Volume-based QC cannot catch an origin or direction error (V6: volume error 0%).
- The anatomy check without a reference is a useful screen at inference time (no ground truth), but it misses LAS-only flips and correct-volume misplacements, so it does not replace the round-trip test.

### Real-world finding: half-voxel origin offset in NCI-ISBI 2013

All 40 Radboud (Prostate3T) masks share the size, spacing and direction of their image series, but their NRRD origin is offset by exactly half a voxel on every axis. The labels are aligned with the image voxels, so the correct handling is a voxel copy. A standard physical-space resample, which is what most pipelines do when two files' origins differ, shifts the labels and reduces agreement with the original annotation to Dice 0.81 for the peripheral zone (minimum 0.52) and 0.90 for the central gland (minimum 0.71). One further case (ProstateDx-01-0055) has a mask drawn on a different series (23 vs 34 slices) and was excluded.

Suggested text:
> During preparation of the NCI-ISBI cohort, all 40 masks from one site carried a half-voxel origin offset relative to their image series. Resampling them in physical space, the default when origins differ, would have reduced their Dice against the original annotation to 0.81 (peripheral zone) and 0.90 (central gland) before any model was applied.

Caveat: the half-voxel finding should be described as a header convention mismatch, not as an annotation error. Cite the dataset DOI and say how it was handled.

## 4. Second external cohort: NCI-ISBI 2013 (confirmatory, pre-specified)

Files:
- Pre-specification: `results/lcc_prespecification.md`. Its SHA-256 is recorded in every output. The erratum, which corrects a cohort count only, is in `results/lcc_prespecification_erratum.md`.
- Script: `scripts/run_nci_external_validation.py`.
- Outputs: `results/external_nci_isbi_e1/{valid_prostate158/decision.json, nci/summary.json, nci/per_case.csv}`.

Model: frozen E1 (SHA-256 2366e3c6…d5d2). Inference only, CPU. The CPU run reproduces the training-time validation scores exactly (CG Dice 0.8662, PZ 0.7527).

### Step 1: internal safety check (Prostate158 validation, 20 cases)

| Rule | Zone | Dice change | HD95 change, mm | Passed |
|---|---|---|---|---|
| R1 (largest CG component) | CG | +0.001 | −1.2 | Yes |
| R2 (R1 + largest PZ component) | PZ | +0.0004 | −0.4 | Yes |

### Step 2: NCI-ISBI 2013

Cohort: 78 cases (39 1.5 T BMC with endorectal coil; 39 3 T Radboud). 2 cases are excluded and reported. The reconstruction geometry check passed in 78 of 78.

| Zone | Metric | All (n=78) | 1.5 T endorectal (n=39) | 3 T (n=39) | PROSTATEx (n=204) |
|---|---|---|---|---|---|
| CG | Dice, mean (SD) | 0.777 (0.122) | 0.741 (0.129) | 0.812 (0.105) | 0.814 |
| CG | HD95, mm, mean (SD); median | 36.5 (24.9); 42.5 | 46.8; 53.5 | 26.2; 19.7 | 20.2 |
| PZ | Dice, mean (SD) | 0.588 (0.154) | 0.563 (0.127) | 0.613 (0.175) | 0.625 |
| PZ | HD95, mm, mean (SD); median | 13.6 (13.2); 9.0 | 13.5 | 13.7 | 11.1 |

95% bootstrap CIs (all 78 cases): CG Dice 0.748–0.802; PZ Dice 0.553–0.621; CG HD95 31.0–42.0 mm.

Pre-specified secondary analysis: the LCC rule, unmodified vs rule, paired Wilcoxon, n=78.

| Rule | Zone | Dice, before → after | HD95 mm, before → after | P (HD95) | Criterion met |
|---|---|---|---|---|---|
| R1 | CG | 0.777 → 0.805 (+0.028; P<.001) | 36.5 → 9.1 (median 42.5 → 7.2) | <.001 | Yes |
| R2 | PZ | 0.588 → 0.590 (+0.002) | 13.6 → 9.7 | .04 | Yes |

- CG cases with HD95 > 30 mm: 43 of 78 before the rule, 1 of 78 after.
- 92% of predicted CGs had more than one connected component.
- After R1, CG HD95 by site: 1.5 T 10.9 mm, 3 T 7.2 mm.

Sensitivity check: excluding Prostate3T-03-0001, which is also ProstateX-0082, changes no value by more than 0.4 mm or 0.001 Dice.

### Model vs 2 independent annotation sets (18 NCI 3T images that are also in MSD Task05)

| Comparison | CG Dice | PZ Dice |
|---|---|---|
| NCI annotation vs MSD annotation | 0.888 | 0.776 |
| Model vs NCI annotation | 0.800 | 0.588 |
| Model vs MSD annotation | 0.773 | 0.595 |

The model is below inter-annotation agreement in both zones on this external data: CG −0.088 and PZ −0.188, both P<.001. On Prostate158 the gap was −0.028 and −0.062. The gap therefore widens under domain shift, most of all in PZ.

Suggested text (Results):
> On a second external cohort that had not been used before (NCI-ISBI 2013, 78 cases from 2 sites), the frozen model reached a central-gland Dice of 0.777 (SD 0.122) and a peripheral-zone Dice of 0.588 (SD 0.154), with all 78 reconstructions passing the geometry check. Performance was lower at the 1.5 T site imaged with an endorectal coil (central gland 0.741) than at the 3 T site (0.812). The central-gland boundary error again showed a heavy tail (HD95 36.5 mm; 43 of 78 cases above 30 mm). The largest-connected-component rule, fixed in advance and checked for harm on the internal validation split, reduced central-gland HD95 to 9.1 mm (P<.001) and increased Dice by 0.028 (P<.001). After the rule, only 1 case remained above 30 mm.

Interpretation for the Discussion:
- This converts the PROSTATEx post hoc diagnosis into a confirmed, pre-specified result on independent data. It also directly answers the manuscript's existing statement that LCC "would require pre-specifying the step and validating it on the internal validation split first".
- The primary result remains the unmodified pipeline. LCC is reported as the pre-specified secondary analysis.
- The endorectal-coil site is the hardest domain, which is plausible: the coil distorts the gland and the coil signal creates bright structures near the PZ.
- The overlap between cohorts must be stated:
  - 19 NCI 3T images are MSD Task05 images;
  - 1 NCI case is a PROSTATEx case;
  - no dataset overlaps with the Prostate158 training data.
