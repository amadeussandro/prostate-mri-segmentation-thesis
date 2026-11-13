# External validation on PROSTATEx -- frozen E1 model

_Generated: 2026-11-11T09:42:06.766624+00:00 | 204 cases | exp_e1_dice_ce epoch [88, 62, 98]_

## 1. Status of this cohort

Inference only -- no training, tuning, threshold or model selection. PROSTATEx is an independent cohort with different scanners, annotators and geometry. Lower scores than the Prostate158 test set are a finding about generalization, not a failure.

Checkpoint SHA-256 `39d5e14cb3d2b6a8c5e22ed28a8bcf2abe4d732c03ae86672ac06a7aaccfd6f1` -- verified identical to the model that produced the RQ1 held-out test result.

## 2. Results

| Metric | CG | PZ |
|---|---|---|

| Dice | 0.8271 +/- 0.1007 | 0.6457 +/- 0.1589 |

| IoU | 0.7158 +/- 0.1260 | 0.4945 +/- 0.1526 |

| HD95 (mm) | 11.7903 +/- 13.3968 | 8.0527 +/- 4.9352 |

| ASD (mm) | 2.2516 +/- 1.6459 | 1.8456 +/- 1.3905 |

| Precision | 0.8541 +/- 0.1012 | 0.8775 +/- 0.1136 |

| Recall | 0.8174 +/- 0.1325 | 0.5308 +/- 0.1617 |


Background reported separately: 0.9965 +/- 0.0017 Dice. Macro Dice is the mean of CG and PZ only.


## 3. Domain shift, measured

- In-plane scale relative to Prostate158 (~0.46875 mm/px): 0.78x to 1.56x. The pipeline does no in-plane resampling, so the prostate is presented to the model at a different pixel size than it was trained on. This is a genuine covariate shift and part of what any score drop measures.
- Cases whose in-plane size exceeds the crop_pad target: **7**. Prostate158 cases are always padded by this transform; these are cropped instead.
- Cases losing annotated foreground to that crop: **0**. No case lost annotated foreground to cropping.


## 4. How to read this

These numbers are not comparable to the Prostate158 test result as a like-for-like contrast: the annotators, scanners, acquisition geometry and zone definitions all differ. They answer a different question -- does a model trained on one cohort retain useful zonal segmentation on an independent one -- and the honest unit of that answer is the gap, reported alongside the shift in Section 3 that partly explains it.
