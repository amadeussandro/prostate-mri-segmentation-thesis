# External validation on PROSTATEx -- frozen E1 model

_Generated: 2026-10-07T09:49:34.810597+00:00 | 204 cases | exp_e1_dice_ce epoch 88_

## 1. Status of this cohort

Inference only -- no training, tuning, threshold or model selection. PROSTATEx is an independent cohort with different scanners, annotators and geometry. Lower scores than the Prostate158 test set are a finding about generalization, not a failure.

Checkpoint SHA-256 `2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2` -- verified identical to the model that produced the RQ1 held-out test result.

## 2. Results

| Metric | CG | PZ |
|---|---|---|

| Dice | 0.8137 +/- 0.0935 | 0.6247 +/- 0.1460 |

| IoU | 0.6954 +/- 0.1207 | 0.4689 +/- 0.1398 |

| HD95 (mm) | 20.2454 +/- 18.5429 | 11.0826 +/- 12.8777 |

| ASD (mm) | 3.1448 +/- 2.2344 | 2.1072 +/- 1.4124 |

| Precision | 0.8193 +/- 0.1153 | 0.8555 +/- 0.1191 |

| Recall | 0.8229 +/- 0.1133 | 0.5076 +/- 0.1483 |


Background reported separately: 0.9961 +/- 0.0017 Dice. Macro Dice is the mean of CG and PZ only.


## 3. Domain shift, measured

- In-plane scale relative to Prostate158 (~0.46875 mm/px): 0.78x to 1.56x. The pipeline does no in-plane resampling, so the prostate is presented to the model at a different pixel size than it was trained on. This is a genuine covariate shift and part of what any score drop measures.
- Cases whose in-plane size exceeds the crop_pad target: **7**. Prostate158 cases are always padded by this transform; these are cropped instead.
- Cases losing annotated foreground to that crop: **0**. No case lost annotated foreground to cropping.


## 4. How to read this

These numbers are not comparable to the Prostate158 test result as a like-for-like contrast: the annotators, scanners, acquisition geometry and zone definitions all differ. They answer a different question -- does a model trained on one cohort retain useful zonal segmentation on an independent one -- and the honest unit of that answer is the gap, reported alongside the shift in Section 3 that partly explains it.
