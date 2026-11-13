# RQ2 -- anatomical zone volumes (mL)

_Generated: 2026-10-07T09:21:07.120930+00:00 | 19 cases | mode: **prediction_vs_ground_truth**_

## 1. Why volumes

Whole-gland volume is the denominator of PSA density (serum PSA / gland volume). Volumes are computed as voxel count x |det(affine)|, so they are only valid because the reconstruction restores the source affine exactly; see the reconstruction ablation for what the same voxels yield without it.

## 2. Predicted volumes

| Quantity | Mean | SD | Min | Median | Max |
|---|---|---|---|---|---|

| CG (mL) | 34.23 | 14.28 | 14.25 | 28.40 | 67.76 |

| PZ (mL) | 12.79 | 5.19 | 2.81 | 11.53 | 25.74 |

| Whole gland (mL) | 47.02 | 14.02 | 24.26 | 45.96 | 78.86 |

| PZ fraction | 0.288 | 0.121 | 0.109 | 0.285 | 0.486 |


## 3. Agreement with ground truth

| Quantity | Bias (mL) | SD (mL) | 95% limits of agreement (mL) |

|---|---|---|---|

| CG | -1.48 | 4.76 | -10.82 to 7.85 |

| PZ | -5.10 | 5.84 | -16.53 to 6.34 |

| Whole gland | -6.58 | 6.94 | -20.18 to 7.02 |


Bias is mean(prediction - ground truth): positive means the model over-estimates. The limits of agreement, not the bias, are what determine whether a volume is usable for an individual patient -- a near-zero bias with wide limits means errors cancel across the cohort while remaining large case by case.
