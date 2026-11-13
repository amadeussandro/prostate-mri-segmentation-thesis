# RQ2 -- anatomical zone volumes (mL)

_Generated: 2026-11-11T09:51:14.669065+00:00 | 19 cases | mode: **prediction_vs_ground_truth**_

## 1. Why volumes

Whole-gland volume is the denominator of PSA density (serum PSA / gland volume). Volumes are computed as voxel count x |det(affine)|, so they are only valid because the reconstruction restores the source affine exactly; see the reconstruction ablation for what the same voxels yield without it.

## 2. Predicted volumes

| Quantity | Mean | SD | Min | Median | Max |
|---|---|---|---|---|---|

| CG (mL) | 34.13 | 14.51 | 13.79 | 28.38 | 69.69 |

| PZ (mL) | 12.87 | 5.13 | 4.73 | 12.10 | 26.02 |

| Whole gland (mL) | 47.00 | 13.97 | 24.43 | 45.32 | 79.87 |

| PZ fraction | 0.292 | 0.121 | 0.085 | 0.286 | 0.478 |


## 3. Agreement with ground truth

| Quantity | Bias (mL) | SD (mL) | 95% limits of agreement (mL) |

|---|---|---|---|

| CG | -1.59 | 4.03 | -9.50 to 6.32 |

| PZ | -5.01 | 5.74 | -16.26 to 6.23 |

| Whole gland | -6.60 | 6.54 | -19.41 to 6.21 |


Bias is mean(prediction - ground truth): positive means the model over-estimates. The limits of agreement, not the bias, are what determine whether a volume is usable for an individual patient -- a near-zero bias with wide limits means errors cancel across the cohort while remaining large case by case.
