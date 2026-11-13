# RQ2 ablation -- the measured cost of naive 3D reconstruction

_Generated: 2026-10-08T01:28:41.042588+00:00 | 19 cases | source: `results/rq2_e1_epoch88`_

## 1. What this measures

Every variant below is reconstructed from the **same saved 2D predictions**, so the model, its weights and its per-slice output are held fixed and the only thing that varies is the reconstruction stage. Differences are therefore attributable to reconstruction alone, with no model error mixed in. No inference is re-run and no ground truth is used: the reference is the verified correct reconstruction.

**Correctness gate:** PASS (19/19 rebuilt references bit-identical to the saved run). The reference was rebuilt here from the 2D predictions and the per-case metadata alone, and had to come out bit-identical to the volumes the main RQ2 run saved. A reference that cannot be reproduced cannot be measured against.

## 2. Variants

| Variant | Shortcut taken |
|---|---|

| `V0_correct` | Correct pipeline (reference) |

| `V1_no_reorientation` | Inverse orientation step skipped |

| `V2_naive_centre_crop` | Centre-crop instead of recorded pad offsets |

| `V3_identity_affine` | Saved with eye(4) instead of the source affine |

| `V4_append_order` | Slices stacked in iteration order, not by index |

| `V5_all_naive` | Every shortcut at once |


## 3. Results

| Variant | Class | Dice vs correct | Reported Dice vs GT | Centroid shift (mm) | Volume error (mL) | Volume error (%) | Caught by geometry check |

|---|---|---|---|---|---|---|---|

| `V1_no_reorientation` | CG | 0.6308 | 0.5960 | 8.20 | 0.00 | 0.0 | 0/19 |

| `V1_no_reorientation` | PZ | 0.1802 | 0.1758 | 17.25 | 0.00 | 0.0 | 0/19 |

| `V2_naive_centre_crop` | CG | 1.0000 | 0.8442 | 0.00 | 0.00 | 0.0 | 0/19 |

| `V2_naive_centre_crop` | PZ | 1.0000 | 0.6852 | 0.00 | 0.00 | 0.0 | 0/19 |

| `V3_identity_affine` | CG | 1.0000 | 0.8442 | 182.82 | 16.02 | 47.0 | 19/19 |

| `V3_identity_affine` | PZ | 1.0000 | 0.6852 | 205.09 | 6.08 | 47.0 | 19/19 |

| `V4_append_order` | CG | 1.0000 | 0.8442 | 0.00 | 0.00 | 0.0 | 0/19 |

| `V4_append_order` | PZ | 1.0000 | 0.6852 | 0.00 | 0.00 | 0.0 | 0/19 |

| `V5_all_naive` | CG | 0.6308 | 0.5960 | 185.07 | 16.02 | 47.0 | 19/19 |

| `V5_all_naive` | PZ | 0.1802 | 0.1758 | 181.65 | 6.08 | 47.0 | 19/19 |


The correct pipeline reports Dice 0.8442 (CG) and 0.6852 (PZ) against the expert annotation. The **Reported Dice vs GT** column is the number a reader would see in a results table if that shortcut had been taken -- the same model, the same predictions, only the reconstruction done differently.


## 4. The finding that matters

Skipping the inverse orientation step (`V1_no_reorientation`) produces a volume that is anatomically **mirrored** -- and it **passes the entire geometry checklist**. Shape, affine, voxel spacing, orientation codes and label set are all correct, because the header is correct; only the voxel data is wrong. The checklist inspects headers, so it cannot see this.

That error is not small. Against the correct reconstruction it scores Dice 0.63 (CG) and 0.18 (PZ), and displaces the peripheral zone by up to ~33 mm. A pipeline that validated geometry alone would have published it as a clean result.

The round-trip fidelity test is what catches it: pushing ground truth through this variant returns a mirrored mask, so the round-trip Dice collapses instead of returning 1.0. **This is the concrete justification for that test.** A round-trip Dice of 1.0 is not an inflated accuracy claim -- it is the only gate in the pipeline that detects a correct-looking file containing wrong voxels.

Conversely, `V3_identity_affine` keeps the voxels perfect (Dice 1.0 against the correct reconstruction, and an **unchanged** Dice against the annotation) while misplacing the zones by ~180-205 mm in scanner coordinates and inflating every volume by ~47%, because a 1 mm isotropic voxel is not a 0.47 x 0.47 x 3.0 mm one. Dice is computed on voxel indices, so it cannot see an affine error at all.

The two failure modes are exactly complementary:

| Failure | Dice detects it | Geometry checklist detects it |
|---|---|---|
| Orientation inversion (`V1`) | **yes** (0.84 -> 0.60 CG, 0.69 -> 0.18 PZ) | no |
| Lost affine (`V3`) | no (unchanged) | **yes** |

**Neither check alone finds both.** A pipeline that reports only overlap metrics and a pipeline that validates only headers are each blind to one of these, which is the argument for running both.


## 4. How to read this

- **Dice vs correct** is voxel agreement against the correct reconstruction, not against the expert annotation. 1.0 means the shortcut happened to be harmless on this cohort; it does not mean the segmentation is accurate.
- **Centroid shift** is the distance between where the zone sits in scanner coordinates under the variant's own affine and where it sits under the correct one. This is the error a downstream tool (biopsy planning, registration, any viewer) would inherit.
- **Volume error** is computed with each variant's own voxel size, which is the point: correct voxels saved under a wrong affine still yield a wrong millilitre figure.
- A shortcut scoring 1.0 here is **latent**, not safe, and the two reasons differ. `V2_naive_centre_crop` is harmless *structurally*: this pipeline always places the pad at `(target - native) // 2`, so a centre-crop can never diverge from the recorded offset for a pad -- but it still cannot invert a forward *crop*, which is what a larger-than-target cohort would produce. `V4_append_order` is harmless only *circumstantially*: this loader happens to emit slices in axial order, which is a property of the loader and can change without any change to the reconstruction code.
