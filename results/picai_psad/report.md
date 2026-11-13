# PI-CAI: PSA density from the final model's whole-gland volume

Prespecification SHA-256 (LF): `6e1b6bab750d696aa8c4f05618a425c5c2bd00820137fc989553735263193c9c`; labels commit `ce4a4723d7c46d882a6cbaacb40ed6c4be86282f`; generated 2026-11-12T21:23:23.266744+00:00.

1500 studies from 1476 patients; 1440 first studies inferred; 1419 in the primary comparison (403 with csPCa).

## Exclusions

| Reason | n |
|---|---|
| later study of the same patient | 24 |
| no PSA | 36 |
| no csPCa label | 0 |
| no reported prostate volume | 21 |

## Primary endpoint

| | AUC (95% CI) |
|---|---|
| PSA density, model volume | 0.732 (0.703-0.761) |
| PSA density, report volume | 0.726 (0.696-0.755) |
| Difference (model - report) | 0.006 (-0.007 to 0.020) |

Noninferiority (margin -0.05): shown; superiority: not shown.

## Volume agreement

| Comparison | n | Bias (mL) | 95% LoA (mL) | Median rel. diff | Spearman |
|---|---|---|---|---|---|
| Model vs report | 1419 | -13.6 | -48.7 to 21.5 | -0.197 | 0.867 |
| Model vs Bosma22b | 1440 | -8.3 | -30.7 to 14.2 | -0.113 | 0.919 |

Whole-gland Dice vs Bosma22b (same grid, n = 1440): mean 0.840, median 0.880.

## Threshold 0.15 ng/mL/mL

| PSA density from | Sensitivity | Specificity |
|---|---|---|
| model | 0.821 | 0.474 |
| report | 0.722 | 0.608 |
| bosma | 0.772 | 0.559 |

Discordant patients: 206 (model positive only 191, report positive only 15).

## By center

| Center | n (csPCa) | AUC model | AUC report | Difference (95% CI) |
|---|---|---|---|---|
| PCNN | 323 (96) | 0.592 | 0.607 | -0.015 (-0.056 to 0.028) |
| RUMC | 761 (230) | 0.764 | 0.758 | 0.006 (-0.008 to 0.020) |
| ZGT | 335 (77) | 0.790 | 0.782 | 0.008 (-0.015 to 0.031) |

## Sensitivity analyses

- rule_R1: difference 0.007 (-0.007 to 0.020), n = 1419
- without_duplicates: difference 0.006 (-0.008 to 0.021), n = 1265, removed 154
- without_geometry_failures: no geometry failures
- Bosma22b PSA density (reference): AUC 0.739, vs report 0.013 (0.002 to 0.024)

## Verification

- Conversion corner check: max 0.00004 mm.
- Header-based geometry check: 1440 of 1440 passed.
- Round-trip of Bosma22b masks: 1439 of 1440 exact (0 empty masks).
- Empty model segmentations: 1.
- Duplicates (same RAS array shape and Pearson r >= 0.99): 156 {'PROSTATEx': 152, 'NCI-ISBI': 4}.
- Share of patients not in the primary comparison: 3.9%.
