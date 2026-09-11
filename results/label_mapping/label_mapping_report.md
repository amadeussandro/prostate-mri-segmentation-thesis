# Gate 2 — Label Mapping Verification (Programmatic)
Sample size: 15 cases (errors: 0)
Sampled patient IDs: ['020', '029', '039', '049', '059', '069', '079', '089', '099', '109', '119', '129', '139', '149', '158']

Union of unique mask values across sample: [0, 1, 2]

## Per-case results
| pid | uniq | vox(1) | vox(2) | slices(1) | slices(2) | ext%(1) | ext%(2) | cdist(1) | cdist(2) | vol1:vol2 |
|---|---|---|---|---|---|---|---|---|---|---|
| 020 | [0, 1, 2] | 59304 | 43237 | (1, 19) | (1, 16) | 0.156 | 0.342 | 0.728 | 0.889 | 1.37 |
| 029 | [0, 1, 2] | 51703 | 54210 | (8, 24) | (8, 21) | 0.207 | 0.317 | 0.679 | 0.960 | 0.95 |
| 039 | [0, 1, 2] | 46633 | 22855 | (9, 21) | (9, 17) | 0.312 | 0.434 | 0.778 | 0.874 | 2.04 |
| 049 | [0, 1, 2] | 278226 | 27117 | (2, 26) | (2, 14) | 0.122 | 0.399 | 0.741 | 0.963 | 10.26 |
| 059 | [0, 1, 2] | 140746 | 85255 | (0, 23) | (0, 17) | 0.129 | 0.318 | 0.708 | 0.897 | 1.65 |
| 069 | [0, 1, 2] | 32342 | 18177 | (5, 17) | (5, 13) | 0.258 | 0.409 | 0.712 | 0.913 | 1.78 |
| 079 | [0, 1, 2] | 33372 | 26494 | (4, 19) | (4, 15) | 0.232 | 0.358 | 0.723 | 0.881 | 1.26 |
| 089 | [0, 1, 2] | 253592 | 61364 | (3, 20) | (3, 13) | 0.163 | 0.422 | 0.733 | 0.948 | 4.13 |
| 099 | [0, 1, 2] | 112362 | 45987 | (1, 19) | (1, 13) | 0.160 | 0.290 | 0.716 | 0.890 | 2.44 |
| 109 | [0, 1, 2] | 27093 | 17193 | (6, 17) | (6, 15) | 0.276 | 0.461 | 0.696 | 0.965 | 1.58 |
| 119 | [0, 1, 2] | 79874 | 36386 | (7, 25) | (7, 18) | 0.165 | 0.422 | 0.709 | 0.937 | 2.20 |
| 129 | [0, 1, 2] | 80159 | 13979 | (5, 24) | (5, 16) | 0.183 | 0.482 | 0.737 | 0.952 | 5.73 |
| 139 | [0, 1, 2] | 208411 | 33626 | (5, 24) | (5, 13) | 0.157 | 0.534 | 0.745 | 0.975 | 6.20 |
| 149 | [0, 1, 2] | 610305 | 39914 | (7, 31) | (7, 14) | 0.123 | 0.502 | 0.758 | 0.996 | 15.29 |
| 158 | [0, 1, 2] | 198708 | 83291 | (5, 20) | (5, 15) | 0.176 | 0.362 | 0.716 | 0.920 | 2.39 |

## Aggregate (mean across sample)
- Mean exterior-contact fraction, label 1: **0.188**
- Mean exterior-contact fraction, label 2: **0.403**
- Mean normalized centroid distance, label 1: **0.725**
- Mean normalized centroid distance, label 2: **0.931**
- Mean volume ratio label1:label2: **3.95**

## Morphological hypothesis (NOT a confirmed ground truth)
- Exterior-contact fraction suggests label **2** is the more peripheral/shell-like structure (higher fraction of its voxels touch the outer gland boundary) -> hypothesis: label 2 = **PZ**, label 1 = **CG/TZ**.
- Centroid-distance test AGREES with the exterior-contact test (on the aggregate mean).
- Per-case consistency: exterior-contact test agrees with the aggregate hypothesis in **15/15** sampled cases; centroid-distance test agrees in **15/15** sampled cases.
- This is quantitative morphological evidence from real data, computed across 15 cases spread across the full patient-ID range. It is consistent with the MONAI convention (CG=1, PZ=2) if label 2 = PZ.

## VERDICT
**LABEL MAPPING REQUIRES HUMAN / 3D SLICER VERIFICATION.**

Rationale: morphological heuristics (exterior contact, centroid distance) are strong supporting evidence but are inference, not ground truth — they cannot distinguish a genuinely ambiguous or atypical case, and published sources on this exact dataset already disagree with each other. Per the research protocol, this gate does not resolve the mapping by assumption. The pipeline continues to read and report labels as neutral **'label 1'** / **'label 2'** (or 'class 1' / 'class 2') until a human confirms the anatomical identity by opening `t2.nii.gz` and `t2_anatomy_reader1.nii.gz` together in 3D Slicer for a sample of cases (suggested: patients 020, 059, 099, 139) and visually identifying which integer is the thin outer rim (PZ) vs the central mass (CG/TZ).

Working (unverified) hypothesis for reference during development: label 2 = PZ, label 1 = CG/TZ. Do not report final thesis per-class results under these names until confirmed.
