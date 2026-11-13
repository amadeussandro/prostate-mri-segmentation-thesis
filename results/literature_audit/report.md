# Literature audit: reporting of the 2D-to-3D step

Search run 2026-11-11T15:49:15.328604+00:00; 57 PMC open-access records; 57 screened on full text; 22 included.

Exclusions: 13 all models 3D; 6 target not prostate; 5 lesion-only; 4 input dimensionality not reported; 3 not MRI; 3 review or non-original; 1 not a segmentation model

| Item | Counts (of included studies) |
|---|---|
| dimensionality | 17 2D, 4 2D+3D, 1 2.5D |
| target | 13 gland, 6 both, 3 zones |
| metric_level | 14 not stated, 7 volume, 1 both |
| metric_space | 20 not stated, 2 resampled or preprocessed grid |
| reassembly_described | 22 no |
| inverse_preprocessing_described | 13 no, 9 n/a |
| geometry_check_reported | 22 no |
| orientation_mentioned | 21 no, 1 yes |
| surface_metric_mm | 16 no, 6 yes |
| code_available | 20 no, 2 yes |

Geometry check reported: 0/22 (exact 95% CI 0.0%-15.4%).
Inverse preprocessing described: 0/13 studies that reported resampling, resizing or cropping (the rest reported none).

## Included studies

| PMCID | First author | Year | Input | Target | Reassembly | Inverse | Geometry check | Metric space |
|---|---|---|---|---|---|---|---|---|
| PMC7192345 | Elguindi | 2019 | 2D | gland | no | n/a | no | not stated |
| PMC7309110 | Khan | 2020 | 2D | both | no | n/a | no | not stated |
| PMC7459118 | Aldoj | 2020 | 2D | both | no | no | no | not stated |
| PMC7555425 | Sunoqrot | 2020 | 2D+3D | gland | no | n/a | no | not stated |
| PMC8321056 | Cem Birbiri | 2020 | 2D | gland | no | no | no | not stated |
| PMC7932306 | Comelli | 2021 | 2D | gland | no | no | no | resampled or preprocessed grid |
| PMC8438764 | Saunders | 2021 | 2D+3D | both | no | no | no | not stated |
| PMC8471645 | Sunoqrot | 2021 | 2D+3D | both | no | n/a | no | not stated |
| PMC8724207 | Liu | 2021 | 2D | gland | no | n/a | no | not stated |
| PMC8821548 | Aldoj | 2022 | 2D | both | no | no | no | not stated |
| PMC9234226 | Fransson | 2022 | 2D | gland | no | no | no | not stated |
| PMC9511435 | Rezaeijo | 2022 | 2D | zones | no | no | no | not stated |
| PMC10135670 | Montazerolghaem | 2023 | 2D | gland | no | no | no | not stated |
| PMC10154598 | Ren | 2023 | 2D | gland | no | n/a | no | not stated |
| PMC9837765 | Zaridis | 2023 | 2D | both | no | n/a | no | not stated |
| PMC9921124 | Isaksson | 2023 | 2D+3D | gland | no | no | no | not stated |
| PMC11607187 | Gade | 2024 | 2D | gland | no | no | no | not stated |
| PMC12689646 | Alqarni | 2025 | 2D | gland | no | n/a | no | not stated |
| PMC12700846 | Krishnan | 2025 | 2.5D | zones | no | no | no | not stated |
| PMC12719691 | Dushatskiy | 2025 | 2D | gland | no | n/a | no | not stated |
| PMC12999624 | Jacobson | 2025 | 2D | zones | no | no | no | resampled or preprocessed grid |
| PMC13165886 | Ba | 2026 | 2D | gland | no | no | no | not stated |

## Limitations (stated in the protocol before the search)

One database (PMC open-access subset), title-level search terms, one reviewer. The audit describes reporting, not practice: a study may have verified its reconstruction without saying so. Uncertain codings are flagged in extraction.csv and were coded `no` unless the text stated the item.

Per-study coding with supporting quotations: extraction.csv. Screening with reasons: screening.csv. Protocol SHA-256 (LF): 15e65d75f5701e17a48b7a92fc698da8608b6ee81a36c04daaac81ed6599c70a.
