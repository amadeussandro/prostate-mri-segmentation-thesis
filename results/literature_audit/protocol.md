# Protocol: how often do 2D prostate MRI segmentation studies report the 2D-to-3D step?

Written 2026-11-11, **before** any search result was screened or any full text was read.
`scripts/literature_audit.py` records this file's SHA-256 (LF line endings) in its outputs.

## Question

In published deep-learning studies that segment the prostate (whole gland or zones) on MRI with a
network that works on 2D slices (2D or 2.5D input), how often does the article report:

1. how the slice-wise output was reassembled into a 3D volume;
2. how preprocessing was inverted, so that the output returns to the source geometry;
3. any check that the reconstructed volume is spatially correct;
4. where the metrics were computed (original voxel space or a resampled grid)?

This supports one sentence of the manuscript: that reconstruction is routinely performed but rarely
reported or verified. It is a structured scoping audit by one reviewer, not a systematic review.

## Source and search (fixed)

- Database: PubMed Central (PMC), open-access subset, through NCBI E-utilities, so that every full
  text can be read and the search can be repeated exactly.
- Query (run once, date recorded):
  `prostate[ti] AND segmentation[ti] AND (MRI[tiab] OR "magnetic resonance"[tiab]) AND ("deep learning"[tiab] OR convolutional[tiab] OR "U-Net"[tiab] OR "neural network"[tiab]) AND 2017:2026[pdat] AND "open access"[filter]`
- No other records are added afterwards. The authors' own work is excluded if it appears.

## Eligibility (judged on the full text)

Include an article if all hold:
- original research (not a review, editorial, protocol, challenge summary or dataset-only paper);
- deep learning segmentation of the prostate gland or its zones on MRI;
- at least one reported model takes 2D slices as input (2D, or 2.5D with neighbouring slices as
  channels). Studies comparing 2D with 3D are included.

Exclude if every model is 3D, if the task is lesion-only segmentation, or if the modality is not MRI.
The reason for every exclusion is recorded.

## Items extracted per included study (fixed before reading)

| Item | Coding |
|---|---|
| `dimensionality` | 2D / 2.5D / 2D+3D / 2.5D+3D |
| `target` | gland / zones / both |
| `metric_level` | volume (per case, 3D) / slice (per 2D slice) / both / not stated |
| `metric_space` | original voxel space / resampled or preprocessed grid / not stated |
| `reassembly_described` | yes if the article states how slice outputs were stacked back into a volume (order, index, or "stacked along z" with any detail); no otherwise |
| `inverse_preprocessing_described` | yes if it states that resampling, cropping or reorientation were undone before evaluation or export; no otherwise; n/a if no preprocessing changed the geometry |
| `geometry_check_reported` | yes if any check of the reconstructed volume's spatial correctness is reported (header/affine comparison, round-trip of a reference mask, alignment check, orientation check); visual inspection counts only if stated as a check of the reconstruction |
| `orientation_mentioned` | yes if image orientation or reorientation is mentioned at all |
| `surface_metric_mm` | yes if a surface distance (Hausdorff, HD95, ASD) is reported in millimetres |
| `code_available` | yes if a public code link is given |

Each `yes` is backed by a short quotation (at most 25 words) from the article, with its section.
Uncertain cases are coded conservatively (`no` unless the text states it) and flagged.

## Analysis

Counts and percentages of included studies for each item, with an exact 95% (Clopper-Pearson)
CI for the proportion reporting a geometry check. No hypothesis test.

## Limitations stated in advance

One database, open-access articles only, one reviewer, title-level search terms. The result
describes reporting, not practice: a study may have verified its reconstruction without saying so.
