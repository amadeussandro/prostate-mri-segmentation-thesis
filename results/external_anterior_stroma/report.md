# Is the external central-gland gap a model failure or an annotation difference?
_204 external cases against 19 internal test cases. Diagnostic only: nothing here changes the reported method or the reported numbers._
## Hypothesis
The external reference includes anterior fibromuscular stroma in its non-PZ mask; Prostate158 does not. The stroma lies at the anterior edge of the gland, so the external reference asks the model to label a structure it never saw in training, at a predictable location. External CG recall should therefore fall toward the anterior far more steeply than internal CG recall does.
## Orientation gate
The anterior axis is read from each volume's affine, not assumed from storage order, and its sign is checked against anatomy: the PZ centroid must lie posterior to the CG centroid.
- External: **204/204** pass
- Internal: **19/19** pass

## Recall along the anterior-posterior axis of the CG reference
Position is normalised per case: 0 is the most posterior CG voxel, 1 the most anterior. Each cell is the fraction of reference CG voxels in that decile that the model labelled CG.

Per-case mean first, then the same quantity pooled over voxels.

| Decile (posterior - anterior) | External | Internal | External (pooled) | Internal (pooled) |
|---|---|---|---|---|
| 0.0-0.1 | 0.460 | 0.624 | 0.602 | 0.722 |
| 0.1-0.2 | 0.627 | 0.783 | 0.759 | 0.845 |
| 0.2-0.3 | 0.755 | 0.849 | 0.814 | 0.853 |
| 0.3-0.4 | 0.823 | 0.878 | 0.847 | 0.878 |
| 0.4-0.5 | 0.856 | 0.871 | 0.872 | 0.873 |
| 0.5-0.6 | 0.872 | 0.855 | 0.884 | 0.851 |
| 0.6-0.7 | 0.874 | 0.822 | 0.888 | 0.819 |
| 0.7-0.8 | 0.854 | 0.793 | 0.871 | 0.801 |
| 0.8-0.9 | 0.771 | 0.712 | 0.801 | 0.740 |
| 0.9-1.0 | 0.499 | 0.582 | 0.538 | 0.576 |

## Headline split at the midpoint of the gland
| | External | Internal | Difference |
|---|---|---|---|
| CG recall, posterior half | 0.7965 | 0.8550 | -0.0585 |
| CG recall, anterior half | 0.8401 | 0.8086 | +0.0316 |
| Anterior deficit (post - ant) | -0.0436 ± 0.1208 | 0.0465 ± 0.0859 | -0.0901 |
| Recall, most anterior decile | 0.4985 | 0.5820 | -0.0835 |

Mann-Whitney on the per-case anterior deficit, external vs internal: p = 0.0020. A deficit of similar size in both cohorts is a property of the anterior gland, not of the external reference.

## Where the cohort gap lives
CG Dice restricted to a half-space, the restriction applied identically to prediction and reference.

| CG Dice | External | Internal | Gap |
|---|---|---|---|
| Whole gland (as reported) | 0.8137 | 0.8442 | -0.0304 |
| Posterior half | 0.7895 | 0.8516 | -0.0621 |
| Anterior half | 0.8307 | 0.8366 | -0.0059 |

Same split taken on the whole-gland extent instead of the CG extent, as a check that the result is not an artefact of which structure set the plane:

| CG Dice | External | Internal | Gap |
|---|---|---|---|
| Posterior half (whole-gland plane) | 0.7688 | 0.8364 | -0.0677 |
| Anterior half (whole-gland plane) | 0.8389 | 0.8460 | -0.0071 |

## What the model put where it missed
Voxel-pooled destination of reference CG the model did not call CG, at the two ends of the gland.

| Reference CG not labelled CG | External | Internal |
|---|---|---|
| Posterior 20%, given background | 23.2% | 9.6% |
| Posterior 20%, given PZ | 4.3% | 8.7% |
| Anterior 20%, given background | 25.8% | 29.3% |
| Anterior 20%, given PZ | 0.3% | 0.3% |

At both ends the misses go to background rather than to the other zone, so they are the gland boundary falling short, not the CG/PZ interface being drawn in the wrong place.

## Unlabelled volume
Reference CG the model did not label, per case: **3.40 mL** anteriorly against 4.00 mL posteriorly externally; 3.29 mL against 2.62 mL internally.

## Verdict: NOT SUPPORTED (1/3 predictions met)
1. Anterior deficit larger externally than internally — **not met**: -0.0436 externally against +0.0465 internally, an excess of -0.0901 (p = 0.0020).
2. Cohort gap smaller when restricted to the posterior gland — **not met**: -0.0304 whole gland against -0.0621 posterior.
3. Anterior misses fall to background rather than to PZ — met: 49.7% background against 0.5% PZ. This one turns out not to discriminate: the misses go to background at both ends of the gland and in both cohorts, so it is consistent with the hypothesis without being evidence for it.

Of the external central-gland gap, **9%** sits in the anterior half and **91%** in the posterior half.

## What follows for the manuscript
The anterior fibromuscular stroma does not account for the external central-gland gap. The anterior half of the gland — exactly where the stroma lies — is where the two cohorts agree **best** (-0.0059), while the gap concentrates posteriorly (-0.0621). The direction of the anterior deficit is reversed between cohorts: internally the anterior half is the weaker one, externally the posterior half is (p = 0.0020).

The differing zone definitions remain a real difference between the two datasets, documented by their authors, and the manuscript should keep saying so. What the manuscript should no longer imply is that this difference explains the external shortfall, because the shortfall is not where that explanation puts it.

What the gap looks like instead: reference central gland that the model leaves as background at the posterior extent of the gland, with little of it handed to PZ. That is the whole-gland boundary falling short out of domain, the same under-segmentation already reported internally in millilitres, rather than a zone-assignment error or a missing structure.

## What this cannot establish either way
The two cohorts differ in reference protocol, reader, scanner and field of view at once, so no single difference is isolated here. No stroma mask exists in either dataset to subtract, so the structure itself is never measured — only the place where the gap sits. The internal arm is 19 cases against 204, and the anterior-posterior axis, while read from each affine and checked against anatomy, is a straight-line approximation of a curved gland.

No number reported in the manuscript changes. The reported external CG Dice remains 0.8137 on the unmodified pipeline.
