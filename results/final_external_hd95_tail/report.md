# Why central-gland HD95 has a heavy tail on PROSTATEx
_204 cases. Diagnostic only: nothing here changes the reported method._
## Hypothesis
The tail is caused by small volumes of predicted central gland lying far from the gland, not by a boundary that is uniformly wrong. Such components carry too little volume to move Dice but dominate a 95th-percentile boundary distance.
## Test
Decompose each predicted class into connected components, then discard everything but the largest and re-measure. If the hypothesis holds, HD95 collapses while Dice barely moves.
## Result
| | All cases | Tail (HD95 > 30 mm) | Rest |
|---|---|---|---|
| Cases | 204 | 26 | 178 |
| HD95 (mm) | 11.8 | 43.5 | 7.2 |
| HD95, largest component only | 6.5 | 7.9 | 6.3 |
| Dice | 0.8271 | 0.7704 | 0.8353 |
| Dice, largest component only | 0.8304 | 0.7959 | 0.8355 |
| Mean components | 4.50 (tail) | 4.50 | 1.83 |
| Cases with >1 component | 45% | 100% | 37% |
| Satellite volume share | — | 9.9% | 0.5% |
| Mean distance to farthest satellite | — | 69.2 mm | 17.5 mm |

## Verdict: SUPPORTED
Discarding all but the largest component takes mean HD95 from 11.8 mm to 6.5 mm while mean Dice moves from 0.8271 to 0.8304. Correlation between component count and HD95 is +0.47.

## Why this is not applied
Keeping only the largest component would be a change to the method, and selecting it after seeing the external scores would be tuning on the external cohort. It is used here to explain the tail, not to improve the reported numbers. Adopting it would require pre-specifying it and validating it on the internal validation split first.
