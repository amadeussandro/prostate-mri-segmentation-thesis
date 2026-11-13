# Why central-gland HD95 has a heavy tail on PROSTATEx
_204 cases. Diagnostic only: nothing here changes the reported method._
## Hypothesis
The tail is caused by small volumes of predicted central gland lying far from the gland, not by a boundary that is uniformly wrong. Such components carry too little volume to move Dice but dominate a 95th-percentile boundary distance.
## Test
Decompose each predicted class into connected components, then discard everything but the largest and re-measure. If the hypothesis holds, HD95 collapses while Dice barely moves.
## Result
| | All cases | Tail (HD95 > 30 mm) | Rest |
|---|---|---|---|
| Cases | 204 | 64 | 140 |
| HD95 (mm) | 20.2 | 44.5 | 9.1 |
| HD95, largest component only | 6.5 | 6.8 | 6.3 |
| Dice | 0.8137 | 0.7770 | 0.8305 |
| Dice, largest component only | 0.8266 | 0.8108 | 0.8339 |
| Mean components | 5.45 (tail) | 5.45 | 3.09 |
| Cases with >1 component | 86% | 100% | 80% |
| Satellite volume share | — | 8.3% | 1.4% |
| Mean distance to farthest satellite | — | 72.9 mm | 50.0 mm |

## Verdict: SUPPORTED
Discarding all but the largest component takes mean HD95 from 20.2 mm to 6.5 mm while mean Dice moves from 0.8137 to 0.8266. Correlation between component count and HD95 is +0.42.

## Why this is not applied
Keeping only the largest component would be a change to the method, and selecting it after seeing the external scores would be tuning on the external cohort. It is used here to explain the tail, not to improve the reported numbers. Adopting it would require pre-specifying it and validating it on the internal validation split first.
