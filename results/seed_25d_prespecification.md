# Prespecification: seed replicates and 2.5D input for E1

Written on 2026-11-07, before any of the runs below was started. Its SHA-256 is
printed by the Colab notebook at the start of every run and stored in each run's
output folder, so a later edit to this file is visible.

## Why

1. Every training configuration was trained once (seed 42), so the variation
   between seeds is unknown and the equivalence results in the manuscript
   describe single runs.
2. The largest remaining weakness is the peripheral zone, whose boundary runs
   obliquely through the axial plane; a 2D network sees no neighbouring slice.
   2.5D input (slices s-1, s, s+1 as 3 channels) keeps the 2D U-Net and the
   slice-wise reconstruction that the study is about.

## Evidence behind the design choices

- **3 input slices (s-1, s, s+1).** In an empirical comparison on the MSD
  prostate task, 3 slices raised Dice over single-slice 2D (0.722 to 0.746) and
  cut HD95 (8.9 to 5.6 mm); 5 slices gave no further gain (0.745) and 7 slices
  was worse (0.726) (Zhang et al, arXiv:2010.06163). In CAT-Net's prostate zonal
  benchmark, nnU-Net with 3 slices improved PZ Dice over 1 slice by 0.2 points
  internally and 2.1 points on PROSTATEx; 5 slices was similar and 7 slices fell
  back, while a 3D U-Net was worse than every 2D/2.5D variant on this anisotropic
  data (Hung et al, arXiv:2203.15163). 3 slices is therefore the best-supported
  and cheapest choice. The expected gain is modest (about 0.2-2.5 Dice points),
  larger out of domain than in domain.
- **Seed replicates.** Seed alone changes segmentation performance, with
  frequent significant differences between models from one algorithm
  (Akesson, Toger, Heiberg, Comput Biol Med 2024;180:108944). Three seeds per
  arm is the most the budget allows; the comparison is paired by seed.
- **Seed ensemble and flip test-time augmentation.** Averaging models and
  mirroring at inference are standard in nnU-Net (Isensee et al, Nat Methods
  2021); mirroring helps single models but adds little once models are
  ensembled (arXiv:2609.15524), so it is tested on validation, not assumed.

## Runs (5 new; budget at most 300 Colab compute units)

| Arm | Config | Seeds | New runs |
|---|---|---|---|
| E1-2D | `configs/experiments/exp_e1_dice_ce.yaml` | 42 (existing), 43, 44 | 2 |
| E1-2.5D | `configs/experiments/exp_e1_25d.yaml` | 42, 43, 44 | 3 |

All runs: 100 epochs, FP32 (AMP off, TF32 off), NVIDIA T4, identical code except
`training.seed`, `data.context_slices` and `model.in_channels`. The two configs
differ only in those input keys (enforced by `tests/test_e1_25d_config.py`).
Within-run checkpoint selection is unchanged (best validation slice-level Dice).

If the budget runs out, runs are done in this order and the analysis uses the
completed seeds that both arms share: 2.5D seed 42, 2D seed 43, 2.5D seed 43,
2D seed 44, 2.5D seed 44.

## Selection rule (validation split only, 20 cases)

The outcome is case-level macro Dice ((CG + PZ) / 2) in original voxel space,
computed with `src/evaluate.py::run_evaluation(split="val")`.

1. For each arm, report every seed and the mean and sample SD across seeds.
2. **2.5D is adopted** if its mean validation macro Dice across shared seeds
   exceeds that of 2D by at least 0.005. Otherwise 2D is kept. The 0.005 is
   about the size of the selection-tolerance used for the connected-component
   rule (`results/lcc_prespecification.md`) and is fixed here.
3. The final model of the adopted arm is the softmax-average ensemble of its
   seeds. Horizontal-flip test-time augmentation is added only if it raises the
   ensemble's validation macro Dice; it is not tuned further.
4. Nothing in steps 1-3 reads the test split.

## After selection

- The adopted model is frozen (SHA-256 of each member's weights recorded) and
  evaluated once on the 19-case test split, with reader 2 as the second
  reference, as before.
- External testing repeats the earlier protocol on PROSTATEx and NCI-ISBI 2013.
  The connected-component rules R1 and R2 of `results/lcc_prespecification.md`
  apply unchanged; the internal no-harm check is repeated on the 20 validation
  cases before NCI-ISBI is run.
- The fault-injection experiment is model-free and is not repeated.
- All seeds of both arms are reported, whichever arm is adopted. A negative
  result (2.5D not better) is reported as such.
