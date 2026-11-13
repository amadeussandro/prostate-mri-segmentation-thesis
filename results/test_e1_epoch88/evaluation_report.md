# E1 — Final Held-Out Test Evaluation (RQ1)

Generated 2026-10-06T08:53:26.863542Z from commit `290fd935a4f4be4822e98bab5fcdae15dd153dfd`.

## 1. Run identity

| Field | Value |
|---|---|
| Experiment | `exp_e1_dice_ce` (Dice + Cross Entropy) |
| Checkpoint epoch | 88 |
| Checkpoint SHA-256 | `2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2` |
| Split | official held-out test, 19 cases |
| Evaluation | per-case, volume-level, original voxel space |
| Augmentation / TTA / post-processing | none |
| Device / seed | cuda / 42 |
| Label mapping | 0 = Background, 1 = CG, 2 = PZ (verified 2026-10-05) |

> E1 was selected on validation data only. These test numbers are reported as
> measured and were not used to reconsider the model, checkpoint or preprocessing.

## 2. Test results

| Class | Dice | IoU | HD95 (mm) | ASD (mm) | Precision | Recall |
|---|---|---|---|---|---|---|
| Background | 0.9923 ± 0.0037 | 0.9847 ± 0.0073 | 1.6973 ± 1.1329 | 0.2056 ± 0.1053 | 0.9885 ± 0.0075 | 0.9961 ± 0.0020 |
| CG | 0.8442 ± 0.0506 | 0.7333 ± 0.0722 | 6.3812 ± 3.1945 | 1.5596 ± 0.6517 | 0.8646 ± 0.0557 | 0.8318 ± 0.0880 |
| PZ | 0.6852 ± 0.1322 | 0.5339 ± 0.1349 | 6.5383 ± 4.1515 | 1.4269 ± 0.8300 | 0.8297 ± 0.0767 | 0.6090 ± 0.1629 |

**Macro Dice (CG + PZ) / 2 = 0.7647**  (per-case 0.7647 ± 0.0770 SD, median 0.7833)

Background is reported above for completeness and is **not** part of macro Dice.

## 3. Validation vs test

| Metric | Validation (20 cases) | Test (19 cases) | Difference |
|---|---|---|---|
| CG Dice | 0.8662 | 0.8442 | -0.0220 |
| PZ Dice | 0.7527 | 0.6852 | -0.0675 |
| Macro Dice | 0.8094 | 0.7647 | -0.0447 |
| CG precision | 0.8823 | 0.8646 | -0.0177 |
| CG recall | 0.8604 | 0.8318 | -0.0286 |
| PZ precision | 0.8073 | 0.8297 | +0.0224 |
| PZ recall | 0.7161 | 0.6090 | -0.1071 |
| CG HD95 (mm) | 5.2467 | 6.3812 | +1.1345 |
| PZ HD95 (mm) | 5.4061 | 6.5383 | +1.1322 |
| CG ASD (mm) | 1.1508 | 1.5596 | +0.4088 |
| PZ ASD (mm) | 0.9453 | 1.4269 | +0.4816 |

Validation drove model selection and is optimistically biased; the test column
is the unbiased estimate and is what the thesis reports.

## 4. Per-case results

See `per_case_metrics.csv` for all cases and classes. Per-case macro Dice:

| Case | CG Dice | PZ Dice | Macro Dice |
|---|---|---|---|
| 1 | 0.7115 | 0.7689 | 0.7402 |
| 2 | 0.8875 | 0.7908 | 0.8392 |
| 3 | 0.8719 | 0.6653 | 0.7686 |
| 4 | 0.7954 | 0.6861 | 0.7408 |
| 5 | 0.9002 | 0.6685 | 0.7844 |
| 6 | 0.8621 | 0.7046 | 0.7833 |
| 7 | 0.8829 | 0.7031 | 0.7930 |
| 8 | 0.8386 | 0.7056 | 0.7721 |
| 9 | 0.8843 | 0.7689 | 0.8266 |
| 10 | 0.8585 | 0.5547 | 0.7066 |
| 11 | 0.8399 | 0.7734 | 0.8066 |
| 12 | 0.8323 | 0.8004 | 0.8163 |
| 13 | 0.8302 | 0.7176 | 0.7739 |
| 14 | 0.8448 | 0.8307 | 0.8377 |
| 15 | 0.8547 | 0.7714 | 0.8131 |
| 16 | 0.7371 | 0.2783 | 0.5077 |
| 17 | 0.8614 | 0.7583 | 0.8099 |
| 18 | 0.8355 | 0.4890 | 0.6623 |
| 19 | 0.9104 | 0.5839 | 0.7471 |

## 5. Provenance and scope

* Test cases: 001, 002, 003, 004, 005, 006, 007, 008, 009, 010, 011, 012, 013, 014, 015, 016, 017, 018, 019
* Preprocessing is the checkpoint's own embedded configuration, identical to
  E1 training and validation.
* The official test set was evaluated once for this model. **Historical note:**
  the test set had previously been evaluated with the baseline experiment
  (2026-09-13) and used for the RQ2 reconstruction geometry study (2026-09-19),
  both with the Baseline epoch-77 checkpoint and both before the E1/E2/E3 arms
  were designed. This E1 run is therefore not the first historical use of the
  official test set.
* Historical experiment directories were not modified by this evaluation.
* **RQ1 only.** RQ2 reconstruction will use the frozen E1 model and is handled
  in a separate reconstruction workflow.

