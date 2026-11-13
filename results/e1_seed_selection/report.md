# E1 seed / 2.5D selection (validation only)

Rule: `results/seed_25d_prespecification.md` (SHA-256 `83ba8e596ec4...`, checked before running). 20 validation cases; the test split was not read.

## Step 1. Every seed (src/evaluate.run_evaluation, split=val)

| Arm | Seed | Epoch | CG Dice | PZ Dice | Macro Dice |
|---|---|---|---|---|---|
| 2d | 42 | 88 | 0.8662 | 0.7527 | 0.8094 |
| 2d | 43 | 62 | 0.8614 | 0.7438 | 0.8026 |
| 2d | 44 | 98 | 0.8639 | 0.7384 | 0.8012 |
| 2.5d | 42 | 74 | 0.8723 | 0.7468 | 0.8096 |
| 2.5d | 43 | 86 | 0.8550 | 0.7249 | 0.7900 |
| 2.5d | 44 | 57 | 0.8735 | 0.7488 | 0.8111 |
| **2d** | mean (SD) | | 0.8638 (0.0024) | 0.7450 (0.0072) | 0.8044 (0.0044) |
| **2.5d** | mean (SD) | | 0.8670 (0.0103) | 0.7402 (0.0132) | 0.8036 (0.0118) |

## Step 2. Arm

2.5D minus 2D mean macro Dice = -0.0008; the prespecified margin is 0.005. Adopted arm: **2d**.

## Step 3. Ensemble and test-time augmentation

| Model | CG Dice | PZ Dice | Macro Dice | CG HD95 (mm) | PZ HD95 (mm) |
|---|---|---|---|---|---|
| 2d seed42 | 0.8662 | 0.7527 | 0.8094 | 5.25 | 5.41 |
| 2d seed43 | 0.8614 | 0.7438 | 0.8026 | 5.23 | 5.21 |
| 2d seed44 | 0.8639 | 0.7384 | 0.8012 | 5.27 | 5.82 |
| 2d ensemble | 0.8726 | 0.7578 | 0.8152 | 4.62 | 4.68 |
| 2d ensemble + tta | 0.8732 | 0.7651 | 0.8192 | 4.30 | 4.92 |

Left-right flip TTA is kept (0.8192 vs 0.8152).

Reproducibility of step 1 (CPU re-score of the adopted seeds): seed42 |diff| 0.00000, seed43 |diff| 0.00000, seed44 |diff| 0.00000.

## Final model (frozen)

2d softmax ensemble of seeds 42, 43, 44 with left-right flip TTA. Member weights (SHA-256):

- seed 42, epoch 88: `2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2`
- seed 43, epoch 62: `0ee59ac4c5511869c4910cfb6a385ae2f758623b8137f42a7052c15eab443179`
- seed 44, epoch 98: `427b134123b64ddfceab0a9043eb7937dd424db9c2ab2924975ab0512012cd98`
