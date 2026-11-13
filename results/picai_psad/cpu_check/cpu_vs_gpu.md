# Laptop CPU vs Colab GPU on the same PI-CAI studies (2026-11-12)

The Colab run (T4, bundle `picai_psad_bundle.zip`, SHA-256
29021394e1c61b3f50420aa8315c3d43eca4e66f8a57398537c0a439361e50bd) produces the main results. To
check that the device does not matter, fold 0 was also converted on the laptop and the first 3
studies were inferred on its CPU with the same script.

- Conversion records (shape, spacing, axis codes, archive member, corner check): 289 of 289
  identical between laptop and Colab.
- Predictions, voxel by voxel:

| Study | Whole gland CPU (mL) | Whole gland GPU (mL) | Voxels differing | Geometry check |
|---|---|---|---|---|
| 10000_1000000 | 28.036 | 28.036 | 0 of 107,404 | passed / passed |
| 10001_1000001 | 89.588 | 89.588 | 0 of 276,507 | passed / passed |
| 10003_1000003 | 67.778 | 67.778 | 0 of 90,371 | passed / passed |

The Bosma22b masks of these studies lie on the T2-weighted grid (whole-gland Dice against the model
0.85, 0.96 and 0.93).

CPU outputs: `per_case_cpu.csv`, `model_identity_cpu.json` (this folder). Script: scratchpad
`cpu_vs_gpu.py`, run once.
