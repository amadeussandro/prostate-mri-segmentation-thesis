# Prostate Zonal Segmentation with a 2D U-Net and a Verified 2D→3D Reconstruction

Code and analysis for an undergraduate thesis at Universitas Multimedia
Nusantara on segmenting the anatomical zones of the prostate from T2-weighted
MRI, and on **verifying** the 2D→3D reconstruction that turns slice-wise output
back into a volume.

**Author:** Benedict Amadeus Sandro ([ORCID 0009-0000-1459-9139](https://orcid.org/0009-0000-1459-9139))
**Supervisor:** Eunike Endariahna Surbakti, S.Kom., M.T.I.

---

## What this study is, and is not

It is **not** a new architecture, not a state-of-the-art claim, and not a
clinical tool. The 2D U-Net is a standard component, deliberately so.

The contribution is on a different axis:

1. **An empirical comparison of four single-factor training configurations**
   under a selection rule fixed before the held-out data were examined,
   including the **negative result** that three interventions moved performance
   by less than the variation between patients.
2. **A measured demonstration that geometric validation alone is insufficient**
   for 2D→3D reconstruction. An anatomically mirrored volume passes every item
   of a geometry checklist; a lost affine leaves Dice completely untouched.
   Neither check finds both, and the cost of each omitted step is quantified in
   Dice, millimetres and millilitres.

### Research questions

| | |
|---|---|
| **RQ1** | What accuracy does a 2D U-Net achieve for prostate zonal segmentation? |
| **RQ2** | How is the 3D reconstruction of that 2D output implemented and verified? |

---

## Headline results

Held-out test set, 19 Prostate158 cases, scored in original voxel space:

| Metric | Central gland | Peripheral zone |
|---|---|---|
| Dice | **0.8442 ± 0.0493** | **0.6852 ± 0.1286** |
| Precision | 0.8646 | 0.8297 |
| Recall | 0.8318 | **0.6090** |

Macro Dice **0.7647**. The peripheral zone pairs high precision with low recall
— it is **under-segmented, not mislocated** — and that pattern recurs in every
downstream analysis.

**Reconstruction ablation** (same predictions, only the reconstruction varies):

| Shortcut | Dice CG | Dice PZ | Centroid shift | Volume error | Caught by checklist |
|---|---|---|---|---|---|
| *(none — correct)* | 0.8442 | 0.6852 | 0.0 mm | 0% | — |
| **No reorientation** | **0.5960** | **0.1758** | 8.2 mm | 0% | **0/19** |
| **Identity affine** | 0.8442 | 0.6852 | **182.8 mm** | **47%** | 19/19 |

The two failure modes are exactly complementary. That is the point of the study.

**External validation:** 204 PROSTATEx examinations, inference only, frozen
model. Macro Dice 0.7192 — a drop of 0.0455. Reconstruction passed 204/204 with
an affine difference of 0.0, across a different orientation convention (LAS vs
LPS) and both branches of the in-plane transform.

**Known limitation:** accuracy depends on gland size. On the external cohort,
central-gland Dice rises from 0.7287 in the smallest size tertile to 0.8754 in
the largest (Spearman r = 0.766).

---

## The frozen model

```
experiment : exp_e1_dice_ce   (Dice + unweighted Cross Entropy)
epoch      : 88
SHA-256    : 2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2
parameters : 7,762,531
```

Identity is confirmed by checksum, never by filename — configurations E1 and E3
are otherwise nearly identical.

---

## Repository layout

```
src/                    core implementation
  transforms.py         config-driven preprocessing + exact inverses
  reconstruction.py     explicit-index 2D→3D assembly, round-trip support
  geometry.py           the geometric validation checklist
  metrics.py            Dice / IoU / HD95 / ASD / precision / recall, bootstrap CI
  model.py              the 2D U-Net
  train.py  evaluate.py  dataset.py  losses.py  splits.py

scripts/                analyses and manuscript build
  run_reconstruction_ablation.py    what each reconstruction step is worth
  run_zone_volumes.py               volumes → Bland–Altman → PSA density
  run_prostatex_external_validation.py
  analyse_external_hd95_tail.py     why the boundary-metric tail exists
  analyse_anterior_stroma.py        tests, and refutes, the stroma explanation
  analyse_gland_size_effect.py      accuracy versus gland size
  manuscript_numbers.py             every reported value, loaded from artifacts
  build_manuscript.py               renders the manuscript
  gen_daftarpustaka.py              generates the thesis bibliography

configs/    tests/    notebooks/    results/
```

---

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt
pytest -q tests/                # 438 passed, 7 skipped
```

CPU is enough for the verification gates, every analysis script, and the
manuscript build. Only training and large-cohort inference need a GPU; those
run on Colab (T4) via generated notebooks, because the development laptop's
GTX 1050 has 4 GB of VRAM — not enough to train at a batch size that keeps the
four configurations comparable.

```bash
python scripts/roundtrip_test.py              # must print Dice = 1.0
python scripts/verify_label_mapping.py
python scripts/analyse_gland_size_effect.py
```

---

## Data

- **Prostate158** (Adams et al, 2022) — development data, official
  patient-level split 119 / 20 / 19.
- **PROSTATEx** (Litjens et al) with zonal annotations by Cuocolo et al,
  released CC BY 4.0 — external cohort, inference only. Underlying imaging from
  The Cancer Imaging Archive.

Neither dataset is redistributed here. `.gitignore` excludes `dataset/`,
`*.nii*`, `*.npz` and checkpoints.

**Label convention:** `0 = background, 1 = central gland (CZ+TZ), 2 = peripheral
zone`, verified by multi-planar inspection rather than assumed.

---

## Reporting conventions

- Standard deviations are population SD (ddof = 0); Bland–Altman limits of
  agreement use ddof = 1, per that statistic's convention.
- Macro Dice is the mean of CG and PZ only; background is reported separately.
- Metrics are computed per case in original voxel space, then averaged across
  cases.
- Every number in the manuscript is loaded programmatically from a result
  artifact. None is typed by hand.
- The held-out split was used twice during earlier pipeline development, before
  these four configurations were designed. That is disclosed rather than
  described as pristine.

---

## Licence

Code released under the MIT Licence. The datasets carry their own terms.
