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
3. **A prespecified test of clinical usefulness** (JMIR manuscript): PSA density
   computed from the model's volume against PSA density from reported volumes,
   on 1,419 patients of the public PI-CAI cohort, reported whatever the outcome.

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

## The frozen models

The numbers above are those of the thesis, which uses a **single-seed model**:

```
experiment : exp_e1_dice_ce   (Dice + unweighted Cross Entropy)
epoch      : 88
SHA-256    : 2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2
parameters : 7,762,531
```

The JMIR manuscript (revision v10) uses a **final model** defined later by a
prespecified rule (`results/seed_25d_prespecification.md`, decision in
`results/e1_seed_selection/selection.json`, validation data only). E1 was
replicated with seeds 42, 43 and 44, with 2D and 2.5D input. 2.5D input was not
adopted (validation macro Dice −0.0008 against a required gain of 0.005). The
final model is the softmax ensemble of the three 2D seeds with left-right flip
averaging:

```
seed 42, epoch 88 : 2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2
seed 43, epoch 62 : 0ee59ac4c5511869c4910cfb6a385ae2f758623b8137f42a7052c15eab443179
seed 44, epoch 98 : 427b134123b64ddfceab0a9043eb7937dd424db9c2ab2924975ab0512012cd98
```

| Final model (sample SD) | CG Dice | PZ Dice | Macro |
|---|---|---|---|
| Held-out test, 19 cases | 0.862 (0.050) | 0.710 (0.133) | 0.786 |
| PROSTATEx, 204 cases | 0.827 (0.101) | 0.646 (0.159) | 0.736 |
| NCI-ISBI 2013, 78 cases | 0.806 (0.143) | 0.616 (0.173) | 0.711 |

On the held-out set it beats the single-seed model by +0.022 macro Dice (better
in 16 of 19 cases, P=.005). It costs 6 forward passes per slice. Results:
`results/final_ensemble_*`, `results/external_prostatex_final`,
`results/external_nci_isbi_final`.

Identity is confirmed by checksum, never by filename — configurations E1 and E3
are otherwise nearly identical.

---

## PSA density on PI-CAI (JMIR manuscript, v10)

A third analysis asks whether the model's volumes are useful where they reach the
clinic. It was specified in writing, and the file hash recorded, before any
PI-CAI file was downloaded (`results/picai_psad_prespecification.md`, lock in
`results/picai_psad/prespec_lock.json`). The frozen final model was applied to
the 1,440 eligible first studies of PI-CAI v2.0 (3 centers), and PSA density
from the model's whole-gland volume was compared with PSA density from the volume
in the radiology report for detecting clinically significant prostate cancer.

| PSA density from | AUC (95% CI) | Sensitivity at 0.15 | Specificity at 0.15 |
|---|---|---|---|
| Model volume | 0.732 (0.703-0.761) | 0.821 | 0.474 |
| Reported volume | 0.726 (0.696-0.755) | 0.722 | 0.608 |

Difference in AUC, model minus report: **0.006 (95% CI -0.007 to 0.020)**
in 1,419 patients (403 with cancer). **Noninferior** at the prespecified margin
of -0.05; **not superior**. The model's volumes are 13.6 mL smaller than the
reported ones (median 19.7%), so the usual 0.15 ng/mL/mL threshold trades
specificity for sensitivity: the two volumes are interchangeable for ranking
patients, not for threshold-based decisions. At 1 of the 3 centers the lower
confidence bound fell below the margin (descriptive analysis).
155 PI-CAI studies duplicate images of the external cohorts (none used for
training); excluding them does not change the result. Full report:
`results/picai_psad/report.md`.

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
  manuscript_numbers_jmir.py        the same for the JMIR revision (both models)
  manuscript_content_{en,id}_jmir.py  JMIR text, with guards that stop the build
                                    if the data no longer support a sentence
  run_nci_external_validation.py    NCI-ISBI 2013, prespecified rule
  nci_two_annotation_overlap.py     model vs two independent annotations
  e1_seed_selection.py              applies the prespecified seed and 2.5D rule
  run_picai_psad.py                 PI-CAI: cohort, conversion, inference, analysis
  merge_picai_colab.py              merges and cross-checks the PI-CAI Colab runs
  literature_audit.py               audit of how published 2D prostate studies report
                                    the 2D-to-3D step (protocol, coding and report in
                                    results/literature_audit/; full texts not committed)
  build_jmir_submission_pack.py     appendices, cover letter, figures
  build_manuscript.py               renders the manuscript (--jmir for JMIR,
                                    --picai adds the PI-CAI analysis = v10)
  gen_daftarpustaka.py              generates the thesis bibliography

configs/    tests/    notebooks/    results/
```

---

## Reproducing

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt
pytest -q tests/                # 469 passed, 7 skipped
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
- **NCI-ISBI 2013** (CC BY 3.0) — confirmatory external cohort for the
  prespecified connected-component rule, inference only.
- **Medical Segmentation Decathlon, Task05** (CC BY-SA 4.0) — fault injection
  and the two-annotation comparison only.
- **PI-CAI** public training and development set, v2.0 (CC BY-NC 4.0), with
  labels from the `picai_labels` repository at a fixed commit — PSA density
  analysis only, inference only.

No dataset is redistributed here. `.gitignore` excludes `dataset/`,
`*.nii*`, `*.npz` and checkpoints.

**Label convention:** `0 = background, 1 = central gland (CZ+TZ), 2 = peripheral
zone`, verified by multi-planar inspection rather than assumed.

---

## Reporting conventions

- Standard deviations are population SD (ddof = 0) in the thesis and sample SD
  (ddof = 1) in the JMIR manuscript; Bland–Altman limits of agreement use
  ddof = 1, per that statistic's convention.
- Macro Dice is the mean of CG and PZ only; background is reported separately.
- Metrics are computed per case in original voxel space, then averaged across
  cases.
- Every number in the manuscript is loaded programmatically from a result
  artifact. None is typed by hand.
- The held-out split was used twice during earlier pipeline development, before
  these four configurations were designed, and once more for the single-seed
  model before the seed extension was written. That is disclosed rather than
  described as pristine.

---

## Licence

Code released under the MIT Licence. The datasets carry their own terms.
