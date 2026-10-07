# Project Handoff — read this first

Living state document for this thesis. A new session should read this before
touching anything; it records what is decided, what is frozen, and what is open,
so settled questions are not re-litigated.

**Last updated:** 2026-10-07 · **HEAD at writing:** `7d805e0`

---

## 1. The thesis

**Title:** *Segmentasi Zona Anatomi pada Citra MRI Pasien Kanker Prostat menggunakan 2D U-Net dan Rekonstruksi 3D*

| | |
|---|---|
| **RQ1** | Berapa hasil evaluasi segmentasi MRI prostat dengan 2D U-Net? |
| **RQ2** | Bagaimana implementasi rekonstruksi 3D dari hasil segmentasi 2D U-Net? |
| Author | Benedict Amadeus Sandro, Universitas Multimedia Nusantara |
| Supervisor | Bu Eunike |
| Dataset | Prostate158 — official split 119 train / 20 validation / 19 test |
| Target journal | **JMIR AI** (not JMIR flagship — see §7) |

---

## 2. Status

| Stage | State |
|---|---|
| Baseline, E1, E2, E3 training | **done** |
| Validation comparison + model selection | **done** — E1 selected |
| E1 frozen | **done** — SHA-256 recorded |
| RQ1 held-out test (19 cases) | **done**, audited, VALID |
| Label-mapping verification | **done** 2026-10-05 |
| RQ2 re-pointed to E1 | **done** — notebook ready to run |
| RQ2 run with E1 | **done** 2026-10-06 — audited 2026-10-07, VALID |
| Reconstruction ablation (task B) | **done** 2026-10-07 |
| Zone volumes in mL (task C) | **done** 2026-10-07 — agreement measured |
| ProstateX external validation (task D) | **ready to run** — notebook + script done, needs Drive upload |
| Manuscript update to E1 | open (audited, findings in §6) |

---

## 3. The frozen model — do not change

```
experiment : exp_e1_dice_ce   (E1 = Dice + Cross Entropy, unweighted CE)
epoch      : 88
checkpoint : results/exp_e1_dice_ce/best_model.pt
SHA-256    : 2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2
size       : 93,272,507 bytes   parameters: 7,762,531
```

Identity is confirmed by SHA-256, never by filename. `ce_class_weights is None`
is what distinguishes E1 from E3, whose config is otherwise nearly identical.

---

## 4. Results on record

All standard deviations use **ddof = 0** (population SD), matching
`src/metrics.py`, which produced every validation, baseline, E2 and E3 SD.
`results/test_e1_epoch88/summary.json` and `evaluation_report.md` carry ddof = 1
values from an earlier convention — **cite `metrics.csv` instead**.

### Validation (20 cases) — the basis for model selection

| Arm | Intervention | Best epoch | CG Dice | PZ Dice | Macro |
|---|---|---|---|---|---|
| Baseline | Cross Entropy | 77 | 0.8642 | 0.7375 | 0.8008 |
| **E1 ← selected** | + SoftDice(CG,PZ) | **88** | **0.8662** | **0.7527** | **0.8094** |
| E2 | + augmentation | 70 | 0.8636 | 0.7327 | 0.7982 |
| E3 | + CE class weights | 99 | 0.8652 | 0.7510 | 0.8081 |

Spread across arms is **0.0112**, smaller than any arm's own per-case SD
(0.05–0.07). E1 was selected because it was **highest**, not because it was
shown superior. The only paired test available (E3 vs E1, n=20) gives p = 0.330.

### Held-out test (19 cases) — the RQ1 result

| Metric | CG | PZ | Macro |
|---|---|---|---|
| **Dice** | **0.8442 ± 0.0493** (CI 0.8213–0.8642) | **0.6852 ± 0.1286** (CI 0.6205–0.7377) | **0.7647** |
| IoU | 0.7333 ± 0.0703 | 0.5339 ± 0.1313 | — |
| HD95 (mm) | 6.381 ± 3.109 | 6.538 ± 4.041 | — |
| ASD (mm) | 1.560 ± 0.634 | 1.427 ± 0.808 | — |
| Precision | 0.8646 ± 0.0542 | 0.8297 ± 0.0746 | — |
| Recall | 0.8318 ± 0.0856 | 0.6090 ± 0.1585 | — |

Background 0.9923 ± 0.0036 (reported separately, **not** part of macro Dice).
Per-case macro range 0.5077–0.8392; cases **016** and **018** are PZ outliers.

### E1 vs Baseline on the same 19 test cases

Macro +0.0098 (E1 better in 13/19 cases); PZ Dice +0.0177 (**16/19**); PZ
precision +0.0390. Wilcoxon paired n=19: **W=47, p=0.055**.
→ Write as *consistent in direction*, **never** as statistically significant.

### RQ2 (reconstruction) — E1 epoch 88, `results/rq2_e1_epoch88/`

Run 2026-10-06, forensically audited 2026-10-07: **VALID**.

- Checkpoint SHA-256 re-hashed locally and matches the frozen E1 exactly.
  Every other checkpoint in the tree hashes differently.
- Geometry **19/19 all_ok**; `affine_max_abs_diff = 0.0` exactly (not merely
  within tolerance), verified by re-reading all 19 output NIfTI headers.
- Reconstruction is **exact voxel reassembly** — with `z_resample: false` and
  `crop_pad`, the mask path has *zero* interpolation, so round-trip Dice = 1.0
  is a structural certainty, not a lucky measurement.
- Every aggregate (mean, SD ddof=0, and both bootstrap CI bounds) reproduces
  from `metrics_per_case.csv` with **0.000e+00** discrepancy. No NaN, no
  duplicates, 19 cases × 3 classes.
- Split verified: train 119 / valid 20, mutually disjoint, and neither contains
  any ID in 001–019. No train/test or val/test overlap.

**Segmentation metrics are identical to the RQ1 test, bit-for-bit, by
construction** — both go through `src.evaluate.predict_case_reconstructed`, and
the RQ1 test was already scored in original voxel space. RQ2 is the same
measurement in 3D form, **not** an independent confirmation of RQ1. What RQ2
adds is the geometry validation, the saved spatially valid volumes, and the
fidelity verification. Say this in the manuscript; a reviewer reading both
files will otherwise catch it.

The older `results/exp04_reconstruction/` is **Baseline epoch 77** and must not
be presented as an E1 result. Contamination check found no computational
baseline reference in the E1 artifacts.

**Figures and report are now current.** The 3D projections were regenerated on
2026-10-07 with the corrected aspect ratio (`13e7d3f`); coronal/sagittal panels
are no longer squashed. `report.md` was regenerated locally the same day.

### Reconstruction ablation (task B) — `results/rq2_ablation_naive/`

`scripts/run_reconstruction_ablation.py`, CPU, seconds, no inference, no GT.
Every variant is rebuilt from the same `predictions_2d/*.npz`, so differences
are attributable to the reconstruction stage alone. A correctness gate requires
the locally rebuilt reference to be bit-identical to the saved reconstruction —
it is, for all 19 cases.

| Variant | CG Dice | PZ Dice | Centroid shift | Volume error | Caught by geometry check |
|---|---|---|---|---|---|
| No reorientation | 0.6308 | **0.1802** | 8.2 / 17.3 mm (max 33.4) | 0% | **0/19 — invisible** |
| Naive centre crop | 1.0000 | 1.0000 | 0 mm | 0% | 0/19 |
| Identity affine | 1.0000 | 1.0000 | 183 / 205 mm | **47%** | 19/19 |
| Append order | 1.0000 | 1.0000 | 0 mm | 0% | 0/19 |
| All naive | 0.6308 | 0.1802 | ~185 mm | 47% | 19/19 |

**The headline finding, and the strongest argument RQ2 has:** skipping the
inverse orientation step produces an anatomically **mirrored** volume that
**passes the entire geometry checklist** — shape, affine, spacing, orientation
codes and labels are all correct, because the header is correct and only the
voxel data is wrong. Only the round-trip fidelity test catches it.

→ This is the concrete answer to the supervisor's "Dice 0.8 not 1" concern
(§5.4). A round-trip Dice of 1.0 is not an inflated accuracy claim; it is the
only gate in the pipeline that detects a correct-looking file containing wrong
voxels. Say this in the manuscript.

The complementary failure is the mirror image: the identity affine keeps Dice at
1.0 while misplacing zones by ~180–205 mm and inflating volumes by 47%. Neither
check alone finds both failures. Two shortcuts are harmless here for *different*
reasons — the centre crop structurally (the forward pad is always centred, but
it still cannot invert a forward *crop*), the append order only circumstantially
(this loader happens to emit slices in axial order).

### Zone volumes (task C) — `results/rq2_zone_volumes/`, **prediction-only so far**

`scripts/run_zone_volumes.py`. Volume = voxel count × |det(affine)|, so it is
valid only because the affine is restored exactly — the ablation above measures
what the same voxels yield without it.

Predicted over the 19 test cases: whole gland **47.02 ± 14.02 mL** (24.26–78.86),
CG 34.23 ± 14.28, PZ 12.79 ± 5.19, PZ fraction 0.288 ± 0.121. Plausible for a
cancer-suspicion cohort.

**Agreement against the annotation (measured 2026-10-07):**

| Quantity | GT mean | Bias | % of GT | Under-est. | 95% LoA |
|---|---|---|---|---|---|
| CG | 35.72 mL | −1.48 mL | −4.2% | 12/19 | −10.82 to +7.85 |
| **PZ** | 17.88 mL | **−5.10 mL** | **−28.5%** | **18/19** | −16.53 to +6.34 |
| Whole gland | 53.60 mL | −6.58 mL | −12.3% | 15/19 | −20.18 to +7.02 |

The model **systematically under-segments**, and PZ worst — under-estimated in
18 of 19 cases by nearly a third of its true volume. This is the same defect the
RQ1 metrics show as low PZ recall (0.609) against high PZ precision (0.830),
now expressed in millilitres: what the model does find is right, it just does
not find enough of it.

**The clinical consequence, which is the task-C payoff.** PSA density is
PSA ÷ gland volume, so under-estimating the volume **inflates** PSAD:

- PSAD over-estimated by **14.7% on average**, up to **50%** in the worst case.
- Biased **upward in 15/19 cases** — i.e. toward *more* biopsies, not fewer.
- Against the commonly cited 0.15 ng/mL/cc threshold, a true PSAD of **0.131**
  already reads as ≥ 0.15 at the mean error factor.

That direction matters for how it is written up: the failure mode is
false-positive-leaning, which is the safer direction clinically but still a
real mis-calibration. Report the **limits of agreement**, not the bias — a
−20 to +7 mL spread on a ~54 mL gland is what decides usability for an
individual patient, and it is wide.

---

## 5. Decisions already made — do not reopen

1. **E1 is the RQ1 model.** Selected on validation only;
   `test_metrics_used_for_selection: false` is recorded in the artifacts. Test
   numbers never revisit the selection.
2. **No E4.** Four arms span 0.0112 macro Dice. E3 proved a provably effective
   loss-level intervention moves the PZ operating point without moving overlap,
   so the limit is boundary accuracy, not loss design. The supervisor's
   preference for E4 was discussed; the defensible one would have been E1+E2
   (the unrun 2×2 cell), and it was declined on the evidence above.
3. **Label mapping: `0=background, 1=CG (CZ+TZ), 2=PZ`.** Verified 2026-10-05 by
   multi-planar inspection of cases 020/059/099/139, corroborated by 15/15
   morphometry, the MONAI convention, and the independently annotated ProstateX
   cohort. Record: `results/label_mapping/label_mapping_verified.md`.
   This contradicts the Nakić et al. reading; the image evidence wins.
4. **The "0.8" question is settled.** The supervisor asked that the reported Dice
   be ~0.8 rather than 1.0 "supaya ga overconfidence". There are two different
   quantities: round-trip Dice = 1.0 is a **verification** of the 2D→3D
   transform (no model involved; 0.8 there would mean the reconstruction is
   broken), and the real accuracy is the 3D segmentation Dice (~0.84 CG).
   **Resolution:** keep 1.0 out of any results table; report it as a pass/fail
   verification outcome, and let the segmentation Dice be the RQ2 headline.
   Never put the two in the same table.
5. **Test-set usage history — disclose, never claim pristine.** The official
   19-case set was used on 2026-09-13 (baseline test evaluation) and 2026-09-19
   (RQ2 geometry study), both with Baseline epoch 77 and both before E1/E2/E3
   were designed, then on 2026-10-06 for the E1 test. The supervisor was shown
   the baseline test numbers on 2026-09-21.
6. **Frozen result artifacts are never edited.** Files written before a later
   fix keep their original contents (e.g. pre-verification class names). The
   verification record is the authority for interpreting them.

---

## 6. Open work, in order

### A. Run RQ2 with E1 — **DONE** 2026-10-06, audited 2026-10-07

Notebook: `notebooks/prostate158_rq2_reconstruction_colab.ipynb`. Output in
`results/rq2_e1_epoch88/`. All three gates passed (SHA, case count 19,
single-case sanity). Results and audit findings in §4.

### B. Reconstruction ablation — **DONE** 2026-10-07, results in §4

Run naive reconstruction (stack slices without explicit indices, without inverse
crop/pad, without affine restoration) over the same 19 cases and measure the
damage: Dice drop, geometric shift in mm, zone-volume error in mL.

Cheap: the 2D predictions are already saved as `predictions_2d/*.npz`, so this
varies only the reconstruction stage — CPU, minutes, no re-inference.

**No Colab needed for the core result.** Everything required is already on the
local disk: 19 `predictions_2d/*.npz`, 19 `metadata/meta_*.json` (affine,
spacing, pad/crop parameters, original shape), and the 19 correct
`reconstructions_3d/*.nii.gz` to compare against. Build both volumes from the
*same* `.npz` — one through the correct path, one naive — so what is measured is
purely the cost of the reconstruction stage, with no model error mixed in. That
is a cleaner ablation than comparing to ground truth.

Correctness gate: the "correct" path recomputed locally must equal the saved
`.nii.gz` exactly. If it does not, the ablation is wrong and must not be
reported. The ablation validates itself.

Caveat: metrics *against ground truth* need the test data, which lives on Drive.
The core numbers (geometric damage, naive-vs-correct volume error) need no GT.

**Why it matters:** the manuscript currently *asserts* that naive reconstruction
"is not trivial" without evidence. This converts the central novelty claim from
hygiene ("we validated it") into a finding ("here is the measured cost of not
validating it"). Without it, a reviewer can answer "that is how it should always
be done — what did we learn?"

### B2. Next Colab trip — two notebooks, Run all

Both are generated by `scripts/_build_colab_notebook_rq2_followup.py` (edit the
builder, not the `.ipynb`).

1. **`notebooks/prostate158_rq2_finalize_colab.ipynb`** — **CPU runtime.**
   Zone volumes against the annotation (finishes task C) + redraws the figures
   with the corrected aspect ratio. Never loads the checkpoint, so no
   segmentation result can change. Figures are redrawn from the saved
   `reconstructions_3d/` by `scripts/regenerate_rq2_figures.py` — **no inference
   re-run is needed or wanted.**
2. **`notebooks/prostate158_prostatex_external_colab.ipynb`** — **GPU.** Task D.

Optional later: ablation Dice against ground truth, to state the naive-vs-GT
drop as well as naive-vs-correct.

### C. Zone volumes in mL — **partly done**, GT half needs Colab

From the reconstructed NIfTI vs ground truth, per case. Connects to PSA density
(needs gland volume) and targeted-biopsy planning — this is the *healthcare
impact* JMIR AI asks for, and it answers the supervisor's standing request
(2026-09-22) for a clinical justification of RQ2.

### D. ProstateX external validation — **ready to run**, needs the Drive upload

Script `scripts/run_prostatex_external_validation.py`, notebook
`notebooks/prostate158_prostatex_external_colab.ipynb`. Inference only, reusing
`predict_case_reconstructed` so the path is identical to RQ1/RQ2 rather than a
second implementation that could drift.

**Upload to** `<DRIVE>/dataset/external/prostatex/` preserving the local layout
(`raw/images/t2/`, `raw/masks/pz/`, `raw/masks/rest/`, `metadata/prostatex_manifest.csv`).
790 MB. The manifest is **required**: 15 cases have non-canonical mask stems
(13 three-digit, 2 `VOLUME-` prefixed) that cannot be derived from the case id.
A short directory stops the run — a partial upload is the realistic failure and
would otherwise be scored as a complete cohort.

Verified locally on 2026-10-07, so these are facts rather than assumptions:

- **Cropping is safe.** Prostate158 is 232² against the 442 crop_pad target so it
  is always padded; PROSTATEx reaches 640², where the same transform crops.
  Measured over all 204 cases: **7 are cropped, 0 lose any annotated
  foreground.** Re-checked per run and reported per case.
- **Zones are disjoint** in all 204 cases (0 overlapping voxels), so combining
  `rest→1, pz→2` needs no precedence rule.
- **ProstateX-0086 has a stray voxel of value 2** in its PZ mask. Harmonization
  uses `mask > 0`, never `mask == 1`, which would silently drop it.
- **Scale shift is real and unhandled by design.** The pipeline does no in-plane
  resampling, so a 0.3 mm/px case presents the prostate at ~1.6× the pixel size
  the model trained on (~0.47 mm/px). Recorded per case in
  `cohort_geometry.csv`. Separating scale shift from genuine domain shift would
  need a second run with a resampling adapter — a scientific decision to take
  with the supervisor, not a knob to turn quietly.

**Expect lower numbers — that is a valid finding, not a failure.** The framing is
already fixed in the notebook and the report template, before the numbers exist.
Do not present them as a like-for-like contrast with the Prostate158 test: the
annotators, scanners, geometry and zone definitions all differ.

### E. Manuscript update — audited, see §6 of the audit below

---

## 7. Journal targeting — evidence, not opinion

`InstructionsForAuthorsOfJMIR.docx` in this repo is **only a formatting
template** (heading styles, table captions, reference format). It says nothing
about acceptance standards.

From JMIR's own Focus and Scope page: ML submissions must demonstrate *clinical
maturity with independent dataset validation and clear healthcare impact* —
otherwise they are directed to JMIR AI or JMIR Medical Informatics; and
*"Highly technical papers (with mathematical formulas) are unsuitable for J Med
Internet Res."*

→ **Target JMIR AI**, whose scope explicitly covers image applications.
Submitting to the flagship risks a desk reject or transfer.

Realistic expectation: major revision, then acceptable — *after* B, C and D.
Honest residual weaknesses that none of that fixes: single public dataset,
performance below the dataset paper (Adams CG 0.877 / PZ 0.754), standard
architecture, no reader study.

### Novelty, stated for the paper

> Reconstructing 3D volumes from 2D segmentation is not a trivial step; done
> naively the geometry breaks and reported results mislead. This study measures
> that cost and shows that a pipeline verifying every transformation yields a
> spatially valid 3D representation and clinically usable zone volumes.

2D segmentation is a **component**, reproducibility is a **property** — neither
is the contribution. Do not sell segmentation accuracy; E1 is not SOTA.

---

## 8. Manuscript audit findings (`JurnalJMIR-BenedictAmadeusSandro.docx`)

Writing quality and JMIR structure are fine (structured abstract, ~5030 words,
10 keywords, Ethics / Acknowledgments / COI present). The problem is that
**every Results number is Baseline epoch 77, not E1 epoch 88**.

**CRITICAL**

1. §Methods says *"Training used cross-entropy loss"* and *"best checkpoint at
   epoch 77"* — that is the Baseline. E1 is `1.0 × CE + 1.0 × SoftDice(CG,PZ)`,
   epoch 88.
2. **E1/E2/E3 are never mentioned.** The reader sees "one model, best
   checkpoint, test" instead of four arms with a pre-committed selection rule.
   This discards the most rigorous part of the work *and* a real finding: three
   single-factor interventions produced no improvement beyond noise.
3. Every number in Abstract, §Results, Table 2, §Discussion and §Conclusions
   needs the §4 values above.
4. Representative cases change under the manuscript's own rule (mean foreground
   Dice): highest 014 → **002**, median 007 → **006**, hardest stays **016**
   (but PZ 0.420 → **0.278**). Figure 4 must be regenerated.

**MAJOR** — M1 the "not trivial" claim is unproven (→ task B); M2 no clinical
quantity (→ task C); M3 no external validation (→ task D).

**MINOR** — label mapping verification not claimed in the text (a strength going
unused); test-set usage history not disclosed in the manuscript; no data/code
availability statement though the repo is public; §Limitations says no
significance testing was done, but the Wilcoxon E1-vs-Baseline (p = 0.055) now
exists.

---

## 9. Conventions

- **Commits:** follow `COMMIT_RULES.md`. Author
  `Benedict Amadeus Sandro <criwbasct77@gmail.com>`; **no AI co-author trailer**
  (the repo rule overrides the default). Conventional-commit style. Review
  `git status` / `git diff` before staging; never `git add .`.
- **Never commit:** datasets, `.nii.gz`, `.npz`, checkpoints, Drive data,
  generated figures (except `results/label_mapping/*.png`, which are the
  evidence behind the mapping verification), `.docx`, `.zip`, bundles.
- **Standard deviation:** ddof = 0 everywhere, matching `src/metrics.py`.
- **Macro Dice** = mean of CG and PZ only; background is reported separately.
- Notebooks are generated by `scripts/_build_colab_notebook_*.py` — edit the
  builder, not the `.ipynb`, except for the RQ2 notebook, which has no builder
  and is maintained directly.
- Test suite: `.venv/Scripts/python.exe -m pytest -q tests/` → 371 passed,
  7 skipped. Run it before committing anything that touches `src/`.
- `results/exp04_reconstruction/` (Baseline RQ2) and `results/test_e1_epoch88/`
  (RQ1 test) are **frozen**. New runs go to new directories.

---

## 10. Known rough edges

- `results/exp_e1_dice_ce51026/best_model.pt` hashes to the **same SHA-256** as
  the frozen E1 checkpoint — a bit-identical copy, harmless and incapable of
  causing a wrong-model run, but the directory name invites confusion.
- `results/test_e1_epoch88/summary.json` carries
  `scope_note: "RQ1 only. No reconstruction... is performed in this notebook."`
  That is **factually wrong** — scoring in original voxel space requires the
  inverse transform. The file is frozen (§5.6) so it stays as-is; do not quote
  that sentence in the manuscript.
- `results/exp_e3_dice_ce_pzweight/` contains a **nested duplicate** directory —
  an artifact of extracting a Google Drive zip. The real artifacts are in
  `results/exp_e3_dice_ce_pzweight/exp_e3_dice_ce_pzweight/`. Harmless but
  confusing; `.gitignore` targets the flat paths, so the nested files are
  untracked and unignored.
- A stray `results/exp_e3_dice_ce_pzweight-*.zip` (166 MB) sits in the tree and
  is not gitignored. Do not commit it.
- `results/exp04_rq2_reconstruction/` holds only a *seeded* template
  `report.md`; the real Baseline RQ2 run is in `results/exp04_reconstruction/`.
- `results/exp_e1_dice_ce/plots/.gitkeep` shows as deleted in the working tree —
  pre-existing, unrelated.
- E1's `metrics.csv` and `training_history.json` cover **epochs 70–99 only**; a
  resume overwrote the earlier history. Epoch-88 selection is still correct
  (the running best lives in the checkpoint), and this is already documented in
  `Outputs/E1_supervisor_meeting_notes.md`.

---

## 11. Supervisor's standing requests

| Date | Request | Status |
|---|---|---|
| 2026-09-21 | Continue RQ1 beyond baseline with tuning scenarios | done (E1/E2/E3) |
| 2026-09-22 | Paper references for each tuning scenario | partly — E2/E3 were null results, which vindicated her doubt about them |
| 2026-09-22 | RQ2 needs a scientific explanation tied to medical conditions | **OPEN** → task C |
| 2026-09-22 | Dice "0.8 not 1" | resolved — see §5.4 |
