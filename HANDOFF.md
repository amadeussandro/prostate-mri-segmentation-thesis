# Project Handoff — read this first

Living state document for this thesis. A new session should read this before
touching anything; it records what is decided, what is frozen, and what is open,
so settled questions are not re-litigated.

**Last updated:** 2026-10-07 · **HEAD at writing:** `e61ff44`

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
| RQ2 run with E1 | **OPEN — next action** |
| Reconstruction ablation, zone volumes | open |
| ProstateX external validation | open (data ready, 204 cases) |
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

### RQ2 (reconstruction) — Baseline epoch 77 only, awaiting E1 re-run

Geometry 19/19 all_ok, affine max diff 0.0. Round-trip Dice = 1.0 exactly.
Existing artifacts in `results/exp04_reconstruction/` are **Baseline-derived**
and must not be presented as E1 results.

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

### A. Run RQ2 with E1 — **next action, needs Colab + Drive**

Notebook: `notebooks/prostate158_rq2_reconstruction_colab.ipynb` → Run all.
Output lands in `<DRIVE>/results/rq2_e1_epoch88/`. Three gates stop it on a SHA
mismatch, a case count ≠ 19, or a failed single-case sanity check.

### B. Reconstruction ablation — **the highest-value remaining experiment**

Run naive reconstruction (stack slices without explicit indices, without inverse
crop/pad, without affine restoration) over the same 19 cases and measure the
damage: Dice drop, geometric shift in mm, zone-volume error in mL.

Cheap: the 2D predictions are already saved as `predictions_2d/*.npz`, so this
varies only the reconstruction stage — CPU, minutes, no re-inference.

**Why it matters:** the manuscript currently *asserts* that naive reconstruction
"is not trivial" without evidence. This converts the central novelty claim from
hygiene ("we validated it") into a finding ("here is the measured cost of not
validating it"). Without it, a reviewer can answer "that is how it should always
be done — what did we learn?"

### C. Zone volumes in mL

From the reconstructed NIfTI vs ground truth, per case. Connects to PSA density
(needs gland volume) and targeted-biopsy planning — this is the *healthcare
impact* JMIR AI asks for, and it answers the supervisor's standing request
(2026-09-22) for a clinical justification of RQ2.

### D. ProstateX external validation

204 cases, already acquired, audited, spatially verified, mapping harmonized
(`dataset/external/prostatex/`). Directly answers JMIR's stated requirement for
independent dataset validation. Needs an inference adapter for domain shift
(in-plane 0.3–0.6 mm vs Prostate158). **Expect lower numbers — that is a valid
finding, not a failure.** Frame it before running.

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
