# Project Handoff — read this first

Living state document for this thesis. A new session reads this before touching
anything: it records what is decided, what is frozen, what is settled, and what
is open, so nothing gets re-derived or re-litigated.

**Last updated:** 2026-10-08 · check `git log -1` for the current HEAD

> **Keep this current.** After any change that alters a result, a decision, or
> what remains open, update the affected section here in the same commit. A
> stale handoff is worse than none: it misleads the next session with
> confidence. If a statement here contradicts an artifact, the artifact wins —
> fix this file.

---

## 1. The thesis

**Title:** *Segmentasi Zona Anatomi pada Citra MRI Pasien Kanker Prostat
menggunakan 2D U-Net dan Rekonstruksi 3D*

| | |
|---|---|
| **RQ1** | Berapa hasil evaluasi segmentasi MRI prostat dengan 2D U-Net? |
| **RQ2** | Bagaimana implementasi rekonstruksi 3D dari hasil segmentasi 2D U-Net? |
| Author | Benedict Amadeus Sandro, Universitas Multimedia Nusantara |
| Supervisor / co-author | Eunike Endariahna Surbakti, S.Kom., M.T.I. (Informatics, FTI UMN) |
| Dataset | Prostate158 — official split 119 train / 20 validation / 19 test |
| External cohort | PROSTATEx — 204 cases, inference only |
| Target journal | **JMIR AI** (not the flagship — see §7) |

**Title and RQs are fixed. Do not reword them.**

### Two separate documents — do not conflate

1. **Journal manuscript** — JMIR format, English + Indonesian. **DONE.** This is
   a task the supervisor asked the student to help with.
2. **Skripsi** — Bab I–V, LaTeX, campus template. **NOT STARTED.** Different
   structure, much longer. Blocked on obtaining the UMN LaTeX template.

---

## 2. Status

| Stage | State |
|---|---|
| Baseline, E1, E2, E3 training | **done** |
| Validation comparison + model selection | **done** — E1 selected |
| E1 frozen | **done** — SHA-256 recorded |
| RQ1 held-out test (19 cases) | **done**, audited, VALID |
| Label-mapping verification | **done** 2026-10-05 |
| RQ2 reconstruction with E1 | **done**, audited, VALID |
| Reconstruction ablation (task B) | **done** |
| Zone volumes + agreement (task C) | **done** |
| External validation, 204 cases (task D) | **done** |
| HD95 tail mechanism | **done** — hypothesis proved |
| Full pre-writing project audit | **done** 2026-10-08 |
| **JMIR manuscript, EN + ID** | **done** 2026-10-08 |
| Supervisor review of the manuscript | **open** — student's action |
| **Skripsi (Bab I–V, LaTeX)** | **open — the real remaining work** |

**No experiments remain.** Every number in the manuscript is derived
programmatically from a result artifact.

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

All standard deviations are **population SD (ddof = 0)**, matching
`src/metrics.py`. `results/test_e1_epoch88/summary.json` carries ddof = 1 values
from an earlier convention — **cite `metrics_per_case.csv` instead**.
Bland–Altman limits of agreement use ddof = 1, the convention for that statistic.

### Validation (20 cases) — the basis for model selection

| Arm | Intervention | Best epoch | CG Dice | PZ Dice | Macro |
|---|---|---|---|---|---|
| Baseline | Cross Entropy | 77 | 0.8642 | 0.7375 | 0.8008 |
| **E1 ← selected** | + SoftDice(CG,PZ) | **88** | **0.8662** | **0.7527** | **0.8094** |
| E2 | + augmentation | 70 | 0.8636 | 0.7327 | 0.7982 |
| E3 | + CE class weights | 99 | 0.8652 | 0.7510 | 0.8081 |

Spread across arms is **0.0112**, smaller than any arm's own per-case SD
(0.0497–0.0661). E1 was selected because it was **highest**, not because it was
shown superior.

All four arms share the same 20 validation cases, so every pair is testable:

| Pair (validation, n=20) | Δ macro | p |
|---|---|---|
| E1 vs Baseline | +0.0086 | 0.0192 |
| E1 vs E2 | +0.0112 | 0.0484 |
| E3 vs E1 | −0.0013 | 0.3300 |
| E2 vs Baseline | −0.0026 | 0.3884 |

**Do not quote these as evidence of superiority.** Validation is the split
selection was performed on, so a test computed there is circular. Descriptive
only; the manuscript labels them as such.

### Held-out test (19 cases) — the RQ1 result

| Metric | CG | PZ |
|---|---|---|
| **Dice** | **0.8442 ± 0.0493** (CI 0.8213–0.8642) | **0.6852 ± 0.1286** (CI 0.6205–0.7377) |
| IoU | 0.7333 ± 0.0703 | 0.5339 ± 0.1313 |
| HD95 (mm) | 6.381 ± 3.109 | 6.538 ± 4.041 |
| ASD (mm) | 1.560 ± 0.634 | 1.427 ± 0.808 |
| Precision | 0.8646 ± 0.0542 | 0.8297 ± 0.0746 |
| Recall | 0.8318 ± 0.0856 | 0.6090 ± 0.1585 |

**Macro 0.7647.** Background 0.9923 ± 0.0036 (reported separately, not in macro).
Per-case macro range 0.5077–0.8392.
Representative cases: good **002**, median **006**, challenging **016**.

PZ combines high precision with low recall — it is **under-segmented**, not
mislocated. This pattern recurs everywhere and is the key qualitative finding.

### E1 vs Baseline, same 19 test cases

| | Δ | p | E1 better in |
|---|---|---|---|
| Macro (pre-specified) | +0.0098 | 0.0546 | 13/19 |
| PZ (**post hoc**) | +0.0177 | **0.0181** | 16/19 |
| CG (**post hoc**) | +0.0019 | 0.4413 | 11/19 |

Macro: write as *consistent in direction*, **never** as statistically
significant. The PZ test was not pre-specified; with three comparisons the
Bonferroni threshold is 0.0167, so report it as exploratory.

### RQ2 reconstruction — `results/rq2_e1_epoch88/`

Geometry **19/19 pass**, `affine_max_abs_diff = 0.0` exactly. Round-trip
fidelity: **PASS** (exact recovery). Reconstruction is exact voxel reassembly —
with `z_resample: false` and `crop_pad`, the mask path has **zero interpolation**,
so round-trip Dice = 1.0 is a structural certainty, not a lucky measurement.

**The segmentation metrics are bit-for-bit identical to the RQ1 test** — both go
through `src.evaluate.predict_case_reconstructed`, and the RQ1 test was already
scored in original voxel space. RQ2 is the same measurement in 3D form, **not an
independent confirmation**. Say so in any write-up.

### Reconstruction ablation (task B) — `results/rq2_ablation_naive/`

Same 2D predictions, only the reconstruction varies. Correctness gate: the
rebuilt reference is bit-identical to the saved run, all 19 cases.

| Shortcut | Reported CG Dice | Reported PZ Dice | Centroid shift | Volume error | Checklist |
|---|---|---|---|---|---|
| *(none — correct)* | 0.8442 | 0.6852 | 0.0 mm | 0% | nothing to detect |
| **No reorientation** | **0.5960** | **0.1758** | 8.2 mm | 0% | **MISSED 0/19** |
| Naive centre crop | 0.8442 | 0.6852 | 0.0 mm | 0% | nothing to detect |
| **Identity affine** | 0.8442 | 0.6852 | **182.8 mm** | **47%** | detected 19/19 |
| Append order | 0.8442 | 0.6852 | 0.0 mm | 0% | nothing to detect |

**The headline finding.** Skipping the inverse orientation step yields an
anatomically **mirrored** volume that **passes the entire geometry checklist** —
header correct, only voxels wrong. The complementary failure, a lost affine,
leaves Dice *untouched* because Dice is computed on voxel indices.

**Neither check alone finds both.** That is the argument for the round-trip
fidelity test, and the answer to the supervisor's "Dice 0.8 not 1" concern (§5.4).

Two shortcuts are harmless for *different* reasons: centre crop is safe
**structurally** (padding is always centred, but it still cannot invert a forward
*crop*); append order is safe only **circumstantially** (this loader happens to
emit slices in axial order).

### Zone volumes (task C) — `results/rq2_zone_volumes/`

| Quantity | GT mean | Bias | % of GT | 95% LoA |
|---|---|---|---|---|
| CG | 35.72 mL | −1.48 | −4.2% | −10.82 … +7.85 |
| **PZ** | 17.88 mL | **−5.10** | **−28.5%** | −16.53 … +6.34 |
| Whole gland | 53.60 mL | −6.58 | −12.3% | −20.18 … +7.02 |

Systematic **under-segmentation**; PZ worst (18/19 cases). This is the low PZ
recall expressed in millilitres.

**Clinical consequence.** PSA density = PSA ÷ gland volume, so under-estimating
the denominator inflates it: **+14.7% on average**, up to **+50%**, biased upward
in **15/19** cases. Against the 0.15 ng/mL/cc threshold, a true density of
**0.131** already reads as 0.15. Direction is toward *more* biopsies — safer
clinically, but a real mis-calibration. **Report the limits of agreement, not the
bias** — ±~14 mL on a ~54 mL gland is what matters for an individual patient.

### External validation (task D) — `results/external_prostatex_e1_v2/`

204 PROSTATEx cases, inference only, frozen E1.

| | CG | PZ |
|---|---|---|
| Dice | 0.8137 ± 0.0935 | 0.6247 ± 0.1460 |
| Recall | 0.8229 | **0.5076** |
| Precision | 0.8193 | 0.8555 |

**Macro 0.7192** — a drop of only **0.0455** from 0.7647 internally, on a fully
independent cohort. The PZ pattern intensifies: recall falls, precision rises —
the model gets *more conservative* out of domain.

**Geometry: 204/204 pass, affine diff 0.0.** Orientation **LAS** (vs LPS
internally); **7 cases exercise the *crop* branch** the development data never
triggers; **0 lose annotated anatomy**. With the 19 internal cases, the
reconstruction is verified on **223 examinations across two orientation
conventions and both branches of the in-plane transform.**

**Zone definitions are not identical.** The external "rest" mask is defined by
its authors as transition zone + central zone + **anterior fibromuscular
stroma**, while Prostate158 CG is transition + central zone only. The external
reference therefore includes a structure the model was never trained to label,
so part of the gap is a difference in *what was annotated*. Stated in the
manuscript.

**Licence:** external zonal annotations are **CC BY 4.0** (reuse with
attribution). Underlying imaging is from the TCIA public archive.

### HD95 tail — mechanism proved — `results/external_hd95_tail/`

Central-gland HD95 is heavy-tailed: median 9.0 mm, mean 20.2 mm, max 81.0 mm,
64/204 cases above 30 mm, while Dice stays high. Connected-component analysis
(on the saved volumes, no inference) **confirms the cause**:

| | All | Tail (HD95 > 30) | Rest |
|---|---|---|---|
| HD95 | 20.2 mm | 44.5 mm | 9.1 mm |
| HD95, largest component only | **6.5 mm** | **6.8 mm** | 6.3 mm |
| Dice | 0.8137 | 0.7770 | 0.8305 |
| Dice, largest component only | 0.8266 | 0.8108 | 0.8339 |
| Cases with >1 component | 86% | **100%** | 80% |
| Satellite volume share | — | 8.3% | 1.4% |
| Distance to farthest satellite | — | 73 mm | 50 mm |

Small, distant false-positive components: too little volume to move Dice, far
enough to dominate a boundary metric. Consistent with the **1.76× wider field of
view** (192 mm vs 108.8 mm) exposing anatomy the model never trained on.

**Deliberately NOT applied.** Keeping only the largest component would change
the method, and choosing it after seeing external scores is tuning on the
external cohort. Reported numbers are the unmodified pipeline.

---

## 5. Decisions already made — do not reopen

1. **E1 is the RQ1 model.** Selected on validation only;
   `test_metrics_used_for_selection: false` is recorded. Test numbers never
   revisit the selection.
2. **No E4.** Four arms span 0.0112 macro Dice. E3 proved a provably effective
   loss-level intervention moves the PZ operating point without moving overlap,
   so the limit is boundary accuracy, not loss design. The supervisor's
   preference for E4 was discussed; the defensible one would have been E1+E2
   (the unrun 2×2 cell), and it was declined on the evidence above.
3. **Label mapping: `0=background, 1=CG (CZ+TZ), 2=PZ`.** Verified 2026-10-05 by
   multi-planar inspection of cases 020/059/099/139, corroborated by morphometry
   and the MONAI convention. Record:
   `results/label_mapping/label_mapping_verified.md`. This contradicts the Nakić
   et al. reading; the image evidence wins.
4. **The "0.8" question is settled.** WhatsApp, 22/09/2026: the student asked
   which Dice she meant; Bu Eunike answered *"Di round trip dice nyaa yang 1,0
   diganti ke 0,8 yaaa"* — the **round-trip Dice**, not millimetres. The "0.8 mm"
   phrasing that circulated was a corruption originating in an older session's
   scope list; a repo-wide grep finds **zero** occurrences of 0.8 mm anywhere,
   and no output volume has a 0.8 mm axis.
   **Resolution:** round-trip Dice = 1.0 is a mathematical certainty for a
   lossless transform and cannot be "changed to 0.8" without falsifying. It is
   reported as a **pass/fail verification**, kept out of every results table. The
   number in the results table is the real 3D segmentation Dice — which *is*
   ~0.8 (macro 0.7647). Her instinct was right; two different quantities were
   both called "Dice". The ablation now proves the round-trip test is the only
   gate that catches a mirrored volume.
5. **Test-set usage history — disclose, never claim pristine.** The official
   19-case set was used on 2026-09-13 (baseline test) and 2026-09-19 (RQ2
   geometry study), both with Baseline epoch 77 and both before E1/E2/E3 were
   designed, then on 2026-10-06 for the E1 test. The supervisor was shown the
   baseline test numbers on 2026-09-21. Disclosed in the manuscript's Limitations.
6. **Frozen result artifacts are never edited.** `results/exp04_reconstruction/`
   and `results/test_e1_epoch88/` keep their original contents. The verification
   record is the authority for interpreting them.
7. **GPU vs CPU reruns differ at float precision only.** The external run was
   repeated locally on CPU: Dice differs in 6/612 rows, max 1.0e-05; HD95 not at
   all; cohort means agree to 7 decimals. Cite `..._v2` (it has the saved volumes
   and geometry records).

---

## 6. Open work

### A. Skripsi (Bab I–V, LaTeX) — **the real remaining work**

Not started. Different document from the journal: different structure, much
longer, campus template, written in LaTeX.

**BLOCKED:** needs the UMN LaTeX thesis template (`.tex`/`.cls`) or the written
formatting guideline. Do not guess the chapter structure — guessing means
rework.

All the material exists. The work is restructuring, not new research.

### B. Supervisor review of the manuscript — student's action

### C. Optional, not blocking

- **Manuscript length.** English is ~7.7k words against JMIR's typical
  3,000–6,000 (instructions say "no rigorous restrictions"). ~800 words could
  come out of Limitations and Comparison With Prior Work without losing a finding.
- **Anterior-stroma quantification.** The external CG reference includes anterior
  fibromuscular stroma; comparing anterior vs posterior CG error would measure
  how much of the external gap is annotation difference rather than model
  failure. ~20 min, local.
- **Disk.** ~960 MB of duplication remains and is safe to delete (sources exist):
  `results/external_prostatex_e1/.../\_harmonized/cases/*/t2.nii.gz` (779 MB,
  duplicates `dataset/external/prostatex`) and `results/exp_e1_dice_ce51026/`
  (178 MB, bit-identical checkpoint copy). Awaiting the student's confirmation.

---

## 7. Journal targeting

`InstructionsForAuthorsOfJMIR.docx` is **only a formatting template**. From
JMIR's Focus and Scope: ML submissions must demonstrate *clinical maturity with
independent dataset validation and clear healthcare impact*, otherwise they are
directed to JMIR AI or JMIR Medical Informatics; and *"Highly technical papers
(with mathematical formulas) are unsuitable for J Med Internet Res."*

→ **Target JMIR AI.** Submitting to the flagship risks a desk reject or transfer.

Requirements verified against the template and met: portrait US Letter, 1.25"
side margins, 6.00" text column; structured abstract ≤450 words; 3–10 keywords;
references numbered in citation order with in-text `[n]` markers; figures and
tables referenced in the body.

### Contribution, as stated in the paper

Not a new architecture, not SOTA, and not "a reproducible pipeline" —
reproducibility is a property, not a finding. The contributions are:

1. **Empirical comparison of four single-factor configurations** under a
   pre-specified selection rule, including the **negative result** that three
   interventions moved performance by less than between-patient variation.
2. **A measured demonstration that geometric validation alone is insufficient**
   for 2D→3D reconstruction: a mirrored volume passes the whole checklist, a
   lost affine is invisible to Dice, and neither check alone finds both — with
   the cost of each omitted step quantified in Dice, millimetres and millilitres,
   and carried through to PSA density.

Honest residual weaknesses: single development dataset, standard architecture,
performance below the dataset paper (Adams CG 0.877 / PZ 0.754), no reader study.

---

## 8. The manuscript — how it is built

Both language versions are **generated, not hand-edited**:

```
scripts/manuscript_numbers.py        every value, loaded from result artifacts
scripts/manuscript_content_en.py     English text + the shared REFERENCES library
scripts/manuscript_content_id.py     Indonesian text
scripts/build_manuscript.py          renders either into .docx
scripts/make_rq2_figures.py          figures F0–F5
scripts/analyse_external_hd95_tail.py  the tail diagnostic
```

```bash
python scripts/build_manuscript.py --lang en
python scripts/build_manuscript.py --lang id --template <a clean copy of the docx>
```

**Edit the content module, never the .docx** — a rebuild overwrites it.

- Citations are **keyed**: write `[@bardis]` or `[@litjens,clark]`; the builder
  numbers them in order of first appearance and emits only cited entries. A
  guard fails the build if a content module still has hard-coded `[n]`.
- Indonesian decimals are converted to commas at build time.
- Document properties are reset to the author alone.
- Media parts the rebuild no longer references are stripped (the old draft left
  2.5 MB of stale figures behind).
- The Indonesian build needs a `--template` pointing at a *clean* docx, because
  it would otherwise use the English output as its template.

Current state: EN ~7.7k words, ID ~7.0k; abstracts 449 / 418; 20 references;
9 figures; 6 tables; portrait 8.5×11".

**No AI/tool attribution anywhere** — not in the text, not in document metadata.
This is a hard requirement.

---

## 9. Conventions

- **Commits:** follow `COMMIT_RULES.md`. Author
  `Benedict Amadeus Sandro <criwbasct77@gmail.com>`; **no AI co-author trailer**.
  Conventional-commit style. Review `git status` / `git diff` before staging;
  never `git add .`. Never push without explicit authorization.
- **Never commit:** datasets, `.nii.gz`, `.npz`, checkpoints, Drive data,
  generated figures (except `results/label_mapping/*.png`), `.docx`, `.zip`,
  bundles.
- **Standard deviation:** ddof = 0 everywhere (ddof = 1 only for Bland–Altman LoA).
- **Macro Dice** = mean of CG and PZ only; background reported separately.
- Notebooks are generated by `scripts/_build_colab_notebook_*.py` — edit the
  builder, not the `.ipynb`.
- Test suite: `.venv/Scripts/python.exe -m pytest -q tests/` → **438 passed,
  7 skipped**. Run before committing anything touching `src/` or `scripts/`.
- Frozen result dirs: `results/exp04_reconstruction/`, `results/test_e1_epoch88/`.
  New runs go to new directories.

### Where the data lives

**Google Drive is mounted locally** at `G:\My Drive\THESIS_PROSTATE158` (Drive
for Desktop). It holds the 19-case test set
(`dataset/test/extracted/prostate158_test/test`), the 204-case ProstateX cohort,
and every `results/` directory.

- Python needs the Windows form `G:/My Drive/...`; the Git Bash `/g/My Drive/...`
  form works for `ls` but `nibabel` and `open()` reject it.
- First read of a file streams from the cloud (~8 s for 2 MB), cached after. A
  93 MB checkpoint takes minutes — prefer the repo copies.

**Division of work:** CPU work runs locally and the result is handed back. Only
GPU work ships as a Colab notebook. A *subset* of a GPU job often runs fine on
CPU (inference on 15 cases to test a hypothesis rather than all 204), so prefer
a targeted local run over a Colab round trip.

---

## 10. Known rough edges

- `results/exp_e1_dice_ce51026/best_model.pt` hashes to the **same SHA-256** as
  the frozen E1 checkpoint — a bit-identical copy. Harmless, confusing name.
- `results/test_e1_epoch88/summary.json` carries
  `scope_note: "RQ1 only. No reconstruction... is performed in this notebook."`
  That is **factually wrong** — scoring in original voxel space requires the
  inverse transform. The file is frozen (§5.6); do not quote that sentence.
- `results/exp_e3_dice_ce_pzweight/` contains a **nested duplicate** directory
  from a Drive zip extraction. Real artifacts are in the inner directory.
- `results/exp04_rq2_reconstruction/` holds only a *seeded* template `report.md`;
  the real Baseline RQ2 run is in `results/exp04_reconstruction/`.
- `results/exp_e1_dice_ce/plots/.gitkeep` shows as deleted in the working tree —
  pre-existing, unrelated.
- E1's `metrics.csv` and `training_history.json` cover **epochs 70–99 only**; a
  resume overwrote the earlier history. Epoch-88 selection is still correct (the
  running best lives in the checkpoint).
- The `results/` directories are untracked but **not** gitignored.

---

## 11. Supervisor's standing requests

| Date | Request | Status |
|---|---|---|
| 2026-09-21 | Continue RQ1 beyond baseline with tuning scenarios | **done** (E1/E2/E3) |
| 2026-09-22 | Paper references for each tuning scenario | **done** — E2/E3 were null results, which vindicated her doubt |
| 2026-09-22 | RQ2 needs a scientific explanation tied to medical conditions | **done** → zone volumes → PSA density |
| 2026-09-22 | Dice "0.8 not 1" | **done** — see §5.4 |
