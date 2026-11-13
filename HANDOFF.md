# Project Handoff — read this first

Living state document for this thesis. A new session reads this before touching
anything: it records what is decided, what is frozen, what is settled, and what
is open, so nothing gets re-derived or re-litigated.

**Last updated:** 2026-11-11 (§0 added; §1–§11 below are older and describe the
thesis and the single-seed model) · check `git log -1` for the current HEAD

> **Keep this current.** After any change that alters a result, a decision, or
> what remains open, update the affected section here in the same commit. A
> stale handoff is worse than none: it misleads the next session with
> confidence. If a statement here contradicts an artifact, the artifact wins —
> fix this file.

---

## 0. JMIR revision v7 and the final model — state on 2026-11-11

**Two models now exist. Do not mix them up.**

- **Single-seed model** (E1, seed 42, epoch 88, SHA `2366e3c6…d5d2`): used by the
  thesis (Bab I–V) and by every result folder named `*_e1_*`, `rq2_*`,
  `jmir_stats`, `jmir_reader2`. Its NCI-ISBI 2013 run is the **confirmatory**
  test of the connected-component rule.
- **Final model** (JMIR v7): softmax ensemble of E1 seeds 42/43/44 (2D) with
  left-right flip averaging, chosen by `results/seed_25d_prespecification.md`
  (SHA on LF bytes `83ba8e59…26f8`) and recorded in
  `results/e1_seed_selection/selection.json` (validation only,
  `test_split_read: false`). 2.5D input was not adopted (−0.0008 < 0.005).
  Member SHAs are in the README and in Table S6 of Appendix 3.

Where the final-model results live: `results/final_ensemble_test`,
`final_ensemble_ablation`, `final_ensemble_zone_volumes`, `final_ensemble_stats`,
`final_ensemble_reader2`, `external_prostatex_final`, `final_external_hd95_tail`,
`final_external_anterior_stroma`, `final_gland_size_effect`,
`external_nci_isbi_final`, `external_overlap_final`, figures in
`results/figures_rq2_final` and `results/figures_jmir_final`.
`scripts/manuscript_numbers_jmir.py` (`MODELS`) maps both models to their folders.

Headline (final model, sample SD): held-out CG 0.862, PZ 0.710; +0.022 macro
Dice over the single-seed model (P=.005); 98.8% / 95.1% of inter-reader Dice,
no significant gap. PROSTATEx CG 0.827, PZ 0.646. NCI-ISBI CG 0.806, PZ 0.616.

**NCI-ISBI repeat with the final model:** R1 passed the internal check and was
applied (CG HD95 19.2 → 8.6 mm). **R2 failed** the internal check (validation PZ
HD95 +0.4 mm), so it was not applied, as the LCC prespecification requires. The
manuscript text and Appendix 2 say so, and their guards fail the build if this
changes.

**Literature audit (v9):** `results/literature_audit/` holds a protocol written
before screening (its LF SHA-256 is recorded in `search.json`), the PMC search
(57 open-access records, run 2026-11-11), `screening.csv` (22 included, reasons
for every exclusion), `extraction.csv` (10 items per study, with quotations and
flags) and `report.md`. Result: 0/22 described reassembly, 0/13 that resampled
or cropped described undoing it, 0/22 reported a geometry check (95% CI
0–15.4%). `manuscript_numbers_jmir.literature_audit()` loads it and checks the
protocol hash; the text sits in Methods, Results, Discussion, Limitations and
the abstract conclusion, guarded. Multimedia Appendix 4 is built from it. Full
texts (`fulltext/`) are gitignored; `literature_audit.py search` refetches them.

Built files (Journal-JMIR folder), latest first:
`JurnalJMIR-BenedictAmadeusSandro-{EN,ID}-JMIRrevision-111126-v9.docx` and
`JMIR_submission_111126-v9/` (adds the audit and Appendix 4); v8 and v7 kept.
Rebuild with `build_manuscript.py --jmir [--lang id] --out <new file>` and
`build_jmir_submission_pack.py --tag <new tag> --template <path>`; never
overwrite an older version.

Still open: Zenodo DOI (`[[TO FILL: Zenodo DOI]]`, after the final push and
release); cover-letter placeholders (date, thesis status and link,
confirmations); CLAIM form transfer; supervisor review. Nothing from this
work is committed yet.

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
2. **Skripsi** — Bab I–V, LaTeX, campus template. **DRAFTED 2026-10-15.** The
   template arrived, so this is no longer blocked. Lives OUTSIDE this repo at
   `D:\1.AmadeusKuliah\Skripsi\2611_Skripsi_Sandro_103840`. Not yet compiled.

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
| **JMIR manuscript, EN + ID** | **done**, rebuilt 2026-10-15 |
| Anterior-stroma test (refutes a manuscript claim) | **done** 2026-10-15 |
| Supervisor review of the manuscript | **open** — student's action |
| **Skripsi (Bab I–V, LaTeX)** | **drafted** 2026-10-15 — needs compiling |

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

### Gland size — accuracy is NOT consistent — `results/gland_size_effect/`

Asked by the supervisor: does a smaller reference structure get segmented
worse, and is the model consistent? **It is not, and the effect is large.**

| Cohort | Zone | Spearman r | p |
|---|---|---|---|
| **External (n=204)** | **CG** | **+0.766** | <1e-39 |
| External | PZ | +0.279 | 0.0001 |
| Internal (n=19) | CG | +0.449 | 0.054 |

External CG Dice by size tertile: **0.7287** (4.6—26.2 mL) / 0.8372 /
**0.8754** (51.6—252.1 mL) — a **0.147 Dice** gap. The internal cohort agrees
in direction but 19 cases leave it underpowered, so the external cohort carries
the finding.

**The compounding follow-on was tested and does NOT hold.** PSA density divides
by gland volume, so if the relative volume error also grew as the gland shrank
the two would multiply. Correlation of reference volume against absolute
percentage volume error: **+0.088 (p=0.72)**, tertiles non-monotone
(13.9% / 7.8% / 15.9%). Recorded as a negative sub-result. With n=19 the honest
statement is that no evidence was found, not that no effect exists.

This is **a limitation, not a strength**, and both documents report it as one:
a small gland is the normal finding in younger men and in those without benign
enlargement, so it bounds where the model can be trusted.

---

## Supervisor review of the manuscript — acted on 2026-11-04

Done: the 2D U-Net choice justified explicitly; a comparison table against the
dataset's reference implementation (96.3% of its CG Dice, 90.9% of its PZ Dice,
with the architectural reason); nine references added with CrossRef metadata
(five JMIR-family, CLAIM 2024, TRIPOD+AI, two surveys) bringing both documents
to **29 cited**; an argument for why those guidelines do not close this gap
(they ask for disclosure, not for a check that compares voxels); ORCID
0009-0000-1459-9139 on the corresponding-author line.

Skripsi-only, deliberately kept out of the journal: hardware specification, the
VS Code — Colab workflow and why it exists (a 4 GB GTX 1050 cannot train these
four configurations at a comparable batch size), and three code listings taken
from the real source.

**Still open from that review:** a preprocessing figure, redrawing the study-flow
figure, an audit that every figure and table is discussed in the text rather
than merely referenced, trimming narrative in favour of result description, and
studying JMIR's house structure to match it.

**Length is now a real problem.** EN body prose is ~7.2k words against JMIR's
typical 3,000—6,000. The additions above are substance, not padding, so the trim
has to come from elsewhere — and the supervisor also asked for less narrative,
which points the same way.

---

### Anterior stroma — hypothesis REFUTED — `results/external_anterior_stroma/`

The manuscript used to imply that part of the external CG gap came from the
external reference including anterior fibromuscular stroma. That was tested
with the internal cohort as the control, and **it does not hold.**

| CG Dice | External | Internal | Gap |
|---|---|---|---|
| Whole gland (as reported) | 0.8137 | 0.8442 | −0.0304 |
| **Posterior half** | 0.7895 | 0.8516 | **−0.0621** |
| **Anterior half** | 0.8307 | 0.8366 | **−0.0059** |

**91% of the gap sits posteriorly, 9% anteriorly.** The anterior half — exactly
where the stroma is — is where the cohorts agree *best*. The anterior deficit is
*reversed* between cohorts (external −0.0436, internal +0.0465, p = 0.0020).
Robust to the dividing plane: the whole-gland plane gives −0.0677 / −0.0071.

Gates passed: the anterior axis is read from each volume's affine and its sign
checked anatomically (PZ centroid posterior to CG centroid) — **223/223 pass**;
and the script independently reproduces CG Dice 0.8442 internally and 0.8137
externally from the raw volumes.

What the gap actually is: reference CG left as **background** at the posterior
extent of the gland (23.2% of posterior misses go to background vs 4.3% to PZ)
— the whole-gland boundary falling short out of domain, not a missing structure.

**The differing zone definitions remain a real fact about the datasets** and the
manuscript still states it. What it no longer claims is that the difference
explains the shortfall. No reported number changed.

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

### A. Skripsi (Bab I–V, LaTeX) — **drafted, not compiled**

**No longer blocked.** The UMN template arrived and the thesis is drafted.

Location — **outside this repo**: `D:\1.AmadeusKuliah\Skripsi\2611_Skripsi_Sandro_103840`

- Entry point `SKRIPSI.tex`; chapter titles come from `settings.tex`; BAB II is
  named "Landasan Teori" by the template, with prior work as a section inside.
- Bab1–Bab5 written, plus `07a-Abstrak.tex` and `07b-Abstract.tex`. As of
  2026-11-05: **22 research figures and 16 tables** (was 10 and 13), see
  "Expanded 2026-11-05" below.
- 20 references appended to the root `pustaka.bib` with the same citation keys
  as the journal; the template's own 10 example entries were left in place.
- Figures copied into `assets/pics/` as `fig_*.png`.
- Map of the template and its styling rules: `claude/Thesis_Latex_Template_Guide.md` (local notes, not in the repository).

Verified statically (no LaTeX installed on this machine): every `\ref` resolves,
no duplicate labels, all 9 figures exist, all 20 `\cite` keys exist, braces
balance, `\addequation` count matches the equation environments. **29 headline
numbers cross-checked against the result artifacts.**

First compile 2026-10-15 produced 66 pages, and it was audited against the
PDF itself — glyph positions read with pdfplumber and compared to the
14.00 cm text block, rather than estimated from the source. Findings and the
remaining to-do list: `claude/Skripsi_PDF_Audit_TODO.md` (local notes, not in the repository).

Fixed since that compile:

- NIM filled (00000103840).
- An unescaped underscore in a DOI, which raises "Missing $ inserted" the
  moment the bibliography is typeset.
- Author lists cut to IEEE's six-plus-et-al. The MONAI entry carried 56 names
  on one 1080-character line; that is both wrong for the style and the shape
  of input that exhausts BibTeX while IEEEtran formats each name.
- The template's author-facing note after `\bibliography` commented out, so
  it stops printing on the DAFTAR PUSTAKA page of the submitted document.
- The three tables that overran the right margin: 4.5 by 2.90 cm, 4.6 by
  1.63 cm, 4.1 by 0.29 cm. Headers moved into the captions, type size and
  column separation reduced. The other seven were measured and left alone.

#### Uploading to Overleaf — the student does it by hand from a local package

**Do not drive Overleaf through the browser** — the student asked for local
work only (2026-11-06); browser upload was too slow.

`scripts/sync_overleaf_folder.py` mirrors the complete thesis into
`D:\1.AmadeusKuliah\Skripsi\SKRIPSI-Overleaf` and writes
`SKRIPSI-Overleaf.zip` (74 files at the zip root, so Overleaf's
New Project > Upload Project builds a working project in one step). It refuses
to write the zip if any `\input`/`\includegraphics` would not resolve or if
there is not exactly one `\documentclass`.

On 2026-11-06 the student emptied the old Overleaf project to start over
(only two stray folders remain; its main document is set to a non-existent
`main.tex`). Their personal pages there had been filled in on Overleaf and now
survive only in that project's **History**:

- `05-Persembahan.tex` — **restored locally**, byte-identical to the Overleaf
  version (verified by hash). Contains the student's motto.
- `02b-HalamanAI.tex`, `03a-Persetujuan.tex`, `04-Publikasi.tex` — local copies
  are **still the template**. The Overleaf entries could not be read back
  (02b had changed again since it was last seen); the student must restore
  them from History or fill them in again. The README in the package says so.

Two Overleaf-side edits were merged into the source before the project was
emptied, and hashing proved they were the only other differences: `\small` +
`\setlength{\tabcolsep}{4pt}` on `tab:terdahulu` in Bab2, and `[!ht]` on the
reconstruction figure in Bab3. `BACKUP_ORI_TEMPLATE_UMN` existed only on
Overleaf and is likewise in its History.

#### Expanded 2026-11-05 — more results, not more padding

The project had computed far more than the thesis showed, and RQ1 — the
four-arm comparison — carried no figure at all. `scripts/make_skripsi_extra.py`
builds everything below from frozen artifacts and writes it into the thesis
folder; the new tables are generated LaTeX in `src/_generated/` and must not
be edited by hand. Backups of the two edited files are in
`_cadangan_20261105_152905/` inside the thesis folder.

Added to Bab 4: training curves and epoch selection (new §4.1 subsections),
the RQ1 overview and per-case spread, the E3 mechanism, per-case test results,
3D projections of the three representative cases, the external cohort
geometry table. Lampiran: the template's example table, figure and VBScript
listing are gone, replaced by per-case test metrics, full external metrics,
and three RQ1 figures. The two **mandatory** appendices (Turnitin, Formulir
Bimbingan) still hold the template's sample files — the student must overwrite
`fig_individual_source_sample_turnitin.png` and
`Form_Bimbingan_atau_Konsultasi_Skripsi.pdf` with their own.

Findings the new material surfaced, all now stated in the text:

- **The selection criterion is batch-level, the reported metric case-level.**
  `src/train.py:1586` selects on `val_metrics["dice"]`, a batch-level 2D
  proxy. That is why the curves sit near 0.68 while Table 4.1 reports 0.8094
  for the same 20 cases. Unexplained, it reads as an inconsistency.
- **Epoch 81 beats epoch 88 on the batch-level macro** (0.7012 vs 0.6963).
  Epoch 88 is the maximum of the criterion actually used, fixed in code before
  training. Whether 81 would win at case level **cannot be tested**: only
  `best_model.pt` and `latest_checkpoint.pt` were kept.
- **E3's failure is mechanistic.** Against E1, PZ recall rose in 20/20
  validation cases and precision fell in 20/20; Dice improved in 8/20. The
  weighting moved the operating point, not the error. E2 has no equivalent
  explanation, and the text says so.
- **Case 077 is the hardest validation case in all four arms.**
- **The thesis had shown only the two extremes.** Text announced case 006 as
  the median and then showed only 002 and 016. All three now appear in 3D.
- **The 3D projections expose slice-to-slice inconsistency** — horizontal bands
  in coronal/sagittal views, a consequence of a 2D model. Added to Keterbatasan.
- The projections are **silhouettes with PZ painted over CG**, in array
  orientation (posterior up). The captions and text say how to read them; a
  silhouette area is not a volume and is never interpreted as one.

Seven `make_rq1_figures.py` defects fixed at source: labels struck through by
their own error bars, a 2x2 panel layout collapsed into the corners by a
misplaced legend anchor, HD95 whiskers below 0 mm (now clipped, and labelled
so), legends over data, a mean label crossed by a curve, two test results drawn
on top of each other. These figures were unused by the journal, which is why
fixing them changed nothing there.

Verified: every `\ref` resolves, no duplicate labels; **every decimal in Bab 4
and the Lampiran prose traced to an artifact** (21 that the journal does not
print were checked one by one against the loaders and result CSVs).
**Not yet compiled** — no LaTeX on this machine.

`SKRIPSI-Overleaf.zip` is built by `scratchpad/make_overleaf_zip.py`, which
refuses to package a tree that still carries placeholders, an unescaped DOI
underscore, an over-long author list, active template guidance, or more than
one `\documentclass`. The template's `artikel_ilmiah.tex` is deliberately left
out: Overleaf otherwise picks it as the main document and builds a 3-page
IEEE sample instead of the thesis, which is what happened on the first upload.

### The bibliography no longer goes through BibTeX — decided 2026-10-15

The BibTeX route printed the DAFTAR PUSTAKA heading with nothing under it and
left all 23 citations as `[?]`. The heading and its TOC entry only appear when a
`thebibliography` environment is actually processed, so BibTeX had run and emitted
an **empty** list — which is what happens when none of the cited keys are found
in the database it read.

Cause: `\bibliography{pustaka}` resolves by **basename**, and the template ships a
second `pustaka.bib` at `src/3.Akhir/` holding three unrelated entries. Whichever
one the search reaches first wins.

Rather than keep betting on search order, the bibliography is now plain LaTeX:

- `src/3.Akhir/daftarpustaka.tex` holds a `thebibliography` with 20 `\bibitem`,
  generated by `scratchpad/gen_daftarpustaka.py` from `pustaka.bib`. Numbering
  follows IEEE — order of first citation — computed from the sources in the
  order `SKRIPSI.tex` inputs them, never hand-maintained.
- `\bibliographystyle` and `\bibliography` are commented in `SKRIPSI.tex`, with a
  note on how to switch back. Nothing now depends on a BibTeX pass, a `.bst`,
  or a basename lookup.
- `pustaka.bib` stays as the machine-readable record. **To add a reference: edit
  `pustaka.bib`, then re-run the generator.** Do not hand-edit
  `daftarpustaka.tex`.

Two bib defects were fixed on the way: an unescaped underscore in a DOI, and
author lists past IEEE's six-plus-et-al. (the MONAI entry had 56 names on one
1080-character line, which is also what exhausts BibTeX while IEEEtran formats
names).

**Open:** confirm on the next compile that citations print as `[1]`-`[20]`.
Earlier state, for reference: 23 citations
printed as `[?]` and DAFTAR PUSTAKA held no entries, which means no
usable `.bbl` ever reached the document. The two bib defects above
are fixed, but they do not by themselves explain it: the DOI one sits on the
fifth cited entry, so the first four would still have printed. It needs a
BibTeX pass that actually runs; on Overleaf, **Recompile from scratch** also
clears a cached broken `.bbl`.

Still to fill, and not guessable: `\pembimbingNIDN`, the
administrative dates, Kata Pengantar, the persembahan/moto page, the
publication page's leftover note, and the appendix pages that still hold
template examples. The title uses the **locked** Indonesian title, not the
journal's extended one.

`src/1.Awal/02b-HalamanAI.tex` is the campus AI-declaration page. It is the
student's own statement and was left untouched.

### B. Supervisor review of the manuscript — student's action

Not defects. The three automated checks are clean (§8); what is left needs the
student's or the supervisor's judgement, not a fix:

- **Title.** The journal uses an extended title ("Spatially Validated …") while
  the skripsi uses the locked Indonesian one. Deliberate, but the pair should be
  a conscious choice rather than drift.
- **Multimedia Appendix.** JMIR allows one. Nothing currently needs it; the
  per-case tables are the obvious candidate if a reviewer asks.
- **Bimbingan item 4 — clinical framing.** Which angle: the model as an aid to a
  radiologist, or as a way of making a rarely performed measurement routine? The
  two read differently and the supervisor's phrasing fits either.
- **Bimbingan item 11 — reduce narrative.** Purely stylistic now: the paper is
  2,306 words *shorter* than the accepted JMIR paper supplied as the model, so
  this is a preference about voice, not a length problem.
- **Single supervisor** — noted 2026-11-04; affects the skripsi front matter
  (`\pembimbingb`), not the manuscript.

### C. Optional, not blocking

- **Manuscript length — mostly settled.** Body prose is now EN 6,295 / ID 5,672
  words. The HANDOFF used to target ~800 words out of Limitations + Comparison;
  **that target was wrong** — those two sections total 644 words and are dense,
  every paragraph carrying a finding. Genuine redundancy was removed instead
  (a self-repeating sentence, twice-quoted field-of-view figures, a
  three-times-stated contribution, a Principal-Results restatement): EN −76,
  ID −36 net, after the stroma finding *added* content. Further cuts cost
  findings.
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
scripts/analyse_anterior_stroma.py     the anterior-stroma test (refuted)
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

Current state: EN 10,949 words, ID 9,743; abstracts 448 / 364; 29 references;
10 figures; 7 tables; portrait 8.5×11". Both 4.8 MB.

### Model selection is two-level — stated explicitly since 2026-11-06

Methods (both languages), Table 1 and Bab 3 previously described only the
between-configuration rule (highest case-level macro Dice), leaving Table 2's
"best epoch" column unexplained. They now state both levels: **within a run**,
the checkpoint is the epoch with the highest batch-level validation Dice
(`save_best_metric: "val_dice"` in the config, `src/train.py:1586`); **between
runs**, the configuration with the highest case-level macro Dice in original
voxel space. All four epochs in Table 2 were verified: E1/E2/E3 against their
logs, the baseline (77) against the `epoch` field stored in its checkpoint, and
each checkpoint's stored `best_metric` equals the log maximum.

The EN abstract had said "The highest validation macro Dice was selected" — the
word "configuration" was lost when the abstract was squeezed under 450 words.
Restored; the abstract is now 447 words.

### The extended version — `--extended`, added 2026-11-05

```bash
python scripts/build_manuscript.py --lang en --extended --out <dir>/...-EN-LENGKAP.docx
python scripts/build_manuscript.py --lang id --extended --out <dir>/...-ID-LENGKAP.docx
```

A second, fuller version for the student and supervisor — **not for
submission**. It is the submission text plus a "Supplementary Results" section
after the Conclusions: Figures S1–S11 and Tables S1–S3 (21 figures and 10
tables in all).

Three properties are deliberate and should be kept:

- **The submission build is untouched.** `build()` is never modified; the
  builder splices the extra blocks in at two `h2` anchors
  (`EXT_ANCHORS` in each content module). `extended()` in
  `manuscript_numbers.py` is called only on this path, so a missing artifact
  can break only the extended build. Proved on 2026-11-05: both submission
  texts were byte-identical before and after.
- **S-numbering.** Main-text figure and table numbers are typed into captions
  and prose, so inserting mid-sequence would shift them all. A separate S
  series cannot collide. `check_manuscript_submission.py` now verifies the S
  series for consecutiveness and counts images and tables across both.
- **Guarded claims.** Every comparative word in the supplementary text —
  "rose", "fell", "largest", "do not overlap" — has a `_require()` beside it.
  If the data ever stop supporting a sentence, the build stops instead of
  printing it. `extended()` also guards that the criterion's maximum is the
  epoch the paper says was frozen, and that its per-case improvement count
  agrees with `test_vs_baseline()`.

All three checks pass on the extended pair; the EN↔ID cross-check covers
177 numbers there (114 in the submission version).

### Verification — run all three after every rebuild

```bash
python scripts/check_manuscript_submission.py    # 17 format checks, EN + ID
python scripts/check_manuscript_content_en.py    # content checks, EN
python scripts/check_manuscript_content_id.py    # content checks, ID + EN↔ID numbers
```

They print per-item results and end in `FAILURES: 0` / `ISSUES: 0`. All three are
clean as of 2026-11-04. Each takes the journal folder as `argv[1]` or `$JOURNAL_DIR`.

The content checks are not a format pass. They test what a reviewer reads:
whether every headline number is stated identically everywhere, whether any claim
is stronger than the statistics support, whether Limitations covers what Results
revealed, whether the abstract carries each finding, and whether the p = .055
result stays hedged. The ID check additionally **cross-checks every reported
number against the EN version** — 114 three- and four-decimal values, all
matching. The two language modules are independent, so this is the only thing
that catches a number drifting in one and not the other.

Two defects these checks found, both invisible to a format pass:

- Results promised the gland-size effect was "reported in Limitations". It was
  in neither Limitations nor the abstract. Fixed; the abstract was at 449/450
  words, so room was made by tightening rather than by exceeding the limit.
- The ID introduction said cited work *membuktikan* ("proves") where the EN said
  "demonstrate". The translation overstated someone else's claim. Now
  *memperlihatkan bahwa*.

A caution for whoever extends these checks: a comma between digits cannot be
classified by shape. Indonesian `0,766` is a decimal and English `2,000` is a
thousands separator, and one rule for both destroys every three-decimal
Indonesian number — it reported 13 false mismatches before being normalised per
language. Avoid `\b` in regexes written through a shell heredoc; backslash
mangling turns it into a literal backspace and the pattern silently stops
matching.

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

## PI-CAI: PSA density from the final model (v10, 2026-11-13)

Prespecified analysis (results/picai_psad_prespecification.md, SHA-256 LF `6e1b6bab…3c9c`, locked
before any PI-CAI file was downloaded; erratum 2026-11-12). Nothing is committed yet.

- Data: PI-CAI v2.0 images in `D:/1.AmadeusKuliah/Skripsi/PICAI` (outside the repo), labels at
  picai_labels commit `ce4a472`. 1500 studies, 1440 eligible, 1419 in the primary comparison.
- Inference: frozen final ensemble on Colab T4, one bundle (`29021394…`), 3 runs: Run 7 folds 0-1
  and Run 8 fold 2 (first Colab account), Run 9 folds 3-4 (second Colab account; results in
  `results/colab_picai/run9_folds34`, zip in `results/colab_picai/zips`). 90 fold-3 studies were
  inferred by both Run 8 and Run 9 and gave identical volumes.
- Merge and analysis: `scripts/merge_picai_colab.py` then `scripts/run_picai_psad.py --phase analyze`
  -> `results/picai_psad/summary.json`, `report.md`, `per_case.csv`, `colab_merge.json`.
- Result: AUC 0.732 (model volume) vs 0.726 (reported volume), difference 0.006 (95% CI -0.007 to
  0.020): noninferior (margin -0.05), not superior. Model volume is 13.6 mL smaller than the report
  (-19.7% median), so at 0.15 ng/mL/mL sensitivity 0.821 vs 0.722 and specificity 0.474 vs 0.608.
  At PCNN the lower bound (-0.056) is below the margin (descriptive, by center). 155 studies
  duplicate PROSTATEx/NCI-ISBI images (all RUMC); without them the difference is 0.006.
- Manuscript: `build_manuscript.py --jmir --picai [--lang id]` builds v10; without `--picai` the
  v9 text is unchanged (checked block by block). v10 files:
  `Journal-JMIR/JurnalJMIR-BenedictAmadeusSandro-{EN,ID}-JMIRrevision-131126-v10.docx`.
  Figure 1 for v10: `JMIR_FIG_OUT=results/figures_jmir_v10 python scripts/make_jmir_workflow_figure.py --picai`.
- Still open for v10: submission pack (CLAIM checklist item numbers, Multimedia Appendix 5 files,
  cover letter), second reviewer for the literature audit, Bu Eunike's review, commit and push.
- Target journal (decided 2026-11-13, supersedes "JMIR AI" in the table above and in §7): **JMIR
  Formative Research**, as in Bu Eunike's link of 2026-11-06 (formative.jmir.org; example article
  e82587 by Pak David and Pak Moeljono). APF listed as 2500 USD on acceptance (check before
  submitting). The v10 cover letter names it; pack: `Journal-JMIR/JMIR_submission_131126-v10`,
  built with `build_jmir_submission_pack.py --picai`.
