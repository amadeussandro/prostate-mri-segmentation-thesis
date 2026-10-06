# Final Pre-Test Readiness Audit

**Audit date:** 2026-10-05
**Repository:** `amadeussandro/prothesis-mri-segmentation-thesis` — branch `main`, HEAD `0e8569cb74ca3517e4bda02f0f647300b728a53f` (in sync with `origin/main`)
**Update 2026-10-05:** Blocker B1 (label mapping) has since been **RESOLVED** — see `results/label_mapping/label_mapping_verified.md`. Blocker B3 (prior test-set usage) is also resolved: the supervisor was informed of the baseline test results on 2026-09-21. Superseded statements below are struck through rather than deleted.

**Scope:** read-only. No training, no retraining, no E4, no test-set evaluation, no reconstruction run, no deletion, no rename, no commit, no push. The only file created is this report.
**Thesis:** *"Segmentasi Zona Anatomi pada Citra MRI Pasien Kanker Prostat menggunakan 2D U-NET dan Rekonstruksi 3D."*

---

## 1. Executive Summary

- **The E1 candidate itself is clean and fully verifiable.** `best_model.pt` records `epoch: 88`, loads into the current code with `load_state_dict(strict=True)`, produces a finite `(1, 3, 442, 442)` forward pass, carries 7,768,437 finite parameters, and passes `verify_checkpoint_compatibility` against both its own run config and the committed `configs/experiments/exp_e1_dice_ce.yaml` with zero mismatches.
- **SHA-256 of the frozen E1 checkpoint: `2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2`** (93,272,507 bytes, mtime 2026-09-23 06:46:18 — unchanged since training).
- **E1 is the correct selection on validation**, but by a narrow margin that is not statistically demonstrated. Wording in §3 is deliberately conservative.
- **~~BLOCKER 1~~ — RESOLVED 2026-10-05.** *(At the time of writing, the label mapping was unverified and the repository contradicted itself — six artifacts said `unverified`, five asserted `verified_3dslicer`, and no verification record existed.)* The check has since been performed: multi-planar figures (axial/coronal/sagittal) for cases 020, 059, 099, 139 confirm **`0 = background, 1 = CG (CZ+TZ), 2 = PZ`**, agreeing with the 15/15 morphometry, the MONAI convention and the ProstateX annotation. Record: `results/label_mapping/label_mapping_verified.md`. All 12 contradictory status statements reconciled.
- **BLOCKER 2 — The official 19-case test data is NOT present locally.** `dataset/` contains only `prostate158_train/` and `external/prostatex/`. The archive (DOI 10.5281/zenodo.6592345) must be in place and audited before any test run.
- **BLOCKER 3 — The official test set is NOT pristine. It has already been used twice**, both times with the **Baseline epoch-77** checkpoint: a full test evaluation on 2026-09-13 (`results/exp01_baseline/eval_test_*`, `is_official_held_out_test: true`, 19 cases) and the RQ2 reconstruction run on 2026-09-19 (`results/exp04_reconstruction/summary.json`, `split: test`, 19 cases, 19/19 geometry OK). This was **not** discovered in the earlier E3 audit and it changes how the final evaluation must be described.
- **Reassurance on that point:** the RQ1 model selection was still made on **validation only**. Every selection artifact for Baseline/E1/E2/E3 reports `split: val`, `n_cases: 20`, `is_official_held_out_test: false`; no arm's config declares `test_csv`/`test_dir`; and no `eval_test_*` artifact exists for E1, E2 or E3. The prior test usage was for the Experiment-1 endpoint and for RQ2 geometry, both before the RQ1 tuning arms existed.
- **RQ2 checkpoint currently points at Baseline epoch 77** in four places (one declarative YAML line, one behavioural notebook line, plus two documentation strings). Behaviourally, `scripts/run_rq2_reconstruction.py` takes `--checkpoint` as a **required CLI argument**, so the implementation is reusable unchanged.
- **E1's per-epoch history on disk covers only epochs 70–99.** This is a known, already-documented logging limitation, **not** a selection error — the running best is persisted inside the checkpoint. Detail in §2.
- **Readiness verdict: NOT READY.** Three blockers, all cheap to close; none requires retraining.
- **E4: do not run** (unchanged from the E3 audit).

---

## 2. E1 Frozen Candidate Verification

### 2.1 Checkpoint identity

| Item | Value |
|---|---|
| Path | `results/exp_e1_dice_ce/best_model.pt` |
| Size | 93,272,507 bytes |
| **SHA-256** | **`2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2`** |
| mtime | 2026-09-23 06:46:18 (before the eval at 08:56; untouched since) |
| `latest_checkpoint.pt` SHA-256 | `13fa845fe6cbebd43975dbe6715c3645c82db6f878b81352167432be93e9df58` (93,293,696 bytes, epoch 99 — **not** the model to freeze) |

### 2.2 Checkpoint contents and loadability

| Check | Result |
|---|---|
| Keys | `best_metric`, `config`, `epoch`, `model_state_dict`, `optimizer_state_dict` |
| `epoch` | **88** ✓ matches the required best epoch |
| `best_metric` | 0.675715449732747 (slice-level selection proxy) |
| `best_epoch` key | **absent (`None`)** — this field was added to the checkpoint payload *after* E1 ran. `epoch: 88` is the authoritative value; see §2.4 |
| Tensors / parameters | 118 / 7,768,437 |
| All floating tensors finite | **yes** |
| Optimizer state present | yes |
| Embedded experiment name | `exp_e1_dice_ce` |
| Embedded loss | `{name: dice_ce, ce_weight: 1.0, dice_weight: 1.0, dice_classes: [1,2], epsilon: 1e-06}` — resolves under current code to `ce_class_weights: None`, i.e. **unweighted CE**, correct for E1 |
| Embedded `data` keys | `dataset_root`, `slice_sampling`, `target_mask`, `train_csv`, `valid_csv` — **no `test_csv`, no `test_dir`** |
| `build_model(...)` + `load_state_dict(strict=True)` | **OK**, no missing/unexpected keys |
| Forward pass `1×1×442×442` | `(1, 3, 442, 442)`, all finite |
| `verify_checkpoint_compatibility` vs E1 run config | **verified**, `mismatches: []` |
| `verify_checkpoint_compatibility` vs committed E1 config | **verified**, `mismatches: []` |

The checkpoint remains fully compatible with the current code base — importantly, *after* the `ce_class_weights` change introduced for E3, which confirms that change was genuinely backward-compatible.

### 2.3 Config integrity

The run config differs from the committed `configs/experiments/exp_e1_dice_ce.yaml` in **exactly one key**: `experiment.output_dir` (`./results/exp_e1_dice_ce` → the Drive path). Nothing scientific drifted.

Controlled values confirmed: `ProstateUNet2D`, 1→3 channels, `init_features: 32`, `dropout_rate: 0.2`, per-volume normalization, `z_resample: false`, `crop_pad` 442×442, AdamW, lr 1e-3, weight decay 1e-4, batch 8, 100 max epochs, seed 42, `amp: false` (FP32), no scheduler, no augmentation block, loss `dice_ce` with `dice_classes: [1, 2]` and `epsilon: 1e-6`.
Fingerprint confirms `references_test_split: false`.

### 2.4 History integrity — one documented limitation

| Source | Records | Range |
|---|---|---|
| `metrics.csv` | 30 | **epochs 70–99 only** |
| `training_history.json` | 30 | **epochs 70–99 only** |
| `summary.json` | `epochs_completed: 99`, `best_checkpoint_epoch: 88` | — |

Epochs 1–69 are **absent** from both history files. This is **not** a new finding and **not** a selection error — `Outputs/E1_supervisor_meeting_notes.md` already records it:

> *"`metrics.csv` and `training_history.json` currently contain epochs 70–99 only. The history file is rewritten from the current session's in-memory list, so earlier resumed sessions were overwritten. This is a logging limitation, not a checkpoint-selection error — `best_metric` is persisted inside the checkpoint and restored across resumes, so the epoch-88 selection correctly accounts for all 99 epochs."*

Three independent confirmations that epoch 88 is correct:
1. the checkpoint's own `epoch: 88` with `best_metric: 0.675715449732747` — written only when the running best improved, and that running best is restored on resume, so it spans all 99 epochs;
2. the proxy argmax over the surviving window (70–99) is **epoch 88** at exactly 0.675715;
3. `eval_val_summary.json` independently records `checkpoint_epoch: 88` and `training_loop_val_metric_slice_level: 0.675715449732747`.

E1 also lacks `reproducibility.json` and `state/training_state.json` — both features postdate E1 (they exist for E2 and E3). Not a defect; worth one sentence in the thesis's reproducibility note.

### 2.5 Test-split isolation for E1

| Evidence | Result |
|---|---|
| `eval_val_summary.json` metadata | `split: val`, `n_cases: 20`, `is_official_held_out_test: **false**` |
| Evaluated checkpoint | `.../results/exp_e1_dice_ce/best_model.pt` |
| Space / aggregation | `original_voxel` / `per_case` |
| `eval_test_*` artifacts in `results/exp_e1_dice_ce/` | **none** |
| `test_csv` / `test_dir` in E1 config or checkpoint config | **absent** |
| Fingerprint `references_test_split` | `false` |

**E1 never accessed the official test set.** (The separate matter of the *Baseline* having done so is §6.)

### 2.6 Internal consistency of E1's validation metrics

20/20 cases evaluated, 0 excluded, no NaN/Inf. Cohort means reproduce from the per-case CSV. Full numbers with SD and bootstrap 95% CI:

| Class | Metric | Mean ± SD | 95% CI |
|---|---|---|---|
| CG (1) | Dice | 0.8662 ± 0.0737 | [0.8285, 0.8912] |
| CG | IoU | 0.7703 ± 0.0971 | [0.7215, 0.8049] |
| CG | HD95 (mm) | 5.2467 ± 5.2305 | [3.4974, 7.9808] |
| CG | ASD (mm) | 1.1508 ± 0.8017 | [0.8519, 1.5519] |
| CG | Precision | 0.8823 ± 0.0525 | [0.8592, 0.9041] |
| CG | Recall | 0.8604 ± 0.1058 | [0.8074, 0.8952] |
| PZ (2) | Dice | 0.7527 ± 0.0706 | [0.7193, 0.7796] |
| PZ | IoU | 0.6082 ± 0.0853 | [0.5689, 0.6408] |
| PZ | HD95 (mm) | 5.4061 ± 5.2517 | [3.8414, 7.9884] |
| PZ | ASD (mm) | 0.9453 ± 0.5436 | [0.7506, 1.2085] |
| PZ | Precision | 0.8073 ± 0.1029 | [0.7581, 0.8496] |
| PZ | Recall | 0.7161 ± 0.0842 | [0.6756, 0.7492] |

Macro Dice = (0.8662 + 0.7527) / 2 = **0.8094**.

---

## 3. Validation Comparison

Volume-level, per-case, original-voxel-space metrics over the **same 20 official validation cases** for all four arms. This is the comparison metric; it is **not** the slice-level checkpoint-selection proxy.

| Experiment | Intervention | Best epoch | CG Dice | PZ Dice | Macro Dice | CG prec | CG rec | PZ prec | PZ rec | CG HD95 | PZ HD95 | CG ASD | PZ ASD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (CE) | none (reference) | 77 | 0.8642 | 0.7375 | 0.8008 | 0.8754 | 0.8634 | 0.7855 | 0.7073 | 6.054 | 6.722 | 1.219 | 1.036 |
| **E1 (Dice + CE)** | +SoftDice(CG,PZ) | **88** | **0.8662** | **0.7527** | **0.8094** | **0.8823** | 0.8604 | **0.8073** | 0.7161 | 5.247 | 5.406 | 1.151 | **0.945** |
| E2 (CE + augmentation) | +augmentation | 70 | 0.8636 | 0.7327 | 0.7982 | 0.8732 | **0.8652** | 0.7781 | 0.7008 | 7.206 | **5.279** | 1.421 | 1.066 |
| E3 (Dice + class-weighted CE) | +CE class weights | 99 | 0.8652 | 0.7510 | 0.8081 | 0.8818 | 0.8599 | 0.7297 | **0.7929** | **4.979** | 5.874 | **1.114** | 0.998 |

Per-case Dice SD (CG, PZ): Baseline (0.0614, 0.0665) · E1 (0.0737, 0.0706) · E2 (0.0529, 0.0644) · E3 (0.0686, 0.0706).
Checkpoint-selection proxy (slice-level, **never** used for cross-arm comparison): Baseline 0.65106 · E1 0.67572 · E2 0.64916 · E3 0.67178.

### Why E1 is selected

> **All four configurations performed within a relatively narrow validation range** — macro Dice spans 0.7982 to 0.8094, a total of 0.0112, which is smaller than the per-case standard deviation of any single arm (≈0.05–0.07). **E1 was selected because it achieved the highest validation macro Dice (0.8094), the highest CG Dice (0.8662) and the highest PZ Dice (0.7527), while also showing the strongest PZ ASD (0.945 mm) and competitive PZ HD95 (5.406 mm).**

Supporting points, stated at their true strength:

- E1 leads on all three primary Dice figures simultaneously — it is not winning an aggregate while losing a component.
- Best PZ surface accuracy of the four arms (ASD 0.945 mm). PZ HD95 is second-best: E2 is marginally better at 5.279 mm (a 0.13 mm difference, well inside an SD of ≈5.25 mm).
- Most balanced PZ operating point (precision 0.8073 / recall 0.7161). E3's 0.7297 / 0.7929 is tilted the other way, not better balanced; its lower PZ precision means more false-positive PZ voxels, which is undesirable for the RQ2 3D reconstruction.
- E3 has the best CG boundary metrics (HD95 4.979 mm, ASD 1.114 mm) — state this honestly; it does not overturn the selection.

### What must NOT be claimed

E1's advantage is **not** statistically demonstrated. The paired evidence available (E3 vs E1, the only paired comparison computed, n = 20) gives macro Dice difference −0.0013 with bootstrap 95% CI **[−0.0091, +0.0073]** (contains zero) and Wilcoxon **p = 0.330**; CG Dice p = 0.812; PZ Dice p = 0.452. **Do not write that E1 is significantly or statistically superior.** Write that it was selected by the pre-committed rule (highest validation macro Dice) among configurations that performed equivalently.

---

## 4. E3 Documentation / Naming

### The discrepancy, precisely

| Source | What "E3" means there |
|---|---|
| `Outputs/rq1_experiment_specification.md` §7 | **E3 = LR scheduler** (`e3_scheduler_cosine`: CosineAnnealingLR, T_max=100, eta_min=1e-6) |
| `Outputs/rq1_experiment_specification.md` §6, line 149 | class-weighted CE `[bg 0.046, CG 1.000, PZ 2.85]` is a **documented fallback for E1**, to be run *"in place of — not in addition to — E1"*, and only *"if `dice_ce` does not beat baseline"* |
| What was executed | **class-weighted CE applied on top of E1**, named `exp_e3_dice_ce_pzweight` |

Two facts follow. First, `dice_ce` *did* beat baseline (+0.0086), so the fallback's trigger condition was never met and the factor was applied as an additional arm rather than a substitute. Second, the executed experiment took a label that the pre-registration had reserved for a different factor. The pre-registered scheduler arm was never run — and `Outputs/E1_supervisor_meeting_notes.md` question 3 explicitly asked the supervisor to agree it need not be (*"Apakah Ibu setuju bahwa E3 (scheduler) tidak perlu dijalankan, mengingat model sudah konvergen pada epoch 88?"*).

**This is a bookkeeping/naming issue, not a scientific defect.** The executed run was verified single-factor against E1 (exhaustive key-by-key config diff → only `experiment.{name,description,output_dir}` and `loss.ce_class_weights` differ; 27/27 contract invariants; 13/13 loss checks), completed 100/100 epochs with contiguous history, and its weight vector `[0.0456, 1.0000, 2.8477]` matches the specification's `[0.046, 1.000, 2.85]` to three significant figures.

### Recommended thesis terminology

Use a factor-descriptive name, never a bare index:

> **E3 — Class-Weighted Cross-Entropy** (`exp_e3_dice_ce_pzweight`)

and rename the unrun arm when it is mentioned:

> **E-sched — Learning-Rate Scheduler** (pre-registered, not executed)

Throughout the thesis, name each arm by its factor rather than its number: *Baseline (CE)*, *E1 (Dice + CE)*, *E2 (CE + Augmentation)*, *E3 (Dice + Class-Weighted CE)*.

### Exact text to include (methodology or limitations)

> The experiment protocol in `Outputs/rq1_experiment_specification.md` originally reserved the label "E3" for a learning-rate-scheduler arm, and listed median-frequency class-weighted cross-entropy as a documented fallback for the E1 loss arm, to be used only if Dice+CE failed to improve on the baseline. Dice+CE did improve on the baseline (macro Dice 0.8094 vs 0.8008), so the fallback's trigger condition was not met. Following the E2 result, class weighting was instead executed as an additional single-factor arm on top of E1 and recorded under the identifier `exp_e3_dice_ce_pzweight`; it is referred to in this thesis as **E3 — Class-Weighted Cross-Entropy**. The pre-registered scheduler arm was not executed, on the grounds that the model had already converged by epoch 88. The executed arm's success criteria were fixed before the run and are stored verbatim inside its own result artifacts (`summary.json → outcome.criteria`, `reproducibility.json → pre_committed_criteria`); an exhaustive configuration diff confirms it differs from E1 only in `loss.ce_class_weights`. This deviation is in experiment numbering, not in experimental control.

**Do not** rewrite `Outputs/rq1_experiment_specification.md`, and do not amend git history. The deviation is disclosed, dated and auditable — that is stronger evidence of rigour than a tidy numbering scheme would be.

---

## 5. Label Mapping Status

### The repository contradicts itself

**"Unverified" (RQ1 side):**

| Source | Statement |
|---|---|
| `results/label_mapping/label_mapping_report.md` (2026-09-11) | **VERDICT: LABEL MAPPING REQUIRES HUMAN / 3D SLICER VERIFICATION** |
| `results/label_mapping/label_mapping_hypothesis.yaml` | `verification_method_required: "3D Slicer visual inspection, sampled cases"` |
| `configs/config_baseline.yaml`, `exp_e1_dice_ce.yaml`, `exp_e2_ce_aug.yaml`, `exp_e3_dice_ce_pzweight.yaml` | `label_mapping_status: unverified` |
| `src/dataset.py:14` | *"a documented hypothesis pending 3D Slicer confirmation"* |
| `README.md:86` | *"remains a documented hypothesis pending 3D Slicer human confirmation"* |
| `Outputs/rq1_experiment_specification.md:262`, `Outputs/baseline_evaluation_audit_report.md:145`, `Outputs/thesis_code_audit.md:302` | unresolved, flagged `[UNKNOWN]` |
| All four arms' `eval_val_summary.json` | `label_mapping_status: "unverified (see results/label_mapping/label_mapping_report.md)"`; class names literally `class1_hypothesis_CG_TZ_unverified`, `class2_hypothesis_PZ_unverified` |

**"Verified" (RQ2 side), all introduced by one commit, `24c5fba` of 2026-09-19:**

| Source | Statement |
|---|---|
| `configs/experiments/exp04_rq2_reconstruction.yaml:23–24` | *"Label mapping human-verified in 3D Slicer (cases 020, 059)"*; `label_mapping_status: "verified_3dslicer"` |
| `scripts/run_rq2_reconstruction.py:36, 65, 626` | same claim |
| `notebooks/prostate158_rq2_reconstruction_colab.ipynb` cell 0 | *"Label mapping (human-verified in 3D Slicer)"* |
| `results/exp04_rq2_reconstruction/report.md:38`, `results/exp04_reconstruction/summary.json:45` | `verified_3dslicer (0=background, 1=CG, 2=PZ)` |
| `code-brief.md:408` (2026-09-20) | *"manually checked against representative Prostate158 cases in 3D Slicer"* |

### Determination

**Chronology settles it.** The `verified_3dslicer` claims date to 2026-09-19/20. But `Outputs/E1_supervisor_meeting_notes.md`, dated **2026-09-23 — four days later** — still lists this as an open question for the supervisor:

> *"**Label CG/PZ** masih berstatus hipotesis terdokumentasi (belum diverifikasi di 3D Slicer). Apakah perlu diverifikasi sebelum penulisan akhir?"*

And **no verification record exists.** `results/label_mapping/` contains only the 15 overview PNGs generated by `scripts/verify_label_mapping.py` — there is no Slicer screenshot, no signed log, no `label_mapping_verified.md`, nothing recording who looked at cases 020 and 059, when, or what they saw.

### What the direct anatomical evidence in this repository shows

Per the instruction not to infer from literature where direct evidence exists — the repository *does* hold direct, quantitative anatomical evidence, computed from the data itself over 15 cases spanning the full ID range (`results/label_mapping/label_mapping_report.md`):

| Measurement | Label 1 | Label 2 | Interpretation |
|---|---|---|---|
| Mean exterior-contact fraction | 0.188 | **0.403** | label 2 is the thin shell touching the gland boundary → PZ |
| Mean normalized centroid distance | 0.725 | **0.931** | label 2 sits farther from the gland centroid → peripheral |
| Mean volume ratio 1:2 | **3.95 : 1** | | label 1 is the bulky central mass → CG |
| Per-case agreement | **15/15** on exterior contact, **15/15** on centroid distance | | no dissenting case |

**Both independent morphological tests agree, in every sampled case, that label 1 = CG and label 2 = PZ** — consistent with the MONAI convention and with the direction asserted on the RQ2 side. So the *direction is very probably right*; the risk is documentary, not numerical. Note also that E3's weight vector is unaffected either way: it was derived from the measured frequencies of *class 1* (4.30%) and *class 2* (1.51%), so "up-weight the minority class" holds regardless of the anatomical names.

### Verdict

Per the audit instruction, I must stop and state this plainly:

> ## ~~Label mapping must be manually verified before final anatomical reporting.~~
>
> **RESOLVED 2026-10-05** — verified by multi-planar visual inspection of cases 020, 059, 099, 139. Confirmed `1 = CG (CZ+TZ)`, `2 = PZ`. See `results/label_mapping/label_mapping_verified.md`.

The anatomical *direction* is strongly supported by in-repository evidence, but the status is formally **unresolved and internally contradictory**, and the latest-dated document says unverified. Closing this costs minutes, not a GPU run:

1. Open `t2.nii.gz` + `t2_anatomy_reader1.nii.gz` together in 3D Slicer for cases **020, 059, 099, 139** (the report's own suggested sample).
2. Visually identify which integer is the thin outer rim (PZ) versus the central mass (CG/TZ).
3. Write the result, the cases inspected, the date and the inspector into a new `results/label_mapping/label_mapping_verified.md`, and update `results/label_mapping/label_mapping_report.md` and `label_mapping_hypothesis.yaml` so they no longer contradict the RQ2 artifacts.
4. Only then switch the RQ1 configs' `label_mapping_status` and report per-class results under the names "CG" and "PZ".

Until step 3 exists, report per-class numbers as **class 1** / **class 2** with the hypothesis stated, exactly as the current artifacts already do. **No metric was altered by this audit.**

---

## 6. Official Test Data Availability

### Availability: **NOT PRESENT LOCALLY**

| Location | Contents |
|---|---|
| `dataset/` | `prostate158_train/`, `external/prostatex/` only |
| `dataset/prostate158_train/` | `train/` (139 case directories), `train.csv`, `valid.csv` |
| Any `prostate158_test/` or test case directory | **absent** |
| `configs/config_baseline.yaml:26–27` | `test_csv` / `test_dir` present but **commented out**, annotated *"not available locally yet"* |
| Runtime behaviour | `src/splits.py::load_official_split` emits a `UserWarning` that the archive (DOI 10.5281/zenodo.6592345) was not found and returns `test_ids = None`; every arm's notebook asserted `split.test_ids is None` and passed |

**Required before the final test run:** obtain the official Prostate158 test archive (DOI **10.5281/zenodo.6592345**), place it on Drive (expected: `/content/drive/MyDrive/THESIS_PROSTATE158/dataset/test/`), extract it, and audit it **read-only** with the tool already written for this purpose:

```
python scripts/audit_test_archive.py \
    --zip /content/drive/MyDrive/THESIS_PROSTATE158/dataset/test/prostate158_test.zip \
    --extract-dir /content/drive/MyDrive/THESIS_PROSTATE158/dataset/test/extracted \
    --train-csv dataset/prostate158_train/train.csv \
    --valid-csv dataset/prostate158_train/valid.csv
```

That script loads no model, runs no inference and computes no segmentation metrics; it reports case count and IDs, per-case presence of `t2.nii.gz` and `t2_anatomy_reader1.nii.gz`, NIfTI geometry and label values, train/val leakage, and whether `load_official_split` resolves exactly 19 test IDs. `tests/test_audit_test_archive.py` and `tests/test_test_split_readiness.py` already cover it.

### The test set is not pristine — two prior uses found

This was **not** surfaced by the earlier E3 audit and must be disclosed.

| Date | Artifact | Checkpoint | Evidence |
|---|---|---|---|
| **2026-09-13** | `results/exp01_baseline/eval_test_per_case_metrics.csv`, `eval_test_summary.csv`, `eval_test_summary.json` | **Baseline epoch 77** | `split: "test"`, `n_cases: 19`, `is_official_held_out_test: **true**`, case IDs `001`–`019`, `config_path: /content/config_baseline_test_colab.yaml`, `generated_utc: 2026-09-13T21:30:41Z` |
| **2026-09-19** | `results/exp04_reconstruction/` (`summary.json`, `metrics_per_case.csv`, `reconstruction_validation.csv`, `predictions_2d/`, `reconstructions_3d/`), and `results/exp04_rq2_reconstruction/` | **Baseline epoch 77** | `task: RQ2_3d_reconstruction`, `checkpoint_epoch: 77`, `split: "test"`, `n_cases: 19`, `is_official_held_out_test: **true**`, `reconstruction_geometry_all_ok: 19/19`, `generated_utc: 2026-09-19T08:11:12Z` |

**What this does and does not compromise:**

- **Does not compromise RQ1 model selection.** Both prior uses were on the Baseline, and both predate the design of E1/E2/E3 as comparison arms. Every selection artifact for all four arms reports `split: val`, `n_cases: 20`, `is_official_held_out_test: false`; no arm's config declares `test_csv`/`test_dir`; no `eval_test_*` artifact exists for E1, E2 or E3. The pre-committed criteria and the validation-only comparison are documented in the artifacts themselves.
- **Does change the framing.** The upcoming E1 evaluation is the **third** use of the test set overall and the **first** on E1. The thesis cannot claim "the test set was touched exactly once". It should state: the test set was evaluated once for the Experiment-1 baseline endpoint and once for the RQ2 reconstruction geometry study, both with the frozen epoch-77 baseline weights and both before the RQ1 tuning arms were designed; the RQ1 configuration was selected on validation data only; the selected configuration was then evaluated on the test set once.
- **Decision required from the supervisor** before proceeding — this is a protocol call, not a technical one. Options: (a) report the prior Baseline test result as a pre-registered reference point and the E1 test result as the final RQ1 figure, disclosing both uses; or (b) report only the E1 test result and disclose the prior uses in the limitations. Option (a) is more transparent and is what the existing artifacts already support.

---

## 7. RQ2 Checkpoint Status

**Current: Baseline epoch 77. Required final: E1 epoch 88.** Four places reference the baseline checkpoint — one declarative, one behavioural, two documentary. **None was changed by this audit.**

| # | File | Line / location | Current value | Nature |
|---|---|---|---|---|
| 1 | `configs/experiments/exp04_rq2_reconstruction.yaml` | **line 17** | `source_checkpoint: "./results/exp01_baseline/best_model.pt"   # frozen Experiment-1 epoch-77 weights` | **declarative only** — no code reads this file (verified: no `.py`/`.ipynb` references `exp04_rq2_reconstruction.yaml` or `source_checkpoint`) |
| 2 | `notebooks/prostate158_rq2_reconstruction_colab.ipynb` | **cell 4** | `CHECKPOINT_PATH = f"{DRIVE_BASE}/results/exp01_baseline/best_model.pt"` | **behavioural in the Colab path** — this value is passed to `--checkpoint` in cell 18 |
| 3 | `notebooks/prostate158_rq2_reconstruction_colab.ipynb` | cell 0 (markdown) | *"Frozen checkpoint: `results/exp01_baseline/best_model.pt` (epoch 77, in=1, out=3)"* | documentation |
| 4 | `scripts/run_rq2_reconstruction.py` | docstring + `--checkpoint` help text (*"the frozen best_model.pt (epoch 77)"*) | epoch-77 wording | documentation |

Required change, when authorised (**do not apply yet**): point all four at `results/exp_e1_dice_ce/best_model.pt` (epoch 88, SHA-256 `2366e3c6…28d5d2`) and update the "epoch 77" wording to "epoch 88". Also note `results/exp04_rq2_reconstruction/report.md`'s own header already states that re-running the script **regenerates that report in place**, so the stale baseline-derived report will be replaced rather than needing manual editing.

---

## 8. RQ2 Implementation Readiness

### Reusable unchanged — yes

`scripts/run_rq2_reconstruction.py` takes the checkpoint as a **required CLI argument** (`--checkpoint`, line 727) and reads the model architecture and preprocessing from **the checkpoint's own embedded config**, so inference always matches how the weights were trained. Swapping Baseline→E1 therefore requires **no code change at all** — only the argument value (and, for tidiness, the four references in §7). Both checkpoints have identical architecture (`ProstateUNet2D`, 1→3, 32 features, dropout 0.2, 7,768,437 parameters) and identical preprocessing (per-volume normalization, no z-resample, crop/pad 442×442), so nothing downstream shifts.

All scientific work is delegated to `src/` (`src.reconstruction`, `src.transforms`, `src.metrics`, `src.evaluate`); the script only composes them. `tests/test_reconstruction.py`, `tests/test_rq2_reconstruction.py`, `tests/test_original_space_gt.py`, `tests/test_geometry.py` and `scripts/roundtrip_test.py` cover it, and the full suite passes (370 passed, 7 skipped).

### Confirmed intended final pipeline

```
E1 frozen checkpoint (epoch 88, SHA-256 2366e3c6…28d5d2)
  → ProstateZonal2DDataset preprocessing, read from the checkpoint's embedded config
  → slice-wise 2D prediction (eval / no_grad, argmax over 3 classes)
  → explicit slice-index reassembly            (src.reconstruction)
  → inverse preprocessing, nearest-neighbour   (src.transforms)
  → original voxel space, source affine restored (src.reconstruction)
  → reconstructed 3D segmentation NIfTI (saved)
  → geometry validation  (src.reconstruction.validate_reconstruction_geometry)
  → quantitative evaluation vs untouched GT    (src.metrics)
  → 3D visualization
```

Ground truth is read untouched from disk via `src/evaluate.py::load_original_space_ground_truth` and is never round-tripped.

### Segmentation accuracy vs reconstruction/geometric fidelity — keep these apart

| | Reconstruction / geometric fidelity | Segmentation accuracy |
|---|---|---|
| What it measures | whether the 2D→3D transform is information-preserving | how well the model's labels match the radiologist's |
| How measured | round-trip the **ground truth** through the identical pipeline and compare to itself; plus shape/affine/spacing/orientation checks | compare the **model's prediction** to the untouched GT in original voxel space |
| Result | **Dice = 1.0 exactly** (`scripts/roundtrip_test.py`), geometry `19/19 OK` on the prior run | per-case Dice / IoU / HD95 / ASD / precision / recall, mean ± SD |
| Involves the model? | **No** | Yes |

> **Round-trip Dice = 1.0 means the reconstruction transformation is lossless. It is NOT a segmentation result, and it must never be reported as one, averaged with one, or placed in the same table as one.** The repository already enforces this distinction in `exp04_rq2_reconstruction.yaml` (`roundtrip_fidelity_note`), in the script docstring, and in `results/exp04_rq2_reconstruction/report.md` §9 — keep that wording in the thesis.

**One caveat:** the existing `results/exp04_reconstruction/` and `results/exp04_rq2_reconstruction/` artifacts are **Baseline epoch-77** derived (§6). They must not be mixed with, or presented as, the final E1-based RQ2 figures. Re-run RQ2 against E1 after the test evaluation.

---

## 9. Repository Hygiene

Working tree relative to HEAD `0e8569c` (in sync with `origin/main`). **Nothing was deleted, renamed, modified or committed.**

| Path | Size | `.gitignore` covered? | Assessment |
|---|---|---|---|
| `results/exp_e3_dice_ce_pzweight-20261004T164011Z-1-001.zip` | **166 MB** | **NO** | Google Drive bulk-download archive. Must never be committed. |
| `results/exp_e3_dice_ce_pzweight/exp_e3_dice_ce_pzweight/` | **179 MB** | partly — `*.pt` and `*.png` are ignored by the global rules, but `metrics.csv`, `summary.json`, `training_history.json`, `reproducibility.json`, `config*.{yaml,json,txt}`, `e3_vs_e1_paired_cases.csv`, `eval_val_*` and `state/training_state.json` are **NOT**, because `.gitignore` lists the *flat* `results/exp_e3_dice_ce_pzweight/<file>` paths | **Nested-directory artifact**, confirmed: the zip's 19 entries all sit under one top-level folder `exp_e3_dice_ce_pzweight/`, extracted into a same-named directory. Not the generated layout (the run wrote flat to Drive) and not intentional. |
| `results/exp_e3_dice_ce_pzweight/prostate158_exp_e3_dice_ce_pzweight_colab.ipynb` | 676 KB | **NO** | Executed notebook with outputs. Cell source is byte-identical to the committed copy. |
| `results/exp_e2_ce_aug/prostate158_exp_e2_ce_aug_colab_RESULTCOLAB.ipynb` | 1.1 MB | **NO** | Same situation for E2. |
| `results/exp_e1_dice_ce/training_history.json` | 20 KB | **NO** | **Coverage gap:** the `.gitignore` E1 block omits `training_history.json`, though the E2 and E3 blocks include it. |
| `results/exp04_reconstruction/` (7 entries) | 6.9 MB | **NO** | Generated RQ2 artifacts (Baseline-derived). |
| `Outputs/` | 5.5 MB | **NO** | Specifications, audit reports, meeting notes, `.pptx`. Mostly valuable documentation; the `.pptx` files are large binaries. |
| `code-brief.md`, `Thesis-Write-Jurnal.md` | small | **NO** | Documentation; committing is a judgement call. |
| `JurnalJMIR-BenedictAmadeusSandro.docx`, `InstructionsForAuthorsOfJMIR.docx` | 2.6 MB | **NO** | Manuscript binaries. |
| `thesis-before-author-cleanup.bundle` | 4.0 MB | **NO** | Git bundle backup — must never be committed. |
| ` D results/exp_e1_dice_ce/plots/.gitkeep` | — | tracked | **Unrelated deletion**, pre-existing since before this audit sequence. |

### Recommended actions — review and run yourself; I executed none of these

Flatten the nested E3 directory (inspect first; this moves 179 MB):

```bash
git status --short results/exp_e3_dice_ce_pzweight
```

Restore the unrelated deleted placeholder:

```bash
git restore results/exp_e1_dice_ce/plots/.gitkeep
```

Add the missing `.gitignore` coverage (one block; review before applying):

```bash
printf '\n# Drive bulk-download archives and executed result notebooks (never commit)\nresults/**/*.zip\nresults/**/*_RESULTCOLAB.ipynb\nresults/**/prostate158_exp_*_colab.ipynb\nresults/exp_e1_dice_ce/training_history.json\nthesis-before-author-cleanup.bundle\n*.docx\n' >> .gitignore
```

Verify coverage took effect before committing anything:

```bash
git status --short && git check-ignore -v results/exp_e3_dice_ce_pzweight-20261004T164011Z-1-001.zip
```

Decide deliberately on `Outputs/`, `code-brief.md` and `Thesis-Write-Jurnal.md`: the Markdown specifications and audit reports are genuinely worth versioning (they are the pre-registration evidence this thesis leans on); the `.pptx`/`.docx` binaries are not.

---

## 10. Final Readiness Decision

# NOT READY

Three blockers, none requiring retraining:

| # | Blocker | Owner | Effort |
|---|---|---|---|
| **B1** | **Label mapping unverified and self-contradictory.** Six artifacts say `unverified`, five say `verified_3dslicer`, no verification record exists, and the latest-dated document (2026-09-23) lists it as an open question. §5. | You, in 3D Slicer | ~30 min |
| **B2** | **Official 19-case test archive not present locally.** Must be obtained (DOI 10.5281/zenodo.6592345), placed on Drive, extracted and audited read-only with `scripts/audit_test_archive.py`. §6. | You | download + 1 script run |
| **B3** | **Test set already used twice** (Baseline test eval 2026-09-13; RQ2 reconstruction 2026-09-19), both with Baseline epoch 77. Requires a disclosure decision before the final run so the thesis describes the protocol accurately. §6. | Supervisor | one conversation |

Everything else is green: E1's checkpoint is verified, hash-identified, loadable, config-compatible and demonstrably validation-only; the four-arm comparison is complete and internally consistent; the RQ2 implementation is reusable unchanged.

---

## 11. Exact Next Steps

1. **Resolve B1 — verify the label mapping.** Open `t2.nii.gz` + `t2_anatomy_reader1.nii.gz` in 3D Slicer for cases **020, 059, 099, 139**. Record the finding, cases, date and inspector in a new `results/label_mapping/label_mapping_verified.md`. Update `label_mapping_report.md` and `label_mapping_hypothesis.yaml` so they no longer contradict the RQ2 artifacts. *(Expected outcome: confirmation of 1 = CG, 2 = PZ — 15/15 morphological agreement already points there.)*
2. **Resolve B3 — agree the disclosure with your supervisor.** Present the two prior test uses and the fact that RQ1 selection was validation-only. Agree the wording.
3. **Resolve B2 — obtain and audit the test archive.** Download DOI 10.5281/zenodo.6592345, place under `<DRIVE_ROOT>/dataset/test/`, extract, then run `scripts/audit_test_archive.py` (read-only). Require: exactly 19 cases, zero overlap with the 119 train / 20 validation IDs, `t2.nii.gz` + `t2_anatomy_reader1.nii.gz` present for all, labels ⊆ {0,1,2}, and `load_official_split` resolving 19 test IDs.
4. **Freeze E1.** Copy `results/exp_e1_dice_ce/best_model.pt` to the Drive location used for the test run and re-verify SHA-256 = `2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2` after transfer. Record the hash in the thesis reproducibility section.
5. **Request explicit approval to proceed to the test evaluation.** Do not proceed without it.
6. **Run the test evaluation ONCE** — `run_evaluation(config, checkpoint=E1 best_model.pt, split="test", class_ids=(0,1,2))` — writing to `results/exp_e1_dice_ce/eval_test_*`. One invocation. No re-runs, no variants, no threshold sweeps, no post-processing, no TTA.
7. **Record the final RQ1 test metrics** as the headline RQ1 result. The validation table in §3 becomes the model-selection evidence.
8. **Do not use the test metrics to revisit model selection.** If E1 performs worse on test than on validation, that is the finding and it gets reported as such. Changing the selected arm after seeing test numbers would invalidate the protocol.
9. **Update the RQ2 checkpoint references** (the four locations in §7) to E1 epoch 88.
10. **Re-run RQ2 reconstruction + geometry validation** against the frozen E1 checkpoint. `report.md` regenerates in place. Do not mix the output with the existing Baseline-derived `results/exp04_reconstruction/` and `results/exp04_rq2_reconstruction/` artifacts.
11. **Report RQ2 reconstruction fidelity separately from segmentation accuracy** (§8). Round-trip Dice = 1.0 never appears in a segmentation-accuracy table.
12. **Update thesis / journal:** §3 table as RQ1 validation evidence; test numbers as the final RQ1 result; E3 written up as a mechanistic/negative finding with the naming note from §4; label-mapping status and the test-set usage history disclosed as stated.
13. **Repository hygiene** (§9) before the final submission commit.

---

## 12. Risks and Blockers

**Blocking (must close before the test run):**

1. **B1 — label mapping unverified and contradictory.** Every per-class number and every sentence containing "CG" or "PZ" is provisional until closed. The numeric results are unaffected; only the anatomical naming is at risk.
2. **B2 — official test archive absent locally.** Cannot run the final evaluation.
3. **B3 — test set already used twice**, both with Baseline epoch 77. Needs disclosure wording agreed before the run so the protocol description is honest.

**Non-blocking, must be disclosed:**

4. **E1's per-epoch history covers only epochs 70–99.** A logging limitation already documented in `Outputs/E1_supervisor_meeting_notes.md`; epoch-88 selection is independently confirmed three ways (§2.4). Plots and loss curves for epochs 1–69 cannot be reconstructed.
5. **E1 lacks `reproducibility.json` and `state/training_state.json`** — both postdate E1. Compensate with the SHA-256 in §2.1 and this audit.
6. **E1's advantage is within noise.** Macro Dice spread across arms is 0.0112 versus per-case SD ≈0.05–0.07; the one paired comparison available (E3 vs E1) gives p = 0.330, CI [−0.0091, +0.0073]. Never write "statistically superior".
7. **"E3" naming collision** with the pre-registered scheduler arm (§4). Disclose; do not rewrite history.
8. **Existing RQ2 artifacts are Baseline-derived** and must not be presented as final.
9. **E2 marginally beats E1 on PZ HD95** (5.279 vs 5.406 mm) and **E3 beats E1 on both CG boundary metrics** (HD95 4.979, ASD 1.114). Report these honestly; they do not overturn the selection.
10. **Nakić et al. 2026 could not be verified** in the prior audit (two web searches found no matching record). Do not cite it as support for class weighting. Cite median-frequency balancing generically instead.
11. **Repository hygiene:** 166 MB stray ZIP and 179 MB nested directory, neither fully `.gitignore`-covered (§9). Risk of an accidental large commit.

---

# FINAL RECOMMENDATION

| Item | Determination |
|---|---|
| **Final RQ1 candidate** | **E1 — Dice + Cross-Entropy** (`exp_e1_dice_ce`) |
| **Best epoch** | **88** |
| **Checkpoint** | `results/exp_e1_dice_ce/best_model.pt` |
| **SHA-256** | `2366e3c68f5ec616b19c0e7df39f9135898fd287b9436ce5dfab7baf8528d5d2` |
| **Validation macro Dice** | **0.8094** |
| **Validation CG Dice** | **0.8662** ± 0.0737 (95% CI [0.8285, 0.8912]) |
| **Validation PZ Dice** | **0.7527** ± 0.0706 (95% CI [0.7193, 0.7796]) |
| **Should E4 be run?** | **No.** Four arms span 0.0112 macro Dice — smaller than any single arm's per-case SD. E3 proved a provably effective loss-level intervention moves the PZ operating point without moving overlap, so the residual error is boundary localisation, not class frequency or augmentation. The pre-registered scheduler arm targets optimisation noise, not that limit. Run it only if the supervisor explicitly requires a fourth tuning factor. |
| **Is test evaluation currently allowed?** | **No — NOT READY.** Blocked by B1 (label mapping), B2 (test data absent), B3 (prior test usage disclosure). And not without your explicit approval after this audit. |
| **Is label mapping verified?** | **Yes, as of 2026-10-05.** Confirmed `1 = CG (CZ+TZ)`, `2 = PZ` by multi-planar visual inspection of cases 020, 059, 099, 139, corroborated by 15/15 morphometry, the MONAI convention and the independently annotated ProstateX cohort. Record: `results/label_mapping/label_mapping_verified.md`. *(This row read "No" when the audit was first written; the verification was performed afterwards.)* |
| **Before the test** | 1) verify the label mapping in 3D Slicer and record it; 2) agree the test-usage disclosure with the supervisor; 3) obtain + extract + read-only-audit the 19-case archive; 4) freeze E1 and re-verify its SHA-256 after transfer; 5) obtain explicit approval. |
| **Immediately after the test** | Record the final RQ1 test metrics once; **do not** revisit model selection in light of them; update the thesis RQ1 results; keep the §3 validation table as the selection evidence. |
| **For RQ2** | Re-point the four checkpoint references (§7) from Baseline epoch 77 to E1 epoch 88; re-run `scripts/run_rq2_reconstruction.py` unchanged (checkpoint is a CLI argument, no code change needed); report reconstruction/geometric fidelity strictly separately from segmentation accuracy; do not mix the output with the existing Baseline-derived artifacts. |

---

*Generated by a read-only audit. No experiment result, metric, config, checkpoint, or thesis artifact was modified. No git operation was performed. This file is the only artifact created.*
