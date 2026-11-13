"""Apply the prespecified seed / 2.5D selection rule on the validation split.

Rule: results/seed_25d_prespecification.md (SHA-256 83ba8e59...26f8, written
before any run). Only the 20 validation cases are read; the test split is not
touched here.

  1. Every seed of both arms, scored with src/evaluate.run_evaluation(split="val")
     (case-level Dice in original voxel space). These are the eval_val_summary
     files the Colab notebook wrote for each run, and the September evaluation of
     the frozen 2D seed-42 model.
  2. Adopt 2.5D only if its mean macro Dice ((CG + PZ) / 2) over the shared
     seeds beats 2D by >= 0.005; otherwise keep 2D.
  3. Final model = softmax-average ensemble of the adopted arm's seeds. Add
     left-right flip TTA only if it raises the ensemble's validation macro Dice.
     Step 3 is computed here on CPU through the same reconstruction and metric
     code as run_evaluation; the adopted arm's single seeds are re-scored in the
     same pass as a reproducibility check of step 1.

Writes results/e1_seed_selection/{per_case_val.csv, selection.json, report.md}.
Run from the repo root:  .venv/Scripts/python.exe scripts/e1_seed_selection.py
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.dataset import ProstateZonal2DDataset  # noqa: E402
from src.ensemble import LEFT_RIGHT_DIM, load_member  # noqa: E402
from src.evaluate import load_original_space_ground_truth  # noqa: E402
from src.metrics import aggregate_case_results, evaluate_case_volume  # noqa: E402
from src.reconstruction import reconstruct_case_to_original_space  # noqa: E402
from src.splits import load_official_split  # noqa: E402
from src.transforms import PreprocessingConfig  # noqa: E402

PRESPEC = "results/seed_25d_prespecification.md"
PRESPEC_SHA = "83ba8e596ec4b0a671855276e583300c4e63119bb8cad33e003df71a7b3326f8"
MARGIN = 0.005
OUT = "results/e1_seed_selection"
SEEDS = (42, 43, 44)
RUN_DIRS = {
    "2d": {42: "results/exp_e1_dice_ce", 43: "results/colab_e1_seeds/e1_2d_seed43",
           44: "results/colab_e1_seeds/e1_2d_seed44"},
    "2.5d": {42: "results/colab_e1_seeds/e1_25d_seed42", 43: "results/colab_e1_seeds/e1_25d_seed43",
             44: "results/colab_e1_seeds/e1_25d_seed44"},
}
DATA = {
    "dataset_root": "dataset/prostate158_train/train",
    "train_csv": "dataset/prostate158_train/train.csv",
    "valid_csv": "dataset/prostate158_train/valid.csv",
    "target_mask": "t2_anatomy_reader1.nii.gz",
}
CHUNK = 4  # slices per forward pass; small to fit an 8 GB laptop


def git_blob_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()


def read_val_summary(run_dir: str):
    s = json.load(open(os.path.join(run_dir, "eval_val_summary.json")))
    md, pc = s["metadata"], s["per_class"]
    cg, pz = pc["1"]["metrics"]["dice"]["mean"], pc["2"]["metrics"]["dice"]["mean"]
    return {"cg": cg, "pz": pz, "macro": (cg + pz) / 2, "n_cases": md["n_cases"],
            "case_ids": md["case_ids"], "epoch": md["checkpoint_epoch"]}


def summarize(case_results):
    agg = aggregate_case_results(case_results, class_ids=(0, 1, 2))
    cg, pz = agg[1]["dice"]["mean"], agg[2]["dice"]["mean"]
    return {"cg": cg, "pz": pz, "macro": (cg + pz) / 2,
            "cg_sd": agg[1]["dice"]["std"], "pz_sd": agg[2]["dice"]["std"],
            "cg_hd95": agg[1]["hd95_mm"]["mean"], "pz_hd95": agg[2]["hd95_mm"]["mean"]}


def main():
    torch.set_num_threads(os.cpu_count() or 4)
    if git_blob_sha256(PRESPEC) != PRESPEC_SHA:
        raise SystemExit(f"{PRESPEC} changed since it was fixed; refusing to run the selection.")
    os.makedirs(OUT, exist_ok=True)
    val_ids = list(load_official_split(train_csv=DATA["train_csv"], valid_csv=DATA["valid_csv"]).val_ids)
    assert len(val_ids) == 20

    # ---- Step 1: per-seed validation scores (run_evaluation output) ----
    single = {arm: {s: read_val_summary(d) for s, d in dirs.items()} for arm, dirs in RUN_DIRS.items()}
    for arm in single:
        for s, r in single[arm].items():
            assert r["n_cases"] == 20 and sorted(r["case_ids"]) == sorted(val_ids), (arm, s)

    def arm_stats(arm):
        g = lambda k: [single[arm][s][k] for s in SEEDS]  # noqa: E731
        return {f"{f}_{k}": fn(g(k)) for k in ("cg", "pz", "macro")
                for f, fn in (("mean", statistics.mean), ("sd", statistics.stdev))}

    stats = {arm: arm_stats(arm) for arm in RUN_DIRS}
    diff = stats["2.5d"]["mean_macro"] - stats["2d"]["mean_macro"]
    adopted = "2.5d" if diff >= MARGIN else "2d"
    print(f"Step 1-2: 2D mean macro {stats['2d']['mean_macro']:.4f}, 2.5D {stats['2.5d']['mean_macro']:.4f}, "
          f"diff {diff:+.4f} (margin {MARGIN}) -> adopt {adopted}", flush=True)

    # ---- Step 3: ensemble of the adopted arm, with and without flip TTA ----
    members = []
    for s in SEEDS:
        # "/" on purpose: the path is recorded in selection.json and read on Linux too.
        m, info = load_member(f"{RUN_DIRS[adopted][s]}/best_model.pt")
        members.append((s, m, info))
    ref = members[0][2]
    for _, _, info in members[1:]:
        assert info["context_slices"] == ref["context_slices"] and info["preprocessing"] == ref["preprocessing"]
    dataset = ProstateZonal2DDataset(
        dataset_root=DATA["dataset_root"], patient_ids=val_ids, image_name="t2.nii.gz",
        mask_name=DATA["target_mask"], slice_sampling="all",
        preprocessing_config=PreprocessingConfig.from_dict(ref["preprocessing"]),
        cache_data=False, context_slices=ref["context_slices"],
    )

    labels = [f"seed{s}" for s in SEEDS] + ["ensemble", "ensemble_tta"]
    results = {k: [] for k in labels}
    t0 = time.time()
    for n, pid in enumerate(val_ids, 1):
        geometry, meta = dataset.get_case_metadata(pid)
        idxs = [i for i, (p, _) in enumerate(dataset.samples) if p == pid]
        stacks = {k: [] for k in labels}
        slice_indices = []
        with torch.no_grad():
            for c0 in range(0, len(idxs), CHUNK):
                samples = [dataset[i] for i in idxs[c0:c0 + CHUNK]]
                x = torch.from_numpy(np.stack([sm["image"] for sm in samples])).float()
                xf = torch.flip(x, dims=[LEFT_RIGHT_DIM])
                p_sum = pf_sum = 0
                for s, m, _ in members:
                    p = F.softmax(m(x), dim=1)
                    pf = torch.flip(F.softmax(m(xf), dim=1), dims=[LEFT_RIGHT_DIM])
                    for b in range(len(samples)):
                        stacks[f"seed{s}"].append(p[b].argmax(0).numpy().astype(np.int64))
                    p_sum, pf_sum = p_sum + p, pf_sum + pf
                ens = p_sum.argmax(1).numpy().astype(np.int64)
                tta = (p_sum + pf_sum).argmax(1).numpy().astype(np.int64)
                for b, sm in enumerate(samples):
                    stacks["ensemble"].append(ens[b])
                    stacks["ensemble_tta"].append(tta[b])
                    slice_indices.append(sm["slice_idx"])
        gt = load_original_space_ground_truth(dataset, pid, geometry)
        for k in labels:
            pred = np.asarray(reconstruct_case_to_original_space(stacks[k], slice_indices, meta, geometry).dataobj)
            results[k].append(evaluate_case_volume(pred, gt, spacing=geometry.zooms,
                                                   class_ids=(0, 1, 2), patient_id=pid))
        print(f"  case {pid} ({n}/20) done, {time.time() - t0:.0f} s", flush=True)

    scored = {k: summarize(v) for k, v in results.items()}
    ens_res, tta_res = scored["ensemble"], scored["ensemble_tta"]
    use_tta = tta_res["macro"] > ens_res["macro"]
    final = tta_res if use_tta else ens_res
    repro = {f"seed{s}": {"step1": single[adopted][s]["macro"], "cpu": scored[f"seed{s}"]["macro"],
                          "abs_diff": abs(single[adopted][s]["macro"] - scored[f"seed{s}"]["macro"])}
             for s in SEEDS}

    with open(os.path.join(OUT, "per_case_val.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "case", "cg_dice", "pz_dice", "macro_dice", "cg_hd95_mm", "pz_hd95_mm"])
        for k in labels:
            for c in results[k]:
                cg, pz = c.per_class[1]["dice"], c.per_class[2]["dice"]
                w.writerow([f"{adopted}_{k}", c.patient_id, cg, pz, (cg + pz) / 2,
                            c.per_class[1]["hd95_mm"], c.per_class[2]["hd95_mm"]])

    final_members = [{"seed": s, "epoch": info["epoch"], "path": info["path"], "sha256": info["sha256"]}
                     for s, _, info in members]
    selection = {
        "prespecification": PRESPEC, "prespecification_sha256": PRESPEC_SHA,
        "split": "val", "n_cases": 20, "case_ids": val_ids, "test_split_read": False,
        "step1_single_seed": {a: {str(s): v for s, v in d.items()} for a, d in single.items()},
        "arm_stats": stats, "diff_25d_minus_2d": diff, "margin": MARGIN, "adopted_arm": adopted,
        "step3": scored, "tta_kept": use_tta, "reproducibility_cpu_vs_step1": repro,
        "final_model": {"arm": adopted, "context_slices": ref["context_slices"],
                        "tta_flip_left_right": use_tta, "members": final_members, "val": final},
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with open(os.path.join(OUT, "selection.json"), "w") as f:
        json.dump(selection, f, indent=2)

    L = ["# E1 seed / 2.5D selection (validation only)", "",
         f"Rule: `{PRESPEC}` (SHA-256 `{PRESPEC_SHA[:12]}...`, checked before running). "
         "20 validation cases; the test split was not read.", "",
         "## Step 1. Every seed (src/evaluate.run_evaluation, split=val)", "",
         "| Arm | Seed | Epoch | CG Dice | PZ Dice | Macro Dice |", "|---|---|---|---|---|---|"]
    for arm in RUN_DIRS:
        for s in SEEDS:
            r = single[arm][s]
            L.append(f"| {arm} | {s} | {r['epoch']} | {r['cg']:.4f} | {r['pz']:.4f} | {r['macro']:.4f} |")
    for arm in RUN_DIRS:
        st = stats[arm]
        L.append(f"| **{arm}** | mean (SD) | | {st['mean_cg']:.4f} ({st['sd_cg']:.4f}) | "
                 f"{st['mean_pz']:.4f} ({st['sd_pz']:.4f}) | {st['mean_macro']:.4f} ({st['sd_macro']:.4f}) |")
    L += ["", "## Step 2. Arm", "",
          f"2.5D minus 2D mean macro Dice = {diff:+.4f}; the prespecified margin is {MARGIN}. "
          f"Adopted arm: **{adopted}**.", "",
          "## Step 3. Ensemble and test-time augmentation", "",
          "| Model | CG Dice | PZ Dice | Macro Dice | CG HD95 (mm) | PZ HD95 (mm) |", "|---|---|---|---|---|---|"]
    for k in labels:
        r = scored[k]
        L.append(f"| {adopted} {k.replace('_', ' + ')} | {r['cg']:.4f} | {r['pz']:.4f} | {r['macro']:.4f} | "
                 f"{r['cg_hd95']:.2f} | {r['pz_hd95']:.2f} |")
    L += ["", f"Left-right flip TTA {'is kept' if use_tta else 'is not kept'} "
          f"({tta_res['macro']:.4f} vs {ens_res['macro']:.4f}).", "",
          "Reproducibility of step 1 (CPU re-score of the adopted seeds): "
          + ", ".join(f"{k} |diff| {v['abs_diff']:.5f}" for k, v in repro.items()) + ".", "",
          "## Final model (frozen)", "",
          f"{adopted} softmax ensemble of seeds {', '.join(map(str, SEEDS))}"
          f"{' with left-right flip TTA' if use_tta else ''}. Member weights (SHA-256):", ""]
    for m in final_members:
        L.append(f"- seed {m['seed']}, epoch {m['epoch']}: `{m['sha256']}`")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
