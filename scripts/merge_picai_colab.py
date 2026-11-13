"""Merge the PI-CAI Colab runs into results/picai_psad/per_case.csv.

The final model ran on Colab in 3 runs from the same bundle: Run 7 (folds 0-1)
and Run 8 (fold 2) on one Colab account, and Run 9 (folds 3-4) on a second
account. Run 8 reached the free GPU limit after 90 studies of fold 3; Run 9 redid
fold 3 in full, and those 90 serve as a repeat check. Each run left, on Google Drive, one JSON per study in
cache_infer/, one conversion record per study in conversions/, a fold<k>.done
marker per finished fold, and its own notebook with the printed outputs.

This script checks, before writing anything:
  - each notebook printed a GPU, "bundle OK" for the same bundle SHA-256 that its
    source asserts, and the torch version (load_final_model() itself refuses
    member weights whose SHA-256 differs from selection.json, so the same bundle
    means the same frozen model);
  - every Colab conversion record equals the laptop's conversion of the same
    archive member (results/picai_psad/conversion.csv);
  - every study is taken from the run that its fold was assigned to below, and
    that run finished the fold; a study that another run also inferred is not
    used from there, but its volumes must equal the assigned run's;
  - every eligible study whose image converted is covered.

It then writes per_case.csv (the columns of `run_picai_psad.py --phase infer`),
model_identity.json (with the GPU of each run) and colab_merge.json (provenance).
`--check-only` runs every check on whatever is there and writes nothing, so it
can be tried while a run is still going.

usage:
  python scripts/merge_picai_colab.py --check-only
  python scripts/merge_picai_colab.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_picai_psad as rp  # noqa: E402

G = "G:/My Drive"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = [
    {"name": "Run 7", "folds": (0, 1), "dir": f"{G}/Prostate158_PICAI",
     "notebook": f"{G}/Sandro - Prostate158 PICAI/Run 7 - PI-CAI PSA density, final ensemble.ipynb"},
    {"name": "Run 8", "folds": (2,), "dir": f"{G}/Prostate158_E1_seeds/picai_folds234",
     "notebook": f"{G}/Sandro - Prostate158 E1 Colab runs/Run 8 - PI-CAI folds 2-4, final ensemble.ipynb"},
    # The second account's Drive is not mounted here: its results zip is unpacked into this folder.
    {"name": "Run 9", "folds": (3, 4), "dir": os.path.join(REPO, "results/colab_picai/run9_folds34"),
     "notebook": f"{G}/Sandro - Prostate158 E1 Colab runs/Run 9 - PI-CAI folds 3-4, final ensemble.ipynb"},
]
SAME_KEYS = ("shape", "cg_ml", "pz_ml", "wg_ml", "wg_ml_R1", "empty", "geometry_ok")
CONV_FIELDS = ("archive", "member", "shape", "spacing", "axcodes", "status")
IDENTITY_KEYS = ("kind", "arm", "seeds", "epochs", "member_sha256", "tta_flip_left_right",
                 "ensemble_id_sha256")


def notebook_facts(path: str) -> dict:
    """GPU, bundle and torch version as the notebook printed them, and the asserted bundle hash."""
    nb = json.load(open(path, encoding="utf-8"))
    src = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
    out = "\n".join("".join(o.get("text", [])) + "".join(o.get("data", {}).get("text/plain", []))
                    for c in nb["cells"] for o in c.get("outputs", []))
    want = re.search(r"BUNDLE_SHA256 = '[^']*', '([0-9a-f]{64})'", src)
    gpu = re.search(r"GPU 0: (.+?) \(UUID", out)
    ok = re.search(r"bundle OK: ([0-9a-f]{16})", out)
    torch = re.search(r"\| torch (\S+)", out)
    facts = {"notebook": os.path.basename(path), "bundle_sha256": want and want.group(1),
             "bundle_ok_printed": bool(ok and want and want.group(1).startswith(ok.group(1))),
             "gpu": gpu and gpu.group(1), "torch": torch and torch.group(1),
             "folds_done_printed": sorted(int(k) for k in re.findall(r"fold (\d): done", out))}
    return facts


def same_conversion(colab: dict, local: dict) -> bool:
    for k in CONV_FIELDS:
        a, b = colab.get(k), local.get(k)
        if isinstance(a, list):
            a = json.dumps(a)
        if str(a) != str(b):
            return False
    return float(colab["corner_max_diff_mm"]) == float(local["corner_max_diff_mm"])


def fold_of(archive: str) -> int:
    return int(re.search(r"fold(\d)\.zip$", archive).group(1))


def same_result(a: dict, b: dict) -> bool:
    """The same study inferred twice must give the same segmentation volumes."""
    return all(a.get(k) == b.get(k) for k in SAME_KEYS)


def main(check_only: bool) -> None:
    rp.check_lock()
    cohort = {r["case"]: r for r in rp.read_csv(os.path.join(rp.OUT, "cohort.csv"))}
    local = rp.conversions()
    expected = sorted(c for c in cohort if local.get(c, {}).get("status") == "ok")
    problems, rows, owner, runs, extras = [], {}, {}, [], []
    for run in RUNS:
        facts = notebook_facts(run["notebook"]) if os.path.exists(run["notebook"]) else {}
        folds = sorted(int(re.search(r"fold(\d)\.done$", f).group(1))
                       for f in glob.glob(os.path.join(run["dir"], "fold*.done")))
        cache = sorted(glob.glob(os.path.join(run["dir"], "cache_infer", "*.json")))
        info = {"name": run["name"], "dir": run["dir"], "folds_assigned": list(run["folds"]),
                "folds_done": folds, "n_cached": len(cache), **facts}
        runs.append(info)
        if not facts:
            problems.append(f"{run['name']}: notebook not found")
        elif not facts["bundle_ok_printed"] or not facts["gpu"]:
            problems.append(f"{run['name']}: notebook did not print a GPU and 'bundle OK'")
        for f in cache:
            r = json.load(open(f, encoding="utf-8"))
            case = r["case"]
            cj = os.path.join(run["dir"], "conversions", f"{case}.json")
            if case not in cohort:
                problems.append(f"{case}: inferred but not in the cohort")
                continue
            if fold_of(local[case]["archive"]) not in run["folds"]:
                extras.append((case, run["name"], r))  # another run owns this fold
                continue
            if not os.path.exists(cj):
                problems.append(f"{case}: no conversion record in {run['name']}")
            elif not same_conversion(json.load(open(cj, encoding="utf-8")), local[case]):
                problems.append(f"{case}: Colab conversion differs from the laptop's")
            if fold_of(local[case]["archive"]) not in folds:
                problems.append(f"{case}: fold {fold_of(local[case]['archive'])} is not finished in {run['name']}")
            if case in rows:
                problems.append(f"{case}: inferred by both {owner[case]} and {run['name']}")
            rows[case], owner[case] = r, run["name"]
    extra_same = 0
    for case, name, r in extras:
        if case in rows and not same_result(r, rows[case]):
            problems.append(f"{case}: {name} and {owner[case]} give different volumes")
        elif case in rows:
            extra_same += 1
    shas = {r.get("bundle_sha256") for r in runs}
    if len(shas) != 1:
        problems.append(f"the runs used different bundles: {shas}")
    missing = [c for c in expected if c not in rows]
    geom_fail = [c for c, r in rows.items() if not r["geometry_ok"]]
    empty = [c for c, r in rows.items() if r["empty"]]
    print(f"cases: {len(rows)} inferred of {len(expected)} expected; missing {len(missing)}")
    for r in runs:
        print(f"  {r['name']}: folds done {r['folds_done']}, cached {r['n_cached']}, "
              f"GPU {r.get('gpu')}, torch {r.get('torch')}, bundle {str(r.get('bundle_sha256'))[:16]}")
    print(f"geometry check failed: {len(geom_fail)}; empty segmentations: {len(empty)}")
    print(f"studies also inferred by a run that does not own their fold: {len(extras)}; "
          f"identical to the owner's result: {extra_same}")
    print(f"problems: {len(problems)}")
    for p_ in problems[:20]:
        print("  -", p_)
    if check_only:
        print("check only: nothing written")
        return
    if problems or missing:
        raise SystemExit("not merged: fix the problems above, or wait until every fold has finished")

    rp.write_csv(os.path.join(rp.OUT, "per_case.csv"), [rows[c] for c in sorted(rows)])
    cpu = json.load(open(os.path.join(rp.OUT, "cpu_check", "model_identity_cpu.json"), encoding="utf-8"))
    ident = {k: cpu[k] for k in IDENTITY_KEYS}
    for run in RUNS:  # written by the notebook's last cell, so only by a run that reached it
        run_ident = os.path.join(run["dir"], "model_identity.json")
        if os.path.exists(run_ident):
            got = json.load(open(run_ident, encoding="utf-8"))
            if any(got[k] != ident[k] for k in IDENTITY_KEYS):
                raise SystemExit(f"{run['name']} model_identity.json differs from the frozen model")
    ident.update({"device": "cuda", "gpus": sorted({r["gpu"] for r in runs}),
                  "torch": sorted({r["torch"] for r in runs}),
                  "bundle_sha256": shas.pop(), "runs": runs,
                  "note": "member hashes are checked against selection.json by load_final_model() "
                          "at every load; all runs loaded them from the same bundle"})
    json.dump(ident, open(os.path.join(rp.OUT, "model_identity.json"), "w"), indent=2)
    json.dump({"generated_utc": rp.now(), "n_cases": len(rows), "by_run": {
        r["name"]: sum(o == r["name"] for o in owner.values()) for r in runs},
        "n_geometry_failed": len(geom_fail), "n_empty": len(empty),
        "n_also_inferred_elsewhere": len(extras), "n_also_inferred_identical": extra_same, "runs": runs},
        open(os.path.join(rp.OUT, "colab_merge.json"), "w"), indent=2)
    print(f"wrote per_case.csv ({len(rows)} studies), model_identity.json, colab_merge.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check-only", action="store_true")
    main(ap.parse_args().check_only)
