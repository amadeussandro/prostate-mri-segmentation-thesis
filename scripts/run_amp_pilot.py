"""AMP pilot runner (FP32 vs AMP), isolated from all official experiments.

Orchestration only -- all logic lives in src/. For ONE pilot arm it:
  1. hard-refuses to run unless the config's output_dir is under results/pilot_amp/
     (never results/exp01_baseline/ or results/exp02_preprocessing/);
  2. snapshots the resolved config + runtime info into that dir;
  3. trains via src.train.run_training(..., history_path=...) -- honoring the
     config's `training.amp` flag -- writing best_model.pt + a per-epoch history;
  4. runs the FINAL volume-level, per-case, original-voxel-space validation
     evaluation in FP32 (src.evaluate.run_evaluation, split='val') on all 20
     validation patients, writing eval_val_* files into the same dir.

The whole point of the pilot is to compare the FP32 and AMP arms' segmentation
metrics (Dice/IoU/HD95/ASD) to check whether AMP is a confound for Experiment 2.

Typical use (run each arm once; NOT run automatically):
    python scripts/run_amp_pilot.py --config configs/experiments/pilot_amp_fp32.yaml
    python scripts/run_amp_pilot.py --config configs/experiments/pilot_amp_amp.yaml

This script never touches Exp01/Exp02 outputs or the Exp02-A checkpoint.
"""

import argparse
import datetime
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import ensure_dir, get_device, load_config  # noqa: E402

ALLOWED_PREFIX = "results/pilot_amp"
FORBIDDEN_SUBSTRINGS = ("exp01_baseline", "exp02_preprocessing", "exp03_", "exp04_")


def _assert_isolated_output_dir(output_dir: str) -> str:
    """Refuse to proceed unless output_dir is under results/pilot_amp/ and does
    not touch any official experiment tree. This is the hard guardrail that
    keeps the pilot from ever overwriting Exp01/Exp02 outputs."""
    norm = output_dir.replace("\\", "/").lstrip("./").rstrip("/")
    if ALLOWED_PREFIX not in norm:
        raise SystemExit(
            f"REFUSING TO RUN: pilot output_dir '{output_dir}' is not under "
            f"'{ALLOWED_PREFIX}/'. The AMP pilot must be fully isolated."
        )
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad in norm:
            raise SystemExit(
                f"REFUSING TO RUN: pilot output_dir '{output_dir}' points at an "
                f"official experiment tree ('{bad}'). Aborting to protect it."
            )
    return norm


def main():
    ap = argparse.ArgumentParser(description="AMP pilot runner (isolated FP32/AMP arm).")
    ap.add_argument("--config", required=True, help="pilot config (pilot_amp_fp32.yaml or pilot_amp_amp.yaml)")
    ap.add_argument("--skip-eval", action="store_true", help="train only; skip the final FP32 validation evaluation")
    args = ap.parse_args()

    config = load_config(args.config)
    output_dir = config.get("experiment", {}).get("output_dir", "")
    _assert_isolated_output_dir(output_dir)
    out = ensure_dir(output_dir)

    device = get_device()

    # --- runtime info + config snapshot (for provenance) ---
    try:
        import torch
        runtime = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "torch": torch.__version__,
            "device": device,
            "cuda_device": (torch.cuda.get_device_name(0) if device == "cuda" else None),
            "amp_config": bool(config.get("training", {}).get("amp", False)),
            "config_path": args.config,
        }
    except ImportError:
        runtime = {"error": "torch not available"}
    with open(os.path.join(out, "runtime_info.json"), "w", encoding="utf-8") as f:
        json.dump(runtime, f, indent=2)
    try:
        import yaml
        with open(os.path.join(out, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(config, f)
    except Exception as exc:  # noqa: BLE001
        print(f"(warning) could not write config snapshot: {exc}")

    print("=" * 70)
    print("AMP PILOT RUN (isolated)")
    print(f"  config     : {args.config}")
    print(f"  output_dir : {out}")
    print(f"  amp        : {runtime.get('amp_config')}")
    print(f"  device     : {device}")
    print("=" * 70)

    # --- train (honors training.amp; writes best_model.pt + pilot_history.json) ---
    from src.train import run_training

    history_path = os.path.join(out, "pilot_history.json")
    summary = run_training(args.config, history_path=history_path)
    print("Training summary:", summary)

    # --- final FP32 volume-level validation evaluation (all 20 val patients) ---
    if not args.skip_eval:
        from src.evaluate import run_evaluation

        checkpoint = os.path.join(out, "best_model.pt")
        if not os.path.exists(checkpoint):
            raise SystemExit(f"Expected checkpoint not found after training: {checkpoint}")
        print("\nFinal FP32 volume-level validation evaluation (split='val')...")
        result = run_evaluation(
            config_path=args.config,
            checkpoint_path=checkpoint,
            output_dir=out,
            split="val",
        )
        print("Eval outputs:")
        print("  per-case CSV:", result["per_case_csv"])
        print("  summary  CSV:", result["summary_csv"])
        print("  summary JSON:", result["summary_json"])

    print("\nPilot arm complete. History:", history_path)


if __name__ == "__main__":
    main()
