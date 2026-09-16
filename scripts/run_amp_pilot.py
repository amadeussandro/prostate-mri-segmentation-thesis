"""AMP pilot runner (FP32 vs AMP), isolated from all official experiments.

Orchestration only -- all logic lives in src/. For ONE pilot arm it:
  1. hard-refuses to run unless the config's output_dir is under results/pilot_amp
     (never results/exp01_baseline/ or results/exp02_preprocessing/);
  2. CHECKPOINT-SAFE dispatch (see resolve_pilot_run): if a checkpoint already
     exists in the output dir it RESUMES from epoch+1 -- it never silently
     restarts at epoch 1 and never silently overwrites an existing checkpoint;
  3. snapshots the resolved config + runtime info into that dir;
  4. trains via src.train.run_training(..., resume_from=..., history_path=...) --
     honoring the config's `training.amp` flag -- writing best_model.pt + a
     per-epoch history;
  5. runs the FINAL volume-level, per-case, original-voxel-space validation
     evaluation in FP32 (src.evaluate.run_evaluation, split='val') on all 20
     validation patients, writing eval_val_* files into the same dir.

INCIDENT FIX: the previous version called run_training() with no resume_from, so
every rerun started at epoch 1 and overwrote the existing best_model.pt (this is
how an epoch-12 pilot checkpoint was lost). It now inspects the checkpoint first
and resumes; a truly fresh run over an existing checkpoint requires an explicit
--force-fresh, and --fresh REFUSES rather than clobber.

Typical use (run each arm once; NOT run automatically):
    python scripts/run_amp_pilot.py --config configs/experiments/pilot_amp_fp32.yaml
    python scripts/run_amp_pilot.py --config configs/experiments/pilot_amp_amp.yaml
Rerun the SAME command after a timeout -> it resumes automatically.

This script never touches Exp01/Exp02 outputs or their checkpoints.
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

# Pilot output dirs must contain this segment and none of the official ones.
# "results/pilot_amp" is a substring of both "results/pilot_amp/..." and
# "results/pilot_amp_v2/..." so both pilot generations are accepted.
ALLOWED_PREFIX = "results/pilot_amp"
FORBIDDEN_SUBSTRINGS = ("exp01_baseline", "exp02_preprocessing", "exp03_", "exp04_")


def _assert_isolated_output_dir(output_dir: str) -> str:
    """Refuse to proceed unless output_dir is under a results/pilot_amp* dir and
    does not touch any official experiment tree. Hard guardrail that keeps the
    pilot from ever overwriting Exp01/Exp02 outputs."""
    norm = output_dir.replace("\\", "/").lstrip("./").rstrip("/")
    if ALLOWED_PREFIX not in norm:
        raise SystemExit(
            f"REFUSING TO RUN: pilot output_dir '{output_dir}' is not under "
            f"'{ALLOWED_PREFIX}*/'. The AMP pilot must be fully isolated."
        )
    for bad in FORBIDDEN_SUBSTRINGS:
        if bad in norm:
            raise SystemExit(
                f"REFUSING TO RUN: pilot output_dir '{output_dir}' points at an "
                f"official experiment tree ('{bad}'). Aborting to protect it."
            )
    return norm


def resolve_pilot_run(checkpoint_path, num_epochs, fresh=False, force_fresh=False):
    """Decide -- WITHOUT training or mutating anything -- how a pilot arm should
    proceed, so an existing checkpoint can NEVER be silently overwritten.

    Returns {"mode", "resume_from", "plan", "message"} where mode is:
      "fresh"        -- no checkpoint present; start at epoch 1.
      "resume"       -- checkpoint at epoch < num_epochs; continue at epoch+1.
      "complete"     -- checkpoint epoch >= num_epochs; skip training (eval only).
      "fresh_forced" -- caller passed --force-fresh; start at epoch 1 EVEN IF a
                        checkpoint exists (it will be overwritten on improvement).

    Raises SystemExit (clear failure, no training) when:
      - the checkpoint is corrupt/incompatible; or
      - --fresh was requested but a checkpoint already exists (would overwrite) --
        the caller must use a new output dir or pass --force-fresh.
    """
    from src.train import plan_training_run

    plan = plan_training_run(checkpoint_path, num_epochs)
    action = plan["action"]

    if action == "corrupt":
        raise SystemExit("REFUSING TO RUN (corrupt/incompatible checkpoint):\n" + plan["message"])

    if force_fresh:
        return {
            "mode": "fresh_forced", "resume_from": None, "plan": plan,
            "message": (
                "FORCE-FRESH requested: starting at epoch 1. Any existing checkpoint in "
                "this output dir WILL be overwritten once validation improves."
            ),
        }

    if fresh:
        if action in ("resume", "already_complete"):
            raise SystemExit(
                "REFUSING FRESH RUN: a checkpoint already exists at\n"
                f"  {checkpoint_path}\n"
                f"(epoch {plan['checkpoint_epoch']}). A --fresh run would overwrite it.\n"
                "Use a NEW --output via the config's experiment.output_dir, or pass "
                "--force-fresh to intentionally overwrite."
            )
        return {"mode": "fresh", "resume_from": None, "plan": plan, "message": plan["message"]}

    # Default: resume if a checkpoint exists, else fresh.
    if action == "fresh":
        return {"mode": "fresh", "resume_from": None, "plan": plan, "message": plan["message"]}
    if action == "resume":
        return {"mode": "resume", "resume_from": checkpoint_path, "plan": plan, "message": plan["message"]}
    # already_complete
    return {"mode": "complete", "resume_from": None, "plan": plan, "message": plan["message"]}


def _print_status(arm_label, out, checkpoint_path, num_epochs, decision):
    plan = decision["plan"]
    info = plan.get("checkpoint_info", {})
    found = info.get("exists", False)
    print("=" * 70)
    print(f"{arm_label}")
    print(f"Output: {out}")
    if not found:
        print("Checkpoint: NOT FOUND")
    elif decision["mode"] == "complete":
        print(f"Checkpoint: FOUND (epoch {plan['checkpoint_epoch']}/{num_epochs})")
    elif info.get("resumable"):
        print(f"Checkpoint: FOUND (epoch {plan['checkpoint_epoch']}/{num_epochs}, "
              f"best {info.get('best_metric')})")
    else:
        print(f"Checkpoint: FOUND but UNUSABLE ({info.get('error')})")
    action_label = {
        "fresh": "FRESH", "resume": "RESUME", "complete": "ALREADY COMPLETE (eval only)",
        "fresh_forced": "FRESH (FORCED OVERWRITE)",
    }[decision["mode"]]
    print(f"Action: {action_label}")
    if decision["mode"] == "resume":
        print(f"Next epoch: {plan['start_epoch']}")
    print(f"Epochs: {num_epochs}")
    print("=" * 70)
    print(decision["message"])


def main():
    ap = argparse.ArgumentParser(description="AMP pilot runner (isolated, checkpoint-safe).")
    ap.add_argument("--config", required=True, help="pilot config (pilot_amp_fp32.yaml or pilot_amp_amp.yaml)")
    ap.add_argument("--skip-eval", action="store_true", help="train only; skip the final FP32 validation evaluation")
    ap.add_argument("--fresh", action="store_true",
                    help="require a fresh run; REFUSES (does not overwrite) if a checkpoint already exists")
    ap.add_argument("--force-fresh", action="store_true",
                    help="EXPLICITLY start at epoch 1 even if a checkpoint exists (will overwrite on improvement)")
    args = ap.parse_args()

    config = load_config(args.config)
    output_dir = config.get("experiment", {}).get("output_dir", "")
    _assert_isolated_output_dir(output_dir)
    out = ensure_dir(output_dir)

    device = get_device()
    num_epochs = int(config.get("training", {}).get("num_epochs", 0))
    checkpoint = os.path.join(out, "best_model.pt")
    arm_label = "AMP PILOT" if config.get("training", {}).get("amp", False) else "FP32 PILOT"

    # --- CHECKPOINT-SAFE decision BEFORE anything is written or trained ---
    decision = resolve_pilot_run(checkpoint, num_epochs, fresh=args.fresh, force_fresh=args.force_fresh)
    _print_status(arm_label, out, checkpoint, num_epochs, decision)

    # --- runtime info + config snapshot (provenance; never touches best_model.pt) ---
    try:
        import torch
        runtime = {
            "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "torch": torch.__version__,
            "device": device,
            "cuda_device": (torch.cuda.get_device_name(0) if device == "cuda" else None),
            "amp_config": bool(config.get("training", {}).get("amp", False)),
            "config_path": args.config,
            "run_mode": decision["mode"],
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

    # --- train (unless already complete) ---
    from src.train import run_training

    history_path = os.path.join(out, "pilot_history.json")
    if decision["mode"] == "complete":
        print("\nTraining already complete for this arm -- skipping training, running eval only.")
    else:
        summary = run_training(args.config, resume_from=decision["resume_from"], history_path=history_path)
        print("Training summary:", summary)

    # --- final FP32 volume-level validation evaluation (all 20 val patients) ---
    if not args.skip_eval:
        from src.evaluate import run_evaluation

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
