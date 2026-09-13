"""CLI entry point for baseline evaluation (Experiment 1) -- inference only.

Loads a trained checkpoint and evaluates it at the volume level, per case, in
original voxel space (the thesis protocol), writing per-case + cohort-summary
CSV/JSON. This does NOT train, retrain, or modify the checkpoint.

Typical Colab usage (checkpoint on Google Drive):

    python scripts/run_evaluation.py \
        --config configs/config_baseline.yaml \
        --checkpoint /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline/best_model.pt \
        --output-dir /content/drive/MyDrive/THESIS_PROSTATE158/results/exp01_baseline \
        --split val

The 19-case official test archive is not required for --split val. When that
archive is later added (data.test_csv/test_dir in the config), rerun with
--split test to get the final held-out numbers -- no code change needed.
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.evaluate import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline evaluation (inference only).")
    parser.add_argument(
        "--config",
        default="configs/config_baseline.yaml",
        help="Experiment YAML (dataset root, official split CSVs, target mask, default output dir).",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained checkpoint (e.g. the Drive-backed best_model.pt).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write result files (defaults to the config's experiment.output_dir).",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "validation", "test", "train"],
        help="Official split to evaluate. Default: val. 'test' requires the 19-case archive.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (cuda/mps/cpu). Auto-detected if omitted.",
    )
    args = parser.parse_args()

    result = run_evaluation(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
    )
    print("\nEvaluation complete.")
    print(f"  per-case CSV : {result['per_case_csv']}")
    print(f"  summary  CSV : {result['summary_csv']}")
    print(f"  summary JSON : {result['summary_json']}")


if __name__ == "__main__":
    main()
