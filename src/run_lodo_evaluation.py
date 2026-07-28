import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_TASKS = ("drug", "symptom2dx", "adr", "redditmh")
DEFAULT_MODELS = ("electra_small", "distilroberta")


def run_evaluator(
    script: Path,
    model_dir: Path,
    dataset: str,
    budget_threshold: float,
    f2_threshold: float,
    output: Path,
    batch_size: int,
    device: str,
) -> None:
    command = [
        sys.executable,
        str(script),
        "--model-dir",
        str(model_dir),
        "--dataset",
        dataset,
        "--source-budget-threshold",
        str(budget_threshold),
        "--source-f2-threshold",
        str(f2_threshold),
        "--output",
        str(output),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select thresholds on LODO source validation and evaluate held-out tests"
    )
    parser.add_argument("--data-root", default="data/medical_redactor/cross_dataset")
    parser.add_argument(
        "--artifact-root", default="artifacts/medical_redactor/cross_dataset_lodo"
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    artifact_root = Path(args.artifact_root)
    evaluator = Path(__file__).with_name("evaluate_medical_cross_dataset.py")

    for task in args.tasks:
        lodo_dir = data_root / "lodo" / f"heldout_{task}"
        target_dir = data_root / task / "prepared"
        for model_name in args.models:
            model_dir = artifact_root / f"heldout_{task}_{model_name}"
            source_output = model_dir / "source_threshold_evaluation.json"
            source_dataset = (
                f"source:{lodo_dir / 'source_validation.jsonl'}:"
                f"{lodo_dir / 'source_test.jsonl'}"
            )
            run_evaluator(
                evaluator, model_dir, source_dataset, 0.5, 0.5,
                source_output, args.batch_size, args.device,
            )
            source_result = json.loads(source_output.read_text(encoding="utf-8"))
            calibrated = source_result["datasets"]["source"]["target_calibrated_no_training"]
            budget_threshold = calibrated["budget_matched"]["validation"]["threshold"]
            f2_threshold = calibrated["f2_optimized"]["validation"]["threshold"]

            heldout_output = model_dir / "heldout_evaluation.json"
            heldout_dataset = (
                f"{task}:{target_dir / 'validation.jsonl'}:{target_dir / 'test.jsonl'}"
            )
            run_evaluator(
                evaluator, model_dir, heldout_dataset, budget_threshold, f2_threshold,
                heldout_output, args.batch_size, args.device,
            )
            heldout_result = json.loads(heldout_output.read_text(encoding="utf-8"))
            fixed = heldout_result["datasets"][task]["zero_shot_fixed"]
            budget = fixed["budget_matched_source_threshold"]
            f2 = fixed["f2_optimized_source_threshold"]
            print(
                f"LODO {task}/{model_name}: source thresholds="
                f"{budget_threshold:.2f}/{f2_threshold:.2f}; "
                f"budget P/R/F1={budget['precision']:.3f}/{budget['recall']:.3f}/"
                f"{budget['f1']:.3f}; F2-mode R/F2={f2['recall']:.3f}/{f2['f2']:.3f}"
            )


if __name__ == "__main__":
    main()
