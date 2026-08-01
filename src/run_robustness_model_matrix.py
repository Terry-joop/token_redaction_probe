from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASETS = {
    "drug": "medterm5",
    "symptom2dx": "medterm5",
    "adr": "medterm5",
    "redditmh": "medterm5",
    "mednli": "medterm5",
    "mentalhealth": "medterm5",
    "bios": "piiclean2",
    "mrpc": "piiclean2",
    "qnli": "piiclean2",
    "finphrasebank": "piiclean2",
}
MODELS = {
    "bert_tiny": "prajjwal1/bert-tiny",
    "electra_small": "google/electra-small-discriminator",
    "distilroberta": "distilroberta-base",
}


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
        raise SystemExit(
            f"command failed ({process.returncode}): {command}\n{tail}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the v1.4 three-model surface-robustness matrix"
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = Path("artifacts/robustness/v14")
    for dataset in args.datasets:
        data = Path("data/robustness/v14") / dataset
        pairs = data / (
            "robustness_pairs.jsonl"
            if (data / "robustness_pairs.jsonl").exists()
            else "paired_noisy_test.jsonl"
        )
        for short_name in args.models:
            model_dir = root / f"{dataset}_{short_name}_seed{args.seed}"
            train_log = root / "logs" / f"{dataset}_{short_name}_train.log"
            clean_log = root / "logs" / f"{dataset}_{short_name}_clean.log"
            robust_log = root / "logs" / f"{dataset}_{short_name}_robustness.log"
            result = root / f"{dataset}_{short_name}_results.json"
            rule_cache = root / "rule_cache" / f"{dataset}.jsonl"

            if args.force or not (model_dir / "model.pt").exists():
                run(
                    [
                        sys.executable,
                        "src/train.py",
                        "--train",
                        str(data / "train.jsonl"),
                        "--validation",
                        str(data / "validation.jsonl"),
                        "--output-dir",
                        str(model_dir),
                        "--model-name",
                        MODELS[short_name],
                        "--epochs",
                        "5",
                        "--batch-size",
                        "32",
                        "--max-length",
                        "128",
                        "--seed",
                        str(args.seed),
                        "--unfreeze-encoder",
                        "--encoder-learning-rate",
                        "2e-5",
                        "--head-learning-rate",
                        "1e-3",
                        "--device",
                        args.device,
                        "--offline",
                    ],
                    train_log,
                )
            if args.force or not (model_dir / "medical_evaluation.json").exists():
                run(
                    [
                        sys.executable,
                        "src/evaluate_medical_student.py",
                        "--model-dir",
                        str(model_dir),
                        "--validation",
                        str(data / "validation.jsonl"),
                        "--test",
                        str(data / "test.jsonl"),
                        "--batch-size",
                        "256",
                        "--device",
                        args.device,
                    ],
                    clean_log,
                )
            if args.force or not result.exists():
                run(
                    [
                        sys.executable,
                        "src/robustness/evaluate.py",
                        "--pairs",
                        str(pairs),
                        "--model-dir",
                        str(model_dir),
                        "--masker",
                        DATASETS[dataset],
                        "--task",
                        dataset,
                        "--noise-group",
                        "all",
                        "--bootstrap-repeats",
                        "2000",
                        "--student-batch-size",
                        "256",
                        "--rule-cache",
                        str(rule_cache),
                        "--seed",
                        str(args.seed),
                        "--output",
                        str(result),
                    ],
                    robust_log,
                )
            print(f"DONE {dataset} {short_name}", flush=True)


if __name__ == "__main__":
    main()
