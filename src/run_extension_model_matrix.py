import argparse
import json
import subprocess
import sys
from pathlib import Path


MODELS = {
    "bert_tiny": "prajjwal1/bert-tiny",
    "electra_small": "google/electra-small-discriminator",
    "distilroberta": "distilroberta-base",
}


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise SystemExit(f"command failed ({process.returncode}): {command}\n{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the fixed three-model matrix on extension datasets")
    parser.add_argument("--datasets", nargs="+", default=["mednli", "mentalhealth"])
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS), default=list(MODELS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reevaluate", action="store_true")
    parser.add_argument(
        "--data-root", default="data/medical_redactor/cross_dataset",
        help="Directory containing <dataset>/train.jsonl, validation.jsonl, and test.jsonl",
    )
    parser.add_argument(
        "--output-root", default="artifacts/medical_redactor/core_matrix/extension_seed42",
    )
    args = parser.parse_args()
    root = Path(args.output_root)
    summary = {}
    for dataset in args.datasets:
        data = Path(args.data_root) / dataset
        for short_name in args.models:
            model_name = MODELS[short_name]
            output = root / f"{dataset}_{short_name}_seed{args.seed}"
            train_log = root / "logs" / f"{dataset}_{short_name}_seed{args.seed}.log"
            eval_log = root / "logs" / f"{dataset}_{short_name}_seed{args.seed}_evaluation.log"
            evaluation = output / "medical_evaluation.json"
            if args.force or args.reevaluate or not evaluation.exists():
                if args.force or not (output / "model.pt").exists():
                    run([
                        sys.executable, "src/train.py",
                        "--train", str(data / "train.jsonl"),
                        "--validation", str(data / "validation.jsonl"),
                        "--output-dir", str(output),
                        "--model-name", model_name,
                        "--epochs", "5", "--batch-size", str(args.batch_size),
                        "--max-length", "256", "--seed", str(args.seed),
                        "--unfreeze-encoder", "--encoder-learning-rate", "2e-5",
                        "--head-learning-rate", "1e-3", "--device", args.device, "--offline",
                    ], train_log)
                run([
                    sys.executable, "src/evaluate_medical_student.py",
                    "--model-dir", str(output),
                    "--validation", str(data / "validation.jsonl"),
                    "--test", str(data / "test.jsonl"),
                    "--batch-size", str(args.batch_size), "--device", args.device,
                ], eval_log)
            result = json.loads(evaluation.read_text(encoding="utf-8"))
            summary[f"{dataset}:{short_name}"] = result
            budget = result["budget_matched"]["test"]
            privacy = result["f2_optimized"]["test"]
            print(
                f"DONE {dataset} {short_name}: "
                f"budget P/R/F1/F2={budget['precision']:.3f}/{budget['recall']:.3f}/"
                f"{budget['f1']:.3f}/{budget['f2']:.3f}; "
                f"privacy R/F2={privacy['recall']:.3f}/{privacy['f2']:.3f}",
                flush=True,
            )
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {root / 'summary.json'}")


if __name__ == "__main__":
    main()
