import argparse
import json
from pathlib import Path


TASKS = ("drug", "symptom2dx", "adr", "redditmh")
MODEL_DIRS = {
    "bert_tiny": "artifacts/medical_redactor/core_matrix/{task}_bert_tiny_seed42",
    "electra_small": "artifacts/medical_redactor/cross_dataset_in_domain/{task}_electra_small",
    "distilroberta": "artifacts/medical_redactor/cross_dataset_in_domain/{task}_distilroberta",
}


def load_result(path: Path, task: str) -> dict:
    payload = json.loads((path / "cross_evaluation.json").read_text(encoding="utf-8"))
    calibrated = payload["datasets"][task]["target_calibrated_no_training"]
    return {
        "budget_matched": calibrated["budget_matched"],
        "f2_optimized": calibrated["f2_optimized"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the fixed 3-model x 4-dataset matrix")
    parser.add_argument(
        "--output", default="artifacts/medical_redactor/core_matrix/seed42_summary.json"
    )
    args = parser.parse_args()

    summary = {
        "protocol": (
            "Same prepared in-domain split and medterm4 teacher per dataset; seed 42; "
            "threshold selected on each validation split and applied to its fixed test split."
        ),
        "models": {},
    }
    for model, template in MODEL_DIRS.items():
        datasets = {}
        f1_values = []
        f2_values = []
        for task in TASKS:
            result = load_result(Path(template.format(task=task)), task)
            datasets[task] = result
            f1_values.append(result["budget_matched"]["test"]["f1"])
            f2_values.append(result["f2_optimized"]["test"]["f2"])
        summary["models"][model] = {
            "macro_budget_f1": sum(f1_values) / len(f1_values),
            "macro_privacy_f2": sum(f2_values) / len(f2_values),
            "datasets": datasets,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    print("| Dataset | BERT-tiny | ELECTRA-small | DistilRoBERTa |")
    print("|---|---:|---:|---:|")
    for task in TASKS:
        values = [
            summary["models"][model]["datasets"][task]["budget_matched"]["test"]["f1"]
            for model in MODEL_DIRS
        ]
        print(f"| {task} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
    macro = [summary["models"][model]["macro_budget_f1"] for model in MODEL_DIRS]
    print("| Macro F1 | " + " | ".join(f"{value:.3f}" for value in macro) + " |")


if __name__ == "__main__":
    main()
