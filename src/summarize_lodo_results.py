import argparse
import json
from pathlib import Path


DEFAULT_TASKS = ("drug", "symptom2dx", "adr", "redditmh")
DEFAULT_MODELS = ("electra_small", "distilroberta")


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize held-out LODO evaluation JSON files")
    parser.add_argument(
        "--artifact-root", default="artifacts/medical_redactor/cross_dataset_lodo"
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--output", default="artifacts/medical_redactor/cross_dataset_lodo/summary.json"
    )
    args = parser.parse_args()

    root = Path(args.artifact_root)
    summary = {
        "protocol": (
            "Train on three source domains; select thresholds on pooled source validation; "
            "apply fixed thresholds to the unseen held-out test."
        ),
        "models": {},
    }
    for model in args.models:
        datasets = {}
        budget_f1 = []
        privacy_f2 = []
        for task in args.tasks:
            path = root / f"heldout_{task}_{model}" / "heldout_evaluation.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            fixed = payload["datasets"][task]["zero_shot_fixed"]
            budget = fixed["budget_matched_source_threshold"]
            privacy = fixed["f2_optimized_source_threshold"]
            datasets[task] = {
                "budget_matched": budget,
                "f2_optimized": privacy,
            }
            budget_f1.append(budget["f1"])
            privacy_f2.append(privacy["f2"])
        summary["models"][model] = {
            "macro_budget_f1": mean(budget_f1),
            "macro_privacy_f2": mean(privacy_f2),
            "datasets": datasets,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    for model, result in summary["models"].items():
        print(
            f"{model}: macro budget F1={result['macro_budget_f1']:.3f}, "
            f"macro privacy F2={result['macro_privacy_f2']:.3f}"
        )


if __name__ == "__main__":
    main()
