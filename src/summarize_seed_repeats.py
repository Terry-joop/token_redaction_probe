import argparse
import json
import statistics
from pathlib import Path


TASKS = ("drug", "symptom2dx", "adr", "redditmh")
MODELS = ("electra_small", "distilroberta")
SEEDS = (42, 43, 44)
METRICS = ("precision", "recall", "f1", "f2", "predicted_mask_rate")


def model_dir(task: str, model: str, seed: int) -> Path:
    if seed == 42:
        return Path(f"artifacts/medical_redactor/cross_dataset_in_domain/{task}_{model}")
    return Path(
        f"artifacts/medical_redactor/core_matrix/seed_repeats/{task}_{model}_seed{seed}"
    )


def load_test(task: str, model: str, seed: int) -> dict:
    path = model_dir(task, model, seed) / "cross_evaluation.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    calibrated = payload["datasets"][task]["target_calibrated_no_training"]
    return {
        "budget_matched": calibrated["budget_matched"]["test"],
        "f2_optimized": calibrated["f2_optimized"]["test"],
    }


def aggregate(rows: list[dict]) -> dict:
    return {
        metric: {
            "mean": statistics.fmean(row[metric] for row in rows),
            "sample_std": statistics.stdev(row[metric] for row in rows),
        }
        for metric in METRICS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate three-seed in-domain experiments")
    parser.add_argument(
        "--output", default="artifacts/medical_redactor/core_matrix/three_seed_summary.json"
    )
    args = parser.parse_args()

    output_payload = {
        "protocol": (
            "Fixed train/validation/test splits and hyperparameters; only training seed changes. "
            "Threshold is independently selected on validation for each trained model."
        ),
        "seeds": list(SEEDS),
        "models": {},
    }
    for model in MODELS:
        datasets = {}
        macro_f1_by_seed = []
        macro_f2_by_seed = []
        raw_by_task = {
            task: {seed: load_test(task, model, seed) for seed in SEEDS}
            for task in TASKS
        }
        for task in TASKS:
            budget_rows = [raw_by_task[task][seed]["budget_matched"] for seed in SEEDS]
            privacy_rows = [raw_by_task[task][seed]["f2_optimized"] for seed in SEEDS]
            datasets[task] = {
                "budget_matched": aggregate(budget_rows),
                "f2_optimized": aggregate(privacy_rows),
                "per_seed": {str(seed): raw_by_task[task][seed] for seed in SEEDS},
            }
        for seed in SEEDS:
            macro_f1_by_seed.append(
                statistics.fmean(raw_by_task[task][seed]["budget_matched"]["f1"] for task in TASKS)
            )
            macro_f2_by_seed.append(
                statistics.fmean(raw_by_task[task][seed]["f2_optimized"]["f2"] for task in TASKS)
            )
        output_payload["models"][model] = {
            "macro_budget_f1": {
                "mean": statistics.fmean(macro_f1_by_seed),
                "sample_std": statistics.stdev(macro_f1_by_seed),
                "per_seed": dict(zip(map(str, SEEDS), macro_f1_by_seed)),
            },
            "macro_privacy_f2": {
                "mean": statistics.fmean(macro_f2_by_seed),
                "sample_std": statistics.stdev(macro_f2_by_seed),
                "per_seed": dict(zip(map(str, SEEDS), macro_f2_by_seed)),
            },
            "datasets": datasets,
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    print("| Dataset | ELECTRA-small F1 | DistilRoBERTa F1 |")
    print("|---|---:|---:|")
    for task in TASKS:
        values = []
        for model in MODELS:
            result = output_payload["models"][model]["datasets"][task]["budget_matched"]["f1"]
            values.append(f"{result['mean']:.3f} ± {result['sample_std']:.3f}")
        print(f"| {task} | " + " | ".join(values) + " |")
    macro = []
    for model in MODELS:
        result = output_payload["models"][model]["macro_budget_f1"]
        macro.append(f"{result['mean']:.3f} ± {result['sample_std']:.3f}")
    print("| Macro F1 | " + " | ".join(macro) + " |")


if __name__ == "__main__":
    main()
