import json
from pathlib import Path
from statistics import mean


ROOT = Path("artifacts/medical_redactor/core_matrix")
OLD_DATASETS = ["drug", "symptom2dx", "adr", "redditmh"]
NEW_DATASETS = ["mednli", "mentalhealth"]
MODELS = ["bert_tiny", "electra_small", "distilroberta"]
METRICS = ["precision", "recall", "f1", "f2", "gold_mask_rate", "predicted_mask_rate"]


def main() -> None:
    old = json.loads((ROOT / "seed42_summary.json").read_text(encoding="utf-8"))
    extension = json.loads(
        (ROOT / "extension_seed42" / "summary.json").read_text(encoding="utf-8")
    )
    datasets = {}
    for dataset in OLD_DATASETS:
        datasets[dataset] = {
            model: old["models"][model]["datasets"][dataset] for model in MODELS
        }
    for dataset in NEW_DATASETS:
        datasets[dataset] = {
            model: extension[f"{dataset}:{model}"] for model in MODELS
        }

    macro = {}
    for model in MODELS:
        macro[model] = {}
        for mode in ("budget_matched", "f2_optimized"):
            macro[model][mode] = {
                metric: mean(datasets[dataset][model][mode]["test"][metric]
                             for dataset in OLD_DATASETS + NEW_DATASETS)
                for metric in METRICS
            }
    output = {
        "protocol": (
            "Six in-domain datasets, seed 42, fixed train/validation/test; threshold selected "
            "on validation and applied to test; medterm4 pseudo-gold, not human-gold."
        ),
        "datasets": datasets,
        "macro": macro,
    }
    path = ROOT / "six_dataset_seed42_summary.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("| Dataset | BERT-tiny F1 | ELECTRA F1 | DistilRoBERTa F1 |")
    print("|---|---:|---:|---:|")
    for dataset in OLD_DATASETS + NEW_DATASETS:
        values = [datasets[dataset][model]["budget_matched"]["test"]["f1"] for model in MODELS]
        print(f"| {dataset} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |")
    values = [macro[model]["budget_matched"]["f1"] for model in MODELS]
    print(f"| Macro | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} |")
    print("\nNew-dataset details:")
    print("| Dataset | Model | P | R | F1 | F2 | Pred. mask | Privacy R | Privacy F2 |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for dataset in NEW_DATASETS:
        for model in MODELS:
            budget = datasets[dataset][model]["budget_matched"]["test"]
            privacy = datasets[dataset][model]["f2_optimized"]["test"]
            print(
                f"| {dataset} | {model} | {budget['precision']:.3f} | "
                f"{budget['recall']:.3f} | {budget['f1']:.3f} | {budget['f2']:.3f} | "
                f"{budget['predicted_mask_rate']:.2%} | {privacy['recall']:.3f} | "
                f"{privacy['f2']:.3f} |"
            )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
