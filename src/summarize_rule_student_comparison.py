import argparse
import json
import statistics
from pathlib import Path


TASKS = ("drug", "symptom2dx", "adr", "redditmh")
STUDENTS = ("bert_tiny", "electra_small", "distilroberta")
METRICS = ("precision", "recall", "f1", "f2", "predicted_mask_rate")


def scalar(value: float) -> dict:
    return {"mean": value, "sample_std": None}


def aggregate_seed_macros(per_seed: dict[str, dict], mode: str) -> dict:
    output = {}
    for metric in METRICS:
        values = [
            statistics.fmean(per_seed[seed][task][mode][metric] for task in TASKS)
            for seed in per_seed
        ]
        output[metric] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine accuracy and efficiency of rule/student redactors")
    parser.add_argument(
        "--core-summary", default="artifacts/medical_redactor/core_matrix/seed42_summary.json"
    )
    parser.add_argument(
        "--seed-summary", default="artifacts/medical_redactor/core_matrix/three_seed_summary.json"
    )
    parser.add_argument(
        "--efficiency-root", default="artifacts/medical_redactor/core_matrix/efficiency"
    )
    parser.add_argument(
        "--bert-warm-load", default="/tmp/bert_tiny_warm_load.json"
    )
    parser.add_argument(
        "--output", default="artifacts/medical_redactor/core_matrix/complete_rule_student_comparison.json"
    )
    args = parser.parse_args()

    core = json.loads(Path(args.core_summary).read_text(encoding="utf-8"))
    seeded = json.loads(Path(args.seed_summary).read_text(encoding="utf-8"))
    per_dataset = {}
    raw_per_seed = {model: {} for model in STUDENTS}

    for task in TASKS:
        bert = core["models"]["bert_tiny"]["datasets"][task]
        gold_mask_rate = bert["budget_matched"]["test"]["gold_mask_rate"]
        task_result = {
            "teacher_mask_rate": gold_mask_rate,
            "medterm4": {
                "seeds": [],
                "basis": "self-reference; not human-gold accuracy",
                "budget_matched": {
                    "precision": scalar(1.0), "recall": scalar(1.0),
                    "f1": scalar(1.0), "f2": scalar(1.0),
                    "predicted_mask_rate": scalar(gold_mask_rate),
                },
            },
        }
        bert_modes = {
            "budget_matched": bert["budget_matched"]["test"],
            "f2_optimized": bert["f2_optimized"]["test"],
        }
        raw_per_seed["bert_tiny"]["42"] = raw_per_seed["bert_tiny"].get("42", {})
        raw_per_seed["bert_tiny"]["42"][task] = bert_modes
        task_result["bert_tiny"] = {
            "seeds": [42],
            **{
                mode: {metric: scalar(values[metric]) for metric in METRICS}
                for mode, values in bert_modes.items()
            },
        }
        for model in ("electra_small", "distilroberta"):
            source = seeded["models"][model]["datasets"][task]
            task_result[model] = {
                "seeds": seeded["seeds"],
                "budget_matched": source["budget_matched"],
                "f2_optimized": source["f2_optimized"],
            }
            for seed, modes in source["per_seed"].items():
                raw_per_seed[model][seed] = raw_per_seed[model].get(seed, {})
                raw_per_seed[model][seed][task] = modes
        per_dataset[task] = task_result

    macro = {
        "medterm4": {
            "budget_matched": {
                "precision": scalar(1.0), "recall": scalar(1.0),
                "f1": scalar(1.0), "f2": scalar(1.0),
                "predicted_mask_rate": scalar(
                    statistics.fmean(per_dataset[task]["teacher_mask_rate"] for task in TASKS)
                ),
            }
        }
    }
    for model in STUDENTS:
        macro[model] = {
            mode: aggregate_seed_macros(raw_per_seed[model], mode)
            for mode in ("budget_matched", "f2_optimized")
        }

    efficiency = {}
    efficiency_root = Path(args.efficiency_root)
    for method in ("medterm4", *STUDENTS):
        payload = json.loads((efficiency_root / f"{method}.json").read_text(encoding="utf-8"))
        efficiency[method] = payload["metrics"]
    warm_path = Path(args.bert_warm_load)
    if warm_path.exists():
        efficiency["bert_tiny"]["warm_cache_load_seconds"] = json.loads(
            warm_path.read_text(encoding="utf-8")
        )["metrics"]["load_seconds"]

    result = {
        "accuracy_basis": (
            "Token agreement with deterministic medterm4 pseudo-teacher. "
            "medterm4 self-scores are 1.0 by definition and are not human-gold correctness."
        ),
        "accuracy_protocol": (
            "Fixed in-domain splits; validation-selected thresholds applied to fixed tests. "
            "BERT-tiny seed 42; ELECTRA/DistilRoBERTa seeds 42,43,44."
        ),
        "efficiency_protocol": (
            "Separate CPU processes; 1 thread, batch 1, 128 evenly sampled multi-domain test "
            "sentences, three repeats. Student input is padded/truncated to 256 tokens."
        ),
        "datasets": per_dataset,
        "macro": macro,
        "efficiency": efficiency,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}")
    print("| Method | P | R | F1 | F2 | Mask | Median ms | p95 ms | sent/s | Peak RSS |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for method in ("medterm4", *STUDENTS):
        metrics = macro[method]["budget_matched"]
        perf = efficiency[method]
        latency = perf["latency"]
        values = [metrics[name]["mean"] for name in ("precision", "recall", "f1", "f2")]
        print(
            f"| {method} | " + " | ".join(f"{value:.3f}" for value in values) +
            f" | {100 * metrics['predicted_mask_rate']['mean']:.2f}% | "
            f"{latency['median_ms']:.2f} | {latency['p95_ms']:.2f} | "
            f"{latency['sentences_per_second']:.1f} | {perf['peak_rss_mb']:.1f} MB |"
        )


if __name__ == "__main__":
    main()
