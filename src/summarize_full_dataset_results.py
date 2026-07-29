"""Collect the fixed 3-model x 10-dataset full-data experiment results.

The full rerun reuses the earlier Symptom2Dx result because that pilot already
used every usable example and the regenerated split is byte-for-byte
equivalent.  Every other dataset is read from ``artifacts/full_redactor``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    "drug",
    "symptom2dx",
    "adr",
    "redditmh",
    "mednli",
    "mentalhealth",
    "bios",
    "mrpc",
    "qnli",
    "finphrasebank",
]
MODELS = ["bert_tiny", "electra_small", "distilroberta"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default="artifacts/full_redactor/seed42")
    parser.add_argument("--data-root", default="data/full_redactor")
    parser.add_argument("--output", default="reports/full_dataset_results.json")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    artifact_root = ROOT / args.artifact_root
    data_root = ROOT / args.data_root
    previous = read_json(
        ROOT / "artifacts/medical_redactor/core_matrix/six_dataset_seed42_summary.json"
    )["datasets"]
    result: dict = {
        "protocol": (
            "Fixed three-model in-domain comparison on every usable deduplicated "
            "example; official splits preserved where available; seed 42; "
            "threshold selected on validation and applied once to test."
        ),
        "data_scope": "full_usable_deduplicated",
        "seed": 42,
        "datasets": {},
        "missing": [],
    }

    for dataset in DATASETS:
        stats_path = data_root / dataset / "stats.json"
        if not stats_path.exists():
            result["missing"].append(f"{dataset}:stats")
            continue
        entry = {
            "stats": read_json(stats_path),
            "models": {},
            "result_source": "full_rerun",
        }
        for model in MODELS:
            if dataset == "symptom2dx":
                entry["models"][model] = previous[dataset][model]
                entry["result_source"] = "reused_full_coverage_pilot"
                continue
            evaluation = (
                artifact_root / f"{dataset}_{model}_seed42" / "medical_evaluation.json"
            )
            if evaluation.exists():
                entry["models"][model] = read_json(evaluation)
            else:
                result["missing"].append(f"{dataset}:{model}")
        result["datasets"][dataset] = entry

    if result["missing"] and not args.allow_partial:
        raise SystemExit("Missing full results: " + ", ".join(result["missing"]))

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    completed = sum(len(x["models"]) for x in result["datasets"].values())
    examples = sum(x["stats"]["all"]["examples"] for x in result["datasets"].values())
    print(
        f"wrote {output}: {completed}/30 runs, "
        f"{examples:,} examples across {len(result['datasets'])}/10 datasets"
    )


if __name__ == "__main__":
    main()
