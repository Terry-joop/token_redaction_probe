from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import StudentPredictor, evaluate_system, target_detected
from v14_rule_adapter import read_jsonl


def labels_by_id(path: str | Path, field: str = "labels") -> dict[str, list[int]]:
    return {row["pair_id"]: row[field] for row in read_jsonl(path)}


def logical_or(left: list[list[int]], right: list[list[int]]) -> list[list[int]]:
    return [
        [int(a or b) for a, b in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def target_by_noise(
    pairs: list[dict], clean: list[list[int]], noisy: list[list[int]]
) -> dict:
    output = {}
    for noise in sorted({row["noise_type"] for row in pairs}):
        indices = [i for i, row in enumerate(pairs) if row["noise_type"] == noise]
        clean_flags = []
        noisy_flags = []
        for index in indices:
            row = pairs[index]
            clean_flags.append(
                target_detected(
                    row["clean_text"], row["clean_words"], row["clean_labels"],
                    clean[index], row["clean_target"],
                )
            )
            noisy_flags.append(
                target_detected(
                    row["text"], row["words"], row["labels"],
                    noisy[index], row["noisy_target"],
                )
            )
        clean_rate = float(np.mean(clean_flags)) if clean_flags else 0.0
        noisy_rate = float(np.mean(noisy_flags)) if noisy_flags else 0.0
        output[noise] = {
            "pairs": len(indices),
            "clean_target_detection": clean_rate,
            "noisy_target_detection": noisy_rate,
            "detection_drop": clean_rate - noisy_rate,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw rule, normalized rule, student, and rule-OR-student"
    )
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--raw-rule-cache", required=True)
    parser.add_argument("--normalized-rule-cache", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    raw_noisy_by_id = labels_by_id(args.raw_rule_cache)
    norm_rows = {row["pair_id"]: row for row in read_jsonl(args.normalized_rule_cache)}
    raw_clean = [row["clean_labels"] for row in pairs]
    raw_noisy = [raw_noisy_by_id[row["pair_id"]] for row in pairs]
    norm_clean = [norm_rows[row["pair_id"]]["clean_labels"] for row in pairs]
    norm_noisy = [norm_rows[row["pair_id"]]["noisy_labels"] for row in pairs]

    predictor = StudentPredictor(args.model_dir, args.device, None)
    clean_by_source = {}
    for row in pairs:
        clean_by_source.setdefault(row["source_id"], row["clean_words"])
    source_ids = list(clean_by_source)
    values = predictor.predict_many(
        [clean_by_source[source_id] for source_id in source_ids], args.batch_size
    )
    clean_cache = dict(zip(source_ids, values))
    student_clean = [clean_cache[row["source_id"]] for row in pairs]
    student_noisy = predictor.predict_many(
        [row["words"] for row in pairs], args.batch_size
    )

    predictions = {
        "raw_rule": (raw_clean, raw_noisy),
        "normalized_rule": (norm_clean, norm_noisy),
        "student": (student_clean, student_noisy),
        "hybrid_raw_rule_or_student": (
            logical_or(raw_clean, student_clean),
            logical_or(raw_noisy, student_noisy),
        ),
    }
    systems = {}
    for name, (clean, noisy) in predictions.items():
        systems[name] = evaluate_system(pairs, clean, noisy)
        systems[name]["target_by_noise"] = target_by_noise(pairs, clean, noisy)

    result = {
        "pairs": len(pairs),
        "unique_source_rows": len(clean_by_source),
        "seed": args.seed,
        "student_threshold": predictor.threshold,
        "gold": "clean RedactFormer v1.4 labels projected through deterministic edits",
        "hybrid_definition": "raw_rule OR student at stable-word level",
        "systems": systems,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "pairs": len(pairs),
                "seed": args.seed,
                "noisy_f2": {
                    name: round(value["noisy"]["f2"], 4)
                    for name, value in systems.items()
                },
                "noisy_target_detection": {
                    name: round(value["robustness"]["noisy_target_detection"], 4)
                    for name, value in systems.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
