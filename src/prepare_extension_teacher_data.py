import argparse
import json
from collections import Counter
from pathlib import Path

from annotate_medterm4 import annotate, load_pipelines
from common import write_jsonl
from medical_common import read_records, validate_labels


DATASETS = ("mednli", "mentalhealth")


def stats(rows: list[dict]) -> dict:
    tokens = sum(len(row["labels"]) for row in rows)
    selected = sum(sum(row["labels"]) for row in rows)
    return {
        "examples": len(rows),
        "task_labels": dict(Counter(str(row.get("task_label")) for row in rows)),
        "tokens": tokens,
        "selected_tokens": selected,
        "mask_rate": selected / max(tokens, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate MedNLI and mentalhealth once, then assemble preserved splits",
    )
    parser.add_argument("--root", default="data/medical_redactor/cross_dataset")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    root = Path(args.root)
    inputs = []
    for dataset in DATASETS:
        inputs.extend(read_records(root / dataset / "input.jsonl"))
    if len(inputs) != len({row["id"] for row in inputs}):
        raise ValueError("duplicate ids across extension datasets")
    write_jsonl(root / "extension_input.jsonl", inputs)

    science, linker, pii = load_pipelines(args.threshold)
    annotations = []
    for number, row in enumerate(inputs, start=1):
        labels, types = annotate(row["text"], row["words"], science, linker, pii)
        annotations.append({
            "id": row["id"],
            "labels": labels,
            "types": types,
            "selected_words": [word for word, label in zip(row["words"], labels) if label],
            "source": "redactformer-medterm-v4-word-adapter@f2c601e3",
        })
        if number % 25 == 0:
            print(f"annotated {number}/{len(inputs)}")
    write_jsonl(root / "extension_medterm4_latest.jsonl", annotations)
    by_id = {row["id"]: row for row in annotations}

    for dataset in DATASETS:
        source_rows = [row for row in inputs if row["dataset_name"] == dataset]
        dataset_annotations = [by_id[row["id"]] for row in source_rows]
        write_jsonl(root / dataset / "medterm4_latest.jsonl", dataset_annotations)
        merged = []
        for source in source_rows:
            annotation = by_id[source["id"]]
            labels = validate_labels(source["id"], source["words"], annotation["labels"])
            merged.append(source | {
                "labels": labels,
                "types": annotation["types"],
                "selected_words": annotation["selected_words"],
                "annotation_source": annotation["source"],
            })
        splits = {
            name: [row for row in merged if row["desired_split"] == name]
            for name in ("train", "validation", "test")
        }
        write_jsonl(root / dataset / "all.jsonl", merged)
        for name, rows in splits.items():
            write_jsonl(root / dataset / f"{name}.jsonl", rows)
        report = {"all": stats(merged)} | {name: stats(rows) for name, rows in splits.items()}
        (root / dataset / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(dataset, json.dumps(report))


if __name__ == "__main__":
    main()
