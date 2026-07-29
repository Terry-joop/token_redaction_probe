import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import write_jsonl
from medical_common import read_records, validate_labels


DATASETS = ("drug", "symptom2dx", "adr", "redditmh", "mednli", "mentalhealth")


def stratified_split(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("task_label"))].append(row)
    output = {"train": [], "validation": [], "test": []}
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
        validation_size = max(1, round(len(group) * 0.1))
        test_size = max(1, round(len(group) * 0.1))
        output["validation"].extend(group[:validation_size])
        output["test"].extend(group[validation_size:validation_size + test_size])
        output["train"].extend(group[validation_size + test_size:])
    for rows in output.values():
        rng.shuffle(rows)
    return output


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
    parser = argparse.ArgumentParser(description="Combine and assemble full medical redactor data")
    parser.add_argument("--root", default="data/full_redactor")
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    root = Path(args.root)
    all_inputs = []
    for dataset in args.datasets:
        rows = read_records(root / dataset / "input.jsonl")
        if any(row.get("dataset_name") != dataset for row in rows):
            raise ValueError(f"{dataset}: dataset_name mismatch")
        all_inputs.extend(rows)
    ids = [row["id"] for row in all_inputs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids across medical inputs")
    combined_path = root / "medical_input.jsonl"
    write_jsonl(combined_path, all_inputs)
    print(f"combined {len(all_inputs)} rows into {combined_path}")

    teacher_path = Path(args.teacher) if args.teacher else root / "medical_teacher.jsonl"
    if not teacher_path.exists():
        print(f"teacher not found yet: {teacher_path}; combine-only stage complete")
        return
    annotations = {row["id"]: row for row in read_records(teacher_path)}
    if set(ids) != set(annotations):
        missing = len(set(ids) - set(annotations))
        extra = len(set(annotations) - set(ids))
        raise ValueError(f"teacher/input id mismatch: missing={missing} extra={extra}")

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for source in all_inputs:
        annotation = annotations[source["id"]]
        labels = validate_labels(source["id"], source["words"], annotation["labels"])
        by_dataset[source["dataset_name"]].append(source | {
            "labels": labels,
            "types": annotation.get("types", ["O"] * len(labels)),
            "selected_words": annotation.get("selected_words", []),
            "annotation_source": annotation.get("source", "medterm4"),
        })
    for dataset in args.datasets:
        rows = by_dataset[dataset]
        if rows and all(row.get("desired_split") in {"train", "validation", "test"} for row in rows):
            splits = {
                split: [row for row in rows if row["desired_split"] == split]
                for split in ("train", "validation", "test")
            }
        else:
            splits = stratified_split(rows, args.seed)
        path = root / dataset
        write_jsonl(path / "all.jsonl", rows)
        for split, split_rows in splits.items():
            write_jsonl(path / f"{split}.jsonl", split_rows)
        report = {"all": stats(rows)} | {
            split: stats(split_rows) for split, split_rows in splits.items()
        }
        (path / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(dataset, {split: len(value) for split, value in splits.items()})


if __name__ == "__main__":
    main()
