import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from common import write_jsonl
from medical_common import read_records, validate_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and split medical pseudo-labels")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument(
        "--preserve-desired-split", action="store_true",
        help="Use each input row's desired_split instead of creating a new stratified split.",
    )
    args = parser.parse_args()

    inputs = read_records(args.inputs)
    annotations = {row["id"]: row for row in read_records(args.annotations)}
    if set(row["id"] for row in inputs) != set(annotations):
        raise ValueError("input and annotation ids must match exactly")
    merged = []
    for source in inputs:
        annotation = annotations[source["id"]]
        labels = validate_labels(source["id"], source["words"], annotation["labels"])
        merged.append(source | {
            "labels": labels,
            "types": annotation.get("types", ["O"] * len(labels)),
            "selected_words": [word for word, label in zip(source["words"], labels) if label],
            "annotation_source": annotation.get("source", "unknown-teacher"),
        })

    rng = random.Random(args.seed)
    if args.preserve_desired_split:
        valid_names = {"train", "validation", "test"}
        unknown = sorted({row.get("desired_split") for row in merged} - valid_names)
        if unknown:
            raise ValueError(f"unknown or missing desired_split values: {unknown}")
        train = [row for row in merged if row["desired_split"] == "train"]
        validation = [row for row in merged if row["desired_split"] == "validation"]
        test = [row for row in merged if row["desired_split"] == "test"]
    else:
        groups = defaultdict(list)
        for row in merged:
            groups[str(row.get("task_label"))].append(row)
        train, validation, test = [], [], []
        for rows in groups.values():
            rng.shuffle(rows)
            validation_size = max(1, round(len(rows) * args.validation_fraction))
            test_size = max(1, round(len(rows) * args.test_fraction))
            validation.extend(rows[:validation_size])
            test.extend(rows[validation_size:validation_size + test_size])
            train.extend(rows[validation_size + test_size:])
    for split in (train, validation, test):
        rng.shuffle(split)

    output = Path(args.output_dir)
    write_jsonl(output / "all.jsonl", merged)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    write_jsonl(output / "test.jsonl", test)
    stats = {}
    for name, rows in (("all", merged), ("train", train), ("validation", validation), ("test", test)):
        tokens = sum(len(row["labels"]) for row in rows)
        selected = sum(sum(row["labels"]) for row in rows)
        stats[name] = {
            "examples": len(rows),
            "task_labels": dict(Counter(str(row.get("task_label")) for row in rows)),
            "tokens": tokens, "selected_tokens": selected,
            "mask_rate": selected / max(tokens, 1),
        }
    output.mkdir(parents=True, exist_ok=True)
    (output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
