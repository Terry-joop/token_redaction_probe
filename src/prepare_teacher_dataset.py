import argparse
import json
import random
from collections import Counter
from pathlib import Path

from common import write_jsonl


def load_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and split teacher annotations")
    parser.add_argument("--inputs", default="teacher/chatgpt_input_1000.jsonl")
    parser.add_argument(
        "--annotations", default="data/sst2_teacher_annotations_1000_validated.jsonl"
    )
    parser.add_argument("--output-dir", default="data/teacher_prepared")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inputs = load_jsonl(args.inputs)
    annotations = load_jsonl(args.annotations)
    input_by_id = {row["id"]: row for row in inputs}
    annotation_by_id = {row["id"]: row for row in annotations}
    if len(input_by_id) != len(inputs) or len(annotation_by_id) != len(annotations):
        raise ValueError("duplicate ids found")
    if set(input_by_id) != set(annotation_by_id):
        missing = set(input_by_id) - set(annotation_by_id)
        unknown = set(annotation_by_id) - set(input_by_id)
        raise ValueError(f"id mismatch: missing={len(missing)} unknown={len(unknown)}")

    merged = []
    for source in inputs:
        annotation = annotation_by_id[source["id"]]
        labels = annotation["labels"]
        if len(labels) != len(source["words"]):
            raise ValueError(f"{source['id']}: words/labels length mismatch")
        if any(type(label) is not int or label not in (0, 1) for label in labels):
            raise ValueError(f"{source['id']}: labels must be integer 0 or 1")
        selected = [word for word, label in zip(source["words"], labels) if label]
        if selected != annotation["selected_words"]:
            raise ValueError(f"{source['id']}: selected_words mismatch")
        merged.append({
            **source,
            "labels": labels,
            "selected_words": selected,
            "needs_review": bool(annotation.get("needs_review", False)),
            "review_reason": annotation.get("review_reason", ""),
            "source": "chatgpt-ui-teacher-v2",
        })

    groups = {}
    for row in merged:
        key = (row["task_label"], row["needs_review"])
        groups.setdefault(key, []).append(row)
    rng = random.Random(args.seed)
    train, validation, test = [], [], []
    for rows in groups.values():
        rng.shuffle(rows)
        validation_size = round(len(rows) * 0.1)
        test_size = round(len(rows) * 0.1)
        validation.extend(rows[:validation_size])
        test.extend(rows[validation_size : validation_size + test_size])
        train.extend(rows[validation_size + test_size :])
    for rows in (train, validation, test):
        rng.shuffle(rows)

    output = Path(args.output_dir)
    write_jsonl(output / "all.jsonl", merged)
    write_jsonl(output / "train.jsonl", train)
    write_jsonl(output / "validation.jsonl", validation)
    write_jsonl(output / "test.jsonl", test)
    write_jsonl(output / "needs_review.jsonl", [r for r in merged if r["needs_review"]])

    stats = {}
    for name, rows in (("all", merged), ("train", train), ("validation", validation), ("test", test)):
        tokens = sum(len(row["labels"]) for row in rows)
        selected = sum(sum(row["labels"]) for row in rows)
        stats[name] = {
            "examples": len(rows),
            "task_labels": dict(Counter(row["task_label"] for row in rows)),
            "needs_review": sum(row["needs_review"] for row in rows),
            "selected_tokens": selected,
            "total_tokens": tokens,
            "redaction_rate": selected / max(tokens, 1),
        }
    (output / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
