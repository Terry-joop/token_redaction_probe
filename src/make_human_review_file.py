import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an editable human-review copy of a teacher-labeled JSONL file."
    )
    parser.add_argument("--input", default="data/teacher_prepared/test.jsonl")
    parser.add_argument(
        "--output", default="data/human_review/sst2_test_100_review.jsonl"
    )
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    rows = read_jsonl(source)
    review_rows = []

    for row in rows:
        words = row["words"]
        teacher_labels = row["labels"]
        if len(words) != len(teacher_labels):
            raise ValueError(f"{row['id']}: words/labels length mismatch")

        review_rows.append(
            {
                "id": row["id"],
                "text": row["text"],
                "words": words,
                "task_label": row["task_label"],
                "teacher_labels": teacher_labels,
                "teacher_selected_words": row.get(
                    "selected_words",
                    [word for word, label in zip(words, teacher_labels) if label],
                ),
                "teacher_needs_review": bool(row.get("needs_review", False)),
                "teacher_review_reason": row.get("review_reason", ""),
                # Starting copies for convenient editing. They are not human gold
                # until review_status is manually changed to "reviewed".
                "human_labels": list(teacher_labels),
                "human_needs_review": bool(row.get("needs_review", False)),
                "human_review_reason": row.get("review_reason", ""),
                "review_status": "pending",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for row in review_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"created={output} examples={len(review_rows)}")


if __name__ == "__main__":
    main()
