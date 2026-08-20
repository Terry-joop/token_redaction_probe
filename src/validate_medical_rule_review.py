"""Validate and summarize a completed medical human-review JSONL file."""

from __future__ import annotations

import argparse
from collections import Counter

from medical_common import read_records, validate_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/human_review/medical_rule_review_v1.jsonl")
    parser.add_argument("--require-reviewed", action="store_true")
    args = parser.parse_args()

    rows = read_records(args.input)
    reviewed = Counter()
    changed = Counter()
    for row in rows:
        labels = validate_labels(row["id"], row["words"], row["human_labels"])
        dataset = row["dataset"]
        if row.get("human_reviewed"):
            reviewed[dataset] += 1
        elif args.require_reviewed:
            raise ValueError(f"{row['id']}: human_reviewed is false")
        rule = validate_labels(row["id"], row["words"], row["rule_labels"])
        if rule != labels:
            changed[dataset] += 1
    print(f"valid rows: {len(rows)}")
    for dataset in sorted({row['dataset'] for row in rows}):
        total = sum(row["dataset"] == dataset for row in rows)
        print(f"{dataset}: reviewed={reviewed[dataset]}/{total}, changed_from_rule={changed[dataset]}")


if __name__ == "__main__":
    main()
