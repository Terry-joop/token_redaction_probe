import argparse

from common import write_jsonl
from medical_common import read_records, validate_labels


def named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a shared human-gold medical review file")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--candidate", action="append", type=named_path, default=[])
    parser.add_argument("--initialize-from", default=None)
    parser.add_argument("--shared-only", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    candidates = {
        name: {row["id"]: row for row in read_records(path)} for name, path in args.candidate
    }
    output = []
    for source in read_records(args.inputs):
        example_candidates = {}
        for name, rows in candidates.items():
            if source["id"] not in rows:
                continue
            candidate = rows[source["id"]]
            labels = validate_labels(source["id"], source["words"], candidate.get("labels"))
            example_candidates[name] = {
                "labels": labels,
                "types": candidate.get("types", ["O"] * len(labels)),
                "selected_words": [
                    word for word, label in zip(source["words"], labels) if label
                ],
            }
        if args.shared_only and len(example_candidates) != len(candidates):
            continue
        initial = (
            list(example_candidates[args.initialize_from]["labels"])
            if args.initialize_from in example_candidates else [0] * len(source["words"])
        )
        distinct = {tuple(value["labels"]) for value in example_candidates.values()}
        output.append({
            "id": source["id"], "text": source["text"], "words": source["words"],
            "task_label": source.get("task_label"), "candidates": example_candidates,
            "candidate_disagreement": len(distinct) > 1,
            "human_labels": initial, "human_types": ["O"] * len(initial),
            "human_reviewed": False, "human_review_reason": "",
        })
    write_jsonl(args.output, output)
    print(f"wrote {len(output)} review rows to {args.output}")
    print("Set human_reviewed=true only after checking human_labels/human_types.")


if __name__ == "__main__":
    main()
