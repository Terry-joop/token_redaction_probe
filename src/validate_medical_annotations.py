import argparse

from common import write_jsonl
from medical_common import read_records, validate_labels, validate_types


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and merge medical teacher annotations")
    parser.add_argument("--inputs", required=True, help="Canonical raw input JSONL")
    parser.add_argument("--annotations", required=True, help="Teacher response JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rejected", default=None)
    parser.add_argument("--source", default="medical-llm-teacher-v1")
    args = parser.parse_args()

    inputs = {row["id"]: row for row in read_records(args.inputs)}
    accepted, rejected, seen = [], [], set()
    for annotation in read_records(args.annotations):
        example_id = annotation.get("id")
        try:
            if example_id not in inputs:
                raise ValueError(f"{example_id}: unknown id")
            if example_id in seen:
                raise ValueError(f"{example_id}: duplicate annotation")
            seen.add(example_id)
            source = inputs[example_id]
            labels = validate_labels(example_id, source["words"], annotation.get("labels"))
            types = validate_types(example_id, source["words"], annotation.get("types"))
            for index, label in enumerate(labels):
                if label and types[index] == "O":
                    types[index] = "OTHER_SENSITIVE"
                elif not label:
                    types[index] = "O"
            accepted.append(source | {
                "labels": labels,
                "types": types,
                "selected_words": [word for word, label in zip(source["words"], labels) if label],
                "annotation_source": annotation.get("source", args.source),
                "needs_review": bool(annotation.get("needs_review", False)),
                "review_reason": str(annotation.get("review_reason", "")),
            })
        except ValueError as error:
            rejected.append({"id": example_id, "error": str(error), "annotation": annotation})

    missing = sorted(set(inputs) - seen)
    write_jsonl(args.output, accepted)
    rejected_path = args.rejected or args.output.replace(".jsonl", "_rejected.jsonl")
    write_jsonl(rejected_path, rejected)
    print(f"accepted={len(accepted)} rejected={len(rejected)} missing={len(missing)}")
    print(f"wrote {args.output}; rejected details: {rejected_path}")
    if missing:
        print("first missing ids:", ", ".join(missing[:10]))


if __name__ == "__main__":
    main()
