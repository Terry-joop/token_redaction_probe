import argparse
import json
import re
from pathlib import Path

from common import write_jsonl


def load_plain_jsonl(path: str) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if line.strip():
                row = json.loads(line)
                row["_line"] = line_number
                rows.append(row)
    return rows


def parse_labels(row: dict) -> list[int]:
    if isinstance(row.get("labels"), list):
        return row["labels"]
    raw = row.get("response") or row.get("output") or row.get("content")
    if not isinstance(raw, str):
        raise ValueError("missing labels or string response/output/content")
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    parsed = json.loads(raw)
    labels = parsed.get("labels") if isinstance(parsed, dict) else parsed
    if not isinstance(labels, list):
        raise ValueError("response JSON must contain a labels list")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and merge LLM teacher responses")
    parser.add_argument("--requests", default="teacher/requests.jsonl")
    parser.add_argument("--responses", default="teacher/responses.jsonl")
    parser.add_argument("--output", default="teacher/validated.jsonl")
    parser.add_argument("--rejected", default="teacher/rejected.jsonl")
    args = parser.parse_args()

    requests = {row["id"]: row for row in load_plain_jsonl(args.requests)}
    accepted, rejected, seen = [], [], set()
    for response in load_plain_jsonl(args.responses):
        example_id = response.get("id")
        try:
            if example_id not in requests:
                raise ValueError("unknown or missing id")
            if example_id in seen:
                raise ValueError("duplicate id")
            labels = parse_labels(response)
            request = requests[example_id]
            if len(labels) != len(request["words"]):
                raise ValueError(f"expected {len(request['words'])} labels, got {len(labels)}")
            if any(type(label) is not int or label not in (0, 1) for label in labels):
                raise ValueError("every label must be integer 0 or 1")
            accepted.append({
                "id": example_id,
                "text": request["text"],
                "words": request["words"],
                "labels": labels,
                "task_label": request.get("task_label"),
                "source": response.get("model", "llm-teacher"),
            })
            seen.add(example_id)
        except (ValueError, json.JSONDecodeError) as error:
            rejected.append({"id": example_id, "line": response.get("_line"), "error": str(error)})

    rejected_ids = {row.get("id") for row in rejected}
    missing = sorted(set(requests) - seen - rejected_ids)
    rejected.extend({"id": example_id, "error": "missing response"} for example_id in missing)
    write_jsonl(args.output, accepted)
    write_jsonl(args.rejected, rejected)
    redact = sum(sum(row["labels"]) for row in accepted)
    total = sum(len(row["labels"]) for row in accepted)
    print(f"accepted={len(accepted)} rejected={len(rejected)}")
    print(f"redact_tokens={redact}/{total} ({redact / max(total, 1):.2%})")
    print(f"validated data: {args.output}")
    if rejected:
        print(f"inspect errors: {args.rejected}")


if __name__ == "__main__":
    main()
