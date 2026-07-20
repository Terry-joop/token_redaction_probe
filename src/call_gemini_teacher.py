import argparse
import json
import os
import time
from pathlib import Path

from google import genai
from pydantic import BaseModel

from common import write_jsonl


class TokenLabels(BaseModel):
    labels: list[int]


def load_jsonl(path: str) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def error_status(error: Exception) -> int | None:
    return getattr(error, "status_code", None) or getattr(error, "code", None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Call a Gemini teacher for token labels")
    parser.add_argument("--requests", default="teacher/requests.jsonl")
    parser.add_argument("--output", default="teacher/gemini_responses.jsonl")
    parser.add_argument("--model", default="gemini-3.5-flash")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requests = load_jsonl(args.requests)[: args.limit]
    existing = load_jsonl(args.output) if Path(args.output).exists() else []
    completed = {row["id"] for row in existing}
    pending = [row for row in requests if row["id"] not in completed]
    print(f"model={args.model} requested={len(requests)} completed={len(completed)} pending={len(pending)}")
    if args.dry_run:
        for row in pending[:3]:
            print(f"\n[{row['id']}]\n{row['user_prompt']}")
        return
    if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        raise SystemExit("GEMINI_API_KEY is not set; no API calls were made")

    client = genai.Client()
    responses = list(existing)
    for index, row in enumerate(pending, start=1):
        last_error = None
        for attempt in range(args.max_retries + 1):
            try:
                interaction = client.interactions.create(
                    model=args.model,
                    input=row["system_prompt"] + "\n\n" + row["user_prompt"],
                    response_format={
                        "type": "text",
                        "mime_type": "application/json",
                        "schema": TokenLabels.model_json_schema(),
                    },
                )
                parsed = TokenLabels.model_validate_json(interaction.output_text)
                labels = parsed.labels
                if len(labels) != len(row["words"]):
                    raise ValueError(f"expected {len(row['words'])} labels, got {len(labels)}")
                if any(type(label) is not int or label not in (0, 1) for label in labels):
                    raise ValueError("labels must contain only integer 0 or 1")
                responses.append({
                    "id": row["id"],
                    "model": args.model,
                    "response_id": getattr(interaction, "id", None),
                    "labels": labels,
                })
                write_jsonl(args.output, responses)
                print(f"[{index}/{len(pending)}] {row['id']} labels={sum(labels)}/{len(labels)}")
                last_error = None
                break
            except Exception as error:
                last_error = error
                status = error_status(error)
                if status in {400, 401, 403, 404, 429}:
                    print(f"FATAL {row['id']}: {error}")
                    print("Stopped before sending any more requests; rerun later to resume")
                    return
                if attempt < args.max_retries:
                    time.sleep(2 ** attempt)
        if last_error is not None:
            print(f"FAILED {row['id']}: {last_error}")
    print(f"Saved {len(responses)} responses to {args.output}")


if __name__ == "__main__":
    main()
