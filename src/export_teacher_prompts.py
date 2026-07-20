import argparse
import json
from pathlib import Path

from common import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SST-2 examples for an LLM teacher")
    parser.add_argument("--input", default="data/train.jsonl")
    parser.add_argument("--prompt", default="prompts/sst2_teacher.txt")
    parser.add_argument("--output", default="teacher/requests.jsonl")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    system_prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    rows = read_jsonl(args.input)[: args.limit]
    requests = []
    for row in rows:
        requests.append({
            "id": row["id"],
            "text": row["text"],
            "words": row["words"],
            "task_label": row.get("task_label"),
            "system_prompt": system_prompt,
            "user_prompt": "Words: " + json.dumps(row["words"], ensure_ascii=False),
        })
    write_jsonl(args.output, requests)
    print(f"Wrote {len(requests)} teacher requests to {args.output}")


if __name__ == "__main__":
    main()
