import argparse
import json
from pathlib import Path

from common import write_jsonl
from medical_common import read_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Export privacy-policy prompts for an LLM teacher")
    parser.add_argument("--input", default="data/medical_redactor/drugreviews/pilot_input.jsonl")
    parser.add_argument("--prompt", default="prompts/medical_sensitive_teacher_v1.txt")
    parser.add_argument("--output", default="teacher/medical_drugreviews_requests.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    system_prompt = Path(args.prompt).read_text(encoding="utf-8").strip()
    requests = []
    for row in read_records(args.input)[:args.limit]:
        # Deliberately omit task_label: sensitive span selection must not become
        # downstream task-evidence selection again.
        payload = {"id": row["id"], "words": row["words"]}
        requests.append({
            "id": row["id"],
            "system_prompt": system_prompt,
            "user_prompt": json.dumps(payload, ensure_ascii=False),
        })
    write_jsonl(args.output, requests)
    print(f"wrote {len(requests)} requests to {args.output}; task labels were not exported")


if __name__ == "__main__":
    main()
