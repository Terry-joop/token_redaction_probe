from __future__ import annotations

import argparse
import json
from pathlib import Path

from v14_rule_adapter import V14RuleAdapter, read_jsonl, write_jsonl

DEFAULT_REDACTFORMER_ROOT = Path(__file__).resolve().parents[3] / "Redactformer"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate raw JSONL text with RedactFormer v1.4"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--masker",
        choices=["medterm5", "piiclean2", "mdccunion"],
        required=True,
    )
    parser.add_argument("--task", required=True)
    parser.add_argument("--redactformer-root", default=DEFAULT_REDACTFORMER_ROOT)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    adapter = V14RuleAdapter(
        args.masker,
        args.task,
        args.redactformer_root,
        max_length=args.max_length,
    )
    annotated = []
    for index, row in enumerate(rows, start=1):
        annotated.append(adapter.annotate_row(row))
        if index % 100 == 0 or index == len(rows):
            print(f"[{index}/{len(rows)}] {args.masker}:{args.task}", flush=True)
    write_jsonl(args.output, annotated)
    positive = sum(sum(row["labels"]) for row in annotated)
    total = sum(len(row["labels"]) for row in annotated)
    summary = {
        "input": str(Path(args.input)),
        "output": str(Path(args.output)),
        "examples": len(annotated),
        "tokens": total,
        "sensitive_tokens": positive,
        "mask_rate": positive / total if total else 0.0,
        "policy": adapter.policy_id,
    }
    Path(args.output).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
