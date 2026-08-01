from __future__ import annotations

import argparse
import json
from pathlib import Path

from v14_rule_adapter import V14RuleAdapter, read_jsonl, write_jsonl

DEFAULT_REDACTFORMER_ROOT = Path(__file__).resolve().parents[3] / "Redactformer"


def annotate_split(
    adapter: V14RuleAdapter,
    input_path: Path,
    output_path: Path,
    limit: int,
) -> dict:
    rows = read_jsonl(input_path)
    if limit:
        rows = rows[:limit]
    annotated = []
    for index, row in enumerate(rows, start=1):
        annotated.append(adapter.annotate_row(row))
        if index % 100 == 0 or index == len(rows):
            print(
                f"[{input_path.stem} {index}/{len(rows)}] "
                f"{adapter.masker}:{adapter.task}",
                flush=True,
            )
    write_jsonl(output_path, annotated)
    positive = sum(sum(row["labels"]) for row in annotated)
    total = sum(len(row["labels"]) for row in annotated)
    return {
        "examples": len(annotated),
        "tokens": total,
        "sensitive_tokens": positive,
        "mask_rate": positive / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate train/validation/test with one loaded v1.4 rule"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--masker", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--redactformer-root", default=DEFAULT_REDACTFORMER_ROOT)
    parser.add_argument("--train-limit", type=int, default=5000)
    parser.add_argument("--validation-limit", type=int, default=500)
    parser.add_argument("--test-limit", type=int, default=1000)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    adapter = V14RuleAdapter(
        args.masker,
        args.task,
        args.redactformer_root,
        max_length=args.max_length,
    )
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    limits = {
        "train": args.train_limit,
        "validation": args.validation_limit,
        "test": args.test_limit,
    }
    splits = {}
    for split, limit in limits.items():
        splits[split] = annotate_split(
            adapter,
            input_dir / f"{split}.jsonl",
            output_dir / f"{split}.jsonl",
            limit,
        )
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "policy": adapter.policy_id,
        "splits": splits,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
