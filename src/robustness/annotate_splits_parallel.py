from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from v14_rule_adapter import V14RuleAdapter, read_jsonl

_ADAPTER = None


def initialize(masker: str, task: str, root: str, max_length: int) -> None:
    global _ADAPTER
    _ADAPTER = V14RuleAdapter(masker, task, root, max_length=max_length)


def annotate(row: dict) -> dict:
    return _ADAPTER.annotate_row(row)


def annotate_split(pool, input_path: Path, output_path: Path, chunksize: int) -> dict:
    rows = read_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    positive = 0
    total = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(
            pool.imap(annotate, rows, chunksize=chunksize), start=1
        ):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            positive += sum(row["labels"])
            total += len(row["labels"])
            if index % 1000 == 0 or index == len(rows):
                print(f"[{input_path.stem} {index:,}/{len(rows):,}]", flush=True)
    temporary.replace(output_path)
    return {
        "examples": len(rows),
        "tokens": total,
        "sensitive_tokens": positive,
        "mask_rate": positive / total if total else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel full-split v1.4 annotation with stable row order"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--masker", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--redactformer-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument(
        "--start-method", choices=["spawn", "fork"], default="spawn",
        help="fork loads one large read-only rule adapter before sharing workers.",
    )
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    context = mp.get_context(args.start_method)
    pool_kwargs = {}
    if args.start_method == "fork":
        initialize(
            args.masker, args.task, args.redactformer_root, args.max_length
        )
    else:
        pool_kwargs = {
            "initializer": initialize,
            "initargs": (
                args.masker,
                args.task,
                args.redactformer_root,
                args.max_length,
            ),
        }
    with context.Pool(args.workers, **pool_kwargs) as pool:
        splits = {
            split: annotate_split(
                pool,
                input_dir / f"{split}.jsonl",
                output_dir / f"{split}.jsonl",
                args.chunksize,
            )
            for split in ("train", "validation", "test")
        }
    first = json.loads((output_dir / "train.jsonl").open(encoding="utf-8").readline())
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "policy": first["teacher_policy"],
        "workers": args.workers,
        "start_method": args.start_method,
        "stable_input_order": True,
        "splits": splits,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
