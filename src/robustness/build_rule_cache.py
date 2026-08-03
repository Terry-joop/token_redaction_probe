from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from v14_rule_adapter import V14RuleAdapter, read_jsonl

_ADAPTER = None


def initialize(masker: str, task: str, redactformer_root: str, max_length: int) -> None:
    global _ADAPTER
    _ADAPTER = V14RuleAdapter(
        masker, task, redactformer_root, max_length=max_length
    )


def predict(item: tuple[str, str]) -> tuple[str, list[int]]:
    pair_id, text = item
    return pair_id, _ADAPTER.predict(text).labels


def complete(path: Path, expected: int) -> bool:
    summary = path.with_suffix(".summary.json")
    if not path.exists() or not summary.exists():
        return False
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("predictions") == expected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reusable v1.4 noisy-rule predictions in parallel"
    )
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--masker", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--redactformer-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument(
        "--start-method",
        choices=["fork", "spawn"],
        default="fork",
        help="fork loads the large rule adapter once and shares it read-only.",
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    output = Path(args.output)
    if not args.force and complete(output, len(pairs)):
        print(f"cache already complete: {output} ({len(pairs):,})")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    context = mp.get_context(args.start_method)
    tasks = ((row["pair_id"], row["text"]) for row in pairs)
    pool_kwargs = {}
    if args.start_method == "fork":
        initialize(
            args.masker,
            args.task,
            args.redactformer_root,
            args.max_length,
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
    with context.Pool(
        args.workers, **pool_kwargs
    ) as pool, temporary.open("w", encoding="utf-8") as handle:
        for index, (pair_id, labels) in enumerate(
            pool.imap(predict, tasks, chunksize=args.chunksize), start=1
        ):
            handle.write(
                json.dumps(
                    {"pair_id": pair_id, "labels": labels},
                    ensure_ascii=False,
                )
                + "\n"
            )
            if index % 1000 == 0 or index == len(pairs):
                print(f"[{index:,}/{len(pairs):,}] rule cache", flush=True)
    temporary.replace(output)
    summary = {
        "pairs": str(Path(args.pairs)),
        "output": str(output),
        "predictions": len(pairs),
        "masker": args.masker,
        "task": args.task,
        "workers": args.workers,
        "start_method": args.start_method,
        "max_length": args.max_length,
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
