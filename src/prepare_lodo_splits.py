import argparse
from pathlib import Path

from common import read_jsonl, write_jsonl


DEFAULT_TASKS = ("drug", "symptom2dx", "adr", "redditmh")
SPLITS = ("train", "validation", "test")


def combine_split(root: Path, tasks: list[str], split: str) -> list[dict]:
    rows: list[dict] = []
    seen_ids: set[str] = set()
    for task in tasks:
        path = root / task / "prepared" / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for row in read_jsonl(path):
            row_id = row["id"]
            if row_id in seen_ids:
                raise ValueError(f"duplicate id in {split}: {row_id}")
            seen_ids.add(row_id)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create leave-one-domain-out source splits from prepared datasets"
    )
    parser.add_argument(
        "--root", default="data/medical_redactor/cross_dataset",
        help="Directory containing TASK/prepared/{train,validation,test}.jsonl",
    )
    parser.add_argument(
        "--output-root", default="data/medical_redactor/cross_dataset/lodo",
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    args = parser.parse_args()

    root = Path(args.root)
    output_root = Path(args.output_root)
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError("--tasks contains duplicates")
    if len(args.tasks) < 2:
        raise ValueError("LODO requires at least two tasks")

    for heldout in args.tasks:
        source_tasks = [task for task in args.tasks if task != heldout]
        heldout_dir = output_root / f"heldout_{heldout}"
        counts = {}
        for split in SPLITS:
            rows = combine_split(root, source_tasks, split)
            output = heldout_dir / f"source_{split}.jsonl"
            write_jsonl(output, rows)
            counts[split] = len(rows)
        print(
            f"heldout={heldout} source={','.join(source_tasks)} "
            f"train/validation/test={counts['train']}/{counts['validation']}/{counts['test']}"
        )


if __name__ == "__main__":
    main()
