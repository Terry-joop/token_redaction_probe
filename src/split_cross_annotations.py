import argparse
from collections import Counter
from pathlib import Path

from medical_common import read_records
from common import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Split combined annotations by dataset id prefix")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    groups = {}
    for row in read_records(args.annotations):
        task = row["id"].split("-", 1)[0]
        groups.setdefault(task, []).append(row)
    root = Path(args.output_root)
    for task, rows in sorted(groups.items()):
        write_jsonl(root / task / "medterm4_latest.jsonl", rows)
    print(dict(Counter({task: len(rows) for task, rows in groups.items()})))


if __name__ == "__main__":
    main()
