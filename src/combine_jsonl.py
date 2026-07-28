import argparse

from common import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine validated JSONL splits")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    ids = set()
    for path in args.inputs:
        for row in read_jsonl(path):
            if row["id"] in ids:
                raise ValueError(f"duplicate id: {row['id']}")
            ids.add(row["id"])
            rows.append(row)
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
