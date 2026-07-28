import argparse
import html
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

from common import word_tokenize, write_jsonl
from medical_common import read_records


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip().strip('"')


def load_source(args) -> list[dict]:
    if args.input_jsonl:
        source = read_records(args.input_jsonl)
    else:
        source = list(load_dataset(args.dataset, args.config, split=args.split))

    rows = []
    seen = set()
    for source_index, example in enumerate(source):
        parts = [clean_text(example.get(column, "")) for column in args.text_columns]
        parts = [part for part in parts if part]
        if not parts:
            continue
        text = args.text_separator.join(parts)
        if len(text) < args.min_chars or len(text) > args.max_chars or text in seen:
            continue
        seen.add(text)
        label = example.get(args.label_column) if args.label_column else None
        if label is None and args.require_label:
            continue
        rows.append({"source_index": source_index, "text": text, "task_label": label})
    return rows


def stratified_sample(rows: list[dict], size: int, top_labels: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if not rows:
        raise ValueError("no usable source examples")
    labeled = [row for row in rows if row["task_label"] is not None]
    if not labeled:
        return rng.sample(rows, min(size, len(rows)))

    counts = Counter(str(row["task_label"]) for row in labeled)
    allowed = {label for label, _ in counts.most_common(top_labels)} if top_labels else set(counts)
    groups = defaultdict(list)
    for row in labeled:
        label = str(row["task_label"])
        if label in allowed:
            groups[label].append(row)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    labels = list(groups)
    rng.shuffle(labels)
    while len(selected) < size:
        progressed = False
        for label in labels:
            if groups[label]:
                selected.append(groups[label].pop())
                progressed = True
                if len(selected) == size:
                    break
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a medical-sensitive redactor pilot")
    parser.add_argument("--dataset", default="lewtun/drug-reviews")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--input-jsonl", default=None)
    parser.add_argument("--text-columns", nargs="+", default=["review"])
    parser.add_argument("--text-separator", default=" [PAIR] ")
    parser.add_argument("--label-column", default="condition")
    parser.add_argument("--require-label", action="store_true")
    parser.add_argument("--top-labels", type=int, default=10)
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-chars", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=1000)
    parser.add_argument("--id-prefix", default="drugreviews-pilot")
    parser.add_argument("--output", default="data/medical_redactor/drugreviews/pilot_input.jsonl")
    parser.add_argument("--chunks-dir", default="data/medical_redactor/drugreviews/chunks")
    parser.add_argument("--chunk-size", type=int, default=50)
    args = parser.parse_args()

    source = load_source(args)
    selected = stratified_sample(source, args.size, args.top_labels, args.seed)
    rows = []
    for row in selected:
        rows.append({
            "id": f"{args.id_prefix}-{row['source_index']}",
            "text": row["text"],
            "words": word_tokenize(row["text"]),
            "task_label": row["task_label"],
            "source": args.input_jsonl or args.dataset,
            "policy_version": "medical-sensitive-v1",
        })
    write_jsonl(args.output, rows)

    chunks_dir = Path(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), args.chunk_size):
        number = start // args.chunk_size + 1
        write_jsonl(chunks_dir / f"chunk_{number:02d}.jsonl", rows[start:start + args.chunk_size])
    selected_counts = Counter(str(row["task_label"]) for row in selected)
    print(f"source_usable={len(source)} selected={len(rows)} seed={args.seed}")
    print(f"labels={dict(selected_counts)}")
    print(f"wrote {args.output} and {(len(rows) + args.chunk_size - 1) // args.chunk_size} chunks")


if __name__ == "__main__":
    main()
