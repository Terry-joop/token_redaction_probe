import argparse
import random
from pathlib import Path

from datasets import load_dataset

from common import word_tokenize, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified SST-2 teacher input")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--output", default="teacher/chatgpt_input_1000.jsonl")
    parser.add_argument("--chunks-dir", default="teacher/chatgpt_chunks_100")
    args = parser.parse_args()

    if args.size % 2:
        raise ValueError("--size must be even for a balanced binary sample")
    dataset = load_dataset("glue", "sst2", split="train")
    by_label = {0: [], 1: []}
    for index, example in enumerate(dataset):
        by_label[int(example["label"])].append(index)

    rng = random.Random(args.seed)
    per_label = args.size // 2
    selected = []
    for label in (0, 1):
        for index in rng.sample(by_label[label], per_label):
            selected.append((index, label))
    rng.shuffle(selected)

    rows = []
    for index, label in selected:
        text = dataset[index]["sentence"]
        rows.append({
            "id": f"sst2-train-{index}",
            "text": text,
            "words": word_tokenize(text),
            "task_label": label,
        })
    write_jsonl(args.output, rows)

    chunks_dir = Path(args.chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(rows), args.chunk_size):
        chunk_number = start // args.chunk_size + 1
        path = chunks_dir / f"chunk_{chunk_number:02d}.jsonl"
        write_jsonl(path, rows[start : start + args.chunk_size])

    negatives = sum(row["task_label"] == 0 for row in rows)
    positives = len(rows) - negatives
    chunks = (len(rows) + args.chunk_size - 1) // args.chunk_size
    print(f"Wrote {len(rows)} examples to {args.output}")
    print(f"label_0={negatives} label_1={positives} seed={args.seed}")
    print(f"Wrote {chunks} chunks to {args.chunks_dir}")


if __name__ == "__main__":
    main()
