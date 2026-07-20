import argparse
import os

from datasets import load_dataset

from common import word_tokenize, write_jsonl


# This is intentionally a transparent weak-label baseline, not ground truth.
SENTIMENT_WORDS = {
    "amazing", "awful", "bad", "beautiful", "best", "boring", "brilliant",
    "charming", "dreadful", "excellent", "fantastic", "fun", "funny", "good",
    "great", "hated", "horrible", "impressive", "interesting", "lame", "loved",
    "masterpiece", "moving", "poor", "powerful", "ridiculous", "stupid", "superb",
    "terrible", "thrilling", "wonderful", "worst", "waste", "weak",
}
NEGATORS = {"not", "never", "no", "neither", "hardly", "barely", "isn't", "wasn't"}
INTENSIFIERS = {"absolutely", "extremely", "really", "so", "too", "very", "surprisingly"}


def heuristic_labels(words: list[str]) -> list[int]:
    lowered = [word.lower() for word in words]
    labels = [int(word in SENTIMENT_WORDS) for word in lowered]
    for i, word in enumerate(lowered):
        if word in NEGATORS | INTENSIFIERS and any(labels[i + 1 : i + 3]):
            labels[i] = 1
    return labels


def convert(split, split_name: str, limit: int) -> list[dict]:
    rows = []
    for i, example in enumerate(split.select(range(min(limit, len(split))))):
        words = word_tokenize(example["sentence"])
        rows.append({
            "id": f"{split_name}-{i}",
            "text": example["sentence"],
            "words": words,
            "labels": heuristic_labels(words),
            "task_label": int(example["label"]),
            "source": "sst2-heuristic-v1",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=300)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
    dataset = load_dataset("glue", "sst2")
    train = convert(dataset["train"], "train", args.train_size)
    validation = convert(dataset["validation"], "validation", args.validation_size)
    write_jsonl(f"{args.output_dir}/train.jsonl", train)
    write_jsonl(f"{args.output_dir}/validation.jsonl", validation)
    positive = sum(sum(row["labels"]) for row in train)
    total = sum(len(row["labels"]) for row in train)
    print(f"Wrote {len(train)} train and {len(validation)} validation examples")
    print(f"Redact tokens: {positive}/{total} ({positive / max(total, 1):.2%})")


if __name__ == "__main__":
    main()

