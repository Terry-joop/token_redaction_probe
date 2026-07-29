import argparse
import html
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import concatenate_datasets, load_dataset

from common import word_tokenize, write_jsonl
from medical_common import read_records


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip().strip('"')


def sample_balanced(rows: list[dict], size: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["task_label"])].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    labels = sorted(groups)
    selected = []
    while len(selected) < min(size, len(rows)):
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


def usable(text: str) -> bool:
    low = text.lower().strip()
    return 10 <= len(text) <= 1000 and low not in {
        "[deleted]", "[removed]", "[ deleted ]", "[ removed ]"
    }


def make_rows(dataset: str, records, text_key: str, label_key: str,
              size: int | None, seed: int, exclude: set[str] | None = None) -> list[dict]:
    exclude = exclude or set()
    seen = set()
    rows = []
    for source_index, example in enumerate(records):
        text = clean_text(example[text_key])
        signature = text.casefold()
        if not usable(text) or signature in seen or signature in exclude:
            continue
        label = example[label_key]
        if label is None:
            continue
        seen.add(signature)
        rows.append({
            "source_index": source_index,
            "text": text,
            "task_label": label,
            "source": dataset,
            "source_split": example.get("_source_split", "train"),
        })
    return sample_balanced(rows, size, seed) if size is not None else rows


def emit(task: str, rows: list[dict], output_root: Path) -> list[dict]:
    output = []
    for row in rows:
        output.append({
            "id": f"{task}-{row['source_split']}-{row['source_index']}",
            "text": row["text"],
            "words": word_tokenize(row["text"]),
            "task_label": row["task_label"],
            "source": row["source"],
            "source_split": row["source_split"],
            "policy_version": "medical-sensitive-v2-medterm4-aligned",
            "dataset_name": task,
        })
    write_jsonl(output_root / task / "input.jsonl", output)
    print(f"{task}: examples={len(output)} labels={dict(Counter(str(r['task_label']) for r in output))}")
    return output


def with_split(dataset, split: str):
    return dataset.map(lambda _: {"_source_split": split})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build balanced medical cross-dataset probes")
    parser.add_argument("--output-root", default="data/medical_redactor/cross_dataset")
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument(
        "--full-data", action="store_true",
        help="Use every usable deduplicated row under the current domain filters.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-drugreviews",
        default="data/medical_redactor/drugreviews_1000/input.jsonl",
    )
    args = parser.parse_args()
    root = Path(args.output_root)

    previous = [] if args.full_data else read_records(args.exclude_drugreviews)
    excluded = {clean_text(row["text"]).casefold() for row in previous}
    sample_size = None if args.full_data else args.size

    drug_raw = load_dataset("lewtun/drug-reviews", split="train")
    counts = Counter(str(x) for x in drug_raw["condition"] if x)
    top = {label for label, _ in counts.most_common(10)}
    drug_filtered = drug_raw.filter(lambda ex: ex["condition"] in top)
    drug = make_rows("lewtun/drug-reviews", drug_filtered, "review", "condition",
                     sample_size, args.seed, excluded)

    symptom_raw = load_dataset("gretelai/symptom_to_diagnosis")
    symptom_all = concatenate_datasets([
        with_split(symptom_raw["train"], "train"),
        with_split(symptom_raw["test"], "test"),
    ])
    symptom = make_rows("gretelai/symptom_to_diagnosis", symptom_all,
                        "input_text", "output_text", None, args.seed)

    adr_raw = with_split(
        load_dataset("ade_corpus_v2", "Ade_corpus_v2_classification", split="train"),
        "train",
    )
    adr = make_rows("ade_corpus_v2/Ade_corpus_v2_classification", adr_raw,
                    "text", "label", sample_size, args.seed)

    reddit_raw = with_split(
        load_dataset("solomonk/reddit_mental_health_posts", split="train"),
        "train",
    )
    allowed = {"OCD", "ADHD", "depression", "ptsd", "aspergers"}
    reddit_filtered = reddit_raw.filter(lambda ex: ex["subreddit"] in allowed)
    reddit = make_rows("solomonk/reddit_mental_health_posts", reddit_filtered,
                       "body", "subreddit", sample_size, args.seed)

    all_rows = []
    for task, rows in (
        ("drug", drug), ("symptom2dx", symptom), ("adr", adr), ("redditmh", reddit)
    ):
        all_rows.extend(emit(task, rows, root))
    write_jsonl(root / "all_input.jsonl", all_rows)
    print(f"all: examples={len(all_rows)} output={root / 'all_input.jsonl'}")


if __name__ == "__main__":
    main()
