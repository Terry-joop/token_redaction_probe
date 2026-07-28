import argparse
import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from common import word_tokenize, write_jsonl


PAIR_SEPARATOR = "[PAIR]"


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip().strip('"')


def usable(text: str) -> bool:
    low = text.casefold()
    return 10 <= len(text) <= 1000 and low not in {
        "[deleted]", "[removed]", "[ deleted ]", "[ removed ]",
    }


def balanced_sample(rows: list[dict], size: int, seed: int) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row["task_label"])].append(row)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    labels = sorted(groups)
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


def deduplicate(rows: list[dict], seen: set[str] | None = None) -> list[dict]:
    seen = set() if seen is None else seen
    output = []
    for row in rows:
        signature = row["text"].casefold()
        if signature in seen or not usable(row["text"]):
            continue
        seen.add(signature)
        output.append(row)
    return output


def emit(task: str, rows: list[dict], output_root: Path) -> None:
    output = []
    for row in rows:
        item = {
            "id": row["id"],
            "text": row["text"],
            "words": word_tokenize(row["text"]),
            "task_label": row["task_label"],
            "source": row["source"],
            "source_split": row["source_split"],
            "desired_split": row["desired_split"],
            "policy_version": "medical-sensitive-v2-medterm4-aligned",
            "dataset_name": task,
        }
        for key in ("sentence1", "sentence2", "pair_id"):
            if key in row:
                item[key] = row[key]
        output.append(item)
    write_jsonl(output_root / task / "input.jsonl", output)
    counts = Counter((row["desired_split"], str(row["task_label"])) for row in output)
    print(f"{task}: examples={len(output)} split_label_counts={dict(counts)}")


def build_mednli(source_root: Path, seed: int) -> list[dict]:
    label_map = {"entailment": 0, "neutral": 1, "contradiction": 2}
    specs = [
        ("mli_train_v1.jsonl", "train", 800),
        ("mli_dev_v1.jsonl", "validation", 100),
        ("mli_test_v1.jsonl", "test", 100),
    ]
    seen: set[str] = set()
    selected = []
    for offset, (filename, split, size) in enumerate(specs):
        rows = []
        path = source_root / filename
        with path.open(encoding="utf-8") as handle:
            for source_index, line in enumerate(handle):
                raw = json.loads(line)
                sentence1 = clean_text(raw["sentence1"])
                sentence2 = clean_text(raw["sentence2"])
                label = raw.get("gold_label")
                if label not in label_map or not usable(sentence1) or not usable(sentence2):
                    continue
                text = f"{sentence1} {PAIR_SEPARATOR} {sentence2}"
                rows.append({
                    "id": f"mednli-{split}-{raw.get('pairID', source_index)}",
                    "text": text,
                    "sentence1": sentence1,
                    "sentence2": sentence2,
                    "pair_id": raw.get("pairID"),
                    "task_label": label_map[label],
                    "source": "MedNLI/mli_v1",
                    "source_split": split,
                    "desired_split": split,
                })
        rows = deduplicate(rows, seen)
        chosen = balanced_sample(rows, size, seed + offset)
        if len(chosen) != size:
            raise ValueError(f"MedNLI {split}: requested {size}, found {len(chosen)}")
        selected.extend(chosen)
    return selected


def build_mentalhealth(seed: int) -> list[dict]:
    dataset = load_dataset("btwitssayan/sentiment-analysis-for-mental-health", split="train")
    seen: set[str] = set()
    rows = []
    for source_index, raw in enumerate(dataset):
        text = clean_text(raw["statement"])
        label = clean_text(raw["status"])
        if not label:
            continue
        rows.append({
            "id": f"mentalhealth-train-{source_index}",
            "text": text,
            "task_label": label,
            "source": "btwitssayan/sentiment-analysis-for-mental-health",
            "source_split": "train",
        })
    rows = deduplicate(rows, seen)
    pool = balanced_sample(rows, 1000, seed)
    labels = [str(row["task_label"]) for row in pool]
    train_rows, heldout = train_test_split(
        pool, test_size=200, random_state=seed, stratify=labels,
    )
    heldout_labels = [str(row["task_label"]) for row in heldout]
    validation_rows, test_rows = train_test_split(
        heldout, test_size=100, random_state=seed, stratify=heldout_labels,
    )
    output = (
        [row | {"desired_split": "train"} for row in train_rows]
        + [row | {"desired_split": "validation"} for row in validation_rows]
        + [row | {"desired_split": "test"} for row in test_rows]
    )
    rng = random.Random(seed)
    rng.shuffle(output)
    counts = Counter(row["desired_split"] for row in output)
    if counts != {"train": 800, "validation": 100, "test": 100}:
        raise ValueError(f"mentalhealth split sizes are not 800/100/100: {dict(counts)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MedNLI and mentalhealth redactor inputs")
    parser.add_argument("--output-root", default="data/medical_redactor/cross_dataset")
    parser.add_argument("--mednli-root", default="/home/jovyan/Redactformer/data/mednli")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", nargs="+", choices=["mednli", "mentalhealth"],
                        default=["mednli", "mentalhealth"])
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if "mednli" in args.datasets:
        emit("mednli", build_mednli(Path(args.mednli_root), args.seed), output_root)
    if "mentalhealth" in args.datasets:
        emit("mentalhealth", build_mentalhealth(args.seed), output_root)


if __name__ == "__main__":
    main()
