import argparse
import html
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import spacy
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from spacy.lang.en.stop_words import STOP_WORDS

from common import word_tokenize, write_jsonl
from medical_common import word_offsets


PAIR_SEPARATOR = "[PAIR]"
FULL_ENTITY_TYPES = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "MONEY", "PERCENT"}
STRICT_PII_TYPES = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP"}
DURATION = re.compile(
    r"^\s*(about |around |~)?\d+\s*(day|week|month|year|hour|min|yr|mo|wk)s?\b", re.I,
)
POLICIES = {
    "bios": "piiclean-v1",
    "mrpc": "piiclean-strict-v1",
    "qnli": "entityclean-v1",
    "finphrasebank": "entityclean-v1",
}


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip().strip('"')


def usable(text: str) -> bool:
    return 10 <= len(text) <= 2000 and text.casefold() not in {"[deleted]", "[removed]"}


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


def unique_rows(rows: list[dict], excluded: set[str]) -> list[dict]:
    output = []
    local = set()
    for row in rows:
        signature = row["text"].casefold()
        if signature in excluded or signature in local or not usable(row["text"]):
            continue
        local.add(signature)
        output.append(row)
    return output


def select_official_splits(
    candidates: dict[str, list[dict]], seed: int, full_data: bool = False,
) -> list[dict]:
    sizes = {"train": 800, "validation": 100, "test": 100}
    selected = []
    seen: set[str] = set()
    for offset, split in enumerate(("train", "validation", "test")):
        available = unique_rows(candidates[split], seen)
        chosen = available if full_data else balanced_sample(available, sizes[split], seed + offset)
        if not full_data and len(chosen) != sizes[split]:
            raise ValueError(f"{split}: requested {sizes[split]}, found {len(chosen)}")
        for row in chosen:
            row["desired_split"] = split
            seen.add(row["text"].casefold())
        selected.extend(chosen)
    return selected


def single_rows(dataset: str, split: str, records, text_key: str, label_key: str) -> list[dict]:
    rows = []
    for index, raw in enumerate(records):
        text = clean_text(raw[text_key])
        rows.append({
            "id": f"{dataset}-{split}-{raw.get('idx', index)}",
            "text": text,
            "task_label": int(raw[label_key]),
            "source": dataset,
            "source_split": split,
        })
    return rows


def pair_rows(dataset: str, split: str, records, key_a: str, key_b: str) -> list[dict]:
    rows = []
    for index, raw in enumerate(records):
        text_a, text_b = clean_text(raw[key_a]), clean_text(raw[key_b])
        text = f"{text_a} {PAIR_SEPARATOR} {text_b}"
        rows.append({
            "id": f"{dataset}-{split}-{raw.get('idx', index)}",
            "text": text,
            "text_a": text_a,
            "text_b": text_b,
            "task_label": int(raw["label"]),
            "source": f"glue/{dataset}",
            "source_split": split,
        })
    return rows


def build_bios(seed: int, full_data: bool = False) -> list[dict]:
    raw = load_dataset("LabHC/bias_in_bios")
    candidates = {
        "train": single_rows("bios", "train", raw["train"], "hard_text", "profession"),
        "validation": single_rows("bios", "validation", raw["dev"], "hard_text", "profession"),
        "test": single_rows("bios", "test", raw["test"], "hard_text", "profession"),
    }
    return select_official_splits(candidates, seed, full_data)


def build_glue(
    task: str, key_a: str, key_b: str, seed: int, full_data: bool = False,
) -> list[dict]:
    raw = load_dataset("glue", task)
    candidates = {
        split: pair_rows(task, split, raw[split], key_a, key_b)
        for split in ("train", "validation", "test")
    }
    return select_official_splits(candidates, seed, full_data)


def build_finphrasebank(seed: int, full_data: bool = False) -> list[dict]:
    raw = load_dataset(
        "takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True,
        split="train",
    )
    rows = single_rows("finphrasebank", "all", raw, "sentence", "label")
    available = unique_rows(rows, set())
    pool = available if full_data else balanced_sample(available, 1000, seed)
    labels = [str(row["task_label"]) for row in pool]
    train_rows, heldout = train_test_split(
        pool, test_size=(0.2 if full_data else 200), random_state=seed, stratify=labels,
    )
    heldout_labels = [str(row["task_label"]) for row in heldout]
    validation_rows, test_rows = train_test_split(
        heldout, test_size=(0.5 if full_data else 100), random_state=seed, stratify=heldout_labels,
    )
    output = []
    for split, part in (
        ("train", train_rows), ("validation", validation_rows), ("test", test_rows),
    ):
        for row in part:
            row["source_split"] = split
            row["desired_split"] = split
            output.append(row)
    random.Random(seed).shuffle(output)
    return output


def keep_word(word: str, strict: bool) -> bool:
    low = word.strip().lower()
    if not low or low in STOP_WORDS:
        return False
    if not any(character.isalpha() or character.isdigit() for character in low):
        return False
    if len(low) == 1 and low.isalpha():
        return False
    if strict and not any(character.isalpha() for character in low):
        return False
    return True


def annotate(row: dict, nlp, doc=None) -> tuple[list[int], list[str]]:
    words, offsets = word_offsets(row["text"])
    if words != row["words"]:
        raise ValueError(f"{row['id']}: tokenizer offset mismatch")
    strict = row["teacher_policy"] == "piiclean-strict-v1"
    allowed = STRICT_PII_TYPES if strict else FULL_ENTITY_TYPES
    spans = []
    document = doc if doc is not None else nlp(row["text"])
    for entity in document.ents:
        if entity.label_ in allowed:
            spans.append((entity.start_char, entity.end_char, entity.label_))
        elif (
            not strict and entity.label_ == "DATE"
            and re.search(r"\b(19|20)\d\d\b", entity.text)
            and not DURATION.match(entity.text)
        ):
            spans.append((entity.start_char, entity.end_char, "DATE"))
    labels, types = [0] * len(words), ["O"] * len(words)
    for index, (start, end) in enumerate(offsets):
        if not keep_word(words[index], strict):
            continue
        for span_start, span_end, entity_type in spans:
            if start < span_end and span_start < end:
                labels[index] = 1
                types[index] = entity_type
                break
    return labels, types


def report(rows: list[dict]) -> dict:
    tokens = sum(len(row["labels"]) for row in rows)
    selected = sum(sum(row["labels"]) for row in rows)
    return {
        "examples": len(rows),
        "task_labels": dict(Counter(str(row["task_label"]) for row in rows)),
        "tokens": tokens,
        "selected_tokens": selected,
        "mask_rate": selected / max(tokens, 1),
        "zero_mask_examples": sum(not any(row["labels"]) for row in rows),
    }


def write_dataset(dataset: str, rows: list[dict], root: Path, nlp) -> None:
    policy = POLICIES[dataset]
    inputs = []
    annotations = []
    merged = []
    documents = nlp.pipe((row["text"] for row in rows), batch_size=128)
    for number, (row, document) in enumerate(zip(rows, documents), start=1):
        source = row | {
            "words": word_tokenize(row["text"]),
            "dataset_name": dataset,
            "teacher_policy": policy,
        }
        labels, types = annotate(source, nlp, document)
        annotation = {
            "id": source["id"], "labels": labels, "types": types,
            "selected_words": [word for word, label in zip(source["words"], labels) if label],
            "source": policy,
        }
        inputs.append(source)
        annotations.append(annotation)
        merged.append(
            source
            | {key: value for key, value in annotation.items() if key != "source"}
            | {"annotation_source": policy}
        )
        if number % 5000 == 0:
            print(f"{dataset}: annotated {number}/{len(rows)}", flush=True)
    path = root / dataset
    write_jsonl(path / "input.jsonl", inputs)
    write_jsonl(path / "teacher.jsonl", annotations)
    write_jsonl(path / "all.jsonl", merged)
    splits = {
        split: [row for row in merged if row["desired_split"] == split]
        for split in ("train", "validation", "test")
    }
    for split, split_rows in splits.items():
        write_jsonl(path / f"{split}.jsonl", split_rows)
    summary = {"policy": policy, "all": report(merged)} | {
        split: report(split_rows) for split, split_rows in splits.items()
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / "stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(dataset, json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare policy-separated nonmedical redactor data")
    parser.add_argument("--output-root", default="data/nonmedical_redactor")
    parser.add_argument("--datasets", nargs="+", choices=sorted(POLICIES), default=list(POLICIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-data", action="store_true")
    args = parser.parse_args()
    builders = {
        "bios": lambda: build_bios(args.seed, args.full_data),
        "mrpc": lambda: build_glue(
            "mrpc", "sentence1", "sentence2", args.seed, args.full_data,
        ),
        "qnli": lambda: build_glue(
            "qnli", "question", "sentence", args.seed, args.full_data,
        ),
        "finphrasebank": lambda: build_finphrasebank(args.seed, args.full_data),
    }
    nlp = spacy.load(
        "en_core_web_sm", disable=["parser", "tagger", "attribute_ruler", "lemmatizer"],
    )
    root = Path(args.output_root)
    for dataset in args.datasets:
        write_dataset(dataset, builders[dataset](), root, nlp)


if __name__ == "__main__":
    main()
