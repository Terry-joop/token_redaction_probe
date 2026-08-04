from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from v14_rule_adapter import read_jsonl, write_jsonl


SPLITS = ("test", "validation", "train")
APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "\x92": "'"})
INVISIBLE = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"}


def exact_key(text: str) -> str:
    return text


def normalized_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(APOSTROPHES)
    text = "".join(
        char
        for char in text
        if char not in INVISIBLE and unicodedata.category(char) != "Cc"
    )
    return " ".join(text.casefold().split())


def label_signature(row: dict) -> tuple[int, ...]:
    return tuple(int(value) for value in row.get("labels", []))


def audit(rows_by_split: dict[str, list[dict]]) -> dict:
    exact_sets = {
        split: {exact_key(row["text"]) for row in rows}
        for split, rows in rows_by_split.items()
    }
    norm_sets = {
        split: {normalized_key(row["text"]) for row in rows}
        for split, rows in rows_by_split.items()
    }
    pair_names = (("train", "validation"), ("train", "test"), ("validation", "test"))
    pairwise = {}
    for left, right in pair_names:
        pairwise[f"{left}__{right}"] = {
            "exact_unique_keys": len(exact_sets[left] & exact_sets[right]),
            "normalized_unique_keys": len(norm_sets[left] & norm_sets[right]),
        }

    signatures: dict[str, set[tuple[int, ...]]] = defaultdict(set)
    occurrences: Counter[str] = Counter()
    internal = {}
    for split, rows in rows_by_split.items():
        keys = [normalized_key(row["text"]) for row in rows]
        counts = Counter(keys)
        internal[split] = {
            "rows": len(rows),
            "normalized_unique_keys": len(counts),
            "duplicate_rows": sum(count - 1 for count in counts.values()),
            "duplicated_keys": sum(count > 1 for count in counts.values()),
        }
        for row, key in zip(rows, keys):
            occurrences[key] += 1
            signatures[key].add(label_signature(row))
    return {
        "internal": internal,
        "pairwise": pairwise,
        "normalized_keys_with_conflicting_labels": sum(
            len(values) > 1 for values in signatures.values()
        ),
        "normalized_duplicate_rows_all_splits": sum(
            count - 1 for count in occurrences.values()
        ),
    }


def rebuild(rows_by_split: dict[str, list[dict]]) -> tuple[dict[str, list[dict]], dict]:
    kept: dict[str, list[dict]] = {}
    claimed: set[str] = set()
    removed = {}
    for split in SPLITS:
        output = []
        overlap = 0
        internal = 0
        seen_here: set[str] = set()
        for row in rows_by_split[split]:
            key = normalized_key(row["text"])
            if key in seen_here:
                internal += 1
                continue
            if key in claimed:
                overlap += 1
                continue
            seen_here.add(key)
            output.append(row)
        claimed.update(seen_here)
        kept[split] = output
        removed[split] = {
            "before": len(rows_by_split[split]),
            "after": len(output),
            "removed_internal_duplicates": internal,
            "removed_higher_priority_overlap": overlap,
        }
    return kept, removed


def process_dataset(input_dir: Path, output_dir: Path) -> dict:
    rows = {split: read_jsonl(input_dir / f"{split}.jsonl") for split in SPLITS}
    before = audit(rows)
    kept, removed = rebuild(rows)
    after = audit(kept)
    for split in SPLITS:
        write_jsonl(output_dir / f"{split}.jsonl", kept[split])
    result = {
        "input": str(input_dir),
        "output": str(output_dir),
        "priority": list(SPLITS),
        "normalization": (
            "Unicode NFKC; apostrophe variants to ASCII; remove Cc and common "
            "zero-width characters; collapse whitespace; Unicode casefold"
        ),
        "before": before,
        "removed": removed,
        "after": after,
    }
    (output_dir / "split_audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if any(
        values["normalized_unique_keys"]
        for values in after["pairwise"].values()
    ):
        raise AssertionError("normalized split overlap remains")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and rebuild leakage-free splits")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    results = {}
    for dataset in datasets:
        results[dataset] = process_dataset(
            input_root / dataset / "clean", output_root / dataset / "clean"
        )
        removed = results[dataset]["removed"]
        print(
            f"{dataset}: "
            + ", ".join(
                f"{split} {values['before']:,}->{values['after']:,}"
                for split, values in removed.items()
            ),
            flush=True,
        )
    summary = {
        "datasets": datasets,
        "normalization": next(iter(results.values()))["normalization"],
        "priority": list(SPLITS),
        "results": results,
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary_path)


if __name__ == "__main__":
    main()
