from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

from text_normalization import normalize_with_alignment, project_normalized_labels
from v14_rule_adapter import V14RuleAdapter, read_jsonl, write_jsonl


_ADAPTER = None


def initialize(masker: str, task: str, root: str, max_length: int) -> None:
    global _ADAPTER
    _ADAPTER = V14RuleAdapter(masker, task, root, max_length=max_length)


def predict(text: str) -> tuple[str, list[int]]:
    return text, _ADAPTER.predict(text).labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generic-normalization rule cache")
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--masker", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--redactformer-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    clean_by_source = {}
    for row in pairs:
        clean_by_source.setdefault(row["source_id"], (row["clean_text"], row["clean_labels"]))

    known: dict[str, list[int]] = {}
    normalized_clean = {}
    for source_id, (text, labels) in clean_by_source.items():
        normalized = normalize_with_alignment(text)
        normalized_clean[source_id] = normalized
        if normalized.text == text:
            known[normalized.text] = labels

    normalized_noisy = {
        row["pair_id"]: normalize_with_alignment(row["text"]) for row in pairs
    }
    required = {
        value.text for value in normalized_clean.values()
    } | {value.text for value in normalized_noisy.values()}
    missing = sorted(required - known.keys())
    if missing:
        context = mp.get_context("fork")
        initialize(args.masker, args.task, args.redactformer_root, args.max_length)
        with context.Pool(args.workers) as pool:
            for index, (text, labels) in enumerate(
                pool.imap(predict, missing, chunksize=args.chunksize), start=1
            ):
                known[text] = labels
                if index % 1000 == 0 or index == len(missing):
                    print(f"[{index:,}/{len(missing):,}] normalized rule", flush=True)

    output = []
    restored = 0
    for row in pairs:
        clean_norm = normalized_clean[row["source_id"]]
        noisy_norm = normalized_noisy[row["pair_id"]]
        restored += int(clean_norm.text == noisy_norm.text)
        output.append(
            {
                "pair_id": row["pair_id"],
                "clean_labels": project_normalized_labels(
                    row["clean_text"], clean_norm, known[clean_norm.text]
                ),
                "noisy_labels": project_normalized_labels(
                    row["text"], noisy_norm, known[noisy_norm.text]
                ),
                "normalized_equal": clean_norm.text == noisy_norm.text,
            }
        )
    write_jsonl(args.output, output)
    summary = {
        "pairs": len(pairs),
        "unique_sources": len(clean_by_source),
        "unique_normalized_texts": len(required),
        "rule_calls": len(missing),
        "normalized_clean_equals_noisy": restored,
        "restoration_rate": restored / len(pairs) if pairs else 0.0,
        "normalizer": (
            "NFKC, whitespace collapse, apostrophe/control/zero-width cleanup, "
            "number-dosage boundary repair, boundary numeric punctuation cleanup"
        ),
        "oracle_noise_type_used": False,
    }
    Path(args.output).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
