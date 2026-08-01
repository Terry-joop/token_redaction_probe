from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from v14_rule_adapter import read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from medical_common import word_offsets  # noqa: E402


DOSAGE_UNITS = {
    "mg",
    "mcg",
    "g",
    "kg",
    "ml",
    "l",
    "iu",
    "units",
    "unit",
    "mm",
    "cm",
}


@dataclass
class Variant:
    text: str
    char_mask: list[int]
    clean_target: tuple[int, int]
    noisy_target: tuple[int, int]
    edit: dict


def char_mask_from_row(row: dict) -> tuple[list[tuple[int, int]], list[int]]:
    words, offsets = word_offsets(row["text"])
    if words != row["words"]:
        raise ValueError(f"{row.get('id')}: words do not match stable tokenizer")
    mask = [0] * len(row["text"])
    for (start, end), label in zip(offsets, row["labels"]):
        if label:
            mask[start:end] = [1] * (end - start)
    return offsets, mask


def apply_edit(
    text: str,
    mask: list[int],
    start: int,
    end: int,
    replacement: str,
    replacement_mask: list[int],
    target: tuple[int, int],
    edit: dict,
) -> Variant:
    if len(replacement) != len(replacement_mask):
        raise ValueError("replacement and replacement_mask length mismatch")
    new_text = text[:start] + replacement + text[end:]
    new_mask = mask[:start] + replacement_mask + mask[end:]
    delta = len(replacement) - (end - start)
    target_start, target_end = target
    if end <= target_start:
        noisy_target = (target_start + delta, target_end + delta)
    elif start >= target_end:
        noisy_target = target
    else:
        noisy_target = (target_start, target_end + delta)
    return Variant(new_text, new_mask, target, noisy_target, edit)


def sensitive_runs(offsets: list[tuple[int, int]], labels: list[int]):
    for index in range(len(offsets) - 1):
        if labels[index] and labels[index + 1]:
            yield index, index + 1


def space_variant(row: dict, replacement: str, name: str) -> Variant | None:
    offsets, mask = char_mask_from_row(row)
    for left, right in sensitive_runs(offsets, row["labels"]):
        start, end = offsets[left][1], offsets[right][0]
        gap = row["text"][start:end]
        if gap and gap.isspace():
            return apply_edit(
                row["text"],
                mask,
                start,
                end,
                replacement,
                [0] * len(replacement),
                (offsets[left][0], offsets[right][1]),
                {"kind": name, "old": gap, "new": replacement},
            )
    return None


def apostrophe_variant(row: dict, replacement: str, name: str) -> Variant | None:
    offsets, mask = char_mask_from_row(row)
    triples = zip(offsets, row["words"], row["labels"])
    for index, ((start, end), word, label) in enumerate(triples):
        if label and ("'" in word or "’" in word):
            local = next(pos for pos, char in enumerate(word) if char in "'’")
            absolute = start + local
            return apply_edit(
                row["text"],
                mask,
                absolute,
                absolute + 1,
                replacement,
                [1] * len(replacement),
                (start, end),
                {
                    "kind": name,
                    "word_index": index,
                    "old": row["text"][absolute],
                    "new": replacement,
                },
            )
    return None


def dosage_variant(row: dict, replacement: str, name: str) -> Variant | None:
    offsets, mask = char_mask_from_row(row)
    for index in range(len(row["words"]) - 1):
        number = row["words"][index]
        unit = row["words"][index + 1].lower().rstrip(".")
        if (
            row["labels"][index]
            and row["labels"][index + 1]
            and re.fullmatch(r"\d+(?:\.\d+)?", number)
            and unit in DOSAGE_UNITS
        ):
            start, end = offsets[index][1], offsets[index + 1][0]
            if row["text"][start:end].isspace():
                return apply_edit(
                    row["text"],
                    mask,
                    start,
                    end,
                    replacement,
                    [1] * len(replacement),
                    (offsets[index][0], offsets[index + 1][1]),
                    {
                        "kind": name,
                        "number": number,
                        "unit": unit,
                        "new": replacement,
                    },
                )
    return None


def punctuation_variant(row: dict, punctuation: str, name: str) -> Variant | None:
    offsets, mask = char_mask_from_row(row)
    triples = zip(offsets, row["words"], row["labels"])
    for index, ((start, end), word, label) in enumerate(triples):
        if label and any(char.isdigit() for char in word):
            if end >= len(row["text"]) or row["text"][end] not in ",;:":
                return apply_edit(
                    row["text"],
                    mask,
                    end,
                    end,
                    punctuation,
                    [0],
                    (start, end),
                    {"kind": name, "word_index": index, "new": punctuation},
                )
    return None


def zero_width_variant(row: dict) -> Variant | None:
    offsets, mask = char_mask_from_row(row)
    triples = zip(offsets, row["words"], row["labels"])
    for index, ((start, end), word, label) in enumerate(triples):
        if label and len(word) >= 5 and word.isalpha():
            point = start + len(word) // 2
            return apply_edit(
                row["text"],
                mask,
                point,
                point,
                "\u200b",
                [1],
                (start, end),
                {
                    "kind": "zero_width_inside",
                    "word_index": index,
                    "new": "U+200B",
                },
            )
    return None


TRANSFORMS: list[tuple[str, str, Callable[[dict], Variant | None]]] = [
    (
        "double_space",
        "seen",
        lambda row: space_variant(row, "  ", "double_space"),
    ),
    (
        "curly_apostrophe",
        "seen",
        lambda row: apostrophe_variant(row, "’", "curly_apostrophe"),
    ),
    (
        "c1_apostrophe",
        "seen",
        lambda row: apostrophe_variant(row, "\x92", "c1_apostrophe"),
    ),
    (
        "dosage_join",
        "seen",
        lambda row: dosage_variant(row, "", "dosage_join"),
    ),
    (
        "comma_after_number",
        "seen",
        lambda row: punctuation_variant(row, ",", "comma_after_number"),
    ),
    (
        "triple_space",
        "unseen",
        lambda row: space_variant(row, "   ", "triple_space"),
    ),
    ("nbsp", "unseen", lambda row: space_variant(row, "\u00a0", "nbsp")),
    (
        "modifier_apostrophe",
        "unseen",
        lambda row: apostrophe_variant(row, "\u02bc", "modifier_apostrophe"),
    ),
    (
        "dosage_hyphen",
        "unseen",
        lambda row: dosage_variant(row, "-", "dosage_hyphen"),
    ),
    (
        "dosage_thin_space",
        "unseen",
        lambda row: dosage_variant(row, "\u2009", "dosage_thin_space"),
    ),
    (
        "semicolon_after_number",
        "unseen",
        lambda row: punctuation_variant(row, ";", "semicolon_after_number"),
    ),
    ("zero_width_inside", "unseen", zero_width_variant),
]


def labels_from_char_mask(text: str, char_mask: list[int]):
    words, offsets = word_offsets(text)
    labels = [int(any(char_mask[start:end])) for start, end in offsets]
    return words, offsets, labels


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paired clean/noisy robustness cases"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="v1.4-annotated JSONL",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--per-noise", type=int, default=100)
    args = parser.parse_args()

    rows = []
    for path in args.input:
        rows.extend(read_jsonl(path))
    pairs = []
    counts = {}
    for noise_name, group, transform in TRANSFORMS:
        count = 0
        for row in rows:
            if count >= args.per_noise:
                break
            variant = transform(row)
            if variant is None:
                continue
            noisy_words, _, noisy_labels = labels_from_char_mask(
                variant.text, variant.char_mask
            )
            pair_id = f"{row['id']}::{noise_name}"
            pairs.append(
                {
                    "id": pair_id,
                    "pair_id": pair_id,
                    "source_id": row["id"],
                    "dataset_name": row.get(
                        "dataset_name", row.get("source", "unknown")
                    ),
                    "teacher_policy": row.get("teacher_policy"),
                    "noise_type": noise_name,
                    "noise_group": group,
                    "clean_text": row["text"],
                    "clean_words": row["words"],
                    "clean_labels": row["labels"],
                    "clean_target": list(variant.clean_target),
                    "text": variant.text,
                    "words": noisy_words,
                    "labels": noisy_labels,
                    "noisy_target": list(variant.noisy_target),
                    "edit": variant.edit,
                }
            )
            count += 1
        counts[noise_name] = count
    write_jsonl(args.output, pairs)
    summary = {
        "inputs": args.input,
        "output": args.output,
        "pairs": len(pairs),
        "per_noise_requested": args.per_noise,
        "counts": counts,
        "note": (
            "Gold labels are projected from clean v1.4 labels through "
            "deterministic edits."
        ),
    }
    Path(args.output).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
