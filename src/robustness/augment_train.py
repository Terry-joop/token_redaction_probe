from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_pairs import TRANSFORMS, labels_from_char_mask
from v14_rule_adapter import read_jsonl, write_jsonl


def stable_tie(row_id: str, noise_name: str) -> str:
    return hashlib.sha256(
        f"{row_id}::{noise_name}".encode("utf-8")
    ).hexdigest()


def make_noisy_row(row: dict, noise_name: str, variant) -> dict:
    words, _, labels = labels_from_char_mask(
        variant.text, variant.char_mask
    )
    output = dict(row)
    output.update(
        {
            "id": f"{row['id']}::aug::{noise_name}",
            "source_id": row["id"],
            "text": variant.text,
            "words": words,
            "labels": labels,
            "augmentation": {
                "group": "seen",
                "noise_type": noise_name,
                "edit": variant.edit,
                "gold_projection": "clean_v1.4_char_mask",
            },
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Add one balanced seen-noise variant per eligible clean row. "
            "Labels are projected from clean v1.4 character spans."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variants-per-row",
        type=int,
        default=1,
        choices=[1, 2],
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="write only augmented rows",
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]
    seen_transforms = [
        (name, transform)
        for name, group, transform in TRANSFORMS
        if group == "seen"
    ]
    counts: Counter[str] = Counter()
    eligible: Counter[str] = Counter()
    output_rows = []

    for row in rows:
        if not args.no_clean:
            clean = dict(row)
            clean["augmentation"] = {
                "group": "clean",
                "noise_type": "none",
            }
            output_rows.append(clean)

        candidates = []
        for noise_name, transform in seen_transforms:
            variant = transform(row)
            if variant is None:
                continue
            eligible[noise_name] += 1
            candidates.append((noise_name, variant))

        selected = []
        for _ in range(min(args.variants_per_row, len(candidates))):
            noise_name, variant = min(
                candidates,
                key=lambda item: (
                    counts[item[0]],
                    stable_tie(str(row["id"]), item[0]),
                ),
            )
            selected.append((noise_name, variant))
            counts[noise_name] += 1
            candidates = [
                item for item in candidates if item[0] != noise_name
            ]

        for noise_name, variant in selected:
            output_rows.append(
                make_noisy_row(row, noise_name, variant)
            )

    write_jsonl(args.output, output_rows)
    summary = {
        "input": args.input,
        "output": args.output,
        "clean_rows": 0 if args.no_clean else len(rows),
        "augmented_rows": int(sum(counts.values())),
        "total_rows": len(output_rows),
        "variants_per_row": args.variants_per_row,
        "eligible_by_noise": dict(sorted(eligible.items())),
        "selected_by_noise": dict(sorted(counts.items())),
        "train_noise_groups": ["seen"],
        "held_out_noise_group": "unseen",
        "gold": (
            "MASKING_FRAMEWORK v1.4 clean labels projected through "
            "deterministic character edits"
        ),
    }
    summary_path = Path(args.output).with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
