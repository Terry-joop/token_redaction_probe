from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROBUSTNESS_ROOT = Path(__file__).resolve().parent / "robustness"
if str(ROBUSTNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBUSTNESS_ROOT))

from robustness.evaluate import StudentPredictor, metric
from robustness.v14_rule_adapter import read_jsonl, write_jsonl
from medical_common import word_offsets


PATTERNS = {
    "glued_dosage": (
        "1.3",
        re.compile(
            r"\b\d+(?:[.,]\d+)?(?:mg|mcg|ug|ml|l|g|kg|cc|units?|iu|mmol|meq|%|"
            r"mmhg|bpm|cm|mm)\b",
            re.IGNORECASE,
        ),
    ),
    "c1_control": ("1.3", re.compile(r"\S*[\u0080-\u009f]\S*")),
    "possessive": (
        "1.3",
        re.compile(r"\b[A-Z][A-Za-z-]+(?:['\u2019]{1,2}s)\b"),
    ),
    "long_identifier": (
        "1.3",
        re.compile(
            r"\b(?:\d{7,}|\d{3}-\d{4}|\d{3}-\d{3}-\d{4}|"
            r"\d{3}\.\d{3}\.\d{4}|\d{3}-\d{2}-\d{4}|"
            r"\d{2,3}-\d{2,3}-\d{2,3})\b"
        ),
    ),
    "email": ("1.4", re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")),
    "url": (
        "1.4",
        re.compile(r"(?:https?://|www\.)[\w./%\-?=&#+~]+", re.IGNORECASE),
    ),
    "social_handle": (
        "1.4",
        re.compile(r"(?<![\w.@])@[A-Za-z][\w.]{2,}"),
    ),
    "zip4": ("1.4", re.compile(r"\b\d{5}-\d{4}\b")),
    "numeric_date": (
        "1.4",
        re.compile(
            r"\b(?:0?[1-9]|[12]\d|3[01])[/.-]"
            r"(?:0?[1-9]|[12]\d|3[01])[/.-](?:\d{4}|\d{2})\b"
        ),
    ),
}

DATASET_DEFECTS = {
    "drug": ("glued_dosage",),
    "bios": (
        "c1_control",
        "possessive",
        "long_identifier",
        "email",
        "url",
        "social_handle",
        "zip4",
        "numeric_date",
    ),
}


def stable_key(*values: object) -> str:
    return hashlib.sha256("::".join(map(str, values)).encode("utf-8")).hexdigest()


def build_candidates(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    defects = DATASET_DEFECTS[args.dataset]
    candidates: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        text = row["text"]
        for defect in defects:
            introduced_in, pattern = PATTERNS[defect]
            for occurrence, match in enumerate(pattern.finditer(text)):
                candidate_id = (
                    f"{row['id']}::temporal::{defect}::{match.start()}::{occurrence}"
                )
                candidates[defect].append(
                    {
                        "id": candidate_id,
                        "source_id": row["id"],
                        "text": text,
                        "task_label": row.get("task_label"),
                        "source": row.get("source", args.dataset),
                        "source_split": row.get("source_split", "test"),
                        "temporal_defect": defect,
                        "introduced_in": introduced_in,
                        "target": [match.start(), match.end()],
                        "target_text": match.group(),
                        "selection": "held-out corpus regex candidate before rule inference",
                    }
                )
    selected = []
    counts = {}
    for defect in defects:
        ordered = sorted(
            candidates[defect],
            key=lambda row: stable_key(row["id"], row["target_text"]),
        )
        chosen = ordered[: args.max_per_defect]
        selected.extend(chosen)
        counts[defect] = {
            "available": len(ordered),
            "selected": len(chosen),
        }
    selected.sort(key=lambda row: row["id"])
    write_jsonl(args.output, selected)
    summary = {
        "dataset": args.dataset,
        "input": str(Path(args.input)),
        "output": str(Path(args.output)),
        "source_rows": len(rows),
        "candidate_targets": len(selected),
        "max_per_defect": args.max_per_defect,
        "counts": counts,
        "protocol": (
            "Deterministic scan of the existing held-out test split. Selection uses "
            "surface form only; old/student/latest predictions are not consulted."
        ),
    }
    Path(args.output).with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def target_indices(row: dict, latest_labels: list[int]) -> list[int]:
    start, end = row["target"]
    words, offsets = word_offsets(row["text"])
    if words != row["words"]:
        raise ValueError(f"stable tokenization mismatch: {row['id']}")
    return [
        index
        for index, ((word_start, word_end), label) in enumerate(
            zip(offsets, latest_labels)
        )
        if label and word_start < end and start < word_end
    ]


def detected(indices: list[int], labels: list[int]) -> bool:
    return bool(indices) and all(labels[index] for index in indices)


def bootstrap_delta(
    old: np.ndarray,
    student: np.ndarray,
    source_ids: list[str],
    seed: int,
    repeats: int,
) -> list[float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, source_id in enumerate(source_ids):
        grouped[source_id].append(index)
    keys = list(grouped)
    cluster_delta = np.asarray(
        [np.mean(student[grouped[key]] - old[grouped[key]]) for key in keys]
    )
    rng = np.random.default_rng(seed)
    sampled = rng.choice(cluster_delta, size=(repeats, len(cluster_delta)), replace=True)
    low, high = np.percentile(sampled.mean(axis=1), [2.5, 97.5])
    return [float(low), float(high)]


def evaluate(args: argparse.Namespace) -> None:
    old_rows = read_jsonl(args.old_annotations)
    latest_rows = read_jsonl(args.latest_annotations)
    if [row["id"] for row in old_rows] != [row["id"] for row in latest_rows]:
        raise ValueError("old/latest annotation order mismatch")
    student = StudentPredictor(args.model_dir, args.device, args.threshold)
    student_predictions = student.predict_many(
        [row["words"] for row in latest_rows], batch_size=args.batch_size
    )
    accepted = []
    for old, latest, student_labels in zip(
        old_rows, latest_rows, student_predictions
    ):
        if old["words"] != latest["words"]:
            raise ValueError(f"tokenization mismatch: {latest['id']}")
        indices = target_indices(latest, latest["labels"])
        if not indices:
            continue
        accepted.append((old, latest, student_labels, indices))
    if not accepted:
        raise ValueError("latest rule did not validate any temporal target")

    systems = {
        "past_rule_v1_2": [item[0]["labels"] for item in accepted],
        "past_student_v1_2": [item[2] for item in accepted],
        "latest_rule_v1_4": [item[1]["labels"] for item in accepted],
    }
    gold_rows = [item[1]["labels"] for item in accepted]
    flat_gold = [label for row in gold_rows for label in row]
    output_systems = {}
    for name, predictions in systems.items():
        flags = np.asarray(
            [detected(item[3], labels) for item, labels in zip(accepted, predictions)],
            dtype=np.float64,
        )
        flat_predictions = [label for row in predictions for label in row]
        output_systems[name] = {
            "target_detection": float(flags.mean()),
            "token_audit_against_latest_rule": metric(flat_gold, flat_predictions),
        }

    by_defect = {}
    old_flags = np.asarray(
        [detected(item[3], item[0]["labels"]) for item in accepted], dtype=np.float64
    )
    student_flags = np.asarray(
        [detected(item[3], item[2]) for item in accepted], dtype=np.float64
    )
    for defect in sorted({item[1]["temporal_defect"] for item in accepted}):
        indices = [
            index
            for index, item in enumerate(accepted)
            if item[1]["temporal_defect"] == defect
        ]
        old_values = old_flags[indices]
        student_values = student_flags[indices]
        source_ids = [accepted[index][1]["source_id"] for index in indices]
        by_defect[defect] = {
            "introduced_in": accepted[indices[0]][1]["introduced_in"],
            "targets": len(indices),
            "unique_sources": len(set(source_ids)),
            "past_rule_detection": float(old_values.mean()),
            "past_student_detection": float(student_values.mean()),
            "student_minus_rule": float((student_values - old_values).mean()),
            "student_minus_rule_ci95": bootstrap_delta(
                old_values,
                student_values,
                source_ids,
                args.seed,
                args.bootstrap_repeats,
            ),
        }
    all_source_ids = [item[1]["source_id"] for item in accepted]
    result = {
        "dataset": args.dataset,
        "protocol": "v1.2 code/labels/student replayed on v1.3-v1.4 patch targets",
        "past_rule_commit": "b8dff7e",
        "latest_rule_commit": args.latest_commit,
        "candidate_targets": len(latest_rows),
        "latest_validated_targets": len(accepted),
        "unique_sources": len(set(all_source_ids)),
        "student_model_dir": str(Path(args.model_dir)),
        "student_threshold": student.threshold,
        "systems": output_systems,
        "student_minus_past_rule_target_detection": float(
            (student_flags - old_flags).mean()
        ),
        "student_minus_past_rule_ci95": bootstrap_delta(
            old_flags,
            student_flags,
            all_source_ids,
            args.seed,
            args.bootstrap_repeats,
        ),
        "by_defect": by_defect,
        "limitations": [
            "Gold targets are latest-rule-validated pseudo-gold, not human-gold.",
            "Source corpora are replayed from current stored splits; code and labels are v1.2.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="RedactFormer rule-version time-axis evaluation")
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-candidates")
    build.add_argument("--dataset", choices=sorted(DATASET_DEFECTS), required=True)
    build.add_argument("--input", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--max-per-defect", type=int, default=200)
    build.set_defaults(run=build_candidates)

    score = commands.add_parser("evaluate")
    score.add_argument("--dataset", choices=sorted(DATASET_DEFECTS), required=True)
    score.add_argument("--old-annotations", required=True)
    score.add_argument("--latest-annotations", required=True)
    score.add_argument("--model-dir", required=True)
    score.add_argument("--threshold", type=float)
    score.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    score.add_argument("--batch-size", type=int, default=128)
    score.add_argument("--bootstrap-repeats", type=int, default=2000)
    score.add_argument("--seed", type=int, default=42)
    score.add_argument("--latest-commit", default="045f3f3")
    score.add_argument("--output", required=True)
    score.set_defaults(run=evaluate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
