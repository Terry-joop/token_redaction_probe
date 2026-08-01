from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_common import word_offsets  # noqa: E402


MASKER_MODULE = {
    "medterm5": "make_medterm_v5",
    "piiclean2": "make_pii_clean_v2",
    "mdccunion": "make_mdcc_union",
}


@dataclass
class RulePrediction:
    words: list[str]
    offsets: list[tuple[int, int]]
    labels: list[int]


class V14RuleAdapter:
    """Run the exact RedactFormer v1.4 builder on raw text.

    The production builders operate on RoBERTa token ids. This adapter keeps that
    path intact, then projects the resulting subtoken mask onto this repository's
    stable word tokens using character offsets.
    """

    def __init__(
        self,
        masker: str,
        task: str,
        redactformer_root: str | Path,
        max_length: int = 128,
    ) -> None:
        if masker not in MASKER_MODULE:
            raise ValueError(f"unknown masker: {masker}")
        self.masker = masker
        self.task = task
        self.root = Path(redactformer_root).resolve()
        self.max_length = max_length
        builders = self.root / "scripts" / "dataset_builders"
        if not builders.is_dir():
            raise FileNotFoundError(f"RedactFormer builders not found: {builders}")
        if str(builders) not in sys.path:
            sys.path.insert(0, str(builders))
        old_argv = sys.argv[:]
        try:
            sys.argv = [MASKER_MODULE[masker] + ".py", task]
            self.module = importlib.import_module(MASKER_MODULE[masker])
        finally:
            sys.argv = old_argv
        manifest = importlib.import_module("_manifest")
        fingerprint = manifest.code_fingerprint(str(Path(self.module.__file__).resolve()))
        self.rule_version = manifest.RULE_VERSION
        self.code_sha256 = fingerprint["code_sha256"]

    @property
    def policy_id(self) -> str:
        return (
            f"redactformer-{self.masker}-rule-v{self.rule_version}"
            f"@{self.code_sha256[:16]}"
        )

    def predict(self, text: str) -> RulePrediction:
        encoded = self.module.tok(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        input_ids = list(encoded["input_ids"])
        offsets = [tuple(pair) for pair in encoded["offset_mapping"]]
        _, subtoken_mask, _ = self.module.process(input_ids, len(input_ids))
        words, word_spans = word_offsets(text)
        labels = []
        for word_start, word_end in word_spans:
            hit = any(
                bool(subtoken_mask[index])
                and token_start < word_end
                and word_start < token_end
                for index, (token_start, token_end) in enumerate(offsets)
                if token_end > token_start
            )
            labels.append(int(hit))
        return RulePrediction(words=words, offsets=word_spans, labels=labels)

    def annotate_row(self, row: dict[str, Any]) -> dict[str, Any]:
        prediction = self.predict(row["text"])
        out = dict(row)
        out.update(
            {
                "words": prediction.words,
                "labels": prediction.labels,
                "selected_words": [
                    word
                    for word, label in zip(prediction.words, prediction.labels)
                    if label
                ],
                "teacher_policy": self.policy_id,
                "annotation_source": self.policy_id,
                "rule_manifest": {
                    "version": self.rule_version,
                    "masker": self.masker,
                    "code_sha256": self.code_sha256,
                    "task": self.task,
                    "max_length": self.max_length,
                },
            }
        )
        return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
