import json
from pathlib import Path

from common import WORD_RE


SENSITIVE_TYPES = {
    "O", "DRUG", "DISEASE", "SYMPTOM", "DOSAGE", "TREATMENT", "TEST",
    "ANATOMY", "MENTAL_HEALTH", "PERSON", "ORG", "LOCATION", "DATE",
    "OTHER_SENSITIVE", "MEDICAL",
}


def read_records(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
    return rows


def word_offsets(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    matches = list(WORD_RE.finditer(text))
    return [match.group() for match in matches], [match.span() for match in matches]


def labels_to_spans(labels: list[int]) -> list[tuple[int, int]]:
    """Convert word labels to half-open contiguous word spans."""
    spans = []
    start = None
    for index, label in enumerate(labels + [0]):
        if label and start is None:
            start = index
        elif not label and start is not None:
            spans.append((start, index))
            start = None
    return spans


def validate_labels(example_id: str, words: list[str], labels: object) -> list[int]:
    if not isinstance(labels, list) or len(labels) != len(words):
        got = len(labels) if isinstance(labels, list) else type(labels).__name__
        raise ValueError(f"{example_id}: labels length must be {len(words)}, got {got}")
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError(f"{example_id}: labels must contain integer 0/1 only")
    return labels


def validate_types(example_id: str, words: list[str], types: object | None) -> list[str]:
    if types is None:
        return ["O"] * len(words)
    if not isinstance(types, list) or len(types) != len(words):
        raise ValueError(f"{example_id}: types length must be {len(words)}")
    normalized = [str(value).upper() for value in types]
    unknown = sorted(set(normalized) - SENSITIVE_TYPES)
    if unknown:
        raise ValueError(f"{example_id}: unknown sensitive types: {unknown}")
    return normalized
