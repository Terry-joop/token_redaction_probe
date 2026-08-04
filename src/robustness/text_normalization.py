from __future__ import annotations

import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medical_common import word_offsets


APOSTROPHES = {"’", "ʼ", "\x92"}
INVISIBLE = {"\u200b", "\u200c", "\u200d", "\ufeff", "\u2060"}
DOSAGE_UNITS = "mg|mcg|g|kg|ml|l|iu|units?|mm|cm"


@dataclass
class NormalizedText:
    text: str
    source_indices: list[set[int]]


def normalize_with_alignment(text: str) -> NormalizedText:
    chars: list[str] = []
    sources: list[set[int]] = []
    for index, original in enumerate(text):
        expanded = unicodedata.normalize("NFKC", original)
        for char in expanded:
            if char in INVISIBLE or unicodedata.category(char) == "Cc":
                continue
            if char in APOSTROPHES:
                char = "'"
            if char.isspace():
                char = " "
            if char == " " and chars and chars[-1] == " ":
                sources[-1].add(index)
                continue
            chars.append(char)
            sources.append({index})

    # A separator inserted between a number and dosage unit is normalized to a
    # plain space. This covers 25-mg and 25\u2009mg without consulting noise labels.
    interim = "".join(chars)
    for match in reversed(list(re.finditer(rf"(?<=\d)-(?=(?:{DOSAGE_UNITS})\b)", interim, re.I))):
        chars[match.start()] = " "

    # Punctuation appended after a numeric token is ignored at a token boundary;
    # thousands separators such as 1,000 are intentionally preserved.
    interim = "".join(chars)
    for match in reversed(list(re.finditer(r"(?<=\d)[,;:](?=\s|$)", interim))):
        del chars[match.start()]
        del sources[match.start()]

    # Adjacent number+unit is split, supporting the previously documented 25mg seam.
    interim = "".join(chars)
    for match in reversed(list(re.finditer(rf"(?<=\d)(?=(?:{DOSAGE_UNITS})\b)", interim, re.I))):
        boundary = match.start()
        anchor = set()
        if boundary:
            anchor.update(sources[boundary - 1])
        if boundary < len(sources):
            anchor.update(sources[boundary])
        chars.insert(boundary, " ")
        sources.insert(boundary, anchor)

    return NormalizedText("".join(chars).strip(), _strip_alignment(chars, sources))


def _strip_alignment(chars: list[str], sources: list[set[int]]) -> list[set[int]]:
    start = 0
    end = len(chars)
    while start < end and chars[start] == " ":
        start += 1
    while end > start and chars[end - 1] == " ":
        end -= 1
    return sources[start:end]


def project_normalized_labels(
    original_text: str,
    normalized: NormalizedText,
    normalized_labels: list[int],
) -> list[int]:
    _, normalized_offsets = word_offsets(normalized.text)
    if len(normalized_offsets) != len(normalized_labels):
        raise ValueError("normalized label/token mismatch")
    original_mask = [0] * len(original_text)
    for (start, end), label in zip(normalized_offsets, normalized_labels):
        if not label:
            continue
        for norm_index in range(start, end):
            for source_index in normalized.source_indices[norm_index]:
                original_mask[source_index] = 1
    _, original_offsets = word_offsets(original_text)
    return [int(any(original_mask[start:end])) for start, end in original_offsets]
