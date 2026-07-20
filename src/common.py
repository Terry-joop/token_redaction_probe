import json
import re
from pathlib import Path


WORD_RE = re.compile(r"\w+(?:['’-]\w+)*|[^\w\s]", re.UNICODE)


def word_tokenize(text: str) -> list[str]:
    """Stable word-level tokenization used by annotator, trainer, and inference."""
    return WORD_RE.findall(text)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if len(row["words"]) != len(row["labels"]):
                raise ValueError(f"{path}:{line_number}: words/labels length mismatch")
            if not set(row["labels"]).issubset({0, 1}):
                raise ValueError(f"{path}:{line_number}: labels must be 0 or 1")
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_redaction(words: list[str], labels: list[int]) -> str:
    rendered = ["[REDACTED]" if label else word for word, label in zip(words, labels)]
    text = " ".join(rendered)
    return re.sub(r"\s+([.,!?;:])", r"\1", text)

