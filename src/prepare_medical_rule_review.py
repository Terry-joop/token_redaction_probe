"""Create a small, readable human-review set for the current medical rule.

This is a qualitative policy audit, not a new training set.  It samples 20
test examples from each medical dataset, initializes `human_labels` with the
current rule, and leaves a reviewer to correct the word-level 0/1 labels.
"""

from __future__ import annotations

import html
import json
import random
from pathlib import Path

from common import write_jsonl


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "drug": ("Drug Reviews", "약물 리뷰"),
    "symptom2dx": ("Symptom2Dx", "증상→진단"),
    "adr": ("ADR", "약물 부작용"),
    "redditmh": ("RedditMH", "정신건강 서술"),
    "mednli": ("MedNLI", "임상 문장쌍"),
    "mentalhealth": ("Mental Health", "정신상태 분류"),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def choose(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Mix light and heavy rule masking so review is not all long drug lists."""
    positive = [row for row in rows if sum(row["labels"]) > 0]
    light = [row for row in positive if 1 <= sum(row["labels"]) <= 3]
    heavy = [row for row in positive if sum(row["labels"]) >= 4]
    rng = random.Random(seed)
    rng.shuffle(light)
    rng.shuffle(heavy)
    first = count // 2
    selected = light[:first] + heavy[: count - first]
    if len(selected) < count:
        remaining = [row for row in positive if row not in selected]
        rng.shuffle(remaining)
        selected += remaining[: count - len(selected)]
    return selected


def marked(words: list[str], labels: list[int]) -> str:
    parts = []
    for index, (word, label) in enumerate(zip(words, labels)):
        value = html.escape(word)
        if label:
            value = f"<mark>{value}</mark>"
        parts.append(f"<span title='word index {index}'>{value}</span>")
    return " ".join(parts)


def write_html(rows: list[dict], path: Path) -> None:
    sections = []
    for dataset, (name, description) in DATASETS.items():
        samples = [row for row in rows if row["dataset"] == dataset]
        body = "\n".join(
            "<tr>"
            f"<td>{html.escape(row['id'])}</td>"
            f"<td>{sum(row['rule_labels'])}</td>"
            f"<td>{html.escape(', '.join(row['rule_selected_words']))}</td>"
            f"<td class='text'>{marked(row['words'], row['rule_labels'])}</td>"
            "</tr>" for row in samples
        )
        sections.append(f"""<section><h2>{name} <small>{description}</small></h2>
        <table><thead><tr><th>id</th><th>규칙 mask 수</th><th>규칙이 가린 단어</th><th>원문 (노랑=규칙 mask)</th></tr></thead><tbody>{body}</tbody></table></section>""")
    path.write_text(f"""<!doctype html><meta charset='utf-8'><title>의료 규칙 검수</title>
    <style>body{{font-family:system-ui,sans-serif;max-width:1550px;margin:32px auto;background:#f8fafc;color:#172033;padding:0 22px}} section{{background:#fff;border:1px solid #dbe4ee;border-radius:12px;margin:22px 0;overflow:hidden}}h1,h2{{padding:0 16px}}small{{font-weight:400;color:#607083}}p{{padding:0 16px;line-height:1.65}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:10px;border-top:1px solid #e7edf3;vertical-align:top;text-align:left}}th{{background:#edf4f8}}.text{{min-width:700px;line-height:2.1}}mark{{background:#ffe08a;padding:2px 3px;border-radius:3px}}</style>
    <h1>의료 규칙 라벨 검수 세트</h1>
    <p>노랑은 현재 RedactFormer medterm 규칙이 가린 토큰입니다. 이 HTML은 읽기 전용입니다. 실제 수정은 같은 폴더의 <code>medical_rule_review_v1.jsonl</code>에서 합니다.</p>
    <p>각 문장에서 (1) 노랑 토큰이 정말 가려야 하는지, (2) 노랑이 아닌데 가려야 할 토큰이 있는지 확인하세요. JSONL의 <code>human_labels</code>를 0/1로 고친 뒤 <code>human_reviewed: true</code>로 바꾸고, 이유는 <code>human_review_reason</code>에 짧게 남기면 됩니다.</p>
    {''.join(sections)}""", encoding="utf-8")


def main() -> None:
    output_dir = ROOT / "data" / "human_review"
    output_dir.mkdir(exist_ok=True)
    out = []
    for number, (dataset, (name, _)) in enumerate(DATASETS.items()):
        test = read_jsonl(ROOT / "data" / "full_redactor" / dataset / "test.jsonl")
        for row in choose(test, count=20, seed=20260820 + number):
            labels = list(row["labels"])
            out.append({
                "id": row["id"], "dataset": dataset, "dataset_name": name,
                "text": row["text"], "words": row["words"],
                "rule_labels": labels,
                "rule_types": row.get("types", ["O"] * len(labels)),
                "rule_selected_words": [word for word, label in zip(row["words"], labels) if label],
                "human_labels": labels,
                "human_reviewed": False,
                "human_review_reason": "",
            })
    jsonl = output_dir / "medical_rule_review_v1.jsonl"
    write_jsonl(jsonl, out)
    write_html(out, output_dir / "medical_rule_review_v1.html")
    (output_dir / "MEDICAL_RULE_REVIEW_GUIDE.md").write_text(
        "# 의료 규칙 검수 방법\n\n"
        "`medical_rule_review_v1.html`을 먼저 열어 노랑으로 표시된 규칙 mask를 읽습니다. "
        "수정은 `medical_rule_review_v1.jsonl`에서 합니다.\n\n"
        "- 가려야 하는 단어: `human_labels`의 해당 위치를 `1`\n"
        "- 가리면 안 되는 단어: 해당 위치를 `0`\n"
        "- 한 문장을 확인했으면 `human_reviewed`를 `true`\n"
        "- 규칙의 문제를 한 줄로 `human_review_reason`에 기록\n\n"
        "`words`와 `human_labels`는 반드시 같은 길이를 유지해야 합니다.\n",
        encoding="utf-8",
    )
    print(f"wrote {len(out)} rows: {jsonl}")


if __name__ == "__main__":
    main()
