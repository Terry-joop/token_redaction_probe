"""Export readable rule-vs-ELECTRA examples from the six medical test sets.

The dashboard has aggregate scores; this script makes the underlying behaviour
inspectable.  It deliberately samples the fixed *test* split and shows both
agreement and disagreement cases.  The `teacher` column is the current
RedactFormer medterm rule pseudo-label, not an LLM label.
"""

from __future__ import annotations

import argparse
import html
import json
import random
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from train import RedactionModel


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "drug": ("Drug Reviews", "의료: 약물·용량·증상", "artifacts/medical_redactor/cross_dataset_in_domain/drug_electra_small"),
    "symptom2dx": ("Symptom2Dx", "의료: 증상→진단", "artifacts/medical_redactor/cross_dataset_in_domain/symptom2dx_electra_small"),
    "adr": ("ADR", "의료: 약물 부작용", "artifacts/medical_redactor/cross_dataset_in_domain/adr_electra_small"),
    "redditmh": ("RedditMH", "비정형: 정신건강 서술", "artifacts/medical_redactor/cross_dataset_in_domain/redditmh_electra_small"),
    "mednli": ("MedNLI", "의료: 임상 문장쌍", "artifacts/medical_redactor/core_matrix/extension_seed42/mednli_electra_small_seed42"),
    "mentalhealth": ("Mental Health", "비정형: 정신건강 서술", "artifacts/medical_redactor/core_matrix/extension_seed42/mentalhealth_electra_small_seed42"),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def threshold_for(dataset: str) -> float:
    result = json.loads((ROOT / "reports/full_dataset_results.json").read_text(encoding="utf-8"))
    return result["datasets"][dataset]["models"]["electra_small"]["budget_matched"]["test"]["threshold"]


def load_model(model_dir: Path):
    config = json.loads((model_dir / "experiment.json").read_text(encoding="utf-8"))
    kwargs = {"local_files_only": True}
    if "roberta" in config["model_name"].lower():
        kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_dir, **kwargs)
    model = RedactionModel(config["model_name"], config["hidden_size"], config["freeze_encoder"])
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return tokenizer, model, config["max_length"]


def predict(tokenizer, model, words: list[str], max_length: int, threshold: float):
    encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=max_length)
    with torch.no_grad():
        scores = model(**encoded).softmax(-1)[0, :, 1]
    output, seen = [], set()
    for index, word_id in enumerate(encoded.word_ids(0)):
        if word_id is not None and word_id not in seen:
            output.append(float(scores[index]))
            seen.add(word_id)
    # Do not present unscored truncation tail as a confident "not redacted" prediction.
    return output + [None] * (len(words) - len(output)), [int(x >= threshold) for x in output] + [None] * (len(words) - len(output))


def outcome(gold: list[int], pred: list[int | None]) -> str:
    if any(value is None for value in pred):
        return "truncated"
    missed = any(a and not b for a, b in zip(gold, pred))
    extra = any(not a and b for a, b in zip(gold, pred))
    if missed and extra:
        return "missed+overmask"
    if missed:
        return "missed"
    if extra:
        return "overmask"
    return "agreement"


def selected_words(words, labels):
    return [word for word, label in zip(words, labels) if label == 1]


def sample_examples(rows: list[dict], per_outcome: int, seed: int):
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["outcome"], []).append(row)
    picked = []
    # Showing errors first makes the file useful for policy review.
    for index, key in enumerate(("missed", "overmask", "missed+overmask", "agreement", "truncated")):
        candidates = grouped.get(key, [])
        random.Random(seed + index).shuffle(candidates)
        picked.extend(candidates[:per_outcome])
    return picked


def word_markup(row: dict) -> str:
    chunks = []
    for word, rule, student, score in zip(row["words"], row["teacher_labels"], row["student_labels"], row["student_scores"]):
        classes = []
        if rule:
            classes.append("rule")
        if student == 1:
            classes.append("student")
        if student is None:
            classes.append("truncated")
        value = html.escape(word)
        if score is not None:
            value += f'<small>{score:.2f}</small>'
        chunks.append(f'<span class="{" ".join(classes)}">{value}</span>')
    return " ".join(chunks)


def render_html(rows_by_dataset: dict[str, list[dict]], output: Path) -> None:
    css = """
body{font-family:system-ui,sans-serif;max-width:1500px;margin:34px auto;background:#f7fafc;color:#172033;line-height:1.5;padding:0 22px}
h1{margin-bottom:4px}.note{color:#526174}.dataset{background:white;border:1px solid #dbe4ee;border-radius:12px;margin:25px 0;overflow:hidden}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:12px;border-bottom:1px solid #e9eef4;vertical-align:top;text-align:left}th{background:#edf4f8;position:sticky;top:0}
.text{min-width:600px}.rule{background:#ffe3e3;border-bottom:2px solid #d64b4b;border-radius:3px;padding:1px 2px}.student{box-shadow:inset 0 -2px #237c6a}.rule.student{background:#dff5e9}.truncated{color:#8b95a1;text-decoration:line-through}.outcome{font-weight:700;white-space:nowrap}.missed{color:#bd2c24}.overmask{color:#a96b00}.agreement{color:#17735f}small{color:#637083;font-size:10px;margin-left:2px}.words{color:#526174;max-width:230px}.legend span{margin-right:12px}
"""
    content = ["<!doctype html><meta charset='utf-8'><title>의료 Redaction 예시</title>", f"<style>{css}</style>", "<h1>의료 데이터셋: 규칙 teacher vs ELECTRA-small Student 예시</h1>", "<p class='note'>빨강 밑줄=현재 RedactFormer medterm 규칙이 가림, 초록 밑줄=ELECTRA-small이 가림, 초록 배경=둘 다 가림. 작은 회색 숫자는 Student의 redaction 확률이다. 이 파일의 teacher는 <b>규칙 pseudo-label</b>이며 GPT teacher가 아니다.</p>", "<p class='note'>각 데이터셋의 고정 test split에서 오류·일치 사례를 균형 있게 뽑았다. 긴 입력이 모델 max length(256)를 넘은 경우는 truncated으로 따로 표시했다.</p>", "<p class='legend'><span class='rule'>규칙만</span><span class='student'>Student만</span><span class='rule student'>둘 다</span></p>"]
    for key, rows in rows_by_dataset.items():
        name, domain, _ = DATASETS[key]
        counts = Counter(row["outcome"] for row in rows)
        content.append(f"<section class='dataset'><h2>{name} <small>{domain}</small></h2><p class='note'>표시 예시: {len(rows)}개 · " + ", ".join(f"{html.escape(k)} {v}" for k, v in sorted(counts.items())) + "</p><table><tr><th>id</th><th>분류</th><th>규칙이 가린 단어</th><th>Student가 가린 단어</th><th class='text'>원문 / 토큰별 표시</th></tr>")
        for row in rows:
            content.append("<tr>"
                f"<td>{html.escape(row['id'])}</td><td class='outcome {html.escape(row['outcome'])}'>{html.escape(row['outcome'])}</td>"
                f"<td class='words'>{html.escape(', '.join(row['teacher_selected']))}</td>"
                f"<td class='words'>{html.escape(', '.join(row['student_selected']))}</td>"
                f"<td class='text'>{word_markup(row)}</td></tr>")
        content.append("</table></section>")
    output.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=160, help="fixed test rows inspected per dataset before selection")
    parser.add_argument("--per-outcome", type=int, default=8)
    parser.add_argument("--output", default="reports/medical_rule_student_examples.html")
    args = parser.parse_args()

    all_rows, displayed = {}, {}
    for number, (key, (_, _, model_rel)) in enumerate(DATASETS.items()):
        test = read_jsonl(ROOT / "data/full_redactor" / key / "test.jsonl")
        # Fixed deterministic sample, limited to inputs model can fully score.
        candidates = [row for row in test if 0 < sum(row["labels"]) and len(row["words"]) <= 256]
        random.Random(20260820 + number).shuffle(candidates)
        tokenizer, model, max_length = load_model(ROOT / model_rel)
        rows = []
        for raw in candidates[:args.candidates]:
            scores, labels = predict(tokenizer, model, raw["words"], max_length, threshold_for(key))
            row = {
                "dataset": key, "id": raw["id"], "text": raw["text"], "words": raw["words"],
                "teacher_labels": raw["labels"], "student_labels": labels, "student_scores": scores,
            }
            row["outcome"] = outcome(row["teacher_labels"], row["student_labels"])
            row["teacher_selected"] = selected_words(row["words"], row["teacher_labels"])
            row["student_selected"] = selected_words(row["words"], row["student_labels"])
            rows.append(row)
        all_rows[key] = rows
        displayed[key] = sample_examples(rows, args.per_outcome, 1000 + number)
        print(f"{key}: candidates={len(rows)} shown={len(displayed[key])} outcomes={dict(Counter(x['outcome'] for x in rows))}")

    render_html(displayed, ROOT / args.output)
    json_output = ROOT / args.output.replace(".html", ".jsonl")
    json_output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for rows in displayed.values() for row in rows), encoding="utf-8")
    print(f"wrote {args.output} and {json_output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
