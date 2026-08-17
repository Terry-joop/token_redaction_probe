"""Build a human-readable catalog for the surface perturbation experiment."""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = ROOT / "src" / "robustness"
if str(ROBUSTNESS) not in sys.path:
    sys.path.insert(0, str(ROBUSTNESS))

from build_pairs import TRANSFORMS  # noqa: E402


OUT = ROOT / "reports" / "perturbation_catalog.html"
MARKDOWN_OUT = ROOT / "PERTURBATION_CATALOG.md"
PAIR_PATH = ROOT / "data/robustness/v14_strict/drug/catalog_examples.jsonl"
STRICT_ROOT = ROOT / "data" / "robustness" / "v14_strict"


CATALOG = [
    {
        "id": "WS-01",
        "name": "double_space",
        "title": "이중 공백",
        "family": "공백 경계",
        "group": "seen",
        "condition": "서로 인접한 두 민감 토큰 사이에 일반 공백이 있을 때",
        "change": "일반 공백 1개를 2개로 교체",
        "codepoint": "U+0020 × 2",
        "reason": "공백 하나의 차이로 엔티티 경계가 깨지는 실제 규칙 취약점을 재현한다.",
    },
    {
        "id": "AP-01",
        "name": "curly_apostrophe",
        "title": "곱슬 아포스트로피",
        "family": "아포스트로피·인코딩",
        "group": "seen",
        "condition": "민감 단어 안에 직선 또는 곱슬 아포스트로피가 있을 때",
        "change": "첫 아포스트로피를 곱슬 문자로 교체",
        "codepoint": "U+2019",
        "reason": "문서 편집기나 복사·붙여넣기에서 흔한 따옴표 정규화 차이를 재현한다.",
    },
    {
        "id": "AP-02",
        "name": "c1_apostrophe",
        "title": "C1 제어문자 아포스트로피",
        "family": "아포스트로피·인코딩",
        "group": "seen",
        "condition": "민감 단어 안에 아포스트로피가 있을 때",
        "change": "아포스트로피를 깨진 인코딩 제어문자로 교체",
        "codepoint": "U+0092",
        "reason": "인코딩이 깨진 텍스트에서 이름·의학 용어 인식이 흔들리는 상황을 재현한다.",
    },
    {
        "id": "DS-01",
        "name": "dosage_join",
        "title": "용량 붙여쓰기",
        "family": "용량 경계",
        "group": "seen",
        "condition": "민감한 숫자와 mg·ml·mcg 등 용량 단위가 공백으로 분리됐을 때",
        "change": "숫자와 단위 사이 공백 제거",
        "codepoint": "U+0020 → 삭제",
        "reason": "같은 용량을 띄어 쓰거나 붙여 쓰는 표기 차이를 재현한다.",
    },
    {
        "id": "PT-01",
        "name": "comma_after_number",
        "title": "민감 숫자 뒤 쉼표",
        "family": "구두점 경계",
        "group": "seen",
        "condition": "숫자를 포함한 민감 토큰 바로 뒤에 구두점이 없을 때",
        "change": "민감 토큰 뒤에 쉼표 삽입",
        "codepoint": "U+002C",
        "reason": "생년월일·용량 같은 숫자 경계가 쉼표 하나로 달라지는 상황을 재현한다.",
    },
    {
        "id": "WS-02",
        "name": "triple_space",
        "title": "삼중 공백",
        "family": "공백 경계",
        "group": "unseen",
        "condition": "서로 인접한 두 민감 토큰 사이에 일반 공백이 있을 때",
        "change": "일반 공백 1개를 3개로 교체",
        "codepoint": "U+0020 × 3",
        "reason": "이중 공백을 학습한 모델이 더 강한 같은 계열 변형에도 일반화하는지 본다.",
    },
    {
        "id": "WS-03",
        "name": "nbsp",
        "title": "줄바꿈 방지 공백",
        "family": "공백 경계",
        "group": "unseen",
        "condition": "서로 인접한 두 민감 토큰 사이에 일반 공백이 있을 때",
        "change": "일반 공백을 NBSP로 교체",
        "codepoint": "U+00A0",
        "reason": "화면에는 공백처럼 보이지만 tokenizer에는 다른 문자인 경우를 시험한다.",
    },
    {
        "id": "AP-03",
        "name": "modifier_apostrophe",
        "title": "Modifier letter apostrophe",
        "family": "아포스트로피·인코딩",
        "group": "unseen",
        "condition": "민감 단어 안에 아포스트로피가 있을 때",
        "change": "첫 아포스트로피를 modifier letter로 교체",
        "codepoint": "U+02BC",
        "reason": "곱슬 아포스트로피를 본 모델이 닮은 미관측 Unicode 문자에도 일반화하는지 본다.",
    },
    {
        "id": "DS-02",
        "name": "dosage_hyphen",
        "title": "용량 하이픈",
        "family": "용량 경계",
        "group": "unseen",
        "condition": "민감한 숫자와 용량 단위가 공백으로 분리됐을 때",
        "change": "숫자와 단위 사이 공백을 하이픈으로 교체",
        "codepoint": "U+002D",
        "reason": "붙여쓰기를 본 모델이 다른 용량 연결 기호에도 일반화하는지 본다.",
    },
    {
        "id": "DS-03",
        "name": "dosage_thin_space",
        "title": "용량 가는 공백",
        "family": "용량 경계",
        "group": "unseen",
        "condition": "민감한 숫자와 용량 단위가 공백으로 분리됐을 때",
        "change": "일반 공백을 thin space로 교체",
        "codepoint": "U+2009",
        "reason": "시각적으로 비슷한 Unicode 공백에서도 용량 span이 유지되는지 본다.",
    },
    {
        "id": "PT-02",
        "name": "semicolon_after_number",
        "title": "민감 숫자 뒤 세미콜론",
        "family": "구두점 경계",
        "group": "unseen",
        "condition": "숫자를 포함한 민감 토큰 바로 뒤에 구두점이 없을 때",
        "change": "민감 토큰 뒤에 세미콜론 삽입",
        "codepoint": "U+003B",
        "reason": "쉼표를 본 모델이 다른 미관측 구두점 경계에도 일반화하는지 본다.",
    },
    {
        "id": "IC-01",
        "name": "zero_width_inside",
        "title": "단어 내부 Zero-width space",
        "family": "비가시 문자",
        "group": "unseen",
        "condition": "길이 5자 이상인 민감 영문 단어가 있을 때",
        "change": "민감 단어 가운데에 보이지 않는 문자 삽입",
        "codepoint": "U+200B",
        "reason": "사람에게는 같은 단어로 보이지만 내부 문자열 경계가 깨지는 강한 미관측 교란이다.",
    },
]

# These are evaluated by dashboard section 4 (not the earlier strict 5/7
# protocol).  They were intentionally not used in Student training,
# validation, or threshold selection.
FUTURE_CATALOG = [
    ("F-01", "zwnj_inside", "단어 내부 ZWNJ", "U+200C", "없음"),
    ("F-02", "word_joiner_inside", "단어 내부 word joiner", "U+2060", "없음"),
    ("F-03", "soft_hyphen_inside", "단어 내부 soft hyphen", "U+00AD", "없음"),
    ("F-04", "fullwidth_apostrophe", "전각 아포스트로피", "U+FF07", "없음"),
    ("F-05", "dosage_nb_hyphen", "용량 non-breaking hyphen", "U+2011", "없음"),
    ("F-06", "narrow_nbsp", "좁은 non-breaking 공백", "U+202F", "없음"),
    ("F-07", "right_paren_after_number", "숫자 뒤 닫는 괄호", "U+0029", "전용 패치 없음"),
]

V14_COVERAGE = {
    "double_space": "부분: token render의 중복 공백 축약. raw 이중 공백 전체 보장은 아님",
    "curly_apostrophe": "부분: 소유격 처리에서 U+2019 인식. 범용 정규화는 아님",
    "c1_apostrophe": "직접: v1.3 C1 U+0091~U+0097 → ASCII 1:1 정규화",
    "dosage_join": "직접: v1.3 glued dosage(예: 25mg) 규칙",
    "comma_after_number": "전용 패치 없음",
    "triple_space": "전용 패치 없음",
    "nbsp": "전용 패치 없음",
    "modifier_apostrophe": "전용 패치 없음",
    "dosage_hyphen": "전용 패치 없음",
    "dosage_thin_space": "전용 패치 없음",
    "semicolon_after_number": "전용 패치 없음",
    "zero_width_inside": "전용 패치 없음",
    **{name: coverage for _, name, _, _, coverage in FUTURE_CATALOG},
}


GENERALIZATION_PAIRS = [
    (
        "공백 개수·종류",
        "이중 공백 (WS-01)",
        "삼중 공백 (WS-02), NBSP (WS-03)",
        "공백 2개를 본 뒤 더 많은 공백과 다른 Unicode 공백도 같은 경계로 처리하는가?",
    ),
    (
        "아포스트로피·인코딩",
        "곱슬 아포스트로피 (AP-01), C1 (AP-02)",
        "Modifier apostrophe (AP-03)",
        "학습에 없던 닮은 Unicode 아포스트로피에서도 민감 단어를 유지하는가?",
    ),
    (
        "용량 경계",
        "붙여쓰기 (DS-01)",
        "하이픈 (DS-02), Thin space (DS-03)",
        "숫자와 단위를 잇는 기호가 달라져도 같은 용량 span으로 보는가?",
    ),
    (
        "구두점 경계",
        "숫자 뒤 쉼표 (PT-01)",
        "숫자 뒤 세미콜론 (PT-02)",
        "학습하지 않은 구두점이 숫자 뒤에 붙어도 민감 숫자를 놓치지 않는가?",
    ),
    (
        "완전히 새로운 계열",
        "직접 대응 없음",
        "단어 내부 Zero-width (IC-01)",
        "학습에 같은 계열이 전혀 없어도 보이지 않는 내부 문자 삽입에 견디는가?",
    ),
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_strict_counts() -> dict:
    train_counts: dict[str, int] = {}
    test_counts: dict[str, int] = {}
    clean_train_rows = 0
    augmented_train_rows = 0
    unseen_pairs = 0
    datasets = 0
    for dataset_dir in sorted(STRICT_ROOT.iterdir()):
        train_path = dataset_dir / "train_seen_augmented.summary.json"
        test_path = dataset_dir / "unseen_pairs.summary.json"
        if not (train_path.exists() and test_path.exists()):
            continue
        train = load_json(train_path)
        test = load_json(test_path)
        datasets += 1
        clean_train_rows += train["clean_rows"]
        augmented_train_rows += train["augmented_rows"]
        unseen_pairs += test["pairs"]
        for name, count in train["selected_by_noise"].items():
            train_counts[name] = train_counts.get(name, 0) + count
        for name, count in test["counts"].items():
            test_counts[name] = test_counts.get(name, 0) + count
    return {
        "datasets": datasets,
        "clean_train_rows": clean_train_rows,
        "augmented_train_rows": augmented_train_rows,
        "unseen_pairs": unseen_pairs,
        "train_counts": train_counts,
        "test_counts": test_counts,
    }


def load_examples() -> dict[str, dict]:
    examples: dict[str, dict] = {}
    with PAIR_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            examples.setdefault(row["noise_type"], row)
    return examples


def visible(text: str) -> str:
    replacements = {
        " ": "·",
        "\u00a0": "⟨NBSP:U+00A0⟩",
        "\u2009": "⟨THIN:U+2009⟩",
        "\u200b": "⟨ZWSP:U+200B⟩",
        "\x92": "⟨C1:U+0092⟩",
    }
    return "".join(replacements.get(char, char) for char in text)


def marked_excerpt(text: str, target: list[int], context: int = 42) -> str:
    start, end = target
    left = max(0, start - context)
    right = min(len(text), end + context)
    prefix = "…" if left else ""
    suffix = "…" if right < len(text) else ""
    before = escape(visible(text[left:start]))
    focus = escape(visible(text[start:end]))
    after = escape(visible(text[end:right]))
    return f"{prefix}{before}<mark>{focus}</mark>{after}{suffix}"


def validate_catalog() -> None:
    # This catalog documents the registered strict seen-5/unseen-7 protocol.
    # Later ``future`` defects live in FUTURE_DEFECT_TIME_AXIS.md instead.
    runtime = [
        (name, group)
        for name, group, _ in TRANSFORMS
        if group in {"seen", "unseen"}
    ]
    documented = [(item["name"], item["group"]) for item in CATALOG]
    if runtime != documented:
        raise ValueError(
            "Perturbation catalog does not match build_pairs.TRANSFORMS: "
            f"runtime={runtime!r}, documented={documented!r}"
        )


def catalog_cards(
    examples: dict[str, dict], train_counts: dict, test_counts: dict
) -> str:
    cards = []
    for item in CATALOG:
        row = examples[item["name"]]
        group_label = "학습에 노출" if item["group"] == "seen" else "학습에서 보류"
        train_count = train_counts.get(item["name"], 0)
        final_test_count = test_counts.get(item["name"], 0) if item["group"] == "unseen" else 0
        cards.append(
            f"""
            <article class="rule" id="{item['id'].lower()}">
              <div class="rule-head">
                <div><span class="rule-id">{item['id']}</span><h3>{escape(item['title'])}</h3></div>
                <span class="badge {item['group']}">{group_label}</span>
              </div>
              <dl>
                <div><dt>계열</dt><dd>{escape(item['family'])}</dd></div>
                <div><dt>적용 조건</dt><dd>{escape(item['condition'])}</dd></div>
                <div><dt>변환</dt><dd>{escape(item['change'])}</dd></div>
                <div><dt>문자 코드</dt><dd><code>{escape(item['codepoint'])}</code></dd></div>
              </dl>
              <p class="why">{escape(item['reason'])}</p>
              <div class="example">
                <div><b>Clean</b><code>{marked_excerpt(row['clean_text'], row['clean_target'])}</code></div>
                <div><b>Noisy</b><code>{marked_excerpt(row['text'], row['noisy_target'])}</code></div>
              </div>
              <div class="counts">전체 10개 데이터셋 합계 · 학습 증강 <b>{train_count:,}</b>행 · 최종 unseen test <b>{final_test_count:,}</b>쌍 · 예문은 Drug Reviews <code>{escape(row['source_id'])}</code></div>
            </article>
            """
        )
    return "".join(cards)


def count_rows(train_counts: dict, test_counts: dict) -> str:
    rows = []
    for item in CATALOG:
        rows.append(
            "<tr>"
            f"<td class='left'><code>{item['id']}</code></td>"
            f"<td class='left'>{escape(item['title'])}</td>"
            f"<td>{'Seen' if item['group'] == 'seen' else 'Unseen'}</td>"
            f"<td>{train_counts.get(item['name'], 0):,}</td>"
            f"<td>{test_counts.get(item['name'], 0) if item['group'] == 'unseen' else 0:,}</td>"
            "</tr>"
        )
    return "".join(rows)


def generalization_rows() -> str:
    return "".join(
        "<tr>"
        f"<td class='left'>{escape(family)}</td>"
        f"<td class='left'>{escape(seen)}</td>"
        f"<td class='left'>{escape(unseen)}</td>"
        f"<td class='left'>{escape(question)}</td>"
        "</tr>"
        for family, seen, unseen, question in GENERALIZATION_PAIRS
    )


def future_coverage_section() -> str:
    rows = []
    for item in CATALOG:
        rows.append(
            f"<tr><td class='left'><code>{item['id']}</code></td><td class='left'>{escape(item['title'])}</td>"
            f"<td>Seen/Unseen</td><td class='left'>{escape(V14_COVERAGE[item['name']])}</td></tr>"
        )
    for identifier, name, title, codepoint, coverage in FUTURE_CATALOG:
        rows.append(
            f"<tr><td class='left'><code>{identifier}</code></td><td class='left'>{escape(title)} <code>{codepoint}</code></td>"
            f"<td>Future</td><td class='left'>{escape(coverage)}</td></tr>"
        )
    return """
<h2>5. Future 7종과 v1.4 규칙의 직접 커버 범위</h2>
<p class='lede'>현재 실험 대시보드의 ‘학습 미포함 입력 교란 평가’는 <strong>Future 7종만</strong> 사용한다. v1.4가 clean 정답을 생성했더라도 이 7종을 정규화하도록 추가된 규칙은 없다. 따라서 noisy에서 raw 규칙 탐지율이 떨어질 수 있다.</p>
<div class='notice warn'><strong>중요:</strong> v1.3/v1.4의 실제 패치와 실험 교란을 구분해야 한다. 명시적으로 겹치는 것은 C1 <code>U+0092</code>와 <code>25 mg → 25mg</code>뿐이며, 둘 다 Seen 학습 교란이다. Future 7종은 전부 학습·검증·threshold 선택에 없고 전용 v1.4 패치도 없다.</div>
<div class='tablewrap'><table><thead><tr><th class='left'>ID</th><th class='left'>교란</th><th>실험 그룹</th><th class='left'>v1.4 직접 커버 여부</th></tr></thead><tbody>""" + "".join(rows) + "</tbody></table></div>"


def build_html() -> str:
    validate_catalog()
    examples = load_examples()
    missing = {item["name"] for item in CATALOG} - examples.keys()
    if missing:
        raise ValueError(f"Missing stored examples for: {sorted(missing)}")
    strict = load_strict_counts()
    train_counts = strict["train_counts"]
    all_test_counts = strict["test_counts"]
    unseen_counts = {
        item["name"]: all_test_counts[item["name"]]
        for item in CATALOG
        if item["group"] == "unseen"
    }
    final_unseen = sum(unseen_counts.values())
    cards = catalog_cards(examples, train_counts, unseen_counts)
    rows = count_rows(train_counts, unseen_counts)
    mapping_rows = generalization_rows()
    c1_actual = examples["c1_apostrophe"]["edit"]["new"]
    if c1_actual != "\x92":
        raise ValueError(f"latest C1 example is not one U+0092: {c1_actual!r}")
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Token Redaction Probe · 오염 규칙 카탈로그</title>
<style>
:root{{--bg:#f4f6f8;--panel:#fff;--ink:#172126;--muted:#63717a;--line:#dce3e7;--line2:#edf1f3;--teal:#087f70;--tealbg:#e4f5f1;--amber:#9a5b08;--amberbg:#fff5e5;--blue:#486581;--bluebg:#eaf0f5;--red:#b43b33;--redbg:#fbe9e7;--mono:ui-monospace,SFMono-Regular,Consolas,monospace;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif}}
:root[data-theme=dark]{{--bg:#11171b;--panel:#192126;--ink:#edf2f4;--muted:#a5b0b7;--line:#303a40;--line2:#253036;--teal:#52cfbb;--tealbg:#173b35;--amber:#f4bd6b;--amberbg:#3b2d18;--blue:#b1c9dd;--bluebg:#23313d;--red:#f08a80;--redbg:#3c211f}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.55}}.wrap{{max-width:1180px;margin:auto;padding:42px 22px 80px}}.hero{{display:flex;justify-content:space-between;gap:24px}}.eyebrow{{color:var(--teal);font:700 12px var(--mono);letter-spacing:.13em}}h1{{font-size:clamp(29px,4vw,44px);line-height:1.15;margin:8px 0 12px;letter-spacing:-.04em}}.hero p,.lede{{color:var(--muted);max-width:820px}}.actions{{display:flex;gap:8px;align-items:flex-start;flex-wrap:wrap}}.button,button{{border:1px solid var(--line);color:var(--ink);background:var(--panel);border-radius:9px;padding:8px 11px;text-decoration:none;cursor:pointer;font:650 12px var(--sans)}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0}}.card,.notice,.rule,.flow-step{{background:var(--panel);border:1px solid var(--line);border-radius:13px}}.card{{padding:16px}}.card b{{display:block;font:750 26px var(--mono)}}.card span{{color:var(--muted);font-size:12px}}h2{{font-size:22px;margin:38px 0 7px}}.lede{{font-size:13px;margin:0 0 16px}}.notice{{border-left:4px solid var(--teal);padding:14px 16px;color:var(--muted);font-size:13px;margin:16px 0}}.notice strong{{color:var(--ink)}}.warn{{border-left-color:var(--amber);background:var(--amberbg)}}.flow{{display:grid;grid-template-columns:repeat(5,1fr);gap:9px}}.flow-step{{padding:14px;position:relative}}.flow-step b{{display:block;color:var(--teal);font-size:12px;margin-bottom:5px}}.flow-step span{{font-size:12px;color:var(--muted)}}.catalog{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}.rule{{padding:17px}}.rule-head{{display:flex;justify-content:space-between;gap:12px;align-items:flex-start}}.rule-head h3{{margin:2px 0 0;font-size:18px}}.rule-id{{font:750 11px var(--mono);color:var(--teal)}}.badge{{padding:3px 8px;border-radius:999px;font-size:10px;font-weight:750;white-space:nowrap}}.seen{{background:var(--tealbg);color:var(--teal)}}.unseen{{background:var(--bluebg);color:var(--blue)}}dl{{margin:13px 0;border-top:1px solid var(--line2)}}dl div{{display:grid;grid-template-columns:80px 1fr;gap:10px;padding:7px 0;border-bottom:1px solid var(--line2)}}dt{{font-size:11px;color:var(--muted)}}dd{{margin:0;font-size:12px}}code{{font-family:var(--mono)}}.why{{font-size:12px;color:var(--muted)}}.example{{background:var(--bg);border-radius:9px;padding:10px;overflow:auto}}.example div{{display:grid;grid-template-columns:50px 1fr;gap:8px;margin:4px 0;min-width:520px}}.example b{{font-size:10px;color:var(--muted)}}.example code{{font-size:11px;white-space:nowrap}}mark{{background:#ffe08a;color:#172126;border-radius:3px;padding:1px 2px}}.counts{{margin-top:10px;color:var(--muted);font-size:10px}}.tablewrap{{overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:12px}}table{{width:100%;border-collapse:collapse;white-space:nowrap;font-size:12px}}th,td{{padding:9px;border-bottom:1px solid var(--line2);text-align:right}}th{{color:var(--muted);font-size:11px}}.left{{text-align:left}}.limits{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}.limit{{background:var(--panel);border:1px solid var(--line);border-radius:11px;padding:14px;font-size:12px;color:var(--muted)}}.limit b{{display:block;color:var(--ink);margin-bottom:5px}}footer{{margin-top:38px;border-top:1px solid var(--line);padding-top:14px;color:var(--muted);font-size:11px}}@media(max-width:820px){{.hero{{display:block}}.actions{{margin-top:14px}}.cards,.catalog,.limits{{grid-template-columns:1fr 1fr}}.flow{{grid-template-columns:1fr}}}}@media(max-width:560px){{.wrap{{padding:24px 12px}}.cards,.catalog,.limits{{grid-template-columns:1fr}}}}
</style></head><body><main class="wrap">
<header class="hero"><div><div class="eyebrow">TOKEN REDACTION PROBE · PERTURBATION CATALOG V1</div><h1>입력 오염 규칙 12종</h1><p>최신 v1.4 clean pseudo-label을 유지한 채 공백·Unicode·용량·구두점 경계만 바꾸어, 규칙과 학습형 redactor가 같은 민감 span을 계속 탐지하는지 비교한 통제 실험 명세입니다.</p></div><div class="actions"><a class="button" href="../">전체 결과</a><a class="button" href="https://github.com/Terry-joop/token_redaction_probe/blob/main/src/robustness/build_pairs.py">생성 코드</a><button id="theme">다크 모드</button></div></header>
<section class="cards"><div class="card"><b>12</b><span>전체 교란</span></div><div class="card"><b>5</b><span>Seen · 학습 증강</span></div><div class="card"><b>7</b><span>Unseen · 최종 평가</span></div><div class="card"><b>{final_unseen:,}</b><span>최종 unseen target-pair</span></div></section>
<div class="notice"><strong>Seen과 Unseen:</strong> 동일한 분리를 10개 데이터셋 모두에 적용했습니다. Seen 5종만 clean train에 증강하고, Unseen 7종은 학습에서 제외한 채 각 데이터셋의 전체 test에서 적용 가능한 target을 모두 평가했습니다.</div>
<h2>1. 문장과 정답을 만드는 순서</h2><p class="lede">오염된 문장에서 규칙을 다시 실행해 정답을 만들지 않습니다. 그래야 규칙이 오염 때문에 놓친 민감정보가 정답에서도 사라지는 오류를 막을 수 있습니다.</p>
<div class="flow"><div class="flow-step"><b>1 · 원문</b><span>원래 문장을 준비</span></div><div class="flow-step"><b>2 · 최신 clean 정답</b><span>medterm5/piiclean2 v1.4로 민감 span 결정</span></div><div class="flow-step"><b>3 · 한 종류 편집</b><span>적용 가능한 민감 span에 교란 하나 삽입</span></div><div class="flow-step"><b>4 · 정답 이동</b><span>문자 길이 변화만큼 clean span 좌표 이동</span></div><div class="flow-step"><b>5 · Paired 비교</b><span>같은 target을 규칙과 Student가 가리는지 평가</span></div></div>
<h2>2. 학습에서 본 교란과 test에서 처음 본 교란</h2><p class="lede">Test 교란은 Seen 문장을 다시 사용한 것이 아닙니다. 네 계열은 학습 교란과 원리는 같지만 표면 문자가 다른 변형이고, zero-width는 학습에 직접 대응하는 예가 없는 완전히 새로운 계열입니다.</p>
<div class="tablewrap"><table><thead><tr><th class="left">일반화 계열</th><th class="left">Train · Seen</th><th class="left">Test · Unseen</th><th class="left">확인하려는 질문</th></tr></thead><tbody>{mapping_rows}</tbody></table></div>
<div class="notice"><strong>최종 test 구성:</strong> strict 최종 표는 10개 데이터셋 모두에서 Unseen 7종만 사용했고 유형당 상한을 두지 않았습니다. 교란의 적용 조건을 만족하지 않는 문장에는 억지로 변형을 넣지 않으므로 데이터셋별 실제 pair 수와 나타난 교란 종류는 다릅니다. 앞선 10데이터셋 × 3모델 탐색 표는 12종 전체를 종류별 최대 100쌍만 사용한 별도 pilot입니다.</div>
<h2>3. 교란 규칙 전체 목록</h2><p class="lede">예문은 실제 Drug Reviews pair에서 가져왔습니다. 가운데점(·)은 일반 공백이고 꺾쇠 표시는 화면에 보이지 않는 Unicode 문자입니다. 노란 영역은 clean 규칙이 민감하다고 정한 target span입니다.</p>
<div class="catalog">{cards}</div>
<h2>4. 실제 사용 개수</h2><p class="lede">10개 데이터셋의 clean train {strict['clean_train_rows']:,}행에 Seen {strict['augmented_train_rows']:,}행을 더했습니다. 최종 표면 일반화 평가는 각 전체 test에서 생성 가능한 Unseen {strict['unseen_pairs']:,}쌍을 모두 사용했습니다.</p>
<div class="tablewrap"><table><thead><tr><th class="left">ID</th><th class="left">교란</th><th>구분</th><th>학습 증강</th><th>최종 unseen test</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>5. 해석 범위와 재현성 주의</h2><div class="limits"><div class="limit"><b>Pseudo-gold</b>정답은 사람이 검수한 개인정보 gold가 아니라 clean v1.4 규칙의 span을 이동한 정답입니다.</div><div class="limit"><b>통제된 단일 교란</b>한 pair에는 한 종류만 넣었습니다. 여러 오류가 동시에 섞인 실제 사용자 입력 전체를 대표하지 않습니다.</div><div class="limit"><b>표면 강건성</b>문장의 의미를 바꾸지 않는 문자·경계 변화만 시험합니다. 문맥적 민감성이나 새로운 개인정보 유형 평가는 아닙니다.</div><div class="limit"><b>원문 단위 통계</b>한 원문에서 여러 pair가 나오므로 신뢰구간은 같은 source_id의 변형을 묶은 cluster bootstrap으로 계산합니다.</div></div>
<div class="notice"><strong>최신 C1 artifact:</strong> strict 데이터와 이 문서의 AP-02 예시는 현재 생성 코드와 동일하게 <code>U+0092</code> 한 문자만 삽입합니다. 과거 두 문자 artifact는 최종 10개 데이터셋 결과에 사용하지 않았습니다.</div>
<footer>생성: <code>src/build_perturbation_catalog.py</code> · 실행 규칙: <code>src/robustness/build_pairs.py</code> · 예문: 최신 Drug Reviews strict pair · 집계: strict 10개 데이터셋 · 2026-08-03</footer>
</main><script>const b=document.getElementById('theme');b.onclick=()=>{{const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';b.textContent=dark?'라이트 모드':'다크 모드'}};</script></body></html>"""


def build_markdown() -> str:
    validate_catalog()
    strict = load_strict_counts()
    train_counts = strict["train_counts"]
    test_counts = strict["test_counts"]
    lines = [
        "# 입력 오염 규칙 카탈로그 v1",
        "",
        "최신 v1.4 clean pseudo-label을 결정한 뒤 표면 교란을 적용하고, clean 문자 span을 편집에 맞춰 이동한다.",
        "",
        "| ID | 계열 | 구분 | 교란 | 적용 조건 | 변환 | 코드포인트 | 학습 행 | 최종 unseen pair |",
        "|---|---|---|---|---|---|---|---:|---:|",
    ]
    for item in CATALOG:
        unseen = test_counts[item["name"]] if item["group"] == "unseen" else 0
        lines.append(
            f"| {item['id']} | {item['family']} | {item['group']} | {item['title']} | "
            f"{item['condition']} | {item['change']} | `{item['codepoint']}` | "
            f"{train_counts.get(item['name'], 0):,} | {unseen:,} |"
        )
    lines.extend(
        [
            "",
            "## 학습 교란과 최종 test 교란의 대응",
            "",
            "| 일반화 계열 | Train · Seen | Test · Unseen | 확인하려는 질문 |",
            "|---|---|---|---|",
        ]
    )
    for family, seen, unseen, question in GENERALIZATION_PAIRS:
        lines.append(f"| {family} | {seen} | {unseen} | {question} |")
    lines.extend(
        [
            "",
            f"strict 최종 평가는 10개 데이터셋 모두에서 Unseen 7종만 사용해 적용 가능한 {strict['unseen_pairs']:,}쌍을 유형당 상한 없이 평가했다. 앞선 10데이터셋 × 3모델 탐색 표는 12종 전체를 종류별 최대 100쌍 사용한 별도 pilot이다.",
            "",
            "## 핵심 원칙",
            "",
            "- 한 clean 문장에서 최신 medterm5/piiclean2 v1.4 민감 span을 먼저 정한다.",
            "- 한 pair에는 한 종류의 교란만 적용한다.",
            "- 오염 문장에서 규칙을 다시 실행해 gold를 만들지 않고 clean span을 결정적으로 이동한다.",
            "- Seen 5종은 학습 증강, Unseen 7종은 최종 일반화 평가에만 사용한다.",
            "- 이 평가는 human-gold 개인정보 정확도가 아니라 규칙 기반 pseudo-gold 표면 강건성이다.",
            "",
            "## 최신 artifact 확인",
            "",
            "strict 10개 데이터셋과 문서 예시는 최신 코드로 다시 생성했으며 C1 교란은 U+0092 한 문자만 삽입한다. 과거 두 문자 artifact는 최종 결과에 사용하지 않았다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = build_html()
    html = html.replace("입력 오염 규칙 12종", "입력 교란 19종")
    html = html.replace("<b>12</b><span>전체 교란</span>", "<b>19</b><span>등록 교란 · Seen 5 / Unseen 7 / Future 7</span>")
    html = html.replace("<h2>5. 해석 범위와 재현성 주의</h2>", future_coverage_section() + "<h2>6. 해석 범위와 재현성 주의</h2>")
    html = "\n".join(line.rstrip() for line in html.splitlines()) + "\n"
    markdown = "\n".join(line.rstrip() for line in build_markdown().splitlines()) + "\n"
    OUT.write_text(html, encoding="utf-8")
    MARKDOWN_OUT.write_text(markdown, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} and {MARKDOWN_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
