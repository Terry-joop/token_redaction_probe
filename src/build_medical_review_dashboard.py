"""Build a filterable web review dashboard for rule-vs-Student examples."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "reports" / "medical_rule_student_examples.jsonl"
HUMAN_REVIEW = ROOT / "data" / "human_review" / "medical_rule_review_v1.jsonl"
OUT = ROOT / "reports" / "medical_review_dashboard.html"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    examples = read_jsonl(EXAMPLES)
    human = read_jsonl(HUMAN_REVIEW)
    counts = Counter(row["dataset"] for row in examples)
    outcomes = Counter(row["outcome"] for row in examples)
    reviewed = sum(bool(row.get("human_reviewed")) for row in human)
    data = json.dumps(examples, ensure_ascii=False).replace("</", "<\\/")
    dataset_options = "".join(
        f"<option value='{key}'>{key} · {value}개 사례</option>" for key, value in sorted(counts.items())
    )
    page = """<!doctype html><html lang='ko'><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>의료 Redaction 검수 대시보드</title>
<style>
:root{{--bg:#f4f7fb;--panel:#fff;--text:#12243b;--muted:#617287;--line:#dfe7f0;--accent:#147d70;--rule:#e65252;--student:#16856e;--warning:#b47200}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.55 system-ui,-apple-system,sans-serif}} main{{max-width:1420px;margin:0 auto;padding:35px 24px 70px}} h1{{font-size:32px;margin:0 0 5px}} h2{{font-size:17px;margin:0}} .sub{{color:var(--muted);margin:0 0 22px}}
.notice,.toolbar,.card,.stat{{background:var(--panel);border:1px solid var(--line);border-radius:12px}}.notice{{border-left:4px solid var(--accent);padding:15px 18px;margin:15px 0 22px;color:#31465c}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:18px}}.stat{{padding:14px 17px}}.stat b{{display:block;font-size:25px;color:var(--accent)}}.stat span{{color:var(--muted)}}
.toolbar{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:12px;margin-bottom:18px;position:sticky;top:8px;z-index:2;box-shadow:0 4px 15px #31465c0b}}button,select,input{{font:inherit;border:1px solid var(--line);border-radius:8px;padding:9px 12px;background:#fff;color:var(--text)}}button{{cursor:pointer}}button.active{{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:700}}input{{min-width:230px;flex:1}}#count{{margin-left:auto;color:var(--muted)}}
#cards{{display:grid;gap:13px}}.card{{padding:18px}.card-head{{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:11px}}.id{{font-family:ui-monospace,monospace;color:var(--muted)}}.badge{{font-weight:700;border-radius:99px;padding:3px 9px;font-size:12px}}.missed,.missed\+overmask{{color:#b42318;background:#ffe9e7}}.overmask{{color:#955d00;background:#fff4d8}}.agreement{{color:#126b5c;background:#e3f7ee}}.truncated{{color:#566575;background:#ebeff3}}.lists{{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:9px 0 12px}}.list{{border-radius:8px;padding:9px 11px;background:#f8fafc;color:#46596d;min-height:47px}}.list b{{display:block;color:var(--text);font-size:12px;margin-bottom:3px}}.tokens{{line-height:2.35;color:#26394d}}.token{{padding:2px 3px;border-radius:3px;white-space:nowrap}}.rule{{background:#ffe5e3;border-bottom:2px solid var(--rule)}}.student{{box-shadow:inset 0 -2px var(--student)}}.rule.student{{background:#dff6eb}}.prob{{font:10px ui-monospace,monospace;color:#6c7d90;margin-left:2px}}.help{{color:var(--muted);font-size:12px;margin-top:9px}}@media(max-width:760px){{main{{padding:20px 12px}}h1{{font-size:25px}}.stats{{grid-template-columns:repeat(2,1fr)}}.lists{{grid-template-columns:1fr}}.toolbar{{position:static}}#count{{margin-left:0}}}}
</style>
<main>
  <h1>의료 Redaction 검수 대시보드</h1>
  <p class='sub'>현재 RedactFormer 의료 규칙 teacher와 기존 ELECTRA-small Student가 실제 test 문장에서 무엇을 가리는지 확인하는 사례집</p>
  <div class='notice'><b>검수 순서</b> — 여기서는 사례를 보고 판단합니다. 실제 수정은 로컬의 <code>data/human_review/medical_rule_review_v1.jsonl</code>에서 <code>human_labels</code>를 고친 뒤 <code>human_reviewed: true</code>로 표시하면 됩니다. 이 페이지의 규칙 라벨은 GPT teacher 라벨이 아니라 현재 medterm 규칙입니다.</div>
  <section class='stats'><div class='stat'><b>{len(examples)}</b><span>표시 사례</span></div><div class='stat'><b>{len(counts)}</b><span>의료 데이터셋</span></div><div class='stat'><b>{outcomes.get('missed',0)+outcomes.get('missed+overmask',0)}</b><span>Student가 하나 이상 놓친 사례</span></div><div class='stat'><b>{reviewed}/{len(human)}</b><span>규칙 human 검수 완료</span></div></section>
  <section class='toolbar'><button class='filter active' data-outcome='all'>전체</button><button class='filter' data-outcome='missed'>놓침</button><button class='filter' data-outcome='overmask'>과잉 가림</button><button class='filter' data-outcome='agreement'>일치</button><select id='dataset'><option value='all'>모든 데이터셋</option>{dataset_options}</select><input id='query' placeholder='문장, 단어, id 검색'><span id='count'></span></section>
  <section id='cards'></section>
</main>
<script>const DATA={data}; let outcome='all'; const el=id=>document.getElementById(id); const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function toks(r){{return r.words.map((w,i)=>{{let c='token';if(r.teacher_labels[i])c+=' rule';if(r.student_labels[i]===1)c+=' student';let p=r.student_scores[i];return `<span class="${{c}}">${{esc(w)}}${{p==null?'':`<span class="prob">${{p.toFixed(2)}}</span>`}}</span>`}}).join(' ')}}
function render(){{let q=el('query').value.toLowerCase(),d=el('dataset').value;let rows=DATA.filter(r=>(outcome==='all'||r.outcome===outcome||(outcome==='missed'&&r.outcome==='missed+overmask'))&&(d==='all'||r.dataset===d)&&(!q||JSON.stringify(r).toLowerCase().includes(q)));el('count').textContent=rows.length+'개 사례';el('cards').innerHTML=rows.map(r=>`<article class="card"><div class="card-head"><h2>${{esc(r.dataset)}} · <span class="id">${{esc(r.id)}}</span></h2><span class="badge ${{r.outcome}}">${{r.outcome}}</span></div><div class="lists"><div class="list"><b>규칙 teacher가 가린 단어</b>${{esc(r.teacher_selected.join(', ')||'없음')}}</div><div class="list"><b>ELECTRA-small이 가린 단어</b>${{esc(r.student_selected.join(', ')||'없음')}}</div></div><div class="tokens">${{toks(r)}}</div><div class="help">빨강=규칙만, 초록 밑줄=Student만, 초록 배경=둘 다 가림 · 회색 숫자=Student redaction 확률</div></article>`).join('')}}
document.querySelectorAll('.filter').forEach(b=>b.onclick=()=>{{document.querySelectorAll('.filter').forEach(x=>x.classList.remove('active'));b.classList.add('active');outcome=b.dataset.outcome;render()}});el('dataset').onchange=render;el('query').oninput=render;render();</script></html>"""
    replacements = {
        "{len(examples)}": str(len(examples)),
        "{len(counts)}": str(len(counts)),
        "{outcomes.get('missed',0)+outcomes.get('missed+overmask',0)}": str(
            outcomes.get("missed", 0) + outcomes.get("missed+overmask", 0)
        ),
        "{reviewed}": str(reviewed),
        "{len(human)}": str(len(human)),
        "{dataset_options}": dataset_options,
        "{data}": data,
    }
    for source, target in replacements.items():
        page = page.replace(source, target)
    OUT.write_text(page.replace("{{", "{").replace("}}", "}"), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
