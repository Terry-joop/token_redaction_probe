# Build a self-contained HTML dashboard and Excel-friendly CSV.
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
MODELS = {'bert_tiny':'BERT-tiny','electra_small':'ELECTRA-small','distilroberta':'DistilRoBERTa'}
GROUPS = {
 'medical':('의료 규칙',['drug','symptom2dx','adr','redditmh','mednli','mentalhealth']),
 'pii':('실제 PII',['bios','mrpc']),
 'entity':('비개인 엔티티 대조',['qnli','finphrasebank']),
}
META = {
 'drug':('Drug Reviews','medterm4-v2','약물 리뷰'),
 'symptom2dx':('Symptom2Dx','medterm4-v2','증상→진단'),
 'adr':('ADR','medterm4-v2','약물 부작용'),
 'redditmh':('RedditMH','medterm4-v2','정신건강 서술'),
 'mednli':('MedNLI','medterm4-v2','의료 문장쌍 NLI'),
 'mentalhealth':('Mental Health','medterm4-v2','정신상태 분류'),
 'bios':('BIOS','piiclean-v1','약력·직업'),
 'mrpc':('MRPC','piiclean-strict-v1','패러프레이즈'),
 'qnli':('QNLI','entityclean-v1','질문·문장 매칭'),
 'finphrasebank':('FinPhraseBank','entityclean-v1','금융 감성'),
}
FIELDS = ['threshold','precision','recall','f1','f2','gold_mask_rate','predicted_mask_rate','residual_sensitive_rate','evaluated_tokens']

def load(path): return json.loads((ROOT/path).read_text(encoding='utf-8'))
def metric(x): return {k:x.get(k) for k in FIELDS}

def rows():
 full_path=ROOT/'reports/full_dataset_results.json'
 if full_path.exists():
  full=json.loads(full_path.read_text(encoding='utf-8'))
  out=[]
  for group,(group_name,datasets) in GROUPS.items():
   for dataset in datasets:
    if dataset not in full['datasets']: continue
    entry=full['datasets'][dataset]; stats=entry['stats']; title,policy,task=META[dataset]
    for model,model_name in MODELS.items():
     if model not in entry['models']: continue
     src=entry['models'][model]
     out.append({'group':group,'group_name':group_name,'dataset':dataset,'dataset_name':title,'task':task,'policy':policy,'model':model,'model_name':model_name,'seed':42,'scope':'full','result_source':entry['result_source'],'examples':stats['all']['examples'],'train_examples':stats['train']['examples'],'validation_examples':stats['validation']['examples'],'test_examples':stats['test']['examples'],'budget':metric(src['budget_matched']['test']),'privacy':metric(src['f2_optimized']['test'])})
  return out
 medical=load(Path('artifacts/medical_redactor/core_matrix/six_dataset_seed42_summary.json'))['datasets']
 other=load(Path('artifacts/nonmedical_redactor/seed42/summary.json'))
 out=[]
 for group,(group_name,datasets) in GROUPS.items():
  for dataset in datasets:
   title,policy,task=META[dataset]
   for model,model_name in MODELS.items():
    src=medical[dataset][model] if group=='medical' else other[f'{dataset}:{model}']
    out.append({'group':group,'group_name':group_name,'dataset':dataset,'dataset_name':title,'task':task,'policy':policy,'model':model,'model_name':model_name,'seed':42,'scope':'pilot','result_source':'pilot','examples':None,'train_examples':None,'validation_examples':None,'test_examples':None,'budget':metric(src['budget_matched']['test']),'privacy':metric(src['f2_optimized']['test'])})
 return out

def macros(data):
 out=[]
 fields=['precision','recall','f1','f2','gold_mask_rate','predicted_mask_rate','residual_sensitive_rate']
 for group,(name,datasets) in GROUPS.items():
  for model,model_name in MODELS.items():
   chosen=[x for x in data if x['group']==group and x['model']==model]
   item={'group':group,'group_name':name,'datasets':len(datasets),'model_name':model_name}
   for mode in ['budget','privacy']:
    item[mode]={f:statistics.fmean(x[mode][f] for x in chosen) for f in fields}
   out.append(item)
 return out

def efficiency():
 source=load(Path('artifacts/medical_redactor/core_matrix/complete_rule_student_comparison.json'))['efficiency']
 names={'medterm4':'medterm4 규칙'}|MODELS
 out=[]
 for key in ['medterm4',*MODELS]:
  x=source[key]; lat=x['latency']
  out.append({'name':names[key],'implementation':x['implementation'],'parameters':x.get('parameters'),'model_state_mb':x.get('model_state_mb'),'load_seconds':x['load_seconds'],'median_ms':lat['median_ms'],'p95_ms':lat['p95_ms'],'sentences_per_second':lat['sentences_per_second'],'peak_rss_mb':x['peak_rss_mb']})
 return out

def exploratory():
 seeded=load(Path('artifacts/medical_redactor/core_matrix/three_seed_summary.json'))
 lodo=load(Path('artifacts/medical_redactor/cross_dataset_lodo/summary.json'))
 out=[]
 for model in ['electra_small','distilroberta']:
  x=seeded['models'][model]
  out.append({'experiment':'3-seed 반복','scope':'의료 4개 in-domain','model':MODELS[model],'f1':x['macro_budget_f1']['mean'],'std':x['macro_budget_f1']['sample_std'],'f2':x['macro_privacy_f2']['mean'],'note':'seed 42·43·44'})
  md=load(Path(f'artifacts/medical_redactor/cross_dataset_multidomain/{model}/per_dataset_evaluation.json'))['datasets']
  out.append({'experiment':'Multi-domain','scope':'의료 4개 공동학습','model':MODELS[model],'f1':statistics.fmean(v['zero_shot_fixed']['budget_matched_source_threshold']['f1'] for v in md.values()),'std':None,'f2':statistics.fmean(v['zero_shot_fixed']['f2_optimized_source_threshold']['f2'] for v in md.values()),'note':'전역 threshold'})
  out.append({'experiment':'LODO','scope':'의료 4개 중 target 제외','model':MODELS[model],'f1':lodo['models'][model]['macro_budget_f1'],'std':None,'f2':lodo['models'][model]['macro_privacy_f2'],'note':'unseen-domain 탐색'})
 history = [
  ('Frozen baseline','Drug Reviews 200','BERT-tiny (MLP만 학습)','medterm4_student_frozen_pilot200'),
  ('Encoder fine-tuning','Drug Reviews 200','BERT-tiny','medterm4_student_finetuned_pilot200'),
  ('데이터 규모 확대','Drug Reviews 1,000','BERT-tiny','medterm4_student_finetuned_1000'),
  ('1,000개 모델 비교','Drug Reviews 1,000','ELECTRA-small','medterm4_electra_small_finetuned_1000'),
  ('1,000개 모델 비교','Drug Reviews 1,000','DistilRoBERTa','medterm4_distilroberta_finetuned_1000'),
 ]
 for experiment,scope,model_name,directory in history:
  value=load(Path('artifacts/medical_redactor')/directory/'medical_evaluation.json')
  out.append({'experiment':experiment,'scope':scope,'model':model_name,'f1':value['budget_matched']['test']['f1'],'std':None,'f2':value['f2_optimized']['test']['f2'],'note':'초기 단일 데이터셋 실험'})
 return out

def sst2():
 source=load(Path('artifacts/leakage/summary.json'))['runs']; out=[]
 for key in ['frozen_mask','frozen_delete','finetuned_mask','finetuned_delete']:
  x=source[key]; a=x['attacker_accuracy']
  out.append({'condition':key,'rate':x['redaction_rate'],'original':a['original'],'teacher':a['teacher'],'student':a['student'],'random':a['random_matched_mean']})
 return out

def write_csv(data,path):
 fields=['scope','group','dataset','task','teacher_policy','model','seed','all_examples','train_examples','validation_examples','test_examples','result_source','mode','threshold','precision','recall','f1','f2','teacher_mask_rate','student_mask_rate','residual_sensitive_rate','evaluated_tokens']
 with path.open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); w.writeheader()
  for row in data:
   for mode,label in [('budget','budget_matched'),('privacy','f2_optimized')]:
    x=row[mode]
    w.writerow({'scope':row['scope'],'group':row['group_name'],'dataset':row['dataset_name'],'task':row['task'],'teacher_policy':row['policy'],'model':row['model_name'],'seed':row['seed'],'all_examples':row['examples'],'train_examples':row['train_examples'],'validation_examples':row['validation_examples'],'test_examples':row['test_examples'],'result_source':row['result_source'],'mode':label,'threshold':x['threshold'],'precision':x['precision'],'recall':x['recall'],'f1':x['f1'],'f2':x['f2'],'teacher_mask_rate':x['gold_mask_rate'],'student_mask_rate':x['predicted_mask_rate'],'residual_sensitive_rate':x['residual_sensitive_rate'],'evaluated_tokens':x['evaluated_tokens']})

def split_table(data):
 unique={row['dataset']:row for row in data}
 body=''.join(
  f"<tr><td class='left meta dataset'>{row['dataset_name']}</td><td>{row['examples']:,}</td><td>{row['train_examples']:,}</td><td>{row['validation_examples']:,}</td><td>{row['test_examples']:,}</td></tr>"
  for row in unique.values()
 )
 return f"<div class='tablewrap solo'><table><thead><tr><th class='left'>데이터셋</th><th>전체</th><th>Train</th><th>Validation</th><th>Test</th></tr></thead><tbody>{body}</tbody></table></div>"

HTML = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Token Redaction Probe · 전체 결과</title><style>
:root{--bg:#f4f6f8;--panel:#fff;--ink:#162027;--muted:#62707a;--faint:#8d99a2;--line:#dde3e7;--line2:#edf0f2;--teal:#087f70;--tealbg:#e4f5f1;--blue:#486581;--bluebg:#eaf0f5;--amber:#9a5b08;--red:#b43b33;--redbg:#fbe9e7;--mono:ui-monospace,Consolas,monospace;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif}:root[data-theme=dark]{--bg:#11171b;--panel:#192126;--ink:#edf2f4;--muted:#a5b0b7;--faint:#78858d;--line:#303a40;--line2:#253036;--teal:#52cfbb;--tealbg:#173b35;--blue:#b1c9dd;--bluebg:#23313d;--amber:#f4bd6b;--red:#f08a80;--redbg:#3c211f}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5}.wrap{max-width:1240px;margin:auto;padding:42px 22px 80px}.hero{display:flex;justify-content:space-between;gap:20px}.eyebrow{color:var(--teal);font:700 12px var(--mono);letter-spacing:.13em}.hero h1{font-size:clamp(27px,4vw,42px);line-height:1.15;margin:8px 0 10px;letter-spacing:-.035em}.hero p{max-width:820px;color:var(--muted);margin:0}.actions{display:flex;gap:8px;align-items:flex-start}.button,button{border:1px solid var(--line);color:var(--ink);background:var(--panel);border-radius:9px;padding:8px 11px;text-decoration:none;cursor:pointer;font:650 12px var(--sans)}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:16px}.card b{font:750 25px var(--mono);display:block}.card span,.lede{font-size:12px;color:var(--muted)}.notice{border:1px solid var(--line);border-left:4px solid var(--teal);background:var(--panel);border-radius:10px;padding:13px 15px;color:var(--muted);font-size:13px;margin:16px 0}.notice strong{color:var(--ink)}.warn{border-left-color:var(--amber)}h2{font-size:21px;margin:34px 0 6px}.lede{margin:0 0 14px}.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);padding:10px;border-radius:12px 12px 0 0}.toolbar select,.toolbar input{border:1px solid var(--line);color:var(--ink);background:var(--bg);border-radius:8px;padding:7px 9px;font:12px var(--sans)}.seg{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}.seg button{border:0;border-radius:0}.seg .active{background:var(--teal);color:#fff}.tablewrap{overflow:auto;background:var(--panel);border:1px solid var(--line);border-top:0;border-radius:0 0 12px 12px}.solo{border-top:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}th{position:sticky;top:0;background:var(--panel);color:var(--muted);font-size:11px;text-align:right;padding:9px 8px;border-bottom:1px solid var(--line)}th.left,td.left{text-align:left}td{padding:7px 8px;border-bottom:1px solid var(--line2);text-align:right;font:500 12px var(--mono)}td.meta{font-family:var(--sans)}tr.start td{border-top:2px solid var(--line)}.pill,.policy,.seed{display:inline-block;border-radius:999px;padding:2px 7px;font:700 10px var(--sans)}.g-medical{background:var(--tealbg);color:var(--teal)}.g-pii{background:var(--redbg);color:var(--red)}.g-entity{background:var(--bluebg);color:var(--blue)}.policy{background:var(--bg);color:var(--muted)}.seed{padding:1px 5px;background:var(--tealbg);color:var(--teal)}.best{color:var(--teal);font-weight:800;background:color-mix(in srgb,var(--teal) 7%,transparent)}.low{color:var(--red)}.over{color:var(--amber)}.dataset{font-weight:750}.task{display:block;font-size:10px;color:var(--faint);margin-top:2px}td.merge{vertical-align:middle;text-align:center!important;border-right:1px solid var(--line);background:color-mix(in srgb,var(--panel) 94%,var(--teal) 6%)}td.merge.dataset-cell{text-align:left!important;min-width:140px}.analysis-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.analysis-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}.analysis-card h3{font-size:14px;margin:0 0 8px;color:var(--teal)}.analysis-card p{font-size:13px;color:var(--muted);margin:0}.analysis-card strong{color:var(--ink)}.macro-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.macro{border:1px solid var(--line);background:var(--panel);border-radius:12px;overflow:hidden}.macro h3{padding:13px;margin:0;border-bottom:1px solid var(--line);font-size:14px}.macro table{white-space:normal}.help-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.help{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;font-size:12px;color:var(--muted)}.help b{display:block;color:var(--ink);font:700 13px var(--mono)}details{background:var(--panel);border:1px solid var(--line);border-radius:11px;margin-top:12px}summary{cursor:pointer;padding:13px 15px;font-weight:700;font-size:13px}details>div{padding:0 15px 15px;color:var(--muted);font-size:12px}.foot{margin-top:36px;border-top:1px solid var(--line);padding-top:14px;color:var(--faint);font-size:11px}@media(max-width:850px){.hero{display:block}.actions{margin-top:14px}.cards,.macro-grid,.help-grid,.analysis-grid{grid-template-columns:1fr 1fr}}@media(max-width:520px){.cards,.macro-grid,.help-grid,.analysis-grid{grid-template-columns:1fr}.wrap{padding:24px 12px}}
</style></head><body><main class="wrap"><div class="hero"><div><div class="eyebrow">TOKEN REDACTION PROBE · 2026-07-29</div><h1>로컬 Student Redactor — 전체 데이터 실험</h1><p>각 데이터셋의 사용 가능한 행을 빈 문장·중복 제거 후 전부 사용해, 규칙 기반 pseudo-teacher를 작은 Transformer+MLP가 얼마나 모방하는지 정리했다. 의미가 다른 Teacher 정책은 그룹별로 분리했다.</p></div><div class="actions"><a class="button" href="redactor_results.csv">CSV · Excel용</a><button id="theme">다크 모드</button></div></div>
<section class="cards"><div class="card"><b>10</b><span>고정 조건 데이터셋</span></div><div class="card"><b>3</b><span>Student 아키텍처</span></div><div class="card"><b>30</b><span>seed 42 전체-data run</span></div><div class="card"><b>__EXAMPLE_COUNT__</b><span>전처리 후 전체 예시</span></div></section><div class="notice"><strong>주 지표:</strong> F1은 균형 일치, F2는 Recall을 더 중시한다. <strong>Token Accuracy는 클래스 불균형 때문에 메인 표에서 제외</strong>했다.</div><div class="notice warn"><strong>해석 제한:</strong> 모두 human-gold가 아닌 deterministic pseudo-gold다. QNLI·FinPhraseBank는 개인정보 탐지가 아닌 엔티티 대조 실험이며 세 그룹의 macro를 합치지 않는다.</div>
<h2>1. 고정 3모델 × 10데이터셋</h2><p class="lede">사용 가능한 전체 데이터를 학습하고, validation에서 threshold를 선택한 뒤 test에 한 번 적용한 seed 42 결과. 전체 및 Train/Val/Test 예시 수는 바로 아래 규모 표에 분리해 표시한다.</p><div class="toolbar"><div class="seg"><button class="mode active" data-mode="budget">동일 마스킹 예산</button><button class="mode" data-mode="privacy">Recall 중심 F2</button></div><select id="group"><option value="all">모든 그룹</option><option value="medical">의료 규칙</option><option value="pii">실제 PII</option><option value="entity">비개인 엔티티 대조</option></select><select id="model"><option value="all">모든 모델</option><option value="bert_tiny">BERT-tiny</option><option value="electra_small">ELECTRA-small</option><option value="distilroberta">DistilRoBERTa</option></select><input id="search" placeholder="데이터셋 검색"><span id="count" style="margin-left:auto;color:var(--faint);font-size:11px"></span></div><div class="tablewrap"><table><thead><tr><th class="left">그룹</th><th class="left">데이터셋</th><th class="left">Teacher</th><th class="left">Student</th><th>Seed</th><th>Th.</th><th>P</th><th>R</th><th>F1</th><th>F2</th><th>Teacher mask</th><th>Student mask</th><th>남은 민감</th><th>Tokens</th></tr></thead><tbody id="results"></tbody></table></div>
<h2>1-1. 전체 데이터 규모</h2><p class="lede">빈 문장·중복 제거 후 실제 사용한 예시 수. 공식 split이 있으면 보존했다.</p>__SPLIT_TABLE__
<h2>2. 그룹 내부 Macro</h2><p class="lede">의미가 같은 데이터셋끼리만 평균.</p><div class="macro-grid" id="macros"></div>
<h2>3. 효율 비교</h2><p class="lede">CPU 1-thread, batch 1, 128문장, 3회 반복.</p><div class="tablewrap solo"><table><thead><tr><th class="left">방식</th><th class="left">구현</th><th>Params</th><th>크기</th><th>Load</th><th>Median</th><th>p95</th><th>처리량</th><th>Peak RSS</th></tr></thead><tbody id="eff"></tbody></table></div><div class="notice"><strong>규칙의 F1=1은 자기 자신과 비교한 정의상 값</strong>이며 human-gold 정확도가 아니다.</div>
<h2>4. 탐색 실험</h2><p class="lede">메인 표와 분리된 안정성·일반화 분석.</p><div class="tablewrap solo"><table><thead><tr><th class="left">실험</th><th class="left">범위</th><th class="left">모델</th><th>F1 / Macro F1</th><th>Std.</th><th>F2 / Macro F2</th><th class="left">비고</th></tr></thead><tbody id="explore"></tbody></table></div>
<details><summary>SST-2 초기 leakage 예비 실험</summary><div><p>파이프라인 검증용이며 개인정보 성능 근거에는 포함하지 않는다. attacker 정확도는 낮을수록 task leakage가 적다.</p><div class="tablewrap solo"><table><thead><tr><th class="left">조건</th><th>Redaction</th><th>원문</th><th>Teacher</th><th>Student</th><th>Random</th></tr></thead><tbody id="sst"></tbody></table></div></div></details>
<h2>5. 결과 분석</h2><p class="lede">동일 마스킹 예산을 메인 기준으로 보고, Recall 중심 운용점과 효율·일반화 실험을 함께 해석했다.</p><div class="analysis-grid" id="analysis"></div><div class="notice warn"><strong>결론의 범위:</strong> 현재 결과는 Teacher 규칙을 Student가 재현하는 능력을 보여준다. 실제 개인정보를 잘 가리는지에 대한 최종 결론은 human-gold PII test와 RedactFormer 연결 후 RTM 복구 평가가 추가되어야 한다.</div><h2>6. 지표 읽는 법</h2><div class="help-grid"><div class="help"><b>Precision</b>Student 선택 중 Teacher와 일치한 비율.</div><div class="help"><b>Recall</b>Teacher 토큰 중 Student가 찾은 비율.</div><div class="help"><b>F1</b>Precision과 Recall의 균형.</div><div class="help"><b>F2</b>Recall을 더 중시한 지표.</div></div><details><summary>제외·보류 데이터셋</summary><div>biosx는 출처를 회수하지 못했고 MDCC는 원본 CSV가 없어 미실행했다. medterm4는 비의료 baseline으로 쓰지 않았다.</div></details><footer class="foot">생성: src/build_results_dashboard.py · 원본: artifacts 평가 JSON · 서버 없이 단독 실행.</footer></main>
<script>const D=__DATA__;let mode='budget';const $=id=>document.getElementById(id),pct=v=>v==null?'—':(v*100).toFixed(2)+'%',num=(v,n=3)=>v==null?'—':Number(v).toFixed(n),int=v=>v==null?'—':Number(v).toLocaleString('ko-KR');function chosen(){let g=$('group').value,m=$('model').value,q=$('search').value.toLowerCase();return D.rows.filter(r=>(g==='all'||r.group===g)&&(m==='all'||r.model===m)&&(!q||(r.dataset_name+' '+r.task+' '+r.policy).toLowerCase().includes(q)))}function render(){let rows=chosen(),best={},groupN={},dataN={};rows.forEach(r=>{best[r.dataset]=Math.max(best[r.dataset]??-1,r[mode].f1);groupN[r.group]=(groupN[r.group]||0)+1;dataN[r.dataset]=(dataN[r.dataset]||0)+1});let seenG=new Set(),seenD=new Set();$('results').innerHTML=rows.map(r=>{let x=r[mode],firstG=!seenG.has(r.group),firstD=!seenD.has(r.dataset);seenG.add(r.group);seenD.add(r.dataset);let groupCell=firstG?`<td rowspan="${groupN[r.group]}" class="meta merge"><span class="pill g-${r.group}">${r.group_name}</span></td>`:'',datasetCell=firstD?`<td rowspan="${dataN[r.dataset]}" class="meta merge dataset-cell"><b class="dataset">${r.dataset_name}</b><span class="task">${r.task}</span></td><td rowspan="${dataN[r.dataset]}" class="meta merge"><span class="policy">${r.policy}</span></td>`:'';return `<tr class="${firstD?'start':''}">${groupCell}${datasetCell}<td class="left meta">${r.model_name}</td><td><span class="seed">${r.seed}</span></td><td>${num(x.threshold,2)}</td><td>${num(x.precision)}</td><td class="${x.recall<.7?'low':''}">${num(x.recall)}</td><td class="${x.f1===best[r.dataset]?'best':''}">${num(x.f1)}</td><td>${num(x.f2)}</td><td>${pct(x.gold_mask_rate)}</td><td class="${x.predicted_mask_rate>x.gold_mask_rate*1.2?'over':''}">${pct(x.predicted_mask_rate)}</td><td>${pct(x.residual_sensitive_rate)}</td><td>${int(x.evaluated_tokens)}</td></tr>`}).join('');$('count').textContent=rows.length+'개 run';$('macros').innerHTML=['medical','pii','entity'].map(g=>{let rs=D.macros.filter(r=>r.group===g);return `<section class="macro"><h3><span class="pill g-${g}">${rs[0].group_name}</span> · ${rs[0].datasets} datasets</h3><table><thead><tr><th class="left">Student</th><th>P</th><th>R</th><th>F1</th><th>F2</th><th>Mask</th></tr></thead><tbody>${rs.map(r=>{let x=r[mode];return `<tr><td class="left meta">${r.model_name}</td><td>${num(x.precision)}</td><td>${num(x.recall)}</td><td>${num(x.f1)}</td><td>${num(x.f2)}</td><td>${pct(x.predicted_mask_rate)}</td></tr>`}).join('')}</tbody></table></section>`}).join('')}
function renderAnalysis(){const bestMacro=(g,modeName,field)=>D.macros.filter(r=>r.group===g).sort((a,b)=>b[modeName][field]-a[modeName][field])[0],medical=bestMacro('medical','budget','f1'),pii=bestMacro('pii','budget','f1'),entity=bestMacro('entity','budget','f1'),privacy=bestMacro('pii','privacy','f2');let perDataset={};D.rows.forEach(r=>{if(!perDataset[r.dataset]||r.budget.f1>perDataset[r.dataset].budget.f1)perDataset[r.dataset]=r});let ranked=Object.values(perDataset).sort((a,b)=>a.budget.f1-b.budget.f1),hard=ranked[0],easy=ranked[ranked.length-1],fast=D.efficiency.filter(r=>r.parameters).sort((a,b)=>b.sentences_per_second-a.sentences_per_second)[0],distil=D.efficiency.find(r=>r.name==='DistilRoBERTa'),multi=D.exploratory.find(r=>r.experiment==='Multi-domain'&&r.model==='DistilRoBERTa'),lodo=D.exploratory.find(r=>r.experiment==='LODO'&&r.model==='DistilRoBERTa'),frozen=D.exploratory.find(r=>r.experiment==='Frozen baseline'),fine=D.exploratory.find(r=>r.experiment==='Encoder fine-tuning'),scale=D.exploratory.find(r=>r.experiment==='데이터 규모 확대'),large=D.exploratory.find(r=>r.experiment==='1,000개 모델 비교'&&r.model==='DistilRoBERTa');$('analysis').innerHTML=`<article class="analysis-card"><h3>그룹별 Student 성능</h3><p>동일 예산 Macro F1 최고 모델은 의료 <strong>${medical.model_name} ${num(medical.budget.f1)}</strong>, 실제 PII <strong>${pii.model_name} ${num(pii.budget.f1)}</strong>, 엔티티 대조 <strong>${entity.model_name} ${num(entity.budget.f1)}</strong>다. 세 그룹 모두 DistilRoBERTa가 평균 최고지만 정책 의미가 달라 그룹 간 점수는 합치지 않는다.</p></article><article class="analysis-card"><h3>쉬운 데이터와 어려운 데이터</h3><p>각 데이터셋의 최고 Student를 기준으로 가장 높은 F1은 <strong>${easy.dataset_name} ${num(easy.budget.f1)}</strong>, 가장 낮은 F1은 <strong>${hard.dataset_name} ${num(hard.budget.f1)}</strong>다. 낮은 점수는 단순 모델 크기뿐 아니라 문장 형태와 규칙 토큰 분포의 영향을 받는다.</p></article><article class="analysis-card"><h3>Recall 중심 운용점</h3><p>실제 PII 그룹의 최고 F2 운용은 <strong>${privacy.model_name}</strong>이며 Recall <strong>${num(privacy.privacy.recall)}</strong>, F2 <strong>${num(privacy.privacy.f2)}</strong>다. 대신 평균 Student mask가 동일예산 ${pct(pii.budget.predicted_mask_rate)}에서 ${pct(privacy.privacy.predicted_mask_rate)}로 증가하므로 privacy와 utility를 같이 봐야 한다.</p></article><article class="analysis-card"><h3>속도·크기 trade-off</h3><p><strong>${fast.name}</strong>가 ${num(fast.sentences_per_second,1)}문장/s, ${num(fast.model_state_mb,1)}MB로 가장 빠르고 작다. <strong>DistilRoBERTa</strong>는 메인 성능이 가장 높지만 ${num(distil.sentences_per_second,1)}문장/s, ${num(distil.model_state_mb,1)}MB이므로 경량 배포에는 ELECTRA/BERT와 별도 절충이 필요하다.</p></article><article class="analysis-card"><h3>학습 방식과 데이터 규모</h3><p>초기 Drug Reviews에서 Frozen BERT-tiny F1 ${num(frozen.f1)} → encoder fine-tuning ${num(fine.f1)} → 1,000개 ${num(scale.f1)}로 개선됐다. 같은 1,000개에서 DistilRoBERTa는 <strong>${num(large.f1)}</strong>까지 올라 encoder 학습·데이터 증가·모델 용량이 모두 영향을 줬다.</p></article><article class="analysis-card"><h3>도메인 일반화</h3><p>DistilRoBERTa는 4개 의료 도메인을 모두 학습한 Multi-domain Macro F1 <strong>${num(multi.f1)}</strong>이지만 target을 제외한 LODO에서는 <strong>${num(lodo.f1)}</strong>로 하락했다. 즉 공통 규칙만 배운 것이 아니라 target 도메인의 표현을 본 효과도 포함된다.</p></article>`}

function staticTables(){$('eff').innerHTML=D.efficiency.map(r=>`<tr><td class="left meta dataset">${r.name}</td><td class="left meta">${r.implementation}</td><td>${int(r.parameters)}</td><td>${r.model_state_mb==null?'외부':num(r.model_state_mb,1)+' MB'}</td><td>${num(r.load_seconds,2)} s</td><td>${num(r.median_ms,2)} ms</td><td>${num(r.p95_ms,2)} ms</td><td>${num(r.sentences_per_second,1)}/s</td><td>${num(r.peak_rss_mb,1)} MB</td></tr>`).join('');$('explore').innerHTML=D.exploratory.map(r=>`<tr><td class="left meta dataset">${r.experiment}</td><td class="left meta">${r.scope}</td><td class="left meta">${r.model}</td><td>${num(r.f1)}</td><td>${r.std==null?'—':'±'+num(r.std)}</td><td>${num(r.f2)}</td><td class="left meta">${r.note}</td></tr>`).join('');$('sst').innerHTML=D.sst2.map(r=>`<tr><td class="left meta">${r.condition}</td><td>${pct(r.rate)}</td><td>${pct(r.original)}</td><td>${pct(r.teacher)}</td><td>${pct(r.student)}</td><td>${pct(r.random)}</td></tr>`).join('')}
document.querySelectorAll('.mode').forEach(b=>b.onclick=()=>{document.querySelectorAll('.mode').forEach(x=>x.classList.remove('active'));b.classList.add('active');mode=b.dataset.mode;render()});$('group').onchange=$('model').onchange=render;$('search').oninput=render;$('theme').onclick=()=>{let dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';$('theme').textContent=dark?'라이트 모드':'다크 모드'};render();staticTables();renderAnalysis();</script></body></html>'''

def main():
 data=rows(); payload={'rows':data,'macros':macros(data),'efficiency':efficiency(),'exploratory':exploratory(),'sst2':sst2()}
 unique={row['dataset']:row for row in data}
 example_count=sum(row['examples'] or 0 for row in unique.values())
 html=HTML.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')))
 html=html.replace('__EXAMPLE_COUNT__',f'{example_count:,}')
 html=html.replace('__SPLIT_TABLE__',split_table(data))
 OUT.mkdir(exist_ok=True); write_csv(data,OUT/'redactor_results.csv')
 (OUT/'redactor_results_dashboard.html').write_text(html,encoding='utf-8'); print(f'wrote dashboard; rows={len(data)}, csv_rows={len(data)*2}, examples={example_count:,}')
if __name__=='__main__': main()
