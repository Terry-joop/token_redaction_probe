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
 'drug':('Drug Reviews','RedactFormer medterm-v4','약물 리뷰'),
 'symptom2dx':('Symptom2Dx','RedactFormer medterm-v4','증상→진단'),
 'adr':('ADR','RedactFormer medterm-v4','약물 부작용'),
 'redditmh':('RedditMH','RedactFormer medterm-v4','정신건강 서술'),
 'mednli':('MedNLI','RedactFormer medterm-v4','의료 문장쌍 NLI'),
 'mentalhealth':('Mental Health','RedactFormer medterm-v4','정신상태 분류'),
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

def robustness_table():
 source=load(Path('reports/robustness_v14_results.json'))
 rows=[]
 ablation=source['augmentation_ablation']
 absolute=ablation['absolute_target_evaluation']
 target=absolute['summary']
 repeated=ablation['full_aug_seed_repeats']['summary']
 noisy_ci_lows=[
  run['student_noisy_advantage_ci95'][0] for run in absolute['runs']
 ]
 noisy_ci_highs=[
  run['student_noisy_advantage_ci95'][1] for run in absolute['runs']
 ]
 ablation_rows=[]
 for run in ablation['runs']:
  ci=run['survival_ci95']
  aug='seen-noise 증강' if run['augmentation'] else 'clean-only'
  ablation_rows.append(
   f"<tr><td class='left meta dataset'>{run['name']}</td><td>{run['train_rows']:,}</td><td class='left meta'>{aug}</td>"
   f"<td>{run['clean_precision']:.3f}</td><td>{run['clean_recall']:.3f}</td><td>{run['clean_f1']:.3f}</td><td>{run['clean_f2']:.3f}</td>"
   f"<td>{run['teacher_mask_rate']*100:.2f}%</td><td>{run['student_mask_rate']*100:.2f}%</td><td>{run['unseen_noisy_f2']:.3f}</td>"
   f"<td>{run['shared_targets']}</td><td>{run['rule_span_survival']*100:.1f}%</td><td>{run['student_span_survival']*100:.1f}%</td>"
   f"<td>{run['survival_delta']*100:+.1f}%p<span class='task'>[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]</span></td></tr>"
  )
 group_counts={}
 operating_rows=[]
 all_models=[]
 for item in source['datasets'].values():
  group_counts[item['group']]=group_counts.get(item['group'],0)+1+len(item['models'])
  all_models.extend(item['models'].values())
 seen_groups=set()
 for item in source['datasets'].values():
  rule=item['rule']; group=item['group']; group_cell=''
  row_count=1+len(item['models'])
  if group not in seen_groups:
   css_group='medical' if group=='medical' else 'pii'
   group_cell=f"<td rowspan='{group_counts[group]}' class='meta merge'><span class='pill g-{css_group}'>{item['group_name']}</span></td>"
   seen_groups.add(group)
  split=item['splits']
  dataset_cell=f"<td rowspan='{row_count}' class='left meta merge dataset-cell'><b class='dataset'>{item['name']}</b><span class='task'>{item['teacher']} · {split['train']:,}/{split['validation']:,}/{split['test']:,} · {item['pairs']:,}쌍</span></td>"
  rows.append(
   f"<tr>{group_cell}{dataset_cell}<td class='left meta dataset'>규칙 v1.4</td><td>—</td><td>{rule['clean_f2']:.3f}</td><td>{rule['noisy_precision']:.3f}</td><td>{rule['noisy_recall']:.3f}</td><td>{rule['noisy_f1']:.3f}</td><td>{rule['noisy_f2']:.3f}</td><td>{rule['f2_drop']:.3f}</td><td>{rule['noisy_mask_rate']*100:.2f}%</td><td>{rule['newly_leaked_span_rate']*100:.2f}%</td><td>—</td></tr>"
  )
  best_noisy=max(model['robustness']['noisy_f2'] for model in item['models'].values())
  for model in item['models'].values():
   robust=model['robustness']; boot=model['bootstrap_delta_f2']; ci=boot['ci95']
   best_class=" class='best'" if robust['noisy_f2']==best_noisy else ''
   rows.append(
    f"<tr><td class='left meta dataset'>{model['name']}</td><td>{model['threshold']:.2f}</td><td>{robust['clean_f2']:.3f}</td><td>{robust['noisy_precision']:.3f}</td><td>{robust['noisy_recall']:.3f}</td><td>{robust['noisy_f1']:.3f}</td><td{best_class}>{robust['noisy_f2']:.3f}</td><td>{robust['f2_drop']:.3f}</td><td>{robust['noisy_mask_rate']*100:.2f}%</td><td>{robust['newly_leaked_span_rate']*100:.2f}%*</td><td>{boot['mean']:.3f}<span class='task'>[{ci[0]:.3f}, {ci[1]:.3f}]</span></td></tr>"
   )
   budget=model['operating_points']['budget_matched']; privacy=model['operating_points']['f2_optimized']
   operating_rows.append(
    f"<tr><td class='left meta dataset'>{item['name']}</td><td class='left meta'>{model['name']}</td><td>{budget['threshold']:.2f}</td><td>{budget['precision']:.3f}</td><td>{budget['recall']:.3f}</td><td>{budget['f2']:.3f}</td><td>{budget['predicted_mask_rate']*100:.2f}%</td><td>{privacy['threshold']:.2f}</td><td>{privacy['precision']:.3f}</td><td>{privacy['recall']:.3f}</td><td>{privacy['f2']:.3f}</td><td>{privacy['predicted_mask_rate']*100:.2f}%</td></tr>"
   )
 model_best_counts={key:0 for key in next(iter(source['datasets'].values()))['models']}
 for item in source['datasets'].values():
  winner=max(item['models'],key=lambda key:item['models'][key]['robustness']['noisy_f2'])
  model_best_counts[winner]+=1
 best_model=max(model_best_counts,key=model_best_counts.get)
 best_model_name=next(iter(source['datasets'].values()))['models'][best_model]['name']
 student_wins=sum(model['robustness']['noisy_f2']>item['rule']['noisy_f2'] for item in source['datasets'].values() for model in item['models'].values())
 clean_passes=sum(model['acceptance']['final_student_quality_gate']['pass'] for model in all_models)
 budget_passes=sum(model['acceptance']['matched_budget_gate']['pass'] for model in all_models)
 privacy_tradeoff_runs=sum(
  model['operating_points']['f2_optimized']['recall']>=model['operating_points']['budget_matched']['recall']
  and model['operating_points']['f2_optimized']['f2']>=model['operating_points']['budget_matched']['f2']
  and model['operating_points']['f2_optimized']['predicted_mask_rate']>=model['operating_points']['budget_matched']['predicted_mask_rate']
  for model in all_models
 )
 analysis=source['matrix_analysis']; analysis_rows=[]
 for model in analysis['models'].values():
  medical=model['groups']['medical']; general=model['groups']['general']
  analysis_rows.append(
   f"<tr><td class='left meta dataset'>{model['name']}</td><td>{model['parameters']/1_000_000:.1f}M</td><td>{model['model_state_mb']:.1f} MB</td><td>{model['sentences_per_second']:.1f}/s</td><td>{medical['clean_f2']:.3f} → {medical['noisy_f2']:.3f}<span class='task'>−{medical['f2_drop']:.3f}</span></td><td>{general['clean_f2']:.3f} → {general['noisy_f2']:.3f}<span class='task'>−{general['f2_drop']:.3f}</span></td><td>{model['clean_gate_passes']}/10</td></tr>"
  )
 bert=analysis['models']['bert_tiny']; electra=analysis['models']['electra_small']; distil=analysis['models']['distilroberta']
 return (
  "<h2>부록 A-1. 최신 v1.4 입력 교란 강건성 — 10개 데이터셋 × 3모델</h2>"
  "<p class='lede'>clean v1.4 라벨을 결정적 편집으로 noisy 문장에 이동한 pseudo-gold 기준이다. BERT-tiny, ELECTRA-small, DistilRoBERTa를 같은 split·학습 조건·동일 마스킹 예산으로 비교한다. Noisy P/R은 이동된 token 정답에 대한 Precision/Recall이다.</p>"
  "<div class='tablewrap solo'><table><thead><tr><th class='left'>그룹</th><th class='left'>데이터셋</th><th class='left'>방식</th><th>Budget Th.</th><th>Clean F2</th><th>Noisy P</th><th>Noisy R</th><th>Noisy F1</th><th>Noisy F2</th><th>F2 하락</th><th>Noisy mask</th><th>신규 누출</th><th>Student−Rule ΔF2<br>95% CI</th></tr></thead><tbody>"
  + ''.join(rows)
  + "</tbody></table></div>"
  f"<div class='notice'><strong>3모델 결과:</strong> 30개 Student run 중 규칙보다 noisy F2가 높은 경우는 {student_wins}개, clean 대체 최소선 통과는 {clean_passes}/30개, 마스킹 예산 ±1%p 통과는 {budget_passes}/30개다. 데이터셋별 noisy F2 최고 Student는 <strong>{best_model_name} {model_best_counts[best_model]}/10개</strong>다. *Student 신규 누출은 clean에서 먼저 맞힌 span만 분모로 한 조건부 값이다.</div>"
  "<div class='notice warn'><strong>0.85의 의미:</strong> clean F1 0.85는 보편적 표준이 아니라 기존 전체 Drug ELECTRA의 F1 0.892보다 낮게 사전 고정한 pilot screening floor다. Recall만 높이려고 과도하게 가리는 모델을 거르는 보조선이며, privacy 핵심선은 F2·Recall 0.90이다. 최종 대체 판정에는 noisy 신뢰구간과 human-gold 검증도 필요하다.</div>"
  f"<details><summary>동일 마스킹 예산 vs Recall 중심 F2 · clean 운용점 30개 보기</summary><div><p><strong>동일 예산</strong>은 Teacher와 비슷한 비율을 가려 모델 간 공정 비교에 적합하다. <strong>Recall 중심 F2</strong>는 민감 누락을 더 비싸게 보므로 privacy-first 배포 후보에 적합하지만 더 많이 가릴 수 있다. 실제로 {privacy_tradeoff_runs}/30개 run 모두 Recall·F2·마스킹률이 함께 증가했다. 따라서 논문 메인은 동일 예산, F2는 보조 운용점으로 제시한다.</p><div class='tablewrap solo'><table><thead><tr><th class='left'>데이터셋</th><th class='left'>Student</th><th>예산 Th.</th><th>예산 P</th><th>예산 R</th><th>예산 F2</th><th>예산 mask</th><th>F2 Th.</th><th>F2 P</th><th>F2 R</th><th>F2</th><th>F2 mask</th></tr></thead><tbody>"
  + ''.join(operating_rows)
  + "</tbody></table></div></div></details>"
  "<h3>부록 A-1-1. 모델 크기별 결과 분석</h3>"
  "<div class='tablewrap solo'><table><thead><tr><th class='left'>Student</th><th>Params</th><th>모델 크기</th><th>처리량</th><th>의료 Clean→Noisy F2</th><th>일반 Clean→Noisy F2</th><th>Clean gate</th></tr></thead><tbody>"
  + ''.join(analysis_rows)
  + "</tbody></table></div>"
  "<div class='analysis-grid' style='margin-top:12px'>"
  f"<article class='analysis-card'><h3>크기가 커질수록 절대 성능 상승</h3><p><strong>{analysis['monotonic_noisy_f2_datasets']}/10개 데이터셋</strong>에서 BERT &lt; ELECTRA &lt; DistilRoBERTa 순으로 noisy F2가 올랐다. 의료는 {bert['groups']['medical']['noisy_f2']:.3f} → {electra['groups']['medical']['noisy_f2']:.3f} → {distil['groups']['medical']['noisy_f2']:.3f}, 일반은 {bert['groups']['general']['noisy_f2']:.3f} → {electra['groups']['general']['noisy_f2']:.3f} → {distil['groups']['general']['noisy_f2']:.3f}다.</p></article>"
  f"<article class='analysis-card'><h3>F2 하락만 보면 안 되는 이유</h3><p>의료 평균 하락은 BERT {bert['groups']['medical']['f2_drop']:.3f}, ELECTRA {electra['groups']['medical']['f2_drop']:.3f}, Distil {distil['groups']['medical']['f2_drop']:.3f}다. BERT가 덜 하락하지만 clean 출발점이 {bert['groups']['medical']['clean_f2']:.3f}로 낮다. 따라서 <strong>절대 noisy F2와 규칙 격차</strong>를 함께 봐야 한다.</p></article>"
  f"<article class='analysis-card'><h3>규칙 대체에는 아직 부족</h3><p>가장 좋은 DistilRoBERTa도 규칙보다 noisy F2가 의료 <strong>{abs(distil['groups']['medical']['rule_gap']):.3f}</strong>, 일반 <strong>{abs(distil['groups']['general']['rule_gap']):.3f}</strong> 낮다. Clean gate는 모델이 커지며 {bert['clean_gate_passes']} → {electra['clean_gate_passes']} → {distil['clean_gate_passes']}/10으로 늘었지만 noisy 규칙 우세는 유지됐다.</p></article>"
  f"<article class='analysis-card'><h3>품질과 비용의 절충</h3><p>ELECTRA→Distil의 noisy F2 이득은 의료 {distil['groups']['medical']['noisy_f2']-electra['groups']['medical']['noisy_f2']:+.3f}, 일반 {distil['groups']['general']['noisy_f2']-electra['groups']['general']['noisy_f2']:+.3f}지만 파라미터·파일은 약 {distil['parameters']/electra['parameters']:.1f}배다. <strong>품질 최우선은 DistilRoBERTa, 경량 절충은 ELECTRA-small, 속도 최우선은 BERT-tiny</strong>로 해석한다.</p></article>"
  "</div>"
  "<h2>부록 A-2. 전체 데이터·표면 교란 증강 비교</h2>"
  f"<p class='lede'>전체 test에서 생성 가능한 unseen target {absolute['unseen_target_pairs']:,}개를 고정 pseudo-gold로 둔 비교다. 같은 원문에서 파생된 여러 오염은 원문 단위로 묶어 통계 처리했다.</p>"
  f"<div class='notice'><strong>데이터 규모:</strong> Drug Reviews 원본 {absolute['source_examples']:,}문장, clean train {absolute['clean_train_examples']:,}, validation {absolute['validation_examples']:,}, test {absolute['test_examples']:,}문장 전체를 사용했다. seen-noise {absolute['augmented_train_examples']:,}행을 추가해 학습 입력은 {absolute['total_augmented_train_rows']:,}행이다. 전체 test에서 unseen target-pair {absolute['unseen_target_pairs']:,}개가 생성됐고 고유 원문은 {absolute['unique_source_examples']:,}개다.</div>"
  f"<h3>부록 A-2-1. 메인 비교 — 전체 unseen target {absolute['unseen_target_pairs']:,}개를 동일 분모로 사용</h3>"
  "<p class='lede'>정확한 target span 전체를 가렸을 때만 성공이다. Student는 seed 42·43·44 평균이다.</p>"
  "<div class='tablewrap solo'><table><thead><tr><th class='left'>방식</th><th>동일 분모</th><th>Clean target 탐지율</th><th>오염 후 target 탐지율</th><th>Clean→오염 하락</th><th>오염 후 Student−규칙</th></tr></thead><tbody>"
  f"<tr><td class='left meta dataset'>규칙 v1.4</td><td>{absolute['unseen_target_pairs']:,}</td><td>{target['rule_clean_target_recall']['mean']*100:.1f}%</td><td>{target['rule_noisy_target_recall']['mean']*100:.1f}%</td><td>−{target['rule_drop']['mean']*100:.1f}%p</td><td>—</td></tr>"
  f"<tr><td class='left meta dataset'>전체+증강 Student<span class='task'>3-seed 평균±표준편차</span></td><td>{absolute['unseen_target_pairs']:,}</td><td>{target['student_clean_target_recall']['mean']*100:.1f}%<span class='task'>±{target['student_clean_target_recall']['sample_std']*100:.1f}%p</span></td><td class='best'>{target['student_noisy_target_recall']['mean']*100:.1f}%<span class='task'>±{target['student_noisy_target_recall']['sample_std']*100:.1f}%p</span></td><td>−{target['student_drop']['mean']*100:.1f}%p<span class='task'>±{target['student_drop']['sample_std']*100:.1f}%p</span></td><td class='best'>+{target['student_noisy_advantage']['mean']*100:.1f}%p<span class='task'>±{target['student_noisy_advantage']['sample_std']*100:.1f}%p · seed별 CI 모두 &gt;0</span></td></tr>"
  "</tbody></table></div>"
  f"<div class='notice'><strong>메인 결론:</strong> 전체 {absolute['unseen_target_pairs']:,}개 target에서 오염 후 정확한 span 탐지율은 규칙 {target['rule_noisy_target_recall']['mean']*100:.1f}%, Student {target['student_noisy_target_recall']['mean']*100:.1f}%다. 규칙은 {target['rule_drop']['mean']*100:.1f}%p, Student는 {target['student_drop']['mean']*100:.1f}%p 하락해 Student가 {target['student_drop_advantage']['mean']*100:.1f}%p 덜 하락했다. 오염 후 절대 탐지율 차이는 +{target['student_noisy_advantage']['mean']*100:.1f}%p이고, 세 seed의 원문-cluster 95% CI는 모두 0보다 컸다(하한 {min(noisy_ci_lows)*100:.1f}%p 이상).</div>"
  f"<div class='notice'><strong>전체 clean 성능:</strong> 전체 validation에서 threshold를 선택해 전체 test에 적용한 3-seed 평균은 Precision {repeated['clean_precision']['mean']:.3f}, Recall {repeated['clean_recall']['mean']:.3f}, F1 {repeated['clean_f1']['mean']:.3f}, F2 {repeated['clean_f2']['mean']:.3f}다.</div>"
  "<h3>부록 A-2-2. 이전 제한본 ablation과 공통-span 보조 분석</h3>"
  f"<p class='lede'>아래 네 행은 모델 선택을 위한 이전 500 validation/1,000 test·457 unseen-pair 제한본(seed 42)이다. 전체 {absolute['unseen_target_pairs']:,}개 메인 결과와 분모가 다르므로 참고 ablation으로만 본다.</p>"
  "<div class='tablewrap solo'><table><thead><tr><th class='left'>조건</th><th>학습 행</th><th class='left'>학습 입력</th><th>Clean P</th><th>Clean R</th><th>Clean F1</th><th>Clean F2</th><th>Teacher mask</th><th>Student mask</th><th>Unseen F2</th><th>공통 span<br>(보조 분모)</th><th>조건부<br>규칙 생존</th><th>조건부<br>Student 생존</th><th>조건부 차이<br>95% CI</th></tr></thead><tbody>"
  + ''.join(ablation_rows)
  + "</tbody></table></div>"
  f"<div class='notice'><strong>제한본 ablation:</strong> 데이터 확대와 seen-noise 증강이 모두 성능을 높여 전체+증강 모델을 최종 조건으로 선택했다. 이 선택 뒤의 최종 주장은 위 전체 validation/test 및 {absolute['unseen_target_pairs']:,} pair 평가를 기준으로 한다.</div>"
  f"<div class='notice warn'><strong>해석:</strong> 전체 고정 target 기준 Student의 오염 탐지율 우위는 학습형 redactor의 표면 강건성 근거다. 그러나 Student의 전체 noisy token F2 평균 {repeated['unseen_noisy_f2']['mean']:.3f}는 규칙 {repeated['rule_unseen_noisy_f2']['mean']:.3f}보다 낮다. 따라서 현재 결론은 완전 대체가 아니라 표면 결함 보완 후보이며 human-gold 검증이 필요하다.</div>"

 )


def strict_matrix_table():
 path=ROOT/'reports/robustness_v14_strict_matrix.json'
 if not path.exists(): return ''
 source=json.loads(path.read_text(encoding='utf-8'))
 rows=[]; seed_rows=[]; analysis_cards=[]; group_counts={}
 for item in source['datasets'].values(): group_counts[item['group']]=group_counts.get(item['group'],0)+1
 seen=set()
 for item in source['datasets'].values():
  s=item['summary']; group=item['group']; group_cell=''
  if group not in seen:
   group_cell=f"<td rowspan='{group_counts[group]}' class='meta merge'><span class='pill g-{group}'>{item['group_name']}</span></td>"
   seen.add(group)
  advantage=s['student_minus_rule_noisy']['mean']; drop_adv=s['student_drop_advantage']['mean']
  if item['absolute_gate_pass_seeds']==3:
   verdict='우세 · CI 3/3'
   verdict_class='best'
  elif advantage>0 and drop_adv>0:
   verdict='평균 우세 · CI 미달'
   verdict_class='over'
  else:
   verdict='미달'
   verdict_class='low'
  rows.append(
   f"<tr>{group_cell}<td class='left meta dataset'>{item['name']}<span class='task'>{item['teacher']}</span></td>"
   f"<td>{item['clean_train_rows']:,}+{item['augmented_train_rows']:,}</td><td>{item['splits']['test']:,}</td><td>{item['unseen_pairs']:,}</td>"
   f"<td>{s['clean_f2']['mean']:.3f}<span class='task'>±{s['clean_f2']['sample_std']:.3f}</span></td>"
   f"<td class='{'best' if item['quality_gate_pass_seeds']==3 else 'low'}'>{item['quality_gate_pass_seeds']}/3</td>"
   f"<td>{s['rule_clean_target_detection']['mean']*100:.1f}% → {s['rule_noisy_target_detection']['mean']*100:.1f}%<span class='task'>−{s['rule_detection_drop']['mean']*100:.1f}%p</span></td>"
   f"<td>{s['student_clean_target_detection']['mean']*100:.1f}% → {s['student_noisy_target_detection']['mean']*100:.1f}%<span class='task'>−{s['student_detection_drop']['mean']*100:.1f}%p</span></td>"
   f"<td>{advantage*100:+.1f}%p<span class='task'>하락폭 이점 {drop_adv*100:+.1f}%p</span></td><td class='{verdict_class}'>{verdict}</td></tr>"
  )
  analysis_cards.append(
   f"<article class='analysis-card'><h3>{item['name']} · {verdict}</h3>"
   f"<p><strong>규모:</strong> test {item['splits']['test']:,}문장, unseen target-pair {item['unseen_pairs']:,}개.</p>"
   f"<p><strong>Clean:</strong> Student F2 {s['clean_f2']['mean']:.3f}±{s['clean_f2']['sample_std']:.3f}, gate {item['quality_gate_pass_seeds']}/3.</p>"
   f"<p><strong>오염 target 탐지:</strong> 규칙 {s['rule_noisy_target_detection']['mean']*100:.1f}% vs Student {s['student_noisy_target_detection']['mean']*100:.1f}%"
   f" (Student−규칙 {advantage*100:+.1f}%p). Clean→오염 하락은 규칙 {s['rule_detection_drop']['mean']*100:.1f}%p vs Student {s['student_detection_drop']['mean']*100:.1f}%p다.</p>"
   f"<p><strong>Noisy token F2:</strong> 규칙 {s['rule_noisy_f2']['mean']:.3f} vs Student {s['student_noisy_f2']['mean']:.3f}. "
   f"<strong>의미:</strong> {item['meaning']} CI gate {item['absolute_gate_pass_seeds']}/3.</p></article>"
  )
  for run in item['runs']:
   seed_rows.append(
    f"<tr><td class='left meta dataset'>{item['name']}</td><td><span class='seed'>{run['seed']}</span></td><td>{run['threshold']:.2f}</td>"
    f"<td>{run['clean_f2']:.3f}</td><td>{run['rule_noisy_target_detection']*100:.1f}%</td><td>{run['student_noisy_target_detection']*100:.1f}%</td>"
    f"<td>{run['student_minus_rule_noisy']*100:+.1f}%p<span class='task'>[{run['student_minus_rule_noisy_ci95'][0]*100:+.1f}, {run['student_minus_rule_noisy_ci95'][1]*100:+.1f}]</span></td>"
    f"<td>{run['rule_detection_drop']*100:.1f}%p</td><td>{run['student_detection_drop']*100:.1f}%p</td></tr>"
   )
 average_wins=sum(
  item['summary']['student_minus_rule_noisy']['mean']>0
  and item['summary']['student_drop_advantage']['mean']>0
  for item in source['datasets'].values()
 )
 strict_wins=sum(
  item['absolute_gate_pass_seeds']==3
  for item in source['datasets'].values()
 )
 pairs=sum(item['unseen_pairs'] for item in source['datasets'].values())
 sources=sum(item['unique_test_sources'] for item in source['datasets'].values())
 privacy_strict=sum(
  item['group'] in {'medical','pii'} and item['absolute_gate_pass_seeds']==3
  for item in source['datasets'].values()
 )
 medical=source['groups']['medical']['metrics']; pii=source['groups']['pii']['metrics']; entity=source['groups']['entity']['metrics']
 return (
  "<h2>4-3. 전 데이터셋 strict 5/7 검증</h2>"
  "<p class='lede'>각 데이터셋의 전체 clean train에 seen 5종만 증강하고, 전체 test에서는 학습에 없던 unseen 7종만 평가했다. Student는 ELECTRA-small, seed 42·43·44로 고정했다.</p>"
  "<div class='notice'><strong>읽는 핵심:</strong> Clean 최신 규칙이 잡은 target을 절대 정답으로 고정한다. 규칙과 Student 각각의 clean→noisy 탐지율 하락을 비교하며, Student noisy 탐지가 더 높고 하락폭이 더 작아야 표면 교란 보완 근거가 된다.</div>"
  "<div class='tablewrap solo'><table><thead><tr><th class='left'>그룹</th><th class='left'>데이터셋</th><th>Train clean+seen</th><th>Test 원문</th><th>Unseen pair</th><th>Student clean F2</th><th>Clean gate</th><th>규칙 clean→noisy 탐지</th><th>Student clean→noisy 탐지</th><th>Student−규칙</th><th>판정</th></tr></thead><tbody>"
  +''.join(rows)+"</tbody></table></div>"
  "<h3>4-3-1. 데이터셋별 수치·의미 해석</h3>"
  "<p class='lede'>target 탐지율과 clean→오염 하락을 함께 보고, 전체-token noisy F2와 3-seed CI로 과대해석을 막는다.</p>"
  "<div class='analysis-grid'>"+''.join(analysis_cards)+"</div>"
  f"<div class='notice'><strong>요약:</strong> 전체 {len(source['datasets'])}개 데이터셋, {pairs:,}개 unseen target pair(고유 원문 {sources:,}개)에서 3-seed를 반복했다. 평균 noisy 탐지와 하락폭이 모두 좋은 데이터셋은 <strong>{average_wins}/10개</strong>, 두 차이의 95% CI가 seed 3개 모두 0보다 큰 엄격 우세는 <strong>{strict_wins}/10개</strong>다.</div>"
  f"<div class='notice'><strong>그룹 평균 noisy 탐지:</strong> 의료는 규칙 {medical['rule_noisy_target_detection']*100:.1f}% vs Student {medical['student_noisy_target_detection']*100:.1f}%, 실제 PII는 {pii['rule_noisy_target_detection']*100:.1f}% vs {pii['student_noisy_target_detection']*100:.1f}%, 비개인 엔티티 대조는 {entity['rule_noisy_target_detection']*100:.1f}% vs {entity['student_noisy_target_detection']*100:.1f}%다.</div>"
  f"<div class='notice warn'><strong>결론:</strong> privacy 관련 8개 중 엄격 우세는 <strong>{privacy_strict}/8개</strong>(Drug Reviews)다. FinPhraseBank의 우세는 비개인 엔티티 대조 결과이므로 privacy 성공으로 세지 않는다. 현재 Student는 일부 도메인의 규칙 보완 후보이지 전체 규칙 대체 모델은 아니다.</div>"
  "<details><summary>데이터셋별 seed 42·43·44와 95% CI 보기</summary><div><div class='tablewrap solo'><table><thead><tr><th class='left'>데이터셋</th><th>Seed</th><th>Th.</th><th>Clean F2</th><th>규칙 noisy 탐지</th><th>Student noisy 탐지</th><th>차이 95% CI</th><th>규칙 하락</th><th>Student 하락</th></tr></thead><tbody>"
  +''.join(seed_rows)+"</tbody></table></div></div></details>"
 )


def future_defect_time_axis_table():
 path=ROOT/'reports/future_defect_time_axis_summary.json'
 if not path.exists(): return ''
 source=load(path); rows=[]
 for item in source['datasets'].values():
  s=item['summary']
  verdict='우세 · CI 3/3' if item['all_seeds_absolute_gate'] else '미달'
  verdict_class='best' if item['all_seeds_absolute_gate'] else 'low'
  rows.append(
   f"<tr><td class='left meta dataset'>{item['name']}<span class='task'>{item['domain']} · {item['policy']}</span></td>"
   f"<td>{item['pairs']:,}</td><td>{s['clean_f2']:.3f}</td>"
   f"<td>{s['rule_noisy_target_detection']*100:.1f}%</td><td class='best'>{s['student_noisy_target_detection']*100:.1f}%</td>"
   f"<td class='best'>{s['student_minus_rule_noisy']*100:+.1f}%p</td>"
   f"<td>{s['rule_detection_drop']*100:.1f}%p</td><td>{s['student_detection_drop']*100:.1f}%p</td>"
   f"<td class='best'>{s['student_drop_advantage']*100:+.1f}%p</td><td class='{verdict_class}'>{verdict}</td></tr>"
  )
 combo_rows=[]
 for item in source['datasets'].values():
  s=item['summary']
  for index,(prefix,label) in enumerate((
   ('rule','Rule only'), ('student','Student only'),
   ('rule_or','Rule OR Student'), ('rule_and','Rule AND Student'),
  )):
   dataset_cell=(
    f"<td rowspan='4' class='left meta merge dataset-cell'><b class='dataset'>{item['name']}</b>"
    f"<span class='task'>{item['domain']} · {item['policy']}</span></td>"
    if index == 0 else ''
   )
   target_detection=s[f'{prefix}_noisy_target_detection']
   target_delta=target_detection-s['rule_noisy_target_detection']
   combo_rows.append(
    f"<tr>{dataset_cell}<td class='left meta'>{label}</td>"
    f"<td class='best'>{target_detection*100:.1f}%</td><td class='best'>{target_delta*100:+.1f}%p</td>"
    f"<td>{s[f'{prefix}_noisy_mask']*100:.1f}%</td><td class='over'>{s[f'{prefix}_noisy_overmask']*100:.1f}%</td>"
    f"<td>{s[f'{prefix}_noisy_precision']:.3f}</td><td>{s[f'{prefix}_noisy_recall']:.3f}</td>"
    f"<td>{s[f'{prefix}_noisy_f1']:.3f}</td><td>{s[f'{prefix}_noisy_f2']:.3f}</td></tr>"
   )
 noise_rows=[]
 for name,item in source['pooled_by_noise'].items():
  noise_rows.append(
   f"<tr><td class='left meta dataset'>{name}</td><td>{item['eligible_shared_clean_targets']:,}</td>"
   f"<td>{item['rule_survival']*100:.1f}%</td><td>{item['student_survival']*100:.1f}%</td>"
   f"<td class='best'>{item['student_minus_rule']*100:+.1f}%p</td></tr>"
  )
 dataset_count=len(source['datasets'])
 pair_total=sum(item['pairs'] for item in source['datasets'].values())
 strict_win_count=sum(item['all_seeds_absolute_gate'] for item in source['datasets'].values())
 return (
  "<h2>4-4. 학습 미포함 입력 교란 평가</h2>"
  f"<p class='lede'>ELECTRA-small + hidden-128 MLP의 기존 strict seen-5 checkpoint를 그대로 사용했다. 학습·validation·threshold 선택에 없던 7종의 입력 교란을 test 전용으로 두고, clean 최신 v1.4 span을 고정 정답으로 이동했다. {dataset_count}개 데이터셋, {pair_total:,} pair, seed 42·43·44 결과다.</p>"
  "<div class='notice warn'><strong>모델 범위:</strong> 이 strict seen-5/unseen-7 프로토콜은 ELECTRA-small만 실행했다. BERT-tiny·DistilRoBERTa의 3모델 결과는 부록 A의 별도 clean-only·제한 표본 조건이므로 이 표의 수치와 직접 비교할 수 없다.</div>"
  "<div class='notice'><strong>읽는 법:</strong> 여기서 성공은 token F2가 아니라 <strong>오염 전 최신 규칙이 잡은 고정 target span을 오염 후에도 가렸는가</strong>다. Student가 규칙보다 미래 target을 더 많이 잡고, clean→future 하락도 더 작아야 우세다. 각 seed에서 source-cluster bootstrap 95% CI가 모두 0보다 큰 경우만 ‘우세’로 표시했다.</div>"
  "<div class='tablewrap solo'><table><thead><tr><th class='left'>데이터셋</th><th>Future pair</th><th>Clean F2</th><th>규칙 미래 탐지</th><th>Student 미래 탐지</th><th>탐지 차이</th><th>규칙 하락</th><th>Student 하락</th><th>하락폭 이점</th><th>판정</th></tr></thead><tbody>"
  + ''.join(rows)
  + "</tbody></table></div>"
  f"<div class='notice'><strong>결론 범위:</strong> {dataset_count}개 중 {strict_win_count}개 데이터셋이 raw 규칙 v1.4 대비 고정 target 탐지와 하락폭에서 3-seed 우세다. 우세 행은 입력 교란 상황에서의 <strong>로컬 fallback/병렬 보완 redactor</strong> 근거다. 최신 규칙 전체를 대체한다는 뜻은 아니며, 아래 결합 방식의 과마스킹도 함께 확인해야 한다.</div>"
  "<h3>4-4-1. 입력 교란에서 Rule/Student 결합 방식 비교</h3>"
  "<p class='lede'><strong>주 지표는 앞의 두 열</strong>이다. 고정 target 탐지는 clean 최신 규칙이 잡은 span을 미래 교란 후에도 전부 가린 비율이고, Rule 대비 차이는 그 차이다. 뒤의 Mask·FP·P/R/F1/F2는 OR·AND의 주 지표 변화가 과마스킹 때문인지 점검하는 보조 지표다. 모든 수치는 학습·검증·threshold 선택에 쓰지 않은 미래 교란 7종 noisy pair에서 측정했다.</p>"
  "<div class='tablewrap solo'><table><thead><tr><th class='left'>데이터셋</th><th class='left'>방식</th><th>고정 target<br>탐지</th><th>Rule 대비<br>차이</th><th>Mask</th><th>불필요 mask<br>(FP)</th><th>P</th><th>R</th><th>F1</th><th>F2</th></tr></thead><tbody>"
  + ''.join(combo_rows)
  + "</tbody></table></div>"
  "<div class='notice warn'><strong>불필요 mask(FP):</strong> pseudo-gold가 0인 토큰 중 실제로 1로 가린 비율이다. 즉 <strong>가리지 않아도 되는 것을 가린 비율</strong>이며 낮을수록 좋다. Mask는 전체 토큰 중 가린 비율이라 민감 토큰을 많이 찾은 결과와 과마스킹을 구분하지 못하므로, OR은 F2·고정 target 탐지와 함께 이 FP 열을 반드시 같이 본다.</div>"
  "<h3>4-4-2. 입력 교란 종류별 공통 clean-correct span 생존</h3>"
  "<p class='lede'>clean에서 규칙과 Student가 모두 맞힌 span만 분모로 둔 보조 분석이다. 특정 교란이 한 데이터셋에 거의 없으면 사례 수준으로만 해석한다.</p>"
  "<div class='tablewrap solo'><table><thead><tr><th class='left'>미래 교란</th><th>공통 span</th><th>규칙 생존</th><th>Student 생존</th><th>차이</th></tr></thead><tbody>"
  + ''.join(noise_rows)
 + "</tbody></table></div>"
 )


def actual_rule_version_time_axis_table():
 path=ROOT/'reports/temporal_rule_version_summary.json'
 if not path.exists(): return ''
 source=load(path); rows=[]; defect_rows=[]
 defect_names={
  'glued_dosage':'붙여 쓴 용량','c1_control':'C1 제어문자',
  'possessive':'소유격','long_identifier':'긴 숫자·식별자',
  'email':'이메일','url':'URL','social_handle':'소셜 핸들',
  'zip4':'ZIP+4','numeric_date':'숫자 날짜',
 }
 for item in source['datasets'].values():
  clean=item['clean']; ci=item['student_minus_rule_ci95']
  verdict_class='best' if item['verdict']=='Student 우세' else ('low' if '규칙' in item['verdict'] else '')
  student_class='best' if item['past_student_detection']>item['past_rule_detection'] else 'low'
  delta_class='best' if item['student_minus_rule']>0 else 'low'
  rows.append(
   f"<tr><td class='left meta dataset'>{item['name']}<span class='task'>{item['domain']}</span></td>"
   f"<td>{item['targets']:,}</td><td>{item['unique_sources']:,}</td><td>{clean['f2']:.3f}</td>"
   f"<td>{clean['predicted_mask_rate']*100:.1f}%</td>"
   f"<td>{item['past_rule_detection']*100:.1f}%</td>"
   f"<td class='{student_class}'>{item['past_student_detection']*100:.1f}%</td>"
   f"<td class='{delta_class}'>{item['student_minus_rule']*100:+.1f}%p<span class='task'>[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]</span></td>"
   f"<td>{item['latest_rule_detection']*100:.1f}%</td>"
   f"<td class='{verdict_class}'>{item['verdict']}</td></tr>"
  )
  first=True
  for defect,row in item['by_defect'].items():
   ci=row['student_minus_rule_ci95']
   dataset_cell=(
    f"<td rowspan='{len(item['by_defect'])}' class='left meta merge dataset-cell'><b class='dataset'>{item['name']}</b></td>"
    if first else ''
   ); first=False
   defect_rows.append(
    f"<tr>{dataset_cell}<td>v{row['introduced_in']}</td>"
    f"<td class='left meta'>{defect_names.get(defect,defect)}</td><td>{row['targets']:,}</td>"
    f"<td>{row['past_rule_detection']*100:.1f}%</td>"
    f"<td>{row['past_student_detection']*100:.1f}%</td>"
    f"<td>{row['student_minus_rule']*100:+.1f}%p<span class='task'>[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}]</span></td></tr>"
   )
 dataset_count=len(source['datasets'])
 target_count=sum(item['targets'] for item in source['datasets'].values())
 wins=sum(item['verdict']=='Student 우세' for item in source['datasets'].values())
 conclusion=(
  f"{wins}/{dataset_count}개에서 Student가 통계적으로 우세해 일부 미래 표면형의 규칙 유지보수 지연 보완 근거가 있다."
  if wins else
  "두 데이터셋 모두 v1.2 규칙이 통계적으로 우세했다. 현재의 단순 규칙 모방 Student가 미래 결함을 선제적으로 일반화한다는 가설은 지지되지 않았다."
 )
 return (
  "<h2>4-5. 실제 규칙 버전 시간축 평가 — v1.2 → v1.3/v1.4</h2>"
  f"<p class='lede'>과거 commit b8dff7e의 v1.2 규칙으로 전체 train을 다시 라벨링하고, 그 라벨만 본 ELECTRA-small Student를 이후 Git에서 실제 추가된 패치 target에 평가했다. {dataset_count}개 데이터셋, 최신 규칙 검증 target {target_count:,}개, seed 42 결과다.</p>"
  "<div class='notice'><strong>4-4와의 차이:</strong> 4-4는 최신 규칙을 기준으로 만든 학습 미포함 합성 교란이다. 이 절은 <strong>실제 Git 시간순서</strong>를 지켜 v1.2 코드·라벨·Student가 나중의 v1.3/v1.4 결함을 잡았는지 본다. 미래 target은 학습과 threshold 선택에 사용하지 않았다.</div>"
  "<div class='tablewrap solo'><table><thead><tr><th class='left'>데이터셋</th><th>Future target</th><th>고유 원문</th><th>v1.2 Clean F2</th><th>Student clean mask</th><th>v1.2 규칙 탐지</th><th>v1.2 Student 탐지</th><th>Student−규칙<br>95% CI</th><th>최신 v1.4<br>참고 상한</th><th>판정</th></tr></thead><tbody>"
  +''.join(rows)+"</tbody></table></div>"
  f"<div class='notice'><strong>판정:</strong> target을 구성하는 모든 word를 가려야 성공이다. source 원문 단위 bootstrap 95% CI의 하한이 0보다 클 때만 Student 우세로 표시했다. 현재 데이터셋 단위 우세는 <strong>{wins}/{dataset_count}</strong>다.</div>"
  "<h3>4-5-1. 실제 후속 패치 결함별 탐지</h3>"
  "<p class='lede'>최신 v1.4가 실제 민감 구간으로 확인한 후보만 분모로 사용한다. 최신 규칙 탐지는 정의상 100%이며, 아래 표는 패치 전 두 방식의 차이를 보여준다.</p>"
  "<div class='tablewrap solo'><table><thead><tr><th class='left'>데이터셋</th><th>추가 버전</th><th class='left'>실제 후속 패치</th><th>Target</th><th>v1.2 규칙</th><th>v1.2 Student</th><th>Student−규칙<br>95% CI</th></tr></thead><tbody>"
  +''.join(defect_rows)+"</tbody></table></div>"
  f"<div class='notice warn'><strong>현재 결론:</strong> {conclusion} pseudo-gold는 human-gold가 아니라 최신 규칙으로 검증했으므로 최신 규칙 전체 대체 여부와도 구분해야 한다.</div>"
 )

HTML = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Token Redaction Probe · 전체 결과</title><style>
:root{--bg:#f4f6f8;--panel:#fff;--ink:#162027;--muted:#62707a;--faint:#8d99a2;--line:#dde3e7;--line2:#edf0f2;--teal:#087f70;--tealbg:#e4f5f1;--blue:#486581;--bluebg:#eaf0f5;--amber:#9a5b08;--red:#b43b33;--redbg:#fbe9e7;--mono:ui-monospace,Consolas,monospace;--sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI","Noto Sans KR",sans-serif}:root[data-theme=dark]{--bg:#11171b;--panel:#192126;--ink:#edf2f4;--muted:#a5b0b7;--faint:#78858d;--line:#303a40;--line2:#253036;--teal:#52cfbb;--tealbg:#173b35;--blue:#b1c9dd;--bluebg:#23313d;--amber:#f4bd6b;--red:#f08a80;--redbg:#3c211f}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5}.wrap{max-width:1240px;margin:auto;padding:42px 22px 80px}.hero{display:flex;justify-content:space-between;gap:20px}.eyebrow{color:var(--teal);font:700 12px var(--mono);letter-spacing:.13em}.hero h1{font-size:clamp(27px,4vw,42px);line-height:1.15;margin:8px 0 10px;letter-spacing:-.035em}.hero p{max-width:820px;color:var(--muted);margin:0}.actions{display:flex;gap:8px;align-items:flex-start}.button,button{border:1px solid var(--line);color:var(--ink);background:var(--panel);border-radius:9px;padding:8px 11px;text-decoration:none;cursor:pointer;font:650 12px var(--sans)}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:24px 0}.card{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:16px}.card b{font:750 25px var(--mono);display:block}.card span,.lede{font-size:12px;color:var(--muted)}.notice{border:1px solid var(--line);border-left:4px solid var(--teal);background:var(--panel);border-radius:10px;padding:13px 15px;color:var(--muted);font-size:13px;margin:16px 0}.notice strong{color:var(--ink)}.warn{border-left-color:var(--amber)}h2{font-size:21px;margin:34px 0 6px}.lede{margin:0 0 14px}.toolbar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;background:var(--panel);border:1px solid var(--line);padding:10px;border-radius:12px 12px 0 0}.toolbar select,.toolbar input{border:1px solid var(--line);color:var(--ink);background:var(--bg);border-radius:8px;padding:7px 9px;font:12px var(--sans)}.seg{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}.seg button{border:0;border-radius:0}.seg .active{background:var(--teal);color:#fff}.tablewrap{overflow:auto;background:var(--panel);border:1px solid var(--line);border-top:0;border-radius:0 0 12px 12px}.solo{border-top:1px solid var(--line);border-radius:12px}table{width:100%;border-collapse:collapse;font-size:12px;white-space:nowrap}th{position:sticky;top:0;background:var(--panel);color:var(--muted);font-size:11px;text-align:right;padding:9px 8px;border-bottom:1px solid var(--line)}th.left,td.left{text-align:left}td{padding:7px 8px;border-bottom:1px solid var(--line2);text-align:right;font:500 12px var(--mono)}td.meta{font-family:var(--sans)}tr.start td{border-top:2px solid var(--line)}.pill,.policy,.seed{display:inline-block;border-radius:999px;padding:2px 7px;font:700 10px var(--sans)}.g-medical{background:var(--tealbg);color:var(--teal)}.g-pii{background:var(--redbg);color:var(--red)}.g-entity{background:var(--bluebg);color:var(--blue)}.policy{background:var(--bg);color:var(--muted)}.seed{padding:1px 5px;background:var(--tealbg);color:var(--teal)}.best{color:var(--teal);font-weight:800;background:color-mix(in srgb,var(--teal) 7%,transparent)}.low{color:var(--red)}.over{color:var(--amber)}.dataset{font-weight:750}.task{display:block;font-size:10px;color:var(--faint);margin-top:2px}td.merge{vertical-align:middle;text-align:center!important;border-right:1px solid var(--line);background:color-mix(in srgb,var(--panel) 94%,var(--teal) 6%)}td.merge.dataset-cell{text-align:left!important;min-width:140px}.analysis-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.analysis-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}.analysis-card h3{font-size:14px;margin:0 0 8px;color:var(--teal)}.analysis-card p{font-size:13px;color:var(--muted);margin:0}.analysis-card strong{color:var(--ink)}.macro-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.macro{border:1px solid var(--line);background:var(--panel);border-radius:12px;overflow:hidden}.macro h3{padding:13px;margin:0;border-bottom:1px solid var(--line);font-size:14px}.macro table{white-space:normal}.help-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}.help{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;font-size:12px;color:var(--muted)}.help b{display:block;color:var(--ink);font:700 13px var(--mono)}details{background:var(--panel);border:1px solid var(--line);border-radius:11px;margin-top:12px}summary{cursor:pointer;padding:13px 15px;font-weight:700;font-size:13px}details>div{padding:0 15px 15px;color:var(--muted);font-size:12px}.foot{margin-top:36px;border-top:1px solid var(--line);padding-top:14px;color:var(--faint);font-size:11px}@media(max-width:850px){.hero{display:block}.actions{margin-top:14px}.cards,.macro-grid,.help-grid,.analysis-grid{grid-template-columns:1fr 1fr}}@media(max-width:520px){.cards,.macro-grid,.help-grid,.analysis-grid{grid-template-columns:1fr}.wrap{padding:24px 12px}}
</style></head><body><main class="wrap"><div class="hero"><div><div class="eyebrow">TOKEN REDACTION PROBE · 2026-08-03</div><h1>로컬 Student Redactor — 전체 데이터 실험</h1><p>각 데이터셋의 사용 가능한 행을 빈 문장·중복 제거 후 전부 사용해, 규칙 기반 pseudo-teacher를 작은 Transformer+MLP가 얼마나 모방하는지 정리했다. 의미가 다른 Teacher 정책은 그룹별로 분리했다.</p></div><div class="actions"><a class="button" href="perturbations/">오염 규칙 12종</a><a class="button" href="redactor_results.csv">CSV · Excel용</a><button id="theme">다크 모드</button></div></div>
<section class="cards"><div class="card"><b>10</b><span>고정 조건 데이터셋</span></div><div class="card"><b>3</b><span>Student 아키텍처</span></div><div class="card"><b>30</b><span>seed 42 전체-data run</span></div><div class="card"><b>__EXAMPLE_COUNT__</b><span>전처리 후 전체 예시</span></div></section><div class="notice"><strong>주 지표:</strong> F1은 균형 일치, F2는 Recall을 더 중시한다. <strong>Token Accuracy는 클래스 불균형 때문에 메인 표에서 제외</strong>했다.</div><div class="notice warn"><strong>해석 제한:</strong> 모두 human-gold가 아닌 deterministic pseudo-gold다. QNLI·FinPhraseBank는 개인정보 탐지가 아닌 엔티티 대조 실험이며 세 그룹의 macro를 합치지 않는다.</div>
<h2>1. 고정 3모델 × 10데이터셋</h2><p class="lede">사용 가능한 전체 데이터를 학습하고, validation에서 threshold를 선택한 뒤 test에 한 번 적용한 seed 42 결과. 전체 및 Train/Val/Test 예시 수는 바로 아래 규모 표에 분리해 표시한다.</p><div class="toolbar"><div class="seg"><button class="mode active" data-mode="budget">동일 마스킹 예산</button><button class="mode" data-mode="privacy">Recall 중심 F2</button></div><select id="group"><option value="all">모든 그룹</option><option value="medical">의료 규칙</option><option value="pii">실제 PII</option><option value="entity">비개인 엔티티 대조</option></select><select id="model"><option value="all">모든 모델</option><option value="bert_tiny">BERT-tiny</option><option value="electra_small">ELECTRA-small</option><option value="distilroberta">DistilRoBERTa</option></select><input id="search" placeholder="데이터셋 검색"><span id="count" style="margin-left:auto;color:var(--faint);font-size:11px"></span></div><div class="tablewrap"><table><thead><tr><th class="left">그룹</th><th class="left">데이터셋</th><th class="left">Teacher</th><th class="left">Student</th><th>Seed</th><th>Th.</th><th>P</th><th>R</th><th>F1</th><th>F2</th><th>Teacher mask</th><th>Student mask</th><th>남은 민감</th><th>Tokens</th></tr></thead><tbody id="results"></tbody></table></div>
<h2>1-1. 전체 데이터 규모</h2><p class="lede">빈 문장·중복 제거 후 실제 사용한 예시 수. 공식 split이 있으면 보존했다.</p>__SPLIT_TABLE__
<h2>2. 그룹 내부 Macro</h2><p class="lede">의미가 같은 데이터셋끼리만 평균.</p><div class="macro-grid" id="macros"></div>
<h2>3. 효율 비교</h2><p class="lede">CPU 1-thread, batch 1, 128문장, 3회 반복.</p><div class="tablewrap solo"><table><thead><tr><th class="left">방식</th><th class="left">구현</th><th>Params</th><th>크기</th><th>Load</th><th>Median</th><th>p95</th><th>처리량</th><th>Peak RSS</th></tr></thead><tbody id="eff"></tbody></table></div><div class="notice"><strong>규칙의 F1=1은 자기 자신과 비교한 정의상 값</strong>이며 human-gold 정확도가 아니다.</div>
<h2>5. 결과 분석</h2><p class="lede">동일 마스킹 예산을 메인 기준으로 보고, Recall 중심 운용점과 효율·일반화 실험을 함께 해석했다.</p><div class="analysis-grid" id="analysis"></div><div class="notice warn"><strong>결론의 범위:</strong> 현재 결과는 Teacher 규칙을 Student가 재현하는 능력을 보여준다. 실제 개인정보를 잘 가리는지에 대한 최종 결론은 human-gold PII test와 RedactFormer 연결 후 RTM 복구 평가가 추가되어야 한다.</div><h2>6. 지표 읽는 법</h2><div class="help-grid"><div class="help"><b>Precision</b>Student 선택 중 Teacher와 일치한 비율.</div><div class="help"><b>Recall</b>Teacher 토큰 중 Student가 찾은 비율.</div><div class="help"><b>F1</b>Precision과 Recall의 균형.</div><div class="help"><b>F2</b>Recall을 더 중시한 지표.</div></div><details><summary>제외·보류 데이터셋</summary><div>biosx는 출처를 회수하지 못했고 MDCC는 원본 CSV가 없어 미실행했다. medterm4는 비의료 baseline으로 쓰지 않았다.</div></details>__APPENDIX__<footer class="foot">생성: src/build_results_dashboard.py · 원본: artifacts 평가 JSON · 서버 없이 단독 실행.</footer></main>
<script>const D=__DATA__;let mode='budget';const $=id=>document.getElementById(id),pct=v=>v==null?'—':(v*100).toFixed(2)+'%',num=(v,n=3)=>v==null?'—':Number(v).toFixed(n),int=v=>v==null?'—':Number(v).toLocaleString('ko-KR');function chosen(){let g=$('group').value,m=$('model').value,q=$('search').value.toLowerCase();return D.rows.filter(r=>(g==='all'||r.group===g)&&(m==='all'||r.model===m)&&(!q||(r.dataset_name+' '+r.task+' '+r.policy).toLowerCase().includes(q)))}function render(){let rows=chosen(),best={},groupN={},dataN={};rows.forEach(r=>{best[r.dataset]=Math.max(best[r.dataset]??-1,r[mode].f1);groupN[r.group]=(groupN[r.group]||0)+1;dataN[r.dataset]=(dataN[r.dataset]||0)+1});let seenG=new Set(),seenD=new Set();$('results').innerHTML=rows.map(r=>{let x=r[mode],firstG=!seenG.has(r.group),firstD=!seenD.has(r.dataset);seenG.add(r.group);seenD.add(r.dataset);let groupCell=firstG?`<td rowspan="${groupN[r.group]}" class="meta merge"><span class="pill g-${r.group}">${r.group_name}</span></td>`:'',datasetCell=firstD?`<td rowspan="${dataN[r.dataset]}" class="meta merge dataset-cell"><b class="dataset">${r.dataset_name}</b><span class="task">${r.task}</span></td><td rowspan="${dataN[r.dataset]}" class="meta merge"><span class="policy">${r.policy}</span></td>`:'';return `<tr class="${firstD?'start':''}">${groupCell}${datasetCell}<td class="left meta">${r.model_name}</td><td><span class="seed">${r.seed}</span></td><td>${num(x.threshold,2)}</td><td>${num(x.precision)}</td><td class="${x.recall<.7?'low':''}">${num(x.recall)}</td><td class="${x.f1===best[r.dataset]?'best':''}">${num(x.f1)}</td><td>${num(x.f2)}</td><td>${pct(x.gold_mask_rate)}</td><td class="${x.predicted_mask_rate>x.gold_mask_rate*1.2?'over':''}">${pct(x.predicted_mask_rate)}</td><td>${pct(x.residual_sensitive_rate)}</td><td>${int(x.evaluated_tokens)}</td></tr>`}).join('');$('count').textContent=rows.length+'개 run';$('macros').innerHTML=['medical','pii','entity'].map(g=>{let rs=D.macros.filter(r=>r.group===g);return `<section class="macro"><h3><span class="pill g-${g}">${rs[0].group_name}</span> · ${rs[0].datasets} datasets</h3><table><thead><tr><th class="left">Student</th><th>P</th><th>R</th><th>F1</th><th>F2</th><th>Mask</th></tr></thead><tbody>${rs.map(r=>{let x=r[mode];return `<tr><td class="left meta">${r.model_name}</td><td>${num(x.precision)}</td><td>${num(x.recall)}</td><td>${num(x.f1)}</td><td>${num(x.f2)}</td><td>${pct(x.predicted_mask_rate)}</td></tr>`}).join('')}</tbody></table></section>`}).join('')}
function renderAnalysis(){const bestMacro=(g,modeName,field)=>D.macros.filter(r=>r.group===g).sort((a,b)=>b[modeName][field]-a[modeName][field])[0],medical=bestMacro('medical','budget','f1'),pii=bestMacro('pii','budget','f1'),entity=bestMacro('entity','budget','f1'),privacy=bestMacro('pii','privacy','f2');let perDataset={};D.rows.forEach(r=>{if(!perDataset[r.dataset]||r.budget.f1>perDataset[r.dataset].budget.f1)perDataset[r.dataset]=r});let ranked=Object.values(perDataset).sort((a,b)=>a.budget.f1-b.budget.f1),hard=ranked[0],easy=ranked[ranked.length-1],fast=D.efficiency.filter(r=>r.parameters).sort((a,b)=>b.sentences_per_second-a.sentences_per_second)[0],distil=D.efficiency.find(r=>r.name==='DistilRoBERTa'),multi=D.exploratory.find(r=>r.experiment==='Multi-domain'&&r.model==='DistilRoBERTa'),lodo=D.exploratory.find(r=>r.experiment==='LODO'&&r.model==='DistilRoBERTa'),frozen=D.exploratory.find(r=>r.experiment==='Frozen baseline'),fine=D.exploratory.find(r=>r.experiment==='Encoder fine-tuning'),scale=D.exploratory.find(r=>r.experiment==='데이터 규모 확대'),large=D.exploratory.find(r=>r.experiment==='1,000개 모델 비교'&&r.model==='DistilRoBERTa');$('analysis').innerHTML=`<article class="analysis-card"><h3>그룹별 Student 성능</h3><p>동일 예산 Macro F1 최고 모델은 의료 <strong>${medical.model_name} ${num(medical.budget.f1)}</strong>, 실제 PII <strong>${pii.model_name} ${num(pii.budget.f1)}</strong>, 엔티티 대조 <strong>${entity.model_name} ${num(entity.budget.f1)}</strong>다. 세 그룹 모두 DistilRoBERTa가 평균 최고지만 정책 의미가 달라 그룹 간 점수는 합치지 않는다.</p></article><article class="analysis-card"><h3>쉬운 데이터와 어려운 데이터</h3><p>각 데이터셋의 최고 Student를 기준으로 가장 높은 F1은 <strong>${easy.dataset_name} ${num(easy.budget.f1)}</strong>, 가장 낮은 F1은 <strong>${hard.dataset_name} ${num(hard.budget.f1)}</strong>다. 낮은 점수는 단순 모델 크기뿐 아니라 문장 형태와 규칙 토큰 분포의 영향을 받는다.</p></article><article class="analysis-card"><h3>Recall 중심 운용점</h3><p>실제 PII 그룹의 최고 F2 운용은 <strong>${privacy.model_name}</strong>이며 Recall <strong>${num(privacy.privacy.recall)}</strong>, F2 <strong>${num(privacy.privacy.f2)}</strong>다. 대신 평균 Student mask가 동일예산 ${pct(pii.budget.predicted_mask_rate)}에서 ${pct(privacy.privacy.predicted_mask_rate)}로 증가하므로 privacy와 utility를 같이 봐야 한다.</p></article><article class="analysis-card"><h3>속도·크기 trade-off</h3><p><strong>${fast.name}</strong>가 ${num(fast.sentences_per_second,1)}문장/s, ${num(fast.model_state_mb,1)}MB로 가장 빠르고 작다. <strong>DistilRoBERTa</strong>는 메인 성능이 가장 높지만 ${num(distil.sentences_per_second,1)}문장/s, ${num(distil.model_state_mb,1)}MB이므로 경량 배포에는 ELECTRA/BERT와 별도 절충이 필요하다.</p></article><article class="analysis-card"><h3>학습 방식과 데이터 규모</h3><p>초기 Drug Reviews에서 Frozen BERT-tiny F1 ${num(frozen.f1)} → encoder fine-tuning ${num(fine.f1)} → 1,000개 ${num(scale.f1)}로 개선됐다. 같은 1,000개에서 DistilRoBERTa는 <strong>${num(large.f1)}</strong>까지 올라 encoder 학습·데이터 증가·모델 용량이 모두 영향을 줬다.</p></article><article class="analysis-card"><h3>도메인 일반화</h3><p>DistilRoBERTa는 4개 의료 도메인을 모두 학습한 Multi-domain Macro F1 <strong>${num(multi.f1)}</strong>이지만 target을 제외한 LODO에서는 <strong>${num(lodo.f1)}</strong>로 하락했다. 즉 공통 규칙만 배운 것이 아니라 target 도메인의 표현을 본 효과도 포함된다.</p></article>`}

function staticTables(){$('eff').innerHTML=D.efficiency.map(r=>`<tr><td class="left meta dataset">${r.name}</td><td class="left meta">${r.implementation}</td><td>${int(r.parameters)}</td><td>${r.model_state_mb==null?'외부':num(r.model_state_mb,1)+' MB'}</td><td>${num(r.load_seconds,2)} s</td><td>${num(r.median_ms,2)} ms</td><td>${num(r.p95_ms,2)} ms</td><td>${num(r.sentences_per_second,1)}/s</td><td>${num(r.peak_rss_mb,1)} MB</td></tr>`).join('')}
document.querySelectorAll('.mode').forEach(b=>b.onclick=()=>{document.querySelectorAll('.mode').forEach(x=>x.classList.remove('active'));b.classList.add('active');mode=b.dataset.mode;render()});$('group').onchange=$('model').onchange=render;$('search').oninput=render;$('theme').onclick=()=>{let dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';$('theme').textContent=dark?'라이트 모드':'다크 모드'};render();staticTables();renderAnalysis();</script></body></html>'''

def main():
 data=rows(); payload={'rows':data,'macros':macros(data),'efficiency':efficiency(),'exploratory':exploratory(),'sst2':sst2()}
 unique={row['dataset']:row for row in data}
 example_count=sum(row['examples'] or 0 for row in unique.values())
 html=HTML.replace('__DATA__',json.dumps(payload,ensure_ascii=False,separators=(',',':')))
 html=html.replace('__EXAMPLE_COUNT__',f'{example_count:,}')
 html=html.replace('__SPLIT_TABLE__',split_table(data))
 html=html.replace(
  '<h2>5. 결과 분석</h2>', future_defect_time_axis_table()
  + actual_rule_version_time_axis_table()
  + '<h2>5. 결과 분석</h2>'
 )
 html=html.replace('__APPENDIX__', robustness_table())
 OUT.mkdir(exist_ok=True); write_csv(data,OUT/'redactor_results.csv')
 (OUT/'redactor_results_dashboard.html').write_text(html,encoding='utf-8'); print(f'wrote dashboard; rows={len(data)}, csv_rows={len(data)*2}, examples={example_count:,}')
if __name__=='__main__': main()
