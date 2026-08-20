# Token Redaction Probe

RedactFormer 앞단에서 민감 토큰을 고르는 **작은 로컬 Student redactor**를 실험하는
독립 프로젝트다. 기존 의료 실험은 RedactFormer의 `medterm-v4`를 사용했고,
최신 입력 교란 실험은 현재 `medterm5/piiclean2 v1.4` 규칙으로 원문을 다시
라벨링했다. 데이터셋 성격에 맞는 규칙을 Transformer+MLP Student가 얼마나 모방하는지
확인하며, 서로 의미가 다른 규칙의 결과는 하나의 macro 값으로 섞지 않는다.

최신 4-1 표면 교란 비교는 기존 메인 표의 10개 데이터셋 모두에 같은 제한
프로토콜(데이터셋별 최대 5,000/500/1,000, seed 42)을 적용하고 BERT-tiny,
ELECTRA-small, DistilRoBERTa를 비교했다.
의료 6개는 `medterm5 v1.4`, 일반 4개는 `piiclean2 v1.4`를 clean pseudo-gold로
사용했으며, 각 데이터셋에서 가능한 12종 paired 교란을 최대 유형당 100개 평가했다.
세부 P/R/F1/F2와 split·pair 수는 대시보드 4-1 및
`ROBUSTNESS_EXPERIMENT_V14.md`에 있다.

최신 strict 확장 실험은 10개 데이터셋 모두에 같은 `Seen 5종 학습 / Unseen 7종 최종 test`
분리를 적용했다. clean train 517,215행에 적용 가능한 Seen 교란 367,609행을 추가해
총 884,824행으로 데이터셋별 ELECTRA-small을 학습했고, seed 42·43·44를 반복했다.
각 전체 test에서 유형당 상한 없이 생성 가능한 Unseen target-pair 352,908개
(고유 원문 113,047개)를 평가했다. 모든 교란이 모든 문장에 적용되는 것은 아니므로 실제
pair 수와 나타난 교란 종류는 데이터셋마다 다르다.

평균 noisy 탐지와 하락폭이 모두 나은 데이터셋은 3/10개였지만, 두 차이의 원문-cluster
95% CI가 세 seed 모두 0보다 큰 엄격 우세는 Drug Reviews와 FinPhraseBank 2/10개였다.
FinPhraseBank는 비개인 엔티티 대조이므로 privacy 관련 8개 중 엄격 우세는 Drug Reviews
1/8개다. 따라서 현재 Student는 일부 표면 결함의 규칙 보완 근거를 보였지만 전체 규칙을
대체한다고 결론 낼 수 없다. 상세 표는 `STRICT_ROBUSTNESS_MATRIX.md`와 대시보드 4-3에 있다.


별도의 실제 버전 시간축 실험은 RedactFormer v1.2 코드(`b8dff7e`)로 전체 학습
split을 다시 라벨링하고, v1.2 라벨만 본 Student를 이후 v1.3/v1.4에서 Git에 실제
추가된 결함에 평가한다. 합성 교란 실험과 달리 미래 패치 정보를 학습과 threshold
선택에서 차단한다. 설계는 `TEMPORAL_RULE_VERSION_EVAL.md`, 완료 결과는
`TEMPORAL_RULE_VERSION_RESULTS.md`와 대시보드 4-5에서 확인한다.


## 결과 대시보드

지금까지의 메인·탐색 결과는 브라우저용 통합 표와 Excel 호환 CSV로 생성한다.

```bash
python src/build_results_dashboard.py
python src/build_perturbation_catalog.py
```

- 웹 표: `reports/redactor_results_dashboard.html`
- Excel/스프레드시트: `reports/redactor_results.csv`
- 외부 공유용 대시보드: <https://terry-joop.github.io/token_redaction_probe/>

`main` 브랜치에 결과가 반영되면 GitHub Actions가 대시보드와 집계 CSV/JSON을
GitHub Pages에 자동으로 배포한다. 최초 한 번은 저장소의
`Settings > Pages > Build and deployment > Source`를 `GitHub Actions`로 설정해야 한다.

HTML은 서버 없이 파일을 직접 열 수 있으며 그룹·모델·threshold 운용점 필터와 다크 모드를
지원한다. 의료·실제 PII·비개인 엔티티 macro는 서로 분리되어 있다.

## 전체 데이터 재실행(2026-07-29)

메인 비교는 고정한 세 Student(BERT-tiny, ELECTRA-small, DistilRoBERTa)를 10개
데이터셋의 사용 가능한 전체 데이터로 다시 학습한다. 빈 문장과 완전 중복을 제거하고,
공식 split이 있으면 보존했다. 공식 test label을 사용할 수 없는 경우에는 기존 정책대로
결정적·층화 80/10/10 split을 만들었다.

| 데이터셋 | 전체 | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Drug Reviews | 49,974 | 39,980 | 4,997 | 4,997 |
| Symptom2Dx | 1,060 | 844 | 108 | 108 |
| ADR | 20,892 | 16,714 | 2,089 | 2,089 |
| RedditMH | 59,607 | 47,685 | 5,961 | 5,961 |
| MedNLI | 14,021 | 11,210 | 1,395 | 1,416 |
| Mental Health | 41,878 | 33,502 | 4,188 | 4,188 |
| BIOS | 395,368 | 257,090 | 39,533 | 98,745 |
| MRPC | 5,801 | 3,668 | 408 | 1,725 |
| QNLI | 115,636 | 104,716 | 5,463 | 5,457 |
| FinPhraseBank | 2,258 | 1,806 | 226 | 226 |

학습 조건은 소규모 메인 표와 동일하게 encoder 전체 fine-tuning, 5 epoch, batch 16,
max length 256, seed 42를 사용한다. threshold는 validation에서 선택한 뒤 test에 한 번만
적용한다. Symptom2Dx는 기존 실험이 이미 사용 가능한 1,060개 전부와 같은 split을
사용했으므로 해당 결과를 재사용한다.

```bash
python src/summarize_full_dataset_results.py
python src/build_results_dashboard.py
```

완료된 전체 결과의 추적 가능한 사본은 `reports/full_dataset_results.json`, 브라우저 표는
`reports/redactor_results_dashboard.html`, Excel용 원자료는 `reports/redactor_results.csv`다.

## 먼저 읽을 문서

루트 문서는 아래 문서만 사용한다.

- `README.md`: 연구 구조와 실행 방법(현재 문서)
- `MEDICAL_REDACTOR_ALL_RESULTS.md`: 날짜·실험 순서별 전체 결과와 해석
- `ROBUSTNESS_EXPERIMENT_V14.md`: 초기 제한 실험과 Drug 전체 확장 결과·합격선
- `STRICT_ROBUSTNESS_MATRIX.md`: 10개 데이터셋 전체 strict 5/7·3-seed 최종 비교
- `PERTURBATION_CATALOG.md`: Seen/Unseen 오염 규칙 12종의 조건·예시·전체 실제 개수
- `TEMPORAL_RULE_VERSION_EVAL.md`: v1.2→v1.3/v1.4 실제 Git 시간축 실험 설계
- `TEMPORAL_RULE_VERSION_RESULTS.md`: 실제 버전 시간축 평가 결과

SST-2 사람 검수 방법만 `data/human_review/README.md`에 별도로 둔다.

## 현재 의료 실험의 구조

```text
공개 의료 문장
→ RedactFormer medterm-v4 규칙 Teacher(scispaCy + UMLS + PII NER)
→ 단어별 0/1 pseudo-label
→ 로컬 Transformer encoder + hidden 128 MLP
→ 단어별 redaction 확률
```

- `1`: 가릴 토큰
- `0`: 유지할 토큰
- 현재 의료 Student의 Teacher는 LLM이 아니다.
- 성능은 우선 Teacher mask 모방 정도인 Token Precision/Recall/F1/F2로 측정한다.
- 실제 개인정보 보호 성능을 주장하려면 별도의 human-gold 검증이 필요하다.
- task 정확도 하락, RTM 복구율, Teacher mask 일치는 서로 다른 평가다.

### 의료 Teacher 출처와 버전

- Upstream 저장소: `hwang-yundo/Redactformer`
- Upstream 파일: `scripts/dataset_builders/make_medterm_v4.py`
- 로컬 원본: `../Redactformer/scripts/dataset_builders/make_medterm_v4.py`
- Student용 word-level adapter: `src/annotate_medterm4.py`
- 사용한 규칙의 마지막 실질 변경 커밋: `f2c601e3` (2026-07-22)
- adapter 메타데이터: `redactformer-medterm-v4-word-adapter@f2c601e3`

이전 대시보드의 `medterm4-v2`는 upstream 버전이 아니라 잘못 붙인 내부 이름이었다.
RedactFormer의 `make_medterm_v2.py`는 사용하지 않았으며, 현재 표기는 모두 실제 출처인
`RedactFormer medterm-v4`로 정정했다.

## 환경

일반 데이터 처리에는 base Python을 사용한다. B200 GPU 학습과 평가는 프로젝트에 준비된
CUDA 13 환경을 사용한다.

```bash
# 일반 데이터 처리
python src/prepare_lodo_splits.py

# GPU 학습·평가
.venv-cu130/bin/python src/train.py --help
.venv-cu130/bin/python src/run_lodo_evaluation.py --help
```

의료 Teacher 의존성은 `requirements-medical.txt`에 있다. `annotate_medterm4.py`를 처음
실행하면 scispaCy/UMLS 리소스 준비에 시간이 걸릴 수 있다.

## 공통 JSONL 형식

```json
{"id":"drug-0","text":"...","words":["..."],"labels":[0,1,0],"task_label":0,"source":"redactformer-medterm-v4-word-adapter@f2c601e3"}
```

`words`와 `labels` 길이는 반드시 같아야 한다. 의료 분류의 `task_label`은 Student가
예측하는 대상이 아니며, 토큰 redaction label과 구분한다.

## 현재 의료 실험 실행 순서

### 1. Teacher 라벨과 학습 데이터

- Teacher 라벨 생성: `src/annotate_medterm4.py`
- 공통 학습 split 생성: `src/prepare_medical_student_data.py`
- 기본 의료 데이터셋 4종 생성: `src/prepare_medical_cross_datasets.py`
- MedNLI·mentalhealth 생성: `src/prepare_medical_extension_datasets.py`
- 여러 JSONL 결합: `src/combine_jsonl.py`

현재 준비된 데이터는 다음 위치에 있다.

```text
data/medical_redactor/drugreviews_1000/
data/medical_redactor/cross_dataset/{drug,symptom2dx,adr,redditmh,mednli,mentalhealth}/
data/medical_redactor/cross_dataset/multidomain/
```

### 2. Student 학습

아래는 현재 공통 설정의 예시다. `--model-name`은
`google/electra-small-discriminator` 또는 `distilroberta-base`를 사용한다.

```bash
.venv-cu130/bin/python src/train.py \
  --train data/medical_redactor/cross_dataset/multidomain/train.jsonl \
  --validation data/medical_redactor/cross_dataset/multidomain/validation.jsonl \
  --output-dir artifacts/medical_redactor/example_model \
  --model-name google/electra-small-discriminator \
  --epochs 5 --batch-size 16 --max-length 256 --seed 42 \
  --unfreeze-encoder \
  --encoder-learning-rate 2e-5 \
  --head-learning-rate 1e-3 \
  --device cuda --offline
```

### 3. 단일·cross-dataset 평가

- 단일 데이터셋: `src/evaluate_medical_student.py`
- source→target 고정 threshold: `src/evaluate_medical_cross_dataset.py`
- 규칙/Teacher 후보 비교: `src/evaluate_medical_redactors.py`

주 결과에는 항상 다음을 함께 기록한다.

- Precision, Recall, F1, F2
- Teacher와 Student mask rate
- `1 - Recall`인 residual sensitive rate
- 동일 마스킹 예산 threshold와 privacy-oriented F2 threshold

Token Accuracy는 비민감 토큰이 많아 쉽게 높아지므로 보조 지표로만 사용한다.

### 4. 핵심 결과 집계와 CPU 벤치마크

현재 본 실험은 고정한 세 모델을 여섯 데이터셋에서 동일하게 비교한다. 원래 네 데이터셋의
ELECTRA·DistilRoBERTa는 3개 seed까지 반복했고, 신규 두 데이터셋은 현재 seed 42 결과다.

```bash
# 원래 4데이터셋의 seed 42 표
python src/summarize_core_model_matrix.py

# MedNLI·mentalhealth 입력 → medterm4 라벨·split → 고정 세 모델 학습·평가
python src/prepare_medical_extension_datasets.py
python src/prepare_extension_teacher_data.py
.venv-cu130/bin/python src/run_extension_model_matrix.py

# 기존 4개와 신규 2개를 합친 seed 42의 3모델 × 6데이터셋 표
python src/summarize_six_dataset_matrix.py

# 기존 세 Student의 CPU 1-thread, batch 1 지연시간
.venv-cu130/bin/python src/benchmark_redactor_models.py

# medterm4와 Student를 각각 격리한 CPU·메모리 벤치마크
python src/benchmark_redactor_worker.py --method medterm4
.venv-cu130/bin/python src/benchmark_redactor_worker.py --method bert_tiny
.venv-cu130/bin/python src/benchmark_redactor_worker.py --method electra_small
.venv-cu130/bin/python src/benchmark_redactor_worker.py --method distilroberta

# ELECTRA/DistilRoBERTa seed 42·43·44 평균과 표준편차
python src/summarize_seed_repeats.py

# 규칙과 Student의 Precision/Recall/F1/F2·속도·메모리 통합 표
python src/summarize_rule_student_comparison.py
```

결과는 `artifacts/medical_redactor/core_matrix/`에 모인다.

### 4-1. 비의료 정책 분리 실험

`bios`와 MRPC는 PII 그룹, QNLI와 FinPhraseBank는 비개인 엔티티 대조 그룹이다.
FinPhraseBank에도 `medterm4`를 적용하지 않는다.

```bash
python src/prepare_nonmedical_rule_datasets.py \
  --datasets bios mrpc qnli finphrasebank \
  --output-root data/nonmedical_redactor --seed 42

.venv-cu130/bin/python src/run_extension_model_matrix.py \
  --datasets bios mrpc qnli finphrasebank \
  --data-root data/nonmedical_redactor \
  --output-root artifacts/nonmedical_redactor/seed42 \
  --seed 42 --batch-size 16 --device cuda
```

데이터는 `data/nonmedical_redactor/`, 평가 JSON은
`artifacts/nonmedical_redactor/seed42/summary.json`에 생성된다.

### 5. 선택적 탐색: Leave-one-domain-out

본 실험의 필수 단계가 아니라 unseen domain 일반화를 주장할 때만 사용하는 탐색 분석이다.
한 target을 완전히 제외하고 나머지 세 도메인으로 학습·threshold 선택 후 target test에
고정 적용하는 실험이다.

```bash
python src/prepare_lodo_splits.py
.venv-cu130/bin/python src/run_lodo_evaluation.py --batch-size 64 --device cuda
python src/summarize_lodo_results.py
```

결과는 `artifacts/medical_redactor/cross_dataset_lodo/summary.json`에 모인다.

## 폴더 구조

```text
data/       생성·검수 데이터
artifacts/  모델 체크포인트와 평가 JSON
prompts/    과거 LLM Teacher 프롬프트
src/        데이터 생성, 학습, 평가 코드
teacher/    과거 Teacher 입력과 소규모 예시
tests/      검증 코드
```

`data/`와 `artifacts/`는 크기가 크고 Git에서 제외될 수 있으므로 필요한 결과는 별도로
백업해야 한다.

## SST-2 예비 실험

SST-2는 파이프라인 검증에 사용한 초기 예비 실험이다. 감성 label의 근거 토큰을 가리는
실험이므로 의료 민감정보 성능의 근거로 사용하지 않는다.

```bash
# 휴리스틱 pseudo-label 생성과 기본 학습
python src/make_pseudo_labels.py --train-size 1000 --validation-size 300
python src/train.py --epochs 3 --batch-size 16

# 기존 ChatGPT annotation 병합·분할
python src/prepare_teacher_dataset.py

# SST-2 attacker와 leakage 평가
python src/train_sst2_attacker.py
python src/evaluate_leakage.py \
  --student-dir artifacts/student_teacher_v2 \
  --attacker-dir artifacts/sst2_attacker_bert_tiny \
  --mode mask
```

과거 ChatGPT Teacher 파일은 재현 가능한 API gold가 아니라 수동 생성 pseudo-label이다.
실제 사용자 개인정보를 외부 Teacher API에 보내면 로컬 redaction이라는 목적과 충돌한다.

## 현재 주의사항

- 의료 정답은 아직 human-gold가 아니라 deterministic pseudo-teacher다.
- 원래 네 데이터셋의 ELECTRA·DistilRoBERTa만 3개 seed를 반복했다. BERT-tiny와 신규
  MedNLI·mentalhealth는 현재 seed 42 한 번이다.
- 모델 간 비교는 같은 split, Teacher 버전, threshold 정책, 마스킹 예산에서 해야 한다.
- MedNLI는 `[PAIR]`로 두 문장을 연결하며, mentalhealth를 포함한 max length 256 이후 토큰은
  현재 평가에서 제외된다.
- `잘 가린다 = downstream 정확도가 떨어진다`로 해석하지 않는다.
- RedactFormer 연결 후에는 token selection, RTM 복구, downstream utility를 각각 보고한다.
