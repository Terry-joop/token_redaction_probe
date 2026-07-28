# Token Redaction Probe

RedactFormer 앞단에서 민감 토큰을 고르는 **작은 로컬 Student redactor**를 실험하는
독립 프로젝트다. 현재 주 실험은 의료 문장에서 RedactFormer의 규칙 기반 `medterm4`
선택을 Transformer+MLP Student가 얼마나 모방하는지 확인하는 것이다.

## 먼저 읽을 문서

루트 문서는 두 개만 사용한다.

- `README.md`: 연구 구조와 실행 방법(현재 문서)
- `MEDICAL_REDACTOR_ALL_RESULTS.md`: 날짜·실험 순서별 전체 결과와 해석

SST-2 사람 검수 방법만 `data/human_review/README.md`에 별도로 둔다.

## 현재 의료 실험의 구조

```text
공개 의료 문장
→ medterm4 규칙 Teacher(scispaCy + UMLS + PII NER)
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
{"id":"drug-0","text":"...","words":["..."],"labels":[0,1,0],"task_label":0,"source":"medterm4-reimplementation-v2-latest-aligned"}
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
