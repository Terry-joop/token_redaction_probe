# Token Redaction Probe

RedactFormer와 분리해서 토큰 선택 모델부터 검증하는 최소 실험 프로젝트입니다.

현재 단계의 목적은 다음 흐름이 제대로 작동하는지 확인하는 것입니다.

1. GLUE SST-2 문장을 단어 단위로 나눕니다.
2. 각 단어에 `1=redact`, `0=keep` pseudo-label을 붙입니다.
3. 로컬 Transformer encoder + 작은 MLP로 token binary classifier를 학습합니다.
4. 문장을 넣어 어떤 단어를 가리는지 출력합니다.

기본 pseudo-label은 파이프라인 검증용 감성 어휘 휴리스틱입니다. 최종 연구에서는
`data/*.jsonl`을 큰 LLM이 생성한 라벨로 교체해야 합니다.

## 빠른 실행

```bash
cd /home/jovyan/token_redaction_probe

# 1) SST-2에서 임시 token label 생성
python src/make_pseudo_labels.py --train-size 1000 --validation-size 300

# 2) encoder를 freeze하고 MLP head 학습
python src/train.py --epochs 3 --batch-size 16

# 3) 직접 확인
python src/predict.py \
  --text "This movie is absolutely wonderful but far too long."
```

출력 예시는 각 단어의 redact 확률과 `[REDACTED]`가 적용된 문장입니다.

## 데이터 형식

한 줄에 JSON 하나를 저장합니다.

```json
{"id":"train-0","text":"a wonderful movie","words":["a","wonderful","movie"],"labels":[0,1,0],"task_label":1,"source":"heuristic-v1"}
```

외부 teacher LLM을 사용할 때도 이 형식만 지키면 학습 코드는 그대로 사용할 수
있습니다. 반드시 `len(words) == len(labels)`이어야 합니다.

## 중요한 제한

- 현재 라벨은 연구용 정답이 아니라 코드 검증용 weak label입니다.
- SST-2는 개인정보 데이터셋이 아닙니다. 여기서 `redact`는 “감성 판단을 드러내는
  task-important token을 선택한다”는 뜻입니다.
- 개인정보 redaction을 연구하려면 이후 PII 정책과 데이터셋을 별도로 추가해야 합니다.
- 실제 비공개 문장을 teacher API에 보내면 안 됩니다. teacher는 공개/합성 학습
  데이터의 라벨 생성에만 사용해야 합니다.


## 실제 teacher 라벨 만들기

먼저 100개 요청을 만듭니다.

```bash
python src/export_teacher_prompts.py --limit 100
```

생성된 `teacher/requests.jsonl`의 `system_prompt`와 `user_prompt`를 teacher LLM에
전달합니다. 응답은 다음 중 한 형식으로 `teacher/responses.jsonl`에 저장합니다.

```json
{"id":"train-0","model":"teacher-model-name","labels":[0,1,0]}
{"id":"train-1","model":"teacher-model-name","response":"{\"labels\":[0,1,0]}"}
```

LLM 라벨을 검증하고 학습 가능한 데이터로 합칩니다.

```bash
python src/validate_teacher_responses.py
```

잘못된 토큰 수, `0/1` 이외 값, 중복 ID, JSON 오류는
`teacher/rejected.jsonl`로 분리됩니다. 정상 데이터는
`teacher/validated.jsonl`에 저장됩니다.

validator 샘플을 시험하려면 다음을 실행합니다.

```bash
cp teacher/responses.example.jsonl teacher/responses.jsonl
python src/validate_teacher_responses.py
```

Teacher를 기준으로 기존 휴리스틱 라벨을 비교할 수 있습니다.

```bash
python src/compare_labels.py \
  --reference teacher/validated.jsonl \
  --candidate data/train.jsonl
```

Teacher 라벨이 충분히 모이면 겹치지 않는 train/validation 파일로 나눈 뒤 student를
다시 학습합니다.

```bash
python src/train.py \
  --train teacher/train.validated.jsonl \
  --validation teacher/validation.validated.jsonl \
  --epochs 8 --batch-size 32
```

이 저장소는 teacher API를 직접 호출하지 않습니다. 생성된 요청을 ChatGPT에 직접 전달하고, 받은 라벨을 JSONL로 저장한 뒤 위 validator로 검증합니다.
