# 미래 결함 시간축 평가

## 질문

규칙 v1.4와 과거에 알려진 표면 결함만으로 학습한 로컬 Student가, **학습이 끝난 뒤 새로 발견된 입력 결함**에서도 현재 규칙보다 민감 span을 덜 놓치는가?

이 평가는 사람 개인정보 정답을 새로 만드는 실험이 아니다. clean 입력에서 최신 v1.4 규칙이 잡은 span을 고정 pseudo-gold로 두고, 그 span 내부 또는 경계에 새 문자를 넣은 뒤에도 같은 span을 잡는지 본다.

## 시간축 고정

| 시점 | 허용 정보 |
|---|---|
| 과거(학습) | clean v1.4 rule label과 기존 `seen` 5종: 이중 공백, 곱슬/C1 아포스트로피, 용량 붙여쓰기, 숫자 뒤 쉼표 |
| 동결 | validation에서 정한 threshold와 strict Student checkpoint를 고정 |
| 미래(test 전용) | 아래 7종. 학습 증강, validation, threshold 선택에 전혀 사용하지 않음 |

따라서 미래 교란에서 Student가 잘 잡으면, 특정 새 규칙을 코드로 추가한 결과가 아니라 과거 결함으로부터 배운 표면 일반화 결과다.

## 미래 결함 7종

| ID | 교란 | 코드포인트 | 대표 실패 상황 |
|---|---|---|---|
| F-01 | 단어 내부 zero-width non-joiner | `U+200C` | 보이지 않는 문자 하나로 이름·용어 경계가 갈라짐 |
| F-02 | 단어 내부 word joiner | `U+2060` | 복사/붙여넣기 후 화면에는 같은 단어지만 문자열이 달라짐 |
| F-03 | 단어 내부 soft hyphen | `U+00AD` | PDF/웹 줄바꿈 처리 뒤 보이지 않는 하이픈이 남음 |
| F-04 | fullwidth apostrophe | `U+FF07` | 입력기·문서 변환이 아포스트로피를 전각 문자로 바꿈 |
| F-05 | 용량 사이 non-breaking hyphen | `U+2011` | `25 mg`가 `25‑mg`처럼 바뀜 |
| F-06 | narrow no-break space | `U+202F` | 화면상 일반 공백과 거의 같은 Unicode 공백 |
| F-07 | 숫자 뒤 닫는 괄호 | `)` | 날짜·용량 숫자 토큰의 새 경계 문자 |

한 pair에는 교란 하나만 적용한다. clean label의 문자 mask를 편집에 맞춰 이동하므로 오염 문장에서 규칙이 놓쳐도 정답은 유지된다.

## 데이터셋

| 데이터셋 | 규칙 | 선택 이유 |
|---|---|---|
| Drug Reviews | `medterm5 v1.4` | 약물명·증상·용량이 풍부해 용량/Unicode 결함의 주 평가 도메인 |
| BIOS | `piiclean2 v1.4` | 인명 중심의 실제 PII 도메인; 아포스트로피·비가시 문자 결함을 확인 |
| MRPC | `piiclean2 strict v1.4` | 연락처·날짜·URL 같은 정형 PII와 숫자 경계를 확인 |

각 유형은 test split에서 적용 가능한 최대 2,000개 pair를 사용한다. Student는 기존 strict 평가에서 사용한 `ELECTRA-small + hidden-128 MLP`, seed 42/43/44이다. 별도 재학습은 하지 않는다.

## 읽을 지표와 판정

`clean→future target 탐지`는 clean에서 잡은 고정 target을 미래 교란 뒤에도 잡은 비율이다.

- **future 절대 탐지율**: 오염 뒤 실제로 가린 target의 비율. 높을수록 좋다.
- **하락폭**: clean 탐지율 − future 탐지율. 작을수록 좋다.
- **Student−규칙**: 미래 절대 탐지율 차이. 양수면 Student가 더 많이 잡았다.
- **95% CI**: 같은 원문에서 여러 pair가 나온 의존성을 고려해 source-cluster bootstrap으로 계산한다.

규칙보다 낫다고 주장하려면, 사전에 다음을 모두 만족해야 한다.

1. clean Student 품질 gate(F1 ≥ 0.85, F2 ≥ 0.90, Recall ≥ 0.90)를 통과한다.
2. 세 seed 모두에서 future 절대 탐지율 차이의 95% CI가 0보다 크다.
3. 세 seed 모두에서 Student 하락폭이 규칙 하락폭보다 작고, 그 차이의 95% CI가 0보다 크다.

이것은 **현재 raw 규칙 v1.4에 대한 보완/대체 근거**다. 새로운 generic normalizer나 새 규칙 패치를 추가한 뒤 그 규칙과 비교하는 것은 별도 강한 기준이며, 해당 기준까지 이기지 못하면 “규칙 완전 대체”라고 주장하지 않는다.

## 재현 명령

```bash
cd /home/jovyan/token_redaction_probe
python src/run_future_defect_eval.py \
  --datasets drug,bios,mrpc \
  --seeds 42,43,44 \
  --per-noise 2000 \
  --device cuda
```

출력은 `reports/future_v14_<dataset>_seed<seed>.json`이고, 생성 pair와 raw 규칙 cache는 `data/robustness/v14_future/<dataset>/`에 저장한다.
