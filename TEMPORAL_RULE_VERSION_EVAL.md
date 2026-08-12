# 실제 규칙 버전 시간축 평가

## 질문

RedactFormer 규칙에 아직 패치가 없던 **v1.2 시점**에 학습한 Student가, 이후
v1.3·v1.4에서 실제로 추가된 표면 결함을 v1.2 규칙보다 먼저 일반화해 잡는가를
측정한다.

이 실험은 현재 대시보드의 `학습 미포함 입력 교란 평가`와 다르다. 기존 실험은
최신 v1.4 규칙에서 인공 교란을 만들었고, 이 실험은 RedactFormer Git 이력에
실제로 나중에 추가된 패치를 시간순으로 재생한다.

## 비교 방식

| 방식 | 사용할 수 있는 정보 |
|---|---|
| 과거 규칙 | RedactFormer `b8dff7e`의 v1.2 코드만 사용 |
| 과거 Student | v1.2 코드가 만든 train/validation 라벨만 사용해 학습 |
| 최신 규칙 | v1.4 패치가 적용된 참고 상한 및 target 검증에만 사용 |

Student의 모델은 기존 주 실험과 같은
`google/electra-small-discriminator + hidden-128 MLP token head`이며 encoder까지
fine-tuning한다. threshold는 v1.2 validation에서만 고른다. v1.3/v1.4 target은
학습·threshold 선택에 들어가지 않는다.

## 실제 후속 패치 유형

RedactFormer Git 이력에서 다음 변경을 사용한다.

| 버전 | Git commit | 실제 추가 결함 |
|---|---|---|
| v1.3 | `33d34c5` | 붙여 쓴 용량(`25mg`), C1 제어문자, 소유격 처리 |
| v1.3 | `11be653` | 긴 숫자열·전화·식별번호 |
| v1.4 | `0767cce` | 이메일, URL, 소셜 핸들, 숫자 날짜, ZIP+4 |

근거 파일은 RedactFormer의 `docs/MASKING_FRAMEWORK.md`,
`scripts/audit/g0_unit_check.py`, `scripts/dataset_builders/_maskcore.py`,
`scripts/dataset_builders/_pii_rules.py`다.

## 데이터와 target 생성

- Drug Reviews: 기존 held-out test 4,997문장에서 붙여 쓴 용량 후보를 찾는다.
- BIOS: 기존 held-out test 98,745문장에서 v1.3/v1.4 표면형 후보를 찾는다.
- 후보 선택은 정규식 표면형과 고정 SHA-256 순서만 사용한다. 규칙이나 Student
  예측을 보고 사례를 고르지 않는다.
- 최신 v1.4가 해당 문자 구간을 실제로 민감 처리한 후보만 평가 target으로
  인정한다.

따라서 정답은 human-gold가 아니라 **최신 규칙으로 검증한 pseudo-gold**다.
또한 원문 split은 현재 보관된 split을 재사용하고 코드와 학습 라벨만 v1.2로
되돌린 재생 실험이다. 정확히 당시 생성됐던 원본 파일을 복원한 것은 아니다.

## 주 지표

주 지표는 token F2가 아니라 `target 완전 탐지율`이다. 최신 규칙이 민감하다고
검증한 target을 구성하는 모든 word를 가렸을 때 1, 하나라도 놓치면 0이다.

- 과거 규칙 탐지율: v1.2 규칙이 미래 target을 완전히 가린 비율
- 과거 Student 탐지율: v1.2 라벨로 학습한 Student가 같은 target을 완전히 가린 비율
- 차이: `Student 탐지율 - v1.2 규칙 탐지율`
- 최신 규칙 탐지율: target 검증 정의상 100%인 참고 상한
- 95% CI: 같은 원문에서 나온 target을 묶은 source-cluster bootstrap

보조 지표로 v1.2 clean validation/test의 P·R·F1·F2와 마스킹 비율, 최신 규칙
전체 토큰에 대한 false positive를 함께 본다. Student가 target을 더 잡더라도
문장 전체를 과도하게 가리면 좋은 결과로 해석하지 않는다.

## 주장 가능한 범위

차이의 95% CI가 0보다 크고 과도한 추가 마스킹이 아니라면 다음처럼 말할 수 있다.

> 학습형 redactor는 고정 규칙에 패치가 추가되기 전에도 일부 미래 표면 변형으로
> 일반화하여 규칙 유지보수 지연을 줄일 가능성을 보였다.

이 결과만으로 최신 규칙 전체를 Student가 대체한다고 주장하지 않는다.

## 재현 코드

- 후보 생성·평가: `src/temporal_version_eval.py`
- 과거 규칙 병렬 라벨링: `src/robustness/annotate_splits_parallel.py`
- Student 학습: `src/train.py`
- threshold 및 clean 평가: `src/evaluate_medical_student.py`
