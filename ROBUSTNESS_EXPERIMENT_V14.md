# RedactFormer v1.4 입력 교란 강건성 실험

## 결론

전체 Drug Reviews와 학습 전용 표면 교란 증강을 사용한 ELECTRA-small은 **규칙 대체 후보의 clean 최소선**과 **조건부 표면 강건성 기준**을 3개 seed에서 통과했다. 다만 최신 규칙보다 전체 noisy F2가 낮아 **독립적인 완전 대체 모델이라고 결론 내릴 단계는 아니다**.

- 최신 MASKING_FRAMEWORK.md v1.4의 의료 구현(medterm5)으로 train 39,980문장을 다시 라벨링했다.
- clean 39,980문장에 문서화된 seen 교란 30,591문장을 더해 총 70,571행으로 학습했다.
- 동일 마스킹 예산의 held-out clean 3-seed 평균은 Precision 0.919, Recall 0.923, F1 0.921, F2 0.922다.
- 457개 미관측 교란쌍의 Student noisy F2는 0.882±0.0049이고 규칙 v1.4는 0.941이다. 절대 성능은 여전히 규칙이 높다.
- 규칙과 Student가 clean에서 모두 맞힌 같은 민감 span만 비교하면 Student 생존율 73.9%, 규칙 59.5%, 차이 +14.4%p였다. seed 42·43·44의 paired 95% CI가 모두 0보다 컸다.
- 따라서 현재 근거가 지지하는 주장은 “규칙 전체를 능가한다”가 아니라, **규칙의 표면 이음매 실패를 보완하거나 장기적으로 일부 대체할 수 있는 학습형 redactor**다.

## 초기 5k 비교에서 무엇을 사용했는가

| 항목 | Drug | BIOS |
|---|---|---|
| clean 원문 | `data/full_redactor/drug` | `data/full_redactor/bios` |
| 최신 Teacher | medterm5 v1.4 | piiclean2 v1.4 |
| 코드 식별자 | `73ee6d572fe2eda2` | `81dda225ecbead4f` |
| Student | ELECTRA-small + hidden-128 MLP token head | 동일 |
| Train / Validation / Test | 5,000 / 500 / 1,000 | 5,000 / 500 / 1,000 |
| Seed | 42 | 42 |
| 학습 | encoder와 token head 전체 fine-tuning, 5 epochs | 동일 |

RedactFormer 기준 커밋은 `39b56279c6c58fdc6732df8d5ee98868e323d344`이며, `docs/MASKING_FRAMEWORK.md`와 `docs/examples_v1.4`의 v1.4 G0 검사를 통과한 빌더를 호출했다.

대용량 산출물 `mapped_dataset_n5_medterm5`, `mapped_dataset_n5_piiclean2`, `mapped_dataset_n5_mdccunion`은 로컬 저장소에 없고 Git에도 포함되지 않는다. 따라서 이 실험은 구 라벨을 재사용하지 않고, 로컬에 있는 clean 원문을 **현재 v1.4 코드로 다시 라벨링한 최신 정책 subset**이다. 대용량 디렉터리가 제공되면 같은 파이프라인으로 전체 실험을 다시 실행해야 한다.

## 실험 설계

1. clean 문장을 현재 v1.4 규칙으로 라벨링한다.
2. clean 문자별 민감 마스크를 만든다.
3. 텍스트 결함을 결정적으로 주입하면서 문자 마스크도 같은 편집으로 이동한다.
4. 이동된 라벨을 noisy 문장의 공통 정답으로 쓴다.
5. 현재 규칙, clean-only Student, seen-noise 증강 Student를 같은 noisy 문장에서 평가한다.
6. validation에서 정한 threshold를 test에서 변경하지 않는다.

주입한 결함은 다음과 같다.

- 문서화된 계열: 이중 공백, 곱슬/C1 아포스트로피, `25 mg→25mg`, 숫자 뒤 쉼표
- 미관측 변형: 삼중 공백, NBSP, modifier apostrophe, `25-mg`, thin space, 세미콜론, 단어 내부 zero-width 문자
- Drug 711쌍, BIOS 900쌍
- Drug의 아포스트로피 17쌍과 dosage 계열 20쌍은 표본이 작아 탐색 결과로만 해석한다.

사람이 noisy 문장을 다시 라벨링하지 않은 이유는 의미와 민감 span을 바꾸지 않는 결정적 표면 편집만 사용하고, clean 라벨을 문자 좌표로 그대로 전달했기 때문이다.

## Clean held-out Student 성능

### 동일 마스킹 예산

| 데이터셋 | Threshold | Precision | Recall | F1 | F2 | Teacher mask | Student mask |
|---|---:|---:|---:|---:|---:|---:|---:|
| Drug | 0.89 | 0.867 | 0.863 | 0.865 | 0.864 | 10.36% | 10.32% |
| BIOS | 0.87 | 0.889 | 0.893 | 0.891 | 0.892 | 18.47% | 18.55% |

### Recall 중심 운용점

| 데이터셋 | Threshold | Precision | Recall | F1 | F2 | Teacher mask | Student mask |
|---|---:|---:|---:|---:|---:|---:|---:|
| Drug | 0.63 | 0.736 | 0.944 | 0.827 | 0.893 | 10.36% | 13.29% |
| BIOS | 0.53 | 0.807 | 0.956 | 0.875 | 0.922 | 18.47% | 21.89% |

Recall 중심 결과는 더 많이 가려서 얻은 값이므로 동일 예산 결과와 직접 우열을 비교하지 않는다.

## 입력 교란 결과

메인 비교는 동일 마스킹 예산 threshold를 사용한다.

| 데이터셋 | 방식 | Clean F2 | Noisy F2 | F2 하락 | Clean Recall | Noisy Recall | 신규 누출 span | Noisy mask |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Drug | 규칙 v1.4 | 1.000 | 0.952 | 0.048 | 1.000 | 0.942 | 32.21% | 9.60% |
| Drug | ELECTRA-small | 0.858 | 0.837 | 0.021 | 0.857 | 0.831 | 24.45%* | 9.79% |
| BIOS | 규칙 v1.4 | 1.000 | 0.967 | 0.033 | 1.000 | 0.959 | 26.33% | 19.79% |
| BIOS | ELECTRA-small | 0.875 | 0.855 | 0.021 | 0.873 | 0.848 | 23.70%* | 19.72% |

`신규 누출 span`의 Student 값은 clean에서 먼저 맞힌 span만 분모로 삼는다. Student는 clean부터 놓친 span이 있으므로 규칙과의 절대 개인정보 성능 비교에 단독 사용하면 안 된다.

paired sentence bootstrap 2,000회의 noisy F2 차이(Student−Rule)는 다음과 같다.

| 데이터셋 | 평균 차이 | 95% CI | 판정 |
|---|---:|---:|---|
| Drug | -0.115 | [-0.124, -0.106] | 규칙 우세 |
| BIOS | -0.112 | [-0.121, -0.104] | 규칙 우세 |

특히 C1 아포스트로피와 단어 내부 zero-width 문자는 두 방식 모두 어려웠다. 그러나 이 조건에서도 Student의 절대 F2가 규칙보다 낮아 현재 가설의 우월성 근거가 되지는 않는다.

## 전체 데이터 및 증강 ablation

메인 확장 실험은 Drug Reviews에만 수행했다. validation 500문장과 test 1,000문장은 모든 조건에서 고정했고, threshold는 validation에서 선택한 뒤 test에 한 번 적용했다. 학습에는 이중 공백·C1/곱슬 아포스트로피·25mg·숫자 뒤 쉼표 같은 seen 변형만 넣었다. 평가는 삼중 공백·NBSP·modifier apostrophe·25-mg·thin space·세미콜론·zero-width 문자로 구성된 457개 unseen 쌍만 사용했다.

| 조건(seed 42) | 실제 학습 행 | Clean P | Clean R | Clean F1 | Clean F2 | Student mask | Unseen F2 | 규칙 span 생존 | Student span 생존 | 차이 (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5k clean | 5,000 | 0.867 | 0.863 | 0.865 | 0.864 | 10.32% | 0.835 | 59.3% | 70.1% | +10.8%p [+7.0, +14.7] |
| 5k + seen-noise | 8,831 | 0.879 | 0.882 | 0.880 | 0.881 | 10.39% | 0.856 | 60.6% | 75.3% | +14.6%p [+11.1, +18.4] |
| 39,980 clean | 39,980 | 0.902 | 0.908 | 0.905 | 0.907 | 10.43% | 0.874 | 60.0% | 72.2% | +12.1%p [+8.5, +15.7] |
| 39,980 + seen-noise | 70,571 | 0.917 | 0.926 | 0.921 | 0.924 | 10.46% | 0.887 | 60.0% | 74.4% | +14.4%p [+11.0, +18.0] |

Teacher clean mask는 10.36%다. 실제 학습 행은 증강 조건에서 clean과 오염 행을 합친 값이다.

데이터 확대는 clean F2를 0.864에서 0.907로, unseen F2를 0.835에서 0.874로 올렸다. 전체 데이터에 seen-noise를 더하면 clean F2 0.924, unseen F2 0.887로 다시 개선됐다. 다만 공통 span 분모는 Student가 clean에서 맞힌 대상에 따라 조건별로 다르므로, 70.1%와 75.3% 같은 조건 간 생존율을 직접 유의성 검정한 것으로 해석하지 않는다. 각 행 내부의 Student−규칙 paired 차이가 주 검정이다.

### 전체+증강 3-seed 반복

| 지표 | 평균 | 표본 표준편차 |
|---|---:|---:|
| Clean Precision | 0.919 | 0.0021 |
| Clean Recall | 0.923 | 0.0026 |
| Clean F1 | 0.921 | 0.0004 |
| Clean F2 | 0.922 | 0.0017 |
| Unseen noisy F2 | 0.882 | 0.0049 |
| 규칙 공통-span 생존율 | 59.5% | 0.4%p |
| Student 공통-span 생존율 | 73.9% | 0.7%p |
| Student−규칙 생존율 차이 | +14.4%p | 0.5%p |

공통-span 차이의 개별 paired 95% CI는 seed 42 [+11.0,+18.0], seed 43 [+10.2,+17.8], seed 44 [+11.3,+18.6]%p로 모두 0보다 컸다.

## 합격선

Token Accuracy는 대부분이 비민감 토큰이라 높게 나오기 쉬우므로 합격 기준으로 사용하지 않는다.

### 탐색 실험 통과선

- clean Recall ≥ 0.85
- clean F2 ≥ 0.80
- 목적: 파이프라인과 학습 가능성 확인
- 현재 Drug와 BIOS 모두 통과

### 규칙 대체 후보 최소선

- clean held-out Recall ≥ 0.90
- clean held-out F2 ≥ 0.90
- clean held-out F1 ≥ 0.85
- Student와 Teacher 마스킹률 절대 차이 ≤ 1%p
- 3개 seed에서 같은 결론
- 초기 5k 동일 예산 Student는 둘 다 미통과였지만, 전체+증강 Drug 모델은 seed 42·43·44 모두 통과

F2 0.90과 Recall 0.90은 임의로 낮춰 잡은 값이 아니라, 기존 전체 Drug ELECTRA 실험이 F1 0.892, F2 0.904, Recall 0.912를 이미 달성한 수준을 최소 기준으로 삼은 것이다.

### “규칙보다 강건하다” 주장선

- 위 clean 최소선을 먼저 통과
- noisy F2가 규칙보다 높거나, 사전 정의한 비열등 마진 0.02 이내
- clean→noisy 성능 하락이 규칙보다 작음
- noisy 마스킹 예산 차이 ≤ 1%p
- paired bootstrap 95% CI가 주장과 일치
- 최소 100개 human-gold 일반 문장에서도 확인

전체+증강 모델도 절대 noisy F2 비열등성은 만족하지 못하므로 **독립적인 완전 대체 기준은 미통과**다. 반면 공통 clean-correct span의 생존율 차이는 3개 seed 모두 +5%p를 넘고 paired CI가 0보다 커 **조건부 표면 강건성/규칙 보완 기준은 통과**한다.

## 다음 실험

1. 일반 의료 문장 100~300개를 두 명이 독립 span 검수해 규칙 자체의 human-gold Precision/Recall과 Student의 실제 민감정보 성능을 확인한다.
2. 규칙과 Student의 union 또는 confidence 기반 fallback을 구성해, 규칙 단독 noisy F2 0.941을 유지하면서 zero-width 등 취약 계열의 span 생존율을 개선하는지 평가한다.
3. 같은 프로토콜을 일반 PII(piiclean2) 데이터셋 하나에 반복해 의료 용어 규칙에만 국한된 현상인지 확인한다.
4. 배포 판단 전 latency·메모리와 false-positive 비용을 포함하고, 마지막에 RedactFormer/RTM 복구 평가를 연결한다.

## 재현 경로

- 라벨러: `src/robustness/v14_rule_adapter.py`
- split 재라벨링: `src/robustness/annotate_splits.py`
- 학습용 교란 증강: `src/robustness/augment_train.py`
- 교란쌍 생성: `src/robustness/build_pairs.py`
- 평가: `src/robustness/evaluate.py`
- 초기 결과 JSON: `artifacts/robustness/v14/drug_results.json`, `artifacts/robustness/v14/bios_results.json`
- 전체·증강 결과: `artifacts/robustness/v14_augmented/drug_full_*_unseen_results.json`
- 통합 표 원본: `reports/robustness_v14_results.json`
- 전체·증강 모델: `artifacts/robustness/v14_augmented/drug_electra_small_full_aug_seed{42,43,44}`

결과 JSON과 모델은 용량 때문에 Git에서 제외될 수 있으며, 이 문서에는 논문용 핵심 수치와 정확한 정책 식별자를 보존한다.
