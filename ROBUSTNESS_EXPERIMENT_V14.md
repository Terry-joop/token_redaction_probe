# RedactFormer v1.4 입력 교란 강건성 실험

## 결론

전체 Drug Reviews와 학습 전용 표면 교란 증강을 사용한 ELECTRA-small은 **규칙 대체 후보의 clean 최소선**과 **조건부 표면 강건성 기준**을 3개 seed에서 통과했다. 다만 최신 규칙보다 전체 noisy F2가 낮아 **독립적인 완전 대체 모델이라고 결론 내릴 단계는 아니다**.

- 최신 MASKING_FRAMEWORK.md v1.4의 의료 구현(medterm5)으로 train 39,980문장을 다시 라벨링했다.
- clean 39,980문장에 문서화된 seen 교란 30,591문장을 더해 총 70,571행으로 학습했다.
- 전체 validation 4,997문장에서 threshold를 선택하고 전체 test 4,997문장에 적용한 clean 3-seed 평균은 Precision 0.923, Recall 0.926, F1 0.924, F2 0.925다.
- 전체 test에서 만들 수 있는 unseen target-pair는 13,901개이며 고유 원문은 4,866개다.
- 이 전체 target을 동일 분모로 두면 오염 후 정확한 span 탐지율은 Student 57.3%, 규칙 52.8%다. clean→오염 하락은 Student 33.2%p, 규칙 47.2%p다.
- 세 seed의 오염 후 Student−규칙 차이는 +3.9~+5.4%p였고, 원문 단위 cluster bootstrap 95% CI가 모두 0보다 컸다.
- 전체-token noisy F2는 Student 0.874, 규칙 v1.4 0.927로 token 단위 절대 성능은 여전히 규칙이 높다.
- 따라서 현재 근거가 지지하는 주장은 “규칙 전체를 능가한다”가 아니라, **규칙의 표면 이음매 실패를 보완하거나 장기적으로 일부 대체할 수 있는 학습형 redactor**다.
- 별도의 동일 조건 10개 데이터셋 제한 비교에서는 Student의 noisy F2 우세가 0/10개, clean 대체 최소선 통과가 2/10개였다. Drug 전체+증강의 표면 강건성 결과를 다른 도메인에 일반화하지 않는다.

## 10개 데이터셋 공통 비교

기존 4-1의 Drug Reviews·BIOS 실험을 기존 메인 표의 나머지 8개 데이터셋까지
같은 프로토콜로 확장했다. 의료 6개는 `medterm5 v1.4`, 일반 4개는
`piiclean2 v1.4`로 clean pseudo-label을 만들었다. Student는 모든 데이터셋에서
`google/electra-small-discriminator + hidden-128 MLP`를 encoder까지 fine-tuning했다.

- 데이터셋별 최대 train / validation / test: 5,000 / 500 / 1,000
- 작은 split은 사용 가능한 행 전부 사용
- 5 epochs, batch 32, max length 128, seed 42
- validation에서 동일 마스킹 예산 threshold 선택 후 test에 고정
- 12종 교란별 최대 100개 paired test 생성
- Noisy P/R/F1/F2 정답: clean v1.4 문자 span을 결정적 편집을 따라 이동한 token pseudo-gold
- RedactFormer 기준 커밋: `39b56279c6c58fdc6732df8d5ee98868e323d344`
- 문서화된 교란: 이중 공백, 곱슬/C1 아포스트로피, `25 mg→25mg`, 숫자 뒤 쉼표
- 미관측 교란: 삼중 공백, NBSP, modifier apostrophe, `25-mg`, thin space, 세미콜론, 단어 내부 zero-width

대용량 `mapped_dataset_n5_medterm5/piiclean2` 산출물은 이 저장소에 없으므로,
아래 결과는 로컬 clean 원문에서 현재 v1.4 코드로 다시 라벨링한 데이터셋별 제한본이다.
즉 10개 **종류 전체**를 비교했지만 각 대규모 데이터셋의 **모든 행**을 학습한 실험은
아니다. 작은 split은 사용 가능한 행 전부를 사용했다.

| 그룹 | 데이터셋 | Teacher | Train / Val / Test | 교란 pair | 고유 원문 |
|---|---|---|---:|---:|---:|
| 의료 규칙 | Drug Reviews | medterm5 v1.4 | 5,000 / 500 / 1,000 | 711 | 214 |
| 의료 규칙 | Symptom2Dx | medterm5 v1.4 | 844 / 108 / 108 | 268 | 101 |
| 의료 규칙 | ADR | medterm5 v1.4 | 5,000 / 500 / 1,000 | 664 | 205 |
| 의료 규칙 | RedditMH | medterm5 v1.4 | 5,000 / 500 / 1,000 | 696 | 252 |
| 의료 규칙 | MedNLI | medterm5 v1.4 | 5,000 / 500 / 1,000 | 732 | 223 |
| 의료 규칙 | Mental Health | medterm5 v1.4 | 5,000 / 500 / 1,000 | 581 | 219 |
| 일반 PII/엔티티 | BIOS | piiclean2 v1.4 | 5,000 / 500 / 1,000 | 900 | 226 |
| 일반 PII/엔티티 | MRPC | piiclean2 v1.4 | 3,668 / 408 / 1,000 | 420 | 153 |
| 일반 PII/엔티티 | QNLI | piiclean2 v1.4 | 5,000 / 500 / 1,000 | 903 | 263 |
| 일반 PII/엔티티 | FinPhraseBank | piiclean2 v1.4 | 1,806 / 226 / 226 | 600 | 150 |

## 10개 데이터셋 입력 교란 결과

메인 비교는 동일 마스킹 예산 threshold다. Clean F2와 Noisy 지표는 동일한
projected pseudo-gold 기준이며, human-gold 개인정보 정확도가 아니다.

| 데이터셋 | 방식 | Clean F2 | Noisy P | Noisy R | Noisy F1 | Noisy F2 | F2 하락 | Noisy mask |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Drug Reviews | 규칙 v1.4 | 1.000 | 0.998 | 0.942 | 0.969 | 0.952 | 0.048 | 9.60% |
| Drug Reviews | ELECTRA-small | 0.858 | 0.864 | 0.831 | 0.847 | 0.837 | 0.021 | 9.79% |
| Symptom2Dx | 규칙 v1.4 | 1.000 | 0.999 | 0.847 | 0.917 | 0.874 | 0.126 | 15.76% |
| Symptom2Dx | ELECTRA-small | 0.889 | 0.877 | 0.738 | 0.802 | 0.762 | 0.126 | 15.64% |
| ADR | 규칙 v1.4 | 1.000 | 0.998 | 0.933 | 0.965 | 0.945 | 0.055 | 26.56% |
| ADR | ELECTRA-small | 0.881 | 0.920 | 0.844 | 0.880 | 0.858 | 0.023 | 26.06% |
| RedditMH | 규칙 v1.4 | 1.000 | 0.997 | 0.918 | 0.956 | 0.933 | 0.067 | 6.47% |
| RedditMH | ELECTRA-small | 0.765 | 0.823 | 0.718 | 0.767 | 0.737 | 0.028 | 6.13% |
| MedNLI | 규칙 v1.4 | 1.000 | 0.999 | 0.939 | 0.968 | 0.950 | 0.050 | 26.02% |
| MedNLI | ELECTRA-small | 0.883 | 0.905 | 0.847 | 0.875 | 0.858 | 0.025 | 25.90% |
| Mental Health | 규칙 v1.4 | 1.000 | 0.998 | 0.909 | 0.951 | 0.926 | 0.074 | 8.96% |
| Mental Health | ELECTRA-small | 0.729 | 0.774 | 0.697 | 0.733 | 0.711 | 0.019 | 8.86% |
| BIOS | 규칙 v1.4 | 1.000 | 0.998 | 0.959 | 0.978 | 0.967 | 0.033 | 19.79% |
| BIOS | ELECTRA-small | 0.875 | 0.885 | 0.848 | 0.866 | 0.855 | 0.021 | 19.72% |
| MRPC | 규칙 v1.4 | 1.000 | 0.998 | 0.923 | 0.959 | 0.937 | 0.063 | 14.35% |
| MRPC | ELECTRA-small | 0.933 | 0.928 | 0.896 | 0.912 | 0.902 | 0.031 | 14.99% |
| QNLI | 규칙 v1.4 | 1.000 | 0.996 | 0.938 | 0.966 | 0.949 | 0.051 | 14.99% |
| QNLI | ELECTRA-small | 0.885 | 0.898 | 0.831 | 0.863 | 0.844 | 0.041 | 14.74% |
| FinPhraseBank | 규칙 v1.4 | 1.000 | 0.998 | 0.932 | 0.964 | 0.945 | 0.055 | 27.51% |
| FinPhraseBank | ELECTRA-small | 0.945 | 0.922 | 0.925 | 0.924 | 0.925 | 0.020 | 29.52% |

paired source-cluster bootstrap 2,000회의 noisy F2 차이(Student−Rule)는 다음과 같다.

| 데이터셋 | 평균 차이 | 95% CI | 판정 |
|---|---:|---:|---|
| Drug Reviews | -0.115 | [-0.124, -0.106] | 규칙 우세 |
| Symptom2Dx | -0.112 | [-0.141, -0.085] | 규칙 우세 |
| ADR | -0.087 | [-0.106, -0.069] | 규칙 우세 |
| RedditMH | -0.195 | [-0.225, -0.167] | 규칙 우세 |
| MedNLI | -0.092 | [-0.107, -0.076] | 규칙 우세 |
| Mental Health | -0.215 | [-0.247, -0.185] | 규칙 우세 |
| BIOS | -0.112 | [-0.121, -0.104] | 규칙 우세 |
| MRPC | -0.035 | [-0.058, -0.013] | 규칙 우세 |
| QNLI | -0.106 | [-0.128, -0.086] | 규칙 우세 |
| FinPhraseBank | -0.020 | [-0.037, -0.004] | 규칙 우세 |

10개 중 Student의 절대 noisy F2가 규칙보다 높은 데이터셋은 **0개**다.
동일 마스킹 예산은 **9/10개**, 사전 정의 clean 대체 최소선
(F1/F2/Recall ≥ 0.85/0.90/0.90)은 **2/10개**가 통과했다.
따라서 이 제한 실험만으로 규칙 대체 성공을 주장할 수 없고, 데이터셋별 취약성과
표면 결함 보완 가능성을 확인하는 비교 결과로 해석한다. 특히 FinPhraseBank는 noisy
마스킹률 차이가 1%p를 넘어 동일 예산 직접 비교에 주의한다.

## 전체 데이터 및 증강 ablation

최종 확장 실험은 Drug Reviews 1개 데이터셋에 수행했다. 원본 49,974문장을
train 39,980 / validation 4,997 / test 4,997로 나누고, clean train에 seen-noise
30,591행을 추가해 총 70,571행으로 학습했다. threshold는 전체 validation에서
선택한 뒤 전체 test에 한 번 적용했다.

전체 test의 모든 문장을 훑어 unseen 변형을 적용할 수 있는 경우를 제한 없이 생성했다.
그 결과 target-pair는 13,901개, 고유 원문은 4,866개였다. 한 원문에 여러 오염이
적용될 수 있으므로 pair 수가 고유 원문 수보다 크다. 신뢰구간은 pair를 독립 표본으로
취급하지 않고 같은 source_id의 모든 변형을 묶는 원문-cluster bootstrap 2,000회로
계산했다.

### 메인: clean 규칙 전체 target 13,901개 동일 분모

| 방식 | 분모 | Clean 정확 span 탐지율 | 오염 후 정확 span 탐지율 | Clean→오염 하락 | 오염 후 Student−규칙 |
|---|---:|---:|---:|---:|---:|
| 규칙 v1.4 | 13,901 | 100.0% | 52.8% | −47.2%p | — |
| 전체+증강 Student, 3-seed 평균 | 13,901 | 90.5%±0.8%p | **57.3%±0.8%p** | −33.2%±0.2%p | **+4.6%±0.8%p** |

정확한 target span 전체를 가렸을 때만 성공으로 계산했다. Student는 규칙보다
14.0%p 덜 하락했고, 오염 후 절대 target 탐지율도 4.6%p 높았다. Student가 clean에서
놓친 target도 분모에서 제외하지 않았다.

### Seed별 원문-cluster 신뢰구간

| Seed | Student clean | Student noisy | Noisy Student−Rule (95% CI) | 하락폭 이점 (95% CI) |
|---:|---:|---:|---:|---:|
| 42 | 90.6% | 57.2% | +4.4%p [+3.6, +5.2] | +13.8%p [+13.2, +14.4] |
| 43 | 89.7% | 56.6% | +3.9%p [+3.1, +4.6] | +14.1%p [+13.5, +14.7] |
| 44 | 91.2% | 58.2% | +5.4%p [+4.7, +6.2] | +14.2%p [+13.6, +14.8] |

세 seed 모두 오염 후 절대 탐지율 차이의 95% CI가 0보다 컸다. 따라서 이 데이터와
주입한 표면 결함 범위에서는 Student가 규칙보다 덜 흔들린다는 근거가 있다.

### 전체 clean 및 noisy token 성능

| 지표 | Student 3-seed 평균 | 규칙 |
|---|---:|---:|
| 전체 clean Precision | 0.923 | — |
| 전체 clean Recall | 0.926 | 1.000* |
| 전체 clean F1 | 0.924 | 1.000* |
| 전체 clean F2 | 0.925 | 1.000* |
| Unseen noisy token F2 | 0.874 | 0.927 |

규칙의 clean 1.000은 규칙이 만든 pseudo-gold와 규칙 자신을 비교한 정의상 값이지
human-gold 개인정보 정확도가 아니다. Student가 target-span 강건성에서는 앞섰지만
전체 noisy token F2는 규칙보다 낮으므로 완전 대체 우월성은 성립하지 않는다.

### 이전 제한본 ablation

최종 조건을 고르는 과정에서는 validation 500 / test 1,000 및 unseen 457-pair 제한본을
사용했다. 이 표는 데이터 확대와 증강의 방향성을 비교한 참고 ablation이며, 최종 수치의
분모는 위 13,901-pair 전체 평가다.

| 조건(seed 42) | 학습 행 | Clean F2 | Unseen F2 | 조건부 Student−규칙 생존 차이 |
|---|---:|---:|---:|---:|
| 5k clean | 5,000 | 0.864 | 0.835 | +10.8%p |
| 5k + seen-noise | 8,831 | 0.881 | 0.856 | +14.6%p |
| 39,980 clean | 39,980 | 0.907 | 0.874 | +12.1%p |
| 39,980 + seen-noise | 70,571 | 0.924 | 0.887 | +14.4%p |

데이터 확대와 seen-noise 증강이 모두 성능을 높였기 때문에 마지막 조건을 최종 모델로
선택했다. 하지만 제한본의 74.4% 같은 조건부 생존율을 전체 결과 57.3%와 직접 비교하면
안 된다. 전자는 Student도 clean에서 맞힌 410개만 분모이고, 후자는 전체 clean-rule
target 13,901개를 고정 분모로 사용한다.

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
