# 미래 결함 시간축 평가 결과

학습이 끝난 뒤 새로 발견됐다고 가정한 7개 표면 교란을 test 전용으로 두었다. clean 최신 v1.4 규칙 span을 고정 정답으로 이동했으며, Student는 기존 strict seen-5 checkpoint를 그대로 사용했다. 즉 미래 교란은 학습·validation·threshold 선택에 들어가지 않았다.

- 평가: 3개 데이터셋, 26,382 future target-pair, ELECTRA-small 3 seed(42/43/44).
- 비교 대상: 현재 RedactFormer raw 규칙 v1.4. 새 normalizer나 사후 규칙 패치는 이 비교에 넣지 않았다.
- 판정: clean quality gate와 미래 절대 탐지 우세·하락폭 우세의 source-cluster 95% CI를 세 seed 모두 통과해야 한다.

## 데이터셋별 결과

| 데이터셋 | 도메인 | Pair | Clean F2 | 규칙 미래 탐지 | Student 미래 탐지 | 차이 | 규칙 하락 | Student 하락 | 하락폭 이점 | 3-seed 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Drug Reviews | 의료: 약물·용량·증상 | 9,519 | 0.910 | 63.8% | 69.8% | +6.0%p | 36.2% | 22.5% | +13.6%p | 우세 |
| BIOS | 실제 PII: 인명 | 12,005 | 0.926 | 57.8% | 70.7% | +12.8%p | 42.2% | 21.9% | +20.3%p | 우세 |
| MRPC | 실제 PII: 날짜·연락처·URL | 4,858 | 0.936 | 59.4% | 73.9% | +14.5%p | 40.6% | 20.3% | +20.3%p | 우세 |

## 해석

세 데이터셋 모두 clean quality gate와 미래 target 탐지의 엄격 우세 조건을 세 seed에서 통과했다. 따라서 **현재 raw 규칙 v1.4가 아직 알지 못한 표면 결함**에서는, 과거 결함으로 증강한 Student가 규칙보다 더 많은 고정 민감 target을 보존했고 clean→future 하락도 더 작았다.

하지만 전체 token F2까지 규칙을 이겼는지는 별도다. 특히 Drug Reviews와 BIOS에서 Student는 주입 target은 더 잘 보존했지만 token precision이 낮아 전체 noisy F2는 규칙보다 낮다. 즉 이 결과는 ‘완전 대체’가 아니라 **규칙 업데이트 전 새 표면 결함을 덜 놓치는 local fallback/병렬 보완기**의 근거다.

## 전체 token 품질과 마스킹 예산

마스킹 비율도 함께 봐야 한다. Student의 target 우세가 단순히 훨씬 더 많이 가려서 생긴 것은 아니다. 다만 규칙의 매우 높은 precision을 완전히 재현하지는 못했으므로, 배포는 `규칙 OR Student`의 즉시 교체가 아니라 Student 양성 후보를 재검사하거나, 미래 결함 계열에 한정한 fallback으로 시작하는 편이 타당하다.

| 데이터셋 | 규칙 mask | Student mask | 규칙 P / R / F2 | Student P / R / F2 |
|---|---:|---:|---|---|
| Drug Reviews | 9.0% | 9.2% | 0.997 / 0.913 / 0.929 | 0.931 / 0.873 / 0.884 |
| BIOS | 19.2% | 20.0% | 0.997 / 0.930 / 0.943 | 0.933 / 0.905 / 0.911 |
| MRPC | 12.3% | 13.3% | 0.996 / 0.871 / 0.894 | 0.946 / 0.895 / 0.905 |


## 교란 종류별 공통 span 생존율

공통 clean-correct span만 분모로 하므로, 이 표는 clean에서 이미 한 쪽이 놓친 span 때문에 생기는 착시를 제거한 보조 분석이다. 너무 적은 적용 사례(예: 특정 데이터셋의 용량/아포스트로피)는 전체 결론보다 사례 분석으로만 해석한다.

| 미래 교란 | 공통 span(3데이터셋×3 seed) | 규칙 생존 | Student 생존 | 차이 |
|---|---:|---:|---:|---:|
| dosage_nb_hyphen | 315 | 0.0% | 0.0% | +0.0%p |
| fullwidth_apostrophe | 5,842 | 0.0% | 12.6% | +12.6%p |
| narrow_nbsp | 12,923 | 88.3% | 100.0% | +11.7%p |
| right_paren_after_number | 9,246 | 91.6% | 98.2% | +6.5%p |
| soft_hyphen_inside | 15,036 | 54.5% | 71.4% | +16.9%p |
| word_joiner_inside | 15,036 | 57.9% | 71.4% | +13.6%p |
| zwnj_inside | 15,036 | 55.7% | 71.4% | +15.7%p |

## 재현 파일

- 설계와 사전 판정: `FUTURE_DEFECT_TIME_AXIS.md`
- 실행기: `src/run_future_defect_eval.py`
- raw 결과: `reports/future_v14_<dataset>_seed<seed>.json`
