# 입력 교란 강건성 결과

- Teacher: redactformer-medterm5-rule-v1.4@73ee6d572fe2eda2
- Student threshold: 0.9000
- paired cases: 13901

## 전체 결과

| 방식 | Clean P | Clean R | Clean F1 | Clean F2 | Noisy P | Noisy R | Noisy F1 | Noisy F2 | 신규 누출 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 규칙 v1.4 | 1.000 | 1.000 | 1.000 | 1.000 | 0.998 | 0.911 | 0.952 | 0.927 | 0.472 |
| ELECTRA-small | 0.927 | 0.910 | 0.918 | 0.913 | 0.929 | 0.862 | 0.894 | 0.875 | 0.370 |

## 전체 고정 target 비교

| 분모 | 규칙 clean | 규칙 noisy | Student clean | Student noisy | Noisy 차이 | 하락폭 이점 |
|---:|---:|---:|---:|---:|---:|---:|
| 13901 pair / 4866 원문 | 1.000 | 0.528 | 0.912 | 0.582 | +0.054 | +0.142 |

Noisy 차이 95% CI: [+0.047, +0.062]. 하락폭 이점 95% CI: [+0.136, +0.148].

## 공통 clean-correct span 생존율

| 대상 | 규칙 | Student | 차이 | 95% CI |
|---|---:|---:|---:|---:|
| 전체 12684개 | 0.523 | 0.630 | +0.108 | [+0.102, +0.113] |
| dosage_hyphen (105) | 0.248 | 0.743 | +0.495 | [+0.381, +0.600] |
| dosage_thin_space (105) | 0.333 | 1.000 | +0.667 | [+0.571, +0.752] |
| modifier_apostrophe (67) | 0.881 | 0.806 | -0.075 | [-0.209, +0.060] |
| nbsp (3309) | 0.813 | 1.000 | +0.187 | [+0.174, +0.201] |
| semicolon_after_number (1147) | 0.847 | 0.995 | +0.147 | [+0.126, +0.168] |
| triple_space (3309) | 0.860 | 1.000 | +0.140 | [+0.128, +0.151] |
| zero_width_inside (4642) | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] |

## 합격 판정

- PASS final_student_quality_gate: clean F1>=0.85, F2>=0.90, Recall>=0.90
- PASS matched_budget_gate: absolute noisy mask-rate gap <= 0.01
- FAIL robustness_superiority_gate: noisy F2 higher, newly leaked span rate lower, and paired bootstrap 95% CI for delta F2 entirely above zero
- PASS absolute_target_robustness_gate: on all fixed clean-rule targets, both noisy detection advantage and smaller-drop advantage have source-cluster 95% CIs entirely above zero
- PASS conditional_surface_robustness_gate: on shared clean-correct spans, Student survival >= rule +5 percentage points and paired 95% CI entirely above zero
- PASS pilot_quality_reference: exploratory pilot only: clean F2>=0.80 and Recall>=0.85

이 평가는 깨끗한 v1.4 라벨의 의미적 정당성이 아니라, 같은 라벨을 표면 교란 뒤에도 유지하는 강건성을 측정한다.
