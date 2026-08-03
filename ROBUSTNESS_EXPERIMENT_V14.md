# RedactFormer v1.4 입력 교란 강건성 실험

## 2026-08-03 최종 strict 10데이터셋 확장

이 문서 아래의 5k 제한·12종 pilot과 Drug-only 전체 확장 뒤에, 같은 설계를 10개
데이터셋 전체로 확장했다. 최종 실험은 데이터셋별 전체 clean train에 Seen 5종만
증강하고 전체 test에서는 Unseen 7종만 유형당 상한 없이 사용했다. ELECTRA-small과
seed 42·43·44를 고정했으며 총 352,908 target-pair를 평가했다.

엄격 우세는 오염 후 절대 탐지율이 규칙보다 높고 clean→noisy 하락도 더 작으며, 두
차이의 source-cluster bootstrap 95% CI 하한이 모두 0보다 큰 상태가 세 seed에서 재현된
경우다. 이 기준은 Drug Reviews와 FinPhraseBank 2/10개만 통과했다. FinPhraseBank는
비개인 엔티티 대조이므로 privacy 관련 데이터셋에서는 Drug Reviews 1/8개만 통과했다.
전체 표와 최신 결론은 `STRICT_ROBUSTNESS_MATRIX.md` 및 대시보드 4-3을 기준으로 한다.

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
- 별도의 동일 조건 10개 데이터셋 × 3모델 제한 비교에서는 Student의 noisy F2 우세가 0/30개, clean 대체 최소선 통과가 9/30개였다. DistilRoBERTa가 10개 데이터셋 모두에서 세 Student 중 noisy F2가 가장 높았지만 규칙에는 미치지 못했다. Drug 전체+증강의 표면 강건성 결과를 다른 도메인에 일반화하지 않는다.

## 10개 데이터셋 × 3모델 공통 비교

기존 4-1을 BERT-tiny, ELECTRA-small, DistilRoBERTa 세 Student로 확장했다.
의료 6개는 `medterm5 v1.4`, 일반 4개는 `piiclean2 v1.4`로 clean
pseudo-label을 만들고, 모든 encoder와 hidden-128 MLP token head를 함께 fine-tuning했다.

- 데이터셋별 최대 train / validation / test: 5,000 / 500 / 1,000
- 작은 split은 사용 가능한 행 전부 사용
- 5 epochs, batch 32, max length 128, seed 42
- validation에서 threshold를 선택하고 test에서는 고정
- 교란 표의 메인 운용점: Teacher와 가리는 양을 맞춘 동일 마스킹 예산
- 12종 교란별 최대 100개 paired test 생성
- Noisy P/R/F1/F2 정답: clean v1.4 문자 span을 결정적 편집을 따라 이동한 token pseudo-gold
- RedactFormer 기준 커밋: `39b56279c6c58fdc6732df8d5ee98868e323d344`
- 문서화된 교란: 이중 공백, 곱슬/C1 아포스트로피, `25 mg→25mg`, 숫자 뒤 쉼표
- 미관측 교란: 삼중 공백, NBSP, modifier apostrophe, `25-mg`, thin space, 세미콜론, 단어 내부 zero-width

대용량 `mapped_dataset_n5_medterm5/piiclean2` 산출물은 저장소에 없으므로,
10개 데이터셋 **종류 전체**를 비교했지만 대규모 데이터셋의 **모든 행**을 학습한
실험은 아니다.

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

## 10개 데이터셋 × 3모델 입력 교란 결과

| 데이터셋 | 방식 | Budget Th. | Clean F2 | Noisy P | Noisy R | Noisy F1 | Noisy F2 | F2 하락 | Noisy mask |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Drug Reviews | 규칙 v1.4 | — | 1.000 | 0.998 | 0.942 | 0.969 | 0.952 | 0.048 | 9.60% |
| Drug Reviews | BERT-tiny | 0.89 | 0.811 | 0.821 | 0.786 | 0.803 | 0.793 | 0.018 | 9.74% |
| Drug Reviews | ELECTRA-small | 0.89 | 0.858 | 0.864 | 0.831 | 0.847 | 0.837 | 0.021 | 9.79% |
| Drug Reviews | DistilRoBERTa | 0.89 | 0.886 | 0.889 | 0.853 | 0.871 | 0.860 | 0.026 | 9.76% |
| Symptom2Dx | 규칙 v1.4 | — | 1.000 | 0.999 | 0.847 | 0.917 | 0.874 | 0.126 | 15.76% |
| Symptom2Dx | BERT-tiny | 0.86 | 0.830 | 0.854 | 0.729 | 0.786 | 0.751 | 0.080 | 15.85% |
| Symptom2Dx | ELECTRA-small | 0.90 | 0.889 | 0.877 | 0.738 | 0.802 | 0.762 | 0.126 | 15.64% |
| Symptom2Dx | DistilRoBERTa | 0.78 | 0.927 | 0.917 | 0.808 | 0.859 | 0.827 | 0.100 | 16.37% |
| ADR | 규칙 v1.4 | — | 1.000 | 0.998 | 0.933 | 0.965 | 0.945 | 0.055 | 26.56% |
| ADR | BERT-tiny | 0.78 | 0.824 | 0.886 | 0.788 | 0.834 | 0.806 | 0.018 | 25.25% |
| ADR | ELECTRA-small | 0.76 | 0.881 | 0.920 | 0.844 | 0.880 | 0.858 | 0.023 | 26.06% |
| ADR | DistilRoBERTa | 0.64 | 0.896 | 0.926 | 0.854 | 0.889 | 0.868 | 0.028 | 26.19% |
| RedditMH | 규칙 v1.4 | — | 1.000 | 0.997 | 0.918 | 0.956 | 0.933 | 0.067 | 6.47% |
| RedditMH | BERT-tiny | 0.90 | 0.675 | 0.778 | 0.630 | 0.696 | 0.655 | 0.020 | 5.69% |
| RedditMH | ELECTRA-small | 0.94 | 0.765 | 0.823 | 0.718 | 0.767 | 0.737 | 0.028 | 6.13% |
| RedditMH | DistilRoBERTa | 0.88 | 0.793 | 0.842 | 0.754 | 0.796 | 0.770 | 0.023 | 6.29% |
| MedNLI | 규칙 v1.4 | — | 1.000 | 0.999 | 0.939 | 0.968 | 0.950 | 0.050 | 26.02% |
| MedNLI | BERT-tiny | 0.84 | 0.868 | 0.869 | 0.847 | 0.858 | 0.851 | 0.017 | 26.97% |
| MedNLI | ELECTRA-small | 0.80 | 0.883 | 0.905 | 0.847 | 0.875 | 0.858 | 0.025 | 25.90% |
| MedNLI | DistilRoBERTa | 0.87 | 0.904 | 0.912 | 0.866 | 0.888 | 0.875 | 0.030 | 26.26% |
| Mental Health | 규칙 v1.4 | — | 1.000 | 0.998 | 0.909 | 0.951 | 0.926 | 0.074 | 8.96% |
| Mental Health | BERT-tiny | 0.92 | 0.676 | 0.712 | 0.647 | 0.678 | 0.659 | 0.017 | 8.93% |
| Mental Health | ELECTRA-small | 0.94 | 0.729 | 0.774 | 0.697 | 0.733 | 0.711 | 0.019 | 8.86% |
| Mental Health | DistilRoBERTa | 0.93 | 0.757 | 0.792 | 0.712 | 0.750 | 0.726 | 0.031 | 8.84% |
| BIOS | 규칙 v1.4 | — | 1.000 | 0.998 | 0.959 | 0.978 | 0.967 | 0.033 | 19.79% |
| BIOS | BERT-tiny | 0.78 | 0.831 | 0.847 | 0.807 | 0.826 | 0.815 | 0.016 | 19.64% |
| BIOS | ELECTRA-small | 0.87 | 0.875 | 0.885 | 0.848 | 0.866 | 0.855 | 0.021 | 19.72% |
| BIOS | DistilRoBERTa | 0.91 | 0.911 | 0.925 | 0.880 | 0.902 | 0.888 | 0.023 | 19.59% |
| MRPC | 규칙 v1.4 | — | 1.000 | 0.998 | 0.923 | 0.959 | 0.937 | 0.063 | 14.35% |
| MRPC | BERT-tiny | 0.87 | 0.879 | 0.874 | 0.844 | 0.859 | 0.850 | 0.029 | 14.98% |
| MRPC | ELECTRA-small | 0.83 | 0.933 | 0.928 | 0.896 | 0.912 | 0.902 | 0.031 | 14.99% |
| MRPC | DistilRoBERTa | 0.93 | 0.951 | 0.957 | 0.903 | 0.929 | 0.913 | 0.038 | 14.64% |
| QNLI | 규칙 v1.4 | — | 1.000 | 0.996 | 0.938 | 0.966 | 0.949 | 0.051 | 14.99% |
| QNLI | BERT-tiny | 0.86 | 0.822 | 0.824 | 0.787 | 0.805 | 0.794 | 0.028 | 15.21% |
| QNLI | ELECTRA-small | 0.90 | 0.885 | 0.898 | 0.831 | 0.863 | 0.844 | 0.041 | 14.74% |
| QNLI | DistilRoBERTa | 0.93 | 0.922 | 0.925 | 0.893 | 0.909 | 0.899 | 0.023 | 15.38% |
| FinPhraseBank | 규칙 v1.4 | — | 1.000 | 0.998 | 0.932 | 0.964 | 0.945 | 0.055 | 27.51% |
| FinPhraseBank | BERT-tiny | 0.80 | 0.900 | 0.876 | 0.883 | 0.880 | 0.882 | 0.018 | 29.69% |
| FinPhraseBank | ELECTRA-small | 0.83 | 0.945 | 0.922 | 0.925 | 0.924 | 0.925 | 0.020 | 29.52% |
| FinPhraseBank | DistilRoBERTa | 0.76 | 0.950 | 0.947 | 0.919 | 0.933 | 0.925 | 0.025 | 28.57% |

paired source-cluster bootstrap 2,000회의 noisy F2 차이(Student−Rule)는 다음과 같다.

| 데이터셋 | Student | 평균 차이 | 95% CI | 판정 |
|---|---|---:|---:|---|
| Drug Reviews | BERT-tiny | -0.160 | [-0.178, -0.142] | 규칙 우세 |
| Drug Reviews | ELECTRA-small | -0.115 | [-0.124, -0.106] | 규칙 우세 |
| Drug Reviews | DistilRoBERTa | -0.092 | [-0.108, -0.078] | 규칙 우세 |
| Symptom2Dx | BERT-tiny | -0.124 | [-0.153, -0.094] | 규칙 우세 |
| Symptom2Dx | ELECTRA-small | -0.112 | [-0.141, -0.085] | 규칙 우세 |
| Symptom2Dx | DistilRoBERTa | -0.047 | [-0.070, -0.023] | 규칙 우세 |
| ADR | BERT-tiny | -0.140 | [-0.163, -0.118] | 규칙 우세 |
| ADR | ELECTRA-small | -0.087 | [-0.106, -0.069] | 규칙 우세 |
| ADR | DistilRoBERTa | -0.078 | [-0.098, -0.058] | 규칙 우세 |
| RedditMH | BERT-tiny | -0.276 | [-0.312, -0.245] | 규칙 우세 |
| RedditMH | ELECTRA-small | -0.195 | [-0.225, -0.167] | 규칙 우세 |
| RedditMH | DistilRoBERTa | -0.162 | [-0.191, -0.134] | 규칙 우세 |
| MedNLI | BERT-tiny | -0.099 | [-0.115, -0.084] | 규칙 우세 |
| MedNLI | ELECTRA-small | -0.092 | [-0.107, -0.076] | 규칙 우세 |
| MedNLI | DistilRoBERTa | -0.075 | [-0.091, -0.061] | 규칙 우세 |
| Mental Health | BERT-tiny | -0.267 | [-0.300, -0.236] | 규칙 우세 |
| Mental Health | ELECTRA-small | -0.215 | [-0.247, -0.185] | 규칙 우세 |
| Mental Health | DistilRoBERTa | -0.199 | [-0.235, -0.166] | 규칙 우세 |
| BIOS | BERT-tiny | -0.152 | [-0.174, -0.132] | 규칙 우세 |
| BIOS | ELECTRA-small | -0.112 | [-0.121, -0.104] | 규칙 우세 |
| BIOS | DistilRoBERTa | -0.078 | [-0.094, -0.064] | 규칙 우세 |
| MRPC | BERT-tiny | -0.087 | [-0.114, -0.061] | 규칙 우세 |
| MRPC | ELECTRA-small | -0.035 | [-0.058, -0.013] | 규칙 우세 |
| MRPC | DistilRoBERTa | -0.024 | [-0.043, -0.007] | 규칙 우세 |
| QNLI | BERT-tiny | -0.156 | [-0.178, -0.135] | 규칙 우세 |
| QNLI | ELECTRA-small | -0.106 | [-0.128, -0.086] | 규칙 우세 |
| QNLI | DistilRoBERTa | -0.050 | [-0.063, -0.038] | 규칙 우세 |
| FinPhraseBank | BERT-tiny | -0.063 | [-0.087, -0.041] | 규칙 우세 |
| FinPhraseBank | ELECTRA-small | -0.020 | [-0.037, -0.004] | 규칙 우세 |
| FinPhraseBank | DistilRoBERTa | -0.020 | [-0.037, -0.004] | 규칙 우세 |

30개 Student run 중 절대 noisy F2가 규칙보다 높은 경우는 **0개**,
동일 마스킹 예산 통과는 **26/30개**, 사전 정의 clean 대체 최소선
통과는 **9/30개**다. 모델별 결과는 각각 BERT-tiny clean 1/10·예산 8/10; ELECTRA-small clean 2/10·예산 9/10; DistilRoBERTa clean 6/10·예산 9/10이다.
DistilRoBERTa는 10개 데이터셋 모두에서 세 Student 중 noisy F2가 가장 높았다.

## 모델 크기별 결과 분석

| Student | Params | 모델 크기 | 처리량 | 의료 Clean→Noisy F2 | 일반 Clean→Noisy F2 | Clean gate |
|---|---:|---:|---:|---:|---:|---:|
| BERT-tiny | 4.4M | 17.6 MB | 264.3/s | 0.781 → 0.752 (−0.028) | 0.858 → 0.835 (−0.023) | 1/10 |
| ELECTRA-small | 13.5M | 54.1 MB | 21.7/s | 0.834 → 0.794 (−0.040) | 0.910 → 0.881 (−0.028) | 2/10 |
| DistilRoBERTa | 82.2M | 328.9 MB | 6.5/s | 0.861 → 0.821 (−0.040) | 0.934 → 0.906 (−0.027) | 6/10 |

- 모델이 커질수록 절대 noisy F2가 높아지는 순서가 **10/10개 데이터셋**에서 동일했다. 의료는 BERT 0.752 → ELECTRA 0.794 → DistilRoBERTa 0.821, 일반은 0.835 → 0.881 → 0.906였다.
- 반면 평균 F2 하락폭은 의료에서 BERT 0.028, ELECTRA 0.040, DistilRoBERTa 0.040였다. BERT의 하락이 작지만 clean F2 자체가 낮아 생긴 효과이므로 **하락폭만으로 강건성을 판정하면 안 된다**.
- 가장 큰 DistilRoBERTa도 규칙 noisy F2보다 의료 0.109, 일반 0.043 낮았다. 모델 크기 증가는 규칙 모방력을 높였지만 규칙 대체까지 만들지는 못했다.
- ELECTRA→DistilRoBERTa의 noisy F2 증가는 의료 +0.027, 일반 +0.025인 반면 파라미터는 6.1배, 모델 파일은 6.1배다. 품질 최우선이면 DistilRoBERTa, 속도·메모리까지 보면 ELECTRA-small이 현실적인 절충점이다.

## 동일 마스킹 예산과 Recall 중심 F2

| 데이터셋 | Student | 예산 Th. | 예산 P | 예산 R | 예산 F2 | 예산 mask | F2 Th. | F2 P | F2 R | F2 | F2 mask |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Drug Reviews | BERT-tiny | 0.89 | 0.825 | 0.826 | 0.826 | 10.37% | 0.72 | 0.711 | 0.918 | 0.867 | 13.38% |
| Drug Reviews | ELECTRA-small | 0.89 | 0.867 | 0.863 | 0.864 | 10.32% | 0.63 | 0.736 | 0.944 | 0.893 | 13.29% |
| Drug Reviews | DistilRoBERTa | 0.89 | 0.888 | 0.880 | 0.882 | 10.23% | 0.51 | 0.760 | 0.956 | 0.909 | 13.00% |
| Symptom2Dx | BERT-tiny | 0.86 | 0.836 | 0.844 | 0.842 | 14.02% | 0.55 | 0.663 | 0.955 | 0.878 | 20.01% |
| Symptom2Dx | ELECTRA-small | 0.90 | 0.876 | 0.913 | 0.905 | 14.47% | 0.71 | 0.829 | 0.951 | 0.924 | 15.94% |
| Symptom2Dx | DistilRoBERTa | 0.78 | 0.903 | 0.939 | 0.932 | 14.45% | 0.46 | 0.843 | 0.970 | 0.941 | 15.96% |
| ADR | BERT-tiny | 0.78 | 0.867 | 0.853 | 0.856 | 25.32% | 0.38 | 0.755 | 0.963 | 0.912 | 32.81% |
| ADR | ELECTRA-small | 0.76 | 0.906 | 0.898 | 0.900 | 25.49% | 0.35 | 0.815 | 0.965 | 0.931 | 30.48% |
| ADR | DistilRoBERTa | 0.64 | 0.907 | 0.910 | 0.909 | 25.81% | 0.14 | 0.821 | 0.970 | 0.936 | 30.40% |
| RedditMH | BERT-tiny | 0.90 | 0.720 | 0.692 | 0.697 | 5.43% | 0.73 | 0.525 | 0.858 | 0.761 | 9.23% |
| RedditMH | ELECTRA-small | 0.94 | 0.769 | 0.784 | 0.781 | 5.75% | 0.84 | 0.651 | 0.860 | 0.808 | 7.46% |
| RedditMH | DistilRoBERTa | 0.88 | 0.789 | 0.801 | 0.798 | 5.74% | 0.56 | 0.636 | 0.894 | 0.827 | 7.95% |
| MedNLI | BERT-tiny | 0.84 | 0.858 | 0.878 | 0.874 | 22.44% | 0.37 | 0.724 | 0.982 | 0.917 | 29.79% |
| MedNLI | ELECTRA-small | 0.80 | 0.897 | 0.899 | 0.899 | 22.00% | 0.21 | 0.794 | 0.976 | 0.933 | 26.98% |
| MedNLI | DistilRoBERTa | 0.87 | 0.893 | 0.913 | 0.909 | 22.45% | 0.27 | 0.814 | 0.974 | 0.937 | 26.27% |
| Mental Health | BERT-tiny | 0.92 | 0.646 | 0.669 | 0.664 | 6.25% | 0.69 | 0.438 | 0.898 | 0.742 | 12.38% |
| Mental Health | ELECTRA-small | 0.94 | 0.711 | 0.736 | 0.731 | 6.24% | 0.78 | 0.558 | 0.873 | 0.785 | 9.43% |
| Mental Health | DistilRoBERTa | 0.93 | 0.728 | 0.761 | 0.754 | 6.31% | 0.79 | 0.594 | 0.865 | 0.793 | 8.77% |
| BIOS | BERT-tiny | 0.78 | 0.843 | 0.847 | 0.846 | 18.56% | 0.51 | 0.734 | 0.940 | 0.890 | 23.64% |
| BIOS | ELECTRA-small | 0.87 | 0.889 | 0.893 | 0.892 | 18.55% | 0.53 | 0.807 | 0.956 | 0.922 | 21.89% |
| BIOS | DistilRoBERTa | 0.91 | 0.920 | 0.918 | 0.919 | 18.52% | 0.52 | 0.861 | 0.978 | 0.952 | 21.08% |
| MRPC | BERT-tiny | 0.87 | 0.850 | 0.884 | 0.877 | 10.32% | 0.67 | 0.752 | 0.951 | 0.903 | 12.55% |
| MRPC | ELECTRA-small | 0.83 | 0.921 | 0.944 | 0.940 | 10.17% | 0.18 | 0.862 | 0.972 | 0.948 | 11.17% |
| MRPC | DistilRoBERTa | 0.93 | 0.950 | 0.963 | 0.960 | 10.04% | 0.48 | 0.925 | 0.983 | 0.971 | 10.53% |
| QNLI | BERT-tiny | 0.86 | 0.820 | 0.817 | 0.817 | 10.90% | 0.62 | 0.684 | 0.922 | 0.862 | 14.75% |
| QNLI | ELECTRA-small | 0.90 | 0.880 | 0.874 | 0.875 | 10.87% | 0.44 | 0.767 | 0.940 | 0.899 | 13.41% |
| QNLI | DistilRoBERTa | 0.93 | 0.915 | 0.916 | 0.916 | 10.96% | 0.25 | 0.837 | 0.966 | 0.937 | 12.62% |
| FinPhraseBank | BERT-tiny | 0.80 | 0.845 | 0.889 | 0.880 | 23.57% | 0.44 | 0.753 | 0.974 | 0.920 | 28.99% |
| FinPhraseBank | ELECTRA-small | 0.83 | 0.895 | 0.941 | 0.931 | 23.53% | 0.30 | 0.818 | 0.986 | 0.947 | 27.00% |
| FinPhraseBank | DistilRoBERTa | 0.76 | 0.932 | 0.944 | 0.941 | 22.69% | 0.46 | 0.896 | 0.978 | 0.960 | 24.43% |

- **동일 마스킹 예산**은 Teacher와 비슷한 비율을 가리는 threshold다. 같은 privacy/utility
  비용에서 모델을 비교하므로 논문의 주 비교에 적합하다.
- **Recall 중심 F2**는 validation F2를 최대화한다. F2는 Recall 오차를 Precision보다
  네 배 크게 반영하므로 민감 토큰 누락이 더 비싼 배포 상황의 보조 운용점이다. 대신
  더 많이 가릴 수 있어 규칙과의 공정한 효율 비교로 단독 사용하면 안 된다. 실제로
  이번 결과에서는 30/30개 run 모두 Recall·F2·마스킹률이 함께 증가했다.
- 따라서 **모델 비교의 메인은 동일 예산**, 실제 privacy-first 배포 후보 선택은 F2 운용점을
  함께 제시하는 것이 적절하다.

## 왜 clean F1 0.85인가

0.85는 외부 표준이나 이론적으로 보장된 숫자가 아니다. 초기 전체 Drug ELECTRA 실험이
F1 0.892, F2 0.904, Recall 0.912를 달성한 뒤 결과를 보며 임의로 통과시키지 않도록,
그보다 낮은 **pilot screening floor**로 미리 고정한 값이다. F1 0.85는 Recall만 높이려고
과도하게 가리는 모델을 거르는 보조 안전장치이고, privacy 측면의 핵심 floor는
F2 0.90과 Recall 0.90이다.

논문에서는 ‘0.85가 보편적 합격선’이라고 쓰지 않고, 사전 정의한 내부 pilot 기준이라고
명시해야 한다. 최종 규칙 대체 판정은 이 절대값 하나가 아니라 동일 예산, noisy
비열등성/우월성의 신뢰구간, human-gold 검증을 함께 요구한다.

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
2. 규칙과 Student의 union 또는 confidence 기반 fallback을 구성해 zero-width 등 취약 계열의 span 생존율을 개선하면서 clean false positive를 통제하는지 평가한다.
3. 동일 예산 메인 결과와 함께 Recall 중심 F2 배포 운용점에서도 noisy 강건성·문장 utility를 별도로 측정한다.
4. 마지막에 RedactFormer/RTM 복구율, latency·메모리와 false-positive 비용을 연결한다.

## 재현 경로

- 라벨러: `src/robustness/v14_rule_adapter.py`
- split 재라벨링: `src/robustness/annotate_splits.py`
- 학습용 교란 증강: `src/robustness/augment_train.py`
- 교란쌍 생성: `src/robustness/build_pairs.py`
- 평가: `src/robustness/evaluate.py`
- 10개×3모델 일괄 실행: `src/run_robustness_model_matrix.py`
- 3모델 결과 JSON: `artifacts/robustness/v14/<dataset>_<model>_results.json` (기존 ELECTRA 결과는 `<dataset>_results.json`도 지원)
- 전체·증강 결과: `artifacts/robustness/v14_augmented/drug_full_*_unseen_results.json`
- 통합 표 원본: `reports/robustness_v14_results.json`
- 전체·증강 모델: `artifacts/robustness/v14_augmented/drug_electra_small_full_aug_seed{42,43,44}`

결과 JSON과 모델은 용량 때문에 Git에서 제외될 수 있으며, 이 문서에는 논문용 핵심 수치와 정확한 정책 식별자를 보존한다.
