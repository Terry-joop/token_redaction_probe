# 의료 민감정보 Student Redactor 전체 실험 결과

> 최종 갱신: 2026-07-28
> 이 문서 하나에 BERT-tiny 파일럿부터 multi-domain ELECTRA/DistilRoBERTa까지의
> 결과와 leave-one-domain-out 일반화 실험을 날짜·순서별로 통합했다. 이 문서가
> 의료 Student 실험 결과의 단일 정본이다.

## 1. 연구 질문과 결론

연구 질문은 다음과 같다.

> RedactFormer가 사용하던 규칙 기반 `medterm4`의 의료 민감 토큰 선택을 작은 로컬
> Student가 모방할 수 있는가? 그리고 하나의 Student가 서로 다른 의료 데이터셋에서도
> 동작할 수 있는가?

현재까지의 결론은 다음과 같다.

1. BERT-tiny encoder까지 fine-tuning하면 frozen encoder보다 좋아지지만 성능은 부족했다.
2. 학습 문장을 160개에서 800개로 늘리자 BERT-tiny F1이 0.590→0.646으로 상승했다.
3. 같은 800문장에서 ELECTRA-small은 F1 0.737, DistilRoBERTa는 0.776을 얻었다.
4. Drug Reviews에서만 학습한 모델은 ADR과 redditmh에서 domain gap을 보였다.
5. 네 의료 데이터셋을 합친 multi-domain Student가 데이터셋별 별도 모델보다 모든
   test에서 높은 F1을 얻었다.
6. 그러나 평가 도메인을 통째로 제외한 LODO에서는 Macro F1이 ELECTRA 0.705,
   DistilRoBERTa 0.729로 하락했다. 현재 모델은 처음 보는 도메인에 완전히 일반화하지 않는다.
7. 고정 3모델×4데이터셋 및 3-seed 결과에서도 **성능 우선=DistilRoBERTa**, **크기·속도 우선=
   ELECTRA-small**이다.

## 2. 무엇을 정답으로 사용했는가

최종 의료 실험의 Teacher는 LLM이 아니다. 최신 RedactFormer `medterm4`와 정합화한
결정적 규칙을 사용했다.

```text
scispaCy 의료 NER
→ UMLS entity linking
→ 민감 semantic type 선택
→ spaCy PII NER 결합
→ 임상 상투어·기능어 제외
→ 단어별 0/1 redaction label
```

- UMLS 민감 유형에는 질환, 증상, 약물, 시술, 검사, 신체, 세균 `T007` 등이 포함된다.
- PII는 PERSON, ORG, GPE, LOC, FAC, NORP, MONEY, PERCENT와 특정 연도 DATE를 포함한다.
- 최신 `CLINICAL_STOP` 규칙을 적용했다.
- GPT/Gemini annotation은 최종 의료 Student 학습에 사용하지 않았다.
- 이 정답은 human-gold가 아니라 **deterministic pseudo-teacher label**이다.

따라서 아래 F1은 “실제 민감정보를 완벽히 찾는 정도”가 아니라 우선적으로
“Student가 medterm4의 토큰 선택을 모방하는 정도”다.

## 3. 평가 기준

- Precision: Student가 가린 토큰 중 Teacher도 가린 비율
- Recall: Teacher가 가린 토큰 중 Student가 찾아낸 비율
- F1: Precision과 Recall의 균형
- F2: 민감 토큰 누락을 더 크게 벌점 주는 지표
- Teacher/Student mask: 전체 토큰 중 가린 비율
- 남은 민감 토큰: `1 - Recall`

Token accuracy는 비민감 토큰이 많아 쉽게 높아지므로 보조 지표로만 사용했다.

두 threshold 조건을 구분했다.

- 동일 마스킹 예산: validation에서 Student mask를 Teacher mask와 가깝게 맞춤
- F2 최적: validation F2를 최대화해 privacy Recall을 우선함

## 4. 공통 Student 구조

```text
원문 단어
→ Transformer encoder
→ 토큰별 문맥 벡터
→ hidden 128 MLP head
→ 토큰별 민감/비민감 확률
```

주요 공통 설정은 encoder와 MLP 전체 fine-tuning, 5 epochs, batch 16,
max length 256, encoder LR 2e-5, head LR 1e-3, seed 42다.

### 실험 타임라인

| 날짜 | 단계 | 한 번에 바꾼 핵심 변수 | 목적 |
|---|---|---|---|
| 2026-07-23 | 초기 파일럿 | Teacher 후보와 frozen BERT-tiny | 파이프라인 동작 확인 |
| 2026-07-26 | 실험 A | encoder frozen → fine-tuning | MLP-only 병목 확인 |
| 2026-07-26 | 실험 B | Drug Reviews 200 → 1,000개 | 데이터 규모 효과 확인 |
| 2026-07-26~27 | 실험 C | BERT-tiny → ELECTRA → DistilRoBERTa | 모델 용량 비교 |
| 2026-07-27 | 실험 D | 의료 데이터셋 1종 → 4종 | 도메인 차이 준비 |
| 2026-07-27 | 실험 E | zero-shot/in-domain/multi-domain | 학습 구성 비교 |
| 2026-07-27 | 실험 F | 한 도메인을 통째로 제외 | unseen domain 일반화 평가 |
| 2026-07-28 | 실험 G | MedNLI·mentalhealth 추가 | 문장쌍·정신건강 장문으로 데이터 다양화 |

## 5. 본 실험 (2026-07-27~28): 고정 3모델 × 6데이터셋

앞선 파일럿에서 정한 구조와 학습 설정을 고정하고, 데이터셋만 바꾸어 세 Student를 동일하게
비교했다. 2026-07-28에 임상 문장쌍 MedNLI와 정신건강 장문 mentalhealth를 추가했다.
이 절이 현재 논문의 중심 결과이며 이후 zero-shot, multi-domain, LODO는 탐색 분석으로 구분한다.

### 고정 조건

| 데이터셋 | Train/Validation/Test | BERT-tiny | ELECTRA-small | DistilRoBERTa |
|---|---:|---:|---:|---:|
| drug | 800/100/100 | 학습 | 학습 | 학습 |
| symptom2dx | 844/108/108 | 학습 | 학습 | 학습 |
| ADR | 800/100/100 | 학습 | 학습 | 학습 |
| redditmh | 800/100/100 | 학습 | 학습 | 학습 |
| MedNLI | 800/100/100 | 학습 | 학습 | 학습 |
| mentalhealth | 800/100/100 | 학습 | 학습 | 학습 |

- 모든 모델은 같은 데이터셋 안에서 동일한 split과 medterm4 Teacher를 사용했다.
- encoder와 hidden 128 MLP를 모두 fine-tuning했다.
- epoch, batch, max length, learning rate와 threshold 선택 정책을 고정했다.
- threshold는 각 validation에서 선택하고 해당 고정 test에 적용했다.

### Seed 42 동일 마스킹 예산 F1

| 데이터셋 | BERT-tiny | ELECTRA-small | DistilRoBERTa |
|---|---:|---:|---:|
| drug | 0.674 | 0.786 | **0.795** |
| symptom2dx | 0.778 | 0.890 | **0.906** |
| ADR | 0.704 | 0.728 | **0.778** |
| redditmh | 0.521 | 0.649 | **0.708** |
| MedNLI | 0.674 | 0.713 | **0.762** |
| mentalhealth | 0.446 | 0.564 | **0.679** |
| Macro F1 | 0.633 | 0.722 | **0.771** |

동일한 여섯 데이터셋 모두에서 DistilRoBERTa가 가장 높았고 ELECTRA-small이 그다음이었다.
Mentalhealth가 세 모델 모두 가장 어려웠으며, 기존 네 데이터셋만 계산한 Macro F1
0.669/0.763/0.797에서 0.633/0.722/0.771로 하락했다. 여섯 데이터셋 privacy-oriented
Macro F2는 BERT 0.733, ELECTRA 0.803, DistilRoBERTa 0.827이었다.

### 신규 데이터셋 상세 결과 (seed 42)

| 데이터셋 | Student | Precision | Recall | F1 | F2 | 예측 마스킹률 | Privacy Recall | Privacy F2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MedNLI | BERT-tiny | 0.681 | 0.668 | 0.674 | 0.671 | 13.65% | 0.929 | 0.786 |
| MedNLI | ELECTRA-small | 0.727 | 0.699 | 0.713 | 0.704 | 13.36% | **0.936** | **0.836** |
| MedNLI | DistilRoBERTa | **0.782** | **0.742** | **0.762** | **0.750** | 13.19% | 0.901 | 0.833 |
| mentalhealth | BERT-tiny | 0.472 | 0.422 | 0.446 | 0.431 | 3.33% | 0.753 | 0.588 |
| mentalhealth | ELECTRA-small | 0.609 | 0.525 | 0.564 | 0.540 | 3.21% | **0.853** | 0.660 |
| mentalhealth | DistilRoBERTa | **0.620** | **0.750** | **0.679** | **0.720** | 4.51% | 0.797 | **0.742** |

MedNLI test Teacher 마스킹률은 13.90%, mentalhealth는 3.73%다. MedNLI는 공식
train/dev/test에서 각각 표본을 뽑았고, premise와 hypothesis를 `[PAIR]`로 연결해 양쪽을
모두 라벨링했다. 경계 토큰이 민감하다고 잘못 선택된 예는 0개였다. Mentalhealth는 원본
train을 중복 제거·라벨 층화한 뒤 800/100/100으로 분리했다. 두 데이터셋의 NLI·정신상태
task label은 Student의 학습 대상이 아니며, Student는 medterm4 토큰 위치만 학습했다.

Privacy threshold는 Recall을 높이는 대신 더 많이 가린다. 예를 들어 MedNLI BERT의
privacy 마스킹률은 26.52%이므로 동일 예산 결과와 직접 우열을 비교하면 안 된다.
Mentalhealth는 민감 토큰 자체가 3.7%로 희소하고 표현이 긴 문맥에 흩어져 있어 세 모델
중 가장 어려웠다.

### 원래 4데이터셋 핵심 모델 3-seed 반복

Train/validation/test는 고정하고 학습 난수만 42, 43, 44로 바꿨다. 각 모델은 자신의
validation에서 threshold를 선택한 뒤 같은 test에 적용했다.

| 데이터셋 | ELECTRA-small F1 | DistilRoBERTa F1 |
|---|---:|---:|
| drug | 0.786 ± 0.003 | **0.797 ± 0.002** |
| symptom2dx | 0.894 ± 0.005 | **0.915 ± 0.009** |
| ADR | 0.736 ± 0.007 | **0.766 ± 0.015** |
| redditmh | 0.645 ± 0.005 | **0.715 ± 0.011** |
| Macro F1 | 0.765 ± 0.002 | **0.798 ± 0.005** |

Macro privacy F2는 ELECTRA 0.829 ± 0.003, DistilRoBERTa 0.847 ± 0.003이었다. seed에 따른
변동은 작았고 모델 순위도 유지되어 한 번의 우연한 학습 결과로 보기는 어렵다.

### 원래 4데이터셋: 규칙 기반 medterm4와 Student의 통합 비교

#### 비교 기준

- 정확도: 각 데이터셋의 `medterm4` 출력을 pseudo-gold로 놓은 token-level 평가다.
- 동일 마스킹 예산: validation에서 Student 마스킹률이 Teacher와 비슷해지는 threshold를
  정한 뒤 test에 고정 적용했다.
- BERT-tiny는 seed 42 한 번, ELECTRA-small과 DistilRoBERTa는 seed 42·43·44 평균이다.
- 효율: CPU 1 thread, batch 1, max length 256, 네 test에서 균등 추출한 128문장을 3회
  반복했다. 토큰화·규칙 처리 또는 모델 forward를 모두 포함한다.

`medterm4`의 Precision/Recall/F1/F2가 1.0인 이유는 **자기 출력을 정답으로 두고 자기 자신과
비교했기 때문**이다. 이는 human-gold에 대한 민감정보 탐지 정확도가 100%라는 뜻이 아니다.

#### 네 데이터셋 Macro 정확도와 마스킹률

| 방식 | Precision | Recall | F1 | F2 | 마스킹률 | 정확도 기준 |
|---|---:|---:|---:|---:|---:|---|
| medterm4 | 1.000 | 1.000 | 1.000 | 1.000 | 10.28% | 자기 규칙 재현 기준 |
| BERT-tiny+MLP | 0.668 | 0.673 | 0.669 | 0.671 | 10.83% | seed 42 |
| ELECTRA-small+MLP | 0.757 | 0.775 | 0.765 | 0.771 | 10.83% | 3-seed 평균 |
| DistilRoBERTa+MLP | **0.804** | **0.794** | **0.798** | **0.795** | 10.38% | 3-seed 평균 |

#### 데이터셋별 동일 마스킹 예산 결과

| 데이터셋 | Student | Precision | Recall | F1 | F2 | 마스킹률 |
|---|---|---:|---:|---:|---:|---:|
| drug | BERT-tiny | 0.684 | 0.665 | 0.674 | 0.669 | 6.59% |
| drug | ELECTRA-small | 0.772 | 0.800 | 0.786 | 0.794 | 7.02% |
| drug | DistilRoBERTa | **0.789** | **0.806** | **0.797** | **0.802** | 6.93% |
| symptom2dx | BERT-tiny | 0.795 | 0.761 | 0.778 | 0.767 | 11.24% |
| symptom2dx | ELECTRA-small | 0.874 | **0.915** | 0.894 | 0.907 | 12.30% |
| symptom2dx | DistilRoBERTa | **0.920** | 0.911 | **0.915** | **0.913** | 11.63% |
| ADR | BERT-tiny | 0.655 | 0.762 | 0.704 | 0.738 | 21.89% |
| ADR | ELECTRA-small | 0.706 | 0.770 | 0.736 | 0.756 | 20.51% |
| ADR | DistilRoBERTa | **0.751** | **0.781** | **0.766** | **0.775** | 19.57% |
| redditmh | BERT-tiny | 0.538 | 0.506 | 0.521 | 0.512 | 3.58% |
| redditmh | ELECTRA-small | 0.677 | 0.616 | 0.645 | 0.628 | 3.47% |
| redditmh | DistilRoBERTa | **0.757** | **0.677** | **0.715** | **0.692** | 3.41% |

#### Privacy-oriented threshold의 Macro 결과

민감 토큰을 놓치지 않는 쪽으로 threshold를 낮추면 Recall과 F2가 올라가지만 더 많은
토큰을 가린다.

| Student | Precision | Recall | F1 | F2 | 마스킹률 |
|---|---:|---:|---:|---:|---:|
| BERT-tiny | 0.469 | 0.905 | 0.613 | 0.757 | 19.19% |
| ELECTRA-small | 0.637 | 0.901 | 0.743 | 0.829 | 14.66% |
| DistilRoBERTa | **0.663** | **0.914** | **0.765** | **0.847** | 14.81% |

#### CPU 지연시간·처리량·메모리·크기

| 방식 | 로딩 시간 | 모델 state | 중앙 지연 | p95 지연 | 처리량 | 최대 RSS |
|---|---:|---:|---:|---:|---:|---:|
| medterm4 | 81.09 s | 단일 파일 없음 | 43.89 ms | 66.20 ms | 21.3문장/s | 10,178.8 MB |
| BERT-tiny+MLP | 6.02 s¹ | 17.6 MB | **3.55 ms** | **4.78 ms** | **264.3문장/s** | **695.3 MB** |
| ELECTRA-small+MLP | 6.52 s | 54.1 MB | 45.23 ms | 52.59 ms | 21.7문장/s | 783.0 MB |
| DistilRoBERTa+MLP | 6.81 s | 328.9 MB | 150.63 ms | 171.18 ms | 6.5문장/s | 1,327.5 MB |

¹ BERT-tiny 로딩은 동일 프로세스 환경의 첫 실행에서 21.82초였고, 캐시가 준비된 별도
재측정은 6.02초였다. 따라서 로딩 시간은 캐시와 저장장치 상태에 영향을 받는다.

이 환경에서 ELECTRA의 처리 속도는 medterm4와 거의 같았지만, 최대 RSS는 약 13분의 1이고
초기 로딩은 약 12배 짧았다. BERT-tiny는 medterm4보다 약 12배 높은 처리량을 보였지만 F1
0.669라서 실제 대체 모델로는 정확도가 부족하다. DistilRoBERTa는 Student 중 정확도가 가장
높지만 medterm4보다 약 3.4배 느리다.

따라서 모델을 쓰는 근거는 단순히 “항상 더 빠르다”가 아니다. 학습된 Student는 큰 UMLS
linker와 여러 NLP 파이프라인 없이 하나의 체크포인트로 배포할 수 있고, ELECTRA 기준으로
메모리와 시작 시간이 크게 줄어든다. 반면 규칙과 완전히 같은 출력을 요구하면 medterm4를
직접 실행하는 것이 맞다. 모델이 유리해지려면 **허용 가능한 모방 오차 안에서 메모리·배포
복잡도를 줄이거나, 향후 human/LLM 라벨처럼 규칙으로 쓰기 어려운 판단까지 학습**해야 한다.

medterm4 실행 환경의 NMSLIB가 SSE/AVX 최적화 없이 설치되어 있어 지연시간은 최적화된
환경에서 더 짧아질 수 있다. 그러므로 위 시간·메모리 수치는 절대적인 제품 사양이 아니라
동일 서버에서 측정한 비교값으로 보고한다.

## 6. 사전 실험 A (2026-07-23~26): 초기 파일럿과 encoder fine-tuning

### 2026-07-23 Teacher 후보 파일럿

첫 10문장에서 당시 LLM Teacher와 `medterm4`를 비교했다. 이는 어느 쪽이 실제 정답인지
측정한 것이 아니라 두 pseudo-label 방법의 일치도를 확인한 예비 실험이다.

| 라벨 생성 방법 | 문장 수 | 토큰 마스킹률 | 비고 |
|---|---:|---:|---|
| medterm4 전체 파일럿 | 200 | 6.98% | 문장별 평균 7.34% ± 4.27% |
| medterm4 공통 표본 | 10 | 7.61% | LLM과 같은 문장 |
| LLM Teacher 공통 표본 | 10 | 12.10% | 4개는 정책 경계 검토 필요 |

medterm4를 임시 비교 기준으로 놓은 방법 간 일치도는 Precision 0.532, Recall 0.844,
F1 0.652였다. LLM이 더 많이 가렸지만 human-gold가 없으므로 더 정확하다고 결론 내리지
않았다. 이후 현재 의료 Student 실험은 재현 가능한 `medterm4` 규칙 Teacher로 통일했다.

Drug Reviews 200개(train 160/validation 20/test 20)의 초기 파일럿이다. Teacher는 당시
`medterm4-reimplementation-v1`이었고 전체 mask는 6.98%였다.

### 동일 마스킹 예산

| BERT-tiny 조건 | Precision | Recall | F1 | F2 | Teacher mask | Student mask |
|---|---:|---:|---:|---:|---:|---:|
| Encoder frozen + MLP만 학습 | 0.507 | 0.618 | 0.557 | 0.592 | 9.01% | 10.98% |
| Encoder+MLP fine-tuning | **0.538** | **0.653** | **0.590** | **0.626** | 9.01% | 10.93% |

Encoder fine-tuning은 유효했지만 F1 0.590은 규칙 필터를 대체하기에 부족했다.

## 7. 사전 실험 B (2026-07-26): 200개에서 1,000개로 확대

동일한 BERT-tiny 전체 fine-tuning 구조에서 Drug Reviews 표본을 확대했다.

| 데이터 규모 | Train/Test | Precision | Recall | F1 | F2 | Student mask |
|---|---:|---:|---:|---:|---:|---:|
| 200개 | 160/20 | 0.538 | **0.653** | 0.590 | 0.626 | 10.93% |
| 1,000개 | 800/100 | **0.659** | 0.633 | **0.646** | **0.638** | 6.55% |

데이터 증가로 Precision과 F1은 개선됐지만 Recall은 개선되지 않았다. 두 실험은 test와
Teacher 버전이 달라 순수한 learning curve로 단정할 수 없다는 한계가 있다.

## 8. 사전 실험 C (2026-07-26~27): Student 모델 크기 비교

최신 정합 Teacher, 동일 Drug Reviews 1,000개, 동일 800/100/100 split에서 encoder만
교체했다.

### 모델 크기

| Student | 구조 | 실제 파라미터 | 체크포인트 |
|---|---|---:|---:|
| BERT-tiny | 2층, hidden 128 | 4,402,690 | 17.6 MB |
| ELECTRA-small | 12층, hidden 256 | 13,516,162 | 54.1 MB |
| DistilRoBERTa | 6층, hidden 768 | 82,217,090 | 328.9 MB |

### 동일 마스킹 예산

Teacher test mask는 6.82%다.

| Student | Precision | Recall | F1 | F2 | Student mask | 남은 민감 토큰 |
|---|---:|---:|---:|---:|---:|---:|
| BERT-tiny | 0.659 | 0.633 | 0.646 | 0.638 | 6.55% | 36.71% |
| ELECTRA-small | 0.762 | 0.713 | 0.737 | 0.722 | 6.38% | 28.69% |
| DistilRoBERTa | **0.807** | **0.747** | **0.776** | **0.758** | 6.31% | **25.32%** |

ELECTRA-small은 DistilRoBERTa보다 F1이 0.039 낮지만 약 6.1배 작다.

### Privacy-oriented F2 threshold

| Student | Precision | Recall | F1 | F2 | Student mask |
|---|---:|---:|---:|---:|---:|
| BERT-tiny | 0.520 | 0.824 | 0.638 | 0.738 | 10.81% |
| ELECTRA-small | 0.625 | 0.883 | 0.732 | 0.816 | 9.63% |
| DistilRoBERTa | **0.654** | **0.900** | **0.758** | **0.837** | **9.38%** |

## 9. 본 실험 데이터 준비 (2026-07-27): 의료 데이터셋 확장

원본 전체 규모로 가기 전 데이터셋별 약 1,000개로 맞춘 파일럿이다.

| 데이터셋 | 출처 | 전체/Train/Valid/Test | Teacher test mask | 의미 |
|---|---|---:|---:|---|
| drug | `lewtun/drug-reviews` | 1000/800/100/100 | 6.78% | 기존 표본과 원문 비중복, 같은 출처 |
| symptom2dx | `gretelai/symptom_to_diagnosis` | 1060/844/108/108 | 11.75% | 증상→22진단, 합성 단문 |
| ADR | `ade_corpus_v2` | 1000/800/100/100 | 18.81% | 약물 부작용 이진분류 |
| redditmh | `solomonk/reddit_mental_health_posts` | 1000/800/100/100 | 3.81% | 정신건강 서술, subreddit 5종 |

`drug`와 기존 Drug Reviews 1,000개 사이 원문 중복은 0개임을 검증했다.

## 10. 탐색 실험 E (2026-07-27): Zero-shot, in-domain, multi-domain 비교

### 학습 방식별 Macro F1

| 학습 방식 | ELECTRA-small | DistilRoBERTa |
|---|---:|---:|
| Drug Reviews only, source threshold 고정 zero-shot | 0.701 | 0.723 |
| Leave-one-domain-out, source threshold 고정 | 0.705 | 0.729 |
| 데이터셋별 별도 in-domain 모델 | 0.763 | 0.797 |
| 네 데이터셋 multi-domain 모델 | **0.796** | **0.831** |

### Drug Reviews only zero-shot F1

Drug Reviews validation에서 고른 threshold를 대상 데이터셋에 그대로 적용했다.

| Target | ELECTRA-small | DistilRoBERTa |
|---|---:|---:|
| drug | 0.785 | 0.788 |
| symptom2dx | 0.759 | 0.767 |
| ADR | 0.695 | 0.707 |
| redditmh | 0.567 | 0.630 |

같은 출처인 drug에서는 잘 재현됐고, 독립 도메인인 ADR과 서술형 redditmh에서 성능이
하락했다. redditmh target validation으로 threshold만 다시 맞춰도 F1은 ELECTRA 0.607,
DistilRoBERTa 0.645라서 단순 calibration 문제만은 아니었다.

### 데이터셋별 별도 in-domain 모델 F1

| Target | ELECTRA-small | DistilRoBERTa |
|---|---:|---:|
| drug | 0.786 | 0.795 |
| symptom2dx | 0.890 | 0.906 |
| ADR | 0.728 | 0.778 |
| redditmh | 0.649 | 0.708 |
| Macro | 0.763 | 0.797 |

### Multi-domain Student: 전역 동일예산 threshold

네 train 3,244문장과 validation 408문장을 합쳐 모델 하나를 학습했다. pooled
validation에서 선택한 하나의 threshold를 모든 test에 고정 적용했다.

- ELECTRA threshold: 0.94
- DistilRoBERTa threshold: 0.93

| Target | Teacher mask | ELECTRA P/R/F1 | ELECTRA mask | Distil P/R/F1 | Distil mask |
|---|---:|---:|---:|---:|---:|
| drug | 6.78% | 0.819 / 0.819 / **0.819** | 6.8% | 0.817 / 0.845 / **0.831** | 7.0% |
| symptom2dx | 11.75% | 0.873 / 0.938 / **0.904** | 12.6% | 0.897 / 0.940 / **0.918** | 12.3% |
| ADR | 18.81% | 0.689 / 0.799 / **0.740** | 21.8% | 0.760 / 0.847 / **0.801** | 20.9% |
| redditmh | 3.81% | 0.819 / 0.641 / **0.719** | 3.0% | 0.845 / 0.714 / **0.774** | 3.2% |
| Macro F1 | — | **0.796** | — | **0.831** | — |

Multi-domain Student는 데이터셋별 별도 모델보다 네 test 모두에서 높은 F1을 얻었다.
특히 redditmh는 ELECTRA 0.649→0.719, DistilRoBERTa 0.708→0.774로 개선됐다.

ADR은 전역 threshold에서 Teacher보다 더 많이 가렸다. 따라서 macro F1과 함께
데이터셋별 mask 차이를 반드시 보고해야 한다.

### Multi-domain Student: 전역 F2 threshold

- ELECTRA threshold: 0.43
- DistilRoBERTa threshold: 0.35

| Target | ELECTRA Recall/F2/mask | Distil Recall/F2/mask |
|---|---:|---:|
| drug | 0.939 / 0.859 / 9.9% | 0.935 / 0.866 / 9.5% |
| symptom2dx | 0.988 / 0.934 / 15.2% | 0.988 / 0.947 / 14.3% |
| ADR | 0.958 / 0.843 / 31.6% | 0.979 / 0.878 / 29.6% |
| redditmh | 0.871 / 0.766 / 6.4% | 0.865 / 0.801 / 5.3% |

F2 설정은 Recall을 높이지만 Teacher보다 많은 토큰을 가린다. 특히 ADR은 약 30%를
가리므로 privacy와 utility를 함께 확인해야 한다.

## 11. 탐색 실험 F (2026-07-27): Leave-one-domain-out 일반화

각 target을 통째로 제외하고 나머지 세 데이터셋만으로 다시 학습했다. threshold도 target
validation이 아니라 세 source validation을 합친 데이터에서 선택했다. 이후 target test에
threshold를 고정 적용했다. 따라서 아래 결과는 target 문장이나 라벨을 학습·보정에 사용하지
않은 엄격한 zero-shot 결과다.

#### 동일 마스킹 예산

| Held-out target | Teacher mask | ELECTRA P/R/F1 | ELECTRA mask | Distil P/R/F1 | Distil mask |
|---|---:|---:|---:|---:|---:|
| drug | 6.78% | 0.759 / 0.699 / **0.728** | 6.24% | 0.748 / 0.769 / **0.758** | 6.96% |
| symptom2dx | 11.75% | 0.716 / 0.825 / **0.767** | 13.55% | 0.784 / 0.766 / **0.775** | 11.47% |
| ADR | 18.81% | 0.658 / 0.783 / **0.715** | 22.39% | 0.697 / 0.780 / **0.737** | 21.04% |
| redditmh | 3.81% | 0.724 / 0.530 / **0.612** | 2.79% | 0.776 / 0.552 / **0.645** | 2.71% |
| Macro F1 | — | **0.705** | — | **0.729** | — |

Multi-domain과 비교하면 Macro F1이 ELECTRA 0.796→0.705, DistilRoBERTa
0.831→0.729로 낮아졌다. 특히 redditmh Recall이 0.530/0.552에 그쳐 Teacher 민감 토큰의
약 45~47%가 남았다. 즉 기존 multi-domain의 높은 점수는 공통 규칙 학습뿐 아니라 target
도메인의 표현을 학습한 효과도 포함한다.

Drug Reviews only zero-shot과 비교하면 LODO Macro F1 향상은 ELECTRA +0.004,
DistilRoBERTa +0.006에 불과하다. 다만 같은 출처인 drug를 제외하면 symptom2dx, ADR,
redditmh에서는 대체로 개선되어 여러 source 도메인을 섞는 효과 자체는 확인됐다.

#### Privacy-oriented F2 threshold

| Held-out target | ELECTRA Recall/F2/mask | Distil Recall/F2/mask |
|---|---:|---:|
| drug | 0.874 / 0.804 / 9.70% | 0.888 / 0.822 / 9.50% |
| symptom2dx | 0.947 / 0.858 / 17.88% | 0.952 / 0.873 / 17.09% |
| ADR | 0.926 / 0.830 / 29.65% | 0.913 / 0.831 / 28.11% |
| redditmh | 0.721 / 0.657 / 5.67% | 0.780 / 0.712 / 5.65% |
| Macro F2 | **0.787** | **0.809** |

F2 설정은 unseen domain에서도 Recall을 높였지만 마스킹 비율이 증가했다. redditmh는
DistilRoBERTa에서도 Recall 0.780으로 여전히 가장 어려운 target이다.

## 12. 최종 모델 선택

| 목표 | 선택 | 근거 |
|---|---|---|
| 가장 높은 본 실험 Teacher 모방 성능 | DistilRoBERTa | 3-seed Macro F1 0.798 ± 0.005 |
| 크기·속도와 성능의 균형 | ELECTRA-small | 54.1 MB, 3-seed Macro F1 0.765 ± 0.002 |
| 최소 크기 baseline | BERT-tiny | 17.6 MB지만 성능 부족 |

현재 논문 본 실험의 중심 Student는 데이터셋마다 같은 구조로 학습한 ELECTRA와 DistilRoBERTa다.
ELECTRA는 제안 경량 모델, DistilRoBERTa는 고성능 기준선이며 multi-domain과 LODO는 탐색 분석으로 둔다.

## 13. 해석 시 주의점

1. `잘 가린다 = downstream 정확도가 떨어진다`가 아니다. 1차 평가는 Teacher token
   mask와의 P/R/F1/F2다.
2. 모든 의료 정답은 pseudo-teacher다. 실제 privacy 주장은 human-gold 검증이 필요하다.
3. 원래 네 데이터셋의 ELECTRA와 DistilRoBERTa만 3개 seed를 반복했다. BERT-tiny와
   신규 MedNLI·mentalhealth는 현재 seed 42 한 번이다.
4. 각 데이터셋은 약 1,000개 파일럿이며 원래 전체 규모와 다르다.
5. symptom2dx는 원래 split을 합친 뒤 다시 stratified split했다. MedNLI는 공식 split을 보존했다.
6. redditmh는 subreddit label 자체가 질환명인 부분 순환성이 있다.
7. drug는 기존 표본과 비중복이지만 같은 `lewtun/drug-reviews` 출처다.
8. 모델 크기가 작다고 지연시간도 항상 짧은 것은 아니다. ELECTRA는 12층이므로 실제
   latency 측정이 필요하다.
9. Mentalhealth는 52,681개 전체가 아니라 중복 제거 후 균형 표집한 1,000개 파일럿이다.
10. max length 256 이후 토큰은 평가에서 제외되므로 장문 전체 보호율은 별도 평가가 필요하다.

## 14. 다음 실험

우선순위는 다음과 같다.

1. MedNLI·mentalhealth의 ELECTRA/DistilRoBERTa를 seed 43·44로 반복
2. 데이터셋별 human-gold 민감정보 표본에서 Recall/F2와 Teacher 오류 측정
3. 원래 전체 데이터 규모로 확대: drug 약 13k, ADR 약 1.8k, redditmh 약 21k
4. 최적화된 NMSLIB 환경에서 medterm4 효율을 재측정해 환경 편향 확인
5. RedactFormer 연결 후 RTM utility와 privacy leakage 평가
6. 필요하면 제7의 의료 데이터셋을 완전 unseen test로 추가

## 15. 산출물 경로

### 데이터

- Drug Reviews 1,000: `data/medical_redactor/drugreviews_1000/`
- Cross-dataset 6종: `data/medical_redactor/cross_dataset/`
- MedNLI: `data/medical_redactor/cross_dataset/mednli/`
- Mentalhealth: `data/medical_redactor/cross_dataset/mentalhealth/`
- Multi-domain train/validation/test: `data/medical_redactor/cross_dataset/multidomain/`

### 모델과 평가

- BERT-tiny 1,000: `artifacts/medical_redactor/medterm4_student_finetuned_1000/`
- ELECTRA 1,000: `artifacts/medical_redactor/medterm4_electra_small_finetuned_1000/`
- DistilRoBERTa 1,000: `artifacts/medical_redactor/medterm4_distilroberta_finetuned_1000/`
- Drug-only zero-shot: `artifacts/medical_redactor/cross_dataset_zero_shot/`
- 데이터셋별 in-domain: `artifacts/medical_redactor/cross_dataset_in_domain/`
- Multi-domain: `artifacts/medical_redactor/cross_dataset_multidomain/`
- LODO 모델·평가·요약: `artifacts/medical_redactor/cross_dataset_lodo/`
- 핵심 3×4 표·CPU·3-seed: `artifacts/medical_redactor/core_matrix/`
- 규칙·Student 통합 JSON: `artifacts/medical_redactor/core_matrix/complete_rule_student_comparison.json`
- 방식별 격리 CPU 벤치마크: `artifacts/medical_redactor/core_matrix/efficiency/`
- 신규 두 데이터셋 체크포인트·평가: `artifacts/medical_redactor/core_matrix/extension_seed42/`
- 3모델×6데이터셋 seed 42 통합 JSON: `artifacts/medical_redactor/core_matrix/six_dataset_seed42_summary.json`

### 주요 코드

- Teacher 라벨: `src/annotate_medterm4.py`
- 학습: `src/train.py`
- 단일 데이터셋 평가: `src/evaluate_medical_student.py`
- Cross-dataset 평가: `src/evaluate_medical_cross_dataset.py`
- Cross-dataset 생성: `src/prepare_medical_cross_datasets.py`
- MedNLI·mentalhealth 입력 생성: `src/prepare_medical_extension_datasets.py`
- 신규 데이터 medterm4 라벨·split: `src/prepare_extension_teacher_data.py`
- 신규 3모델 학습·평가: `src/run_extension_model_matrix.py`
- 6데이터셋 집계: `src/summarize_six_dataset_matrix.py`
- LODO 분할: `src/prepare_lodo_splits.py`
- LODO 평가: `src/run_lodo_evaluation.py`
- LODO 집계: `src/summarize_lodo_results.py`
- 핵심 3×4 집계: `src/summarize_core_model_matrix.py`
- CPU 벤치마크: `src/benchmark_redactor_models.py`
- 규칙·Student 격리 벤치마크: `src/benchmark_redactor_worker.py`
- 규칙·Student 통합 집계: `src/summarize_rule_student_comparison.py`
- 3-seed 집계: `src/summarize_seed_repeats.py`

## 16. 비의료 정책별 Student 실험 (2026-07-28)

의료 `medterm4` 결과와 섞지 않는 별도 실험이다. 모든 데이터셋은 1,000개를
train/validation/test = 800/100/100으로 나눴고, 세 Student를 encoder까지 5 epochs
fine-tuning했다(seed 42). 수치는 test token 기준이다.

### 16.1 Teacher 정책과 역할

| 그룹 | 데이터셋 | Teacher 정책 | test mask | 해석 |
|---|---|---|---:|---|
| 실제 PII | bios | `piiclean-v1` | 13.48% | 인명·기관·지역·금액·비율·연도 |
| 실제 PII | MRPC | `piiclean-strict-v1` | 9.03% | 인명·기관·지역 중심의 엄격 PII |
| 비개인 엔티티 대조 | QNLI | `entityclean-v1` | 11.65% | NER 엔티티 모방; 개인정보 성능 주장이 아님 |
| 비개인 엔티티 대조 | FinPhraseBank | `entityclean-v1` | 15.88% | 기업·인명·지역·금액·연도; `medterm4` 미사용 |

라벨은 LLM이 아니라 RedactFormer의 `piiclean` 구현에 맞춘 spaCy NER 규칙에서 생성한
deterministic pseudo-gold다. 따라서 아래 F1은 규칙 모방 점수이지 human-gold 개인정보
탐지 점수가 아니다.

### 16.2 실제 PII 그룹: 동일 마스킹 예산

| Dataset | Student | Precision | Recall | F1 | F2 | Student mask |
|---|---|---:|---:|---:|---:|---:|
| bios | BERT-tiny | 0.749 | 0.802 | 0.775 | 0.791 | 14.42% |
| bios | ELECTRA-small | 0.844 | 0.813 | 0.828 | 0.819 | 12.97% |
| bios | DistilRoBERTa | **0.870** | **0.887** | **0.878** | **0.883** | 13.74% |
| MRPC | BERT-tiny | 0.718 | 0.790 | 0.752 | 0.774 | 9.93% |
| MRPC | ELECTRA-small | 0.916 | **0.914** | **0.915** | **0.914** | 9.01% |
| MRPC | DistilRoBERTa | **0.917** | 0.904 | 0.911 | 0.907 | 8.90% |

PII 두 데이터셋 내부 Macro F1은 BERT-tiny 0.763, ELECTRA-small 0.871,
DistilRoBERTa 0.894다. DistilRoBERTa가 평균 최고이며, MRPC에서는 ELECTRA와 사실상
비슷하다. 한 seed의 파일럿이므로 작은 차이를 우열로 확정하지 않는다.

### 16.3 비개인 엔티티 대조 그룹: 동일 마스킹 예산

| Dataset | Student | Precision | Recall | F1 | F2 | Student mask |
|---|---|---:|---:|---:|---:|---:|
| QNLI | BERT-tiny | 0.759 | 0.737 | 0.748 | 0.742 | 11.32% |
| QNLI | ELECTRA-small | 0.844 | 0.809 | 0.826 | 0.816 | 11.17% |
| QNLI | DistilRoBERTa | **0.880** | **0.871** | **0.875** | **0.872** | 11.53% |
| FinPhraseBank | BERT-tiny | 0.795 | 0.742 | 0.768 | 0.752 | 14.82% |
| FinPhraseBank | ELECTRA-small | 0.820 | 0.839 | 0.830 | 0.835 | 16.24% |
| FinPhraseBank | DistilRoBERTa | **0.881** | **0.872** | **0.877** | **0.874** | 15.71% |

이 그룹 내부 Macro F1은 BERT-tiny 0.758, ELECTRA-small 0.828,
DistilRoBERTa 0.876이다. 이 값은 ‘일반 엔티티 규칙 압축 가능성’을 보여줄 뿐 PII recall로
인용하면 안 된다.

### 16.4 Recall 중심 F2 threshold

| 그룹 | Student | Macro Precision | Macro Recall | Macro F2 | 평균 Student mask |
|---|---|---:|---:|---:|---:|
| 실제 PII | BERT-tiny | 0.551 | 0.920 | 0.811 | 18.64% |
| 실제 PII | ELECTRA-small | 0.717 | 0.962 | 0.899 | 15.42% |
| 실제 PII | DistilRoBERTa | **0.798** | **0.981** | **0.936** | 14.23% |
| 비개인 엔티티 대조 | BERT-tiny | 0.562 | 0.922 | 0.817 | 22.65% |
| 비개인 엔티티 대조 | ELECTRA-small | 0.692 | 0.940 | 0.877 | 18.79% |
| 비개인 엔티티 대조 | DistilRoBERTa | **0.771** | **0.963** | **0.917** | 17.19% |

Recall 중심 threshold는 놓치는 규칙 토큰을 줄이지만 더 많이 가린다. 그래서 본 표는 동일
예산 표를 대체하지 않고 privacy-oriented 운용점으로 함께 제시한다.

### 16.5 제외한 데이터셋

- `biosx`: RedactFormer 감사 문서에서도 원본/정제 출처가 회수되지 않아 임의 데이터로
  대체하지 않았다.
- `MDCC`: Empath adversity 정책 코드는 확인했지만 실행에 필요한 원본 CSV가 저장소에 없어
  재현하지 않았다.
- `medterm4`: 의료 용어 선택 규칙이므로 위 네 비의료 데이터셋의 baseline으로 사용하지 않았다.

### 16.6 산출물

- 생성 코드: `src/prepare_nonmedical_rule_datasets.py`
- 데이터: `data/nonmedical_redactor/{bios,mrpc,qnli,finphrasebank}/`
- 학습 실행기: `src/run_extension_model_matrix.py --data-root ...`
- 모델·로그·개별 평가: `artifacts/nonmedical_redactor/seed42/`
- 통합 평가 JSON: `artifacts/nonmedical_redactor/seed42/summary.json`
## 17. 메인 실험 갱신 (2026-07-29): 전체 데이터 3모델 × 10데이터셋

앞 절의 약 1,000개 결과는 파일럿·탐색 기록으로 보존한다. 모델 정확도 비교의 최신 메인
결과는 빈 문장과 완전 중복을 제거한 뒤 사용 가능한 데이터 706,495개를 전부 사용한 아래
표다. 공식 split이 있으면 보존했고, 나머지는 결정적·층화 split을 사용했다. 학습 조건은
encoder 전체 fine-tuning, 5 epoch, batch 16, max length 256, seed 42로 고정했다.
threshold는 validation에서 선택하고 test에 한 번만 적용했다.

Symptom2Dx는 기존 실험부터 사용 가능한 1,060개 전부를 썼고 재생성한 split도 동일하므로
기존 전체-coverage 결과를 재사용했다. 나머지 9개 데이터셋은 전체 데이터로 다시 학습했다.

### 17.1 동일 마스킹 예산 test F1

| 그룹 | 데이터셋 | 전체 예시 | BERT-tiny | ELECTRA-small | DistilRoBERTa |
|---|---|---:|---:|---:|---:|
| 의료 규칙 | Drug Reviews | 49,974 | 0.855 | 0.892 | **0.910** |
| 의료 규칙 | Symptom2Dx | 1,060 | 0.778 | 0.890 | **0.906** |
| 의료 규칙 | ADR | 20,892 | 0.783 | 0.840 | **0.857** |
| 의료 규칙 | RedditMH | 59,607 | 0.783 | 0.835 | **0.873** |
| 의료 규칙 | MedNLI | 14,021 | 0.798 | 0.856 | **0.859** |
| 의료 규칙 | Mental Health | 41,878 | 0.735 | 0.786 | **0.813** |
| 실제 PII | BIOS | 395,368 | 0.892 | 0.910 | **0.944** |
| 실제 PII | MRPC | 5,801 | 0.841 | 0.906 | **0.935** |
| 비개인 엔티티 대조 | QNLI | 115,636 | 0.861 | 0.887 | **0.914** |
| 비개인 엔티티 대조 | FinPhraseBank | 2,258 | 0.827 | 0.863 | **0.890** |

### 17.2 의미가 같은 그룹 내부 Macro

| 그룹 | 데이터셋 수 | BERT F1 | ELECTRA F1 | Distil F1 | BERT privacy F2 | ELECTRA privacy F2 | Distil privacy F2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 의료 규칙 | 6 | 0.788 | 0.850 | **0.870** | 0.859 | 0.895 | **0.909** |
| 실제 PII | 2 | 0.867 | 0.908 | **0.939** | 0.901 | 0.934 | **0.961** |
| 비개인 엔티티 대조 | 2 | 0.844 | 0.875 | **0.902** | 0.878 | 0.904 | **0.935** |

세 그룹은 Teacher 의미가 다르므로 10개를 하나의 macro로 합치지 않는다. DistilRoBERTa는
10개 데이터셋 모두에서 동일 예산 F1이 가장 높았고, ELECTRA-small은 일관되게 중간,
BERT-tiny는 최소 크기 baseline이었다. 따라서 정확도 최우선 기준선은 DistilRoBERTa,
경량성과 성능 절충 후보는 ELECTRA-small이라는 기존 선택이 전체 데이터에서도 유지된다.

다만 이 표는 규칙 Teacher 모방 점수다. human-gold 개인정보 정답률이나 RTM 복구 저항성을
직접 뜻하지 않는다. 실제 privacy 결론에는 human-gold 민감 토큰 Recall/F2, 같은 마스킹
예산의 기존 규칙·random 비교, RedactFormer 연결 후 RTM 복구율과 downstream utility가
각각 필요하다.

### 17.3 산출물

- 전체 split과 pseudo-label: `data/full_redactor/<dataset>/`
- 모델·평가: `artifacts/full_redactor/seed42/`
- 30개 통합 결과: `reports/full_dataset_results.json`
- 브라우저 표: `reports/redactor_results_dashboard.html`
- Excel용 원자료: `reports/redactor_results.csv`
- 집계 코드: `src/summarize_full_dataset_results.py`
- HTML/CSV 생성: `src/build_results_dashboard.py`
