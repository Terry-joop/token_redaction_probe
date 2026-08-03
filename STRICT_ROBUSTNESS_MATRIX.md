# 전 데이터셋 strict 5/7 입력 교란 실험

학습에는 Seen 5종만 넣고, 최종 test에는 학습에서 한 번도 쓰지 않은 Unseen 7종만 사용했다. Student는 ELECTRA-small로 고정하고 seed 42·43·44를 반복했다.

## 한눈에 보는 결과

- 10개 데이터셋의 clean train 517,215행에 Seen 증강 367,609행을 추가했다.
- 전체 test에서 적용 가능한 Unseen target-pair 352,908개, 고유 원문 113,047개를 평가했다.
- clean 품질 gate(F1≥0.85, F2≥0.90, Recall≥0.90)를 세 seed 모두 통과한 데이터셋은 8/10개다: Drug Reviews, Symptom2Dx, ADR, MedNLI, BIOS, MRPC, QNLI, FinPhraseBank.
- 평균 noisy 탐지와 하락폭이 모두 좋은 데이터셋은 3/10개다: Drug Reviews, Symptom2Dx, FinPhraseBank.
- 두 차이의 95% CI가 세 seed 모두 0보다 큰 엄격 우세는 2/10개다: Drug Reviews, FinPhraseBank.

| 그룹 | 데이터셋 | Train clean+증강 | Test 원문 | Unseen target pair | Clean F2 | 규칙 noisy 탐지 | Student noisy 탐지 | Student−규칙 | 규칙 하락 | Student 하락 | Clean gate | CI gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 의료 규칙 | Drug Reviews | 39,980+30,591 | 4,997 | 13,901 | 0.925±0.001 | 0.528 | 0.573±0.008 | +0.045 | 0.472 | 0.332±0.003 | 3/3 | 3/3 |
| 의료 규칙 | Symptom2Dx | 844+473 | 108 | 217 | 0.927±0.002 | 0.429 | 0.439±0.014 | +0.011 | 0.571 | 0.464±0.003 | 3/3 | 0/3 |
| 의료 규칙 | ADR | 16,714+13,653 | 2,089 | 5,554 | 0.928±0.000 | 0.584 | 0.569±0.002 | -0.015 | 0.416 | 0.333±0.002 | 3/3 | 0/3 |
| 의료 규칙 | RedditMH | 47,685+21,346 | 5,961 | 10,459 | 0.871±0.002 | 0.440 | 0.395±0.004 | -0.045 | 0.560 | 0.416±0.001 | 0/3 | 0/3 |
| 의료 규칙 | MedNLI | 11,210+8,594 | 1,416 | 3,759 | 0.916±0.000 | 0.561 | 0.545±0.008 | -0.016 | 0.439 | 0.323±0.006 | 3/3 | 0/3 |
| 의료 규칙 | Mental Health | 33,502+11,179 | 4,188 | 5,612 | 0.812±0.009 | 0.455 | 0.367±0.013 | -0.088 | 0.545 | 0.397±0.011 | 0/3 | 0/3 |
| 실제 PII | BIOS | 257,090+215,694 | 98,745 | 298,330 | 0.925±0.001 | 0.662 | 0.633±0.002 | -0.029 | 0.338 | 0.278±0.000 | 3/3 | 0/3 |
| 실제 PII | MRPC | 3,668+1,725 | 1,725 | 2,986 | 0.934±0.003 | 0.534 | 0.505±0.002 | -0.029 | 0.466 | 0.426±0.001 | 3/3 | 0/3 |
| 비개인 엔티티 대조 | QNLI | 104,716+63,097 | 5,457 | 11,572 | 0.909±0.001 | 0.595 | 0.583±0.001 | -0.012 | 0.405 | 0.314±0.002 | 3/3 | 0/3 |
| 비개인 엔티티 대조 | FinPhraseBank | 1,806+1,257 | 226 | 518 | 0.939±0.002 | 0.625 | 0.669±0.005 | +0.044 | 0.375 | 0.262±0.003 | 3/3 | 3/3 |

## 데이터셋별 수치 해석

아래 target 탐지율은 clean 최신 규칙이 고른 민감 span 전체를 오염 문장에서도 전부 가린 비율이다. noisy token F2는 오염 문장의 모든 토큰을 대상으로 한 별도 지표이므로, 특정 target의 생존율과 같은 숫자로 해석하면 안 된다.

### Drug Reviews — 엄격 우세

- **평가 규모:** test 원문 4,997개에서 적용 가능한 Unseen target-pair 13,901개를 평가했다.
- **Clean 모방:** Student F2 0.925±0.001, clean gate 3/3이다.
- **오염 후 target 탐지:** 규칙 52.8% 대 Student 57.3%±0.8%p로, Student−규칙은 +4.5%p다.
- **Clean→오염 하락:** 규칙 47.2%p 대 Student 33.2%p이며 Student의 하락폭 이점은 +14.0%p다. noisy token F2는 규칙 0.927, Student 0.874이다.
- **의미:** 실제 의료 privacy 도메인에서 Student가 규칙의 표면 이음매를 보완할 수 있다는 가장 강한 근거다. 다만 전체 noisy token F2는 규칙보다 낮으므로 독립 대체가 아니라 규칙과 병렬로 쓰는 보완 후보로 해석한다. 엄격 CI gate는 3/3이다.

### Symptom2Dx — 평균 우세지만 통계적 재현성 미달

- **평가 규모:** test 원문 108개에서 적용 가능한 Unseen target-pair 217개를 평가했다.
- **Clean 모방:** Student F2 0.927±0.002, clean gate 3/3이다.
- **오염 후 target 탐지:** 규칙 42.9% 대 Student 43.9%±1.4%p로, Student−규칙은 +1.1%p다.
- **Clean→오염 하락:** 규칙 57.1%p 대 Student 46.4%p이며 Student의 하락폭 이점은 +10.8%p다. noisy token F2는 규칙 0.842, Student 0.766이다.
- **의미:** 평균값은 Student 쪽이 약간 좋지만 test와 target-pair가 작고 3-seed CI gate를 통과하지 못했다. 표본을 늘려 재검증하기 전에는 우세라고 주장하지 않는다. 엄격 CI gate는 0/3이다.

### ADR — 오염 후 절대 탐지율에서 규칙 우세

- **평가 규모:** test 원문 2,089개에서 적용 가능한 Unseen target-pair 5,554개를 평가했다.
- **Clean 모방:** Student F2 0.928±0.000, clean gate 3/3이다.
- **오염 후 target 탐지:** 규칙 58.4% 대 Student 56.9%±0.2%p로, Student−규칙은 -1.5%p다.
- **Clean→오염 하락:** 규칙 41.6%p 대 Student 33.3%p이며 Student의 하락폭 이점은 +8.3%p다. noisy token F2는 규칙 0.915, Student 0.877이다.
- **의미:** Student의 하락폭은 작지만 오염 후 절대 탐지율은 규칙보다 낮다. 낮은 clean 시작점 때문에 덜 하락해 보이는 효과가 섞였으므로 규칙 대체 근거가 아니다. 엄격 CI gate는 0/3이다.

### RedditMH — clean 품질 및 규칙 대비 성능 미달

- **평가 규모:** test 원문 5,961개에서 적용 가능한 Unseen target-pair 10,459개를 평가했다.
- **Clean 모방:** Student F2 0.871±0.002, clean gate 0/3이다.
- **오염 후 target 탐지:** 규칙 44.0% 대 Student 39.5%±0.4%p로, Student−규칙은 -4.5%p다.
- **Clean→오염 하락:** 규칙 56.0%p 대 Student 41.6%p이며 Student의 하락폭 이점은 +14.3%p다. noisy token F2는 규칙 0.866, Student 0.766이다.
- **의미:** 비정형 정신건강 서술에서 clean 품질 gate부터 통과하지 못했고 오염 후에도 규칙보다 낮다. 현재 모델·라벨 구성으로는 규칙 보완이나 대체를 주장할 수 없다. 엄격 CI gate는 0/3이다.

### MedNLI — 오염 후 절대 탐지율에서 규칙 우세

- **평가 규모:** test 원문 1,416개에서 적용 가능한 Unseen target-pair 3,759개를 평가했다.
- **Clean 모방:** Student F2 0.916±0.000, clean gate 3/3이다.
- **오염 후 target 탐지:** 규칙 56.1% 대 Student 54.5%±0.8%p로, Student−규칙은 -1.6%p다.
- **Clean→오염 하락:** 규칙 43.9%p 대 Student 32.3%p이며 Student의 하락폭 이점은 +11.7%p다. noisy token F2는 규칙 0.915, Student 0.866이다.
- **의미:** clean 규칙 모방은 합격했지만 학습에 없던 표면 교란에서는 규칙의 절대 탐지율이 더 높다. 임상 문장쌍에 대한 추가 증강 없이는 규칙 유지가 타당하다. 엄격 CI gate는 0/3이다.

### Mental Health — clean 품질 및 규칙 대비 성능 미달

- **평가 규모:** test 원문 4,188개에서 적용 가능한 Unseen target-pair 5,612개를 평가했다.
- **Clean 모방:** Student F2 0.812±0.009, clean gate 0/3이다.
- **오염 후 target 탐지:** 규칙 45.5% 대 Student 36.7%±1.3%p로, Student−규칙은 -8.8%p다.
- **Clean→오염 하락:** 규칙 54.5%p 대 Student 39.7%p이며 Student의 하락폭 이점은 +14.8%p다. noisy token F2는 규칙 0.853, Student 0.735이다.
- **의미:** clean F2와 오염 후 절대 탐지율이 모두 가장 약한 축에 속한다. 자유서술 표현의 다양성을 현재 단일 ELECTRA-small이 충분히 학습하지 못한 실패 사례다. 엄격 CI gate는 0/3이다.

### BIOS — 오염 후 절대 탐지율에서 규칙 우세

- **평가 규모:** test 원문 98,745개에서 적용 가능한 Unseen target-pair 298,330개를 평가했다.
- **Clean 모방:** Student F2 0.925±0.001, clean gate 3/3이다.
- **오염 후 target 탐지:** 규칙 66.2% 대 Student 63.3%±0.2%p로, Student−규칙은 -2.9%p다.
- **Clean→오염 하락:** 규칙 33.8%p 대 Student 27.8%p이며 Student의 하락폭 이점은 +6.0%p다. noisy token F2는 규칙 0.959, Student 0.906이다.
- **의미:** 가장 큰 실제 PII 평가이므로 중요한 반례다. Student가 덜 하락하더라도 오염 후 절대 탐지율은 규칙보다 낮아, 규모만 늘리는 것으로 규칙을 이기지는 못했다. 엄격 CI gate는 0/3이다.

### MRPC — 오염 후 절대 탐지율에서 규칙 우세

- **평가 규모:** test 원문 1,725개에서 적용 가능한 Unseen target-pair 2,986개를 평가했다.
- **Clean 모방:** Student F2 0.934±0.003, clean gate 3/3이다.
- **오염 후 target 탐지:** 규칙 53.4% 대 Student 50.5%±0.2%p로, Student−규칙은 -2.9%p다.
- **Clean→오염 하락:** 규칙 46.6%p 대 Student 42.6%p이며 Student의 하락폭 이점은 +4.0%p다. noisy token F2는 규칙 0.905, Student 0.878이다.
- **의미:** 엄격 PII 정책의 clean 모방은 좋지만 오염 후에는 규칙이 앞선다. Student를 단독 필터로 교체하기보다 규칙의 미탐 후보를 재검사하는 보조기로 보는 편이 맞다. 엄격 CI gate는 0/3이다.

### QNLI — 오염 후 절대 탐지율에서 규칙 우세

- **평가 규모:** test 원문 5,457개에서 적용 가능한 Unseen target-pair 11,572개를 평가했다.
- **Clean 모방:** Student F2 0.909±0.001, clean gate 3/3이다.
- **오염 후 target 탐지:** 규칙 59.5% 대 Student 58.3%±0.1%p로, Student−규칙은 -1.2%p다.
- **Clean→오염 하락:** 규칙 40.5%p 대 Student 31.4%p이며 Student의 하락폭 이점은 +9.0%p다. noisy token F2는 규칙 0.915, Student 0.867이다.
- **의미:** 규칙과 Student의 차이는 작지만 비개인 엔티티 대조군이므로 privacy 성과가 아니다. 학습형 모델이 일반 엔티티 경계를 어느 정도 유지하는지 보는 통제 결과다. 엄격 CI gate는 0/3이다.

### FinPhraseBank — 엄격 우세

- **평가 규모:** test 원문 226개에서 적용 가능한 Unseen target-pair 518개를 평가했다.
- **Clean 모방:** Student F2 0.939±0.002, clean gate 3/3이다.
- **오염 후 target 탐지:** 규칙 62.5% 대 Student 66.9%±0.5%p로, Student−규칙은 +4.4%p다.
- **Clean→오염 하락:** 규칙 37.5%p 대 Student 26.2%p이며 Student의 하락폭 이점은 +11.3%p다. noisy token F2는 규칙 0.922, Student 0.917이다.
- **의미:** 수치와 3-seed CI에서는 Student가 규칙보다 강건하지만 비개인 엔티티 대조군이다. 학습형 redactor의 표면 일반화 가능성은 지지해도 개인정보 보호 성공으로 세지 않는다. 엄격 CI gate는 3/3이다.


## 해석

- noisy 탐지는 clean 최신 규칙이 선택한 고정 target span을 오염 문장에서 전부 가렸는지 본다.
- Student−규칙이 양수이고 Student 하락이 더 작아야 규칙보다 표면 교란에 강하다고 본다. 하락폭만 작은 것은 Student의 clean 시작점이 낮아서 생길 수도 있어 성공으로 세지 않는다.
- 엄격 우세는 위 두 차이 모두의 source-cluster bootstrap 95% CI 하한이 0보다 큰 상태를 seed 3개에서 모두 재현한 경우다.
- Drug Reviews는 실제 의료 privacy 규칙에서 조건부 보완 근거를 보였지만, FinPhraseBank는 비개인 엔티티 대조이므로 privacy 성공으로 해석하지 않는다.
- 전체 의료·PII에서 규칙을 일괄 대체한다는 근거는 아직 없다. human-gold privacy와 실제 비정형 복합 오염 평가가 별도로 필요하다.
