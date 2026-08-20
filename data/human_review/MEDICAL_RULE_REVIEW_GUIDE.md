# 의료 규칙 검수 방법

`medical_rule_review_v1.html`을 먼저 열어 노랑으로 표시된 규칙 mask를 읽습니다. 수정은 `medical_rule_review_v1.jsonl`에서 합니다.

- 가려야 하는 단어: `human_labels`의 해당 위치를 `1`
- 가리면 안 되는 단어: 해당 위치를 `0`
- 한 문장을 확인했으면 `human_reviewed`를 `true`
- 규칙의 문제를 한 줄로 `human_review_reason`에 기록

`words`와 `human_labels`는 반드시 같은 길이를 유지해야 합니다.
