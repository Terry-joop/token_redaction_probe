from __future__ import annotations

import json
from pathlib import Path


LABELS = {
    "raw_rule": "원시 규칙",
    "normalized_rule": "정규화+규칙",
    "student": "ELECTRA-small",
    "hybrid_raw_rule_or_student": "규칙 OR Student",
}


def render(root: Path, meta: dict) -> str:
    path = root / "reports" / "robustness_v14_disjoint_four_systems.json"
    if not path.exists():
        return ""
    source = json.loads(path.read_text(encoding="utf-8"))
    systems = list(LABELS)
    macro_rows = "".join(
        f"<tr><td class='left meta dataset'>{LABELS[key]}</td>"
        f"<td>{value['clean_f2']:.3f}</td><td>{value['noisy_precision']:.3f}</td>"
        f"<td>{value['noisy_recall']:.3f}</td><td>{value['noisy_f1']:.3f}</td>"
        f"<td>{value['noisy_f2']:.3f}</td><td>{value['noisy_mask_rate']*100:.1f}%</td>"
        f"<td>{value['target_detection']*100:.1f}%</td></tr>"
        for key, value in source["macro"].items()
    )
    dataset_rows = []
    direct_rows = []
    for key, item in source["datasets"].items():
        values = item["systems"]
        audit = item["split_audit"]["removed"]
        dataset_rows.append(
            f"<tr><td class='left meta dataset'>{meta[key][0]}"
            f"<span class='task'>train {audit['train']['before']:,}→{audit['train']['after']:,}</span></td>"
            f"<td>{item['pairs']:,}</td>"
            + "".join(
                f"<td>{values[name]['robustness']['noisy_target_detection']*100:.1f}%</td>"
                for name in systems
            )
            + f"<td>{values['student']['clean']['f2']:.3f}</td>"
            f"<td>{values['hybrid_raw_rule_or_student']['noisy']['predicted_mask_rate']*100:.1f}%</td></tr>"
        )
        normalized = values["normalized_rule"]
        student_value = values["student"]
        f2_delta = student_value["noisy"]["f2"] - normalized["noisy"]["f2"]
        target_delta = (
            student_value["robustness"]["noisy_target_detection"]
            - normalized["robustness"]["noisy_target_detection"]
        )
        mask_delta = (
            student_value["noisy"]["predicted_mask_rate"]
            - normalized["noisy"]["predicted_mask_rate"]
        )
        if f2_delta > 0 and target_delta > 0:
            verdict = "Student 우세"
        elif f2_delta < 0 and target_delta < 0:
            verdict = "정규화 규칙 우세"
        else:
            verdict = "지표별 우세가 다름"
        direct_rows.append(
            f"<tr><td class='left meta dataset'>{meta[key][0]}</td><td>{item['pairs']:,}</td>"
            f"<td>{normalized['noisy']['f2']:.3f}</td><td>{student_value['noisy']['f2']:.3f}</td>"
            f"<td>{f2_delta:+.3f}</td>"
            f"<td>{normalized['robustness']['noisy_target_detection']*100:.1f}%</td>"
            f"<td>{student_value['robustness']['noisy_target_detection']*100:.1f}%</td>"
            f"<td>{target_delta*100:+.1f}%p</td>"
            f"<td>{normalized['noisy']['predicted_mask_rate']*100:.1f}%</td>"
            f"<td>{student_value['noisy']['predicted_mask_rate']*100:.1f}%</td>"
            f"<td>{mask_delta*100:+.1f}%p</td><td class='left'>{verdict}</td></tr>"
        )
    noise_rows = []
    for noise, item in source["by_noise"].items():
        noise_rows.append(
            f"<tr><td class='left meta dataset'>{noise}</td>"
            f"<td>{item['raw_rule']['pairs']:,}</td>"
            + "".join(
                f"<td>{item[name]['noisy_target_detection']*100:.1f}%</td>"
                for name in systems
            )
            + "</tr>"
        )
    raw = source["macro"]["raw_rule"]
    norm = source["macro"]["normalized_rule"]
    student = source["macro"]["student"]
    hybrid = source["macro"]["hybrid_raw_rule_or_student"]
    removed = sum(
        values["before"] - values["after"]
        for item in source["datasets"].values()
        for values in item["split_audit"]["removed"].values()
    )
    return (
        "<h2>4-4. split 누수 제거 후 네 방식 비교</h2>"
        f"<p class='lede'>정규화 중복 {removed:,}행을 제거해 split 교집합을 0으로 만든 뒤, 동일 unseen-7 pair에서 네 방식을 비교했다. P/R/F1/F2/Mask는 10개 데이터셋 macro, target 탐지는 352,905개 pair 가중 평균이다.</p>"
        "<div class='notice'><strong>비교 기준:</strong> Clean 최신 v1.4 규칙 span이 고정 정답이다. 정규화는 교란 종류·위치를 입력받지 않고 NFKC·공백·아포스트로피·제어문자·숫자/용량 경계만 처리한다. Hybrid는 원시 규칙과 Student mask의 합집합이다.</div>"
        "<h3>4-4-1. 10개 데이터셋 종합</h3><div class='tablewrap solo'><table><thead><tr><th class='left'>방식</th><th>Clean F2</th><th>Noisy P</th><th>Noisy R</th><th>Noisy F1</th><th>Noisy F2</th><th>Mask</th><th>오염 target 탐지</th></tr></thead><tbody>"
        + macro_rows
        + "</tbody></table></div>"
        "<h3>4-4-2. 정규화+규칙 vs Student 직접 비교</h3>"
        "<p class='lede'>Δ는 Student−정규화 규칙이다. F2와 target Δ가 모두 음수면 정규화 규칙이 두 핵심 지표에서 우세하다는 뜻이다.</p>"
        "<div class='tablewrap solo'><table><thead><tr><th class='left'>데이터셋</th><th>Pair</th><th>정규화 F2</th><th>Student F2</th><th>ΔF2</th><th>정규화 target</th><th>Student target</th><th>Δtarget</th><th>정규화 Mask</th><th>Student Mask</th><th>ΔMask</th><th class='left'>판단</th></tr></thead><tbody>"
        + "".join(direct_rows)
        + "</tbody></table></div>"
        "<h3>4-4-3. 데이터셋별 네 방식 target 탐지</h3><div class='tablewrap solo'><table><thead><tr><th class='left'>데이터셋</th><th>Pair</th><th>원시 규칙</th><th>정규화 규칙</th><th>Student</th><th>Hybrid</th><th>Student Clean F2</th><th>Hybrid Mask</th></tr></thead><tbody>"
        + "".join(dataset_rows)
        + "</tbody></table></div>"
        "<h3>4-4-4. 학습에 없던 교란별 target 탐지</h3><div class='tablewrap solo'><table><thead><tr><th class='left'>교란</th><th>Pair</th><th>원시 규칙</th><th>정규화 규칙</th><th>Student</th><th>Hybrid</th></tr></thead><tbody>"
        + "".join(noise_rows)
        + "</tbody></table></div>"
        f"<div class='analysis-grid'><article class='analysis-card'><h3>전처리 baseline</h3><p>정규화 규칙 macro noisy F2는 <strong>{norm['noisy_f2']:.3f}</strong>, target 탐지는 <strong>{norm['target_detection']*100:.1f}%</strong>다. 원시 규칙 대비 각각 {norm['noisy_f2']-raw['noisy_f2']:+.3f}, {(norm['target_detection']-raw['target_detection'])*100:+.1f}%p다.</p></article>"
        f"<article class='analysis-card'><h3>Student 단독 대체</h3><p>Student noisy F2 <strong>{student['noisy_f2']:.3f}</strong>, target 탐지 <strong>{student['target_detection']*100:.1f}%</strong>를 정규화 규칙과 비교한다. clean fidelity와 마스킹 예산도 함께 충족해야 대체 근거가 된다.</p></article>"
        f"<article class='analysis-card'><h3>Hybrid 절충</h3><p>Hybrid target 탐지는 <strong>{hybrid['target_detection']*100:.1f}%</strong>, mask는 <strong>{hybrid['noisy_mask_rate']*100:.1f}%</strong>다. 누락 감소와 과다 마스킹을 함께 본다.</p></article>"
        "<article class='analysis-card'><h3>주장의 범위</h3><p>최신 규칙 모방과 표면 교란 강건성 결과다. 실제 PII 타당성은 human-gold, 최종 privacy 효과는 RTM 복구율로 별도 검증해야 한다.</p></article></div>"
    )
