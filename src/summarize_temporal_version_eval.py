"""Summarize the actual RedactFormer v1.2 -> v1.3/v1.4 replay."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "drug": {
        "name": "Drug Reviews",
        "domain": "의료·약물 리뷰",
        "model_dir": ROOT / "artifacts/temporal_v12_v14/drug_electra_small_seed42",
    },
    "bios": {
        "name": "BIOS",
        "domain": "실제 PII·인명/연락처",
        "model_dir": ROOT / "artifacts/temporal_v12_v14/bios_electra_small_seed42",
    },
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verdict(delta: float, ci: list[float]) -> str:
    if ci[0] > 0:
        return "Student 우세"
    if ci[1] < 0:
        return "v1.2 규칙 우세"
    if delta == 0:
        return "동률"
    return "불확실"


def main() -> None:
    datasets = {}
    markdown = [
        "# 실제 규칙 버전 시간축 평가 결과",
        "",
        "v1.2 규칙과 v1.2 라벨만 학습한 ELECTRA-small Student를, 이후 "
        "v1.3/v1.4에서 실제 추가된 패치 target에 평가한 결과다.",
        "",
        "| 데이터셋 | Future target | Clean F2 | v1.2 규칙 탐지 | v1.2 Student 탐지 | 차이 (95% CI) | 판정 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for key, metadata in DATASETS.items():
        result_path = ROOT / f"reports/temporal_version_{key}_seed42.json"
        evaluation_path = metadata["model_dir"] / "medical_evaluation.json"
        if not result_path.exists() or not evaluation_path.exists():
            continue
        result = load(result_path)
        evaluation = load(evaluation_path)
        clean = evaluation["budget_matched"]["test"]
        past_rule = result["systems"]["past_rule_v1_2"]
        past_student = result["systems"]["past_student_v1_2"]
        latest = result["systems"]["latest_rule_v1_4"]
        delta = result["student_minus_past_rule_target_detection"]
        ci = result["student_minus_past_rule_ci95"]
        item = {
            **metadata,
            "model_dir": str(metadata["model_dir"].relative_to(ROOT)),
            "candidate_targets": result["candidate_targets"],
            "targets": result["latest_validated_targets"],
            "unique_sources": result["unique_sources"],
            "clean": clean,
            "past_rule_detection": past_rule["target_detection"],
            "past_student_detection": past_student["target_detection"],
            "latest_rule_detection": latest["target_detection"],
            "student_minus_rule": delta,
            "student_minus_rule_ci95": ci,
            "verdict": verdict(delta, ci),
            "student_token_audit_against_latest": past_student[
                "token_audit_against_latest_rule"
            ],
            "by_defect": result["by_defect"],
        }
        datasets[key] = item
        markdown.append(
            f"| {metadata['name']} | {item['targets']:,} | {clean['f2']:.3f} | "
            f"{item['past_rule_detection']*100:.1f}% | "
            f"{item['past_student_detection']*100:.1f}% | "
            f"{delta*100:+.1f}%p [{ci[0]*100:+.1f}, {ci[1]*100:+.1f}] | "
            f"{item['verdict']} |"
        )

    if not datasets:
        raise FileNotFoundError("No completed temporal-version result/evaluation pairs")

    wins = sum(item["verdict"] == "Student 우세" for item in datasets.values())
    if wins:
        conclusion = (
            f"{wins}/{len(datasets)}개 데이터셋에서 Student가 통계적으로 우세했다. "
            "우세 결함에 한해 규칙 유지보수 지연을 보완할 가능성이 있다."
        )
    else:
        conclusion = (
            "두 데이터셋 모두 v1.2 규칙이 통계적으로 우세했다. 높은 clean 모방 "
            "성능만으로 미래 결함 일반화가 생긴다는 가설은 이번 실험에서 지지되지 않았다."
        )
    markdown.extend(["", "## 주 결론", "", conclusion, "", "## 결함별 결과", ""])
    defect_names = {
        "glued_dosage": "붙여 쓴 용량",
        "c1_control": "C1 제어문자",
        "possessive": "소유격",
        "long_identifier": "긴 숫자·식별자",
        "email": "이메일",
        "url": "URL",
        "social_handle": "소셜 핸들",
        "zip4": "ZIP+4",
        "numeric_date": "숫자 날짜",
    }
    markdown.extend([
        "| 데이터셋 | 추가 버전 | 결함 | Target | v1.2 규칙 | v1.2 Student | 차이 (95% CI) |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for item in datasets.values():
        for defect, row in item["by_defect"].items():
            ci = row["student_minus_rule_ci95"]
            markdown.append(
                f"| {item['name']} | v{row['introduced_in']} | "
                f"{defect_names.get(defect, defect)} | {row['targets']:,} | "
                f"{row['past_rule_detection']*100:.1f}% | "
                f"{row['past_student_detection']*100:.1f}% | "
                f"{row['student_minus_rule']*100:+.1f}%p "
                f"[{ci[0]*100:+.1f}, {ci[1]*100:+.1f}] |"
            )

    markdown.extend([
        "",
        "## 해석 제한",
        "",
        "- 정답은 human-gold가 아니라 최신 v1.4가 해당 문자 구간을 민감하다고 "
        "확인한 pseudo-gold다.",
        "- 과거 Student의 학습과 threshold 선택에는 v1.3/v1.4 target을 사용하지 않았다.",
        "- Student가 이기더라도 최신 규칙 전체 대체가 아니라, 규칙 패치 전 일부 "
        "표면형에 일반화하여 유지보수 지연을 줄일 가능성만 주장한다.",
        "- 세부 설계는 `TEMPORAL_RULE_VERSION_EVAL.md`에 있다.",
    ])

    summary = {
        "protocol": "actual RedactFormer rule-version replay: v1.2 -> v1.3/v1.4",
        "past_rule_commit": "b8dff7e",
        "latest_reference_commit": "045f3f3",
        "student": "ELECTRA-small + hidden-128 MLP, encoder fine-tuned, seed 42",
        "datasets": datasets,
    }
    (ROOT / "reports/temporal_rule_version_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ROOT / "TEMPORAL_RULE_VERSION_RESULTS.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    print(json.dumps({"datasets": list(datasets)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
