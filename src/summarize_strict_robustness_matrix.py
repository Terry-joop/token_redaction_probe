from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]
DATASETS = {
    "drug": ("Drug Reviews", "medical", "의료 규칙", "medterm5 v1.4"),
    "symptom2dx": ("Symptom2Dx", "medical", "의료 규칙", "medterm5 v1.4"),
    "adr": ("ADR", "medical", "의료 규칙", "medterm5 v1.4"),
    "redditmh": ("RedditMH", "medical", "의료 규칙", "medterm5 v1.4"),
    "mednli": ("MedNLI", "medical", "의료 규칙", "medterm5 v1.4"),
    "mentalhealth": ("Mental Health", "medical", "의료 규칙", "medterm5 v1.4"),
    "bios": ("BIOS", "pii", "실제 PII", "piiclean2 v1.4"),
    "mrpc": ("MRPC", "pii", "실제 PII", "piiclean2 strict v1.4"),
    "qnli": ("QNLI", "entity", "비개인 엔티티 대조", "piiclean2 v1.4"),
    "finphrasebank": ("FinPhraseBank", "entity", "비개인 엔티티 대조", "piiclean2 v1.4"),
}
DATASET_MEANINGS = {
    "drug": (
        "실제 의료 privacy 도메인에서 Student가 규칙의 표면 이음매를 보완할 수 있다는 "
        "가장 강한 근거다. 다만 전체 noisy token F2는 규칙보다 낮으므로 독립 대체가 "
        "아니라 규칙과 병렬로 쓰는 보완 후보로 해석한다."
    ),
    "symptom2dx": (
        "평균값은 Student 쪽이 약간 좋지만 test와 target-pair가 작고 3-seed CI gate를 "
        "통과하지 못했다. 표본을 늘려 재검증하기 전에는 우세라고 주장하지 않는다."
    ),
    "adr": (
        "Student의 하락폭은 작지만 오염 후 절대 탐지율은 규칙보다 낮다. 낮은 clean "
        "시작점 때문에 덜 하락해 보이는 효과가 섞였으므로 규칙 대체 근거가 아니다."
    ),
    "redditmh": (
        "비정형 정신건강 서술에서 clean 품질 gate부터 통과하지 못했고 오염 후에도 "
        "규칙보다 낮다. 현재 모델·라벨 구성으로는 규칙 보완이나 대체를 주장할 수 없다."
    ),
    "mednli": (
        "clean 규칙 모방은 합격했지만 학습에 없던 표면 교란에서는 규칙의 절대 "
        "탐지율이 더 높다. 임상 문장쌍에 대한 추가 증강 없이는 규칙 유지가 타당하다."
    ),
    "mentalhealth": (
        "clean F2와 오염 후 절대 탐지율이 모두 가장 약한 축에 속한다. 자유서술 표현의 "
        "다양성을 현재 단일 ELECTRA-small이 충분히 학습하지 못한 실패 사례다."
    ),
    "bios": (
        "가장 큰 실제 PII 평가이므로 중요한 반례다. Student가 덜 하락하더라도 오염 후 "
        "절대 탐지율은 규칙보다 낮아, 규모만 늘리는 것으로 규칙을 이기지는 못했다."
    ),
    "mrpc": (
        "엄격 PII 정책의 clean 모방은 좋지만 오염 후에는 규칙이 앞선다. Student를 "
        "단독 필터로 교체하기보다 규칙의 미탐 후보를 재검사하는 보조기로 보는 편이 맞다."
    ),
    "qnli": (
        "규칙과 Student의 차이는 작지만 비개인 엔티티 대조군이므로 privacy 성과가 "
        "아니다. 학습형 모델이 일반 엔티티 경계를 어느 정도 유지하는지 보는 통제 결과다."
    ),
    "finphrasebank": (
        "수치와 3-seed CI에서는 Student가 규칙보다 강건하지만 비개인 엔티티 대조군이다. "
        "학습형 redactor의 표면 일반화 가능성은 지지해도 개인정보 보호 성공으로 세지 않는다."
    ),
}
FIELDS = [
    "clean_precision",
    "clean_recall",
    "clean_f1",
    "clean_f2",
    "rule_clean_target_detection",
    "rule_noisy_target_detection",
    "rule_detection_drop",
    "student_clean_target_detection",
    "student_noisy_target_detection",
    "student_detection_drop",
    "student_minus_rule_noisy",
    "student_drop_advantage",
    "student_noisy_f2",
    "rule_noisy_f2",
]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(runs: list[dict], field: str) -> dict:
    values = [run[field] for run in runs]
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def build() -> dict:
    datasets = {}
    for key, (name, group, group_name, teacher) in DATASETS.items():
        data_dir = ROOT / "data" / "robustness" / "v14_strict" / key
        clean_summary = load(data_dir / "clean" / "summary.json")
        augmentation = load(data_dir / "train_seen_augmented.summary.json")
        pairs = load(data_dir / "unseen_pairs.summary.json")
        runs = []
        for seed in SEEDS:
            robust = load(ROOT / "reports" / f"robustness_v14_strict_{key}_seed{seed}.json")
            calibration = load(
                ROOT
                / "artifacts"
                / "robustness"
                / "v14_strict"
                / f"{key}_electra_small_seed{seed}"
                / "medical_evaluation.json"
            )["budget_matched"]["test"]
            absolute = robust["absolute_target_robustness"]
            runs.append(
                {
                    "seed": seed,
                    "threshold": calibration["threshold"],
                    "clean_precision": calibration["precision"],
                    "clean_recall": calibration["recall"],
                    "clean_f1": calibration["f1"],
                    "clean_f2": calibration["f2"],
                    "clean_mask_rate": calibration["predicted_mask_rate"],
                    "final_quality_pass": (
                        calibration["f1"] >= 0.85
                        and calibration["f2"] >= 0.90
                        and calibration["recall"] >= 0.90
                    ),
                    "rule_clean_target_detection": absolute["rule_clean_target_detection"],
                    "rule_noisy_target_detection": absolute["rule_noisy_target_detection"],
                    "rule_detection_drop": absolute["rule_detection_drop"],
                    "student_clean_target_detection": absolute["student_clean_target_detection"],
                    "student_noisy_target_detection": absolute["student_noisy_target_detection"],
                    "student_detection_drop": absolute["student_detection_drop"],
                    "student_minus_rule_noisy": absolute["student_minus_rule_noisy"],
                    "student_minus_rule_noisy_ci95": absolute["student_minus_rule_noisy_ci95"],
                    "student_drop_advantage": absolute["student_drop_advantage"],
                    "student_drop_advantage_ci95": absolute["student_drop_advantage_ci95"],
                    "student_noisy_f2": robust["student"]["noisy"]["f2"],
                    "rule_noisy_f2": robust["rule_v1_4"]["noisy"]["f2"],
                    "absolute_gate_pass": robust["acceptance"][
                        "absolute_target_robustness_gate"
                    ]["pass"],
                }
            )
        datasets[key] = {
            "name": name,
            "group": group,
            "group_name": group_name,
            "teacher": teacher,
            "meaning": DATASET_MEANINGS[key],
            "policy_id": clean_summary["policy"],
            "splits": {
                split: clean_summary["splits"][split]["examples"]
                for split in ("train", "validation", "test")
            },
            "clean_train_rows": augmentation["clean_rows"],
            "augmented_train_rows": augmentation["augmented_rows"],
            "total_train_rows": augmentation["total_rows"],
            "seen_counts": augmentation["selected_by_noise"],
            "unseen_pairs": pairs["pairs"],
            "unique_test_sources": pairs["unique_source_rows"],
            "unseen_counts": pairs["counts"],
            "runs": runs,
            "absolute_gate_pass_seeds": sum(
                run["absolute_gate_pass"] for run in runs
            ),
            "quality_gate_pass_seeds": sum(
                run["final_quality_pass"] for run in runs
            ),
            "summary": {field: mean_std(runs, field) for field in FIELDS},
        }

    group_summary = {}
    for group in ("medical", "pii", "entity"):
        items = [item for item in datasets.values() if item["group"] == group]
        group_summary[group] = {
            "datasets": len(items),
            "metrics": {
                field: statistics.fmean(
                    item["summary"][field]["mean"] for item in items
                )
                for field in FIELDS
            },
        }
    return {
        "protocol": {
            "design": "seen-5 train augmentation / unseen-7 final test",
            "student": "google/electra-small-discriminator + hidden-128 MLP",
            "seeds": SEEDS,
            "epochs": 5,
            "batch_size": 32,
            "max_length": 128,
            "train_scope": "all clean train rows plus at most one balanced seen perturbation per eligible row",
            "test_scope": "all eligible unseen perturbations from the full test split",
            "threshold": "validation mask-rate matched, then fixed for test",
            "gold": "latest v1.4 clean rule target projected through deterministic edits",
            "bootstrap": "2000 source-cluster resamples per seed",
        },
        "datasets": datasets,
        "groups": group_summary,
    }


def markdown(result: dict) -> str:
    items = list(result["datasets"].values())
    pairs = sum(item["unseen_pairs"] for item in items)
    sources = sum(item["unique_test_sources"] for item in items)
    clean_rows = sum(item["clean_train_rows"] for item in items)
    augmented_rows = sum(item["augmented_train_rows"] for item in items)
    average_wins = [
        item["name"]
        for item in items
        if item["summary"]["student_minus_rule_noisy"]["mean"] > 0
        and item["summary"]["student_drop_advantage"]["mean"] > 0
    ]
    strict_wins = [
        item["name"] for item in items if item["absolute_gate_pass_seeds"] == 3
    ]
    clean_passes = [
        item["name"] for item in items if item["quality_gate_pass_seeds"] == 3
    ]
    lines = [
        "# 전 데이터셋 strict 5/7 입력 교란 실험",
        "",
        "학습에는 Seen 5종만 넣고, 최종 test에는 학습에서 한 번도 쓰지 않은 Unseen 7종만 사용했다. Student는 ELECTRA-small로 고정하고 seed 42·43·44를 반복했다.",
        "",
        "## 한눈에 보는 결과",
        "",
        f"- 10개 데이터셋의 clean train {clean_rows:,}행에 Seen 증강 {augmented_rows:,}행을 추가했다.",
        f"- 전체 test에서 적용 가능한 Unseen target-pair {pairs:,}개, 고유 원문 {sources:,}개를 평가했다.",
        f"- clean 품질 gate(F1≥0.85, F2≥0.90, Recall≥0.90)를 세 seed 모두 통과한 데이터셋은 {len(clean_passes)}/10개다: {', '.join(clean_passes)}.",
        f"- 평균 noisy 탐지와 하락폭이 모두 좋은 데이터셋은 {len(average_wins)}/10개다: {', '.join(average_wins)}.",
        f"- 두 차이의 95% CI가 세 seed 모두 0보다 큰 엄격 우세는 {len(strict_wins)}/10개다: {', '.join(strict_wins)}.",
        "",
        "| 그룹 | 데이터셋 | Train clean+증강 | Test 원문 | Unseen target pair | Clean F2 | 규칙 noisy 탐지 | Student noisy 탐지 | Student−규칙 | 규칙 하락 | Student 하락 | Clean gate | CI gate |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in items:
        s = item["summary"]
        lines.append(
            f"| {item['group_name']} | {item['name']} | {item['clean_train_rows']:,}+{item['augmented_train_rows']:,} | "
            f"{item['splits']['test']:,} | {item['unseen_pairs']:,} | "
            f"{s['clean_f2']['mean']:.3f}±{s['clean_f2']['sample_std']:.3f} | "
            f"{s['rule_noisy_target_detection']['mean']:.3f} | "
            f"{s['student_noisy_target_detection']['mean']:.3f}±{s['student_noisy_target_detection']['sample_std']:.3f} | "
            f"{s['student_minus_rule_noisy']['mean']:+.3f} | "
            f"{s['rule_detection_drop']['mean']:.3f} | "
            f"{s['student_detection_drop']['mean']:.3f}±{s['student_detection_drop']['sample_std']:.3f} | "
            f"{item['quality_gate_pass_seeds']}/3 | {item['absolute_gate_pass_seeds']}/3 |"
        )
    lines.extend([
        "",
        "## 데이터셋별 수치 해석",
        "",
        "아래 target 탐지율은 clean 최신 규칙이 고른 민감 span 전체를 오염 문장에서도 "
        "전부 가린 비율이다. noisy token F2는 오염 문장의 모든 토큰을 대상으로 한 별도 "
        "지표이므로, 특정 target의 생존율과 같은 숫자로 해석하면 안 된다.",
        "",
    ])
    for item in items:
        s = item["summary"]
        advantage = s["student_minus_rule_noisy"]["mean"]
        drop_advantage = s["student_drop_advantage"]["mean"]
        if item["absolute_gate_pass_seeds"] == 3:
            verdict = "엄격 우세"
        elif advantage > 0 and drop_advantage > 0:
            verdict = "평균 우세지만 통계적 재현성 미달"
        elif item["quality_gate_pass_seeds"] < 3:
            verdict = "clean 품질 및 규칙 대비 성능 미달"
        else:
            verdict = "오염 후 절대 탐지율에서 규칙 우세"
        lines.extend([
            f"### {item['name']} — {verdict}",
            "",
            f"- **평가 규모:** test 원문 {item['splits']['test']:,}개에서 적용 가능한 "
            f"Unseen target-pair {item['unseen_pairs']:,}개를 평가했다.",
            f"- **Clean 모방:** Student F2 {s['clean_f2']['mean']:.3f}±"
            f"{s['clean_f2']['sample_std']:.3f}, clean gate {item['quality_gate_pass_seeds']}/3이다.",
            f"- **오염 후 target 탐지:** 규칙 {s['rule_noisy_target_detection']['mean']*100:.1f}% "
            f"대 Student {s['student_noisy_target_detection']['mean']*100:.1f}%±"
            f"{s['student_noisy_target_detection']['sample_std']*100:.1f}%p로, Student−규칙은 "
            f"{advantage*100:+.1f}%p다.",
            f"- **Clean→오염 하락:** 규칙 {s['rule_detection_drop']['mean']*100:.1f}%p 대 "
            f"Student {s['student_detection_drop']['mean']*100:.1f}%p이며 Student의 하락폭 이점은 "
            f"{drop_advantage*100:+.1f}%p다. noisy token F2는 규칙 "
            f"{s['rule_noisy_f2']['mean']:.3f}, Student {s['student_noisy_f2']['mean']:.3f}이다.",
            f"- **의미:** {item['meaning']} 엄격 CI gate는 "
            f"{item['absolute_gate_pass_seeds']}/3이다.",
            "",
        ])
    lines.extend([
        "",
        "## 해석",
        "",
        "- noisy 탐지는 clean 최신 규칙이 선택한 고정 target span을 오염 문장에서 전부 가렸는지 본다.",
        "- Student−규칙이 양수이고 Student 하락이 더 작아야 규칙보다 표면 교란에 강하다고 본다. 하락폭만 작은 것은 Student의 clean 시작점이 낮아서 생길 수도 있어 성공으로 세지 않는다.",
        "- 엄격 우세는 위 두 차이 모두의 source-cluster bootstrap 95% CI 하한이 0보다 큰 상태를 seed 3개에서 모두 재현한 경우다.",
        "- Drug Reviews는 실제 의료 privacy 규칙에서 조건부 보완 근거를 보였지만, FinPhraseBank는 비개인 엔티티 대조이므로 privacy 성공으로 해석하지 않는다.",
        "- 전체 의료·PII에서 규칙을 일괄 대체한다는 근거는 아직 없다. human-gold privacy와 실제 비정형 복합 오염 평가가 별도로 필요하다.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    result = build()
    json_path = ROOT / "reports" / "robustness_v14_strict_matrix.json"
    md_path = ROOT / "STRICT_ROBUSTNESS_MATRIX.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown(result), encoding="utf-8")
    print(f"wrote {json_path} and {md_path}; datasets={len(result['datasets'])}")


if __name__ == "__main__":
    main()
