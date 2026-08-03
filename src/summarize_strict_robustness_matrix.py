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
