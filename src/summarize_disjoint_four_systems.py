from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_strict_robustness_matrix import DATASETS, ROOT


EXPERIMENT = "v14_disjoint"
SEEDS = (42, 43, 44)
SYSTEMS = (
    "raw_rule",
    "normalized_rule",
    "student",
    "hybrid_raw_rule_or_student",
)
LABELS = {
    "raw_rule": "원시 규칙",
    "normalized_rule": "정규화+규칙",
    "student": "ELECTRA-small",
    "hybrid_raw_rule_or_student": "규칙 OR Student",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values) -> float:
    return float(np.mean(list(values)))


def main() -> None:
    audit = load(ROOT / "reports" / "split_overlap_audit_v14.json")
    datasets = {}
    for dataset in DATASETS:
        runs = [
            load(ROOT / "reports" / f"four_system_{EXPERIMENT}_{dataset}_seed{seed}.json")
            for seed in SEEDS
        ]
        system_summary = {}
        for system in SYSTEMS:
            system_summary[system] = {
                split: {
                    metric: mean(run["systems"][system][split][metric] for run in runs)
                    for metric in (
                        "precision", "recall", "f1", "f2", "predicted_mask_rate",
                        "residual_sensitive_rate", "overmask_rate",
                    )
                }
                for split in ("clean", "noisy")
            }
            system_summary[system]["robustness"] = {
                metric: mean(run["systems"][system]["robustness"][metric] for run in runs)
                for metric in (
                    "clean_target_detection", "noisy_target_detection",
                    "span_survival_rate", "newly_leaked_span_rate",
                )
            }
            noises = sorted(runs[0]["systems"][system]["target_by_noise"])
            system_summary[system]["by_noise"] = {}
            for noise in noises:
                system_summary[system]["by_noise"][noise] = {
                    "pairs": runs[0]["systems"][system]["target_by_noise"][noise]["pairs"],
                    "clean_target_detection": mean(
                        run["systems"][system]["target_by_noise"][noise]["clean_target_detection"]
                        for run in runs
                    ),
                    "noisy_target_detection": mean(
                        run["systems"][system]["target_by_noise"][noise]["noisy_target_detection"]
                        for run in runs
                    ),
                    "detection_drop": mean(
                        run["systems"][system]["target_by_noise"][noise]["detection_drop"]
                        for run in runs
                    ),
                    "noisy_f2": mean(
                        run["systems"][system]["by_noise"][noise]["f2"] for run in runs
                    ),
                }
        datasets[dataset] = {
            "pairs": runs[0]["pairs"],
            "unique_source_rows": runs[0]["unique_source_rows"],
            "systems": system_summary,
            "split_audit": audit["results"][dataset],
        }

    macro = {}
    for system in SYSTEMS:
        macro[system] = {
            "clean_f2": mean(datasets[d]["systems"][system]["clean"]["f2"] for d in datasets),
            "noisy_precision": mean(datasets[d]["systems"][system]["noisy"]["precision"] for d in datasets),
            "noisy_recall": mean(datasets[d]["systems"][system]["noisy"]["recall"] for d in datasets),
            "noisy_f1": mean(datasets[d]["systems"][system]["noisy"]["f1"] for d in datasets),
            "noisy_f2": mean(datasets[d]["systems"][system]["noisy"]["f2"] for d in datasets),
            "noisy_mask_rate": mean(datasets[d]["systems"][system]["noisy"]["predicted_mask_rate"] for d in datasets),
            "target_detection": sum(
                datasets[d]["systems"][system]["robustness"]["noisy_target_detection"]
                * datasets[d]["pairs"] for d in datasets
            ) / sum(datasets[d]["pairs"] for d in datasets),
        }

    noise_rollup = defaultdict(dict)
    for noise in sorted(
        {noise for data in datasets.values() for noise in data["systems"]["raw_rule"]["by_noise"]}
    ):
        for system in SYSTEMS:
            entries = [
                data["systems"][system]["by_noise"][noise]
                for data in datasets.values()
                if noise in data["systems"][system]["by_noise"]
            ]
            total = sum(entry["pairs"] for entry in entries)
            noise_rollup[noise][system] = {
                "pairs": total,
                "noisy_target_detection": sum(
                    entry["noisy_target_detection"] * entry["pairs"] for entry in entries
                ) / total,
                "dataset_macro_noisy_f2": mean(entry["noisy_f2"] for entry in entries),
            }

    result = {
        "experiment": EXPERIMENT,
        "datasets": datasets,
        "macro": macro,
        "by_noise": noise_rollup,
        "split_priority": audit["priority"],
        "split_normalization": audit["normalization"],
        "interpretation_scope": (
            "Teacher-rule fidelity and surface robustness; not human-gold privacy validity"
        ),
    }
    output = ROOT / "reports" / "robustness_v14_disjoint_four_systems.json"
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# 중복 제거 후 네 방식 입력 교란 비교", "",
        "평가 순서: split 중복 감사 → 누수 제거 재학습 → 원시 규칙/정규화 규칙/Student/Hybrid 비교 → 교란별 분석.",
        "P/R/F1/F2/Mask는 10개 데이터셋 macro 평균이고, 오염 target 탐지는 352,905개 pair 가중 평균이다.", "",
        "## 1. split 중복 감사", "",
        "| 데이터셋 | Train 전→후 | Validation 전→후 | Test 전→후 | 재구성 후 split 교집합 |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, data in datasets.items():
        removed = data["split_audit"]["removed"]
        overlap = sum(
            value["normalized_unique_keys"]
            for value in data["split_audit"]["after"]["pairwise"].values()
        )
        lines.append(
            f"| {dataset} | {removed['train']['before']:,}→{removed['train']['after']:,} | "
            f"{removed['validation']['before']:,}→{removed['validation']['after']:,} | "
            f"{removed['test']['before']:,}→{removed['test']['after']:,} | {overlap} |"
        )
    lines += ["", "## 2. 전체 10개 데이터셋 요약", "",
        "| 방식 | Clean F2 | Noisy P | Noisy R | Noisy F1 | Noisy F2 | Mask | 오염 target 탐지 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        value = macro[system]
        lines.append(
            f"| {LABELS[system]} | {value['clean_f2']:.3f} | {value['noisy_precision']:.3f} | "
            f"{value['noisy_recall']:.3f} | {value['noisy_f1']:.3f} | {value['noisy_f2']:.3f} | "
            f"{value['noisy_mask_rate']:.1%} | {value['target_detection']:.1%} |"
        )
    lines += ["", "## 3. 정규화+규칙 vs Student 직접 비교", "",
        "Δ는 Student−정규화 규칙이다. F2와 target Δ가 모두 음수면 정규화 규칙이 두 핵심 지표에서 우세하다.", "",
        "| 데이터셋 | Pair | 정규화 F2 | Student F2 | ΔF2 | 정규화 target | Student target | Δtarget | 정규화 Mask | Student Mask | ΔMask | 판단 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, data in datasets.items():
        systems = data["systems"]
        normalized = systems["normalized_rule"]
        student_value = systems["student"]
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
        lines.append(
            f"| {dataset} | {data['pairs']:,} | "
            f"{normalized['noisy']['f2']:.3f} | {student_value['noisy']['f2']:.3f} | {f2_delta:+.3f} | "
            f"{normalized['robustness']['noisy_target_detection']:.1%} | {student_value['robustness']['noisy_target_detection']:.1%} | "
            f"{target_delta:+.1%}p | {normalized['noisy']['predicted_mask_rate']:.1%} | "
            f"{student_value['noisy']['predicted_mask_rate']:.1%} | {mask_delta:+.1%}p | {verdict} |"
        )
    lines += ["", "## 4. 데이터셋별 네 방식 오염 target 탐지율", "",
        "| 데이터셋 | Pair | 원시 규칙 | 정규화+규칙 | Student | 규칙 OR Student | Student Clean F2 | Student-정규화 | 해석 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for dataset, data in datasets.items():
        systems = data["systems"]
        normalized_target = systems['normalized_rule']['robustness']['noisy_target_detection']
        student_target = systems['student']['robustness']['noisy_target_detection']
        target_delta = student_target - normalized_target
        if target_delta > 0.005:
            verdict = "Student가 정규화 규칙보다 target 탐지 우세"
        elif target_delta < -0.005:
            verdict = "알려진 결함은 정규화 규칙이 우세"
        else:
            verdict = "탐지 차이 0.5%p 이내"
        lines.append(
            f"| {dataset} | {data['pairs']:,} | "
            f"{systems['raw_rule']['robustness']['noisy_target_detection']:.1%} | "
            f"{normalized_target:.1%} | "
            f"{student_target:.1%} | "
            f"{systems['hybrid_raw_rule_or_student']['robustness']['noisy_target_detection']:.1%} | "
            f"{systems['student']['clean']['f2']:.3f} | {target_delta:+.1%}p | {verdict} |"
        )
    lines += ["", "## 5. 학습에 없던 교란 종류별 결과", "",
        "| 교란 | Pair | 원시 규칙 탐지 | 정규화 규칙 탐지 | Student 탐지 | Hybrid 탐지 | Student-정규화 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for noise, systems in noise_rollup.items():
        lines.append(
            f"| {noise} | {systems['raw_rule']['pairs']:,} | "
            f"{systems['raw_rule']['noisy_target_detection']:.1%} | "
            f"{systems['normalized_rule']['noisy_target_detection']:.1%} | "
            f"{systems['student']['noisy_target_detection']:.1%} | "
            f"{systems['hybrid_raw_rule_or_student']['noisy_target_detection']:.1%} | "
            f"{systems['student']['noisy_target_detection'] - systems['normalized_rule']['noisy_target_detection']:+.1%}p |"
        )
    lines += ["", "## 6. 결과를 논문 주장으로 읽는 순서", "",
        "1. **규칙 결함 확인:** 원시 규칙의 clean→noisy 하락과 target 탐지율을 본다.",
        "2. **값싼 대안 확인:** 정규화+규칙이 회복하면 알려진 이음매는 모델 없이 고칠 수 있다.",
        "3. **Student의 추가 가치 확인:** 같은 정답·pair에서 정규화 규칙보다 noisy F2와 target 탐지가 높아야 단독 대체 근거가 된다.",
        "4. **Hybrid의 비용 확인:** target 탐지가 늘어도 mask rate와 overmask가 크게 늘면 보완책이지 대체책은 아니다.",
        "5. **주장 범위 제한:** 이 실험은 clean 최신 규칙 모방과 표면 교란 강건성을 측정한다. 실제 민감정보 정답성은 human-gold로 별도 검증한다.",
        "", "## 해석 원칙", "",
        "- 정규화+규칙이 Student보다 좋으면 해당 결함은 모델 없이 전처리로 해결하는 편이 낫다.",
        "- Student 단독 대체는 clean fidelity, noisy F2, target 탐지, 마스킹 예산을 모두 규칙과 비교해야 한다.",
        "- Hybrid는 누락 감소가 목적이다. 탐지율 상승과 함께 과다 마스킹 증가를 반드시 보고한다.",
        "- 정답은 최신 clean v1.4 규칙의 투영 라벨이므로 실제 PII 타당성 결론에는 human-gold가 별도로 필요하다.", "",
    ]
    markdown = ROOT / "DISJOINT_FOUR_SYSTEM_RESULTS.md"
    markdown.write_text("\n".join(lines), encoding="utf-8")
    print(output)
    print(markdown)


if __name__ == "__main__":
    main()
