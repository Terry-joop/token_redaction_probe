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
    "bios": ("BIOS", "general", "일반 PII/엔티티", "piiclean2 v1.4"),
    "mrpc": ("MRPC", "general", "일반 PII/엔티티", "piiclean2 v1.4"),
    "qnli": ("QNLI", "general", "일반 PII/엔티티", "piiclean2 v1.4"),
    "finphrasebank": (
        "FinPhraseBank",
        "general",
        "일반 PII/엔티티",
        "piiclean2 v1.4",
    ),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate(runs: list[dict], fields: list[str]) -> dict:
    return {
        field: {
            "mean": statistics.fmean(run[field] for run in runs),
            "sample_std": statistics.stdev(run[field] for run in runs),
        }
        for field in fields
    }


def line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def sync_dataset_results(summary: dict) -> None:
    datasets = {}
    for key, (name, group, group_name, teacher) in DATASETS.items():
        result_path = (
            ROOT / "artifacts" / "robustness" / "v14" / f"{key}_results.json"
        )
        if not result_path.exists():
            raise FileNotFoundError(f"Missing robustness result: {result_path}")
        result = load(result_path)
        rule = result["rule_v1_4"]
        student = result["student"]
        split_dir = ROOT / "data" / "robustness" / "v14" / key
        splits = {
            split: line_count(split_dir / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        }
        pair_path = split_dir / "paired_noisy_test.jsonl"
        if not pair_path.exists():
            pair_path = split_dir / "robustness_pairs.jsonl"
        unique_source_rows = result.get("unique_source_rows")
        if unique_source_rows is None:
            source_ids = {
                json.loads(line)["source_id"]
                for line in pair_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            unique_source_rows = len(source_ids)
        datasets[key] = {
            "name": name,
            "group": group,
            "group_name": group_name,
            "pairs": result["pairs"],
            "unique_source_rows": unique_source_rows,
            "splits": splits,
            "teacher": teacher,
            "policy_id": result["rule_policy"],
            "threshold": result["student_threshold"],
            "rule": {
                "clean_f2": rule["clean"]["f2"],
                "noisy_precision": rule["noisy"]["precision"],
                "noisy_recall": rule["noisy"]["recall"],
                "noisy_f1": rule["noisy"]["f1"],
                "noisy_f2": rule["noisy"]["f2"],
                "noisy_mask_rate": rule["noisy"]["predicted_mask_rate"],
                "f2_drop": rule["clean"]["f2"] - rule["noisy"]["f2"],
                "newly_leaked_span_rate": rule["robustness"][
                    "newly_leaked_span_rate"
                ],
            },
            "student_result": {
                "clean_precision": student["clean"]["precision"],
                "clean_recall": student["clean"]["recall"],
                "clean_f1": student["clean"]["f1"],
                "clean_f2": student["clean"]["f2"],
                "noisy_precision": student["noisy"]["precision"],
                "noisy_recall": student["noisy"]["recall"],
                "noisy_f1": student["noisy"]["f1"],
                "noisy_f2": student["noisy"]["f2"],
                "noisy_mask_rate": student["noisy"]["predicted_mask_rate"],
                "f2_drop": student["clean"]["f2"]
                - student["noisy"]["f2"],
                "newly_leaked_span_rate": student["robustness"][
                    "newly_leaked_span_rate"
                ],
            },
            "bootstrap_delta_f2": result["paired_bootstrap"],
            "acceptance": result["acceptance"],
        }
    summary["datasets"] = datasets
    summary["protocol"] = {
        "student": "google/electra-small-discriminator + hidden-128 MLP",
        "seed": 42,
        "epochs": 5,
        "max_length": 128,
        "train_validation_test_cap": [5000, 500, 1000],
        "noise_types": 12,
        "per_noise_cap": 100,
        "pseudo_gold": (
            "clean v1.4 character spans projected through deterministic edits"
        ),
    }


def write_dataset_report(summary: dict) -> None:
    report_path = ROOT / "ROBUSTNESS_EXPERIMENT_V14.md"
    report = report_path.read_text(encoding="utf-8")
    start = "## 10개 데이터셋 공통 비교"
    end = "## 전체 데이터 및 증강 ablation"
    if start not in report or end not in report:
        raise ValueError("Could not find robustness report section markers")

    setup_rows = []
    metric_rows = []
    bootstrap_rows = []
    for item in summary["datasets"].values():
        split = item["splits"]
        setup_rows.append(
            f"| {item['group_name']} | {item['name']} | {item['teacher']} | "
            f"{split['train']:,} / {split['validation']:,} / {split['test']:,} | "
            f"{item['pairs']:,} | {item['unique_source_rows']:,} |"
        )
        rule = item["rule"]
        student = item["student_result"]
        metric_rows.extend(
            [
                f"| {item['name']} | 규칙 v1.4 | {rule['clean_f2']:.3f} | "
                f"{rule['noisy_precision']:.3f} | {rule['noisy_recall']:.3f} | "
                f"{rule['noisy_f1']:.3f} | {rule['noisy_f2']:.3f} | "
                f"{rule['f2_drop']:.3f} | {rule['noisy_mask_rate'] * 100:.2f}% |",
                f"| {item['name']} | ELECTRA-small | {student['clean_f2']:.3f} | "
                f"{student['noisy_precision']:.3f} | {student['noisy_recall']:.3f} | "
                f"{student['noisy_f1']:.3f} | {student['noisy_f2']:.3f} | "
                f"{student['f2_drop']:.3f} | {student['noisy_mask_rate'] * 100:.2f}% |",
            ]
        )
        boot = item["bootstrap_delta_f2"]
        ci = boot["ci95"]
        bootstrap_rows.append(
            f"| {item['name']} | {boot['mean']:+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | "
            f"{'Student 우세' if ci[0] > 0 else '규칙 우세' if ci[1] < 0 else '유의 차이 없음'} |"
        )

    clean_passes = sum(
        item["acceptance"]["final_student_quality_gate"]["pass"]
        for item in summary["datasets"].values()
    )
    budget_passes = sum(
        item["acceptance"]["matched_budget_gate"]["pass"]
        for item in summary["datasets"].values()
    )
    student_wins = sum(
        item["student_result"]["noisy_f2"] > item["rule"]["noisy_f2"]
        for item in summary["datasets"].values()
    )
    section = f"""## 10개 데이터셋 공통 비교

기존 4-1의 Drug Reviews·BIOS 실험을 기존 메인 표의 나머지 8개 데이터셋까지
같은 프로토콜로 확장했다. 의료 6개는 `medterm5 v1.4`, 일반 4개는
`piiclean2 v1.4`로 clean pseudo-label을 만들었다. Student는 모든 데이터셋에서
`google/electra-small-discriminator + hidden-128 MLP`를 encoder까지 fine-tuning했다.

- 데이터셋별 최대 train / validation / test: 5,000 / 500 / 1,000
- 작은 split은 사용 가능한 행 전부 사용
- 5 epochs, batch 32, max length 128, seed 42
- validation에서 동일 마스킹 예산 threshold 선택 후 test에 고정
- 12종 교란별 최대 100개 paired test 생성
- Noisy P/R/F1/F2 정답: clean v1.4 문자 span을 결정적 편집을 따라 이동한 token pseudo-gold
- RedactFormer 기준 커밋: `39b56279c6c58fdc6732df8d5ee98868e323d344`
- 문서화된 교란: 이중 공백, 곱슬/C1 아포스트로피, `25 mg→25mg`, 숫자 뒤 쉼표
- 미관측 교란: 삼중 공백, NBSP, modifier apostrophe, `25-mg`, thin space, 세미콜론, 단어 내부 zero-width

대용량 `mapped_dataset_n5_medterm5/piiclean2` 산출물은 이 저장소에 없으므로,
아래 결과는 로컬 clean 원문에서 현재 v1.4 코드로 다시 라벨링한 데이터셋별 제한본이다.
즉 10개 **종류 전체**를 비교했지만 각 대규모 데이터셋의 **모든 행**을 학습한 실험은
아니다. 작은 split은 사용 가능한 행 전부를 사용했다.

| 그룹 | 데이터셋 | Teacher | Train / Val / Test | 교란 pair | 고유 원문 |
|---|---|---|---:|---:|---:|
{chr(10).join(setup_rows)}

## 10개 데이터셋 입력 교란 결과

메인 비교는 동일 마스킹 예산 threshold다. Clean F2와 Noisy 지표는 동일한
projected pseudo-gold 기준이며, human-gold 개인정보 정확도가 아니다.

| 데이터셋 | 방식 | Clean F2 | Noisy P | Noisy R | Noisy F1 | Noisy F2 | F2 하락 | Noisy mask |
|---|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

paired source-cluster bootstrap 2,000회의 noisy F2 차이(Student−Rule)는 다음과 같다.

| 데이터셋 | 평균 차이 | 95% CI | 판정 |
|---|---:|---:|---|
{chr(10).join(bootstrap_rows)}

10개 중 Student의 절대 noisy F2가 규칙보다 높은 데이터셋은 **{student_wins}개**다.
동일 마스킹 예산은 **{budget_passes}/10개**, 사전 정의 clean 대체 최소선
(F1/F2/Recall ≥ 0.85/0.90/0.90)은 **{clean_passes}/10개**가 통과했다.
따라서 이 제한 실험만으로 규칙 대체 성공을 주장할 수 없고, 데이터셋별 취약성과
표면 결함 보완 가능성을 확인하는 비교 결과로 해석한다. 특히 FinPhraseBank는 noisy
마스킹률 차이가 1%p를 넘어 동일 예산 직접 비교에 주의한다.

"""
    prefix, suffix = report.split(start, 1)
    _, suffix = suffix.split(end, 1)
    report_path.write_text(prefix + section + end + suffix, encoding="utf-8")


def main() -> None:
    summary_path = ROOT / "reports" / "robustness_v14_results.json"
    summary = load(summary_path)
    sync_dataset_results(summary)
    robust_runs = []
    repeated_runs = []
    for seed in SEEDS:
        robustness = load(
            ROOT / "reports" / f"robustness_v14_full_eval_seed{seed}.json"
        )
        absolute = robustness["absolute_target_robustness"]
        shared = robustness["shared_clean_target_robustness"]
        evaluation = load(
            ROOT
            / "artifacts"
            / "robustness"
            / "v14_augmented"
            / f"drug_electra_small_full_aug_seed{seed}"
            / "medical_evaluation.json"
        )
        clean = evaluation["budget_matched"]["test"]
        robust_runs.append(
            {
                "seed": seed,
                "rule_clean_target_recall": absolute[
                    "rule_clean_target_detection"
                ],
                "rule_noisy_target_recall": absolute[
                    "rule_noisy_target_detection"
                ],
                "rule_drop": absolute["rule_detection_drop"],
                "student_clean_target_recall": absolute[
                    "student_clean_target_detection"
                ],
                "student_noisy_target_recall": absolute[
                    "student_noisy_target_detection"
                ],
                "student_drop": absolute["student_detection_drop"],
                "student_noisy_advantage": absolute[
                    "student_minus_rule_noisy"
                ],
                "student_noisy_advantage_ci95": absolute[
                    "student_minus_rule_noisy_ci95"
                ],
                "student_drop_advantage": absolute[
                    "student_drop_advantage"
                ],
                "student_drop_advantage_ci95": absolute[
                    "student_drop_advantage_ci95"
                ],
            }
        )
        repeated_runs.append(
            {
                "seed": seed,
                "threshold": clean["threshold"],
                "clean_precision": clean["precision"],
                "clean_recall": clean["recall"],
                "clean_f1": clean["f1"],
                "clean_f2": clean["f2"],
                "student_mask_rate": clean["predicted_mask_rate"],
                "unseen_noisy_f2": robustness["student"]["noisy"]["f2"],
                "rule_unseen_noisy_f2": robustness["rule_v1_4"]["noisy"]["f2"],
                "shared_targets": shared["eligible_shared_clean_targets"],
                "rule_span_survival": shared["rule_span_survival_rate"],
                "student_span_survival": shared["student_span_survival_rate"],
                "survival_delta": shared["student_minus_rule"],
                "survival_ci95": shared["ci95"],
            }
        )

    ablation = summary["augmentation_ablation"]
    ablation["full_eval"] = {
        "validation_examples": 4997,
        "test_examples": 4997,
        "unseen_target_pairs": robust_runs[0]
        and load(ROOT / "reports" / "robustness_v14_full_eval_seed42.json")[
            "pairs"
        ],
        "unique_source_examples": load(
            ROOT / "reports" / "robustness_v14_full_eval_seed42.json"
        )["unique_source_rows"],
        "threshold_selection": "full validation split",
        "bootstrap_unit": (
            "source-cluster bootstrap; all perturbations from one source "
            "sentence stay together"
        ),
    }
    ablation["full_aug_seed_repeats"] = {
        "seeds": SEEDS,
        "evaluation_scope": "full validation/test and all eligible unseen pairs",
        "runs": repeated_runs,
        "summary": aggregate(
            repeated_runs,
            [
                "clean_precision",
                "clean_recall",
                "clean_f1",
                "clean_f2",
                "student_mask_rate",
                "unseen_noisy_f2",
                "rule_unseen_noisy_f2",
                "rule_span_survival",
                "student_span_survival",
                "survival_delta",
            ],
        ),
    }
    fields = [
        "rule_clean_target_recall",
        "rule_noisy_target_recall",
        "rule_drop",
        "student_clean_target_recall",
        "student_noisy_target_recall",
        "student_drop",
        "student_noisy_advantage",
        "student_drop_advantage",
    ]
    ablation["absolute_target_evaluation"] = {
        "dataset_count": 1,
        "dataset": "Drug Reviews",
        "source_examples": 49974,
        "clean_train_examples": 39980,
        "validation_examples": 4997,
        "test_examples": 4997,
        "augmented_train_examples": 30591,
        "total_augmented_train_rows": 70571,
        "unseen_target_pairs": ablation["full_eval"]["unseen_target_pairs"],
        "unique_source_examples": ablation["full_eval"][
            "unique_source_examples"
        ],
        "reference": (
            "clean MASKING_FRAMEWORK v1.4 medterm5 target spans projected "
            "through deterministic edits"
        ),
        "bootstrap_unit": ablation["full_eval"]["bootstrap_unit"],
        "runs": robust_runs,
        "summary": aggregate(robust_runs, fields),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_dataset_report(summary)
    print(json.dumps(ablation["absolute_target_evaluation"], indent=2))


if __name__ == "__main__":
    main()
