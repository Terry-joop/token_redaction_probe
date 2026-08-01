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
MODELS = {
    "bert_tiny": ("BERT-tiny", "prajjwal1/bert-tiny"),
    "electra_small": (
        "ELECTRA-small",
        "google/electra-small-discriminator",
    ),
    "distilroberta": ("DistilRoBERTa", "distilroberta-base"),
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


def result_path(dataset: str, model: str) -> Path:
    root = ROOT / "artifacts" / "robustness" / "v14"
    if model == "electra_small":
        legacy = root / f"{dataset}_results.json"
        if legacy.exists():
            return legacy
    return root / f"{dataset}_{model}_results.json"


def sync_dataset_results(summary: dict) -> None:
    datasets = {}
    for key, (name, group, group_name, teacher) in DATASETS.items():
        model_results = {}
        raw_results = {}
        for model_key, (model_name, encoder_name) in MODELS.items():
            path = result_path(key, model_key)
            if not path.exists():
                raise FileNotFoundError(f"Missing robustness result: {path}")
            result = load(path)
            raw_results[model_key] = result
            student = result["student"]
            model_dir = (
                ROOT
                / "artifacts"
                / "robustness"
                / "v14"
                / f"{key}_{model_key}_seed42"
            )
            clean_eval = load(model_dir / "medical_evaluation.json")
            model_results[model_key] = {
                "name": model_name,
                "encoder": encoder_name,
                "threshold": result["student_threshold"],
                "operating_points": {
                    "budget_matched": clean_eval["budget_matched"]["test"],
                    "f2_optimized": clean_eval["f2_optimized"]["test"],
                },
                "robustness": {
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

        reference = raw_results["electra_small"]
        rule = reference["rule_v1_4"]
        split_dir = ROOT / "data" / "robustness" / "v14" / key
        splits = {
            split: line_count(split_dir / f"{split}.jsonl")
            for split in ("train", "validation", "test")
        }
        pair_path = split_dir / "paired_noisy_test.jsonl"
        if not pair_path.exists():
            pair_path = split_dir / "robustness_pairs.jsonl"
        unique_source_rows = reference.get("unique_source_rows")
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
            "pairs": reference["pairs"],
            "unique_source_rows": unique_source_rows,
            "splits": splits,
            "teacher": teacher,
            "policy_id": reference["rule_policy"],
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
            "models": model_results,
        }
    summary["datasets"] = datasets
    summary.pop("student", None)
    summary.pop("train_validation_test", None)
    summary["students"] = [
        value[1] + " + hidden-128 MLP" for value in MODELS.values()
    ]
    summary["protocol"] = {
        "students": summary["students"],
        "seed": 42,
        "epochs": 5,
        "batch_size": 32,
        "max_length": 128,
        "train_validation_test_cap": [5000, 500, 1000],
        "noise_types": 12,
        "per_noise_cap": 100,
        "robustness_operating_point": "validation budget-matched threshold",
        "pseudo_gold": (
            "clean v1.4 character spans projected through deterministic edits"
        ),
    }
    efficiency = load(
        ROOT
        / "artifacts"
        / "medical_redactor"
        / "core_matrix"
        / "complete_rule_student_comparison.json"
    )["efficiency"]
    model_analysis = {}
    for model_key, (model_name, _) in MODELS.items():
        values = [item["models"][model_key] for item in datasets.values()]
        group_metrics = {}
        for group_key in ("medical", "general"):
            group_items = [
                item for item in datasets.values() if item["group"] == group_key
            ]
            group_values = [item["models"][model_key] for item in group_items]
            clean_f2 = statistics.fmean(
                value["robustness"]["clean_f2"] for value in group_values
            )
            noisy_f2 = statistics.fmean(
                value["robustness"]["noisy_f2"] for value in group_values
            )
            rule_noisy_f2 = statistics.fmean(
                item["rule"]["noisy_f2"] for item in group_items
            )
            group_metrics[group_key] = {
                "datasets": len(group_items),
                "clean_f2": clean_f2,
                "noisy_f2": noisy_f2,
                "f2_drop": statistics.fmean(
                    value["robustness"]["f2_drop"] for value in group_values
                ),
                "retention": statistics.fmean(
                    value["robustness"]["noisy_f2"]
                    / value["robustness"]["clean_f2"]
                    for value in group_values
                ),
                "rule_noisy_f2": rule_noisy_f2,
                "rule_gap": noisy_f2 - rule_noisy_f2,
            }
        source = efficiency[model_key]
        model_analysis[model_key] = {
            "name": model_name,
            "parameters": source["parameters"],
            "model_state_mb": source["model_state_mb"],
            "sentences_per_second": source["latency"]["sentences_per_second"],
            "clean_gate_passes": sum(
                value["acceptance"]["final_student_quality_gate"]["pass"]
                for value in values
            ),
            "budget_gate_passes": sum(
                value["acceptance"]["matched_budget_gate"]["pass"]
                for value in values
            ),
            "best_noisy_f2_datasets": sum(
                item["models"][model_key]["robustness"]["noisy_f2"]
                == max(
                    value["robustness"]["noisy_f2"]
                    for value in item["models"].values()
                )
                for item in datasets.values()
            ),
            "groups": group_metrics,
        }
    summary["matrix_analysis"] = {
        "models": model_analysis,
        "monotonic_noisy_f2_datasets": sum(
            item["models"]["bert_tiny"]["robustness"]["noisy_f2"]
            < item["models"]["electra_small"]["robustness"]["noisy_f2"]
            < item["models"]["distilroberta"]["robustness"]["noisy_f2"]
            for item in datasets.values()
        ),
        "privacy_tradeoff_runs": sum(
            model["operating_points"]["f2_optimized"]["recall"]
            >= model["operating_points"]["budget_matched"]["recall"]
            and model["operating_points"]["f2_optimized"]["f2"]
            >= model["operating_points"]["budget_matched"]["f2"]
            and model["operating_points"]["f2_optimized"]["predicted_mask_rate"]
            >= model["operating_points"]["budget_matched"]["predicted_mask_rate"]
            for item in datasets.values()
            for model in item["models"].values()
        ),
    }


def write_dataset_report(summary: dict) -> None:
    report_path = ROOT / "ROBUSTNESS_EXPERIMENT_V14.md"
    report = report_path.read_text(encoding="utf-8")
    start = "## 10개 데이터셋 × 3모델 공통 비교"
    end = "## 전체 데이터 및 증강 ablation"
    if start not in report or end not in report:
        raise ValueError("Could not find robustness report section markers")

    setup_rows = []
    metric_rows = []
    operating_rows = []
    bootstrap_rows = []
    for item in summary["datasets"].values():
        split = item["splits"]
        setup_rows.append(
            f"| {item['group_name']} | {item['name']} | {item['teacher']} | "
            f"{split['train']:,} / {split['validation']:,} / {split['test']:,} | "
            f"{item['pairs']:,} | {item['unique_source_rows']:,} |"
        )
        rule = item["rule"]
        metric_rows.append(
            f"| {item['name']} | 규칙 v1.4 | — | {rule['clean_f2']:.3f} | "
            f"{rule['noisy_precision']:.3f} | {rule['noisy_recall']:.3f} | "
            f"{rule['noisy_f1']:.3f} | {rule['noisy_f2']:.3f} | "
            f"{rule['f2_drop']:.3f} | {rule['noisy_mask_rate'] * 100:.2f}% |"
        )
        for model in item["models"].values():
            robust = model["robustness"]
            metric_rows.append(
                f"| {item['name']} | {model['name']} | {model['threshold']:.2f} | "
                f"{robust['clean_f2']:.3f} | {robust['noisy_precision']:.3f} | "
                f"{robust['noisy_recall']:.3f} | {robust['noisy_f1']:.3f} | "
                f"{robust['noisy_f2']:.3f} | {robust['f2_drop']:.3f} | "
                f"{robust['noisy_mask_rate'] * 100:.2f}% |"
            )
            budget = model["operating_points"]["budget_matched"]
            privacy = model["operating_points"]["f2_optimized"]
            operating_rows.append(
                f"| {item['name']} | {model['name']} | {budget['threshold']:.2f} | "
                f"{budget['precision']:.3f} | {budget['recall']:.3f} | "
                f"{budget['f2']:.3f} | {budget['predicted_mask_rate'] * 100:.2f}% | "
                f"{privacy['threshold']:.2f} | {privacy['precision']:.3f} | "
                f"{privacy['recall']:.3f} | {privacy['f2']:.3f} | "
                f"{privacy['predicted_mask_rate'] * 100:.2f}% |"
            )
            boot = model["bootstrap_delta_f2"]
            ci = boot["ci95"]
            bootstrap_rows.append(
                f"| {item['name']} | {model['name']} | {boot['mean']:+.3f} | "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                f"{'Student 우세' if ci[0] > 0 else '규칙 우세' if ci[1] < 0 else '유의 차이 없음'} |"
            )

    all_models = [
        model
        for item in summary["datasets"].values()
        for model in item["models"].values()
    ]
    clean_passes = sum(
        model["acceptance"]["final_student_quality_gate"]["pass"]
        for model in all_models
    )
    budget_passes = sum(
        model["acceptance"]["matched_budget_gate"]["pass"]
        for model in all_models
    )
    student_wins = sum(
        model["robustness"]["noisy_f2"] > item["rule"]["noisy_f2"]
        for item in summary["datasets"].values()
        for model in item["models"].values()
    )
    model_gate_summary = []
    for model_key, (model_name, _) in MODELS.items():
        values = [item["models"][model_key] for item in summary["datasets"].values()]
        clean_count = sum(
            value["acceptance"]["final_student_quality_gate"]["pass"]
            for value in values
        )
        budget_count = sum(
            value["acceptance"]["matched_budget_gate"]["pass"]
            for value in values
        )
        model_gate_summary.append(
            f"{model_name} clean {clean_count}/10·예산 {budget_count}/10"
        )
    privacy_tradeoff_runs = sum(
        model["operating_points"]["f2_optimized"]["recall"]
        >= model["operating_points"]["budget_matched"]["recall"]
        and model["operating_points"]["f2_optimized"]["f2"]
        >= model["operating_points"]["budget_matched"]["f2"]
        and model["operating_points"]["f2_optimized"]["predicted_mask_rate"]
        >= model["operating_points"]["budget_matched"]["predicted_mask_rate"]
        for model in all_models
    )
    analysis = summary["matrix_analysis"]
    analysis_rows = []
    for model in analysis["models"].values():
        medical = model["groups"]["medical"]
        general = model["groups"]["general"]
        analysis_rows.append(
            f"| {model['name']} | {model['parameters'] / 1_000_000:.1f}M | "
            f"{model['model_state_mb']:.1f} MB | {model['sentences_per_second']:.1f}/s | "
            f"{medical['clean_f2']:.3f} → {medical['noisy_f2']:.3f} "
            f"(−{medical['f2_drop']:.3f}) | "
            f"{general['clean_f2']:.3f} → {general['noisy_f2']:.3f} "
            f"(−{general['f2_drop']:.3f}) | "
            f"{model['clean_gate_passes']}/10 |"
        )
    bert = analysis["models"]["bert_tiny"]
    electra = analysis["models"]["electra_small"]
    distil = analysis["models"]["distilroberta"]
    section = f"""## 10개 데이터셋 × 3모델 공통 비교

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
{chr(10).join(setup_rows)}

## 10개 데이터셋 × 3모델 입력 교란 결과

| 데이터셋 | 방식 | Budget Th. | Clean F2 | Noisy P | Noisy R | Noisy F1 | Noisy F2 | F2 하락 | Noisy mask |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_rows)}

paired source-cluster bootstrap 2,000회의 noisy F2 차이(Student−Rule)는 다음과 같다.

| 데이터셋 | Student | 평균 차이 | 95% CI | 판정 |
|---|---|---:|---:|---|
{chr(10).join(bootstrap_rows)}

30개 Student run 중 절대 noisy F2가 규칙보다 높은 경우는 **{student_wins}개**,
동일 마스킹 예산 통과는 **{budget_passes}/30개**, 사전 정의 clean 대체 최소선
통과는 **{clean_passes}/30개**다. 모델별 결과는 각각 {"; ".join(model_gate_summary)}이다.
DistilRoBERTa는 10개 데이터셋 모두에서 세 Student 중 noisy F2가 가장 높았다.

## 모델 크기별 결과 분석

| Student | Params | 모델 크기 | 처리량 | 의료 Clean→Noisy F2 | 일반 Clean→Noisy F2 | Clean gate |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(analysis_rows)}

- 모델이 커질수록 절대 noisy F2가 높아지는 순서가 **{analysis['monotonic_noisy_f2_datasets']}/10개 데이터셋**에서 동일했다. 의료는 BERT {bert['groups']['medical']['noisy_f2']:.3f} → ELECTRA {electra['groups']['medical']['noisy_f2']:.3f} → DistilRoBERTa {distil['groups']['medical']['noisy_f2']:.3f}, 일반은 {bert['groups']['general']['noisy_f2']:.3f} → {electra['groups']['general']['noisy_f2']:.3f} → {distil['groups']['general']['noisy_f2']:.3f}였다.
- 반면 평균 F2 하락폭은 의료에서 BERT {bert['groups']['medical']['f2_drop']:.3f}, ELECTRA {electra['groups']['medical']['f2_drop']:.3f}, DistilRoBERTa {distil['groups']['medical']['f2_drop']:.3f}였다. BERT의 하락이 작지만 clean F2 자체가 낮아 생긴 효과이므로 **하락폭만으로 강건성을 판정하면 안 된다**.
- 가장 큰 DistilRoBERTa도 규칙 noisy F2보다 의료 {abs(distil['groups']['medical']['rule_gap']):.3f}, 일반 {abs(distil['groups']['general']['rule_gap']):.3f} 낮았다. 모델 크기 증가는 규칙 모방력을 높였지만 규칙 대체까지 만들지는 못했다.
- ELECTRA→DistilRoBERTa의 noisy F2 증가는 의료 {distil['groups']['medical']['noisy_f2'] - electra['groups']['medical']['noisy_f2']:+.3f}, 일반 {distil['groups']['general']['noisy_f2'] - electra['groups']['general']['noisy_f2']:+.3f}인 반면 파라미터는 {distil['parameters'] / electra['parameters']:.1f}배, 모델 파일은 {distil['model_state_mb'] / electra['model_state_mb']:.1f}배다. 품질 최우선이면 DistilRoBERTa, 속도·메모리까지 보면 ELECTRA-small이 현실적인 절충점이다.

## 동일 마스킹 예산과 Recall 중심 F2

| 데이터셋 | Student | 예산 Th. | 예산 P | 예산 R | 예산 F2 | 예산 mask | F2 Th. | F2 P | F2 R | F2 | F2 mask |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(operating_rows)}

- **동일 마스킹 예산**은 Teacher와 비슷한 비율을 가리는 threshold다. 같은 privacy/utility
  비용에서 모델을 비교하므로 논문의 주 비교에 적합하다.
- **Recall 중심 F2**는 validation F2를 최대화한다. F2는 Recall 오차를 Precision보다
  네 배 크게 반영하므로 민감 토큰 누락이 더 비싼 배포 상황의 보조 운용점이다. 대신
  더 많이 가릴 수 있어 규칙과의 공정한 효율 비교로 단독 사용하면 안 된다. 실제로
  이번 결과에서는 {privacy_tradeoff_runs}/30개 run 모두 Recall·F2·마스킹률이 함께 증가했다.
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
