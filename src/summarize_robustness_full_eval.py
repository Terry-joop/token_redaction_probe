from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 43, 44]


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


def main() -> None:
    summary_path = ROOT / "reports" / "robustness_v14_results.json"
    summary = load(summary_path)
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
    print(json.dumps(ablation["absolute_target_evaluation"], indent=2))


if __name__ == "__main__":
    main()
