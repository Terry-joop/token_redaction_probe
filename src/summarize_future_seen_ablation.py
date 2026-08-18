"""Summarize fixed Future-7 results: clean-only vs Seen-5 augmentation."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from summarize_future_defect_eval import DATASETS, SEEDS

ROOT = Path(__file__).resolve().parents[1]


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(rows: list[dict], field: str) -> float:
    return statistics.fmean(row[field] for row in rows)


def main() -> None:
    seen = load(ROOT / "reports/future_defect_time_axis_summary.json")
    output = {
        "protocol": {
            "comparison": "Same clean split, ELECTRA-small + hidden-128 MLP, 5 epochs, seed 42/43/44, and fixed Future-7 pairs/rule cache.",
            "only_difference": "clean-only trains on clean v1.4 teacher labels; seen-5 additionally trains on one Seen perturbation per eligible clean train row.",
            "future_noise_use": "Future-7 never enters either training, validation, or threshold selection.",
        },
        "datasets": {},
    }
    for key, (name, domain, policy) in DATASETS.items():
        clean_runs = []
        for seed_id in SEEDS:
            payload = load(ROOT / f"reports/future_v14_cleanonly_{key}_seed{seed_id}.json")
            absolute = payload["absolute_target_robustness"]
            absolute_or = payload["absolute_target_rule_or_robustness"]
            student = payload["student"]
            rule_or = payload["rule_or_student"]
            rule_and = payload["rule_and_student"]
            clean_runs.append({
                "seed": seed_id,
                "threshold": payload["student_threshold"],
                "clean_f2": student["clean"]["f2"],
                "clean_recall": student["clean"]["recall"],
                "clean_target": absolute["student_clean_target_detection"],
                "future_target": absolute["student_noisy_target_detection"],
                "drop": absolute["student_detection_drop"],
                "future_precision": student["noisy"]["precision"],
                "future_recall": student["noisy"]["recall"],
                "future_f2": student["noisy"]["f2"],
                "future_mask": student["noisy"]["predicted_mask_rate"],
                "future_overmask": student["noisy"]["overmask_rate"],
                "rule_or_clean_target": absolute_or["student_clean_target_detection"],
                "rule_or_future_target": absolute_or["student_noisy_target_detection"],
                "rule_or_drop": absolute_or["student_detection_drop"],
                "rule_or_future_mask": rule_or["noisy"]["predicted_mask_rate"],
                "rule_or_future_overmask": rule_or["noisy"]["overmask_rate"],
                "rule_and_future_target": rule_and["robustness"]["noisy_target_detection"],
                "quality_gate": payload["acceptance"]["final_student_quality_gate"]["pass"],
                "absolute_gate": payload["acceptance"]["absolute_target_robustness_gate"]["pass"],
                "rule_or_absolute_gate": (
                    absolute_or["student_minus_rule_noisy_ci95"][0] > 0
                    and absolute_or["student_drop_advantage_ci95"][0] > 0
                ),
            })
        seen_runs = seen["datasets"][key]["runs"]
        seen_values = [{
            "seed": row["seed"], "clean_f2": row["clean_f2"],
            "clean_recall": row["clean_recall"],
            "clean_target": row["student_clean_target_detection"],
            "future_target": row["student_noisy_target_detection"],
            "drop": row["student_detection_drop"],
            "future_precision": row["student_noisy_precision"],
            "future_recall": row["student_noisy_recall"],
            "future_f2": row["student_noisy_f2"],
            "future_mask": row["student_noisy_mask"],
        } for row in seen_runs]
        shared_fields = ["clean_f2", "clean_recall", "clean_target", "future_target", "drop", "future_precision", "future_recall", "future_f2", "future_mask"]
        clean_summary = {field: mean(clean_runs, field) for field in shared_fields}
        clean_summary.update({
            field: mean(clean_runs, field)
            for field in ("future_overmask", "rule_or_clean_target", "rule_or_future_target", "rule_or_drop", "rule_or_future_mask", "rule_or_future_overmask", "rule_and_future_target")
        })
        seen_summary = {field: mean(seen_values, field) for field in shared_fields}
        rule = seen["datasets"][key]["summary"]
        output["datasets"][key] = {
            "name": name, "domain": domain, "policy": policy,
            "pairs": seen["datasets"][key]["pairs"],
            "rule": {
                "clean_target": 1.0,
                "future_target": rule["rule_noisy_target_detection"],
                "drop": rule["rule_detection_drop"],
            },
            "clean_only_runs": clean_runs,
            "seen5_runs": seen_values,
            "clean_only": clean_summary,
            "seen5": seen_summary,
            "seen5_minus_clean_only": {
                field: seen_summary[field] - clean_summary[field] for field in shared_fields
            },
            "clean_only_all_seeds_quality_gate": all(row["quality_gate"] for row in clean_runs),
            "clean_only_all_seeds_absolute_gate": all(row["absolute_gate"] for row in clean_runs),
            "clean_only_all_seeds_rule_or_absolute_gate": all(row["rule_or_absolute_gate"] for row in clean_runs),
        }
    (ROOT / "reports/future_seen_ablation_summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
