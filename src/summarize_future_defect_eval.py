"""Summarize the fixed-protocol future-defect time-axis evaluation."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
DATASETS = {
    "drug": ("Drug Reviews", "의료: 약물·용량·증상", "medterm5 v1.4"),
    "symptom2dx": ("Symptom2Dx", "의료: 증상·용량", "medterm5 v1.4"),
    "mednli": ("MedNLI", "의료: 임상 문장쌍", "medterm5 v1.4"),
    "redditmh": ("RedditMH", "비정형: 정신건강 서술", "medterm5 v1.4"),
    "adr": ("ADR", "의료: 약물 부작용", "medterm5 v1.4"),
    "mentalhealth": ("Mental Health", "비정형: 정신건강 서술", "medterm5 v1.4"),
    "qnli": ("QNLI", "비개인 엔티티: 질문·문장", "piiclean2 v1.4"),
    "finphrasebank": ("FinPhraseBank", "비개인 엔티티: 금융", "piiclean2 v1.4"),
    "bios": ("BIOS", "실제 PII: 인명", "piiclean2 v1.4"),
    "mrpc": ("MRPC", "실제 PII: 날짜·연락처·URL", "piiclean2 strict v1.4"),
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def build() -> dict:
    result = {"protocol": {}, "datasets": {}, "pooled_by_noise": {}}
    result["protocol"] = {
        "student": "ELECTRA-small + hidden-128 MLP; strict seen-5 checkpoints",
        "seeds": list(SEEDS),
        "future_noise_training_or_threshold_use": False,
        "gold": "latest clean v1.4 rule spans projected through a single deterministic future edit",
        "criterion": (
            "all 3 seeds: clean gate passes; future absolute target-detection delta "
            "and smaller-drop delta each have source-cluster 95% CI entirely above 0"
        ),
    }

    noise_runs: dict[str, list[dict]] = {}
    for key, (name, domain, policy) in DATASETS.items():
        pairs = load(
            ROOT / "data" / "robustness" / "v14_future" / key / "future_pairs.summary.json"
        )
        runs = []
        for seed in SEEDS:
            payload = load(ROOT / "reports" / f"future_v14_{key}_seed{seed}.json")
            absolute = payload["absolute_target_robustness"]
            clean = payload["student"]["clean"]
            rule_noisy = payload["rule_v1_4"]["noisy"]
            student_noisy = payload["student"]["noisy"]
            quality = payload["acceptance"]["final_student_quality_gate"]["pass"]
            absolute_gate = payload["acceptance"]["absolute_target_robustness_gate"]["pass"]
            run = {
                "seed": seed,
                "clean_f1": clean["f1"],
                "clean_f2": clean["f2"],
                "clean_recall": clean["recall"],
                "quality_gate": quality,
                "absolute_gate": absolute_gate,
                "rule_noisy_mask": rule_noisy["predicted_mask_rate"],
                "student_noisy_mask": student_noisy["predicted_mask_rate"],
                "rule_noisy_precision": rule_noisy["precision"],
                "student_noisy_precision": student_noisy["precision"],
                "rule_noisy_recall": rule_noisy["recall"],
                "student_noisy_recall": student_noisy["recall"],
                "rule_noisy_f2": rule_noisy["f2"],
                "student_noisy_f2": student_noisy["f2"],
                **absolute,
            }
            runs.append(run)
            for noise, row in payload["shared_clean_target_by_noise"].items():
                noise_runs.setdefault(noise, []).append(
                    {
                        "dataset": key,
                        "seed": seed,
                        "eligible": row["eligible_shared_clean_targets"],
                        "rule": row["rule_span_survival_rate"],
                        "student": row["student_span_survival_rate"],
                        "delta": row["student_minus_rule"],
                    }
                )
        result["datasets"][key] = {
            "name": name,
            "domain": domain,
            "policy": policy,
            "pairs": pairs["pairs"],
            "counts": pairs["counts"],
            "runs": runs,
            "summary": {
                field: mean([run[field] for run in runs])
                for field in (
                    "clean_f1",
                    "clean_f2",
                    "clean_recall",
                    "rule_noisy_target_detection",
                    "student_noisy_target_detection",
                    "student_minus_rule_noisy",
                    "rule_detection_drop",
                    "student_detection_drop",
                    "student_drop_advantage",
                    "rule_noisy_mask",
                    "student_noisy_mask",
                    "rule_noisy_precision",
                    "student_noisy_precision",
                    "rule_noisy_recall",
                    "student_noisy_recall",
                    "rule_noisy_f2",
                    "student_noisy_f2",
                )
            },
            "all_seeds_quality_gate": all(run["quality_gate"] for run in runs),
            "all_seeds_absolute_gate": all(run["absolute_gate"] for run in runs),
        }

    for noise, rows in sorted(noise_runs.items()):
        eligible = sum(row["eligible"] for row in rows)
        if not eligible:
            continue
        result["pooled_by_noise"][noise] = {
            "eligible_shared_clean_targets": eligible,
            "rule_survival": sum(row["eligible"] * row["rule"] for row in rows) / eligible,
            "student_survival": sum(row["eligible"] * row["student"] for row in rows) / eligible,
            "student_minus_rule": sum(row["eligible"] * row["delta"] for row in rows) / eligible,
        }
    return result


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def markdown(result: dict) -> str:
    items = result["datasets"]
    total_pairs = sum(item["pairs"] for item in items.values())
    lines = [
        "# 미래 결함 시간축 평가 결과",
        "",
        "학습이 끝난 뒤 새로 발견됐다고 가정한 7개 표면 교란을 test 전용으로 두었다. clean 최신 v1.4 규칙 span을 고정 정답으로 이동했으며, Student는 기존 strict seen-5 checkpoint를 그대로 사용했다. 즉 미래 교란은 학습·validation·threshold 선택에 들어가지 않았다.",
        "",
        f"- 평가: {len(items)}개 데이터셋, {total_pairs:,} future target-pair, ELECTRA-small 3 seed(42/43/44).",
        "- 비교 대상: 현재 RedactFormer raw 규칙 v1.4. 새 normalizer나 사후 규칙 패치는 이 비교에 넣지 않았다.",
        "- 판정: clean quality gate와 미래 절대 탐지 우세·하락폭 우세의 source-cluster 95% CI를 세 seed 모두 통과해야 한다.",
        "",
        "## 데이터셋별 결과",
        "",
        "| 데이터셋 | 도메인 | Pair | Clean F2 | 규칙 미래 탐지 | Student 미래 탐지 | 차이 | 규칙 하락 | Student 하락 | 하락폭 이점 | 3-seed 판정 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in items.values():
        s = item["summary"]
        verdict = "우세" if item["all_seeds_absolute_gate"] else "미달"
        lines.append(
            f"| {item['name']} | {item['domain']} | {item['pairs']:,} | {s['clean_f2']:.3f} | "
            f"{pct(s['rule_noisy_target_detection'])} | {pct(s['student_noisy_target_detection'])} | "
            f"{s['student_minus_rule_noisy'] * 100:+.1f}%p | {pct(s['rule_detection_drop'])} | "
            f"{pct(s['student_detection_drop'])} | {s['student_drop_advantage'] * 100:+.1f}%p | {verdict} |"
        )
    lines.extend([
        "",
        "## 해석",
        "",
        "데이터셋별 3-seed 판정을 우선 본다. 우세 데이터셋에서는 **현재 raw 규칙 v1.4가 아직 알지 못한 표면 결함**에 대해, 과거 결함으로 증강한 Student가 규칙보다 더 많은 고정 민감 target을 보존했고 clean→future 하락도 더 작았다.",
        "",
        "하지만 전체 token F2까지 규칙을 이겼는지는 별도다. 특히 Drug Reviews와 BIOS에서 Student는 주입 target은 더 잘 보존했지만 token precision이 낮아 전체 noisy F2는 규칙보다 낮다. 즉 이 결과는 ‘완전 대체’가 아니라 **규칙 업데이트 전 새 표면 결함을 덜 놓치는 local fallback/병렬 보완기**의 근거다.",
        "",
        "## 전체 token 품질과 마스킹 예산",
        "",
        "마스킹 비율도 함께 봐야 한다. Student의 target 우세가 단순히 훨씬 더 많이 가려서 생긴 것은 아니다. 다만 규칙의 매우 높은 precision을 완전히 재현하지는 못했으므로, 배포는 `규칙 OR Student`의 즉시 교체가 아니라 Student 양성 후보를 재검사하거나, 미래 결함 계열에 한정한 fallback으로 시작하는 편이 타당하다.",
        "",
        "| 데이터셋 | 규칙 mask | Student mask | 규칙 P / R / F2 | Student P / R / F2 |",
        "|---|---:|---:|---|---|",
    ])
    for item in items.values():
        s = item["summary"]
        lines.append(
            f"| {item['name']} | {pct(s['rule_noisy_mask'])} | {pct(s['student_noisy_mask'])} | "
            f"{s['rule_noisy_precision']:.3f} / {s['rule_noisy_recall']:.3f} / {s['rule_noisy_f2']:.3f} | "
            f"{s['student_noisy_precision']:.3f} / {s['student_noisy_recall']:.3f} / {s['student_noisy_f2']:.3f} |"
        )
    lines.extend([
        "",
        "",
        "## 교란 종류별 공통 span 생존율",
        "",
        "공통 clean-correct span만 분모로 하므로, 이 표는 clean에서 이미 한 쪽이 놓친 span 때문에 생기는 착시를 제거한 보조 분석이다. 너무 적은 적용 사례(예: 특정 데이터셋의 용량/아포스트로피)는 전체 결론보다 사례 분석으로만 해석한다.",
        "",
        f"| 미래 교란 | 공통 span({len(items)}데이터셋×3 seed) | 규칙 생존 | Student 생존 | 차이 |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, item in result["pooled_by_noise"].items():
        lines.append(
            f"| {name} | {item['eligible_shared_clean_targets']:,} | {pct(item['rule_survival'])} | "
            f"{pct(item['student_survival'])} | {item['student_minus_rule'] * 100:+.1f}%p |"
        )
    lines.extend([
        "",
        "## 재현 파일",
        "",
        "- 설계와 사전 판정: `FUTURE_DEFECT_TIME_AXIS.md`",
        "- 실행기: `src/run_future_defect_eval.py`",
        "- raw 결과: `reports/future_v14_<dataset>_seed<seed>.json`",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    result = build()
    (ROOT / "reports" / "future_defect_time_axis_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (ROOT / "FUTURE_DEFECT_TIME_AXIS_RESULTS.md").write_text(
        markdown(result), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
