from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import fbeta_score, precision_recall_fscore_support
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REDACTFORMER_ROOT = ROOT.parents[1] / "Redactformer"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from medical_common import word_offsets  # noqa: E402
from train import RedactionModel  # noqa: E402
from v14_rule_adapter import V14RuleAdapter, read_jsonl, write_jsonl  # noqa: E402


def metric(gold: list[int], predicted: list[int]) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, predicted, average="binary", zero_division=0
    )
    gold_array = np.asarray(gold)
    predicted_array = np.asarray(predicted)
    negative = max(int(np.sum(gold_array == 0)), 1)
    false_positive = int(
        np.sum((gold_array == 0) & (predicted_array == 1))
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(
            fbeta_score(
                gold,
                predicted,
                beta=2,
                average="binary",
                zero_division=0,
            )
        ),
        "gold_mask_rate": float(np.mean(gold_array)) if len(gold) else 0.0,
        "predicted_mask_rate": (
            float(np.mean(predicted_array)) if len(predicted) else 0.0
        ),
        "residual_sensitive_rate": float(1 - recall),
        "overmask_rate": false_positive / negative,
        "tokens": len(gold),
    }


def target_detected(
    text: str,
    words: list[str],
    gold: list[int],
    predicted: list[int],
    target: list[int],
) -> bool:
    stable_words, offsets = word_offsets(text)
    if stable_words != words:
        raise ValueError("target detection tokenization mismatch")
    start, end = target
    target_indices = [
        index
        for index, ((word_start, word_end), label) in enumerate(
            zip(offsets, gold)
        )
        if label and word_start < end and start < word_end
    ]
    return bool(target_indices) and all(
        predicted[index] for index in target_indices
    )





def detection_flags(
    pairs: list[dict], predictions: list[list[int]], clean: bool
) -> np.ndarray:
    return np.asarray(
        [
            target_detected(
                row["clean_text"] if clean else row["text"],
                row["clean_words"] if clean else row["words"],
                row["clean_labels"] if clean else row["labels"],
                predicted,
                row["clean_target"] if clean else row["noisy_target"],
            )
            for row, predicted in zip(pairs, predictions)
        ],
        dtype=np.float64,
    )


def source_cluster_totals(
    pairs: list[dict], values: np.ndarray
) -> np.ndarray:
    """Aggregate pair-level values once per source sentence."""
    matrix = np.asarray(values)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if len(matrix) != len(pairs):
        raise ValueError("pair/value length mismatch")
    source_to_cluster: dict[str, int] = {}
    cluster_ids = np.empty(len(pairs), dtype=np.int64)
    for index, row in enumerate(pairs):
        source_id = row["source_id"]
        cluster_ids[index] = source_to_cluster.setdefault(
            source_id, len(source_to_cluster)
        )
    totals = np.zeros((len(source_to_cluster), matrix.shape[1]), dtype=np.float64)
    np.add.at(totals, cluster_ids, matrix)
    return totals


def bootstrap_cluster_sums(
    cluster_totals: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
    batch_size: int = 32,
) -> np.ndarray:
    """Resample source clusters without rebuilding/flattening them per repeat."""
    cluster_count = len(cluster_totals)
    output = np.empty((repeats, cluster_totals.shape[1]), dtype=np.float64)
    for start in range(0, repeats, batch_size):
        stop = min(start + batch_size, repeats)
        chosen = rng.integers(
            0, cluster_count, size=(stop - start, cluster_count)
        )
        output[start:stop] = cluster_totals[chosen].sum(axis=1)
    return output


def absolute_target_robustness(
    pairs: list[dict],
    rule_clean: list[list[int]],
    rule_noisy: list[list[int]],
    student_clean: list[list[int]],
    student_noisy: list[list[int]],
    repeats: int,
    seed: int,
) -> dict:
    """Compare both systems on every fixed clean-rule target."""
    rule_clean_values = detection_flags(pairs, rule_clean, True)
    rule_noisy_values = detection_flags(pairs, rule_noisy, False)
    student_clean_values = detection_flags(pairs, student_clean, True)
    student_noisy_values = detection_flags(pairs, student_noisy, False)
    noisy_delta = student_noisy_values - rule_noisy_values
    drop_advantage = (
        rule_clean_values
        - rule_noisy_values
        - student_clean_values
        + student_noisy_values
    )
    rng = np.random.default_rng(seed)
    cluster_values = source_cluster_totals(
        pairs,
        np.column_stack(
            [noisy_delta, drop_advantage, np.ones(len(pairs), dtype=np.float64)]
        ),
    )
    sampled = bootstrap_cluster_sums(cluster_values, repeats, rng)
    noisy_bootstrap = sampled[:, 0] / sampled[:, 2]
    drop_bootstrap = sampled[:, 1] / sampled[:, 2]
    noisy_ci = np.percentile(noisy_bootstrap, [2.5, 97.5])
    drop_ci = np.percentile(drop_bootstrap, [2.5, 97.5])
    return {
        "target_pairs": len(pairs),
        "unique_source_rows": len({row["source_id"] for row in pairs}),
        "rule_clean_target_detection": float(np.mean(rule_clean_values)),
        "rule_noisy_target_detection": float(np.mean(rule_noisy_values)),
        "rule_detection_drop": float(
            np.mean(rule_clean_values) - np.mean(rule_noisy_values)
        ),
        "student_clean_target_detection": float(np.mean(student_clean_values)),
        "student_noisy_target_detection": float(np.mean(student_noisy_values)),
        "student_detection_drop": float(
            np.mean(student_clean_values) - np.mean(student_noisy_values)
        ),
        "student_minus_rule_noisy": float(np.mean(noisy_delta)),
        "student_minus_rule_noisy_ci95": [
            float(noisy_ci[0]),
            float(noisy_ci[1]),
        ],
        "student_drop_advantage": float(np.mean(drop_advantage)),
        "student_drop_advantage_ci95": [
            float(drop_ci[0]),
            float(drop_ci[1]),
        ],
        "repeats": repeats,
        "unit": "source-cluster bootstrap over fixed target pairs",
    }


class StudentPredictor:
    def __init__(
        self,
        model_dir: str | Path,
        device: str,
        threshold: float | None,
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        with (self.model_dir / "experiment.json").open(
            encoding="utf-8"
        ) as handle:
            self.config = json.load(handle)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = RedactionModel(
            self.config["model_name"],
            self.config["hidden_size"],
            self.config["freeze_encoder"],
        )
        self.model.load_state_dict(
            torch.load(
                self.model_dir / "model.pt",
                map_location="cpu",
                weights_only=True,
            )
        )
        self.model.to(self.device)
        self.model.eval()
        if threshold is None:
            evaluation_path = self.model_dir / "medical_evaluation.json"
            if not evaluation_path.exists():
                raise FileNotFoundError(
                    "threshold not supplied and medical_evaluation.json is absent"
                )
            evaluation = json.loads(
                evaluation_path.read_text(encoding="utf-8")
            )
            threshold = evaluation["budget_matched"]["test"]["threshold"]
        self.threshold = float(threshold)

    def predict_many(
        self, rows: list[list[str]], batch_size: int = 128
    ) -> list[list[int]]:
        predictions = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=self.config["max_length"],
                return_tensors="pt",
            )
            word_ids = [
                encoded.word_ids(batch_index=index)
                for index in range(len(batch))
            ]
            with torch.no_grad():
                logits = self.model(
                    **{
                        key: value.to(self.device)
                        for key, value in encoded.items()
                    }
                )
            scores = logits.softmax(-1)[:, :, 1].cpu().numpy()
            for batch_index, words in enumerate(batch):
                output = [0] * len(words)
                seen = set()
                for token_index, word_id in enumerate(word_ids[batch_index]):
                    if word_id is None or word_id in seen:
                        continue
                    seen.add(word_id)
                    if word_id < len(output):
                        output[word_id] = int(
                            scores[batch_index, token_index] >= self.threshold
                        )
                predictions.append(output)
        return predictions

    def predict(self, words: list[str]) -> list[int]:
        return self.predict_many([words], batch_size=1)[0]


def evaluate_system(
    pairs: list[dict],
    clean_predictions: list[list[int]],
    noisy_predictions: list[list[int]],
):
    clean_gold = []
    clean_pred = []
    noisy_gold = []
    noisy_pred = []
    clean_targets = []
    noisy_targets = []
    by_noise = defaultdict(lambda: {"gold": [], "pred": []})
    for row, clean, noisy in zip(
        pairs, clean_predictions, noisy_predictions
    ):
        clean_gold.extend(row["clean_labels"])
        clean_pred.extend(clean)
        noisy_gold.extend(row["labels"])
        noisy_pred.extend(noisy)
        by_noise[row["noise_type"]]["gold"].extend(row["labels"])
        by_noise[row["noise_type"]]["pred"].extend(noisy)
        clean_targets.append(
            target_detected(
                row["clean_text"],
                row["clean_words"],
                row["clean_labels"],
                clean,
                row["clean_target"],
            )
        )
        noisy_targets.append(
            target_detected(
                row["text"],
                row["words"],
                row["labels"],
                noisy,
                row["noisy_target"],
            )
        )
    eligible = [
        index for index, value in enumerate(clean_targets) if value
    ]
    survived = sum(noisy_targets[index] for index in eligible)
    return {
        "clean": metric(clean_gold, clean_pred),
        "noisy": metric(noisy_gold, noisy_pred),
        "robustness": {
            "clean_target_detection": float(np.mean(clean_targets)),
            "noisy_target_detection": float(np.mean(noisy_targets)),
            "eligible_clean_targets": len(eligible),
            "span_survival_rate": (
                survived / len(eligible) if eligible else 0.0
            ),
            "newly_leaked_span_rate": (
                1 - survived / len(eligible) if eligible else 1.0
            ),
        },
        "by_noise": {
            name: metric(values["gold"], values["pred"])
            for name, values in sorted(by_noise.items())
        },
    }


def combine_predictions(
    left: list[list[int]], right: list[list[int]], mode: str
) -> list[list[int]]:
    """Combine aligned token masks with logical OR or AND."""
    if mode not in {"or", "and"}:
        raise ValueError(f"unsupported combination mode: {mode}")
    output = []
    for left_row, right_row in zip(left, right):
        if len(left_row) != len(right_row):
            raise ValueError("cannot combine masks with different lengths")
        if mode == "or":
            output.append([int(a or b) for a, b in zip(left_row, right_row)])
        else:
            output.append([int(a and b) for a, b in zip(left_row, right_row)])
    return output


def shared_target_robustness(
    pairs: list[dict],
    rule_clean: list[list[int]],
    rule_noisy: list[list[int]],
    student_clean: list[list[int]],
    student_noisy: list[list[int]],
    repeats: int,
    seed: int,
) -> dict:
    rule_clean_flags = detection_flags(pairs, rule_clean, True)
    student_clean_flags = detection_flags(pairs, student_clean, True)
    rule_noisy_flags = detection_flags(pairs, rule_noisy, False)
    student_noisy_flags = detection_flags(pairs, student_noisy, False)
    eligible = [
        index
        for index, (rule_ok, student_ok) in enumerate(
            zip(rule_clean_flags, student_clean_flags)
        )
        if rule_ok and student_ok
    ]
    if not eligible:
        return {
            "eligible_shared_clean_targets": 0,
            "rule_span_survival_rate": 0.0,
            "student_span_survival_rate": 0.0,
            "student_minus_rule": 0.0,
            "ci95": [0.0, 0.0],
            "repeats": repeats,
        }

    rule_values = np.asarray(
        [rule_noisy_flags[index] for index in eligible],
        dtype=np.float64,
    )
    student_values = np.asarray(
        [student_noisy_flags[index] for index in eligible],
        dtype=np.float64,
    )
    differences = student_values - rule_values
    rng = np.random.default_rng(seed)
    eligible_pairs = [pairs[index] for index in eligible]
    cluster_values = source_cluster_totals(
        eligible_pairs,
        np.column_stack(
            [differences, np.ones(len(eligible_pairs), dtype=np.float64)]
        ),
    )
    sampled = bootstrap_cluster_sums(cluster_values, repeats, rng)
    bootstrap = sampled[:, 0] / sampled[:, 1]
    low, high = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "eligible_shared_clean_targets": len(eligible),
        "rule_span_survival_rate": float(np.mean(rule_values)),
        "student_span_survival_rate": float(np.mean(student_values)),
        "student_minus_rule": float(np.mean(differences)),
        "ci95": [float(low), float(high)],
        "repeats": repeats,
        "unit": "source-cluster bootstrap over shared clean-correct targets",
    }
def shared_target_by_noise(
    pairs: list[dict],
    rule_clean: list[list[int]],
    rule_noisy: list[list[int]],
    student_clean: list[list[int]],
    student_noisy: list[list[int]],
    repeats: int,
    seed: int,
) -> dict:
    output = {}
    names = sorted({row["noise_type"] for row in pairs})
    for offset, name in enumerate(names, start=1):
        indices = [
            index
            for index, row in enumerate(pairs)
            if row["noise_type"] == name
        ]
        output[name] = shared_target_robustness(
            [pairs[index] for index in indices],
            [rule_clean[index] for index in indices],
            [rule_noisy[index] for index in indices],
            [student_clean[index] for index in indices],
            [student_noisy[index] for index in indices],
            repeats,
            seed + offset,
        )
    return output


def bootstrap_delta(
    pairs: list[dict],
    rule_predictions: list[list[int]],
    student_predictions: list[list[int]],
    repeats: int,
    seed: int,
) -> dict:
    def pair_counts(predictions: list[list[int]]) -> np.ndarray:
        counts = []
        for row, predicted in zip(pairs, predictions):
            gold = np.asarray(row["labels"], dtype=np.int8)
            pred = np.asarray(predicted, dtype=np.int8)
            counts.append(
                (
                    int(np.sum((gold == 1) & (pred == 1))),
                    int(np.sum((gold == 0) & (pred == 1))),
                    int(np.sum((gold == 1) & (pred == 0))),
                )
            )
        return np.asarray(counts, dtype=np.int64)

    rng = np.random.default_rng(seed)
    rule_counts = pair_counts(rule_predictions)
    student_counts = pair_counts(student_predictions)
    cluster_values = source_cluster_totals(
        pairs, np.column_stack([rule_counts, student_counts])
    )
    sampled = bootstrap_cluster_sums(cluster_values, repeats, rng)

    def f2_from_rows(counts: np.ndarray) -> np.ndarray:
        true_positive = counts[:, 0]
        false_positive = counts[:, 1]
        false_negative = counts[:, 2]
        denominator = 5 * true_positive + false_positive + 4 * false_negative
        return np.divide(
            5 * true_positive,
            denominator,
            out=np.zeros_like(denominator, dtype=np.float64),
            where=denominator != 0,
        )

    deltas = f2_from_rows(sampled[:, 3:]) - f2_from_rows(sampled[:, :3])
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "metric": "noisy_f2_student_minus_rule",
        "mean": float(np.mean(deltas)),
        "ci95": [float(low), float(high)],
        "repeats": repeats,
        "unit": "source-clustered paired sentence",
    }


def acceptance(result: dict) -> dict:
    student = result["student"]
    rule = result["rule_v1_4"]
    clean = student["clean"]
    noisy = student["noisy"]
    mask_gap = abs(
        noisy["predicted_mask_rate"]
        - rule["noisy"]["predicted_mask_rate"]
    )
    final_quality = (
        clean["f1"] >= 0.85
        and clean["f2"] >= 0.90
        and clean["recall"] >= 0.90
    )
    robustness_win = (
        noisy["f2"] > rule["noisy"]["f2"]
        and student["robustness"]["newly_leaked_span_rate"]
        < rule["robustness"]["newly_leaked_span_rate"]
        and result["paired_bootstrap"]["ci95"][0] > 0
    )
    shared = result["shared_clean_target_robustness"]
    conditional_win = (
        shared["ci95"][0] > 0 and shared["student_minus_rule"] >= 0.05
    )
    absolute = result["absolute_target_robustness"]
    absolute_win = (
        absolute["student_minus_rule_noisy_ci95"][0] > 0
        and absolute["student_drop_advantage_ci95"][0] > 0
    )
    return {
        "final_student_quality_gate": {
            "pass": final_quality,
            "criteria": "clean F1>=0.85, F2>=0.90, Recall>=0.90",
        },
        "matched_budget_gate": {
            "pass": mask_gap <= 0.01,
            "mask_rate_gap": mask_gap,
            "criteria": "absolute noisy mask-rate gap <= 0.01",
        },
        "robustness_superiority_gate": {
            "pass": robustness_win,
            "criteria": (
                "noisy F2 higher, newly leaked span rate lower, and paired "
                "bootstrap 95% CI for delta F2 entirely above zero"
            ),
        },
        "absolute_target_robustness_gate": {
            "pass": absolute_win,
            "student_minus_rule_noisy": absolute[
                "student_minus_rule_noisy"
            ],
            "student_minus_rule_noisy_ci95": absolute[
                "student_minus_rule_noisy_ci95"
            ],
            "student_drop_advantage": absolute[
                "student_drop_advantage"
            ],
            "student_drop_advantage_ci95": absolute[
                "student_drop_advantage_ci95"
            ],
            "criteria": (
                "on all fixed clean-rule targets, both noisy detection "
                "advantage and smaller-drop advantage have source-cluster "
                "95% CIs entirely above zero"
            ),
        },
        "conditional_surface_robustness_gate": {
            "pass": conditional_win,
            "student_minus_rule": shared["student_minus_rule"],
            "ci95": shared["ci95"],
            "criteria": (
                "on shared clean-correct spans, Student survival >= rule "
                "+5 percentage points and paired 95% CI entirely above zero"
            ),
        },
        "pilot_quality_reference": {
            "pass": (
                clean["f2"] >= 0.80 and clean["recall"] >= 0.85
            ),
            "criteria": (
                "exploratory pilot only: clean F2>=0.80 and Recall>=0.85"
            ),
        },
    }


def render_markdown(result: dict) -> str:
    lines = [
        "# 입력 교란 강건성 결과",
        "",
        f"- Teacher: {result['rule_policy']}",
        f"- Student threshold: {result['student_threshold']:.4f}",
        f"- paired cases: {result['pairs']}",
        "",
        "## 전체 결과",
        "",
        (
            "| 방식 | Clean P | Clean R | Clean F1 | Clean F2 | "
            "Noisy P | Noisy R | Noisy F1 | Noisy F2 | 신규 누출 |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    systems = [
        ("rule_v1_4", "규칙 v1.4"),
        ("student", "ELECTRA-small"),
        ("rule_or_student", "규칙 OR Student"),
        ("rule_and_student", "규칙 AND Student"),
    ]
    for key, label in systems:
        system = result[key]
        values = [
            system["clean"]["precision"],
            system["clean"]["recall"],
            system["clean"]["f1"],
            system["clean"]["f2"],
            system["noisy"]["precision"],
            system["noisy"]["recall"],
            system["noisy"]["f1"],
            system["noisy"]["f2"],
            system["robustness"]["newly_leaked_span_rate"],
        ]
        lines.append(
            "| "
            + label
            + " | "
            + " | ".join(f"{value:.3f}" for value in values)
            + " |"
        )
    absolute = result["absolute_target_robustness"]
    lines.extend(
        [
            "",
            "## 전체 고정 target 비교",
            "",
            "| 분모 | 규칙 clean | 규칙 noisy | Student clean | Student noisy | Noisy 차이 | 하락폭 이점 |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {absolute['target_pairs']} pair / "
                f"{absolute['unique_source_rows']} 원문 | "
                f"{absolute['rule_clean_target_detection']:.3f} | "
                f"{absolute['rule_noisy_target_detection']:.3f} | "
                f"{absolute['student_clean_target_detection']:.3f} | "
                f"{absolute['student_noisy_target_detection']:.3f} | "
                f"{absolute['student_minus_rule_noisy']:+.3f} | "
                f"{absolute['student_drop_advantage']:+.3f} |"
            ),
            "",
            (
                "Noisy 차이 95% CI: "
                f"[{absolute['student_minus_rule_noisy_ci95'][0]:+.3f}, "
                f"{absolute['student_minus_rule_noisy_ci95'][1]:+.3f}]. "
                "하락폭 이점 95% CI: "
                f"[{absolute['student_drop_advantage_ci95'][0]:+.3f}, "
                f"{absolute['student_drop_advantage_ci95'][1]:+.3f}]."
            ),
        ]
    )
    shared = result["shared_clean_target_robustness"]
    lines.extend(
        [
            "",
            "## 공통 clean-correct span 생존율",
            "",
            "| 대상 | 규칙 | Student | 차이 | 95% CI |",
            "|---|---:|---:|---:|---:|",
            (
                f"| 전체 {shared['eligible_shared_clean_targets']}개 | "
                f"{shared['rule_span_survival_rate']:.3f} | "
                f"{shared['student_span_survival_rate']:.3f} | "
                f"{shared['student_minus_rule']:+.3f} | "
                f"[{shared['ci95'][0]:+.3f}, {shared['ci95'][1]:+.3f}] |"
            ),
        ]
    )
    for name, values in result["shared_clean_target_by_noise"].items():
        lines.append(
            f"| {name} ({values['eligible_shared_clean_targets']}) | "
            f"{values['rule_span_survival_rate']:.3f} | "
            f"{values['student_span_survival_rate']:.3f} | "
            f"{values['student_minus_rule']:+.3f} | "
            f"[{values['ci95'][0]:+.3f}, {values['ci95'][1]:+.3f}] |"
        )
    lines.extend(["", "## 합격 판정", ""])
    for name, gate in result["acceptance"].items():
        verdict = "PASS" if gate["pass"] else "FAIL"
        lines.append(f"- {verdict} {name}: {gate['criteria']}")
    lines.extend(
        [
            "",
            (
                "이 평가는 깨끗한 v1.4 라벨의 의미적 정당성이 아니라, "
                "같은 라벨을 표면 교란 뒤에도 유지하는 강건성을 측정한다."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare v1.4 rules and a student on paired perturbations"
        )
    )
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--masker", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--redactformer-root", default=DEFAULT_REDACTFORMER_ROOT)
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cuda"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--noise-group",
        default="all",
        help="Optional group label to retain (for example: unseen or future).",
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--student-batch-size", type=int, default=128)
    parser.add_argument(
        "--rule-cache",
        help="Optional JSONL cache for rule predictions keyed by pair_id",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs = read_jsonl(args.pairs)
    if args.noise_group != "all":
        pairs = [
            row for row in pairs
            if row.get("noise_group") == args.noise_group
        ]
    if not pairs:
        raise ValueError("no robustness pairs")
    student = StudentPredictor(
        args.model_dir, args.device, args.threshold
    )
    cached_rule = {}
    if args.rule_cache and Path(args.rule_cache).exists():
        cached_rule = {
            row["pair_id"]: row["labels"]
            for row in read_jsonl(args.rule_cache)
        }
        print(
            f"loaded {len(cached_rule)} cached rule predictions",
            flush=True,
        )
    missing_rule = [
        row for row in pairs if row["pair_id"] not in cached_rule
    ]
    rule = None
    if missing_rule:
        rule = V14RuleAdapter(
            args.masker,
            args.task,
            args.redactformer_root,
        )

    clean_by_source = {}
    for row in pairs:
        clean_by_source.setdefault(row["source_id"], row["clean_words"])
    source_ids = list(clean_by_source)
    clean_values = student.predict_many(
        [clean_by_source[source_id] for source_id in source_ids],
        batch_size=args.student_batch_size,
    )
    clean_cache = dict(zip(source_ids, clean_values))
    student_clean = [clean_cache[row["source_id"]] for row in pairs]
    student_noisy = student.predict_many(
        [row["words"] for row in pairs],
        batch_size=args.student_batch_size,
    )
    print(
        f"student predicted {len(source_ids)} clean sources and "
        f"{len(pairs)} noisy pairs",
        flush=True,
    )

    rule_clean = [row["clean_labels"] for row in pairs]
    rule_noisy = []
    for index, row in enumerate(pairs, start=1):
        if row["pair_id"] in cached_rule:
            labels = cached_rule[row["pair_id"]]
        else:
            labels = rule.predict(row["text"]).labels
            cached_rule[row["pair_id"]] = labels
        rule_noisy.append(labels)
        if index % 1000 == 0 or index == len(pairs):
            print(f"[{index}/{len(pairs)}] rule predictions ready", flush=True)

    if args.rule_cache:
        write_jsonl(
            args.rule_cache,
            [
                {"pair_id": row["pair_id"], "labels": cached_rule[row["pair_id"]]}
                for row in pairs
            ],
        )
    rule_policy = (
        rule.policy_id
        if rule is not None
        else pairs[0].get("teacher_policy", "cached-rule")
    )
    rule_or_clean = combine_predictions(rule_clean, student_clean, "or")
    rule_or_noisy = combine_predictions(rule_noisy, student_noisy, "or")
    rule_and_clean = combine_predictions(rule_clean, student_clean, "and")
    rule_and_noisy = combine_predictions(
        rule_noisy, student_noisy, "and"
    )

    result = {
        "pairs": len(pairs),
        "unique_source_rows": len({row["source_id"] for row in pairs}),
        "noise_group": args.noise_group,
        "rule_policy": rule_policy,
        "student_model_dir": str(Path(args.model_dir)),
        "student_threshold": student.threshold,
        "rule_v1_4": evaluate_system(
            pairs, rule_clean, rule_noisy
        ),
        "student": evaluate_system(
            pairs, student_clean, student_noisy
        ),
        "rule_or_student": evaluate_system(
            pairs, rule_or_clean, rule_or_noisy
        ),
        "rule_and_student": evaluate_system(
            pairs, rule_and_clean, rule_and_noisy
        ),
        "absolute_target_robustness": absolute_target_robustness(
            pairs,
            rule_clean,
            rule_noisy,
            student_clean,
            student_noisy,
            args.bootstrap_repeats,
            args.seed,
        ),
        "paired_bootstrap": bootstrap_delta(
            pairs,
            rule_noisy,
            student_noisy,
            args.bootstrap_repeats,
            args.seed,
        ),
        "shared_clean_target_robustness": shared_target_robustness(
            pairs,
            rule_clean,
            rule_noisy,
            student_clean,
            student_noisy,
            args.bootstrap_repeats,
            args.seed,
        ),
        "shared_clean_target_by_noise": shared_target_by_noise(
            pairs,
            rule_clean,
            rule_noisy,
            student_clean,
            student_noisy,
            args.bootstrap_repeats,
            args.seed,
        ),
    }
    result["acceptance"] = acceptance(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output.with_suffix(".md").write_text(
        render_markdown(result),
        encoding="utf-8",
    )
    print(json.dumps(result["acceptance"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
