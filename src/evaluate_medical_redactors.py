import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import fbeta_score, precision_recall_fscore_support

from medical_common import labels_to_spans, read_records, validate_labels


def score_labels(gold_by_id: dict[str, list[int]], pred_by_id: dict[str, list[int]]) -> dict:
    gold, pred = [], []
    gold_spans, pred_spans = set(), set()
    per_example_mask_rates = []
    for example_id, gold_labels in gold_by_id.items():
        pred_labels = pred_by_id[example_id]
        gold.extend(gold_labels)
        pred.extend(pred_labels)
        gold_spans.update((example_id, *span) for span in labels_to_spans(gold_labels))
        pred_spans.update((example_id, *span) for span in labels_to_spans(pred_labels))
        per_example_mask_rates.append(sum(pred_labels) / max(len(pred_labels), 1))
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, average="binary", zero_division=0
    )
    f2 = fbeta_score(gold, pred, beta=2, average="binary", zero_division=0)
    span_tp = len(gold_spans & pred_spans)
    span_precision = span_tp / max(len(pred_spans), 1)
    span_recall = span_tp / max(len(gold_spans), 1)
    span_f1 = (
        2 * span_precision * span_recall / (span_precision + span_recall)
        if span_precision + span_recall else 0.0
    )
    return {
        "examples": len(gold_by_id), "tokens": len(gold),
        "accuracy_secondary": float(np.mean(np.equal(gold, pred))),
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "f2": float(f2),
        "residual_sensitive_rate": float(1 - recall),
        "exact_span_precision": span_precision, "exact_span_recall": span_recall,
        "exact_span_f1": span_f1,
        "mean_example_mask_rate": float(np.mean(per_example_mask_rates)),
        "micro_mask_rate": sum(pred) / max(len(pred), 1),
    }


def random_at_same_per_example_budget(gold_by_id, budget_by_id, seeds):
    scores = []
    for seed in range(seeds):
        rng = random.Random(seed)
        predictions = {}
        for example_id, gold in gold_by_id.items():
            k = min(budget_by_id[example_id], len(gold))
            selected = set(rng.sample(range(len(gold)), k))
            predictions[example_id] = [int(index in selected) for index in range(len(gold))]
        scores.append(score_labels(gold_by_id, predictions))
    keys = scores[0]
    return {
        key: {
            "mean": float(np.mean([score[key] for score in scores])),
            "std": float(np.std([score[key] for score in scores])),
        }
        for key in keys if isinstance(scores[0][key], (int, float))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate redactors against human-reviewed gold")
    parser.add_argument("--review", required=True)
    parser.add_argument("--include-unreviewed", action="store_true")
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    rows = [
        row for row in read_records(args.review)
        if args.include_unreviewed or row.get("human_reviewed") is True
    ]
    if not rows:
        raise ValueError("no reviewed rows; set human_reviewed=true after manual review")
    gold = {
        row["id"]: validate_labels(row["id"], row["words"], row["human_labels"])
        for row in rows
    }
    method_names = sorted(set.intersection(*[set(row.get("candidates", {})) for row in rows]))
    if not method_names:
        raise ValueError("no candidate method is present in every reviewed row")

    results = {
        "gold_mask_rate": sum(map(sum, gold.values())) / sum(map(len, gold.values())),
        "methods": {},
    }
    for method in method_names:
        predictions = {
            row["id"]: validate_labels(
                row["id"], row["words"], row["candidates"][method]["labels"]
            ) for row in rows
        }
        result = score_labels(gold, predictions)
        result["rate_matched_random"] = random_at_same_per_example_budget(
            gold, {key: sum(labels) for key, labels in predictions.items()}, args.random_seeds
        )
        results["methods"][method] = result

    print(f"human_gold examples={len(rows)} gold_mask_rate={results['gold_mask_rate']:.2%}")
    print("method\tmask%\tprecision\trecall\tF1\tF2\texact-span-F1\tresidual%")
    for method, result in results["methods"].items():
        print(
            f"{method}\t{100 * result['micro_mask_rate']:.2f}\t{result['precision']:.4f}\t"
            f"{result['recall']:.4f}\t{result['f1']:.4f}\t{result['f2']:.4f}\t"
            f"{result['exact_span_f1']:.4f}\t{100 * result['residual_sensitive_rate']:.2f}"
        )
        random_f2 = result["rate_matched_random"]["f2"]
        print(
            f"  random@same-per-example-budget "
            f"F2={random_f2['mean']:.4f}±{random_f2['std']:.4f}"
        )
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
