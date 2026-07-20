import argparse

from sklearn.metrics import precision_recall_fscore_support

from common import read_jsonl, render_redaction


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare candidate labels against reference labels")
    parser.add_argument("--reference", required=True, help="Teacher/human JSONL used as gold")
    parser.add_argument("--candidate", required=True, help="Heuristic/student JSONL to evaluate")
    parser.add_argument("--show-errors", type=int, default=5)
    args = parser.parse_args()

    reference = {row["id"]: row for row in read_jsonl(args.reference)}
    candidate = {row["id"]: row for row in read_jsonl(args.candidate)}
    shared = sorted(set(reference) & set(candidate))
    if not shared:
        raise ValueError("no shared ids between files")

    gold, pred, errors = [], [], []
    for example_id in shared:
        ref, cand = reference[example_id], candidate[example_id]
        if ref["words"] != cand["words"]:
            raise ValueError(f"{example_id}: word lists differ")
        gold.extend(ref["labels"])
        pred.extend(cand["labels"])
        if ref["labels"] != cand["labels"]:
            errors.append((ref, cand))

    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, average="binary", zero_division=0
    )
    accuracy = sum(a == b for a, b in zip(gold, pred)) / len(gold)
    print(f"examples={len(shared)} tokens={len(gold)}")
    print(f"accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    print(f"reference_redact_rate={sum(gold) / len(gold):.2%}")
    print(f"candidate_redact_rate={sum(pred) / len(pred):.2%}")
    for ref, cand in errors[: args.show_errors]:
        print(f"\n[{ref['id']}] {ref['text']}")
        print("reference:", render_redaction(ref["words"], ref["labels"]))
        print("candidate:", render_redaction(cand["words"], cand["labels"]))


if __name__ == "__main__":
    main()
