import argparse

from sklearn.metrics import precision_recall_fscore_support

from medical_common import read_records, validate_labels


def named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("annotation must be NAME=PATH")
    return tuple(value.split("=", 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two pseudo-label sources without calling either gold")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--annotation", action="append", type=named_path, required=True)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()
    if len(args.annotation) != 2:
        raise ValueError("provide exactly two --annotation NAME=PATH arguments")

    inputs = {row["id"]: row for row in read_records(args.inputs)}
    (name_a, path_a), (name_b, path_b) = args.annotation
    rows_a = {row["id"]: row for row in read_records(path_a)}
    rows_b = {row["id"]: row for row in read_records(path_b)}
    shared = [example_id for example_id in inputs if example_id in rows_a and example_id in rows_b]
    labels_a, labels_b = [], []
    for example_id in shared:
        words = inputs[example_id]["words"]
        labels_a.extend(validate_labels(example_id, words, rows_a[example_id]["labels"]))
        labels_b.extend(validate_labels(example_id, words, rows_b[example_id]["labels"]))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_a, labels_b, average="binary", zero_division=0
    )
    print(f"shared_examples={len(shared)} tokens={len(labels_a)}")
    print(f"{name_a}_mask_rate={sum(labels_a) / max(len(labels_a), 1):.2%}")
    print(f"{name_b}_mask_rate={sum(labels_b) / max(len(labels_b), 1):.2%}")
    print(
        f"agreement_if_{name_a}_is_reference: precision={precision:.4f} "
        f"recall={recall:.4f} f1={f1:.4f} (not gold performance)"
    )
    for example_id in shared[:args.show]:
        source = inputs[example_id]
        a = rows_a[example_id]["labels"]
        b = rows_b[example_id]["labels"]
        selected_a = [word for word, label in zip(source["words"], a) if label]
        selected_b = [word for word, label in zip(source["words"], b) if label]
        print(f"\n[{example_id}] {source['text']}")
        print(f"{name_a}: {selected_a}")
        print(f"{name_b}: {selected_b}")


if __name__ == "__main__":
    main()
