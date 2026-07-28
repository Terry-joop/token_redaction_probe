import argparse

from common import render_redaction
from medical_common import read_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Render word-level medical annotations")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    inputs = {row["id"]: row for row in read_records(args.inputs)}
    annotations = read_records(args.annotations)
    selected = total = 0
    for annotation in annotations[:args.limit]:
        source = inputs[annotation["id"]]
        labels = annotation["labels"]
        chosen = [word for word, label in zip(source["words"], labels) if label]
        selected += sum(labels)
        total += len(labels)
        print(f"\n[{source['id']}] task_label={source.get('task_label')}")
        print("selected:", chosen)
        print("redacted:", render_redaction(source["words"], labels))
    print(f"\nshown={min(args.limit, len(annotations))} token_mask_rate={selected / max(total, 1):.2%}")


if __name__ == "__main__":
    main()
