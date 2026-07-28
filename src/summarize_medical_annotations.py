import argparse
from collections import defaultdict

import numpy as np

from medical_common import read_records, validate_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize medical word-label coverage")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--annotations", required=True)
    args = parser.parse_args()

    inputs = {row["id"]: row for row in read_records(args.inputs)}
    annotations = read_records(args.annotations)
    rates, total_tokens, total_selected, zero = [], 0, 0, 0
    by_task = defaultdict(lambda: [0, 0, 0])  # examples, selected, tokens
    for annotation in annotations:
        source = inputs[annotation["id"]]
        labels = validate_labels(annotation["id"], source["words"], annotation["labels"])
        selected = sum(labels)
        count = len(labels)
        total_selected += selected
        total_tokens += count
        zero += selected == 0
        rates.append(selected / max(count, 1))
        task = str(source.get("task_label"))
        by_task[task][0] += 1
        by_task[task][1] += selected
        by_task[task][2] += count

    print(f"examples={len(annotations)} tokens={total_tokens} selected={total_selected}")
    print(f"micro_mask_rate={total_selected / max(total_tokens, 1):.2%}")
    print(f"mean_example_mask_rate={np.mean(rates):.2%} std={np.std(rates):.2%}")
    print(f"p10/p50/p90={np.percentile(rates, 10):.2%}/{np.percentile(rates, 50):.2%}/{np.percentile(rates, 90):.2%}")
    print(f"zero_mask_examples={zero}/{len(annotations)} ({zero / max(len(annotations), 1):.2%})")
    print("task_label\texamples\tmask_rate")
    for task, (examples, selected, tokens) in sorted(by_task.items()):
        print(f"{task}\t{examples}\t{selected / max(tokens, 1):.2%}")


if __name__ == "__main__":
    main()
