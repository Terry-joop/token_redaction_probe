from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_strict_robustness_matrix import DATASETS, PYTHON, ROOT, parse_csv, run


def complete(path: Path, expected: int) -> bool:
    summary = path.with_suffix(".summary.json")
    if not path.exists() or not summary.exists():
        return False
    try:
        return json.loads(summary.read_text(encoding="utf-8"))["pairs"] == expected
    except (OSError, json.JSONDecodeError, KeyError):
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build normalized-rule caches")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--experiment", default="v14_disjoint")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    datasets = parse_csv(args.datasets, list(DATASETS), "dataset")
    log_dir = ROOT / "artifacts" / "robustness" / args.experiment / "logs"
    for dataset in datasets:
        config = DATASETS[dataset]
        data_dir = ROOT / "data" / "robustness" / args.experiment / dataset
        pairs = data_dir / "unseen_pairs.jsonl"
        expected = json.loads(
            pairs.with_suffix(".summary.json").read_text(encoding="utf-8")
        )["pairs"]
        output = data_dir / "unseen_normalized_rule_cache.jsonl"
        if complete(output, expected):
            print(f"NORMALIZED CACHE READY {dataset}", flush=True)
            continue
        run(
            [
                PYTHON,
                "src/robustness/build_normalized_rule_cache.py",
                "--pairs",
                str(pairs),
                "--output",
                str(output),
                "--masker",
                config.masker,
                "--task",
                config.task,
                "--redactformer-root",
                str(ROOT.parent / "Redactformer"),
                "--workers",
                str(args.workers),
            ],
            log_dir / f"{dataset}_normalized_rule_cache.log",
        )
        print(f"NORMALIZED CACHE COMPLETE {dataset}", flush=True)


if __name__ == "__main__":
    main()
