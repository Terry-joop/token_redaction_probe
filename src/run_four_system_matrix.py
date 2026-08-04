from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_strict_robustness_matrix import DATASETS, PYTHON, ROOT, complete_json, parse_csv, run


def evaluate(dataset: str, seed: int, experiment: str, force: bool) -> None:
    data_dir = ROOT / "data" / "robustness" / experiment / dataset
    model_dir = (
        ROOT / "artifacts" / "robustness" / experiment
        / f"{dataset}_electra_small_seed{seed}"
    )
    output = ROOT / "reports" / f"four_system_{experiment}_{dataset}_seed{seed}.json"
    if not force and complete_json(output):
        print(f"FOUR SYSTEM READY {dataset} seed={seed}", flush=True)
        return
    run(
        [
            PYTHON,
            "src/robustness/evaluate_four_systems.py",
            "--pairs",
            str(data_dir / "unseen_pairs.jsonl"),
            "--raw-rule-cache",
            str(data_dir / "unseen_rule_cache.jsonl"),
            "--normalized-rule-cache",
            str(data_dir / "unseen_normalized_rule_cache.jsonl"),
            "--model-dir",
            str(model_dir),
            "--device",
            "cuda",
            "--batch-size",
            "256",
            "--seed",
            str(seed),
            "--output",
            str(output),
        ],
        ROOT / "artifacts" / "robustness" / experiment / "logs"
        / f"{dataset}_seed{seed}_four_system.log",
    )
    print(f"FOUR SYSTEM COMPLETE {dataset} seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run four-system robustness matrix")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--experiment", default="v14_disjoint")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = parse_csv(args.datasets, list(DATASETS), "dataset")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(evaluate, dataset, seed, args.experiment, args.force): (dataset, seed)
            for dataset, seed in jobs
        }
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
