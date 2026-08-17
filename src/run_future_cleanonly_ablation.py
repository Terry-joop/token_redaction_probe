"""Clean-only Student ablation on the fixed Future-7 robustness test.

This is deliberately identical to ``run_future_defect_eval.py`` except that
the Student is trained on clean v1.4 teacher labels only.  It never sees the
five Seen perturbations, Unseen-7, or Future-7 during training, validation,
or threshold selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_future_defect_eval import DATASETS, PYTHON, ROOT, complete_json, run


def train_and_evaluate(dataset: str, seed: int, device: str, force: bool) -> None:
    masker, task = DATASETS[dataset]
    clean_dir = ROOT / "data" / "robustness" / "v14_strict" / dataset / "clean"
    future_dir = ROOT / "data" / "robustness" / "v14_future" / dataset
    pairs = future_dir / "future_pairs.jsonl"
    cache = future_dir / "future_rule_cache.jsonl"
    root = ROOT / "artifacts" / "robustness" / "v14_future_cleanonly"
    model_dir = root / f"{dataset}_electra_small_seed{seed}"
    log_dir = root / "logs"
    experiment = model_dir / "experiment.json"

    if force or not (complete_json(experiment) and (model_dir / "model.pt").exists()):
        run(
            [
                PYTHON, "src/train.py",
                "--train", str(clean_dir / "train.jsonl"),
                "--validation", str(clean_dir / "validation.jsonl"),
                "--output-dir", str(model_dir),
                "--model-name", "google/electra-small-discriminator",
                "--epochs", "5", "--batch-size", "32",
                "--learning-rate", "0.001", "--encoder-learning-rate", "2e-5",
                "--head-learning-rate", "0.001", "--hidden-size", "128",
                "--max-length", "128", "--seed", str(seed),
                "--unfreeze-encoder", "--offline", "--device", device,
            ],
            log_dir / f"{dataset}_seed{seed}_train.log",
        )

    calibration = model_dir / "medical_evaluation.json"
    if force or not complete_json(calibration):
        run(
            [
                PYTHON, "src/evaluate_medical_student.py",
                "--model-dir", str(model_dir),
                "--validation", str(clean_dir / "validation.jsonl"),
                "--test", str(clean_dir / "test.jsonl"),
                "--batch-size", "128", "--device", device,
            ],
            log_dir / f"{dataset}_seed{seed}_clean_eval.log",
        )

    result = ROOT / "reports" / f"future_v14_cleanonly_{dataset}_seed{seed}.json"
    if force or not complete_json(result):
        if not (pairs.exists() and cache.exists()):
            raise FileNotFoundError(
                f"Future pairs/cache missing for {dataset}; run run_future_defect_eval.py first"
            )
        run(
            [
                PYTHON, "src/robustness/evaluate.py",
                "--pairs", str(pairs), "--model-dir", str(model_dir),
                "--masker", masker, "--task", task, "--noise-group", "future",
                "--device", device, "--student-batch-size", "256",
                "--bootstrap-repeats", "2000", "--rule-cache", str(cache),
                "--seed", str(seed), "--output", str(result),
            ],
            log_dir / f"{dataset}_seed{seed}_evaluate.log",
        )
    print(f"COMPLETE clean-only {dataset} seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train clean-only Student and evaluate fixed Future-7 pairs"
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = list(DATASETS) if args.datasets == "all" else [x.strip() for x in args.datasets.split(",") if x.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    for dataset in datasets:
        for seed in seeds:
            train_and_evaluate(dataset, seed, args.device, args.force)


if __name__ == "__main__":
    main()
