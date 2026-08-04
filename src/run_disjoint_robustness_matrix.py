from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_strict_robustness_matrix import (
    DATASETS,
    PYTHON,
    ROOT,
    complete_json,
    parse_csv,
    run,
)


EXPERIMENT = "v14_disjoint"


def prepare(dataset: str, force: bool) -> None:
    config = DATASETS[dataset]
    data_dir = ROOT / "data" / "robustness" / EXPERIMENT / dataset
    clean_dir = data_dir / "clean"
    artifact_root = ROOT / "artifacts" / "robustness" / EXPERIMENT
    log_dir = artifact_root / "logs"
    for split in ("train", "validation", "test"):
        if not (clean_dir / f"{split}.jsonl").exists():
            raise FileNotFoundError(f"disjoint split missing: {dataset}/{split}")

    augmented = data_dir / "train_seen_augmented.jsonl"
    if force or not (
        augmented.exists() and complete_json(augmented.with_suffix(".summary.json"))
    ):
        run(
            [
                PYTHON,
                "src/robustness/augment_train.py",
                "--input",
                str(clean_dir / "train.jsonl"),
                "--output",
                str(augmented),
                "--variants-per-row",
                "1",
            ],
            log_dir / f"{dataset}_augment.log",
        )

    pairs = data_dir / "unseen_pairs.jsonl"
    if force or not (
        pairs.exists() and complete_json(pairs.with_suffix(".summary.json"))
    ):
        run(
            [
                PYTHON,
                "src/robustness/build_pairs.py",
                "--input",
                str(clean_dir / "test.jsonl"),
                "--output",
                str(pairs),
                "--per-noise",
                "0",
                "--noise-group",
                "unseen",
            ],
            log_dir / f"{dataset}_pairs.log",
        )

    rule_cache = data_dir / "unseen_rule_cache.jsonl"
    pair_summary = complete_json(pairs.with_suffix(".summary.json"))
    cache_summary = rule_cache.with_suffix(".summary.json")
    cache_ready = False
    if pair_summary and complete_json(cache_summary) and rule_cache.exists():
        expected = json.loads(
            pairs.with_suffix(".summary.json").read_text(encoding="utf-8")
        )["pairs"]
        actual = json.loads(cache_summary.read_text(encoding="utf-8"))["predictions"]
        cache_ready = expected == actual
    old_cache = (
        ROOT / "data" / "robustness" / "v14_strict" / dataset
        / "unseen_rule_cache.jsonl"
    )
    if not force and not cache_ready and old_cache.exists():
        from robustness.v14_rule_adapter import read_jsonl, write_jsonl

        old = {row["pair_id"]: row for row in read_jsonl(old_cache)}
        pair_rows = read_jsonl(pairs)
        if all(row["pair_id"] in old for row in pair_rows):
            write_jsonl(rule_cache, [old[row["pair_id"]] for row in pair_rows])
            cache_summary.write_text(
                json.dumps(
                    {
                        "pairs": str(pairs),
                        "output": str(rule_cache),
                        "predictions": len(pair_rows),
                        "masker": config.masker,
                        "task": config.task,
                        "reused_from": str(old_cache),
                        "reason": "disjoint test is an unchanged-text subset",
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            cache_ready = True
            print(f"REUSED RULE CACHE {dataset}: {len(pair_rows):,}", flush=True)
    if force or not cache_ready:
        run(
            [
                PYTHON,
                "src/robustness/build_rule_cache.py",
                "--pairs",
                str(pairs),
                "--output",
                str(rule_cache),
                "--masker",
                config.masker,
                "--task",
                config.task,
                "--redactformer-root",
                str(ROOT.parent / "Redactformer"),
                "--workers",
                "4",
                "--chunksize",
                "64",
            ],
            log_dir / f"{dataset}_rule_cache.log",
        )
    print(f"PREPARED {dataset}", flush=True)


def train_and_evaluate(
    dataset: str, seed: int, force: bool, legacy_eval: bool = False
) -> None:
    config = DATASETS[dataset]
    data_dir = ROOT / "data" / "robustness" / EXPERIMENT / dataset
    clean_dir = data_dir / "clean"
    artifact_root = ROOT / "artifacts" / "robustness" / EXPERIMENT
    model_dir = artifact_root / f"{dataset}_electra_small_seed{seed}"
    log_dir = artifact_root / "logs"
    experiment = model_dir / "experiment.json"
    if force or not (complete_json(experiment) and (model_dir / "model.pt").exists()):
        run(
            [
                PYTHON,
                "src/train.py",
                "--train",
                str(data_dir / "train_seen_augmented.jsonl"),
                "--validation",
                str(clean_dir / "validation.jsonl"),
                "--output-dir",
                str(model_dir),
                "--model-name",
                "google/electra-small-discriminator",
                "--epochs",
                "5",
                "--batch-size",
                "32",
                "--learning-rate",
                "0.001",
                "--encoder-learning-rate",
                "2e-5",
                "--head-learning-rate",
                "0.001",
                "--hidden-size",
                "128",
                "--max-length",
                "128",
                "--seed",
                str(seed),
                "--unfreeze-encoder",
                "--offline",
                "--device",
                "cuda",
            ],
            log_dir / f"{dataset}_seed{seed}_train.log",
        )

    calibration = model_dir / "medical_evaluation.json"
    if force or not complete_json(calibration):
        run(
            [
                PYTHON,
                "src/evaluate_medical_student.py",
                "--model-dir",
                str(model_dir),
                "--validation",
                str(clean_dir / "validation.jsonl"),
                "--test",
                str(clean_dir / "test.jsonl"),
                "--batch-size",
                "256",
                "--device",
                "cuda",
            ],
            log_dir / f"{dataset}_seed{seed}_clean_eval.log",
        )

    output = ROOT / "reports" / f"robustness_{EXPERIMENT}_{dataset}_seed{seed}.json"
    if legacy_eval and (force or not complete_json(output)):
        run(
            [
                PYTHON,
                "src/robustness/evaluate.py",
                "--pairs",
                str(data_dir / "unseen_pairs.jsonl"),
                "--model-dir",
                str(model_dir),
                "--masker",
                config.masker,
                "--task",
                config.task,
                "--noise-group",
                "unseen",
                "--device",
                "cuda",
                "--student-batch-size",
                "256",
                "--bootstrap-repeats",
                "2000",
                "--rule-cache",
                str(data_dir / "unseen_rule_cache.jsonl"),
                "--seed",
                str(seed),
                "--output",
                str(output),
            ],
            log_dir / f"{dataset}_seed{seed}_robustness.log",
        )
    print(f"COMPLETE {dataset} seed={seed}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leakage-free full-data strict seen-5/unseen-7 experiment"
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--stage", choices=["prepare", "train", "all"], default="all")
    parser.add_argument("--prepare-jobs", type=int, default=1)
    parser.add_argument("--train-jobs", type=int, default=3)
    parser.add_argument(
        "--legacy-eval",
        action="store_true",
        help="also run the older two-system bootstrap report",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = parse_csv(args.datasets, list(DATASETS), "dataset")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]

    if args.stage in {"prepare", "all"}:
        with ThreadPoolExecutor(max_workers=args.prepare_jobs) as pool:
            futures = [pool.submit(prepare, dataset, args.force) for dataset in datasets]
            for future in as_completed(futures):
                future.result()
    if args.stage in {"train", "all"}:
        jobs = [(dataset, seed) for dataset in datasets for seed in seeds]
        with ThreadPoolExecutor(max_workers=args.train_jobs) as pool:
            futures = {
                pool.submit(
                    train_and_evaluate,
                    dataset,
                    seed,
                    args.force,
                    args.legacy_eval,
                ): (dataset, seed)
                for dataset, seed in jobs
            }
            for future in as_completed(futures):
                future.result()


if __name__ == "__main__":
    main()
