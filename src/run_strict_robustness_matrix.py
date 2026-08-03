from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


@dataclass(frozen=True)
class DatasetConfig:
    masker: str
    task: str


DATASETS = {
    "drug": DatasetConfig("medterm5", "drug"),
    "symptom2dx": DatasetConfig("medterm5", "symptom2dx"),
    "adr": DatasetConfig("medterm5", "adr"),
    "redditmh": DatasetConfig("medterm5", "redditmh"),
    "mednli": DatasetConfig("medterm5", "mednli"),
    "mentalhealth": DatasetConfig("medterm5", "mentalhealth"),
    "bios": DatasetConfig("piiclean2", "bios"),
    "mrpc": DatasetConfig("piiclean2", "mrpc"),
    "qnli": DatasetConfig("piiclean2", "qnli"),
    "finphrasebank": DatasetConfig("piiclean2", "finphrasebank"),
}


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
        raise RuntimeError(f"command failed ({process.returncode}): {log_path}\n{tail}")


def complete_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return True


def prepare(dataset: str, force: bool) -> None:
    config = DATASETS[dataset]
    data_dir = ROOT / "data" / "robustness" / "v14_strict" / dataset
    clean_dir = data_dir / "clean"
    log_dir = ROOT / "artifacts" / "robustness" / "v14_strict" / "logs"
    summary = clean_dir / "summary.json"
    required_clean = [clean_dir / f"{split}.jsonl" for split in ("train", "validation", "test")]
    if force or not (complete_json(summary) and all(path.exists() for path in required_clean)):
        run(
            [
                PYTHON,
                "src/robustness/annotate_splits.py",
                "--input-dir",
                f"data/full_redactor/{dataset}",
                "--output-dir",
                f"data/robustness/v14_strict/{dataset}/clean",
                "--masker",
                config.masker,
                "--task",
                config.task,
                "--train-limit",
                "0",
                "--validation-limit",
                "0",
                "--test-limit",
                "0",
                "--max-length",
                "128",
            ],
            log_dir / f"{dataset}_annotate.log",
        )

    augmented = data_dir / "train_seen_augmented.jsonl"
    augmented_summary = augmented.with_suffix(".summary.json")
    if force or not (augmented.exists() and complete_json(augmented_summary)):
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
    pairs_summary = pairs.with_suffix(".summary.json")
    if force or not (pairs.exists() and complete_json(pairs_summary)):
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
    print(f"PREPARED {dataset}", flush=True)


def train_and_evaluate(dataset: str, seed: int, force: bool) -> None:
    config = DATASETS[dataset]
    data_dir = ROOT / "data" / "robustness" / "v14_strict" / dataset
    clean_dir = data_dir / "clean"
    artifact_root = ROOT / "artifacts" / "robustness" / "v14_strict"
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
                "128",
                "--device",
                "cuda",
            ],
            log_dir / f"{dataset}_seed{seed}_clean_eval.log",
        )

    rule_cache = data_dir / "unseen_rule_cache.jsonl"
    output = ROOT / "reports" / f"robustness_v14_strict_{dataset}_seed{seed}.json"
    if force or not complete_json(output):
        rule_cache_summary = rule_cache.with_suffix(".summary.json")
        pair_summary = json.loads(
            (data_dir / "unseen_pairs.summary.json").read_text(encoding="utf-8")
        )
        cache_ready = (
            complete_json(rule_cache_summary)
            and rule_cache.exists()
            and json.loads(rule_cache_summary.read_text(encoding="utf-8"))[
                "predictions"
            ]
            == pair_summary["pairs"]
        )
        if force or not cache_ready:
            run(
                [
                    PYTHON,
                    "src/robustness/build_rule_cache.py",
                    "--pairs",
                    str(data_dir / "unseen_pairs.jsonl"),
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
                str(rule_cache),
                "--seed",
                str(seed),
                "--output",
                str(output),
            ],
            log_dir / f"{dataset}_seed{seed}_robustness.log",
        )
    print(f"COMPLETE {dataset} seed={seed}", flush=True)


def parse_csv(value: str, valid: list[str], label: str) -> list[str]:
    values = list(valid) if value == "all" else [item.strip() for item in value.split(",") if item.strip()]
    unknown = [item for item in values if item not in valid]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full-data strict seen-5 train / unseen-7 test robustness matrix"
    )
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--stage", choices=["prepare", "train", "all"], default="all")
    parser.add_argument("--prepare-jobs", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    datasets = parse_csv(args.datasets, list(DATASETS), "dataset")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]

    if args.stage in {"prepare", "all"}:
        with ThreadPoolExecutor(max_workers=args.prepare_jobs) as pool:
            futures = {pool.submit(prepare, dataset, args.force): dataset for dataset in datasets}
            for future in as_completed(futures):
                future.result()

    if args.stage in {"train", "all"}:
        for dataset in datasets:
            data_dir = ROOT / "data" / "robustness" / "v14_strict" / dataset
            if not (data_dir / "unseen_pairs.jsonl").exists():
                raise FileNotFoundError(f"prepare stage missing for {dataset}")
            for seed in seeds:
                train_and_evaluate(dataset, seed, args.force)


if __name__ == "__main__":
    main()
