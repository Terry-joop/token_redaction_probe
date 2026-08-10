"""Evaluate previously trained Students on post-training future defects.

The Student checkpoints are the strict v1.4 models: they only saw clean text
and the original five ``seen`` perturbations.  This runner never trains on the
``future`` group and never uses it to select a threshold.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DATASETS = {
    "drug": ("medterm5", "drug"),
    "symptom2dx": ("medterm5", "symptom2dx"),
    "mednli": ("medterm5", "mednli"),
    "redditmh": ("medterm5", "redditmh"),
    "bios": ("piiclean2", "bios"),
    "mrpc": ("piiclean2", "mrpc"),
}


def complete_json(path: Path) -> bool:
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except (OSError, json.JSONDecodeError):
        return False


def run(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-40:])
        raise RuntimeError(f"failed ({process.returncode}): {log_path}\n{tail}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Future-defect time-axis evaluation; no future noise enters training."
    )
    parser.add_argument("--datasets", default="drug,bios,mrpc")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--per-noise", type=int, default=2000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]

    for dataset in datasets:
        masker, task = DATASETS[dataset]
        clean_test = ROOT / "data" / "robustness" / "v14_strict" / dataset / "clean" / "test.jsonl"
        output_dir = ROOT / "data" / "robustness" / "v14_future" / dataset
        pairs = output_dir / "future_pairs.jsonl"
        cache = output_dir / "future_rule_cache.jsonl"
        logs = ROOT / "artifacts" / "robustness" / "v14_future" / "logs"

        if args.force or not (pairs.exists() and complete_json(pairs.with_suffix(".summary.json"))):
            run(
                [
                    PYTHON,
                    "src/robustness/build_pairs.py",
                    "--input",
                    str(clean_test),
                    "--output",
                    str(pairs),
                    "--per-noise",
                    str(args.per_noise),
                    "--noise-group",
                    "future",
                ],
                logs / f"{dataset}_pairs.log",
            )

        cache_summary = cache.with_suffix(".summary.json")
        pairs_summary = json.loads(pairs.with_suffix(".summary.json").read_text(encoding="utf-8"))
        cache_ok = cache.exists() and complete_json(cache_summary)
        if cache_ok:
            cache_ok = json.loads(cache_summary.read_text(encoding="utf-8"))["predictions"] == pairs_summary["pairs"]
        if args.force or not cache_ok:
            run(
                [
                    PYTHON,
                    "src/robustness/build_rule_cache.py",
                    "--pairs",
                    str(pairs),
                    "--output",
                    str(cache),
                    "--masker",
                    masker,
                    "--task",
                    task,
                    "--redactformer-root",
                    str(ROOT.parent / "Redactformer"),
                    "--workers",
                    "4",
                    "--chunksize",
                    "64",
                ],
                logs / f"{dataset}_rule_cache.log",
            )

        for seed in seeds:
            model_dir = ROOT / "artifacts" / "robustness" / "v14_strict" / f"{dataset}_electra_small_seed{seed}"
            if not (model_dir / "model.pt").exists():
                raise FileNotFoundError(f"missing strict seen-5 checkpoint: {model_dir}")
            result = ROOT / "reports" / f"future_v14_{dataset}_seed{seed}.json"
            if args.force or not complete_json(result):
                run(
                    [
                        PYTHON,
                        "src/robustness/evaluate.py",
                        "--pairs",
                        str(pairs),
                        "--model-dir",
                        str(model_dir),
                        "--masker",
                        masker,
                        "--task",
                        task,
                        "--noise-group",
                        "future",
                        "--device",
                        args.device,
                        "--student-batch-size",
                        "256",
                        "--bootstrap-repeats",
                        "2000",
                        "--rule-cache",
                        str(cache),
                        "--seed",
                        str(seed),
                        "--output",
                        str(result),
                    ],
                    logs / f"{dataset}_seed{seed}_evaluate.log",
                )
            print(f"COMPLETE {dataset} seed={seed}", flush=True)


if __name__ == "__main__":
    main()
