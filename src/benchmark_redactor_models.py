import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from common import read_jsonl
from train import RedactionModel


MODEL_DIRS = {
    "bert_tiny": "artifacts/medical_redactor/core_matrix/drug_bert_tiny_seed42",
    "electra_small": "artifacts/medical_redactor/cross_dataset_in_domain/drug_electra_small",
    "distilroberta": "artifacts/medical_redactor/cross_dataset_in_domain/drug_distilroberta",
}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def select_evenly(rows: list[dict], count: int) -> list[dict]:
    if count >= len(rows):
        return rows
    indexes = np.linspace(0, len(rows) - 1, count, dtype=int)
    return [rows[index] for index in indexes]


def artifact_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def benchmark_model(
    model_dir: Path,
    rows: list[dict],
    warmup: int,
    repeats: int,
) -> dict:
    started = time.perf_counter()
    config = json.loads((model_dir / "experiment.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = RedactionModel(config["model_name"], config["hidden_size"], config["freeze_encoder"])
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    load_seconds = time.perf_counter() - started

    encoded = [
        tokenizer(
            row["words"],
            is_split_into_words=True,
            truncation=True,
            max_length=config["max_length"],
            padding="max_length",
            return_tensors="pt",
        )
        for row in rows
    ]
    with torch.inference_mode():
        for item in encoded[:warmup]:
            model(input_ids=item["input_ids"], attention_mask=item["attention_mask"])

    forward_ms = []
    with torch.inference_mode():
        for _ in range(repeats):
            for item in encoded:
                started = time.perf_counter()
                model(input_ids=item["input_ids"], attention_mask=item["attention_mask"])
                forward_ms.append((time.perf_counter() - started) * 1000)

    end_to_end_ms = []
    with torch.inference_mode():
        for _ in range(repeats):
            for row in rows:
                started = time.perf_counter()
                item = tokenizer(
                    row["words"],
                    is_split_into_words=True,
                    truncation=True,
                    max_length=config["max_length"],
                    padding="max_length",
                    return_tensors="pt",
                )
                model(input_ids=item["input_ids"], attention_mask=item["attention_mask"])
                end_to_end_ms.append((time.perf_counter() - started) * 1000)

    parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "model_name": config["model_name"],
        "parameters": parameters,
        "model_state_mb": (model_dir / "model.pt").stat().st_size / 1_000_000,
        "artifact_total_mb": artifact_size(model_dir) / 1_000_000,
        "load_seconds": load_seconds,
        "forward_latency_ms": {
            "mean": statistics.fmean(forward_ms),
            "median": statistics.median(forward_ms),
            "p95": percentile(forward_ms, 95),
        },
        "end_to_end_latency_ms": {
            "mean": statistics.fmean(end_to_end_ms),
            "median": statistics.median(end_to_end_ms),
            "p95": percentile(end_to_end_ms, 95),
        },
        "end_to_end_sentences_per_second": 1000 / statistics.fmean(end_to_end_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local redactor model size and CPU latency")
    parser.add_argument(
        "--dataset", default="data/medical_redactor/cross_dataset/multidomain/test.jsonl"
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output", default="artifacts/medical_redactor/core_matrix/cpu_benchmark.json"
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    rows = select_evenly(read_jsonl(args.dataset), args.samples)
    result = {
        "protocol": {
            "device": "cpu",
            "threads": args.threads,
            "batch_size": 1,
            "samples": len(rows),
            "repeats": args.repeats,
            "max_length": 256,
            "dataset": args.dataset,
            "note": "Latency excludes disk loading; end-to-end includes tokenization and model forward.",
        },
        "models": {},
    }
    for name, path in MODEL_DIRS.items():
        measured = benchmark_model(Path(path), rows, args.warmup, args.repeats)
        result["models"][name] = measured
        print(
            f"{name}: params={measured['parameters']:,}, state={measured['model_state_mb']:.1f} MB, "
            f"end-to-end median/p95={measured['end_to_end_latency_ms']['median']:.2f}/"
            f"{measured['end_to_end_latency_ms']['p95']:.2f} ms"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
