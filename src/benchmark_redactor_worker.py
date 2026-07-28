import argparse
import json
import resource
import statistics
import time
from pathlib import Path


MODEL_DIRS = {
    "bert_tiny": "artifacts/medical_redactor/core_matrix/drug_bert_tiny_seed42",
    "electra_small": "artifacts/medical_redactor/cross_dataset_in_domain/drug_electra_small",
    "distilroberta": "artifacts/medical_redactor/cross_dataset_in_domain/drug_distilroberta",
}


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_evenly(rows: list[dict], count: int) -> list[dict]:
    if count >= len(rows):
        return rows
    if count == 1:
        return [rows[0]]
    return [rows[round(index * (len(rows) - 1) / (count - 1))] for index in range(count)]


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def summarize_latency(values: list[float]) -> dict:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))
    mean = statistics.fmean(values)
    return {
        "mean_ms": mean,
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "sentences_per_second": 1000 / mean,
    }


def benchmark_medterm4(rows: list[dict], warmup: int, repeats: int) -> dict:
    baseline_rss = rss_mb()
    started = time.perf_counter()
    from annotate_medterm4 import annotate, load_pipelines

    science, linker, pii = load_pipelines(0.85)
    load_seconds = time.perf_counter() - started
    loaded_rss = rss_mb()
    for row in rows[:warmup]:
        annotate(row["text"], row["words"], science, linker, pii)
    latencies = []
    selected_tokens = 0
    total_tokens = 0
    for _ in range(repeats):
        for row in rows:
            started = time.perf_counter()
            labels, _ = annotate(row["text"], row["words"], science, linker, pii)
            latencies.append((time.perf_counter() - started) * 1000)
            selected_tokens += sum(labels)
            total_tokens += len(labels)
    peak_rss = rss_mb()
    return {
        "implementation": "scispaCy + UMLS linker + spaCy PII NER + policy filters",
        "load_seconds": load_seconds,
        "baseline_rss_mb": baseline_rss,
        "loaded_rss_mb": loaded_rss,
        "peak_rss_mb": peak_rss,
        "incremental_peak_rss_mb": peak_rss - baseline_rss,
        "model_state_mb": None,
        "artifact_total_mb": None,
        "latency": summarize_latency(latencies),
        "measured_mask_rate": selected_tokens / max(total_tokens, 1),
        "storage_note": "External shared spaCy/scispaCy/UMLS resources; no single self-contained checkpoint.",
    }


def benchmark_model(method: str, rows: list[dict], warmup: int, repeats: int, threads: int) -> dict:
    baseline_rss = rss_mb()
    started = time.perf_counter()
    import torch
    from transformers import AutoTokenizer
    from train import RedactionModel

    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    model_dir = Path(MODEL_DIRS[method])
    config = json.loads((model_dir / "experiment.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = RedactionModel(config["model_name"], config["hidden_size"], config["freeze_encoder"])
    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    load_seconds = time.perf_counter() - started
    loaded_rss = rss_mb()

    def infer(row: dict):
        item = tokenizer(
            row["words"],
            is_split_into_words=True,
            truncation=True,
            max_length=config["max_length"],
            padding="max_length",
            return_tensors="pt",
        )
        return model(input_ids=item["input_ids"], attention_mask=item["attention_mask"])

    with torch.inference_mode():
        for row in rows[:warmup]:
            infer(row)
        latencies = []
        for _ in range(repeats):
            for row in rows:
                started = time.perf_counter()
                infer(row)
                latencies.append((time.perf_counter() - started) * 1000)
    peak_rss = rss_mb()
    return {
        "implementation": config["model_name"] + " + hidden-128 MLP token head",
        "load_seconds": load_seconds,
        "baseline_rss_mb": baseline_rss,
        "loaded_rss_mb": loaded_rss,
        "peak_rss_mb": peak_rss,
        "incremental_peak_rss_mb": peak_rss - baseline_rss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_state_mb": (model_dir / "model.pt").stat().st_size / 1_000_000,
        "artifact_total_mb": directory_size(model_dir) / 1_000_000,
        "latency": summarize_latency(latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated CPU benchmark worker for one redactor")
    parser.add_argument(
        "--method", choices=("medterm4", *MODEL_DIRS), required=True
    )
    parser.add_argument(
        "--dataset", default="data/medical_redactor/cross_dataset/multidomain/test.jsonl"
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = select_evenly(read_jsonl(args.dataset), args.samples)
    if args.method == "medterm4":
        metrics = benchmark_medterm4(rows, min(args.warmup, 3), args.repeats)
    else:
        metrics = benchmark_model(args.method, rows, args.warmup, args.repeats, args.threads)
    result = {
        "method": args.method,
        "protocol": {
            "device": "cpu",
            "threads": args.threads,
            "batch_size": 1,
            "samples": len(rows),
            "repeats": args.repeats,
            "dataset": args.dataset,
            "student_max_length": 256,
        },
        "metrics": metrics,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    latency = metrics["latency"]
    print(
        f"{args.method}: median/p95={latency['median_ms']:.2f}/{latency['p95_ms']:.2f} ms, "
        f"throughput={latency['sentences_per_second']:.2f}/s, "
        f"peak_rss={metrics['peak_rss_mb']:.1f} MB, load={metrics['load_seconds']:.2f}s"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
