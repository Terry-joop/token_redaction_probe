import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import fbeta_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common import read_jsonl
from train import RedactionModel, TokenDataset


@torch.no_grad()
def collect(model, loader, device):
    gold, scores = [], []
    model.eval()
    for batch in loader:
        labels = batch.pop("labels").to(device)
        logits = model(**{key: value.to(device) for key, value in batch.items()})
        mask = labels != -100
        gold.extend(labels[mask].cpu().tolist())
        scores.extend(logits.softmax(-1)[..., 1][mask].cpu().tolist())
    return np.asarray(gold), np.asarray(scores)


def calculate(gold, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, predicted, average="binary", zero_division=0
    )
    return {
        "threshold": float(threshold),
        "accuracy_secondary": float(np.mean(gold == predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(fbeta_score(gold, predicted, beta=2, average="binary", zero_division=0)),
        "gold_mask_rate": float(np.mean(gold)),
        "predicted_mask_rate": float(np.mean(predicted)),
        "residual_sensitive_rate": float(1 - recall),
        "evaluated_tokens": int(len(gold)),
    }


def parse_dataset(value: str):
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("dataset must be NAME:VALIDATION_JSONL:TEST_JSONL")
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fixed-threshold cross-dataset redactor evaluation")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True)
    parser.add_argument("--source-budget-threshold", type=float, required=True)
    parser.add_argument("--source-f2-threshold", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    with open(Path(args.model_dir) / "experiment.json", encoding="utf-8") as handle:
        config = json.load(handle)
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = RedactionModel(config["model_name"], config["hidden_size"], config["freeze_encoder"])
    state = torch.load(Path(args.model_dir) / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device)

    candidates = np.linspace(0.05, 0.95, 91)
    results = {
        "model_dir": args.model_dir,
        "model_name": config["model_name"],
        "source_thresholds": {
            "budget_matched": args.source_budget_threshold,
            "f2_optimized": args.source_f2_threshold,
        },
        "datasets": {},
        "warning": "Fixed thresholds are true source-to-target zero-shot; calibrated thresholds use target validation.",
    }
    for name, validation_path, test_path in args.dataset:
        validation_loader = DataLoader(
            TokenDataset(read_jsonl(validation_path), tokenizer, config["max_length"]),
            batch_size=args.batch_size,
        )
        test_loader = DataLoader(
            TokenDataset(read_jsonl(test_path), tokenizer, config["max_length"]),
            batch_size=args.batch_size,
        )
        validation_gold, validation_scores = collect(model, validation_loader, device)
        test_gold, test_scores = collect(model, test_loader, device)
        validation_grid = [calculate(validation_gold, validation_scores, value) for value in candidates]
        best_f2 = max(validation_grid, key=lambda result: result["f2"])
        budget = min(
            validation_grid,
            key=lambda result: abs(result["predicted_mask_rate"] - result["gold_mask_rate"]),
        )
        results["datasets"][name] = {
            "zero_shot_fixed": {
                "budget_matched_source_threshold": calculate(
                    test_gold, test_scores, args.source_budget_threshold
                ),
                "f2_optimized_source_threshold": calculate(
                    test_gold, test_scores, args.source_f2_threshold
                ),
            },
            "target_calibrated_no_training": {
                "budget_matched": {
                    "validation": budget,
                    "test": calculate(test_gold, test_scores, budget["threshold"]),
                },
                "f2_optimized": {
                    "validation": best_f2,
                    "test": calculate(test_gold, test_scores, best_f2["threshold"]),
                },
            },
        }
        fixed = results["datasets"][name]["zero_shot_fixed"]["budget_matched_source_threshold"]
        print(f"{name}: fixed-budget P/R/F1={fixed['precision']:.3f}/{fixed['recall']:.3f}/{fixed['f1']:.3f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
