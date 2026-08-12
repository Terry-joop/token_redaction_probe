import argparse
import json

import numpy as np
import torch
from sklearn.metrics import fbeta_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common import read_jsonl
from train import RedactionModel, TokenDataset, pretokenize_rows


def collect(model, loader, device):
    gold, scores = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(**{key: value.to(device) for key, value in batch.items()})
            mask = labels != -100
            gold.extend(labels[mask].tolist())
            scores.extend(logits.softmax(-1)[..., 1].cpu()[mask].tolist())
    return np.asarray(gold), np.asarray(scores)


def calculate(gold, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, predicted, average="binary", zero_division=0
    )
    return {
        "threshold": float(threshold),
        "accuracy_secondary": float(np.mean(gold == predicted)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "f2": float(fbeta_score(gold, predicted, beta=2, average="binary", zero_division=0)),
        "gold_mask_rate": float(np.mean(gold)), "predicted_mask_rate": float(np.mean(predicted)),
        "residual_sensitive_rate": float(1 - recall), "evaluated_tokens": int(len(gold)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a medical student against its pseudo-teacher")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--pretokenize", action="store_true",
        help="Tokenize validation/test once in memory before GPU evaluation.",
    )
    parser.add_argument("--tokenization-batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    with open(f"{args.model_dir}/experiment.json", encoding="utf-8") as handle:
        config = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = RedactionModel(config["model_name"], config["hidden_size"], config["freeze_encoder"])
    model.load_state_dict(torch.load(f"{args.model_dir}/model.pt", map_location="cpu", weights_only=True))
    model.to(device)
    validation_rows = read_jsonl(args.validation)
    test_rows = read_jsonl(args.test)
    if args.pretokenize:
        validation_data = pretokenize_rows(
            validation_rows, tokenizer, config["max_length"],
            args.tokenization_batch_size, "validation",
        )
        test_data = pretokenize_rows(
            test_rows, tokenizer, config["max_length"],
            args.tokenization_batch_size, "test",
        )
    else:
        validation_data = TokenDataset(
            validation_rows, tokenizer, config["max_length"]
        )
        test_data = TokenDataset(test_rows, tokenizer, config["max_length"])
    loader_kwargs = {
        "batch_size": args.batch_size,
        "pin_memory": device.type == "cuda",
    }
    validation_loader = DataLoader(validation_data, **loader_kwargs)
    test_loader = DataLoader(test_data, **loader_kwargs)
    validation_gold, validation_scores = collect(model, validation_loader, device)
    test_gold, test_scores = collect(model, test_loader, device)
    candidates = np.linspace(0.05, 0.95, 91)
    validation_results = [calculate(validation_gold, validation_scores, value) for value in candidates]
    best_f2 = max(validation_results, key=lambda result: result["f2"])
    budget_matched = min(
        validation_results,
        key=lambda result: abs(result["predicted_mask_rate"] - result["gold_mask_rate"]),
    )
    result = {
        "f2_optimized": {
            "threshold_selected_by": "validation_f2",
            "validation": best_f2,
            "test": calculate(test_gold, test_scores, best_f2["threshold"]),
        },
        "budget_matched": {
            "threshold_selected_by": "validation_mask_rate",
            "validation": budget_matched,
            "test": calculate(test_gold, test_scores, budget_matched["threshold"]),
        },
        "warning": "This measures pseudo-teacher imitation, not human-gold privacy correctness.",
    }
    with open(f"{args.model_dir}/medical_evaluation.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
