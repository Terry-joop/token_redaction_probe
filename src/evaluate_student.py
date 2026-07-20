import argparse
import json

import numpy as np
import torch
from sklearn.metrics import fbeta_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from common import read_jsonl
from train import RedactionModel, TokenDataset


def collect(model, loader):
    gold, scores = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels")
            logits = model(**batch)
            mask = labels != -100
            gold.extend(labels[mask].tolist())
            scores.extend(logits.softmax(-1)[..., 1][mask].tolist())
    return np.asarray(gold), np.asarray(scores)


def calculate(gold, scores, threshold):
    predicted = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, predicted, average="binary", zero_division=0
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(np.mean(gold == predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(fbeta_score(gold, predicted, beta=2, average="binary", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="artifacts/student_teacher_v2")
    parser.add_argument("--validation", default="data/teacher_prepared/validation.jsonl")
    parser.add_argument("--test", default="data/teacher_prepared/test.jsonl")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = json.load(open(f"{args.model_dir}/experiment.json", encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = RedactionModel(config["model_name"], config["hidden_size"], config["freeze_encoder"])
    model.load_state_dict(
        torch.load(f"{args.model_dir}/model.pt", map_location="cpu", weights_only=True)
    )
    validation_loader = DataLoader(
        TokenDataset(read_jsonl(args.validation), tokenizer, config["max_length"]),
        batch_size=args.batch_size,
    )
    test_loader = DataLoader(
        TokenDataset(read_jsonl(args.test), tokenizer, config["max_length"]),
        batch_size=args.batch_size,
    )
    validation_gold, validation_scores = collect(model, validation_loader)
    test_gold, test_scores = collect(model, test_loader)
    candidates = np.linspace(0.05, 0.95, 91)
    validation_results = [
        calculate(validation_gold, validation_scores, threshold) for threshold in candidates
    ]
    best = max(validation_results, key=lambda result: result["f1"])
    result = {
        "validation": best,
        "test": calculate(test_gold, test_scores, best["threshold"]),
        "validation_tokens": len(validation_gold),
        "test_tokens": len(test_gold),
    }
    with open(f"{args.model_dir}/evaluation.json", "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
