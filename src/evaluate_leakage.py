import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common import read_jsonl
from train import RedactionModel, TokenDataset


def join_words(words):
    text = " ".join(words)
    return re.sub(r"\s+([.,!?;:])", r"\1", text)


def redact(words, labels, replacement):
    rendered = []
    for word, label in zip(words, labels):
        if not label:
            rendered.append(word)
        elif replacement is not None:
            rendered.append(replacement)
    return join_words(rendered)


def predict_student(rows, model_dir, batch_size):
    config = json.loads((Path(model_dir) / "experiment.json").read_text())
    evaluation = json.loads((Path(model_dir) / "evaluation.json").read_text())
    threshold = evaluation["validation"]["threshold"]
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = RedactionModel(
        config["model_name"], config["hidden_size"], config["freeze_encoder"]
    )
    model.load_state_dict(
        torch.load(Path(model_dir) / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    loader = DataLoader(
        TokenDataset(rows, tokenizer, config["max_length"]), batch_size=batch_size
    )
    predictions = []
    scores = []
    with torch.no_grad():
        for batch in loader:
            first_subword_mask = batch.pop("labels") != -100
            probabilities = model(**batch).softmax(-1)[..., 1]
            for row_scores, row_mask in zip(probabilities, first_subword_mask):
                word_scores = row_scores[row_mask].tolist()
                scores.append(word_scores)
                predictions.append([int(score >= threshold) for score in word_scores])
    for row, labels in zip(rows, predictions):
        if len(row["words"]) != len(labels):
            raise ValueError(f"{row['id']}: Student output length mismatch")
    return predictions, scores, threshold


def attacker_predictions(texts, tokenizer, model, batch_size):
    predictions, probabilities = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )
            logits = model(**encoded).logits
            probs = logits.softmax(-1)
            predictions.extend(logits.argmax(-1).tolist())
            probabilities.extend(probs.tolist())
    return np.asarray(predictions), np.asarray(probabilities)


def score_condition(texts, gold, tokenizer, model, batch_size, original_predictions=None):
    predicted, probabilities = attacker_predictions(
        texts, tokenizer, model, batch_size
    )
    result = {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro")),
        "mean_gold_label_probability": float(
            np.mean(probabilities[np.arange(len(gold)), gold])
        ),
    }
    if original_predictions is not None:
        originally_correct = original_predictions == gold
        result["originally_correct_examples"] = int(originally_correct.sum())
        result["accuracy_on_originally_correct"] = float(
            np.mean(predicted[originally_correct] == gold[originally_correct])
        )
        result["prediction_change_rate"] = float(np.mean(predicted != original_predictions))
    return result, predicted


def random_masks(rows, budgets, seed):
    rng = random.Random(seed)
    masks = []
    for row, budget in zip(rows, budgets):
        # Match the number of redacted lexical tokens. Teacher/Student almost never
        # select punctuation; keep the control within the same candidate space.
        eligible = [
            index for index, word in enumerate(row["words"])
            if any(character.isalnum() for character in word)
        ]
        budget = min(budget, len(eligible))
        selected = set(rng.sample(eligible, budget))
        masks.append([int(index in selected) for index in range(len(row["words"]))])
    return masks


def mask_statistics(rows, masks):
    total = sum(len(row["words"]) for row in rows)
    selected = sum(sum(mask) for mask in masks)
    return {
        "selected_tokens": selected,
        "total_tokens": total,
        "redaction_rate": selected / max(total, 1),
        "average_selected_tokens_per_example": selected / max(len(rows), 1),
        "zero_redaction_examples": sum(sum(mask) == 0 for mask in masks),
    }


def aggregate_random(results):
    keys = (
        "accuracy",
        "macro_f1",
        "mean_gold_label_probability",
        "accuracy_on_originally_correct",
        "prediction_change_rate",
    )
    return {
        key: {
            "mean": float(np.mean([result[key] for result in results])),
            "std": float(np.std([result[key] for result in results])),
        }
        for key in keys
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default="data/teacher_prepared/test.jsonl")
    parser.add_argument("--student-dir", default="artifacts/student_teacher_v2_finetuned")
    parser.add_argument("--attacker-dir", default="artifacts/sst2_attacker_bert_tiny")
    parser.add_argument(
        "--output", default="artifacts/leakage/bert_tiny_finetuned_mask.json"
    )
    parser.add_argument("--mode", choices=["mask", "delete"], default="mask")
    parser.add_argument("--random-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    rows = read_jsonl(args.test)
    gold = np.asarray([row["task_label"] for row in rows])
    teacher_masks = [row["labels"] for row in rows]
    student_masks, student_scores, threshold = predict_student(
        rows, args.student_dir, args.batch_size
    )

    attacker_tokenizer = AutoTokenizer.from_pretrained(
        args.attacker_dir, local_files_only=True
    )
    attacker_model = AutoModelForSequenceClassification.from_pretrained(
        args.attacker_dir, local_files_only=True
    )
    replacement = attacker_tokenizer.mask_token if args.mode == "mask" else None

    original_texts = [join_words(row["words"]) for row in rows]
    teacher_texts = [
        redact(row["words"], labels, replacement)
        for row, labels in zip(rows, teacher_masks)
    ]
    student_texts = [
        redact(row["words"], labels, replacement)
        for row, labels in zip(rows, student_masks)
    ]

    original_metrics, original_predictions = score_condition(
        original_texts, gold, attacker_tokenizer, attacker_model, args.batch_size
    )
    teacher_metrics, teacher_predictions = score_condition(
        teacher_texts,
        gold,
        attacker_tokenizer,
        attacker_model,
        args.batch_size,
        original_predictions,
    )
    student_metrics, student_predictions = score_condition(
        student_texts,
        gold,
        attacker_tokenizer,
        attacker_model,
        args.batch_size,
        original_predictions,
    )

    teacher_budgets = [sum(mask) for mask in teacher_masks]
    student_budgets = [sum(mask) for mask in student_masks]
    random_teacher_results, random_student_results = [], []
    for trial in range(args.random_trials):
        trial_seed = args.seed + trial
        for budgets, destination in (
            (teacher_budgets, random_teacher_results),
            (student_budgets, random_student_results),
        ):
            masks = random_masks(rows, budgets, trial_seed)
            texts = [
                redact(row["words"], labels, replacement)
                for row, labels in zip(rows, masks)
            ]
            metrics, _ = score_condition(
                texts,
                gold,
                attacker_tokenizer,
                attacker_model,
                args.batch_size,
                original_predictions,
            )
            destination.append(metrics)

    result = {
        "definition": (
            "Fixed-attacker task-label leakage; lower attacker accuracy after "
            "redaction means less SST-2 label leakage."
        ),
        "test_file": args.test,
        "test_examples": len(rows),
        "attacker_dir": args.attacker_dir,
        "student_dir": args.student_dir,
        "student_threshold_selected_on_validation": threshold,
        "redaction_mode": args.mode,
        "random_trials": args.random_trials,
        "mask_statistics": {
            "teacher": mask_statistics(rows, teacher_masks),
            "student": mask_statistics(rows, student_masks),
        },
        "conditions": {
            "original": original_metrics,
            "teacher": teacher_metrics,
            "student": student_metrics,
            "random_teacher_budget": aggregate_random(random_teacher_results),
            "random_student_budget": aggregate_random(random_student_results),
        },
        "accuracy_drop_from_original": {
            "teacher": original_metrics["accuracy"] - teacher_metrics["accuracy"],
            "student": original_metrics["accuracy"] - student_metrics["accuracy"],
            "random_teacher_budget_mean": (
                original_metrics["accuracy"]
                - aggregate_random(random_teacher_results)["accuracy"]["mean"]
            ),
            "random_student_budget_mean": (
                original_metrics["accuracy"]
                - aggregate_random(random_student_results)["accuracy"]["mean"]
            ),
        },
        "examples": [
            {
                "id": row["id"],
                "task_label": int(label),
                "original": original,
                "teacher_redacted": teacher,
                "student_redacted": student,
                "teacher_labels": teacher_mask,
                "student_labels": student_mask,
                "student_scores": [round(score, 6) for score in scores],
                "attacker_predictions": {
                    "original": int(original_prediction),
                    "teacher": int(teacher_prediction),
                    "student": int(student_prediction),
                },
            }
            for (
                row,
                label,
                original,
                teacher,
                student,
                teacher_mask,
                student_mask,
                scores,
                original_prediction,
                teacher_prediction,
                student_prediction,
            ) in zip(
                rows,
                gold,
                original_texts,
                teacher_texts,
                student_texts,
                teacher_masks,
                student_masks,
                student_scores,
                original_predictions,
                teacher_predictions,
                student_predictions,
            )
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "examples"}, indent=2))


if __name__ == "__main__":
    main()
