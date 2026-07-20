import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ID_RE = re.compile(r"^sst2-train-(\d+)$")


class SST2Dataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["sentence"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(row["label"], dtype=torch.long)
        return item


def excluded_teacher_indices(path):
    excluded = set()
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            identifier = json.loads(line)["id"]
            match = ID_RE.match(identifier)
            if match:
                excluded.add(int(match.group(1)))
    return excluded


def evaluate(model, loader, device):
    model.eval()
    gold, predicted = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            gold.extend(labels.cpu().tolist())
            predicted.extend(logits.argmax(-1).cpu().tolist())
    return {
        "accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": float(f1_score(gold, predicted, average="macro")),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="prajjwal1/bert-tiny")
    parser.add_argument("--teacher-input", default="teacher/chatgpt_input_1000.jsonl")
    parser.add_argument("--output-dir", default="artifacts/sst2_attacker_bert_tiny")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    raw = load_dataset("glue", "sst2")
    excluded = excluded_teacher_indices(args.teacher_input)
    available = [index for index in range(len(raw["train"])) if index not in excluded]
    random.Random(args.seed).shuffle(available)
    selected = available[: min(args.train_size, len(available))]
    train_rows = [raw["train"][index] for index in selected]
    validation_rows = list(raw["validation"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    train_loader = DataLoader(
        SST2Dataset(train_rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        SST2Dataset(validation_rows, tokenizer, args.max_length),
        batch_size=args.batch_size,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, local_files_only=True
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    best_accuracy = -1.0
    best_epoch = None
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            optimizer.zero_grad()
            output = model(
                **{key: value.to(device) for key, value in batch.items()},
                labels=labels,
            )
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(output.loss.detach().cpu()))
        metrics = evaluate(model, validation_loader, device)
        entry = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    experiment = {
        **vars(args),
        "excluded_teacher_examples": len(excluded),
        "actual_train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "history": history,
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    print(f"saved={output_dir} best_epoch={best_epoch}", flush=True)


if __name__ == "__main__":
    main()
