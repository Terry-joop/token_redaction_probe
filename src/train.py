import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import fbeta_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from common import read_jsonl


class TokenDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = self.tokenizer(
            row["words"], is_split_into_words=True, truncation=True,
            max_length=self.max_length, padding="max_length", return_tensors="pt",
        )
        word_ids = encoded.word_ids(batch_index=0)
        labels = []
        previous = None
        for word_id in word_ids:
            # Supervise only the first subword so long words do not get extra weight.
            if word_id is None or word_id == previous:
                labels.append(-100)
            else:
                labels.append(row["labels"][word_id])
            previous = word_id
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(labels, dtype=torch.long)
        return item


class RawRowDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class TokenBatchCollator:
    """Tokenize a full batch while preserving first-subword supervision."""

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        encoded = self.tokenizer(
            [row["words"] for row in rows],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        batch_labels = []
        for batch_index, row in enumerate(rows):
            labels = []
            previous = None
            for word_id in encoded.word_ids(batch_index=batch_index):
                if word_id is None or word_id == previous:
                    labels.append(-100)
                else:
                    labels.append(row["labels"][word_id])
                previous = word_id
            batch_labels.append(labels)
        encoded["labels"] = torch.tensor(batch_labels, dtype=torch.long)
        return encoded


class TensorDictDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors
        self.length = len(tensors["labels"])

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.tensors.items()}


def pretokenize_rows(rows, tokenizer, max_length, batch_size, split):
    collator = TokenBatchCollator(tokenizer, max_length)
    chunks = {}
    for start in range(0, len(rows), batch_size):
        stop = min(start + batch_size, len(rows))
        encoded = collator(rows[start:stop])
        for key, value in encoded.items():
            chunks.setdefault(key, []).append(value)
        if stop % 10000 < batch_size or stop == len(rows):
            print(f"pretokenize {split} {stop:,}/{len(rows):,}", flush=True)
    return TensorDictDataset(
        {key: torch.cat(values, dim=0) for key, values in chunks.items()}
    )


class RedactionModel(nn.Module):
    def __init__(self, model_name: str, hidden_size: int, freeze_encoder: bool):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        width = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(width, hidden_size), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden_size, 2)
        )

    def forward(self, input_ids, attention_mask, **kwargs):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(output.last_hidden_state)


def evaluate(model, loader, device):
    model.eval()
    gold, pred = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            logits = model(**{k: v.to(device) for k, v in batch.items()})
            mask = labels != -100
            gold.extend(labels[mask].cpu().tolist())
            pred.extend(logits.argmax(-1)[mask].cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, average="binary", zero_division=0
    )
    accuracy = float(np.mean(np.array(gold) == np.array(pred)))
    f2 = fbeta_score(gold, pred, beta=2, average="binary", zero_division=0)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--output-dir", default="artifacts/model")
    parser.add_argument("--model-name", default="prajjwal1/bert-tiny")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="Parallel DataLoader workers for per-example tokenization.",
    )
    parser.add_argument(
        "--batched-tokenization", action="store_true",
        help="Tokenize each full batch in one tokenizer call without changing alignment.",
    )
    parser.add_argument(
        "--pretokenize", action="store_true",
        help="Tokenize each split once in memory and reuse it across epochs.",
    )
    parser.add_argument("--tokenization-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--encoder-learning-rate", type=float, default=None,
        help="Encoder LR when fine-tuning; defaults to 2e-5 when --unfreeze-encoder is set.",
    )
    parser.add_argument(
        "--head-learning-rate", type=float, default=None,
        help="MLP classifier LR; defaults to --learning-rate.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unfreeze-encoder", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        cuda_usable = torch.cuda.is_available()
        if cuda_usable:
            capability = "sm_" + "".join(map(str, torch.cuda.get_device_capability()))
            cuda_usable = capability in torch.cuda.get_arch_list()
            if not cuda_usable:
                print(f"CUDA build does not support {capability}; falling back to CPU")
        device = torch.device("cuda" if cuda_usable else "cpu")
    else:
        device = torch.device(args.device)
    tokenizer_kwargs = {"local_files_only": args.offline}
    # RoBERTa byte-level BPE needs a leading-space marker for pre-tokenized words.
    if "roberta" in args.model_name.lower():
        tokenizer_kwargs["add_prefix_space"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, **tokenizer_kwargs)
    train_rows, val_rows = read_jsonl(args.train), read_jsonl(args.validation)
    if args.pretokenize:
        train_data = pretokenize_rows(
            train_rows, tokenizer, args.max_length,
            args.tokenization_batch_size, "train",
        )
        val_data = pretokenize_rows(
            val_rows, tokenizer, args.max_length,
            args.tokenization_batch_size, "validation",
        )
        collate_fn = None
    elif args.batched_tokenization:
        train_data = RawRowDataset(train_rows)
        val_data = RawRowDataset(val_rows)
        collate_fn = TokenBatchCollator(tokenizer, args.max_length)
    else:
        train_data = TokenDataset(train_rows, tokenizer, args.max_length)
        val_data = TokenDataset(val_rows, tokenizer, args.max_length)
        collate_fn = None
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, **loader_kwargs
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, collate_fn=collate_fn, **loader_kwargs
    )

    model = RedactionModel(args.model_name, args.hidden_size, not args.unfreeze_encoder).to(device)
    positive = sum(sum(row["labels"]) for row in train_rows)
    negative = sum(len(row["labels"]) - sum(row["labels"]) for row in train_rows)
    weights = torch.tensor([1.0, min(negative / max(positive, 1), 20.0)], device=device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=weights)
    head_lr = args.head_learning_rate or args.learning_rate
    encoder_lr = args.encoder_learning_rate or 2e-5
    if args.unfreeze_encoder:
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": encoder_lr},
            {"params": model.classifier.parameters(), "lr": head_lr},
        ])
    else:
        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=head_lr)

    best_metrics = None
    best_epoch = None
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            optimizer.zero_grad()
            logits = model(**{k: v.to(device) for k, v in batch.items()})
            loss = loss_fn(logits.view(-1, 2), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(loss.item())
        metrics = evaluate(model, val_loader, device)
        print(f"epoch={epoch} loss={np.mean(losses):.4f} metrics={metrics}", flush=True)
        if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
            best_metrics = metrics
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output / "model.pt")
    tokenizer.save_pretrained(output)
    config = vars(args) | {
        "freeze_encoder": not args.unfreeze_encoder,
        "effective_encoder_learning_rate": encoder_lr if args.unfreeze_encoder else None,
        "effective_head_learning_rate": head_lr,
        "class_weights": weights.detach().cpu().tolist(),
        "metrics": best_metrics,
        "best_epoch": best_epoch,
    }
    (output / "experiment.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved best epoch {best_epoch} to {output}")


if __name__ == "__main__":
    main()
