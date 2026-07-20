import argparse
import json

import torch
from transformers import AutoTokenizer

from common import render_redaction, word_tokenize
from train import RedactionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="artifacts/model")
    parser.add_argument("--text", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    config = json.load(open(f"{args.model_dir}/experiment.json", encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = RedactionModel(
        config["model_name"], config["hidden_size"], config["freeze_encoder"]
    )
    model.load_state_dict(torch.load(f"{args.model_dir}/model.pt", map_location="cpu", weights_only=True))
    model.eval()
    words = word_tokenize(args.text)
    encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True)
    with torch.no_grad():
        probabilities = model(**encoded).softmax(-1)[0, :, 1]
    word_scores = []
    seen = set()
    for token_index, word_id in enumerate(encoded.word_ids(0)):
        if word_id is not None and word_id not in seen:
            word_scores.append(float(probabilities[token_index]))
            seen.add(word_id)
    labels = [int(score >= args.threshold) for score in word_scores]
    for word, score, label in zip(words, word_scores, labels):
        print(f"{word:20s} redact_probability={score:.4f} label={label}")
    print("redacted:", render_redaction(words, labels))


if __name__ == "__main__":
    main()

