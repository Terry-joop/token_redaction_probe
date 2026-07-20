# Token Redaction Probe

A small research project for testing task-aware token redaction on the GLUE SST-2
sentiment dataset. It trains a token classifier to identify the smallest set of words
that reveals whether a review is positive or negative.

The project is intentionally separate from RedactFormer so the labeling, training,
and leakage-evaluation pipeline can be tested in isolation.

## What it does

1. Tokenizes SST-2 reviews into words.
2. Assigns each word a binary label: `1` for redact and `0` for keep.
3. Trains a Transformer encoder with a token-classification head.
4. Redacts predicted sentiment evidence from new text.
5. Measures how much task-label information remains after redaction.

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model and dataset downloads require internet access on the first run. Later runs can
use the scripts' offline options or locally cached models where supported.

## Quick start with heuristic labels

Use this path to verify that the basic pipeline works without teacher annotations.

```bash
# 1. Create weak token labels from a sentiment-word heuristic.
python src/make_pseudo_labels.py --train-size 1000 --validation-size 300

# 2. Train the token classifier with a frozen encoder.
python src/train.py --epochs 3 --batch-size 16

# 3. Inspect a prediction.
python src/predict.py \
  --text "This movie is absolutely wonderful but far too long."
```

The prediction output includes a redaction probability for each word and a version of
the sentence with selected words replaced by `[REDACTED]`.

## Data format

All datasets use JSON Lines: one JSON object per line.

```json
{"id":"train-0","text":"a wonderful movie","words":["a","wonderful","movie"],"labels":[0,1,0],"task_label":1,"source":"heuristic-v1"}
```

`words` and `labels` must have the same length. A label of `1` marks a word for
redaction; `0` keeps it visible.

## Manual ChatGPT teacher workflow

This repository does not call a teacher API. Teacher annotations were created by
submitting prepared JSONL batches to ChatGPT manually.

### 1. Create a balanced teacher input set

```bash
python src/make_chatgpt_teacher_input.py --size 1000 --chunk-size 100
```

This creates:

- `teacher/chatgpt_input_1000.jsonl`: the canonical 1,000-example input file.
- `teacher/chatgpt_chunks_100/`: temporary 100-example batches for manual submission.

The chunk directory is ignored by Git because it can be regenerated from the canonical
input file.

### 2. Ask ChatGPT to annotate each batch

Use `prompts/sst2_leakage_teacher_v2.txt` as the annotation instructions. ChatGPT must
return JSONL in the same order, using this schema:

```json
{"id":"sst2-train-123","labels":[0,1,0],"selected_words":["wonderful"],"needs_review":false,"review_reason":""}
```

Combine the returned batches into:

```text
data/sst2_teacher_annotations_1000_validated.jsonl
```

Before continuing, confirm that every input ID appears exactly once and that each
`labels` array matches the corresponding `words` array in length.

### 3. Merge and split the annotations

```bash
python src/prepare_teacher_dataset.py
```

The script validates the annotations, merges them with the source text, and writes
train, validation, test, and review subsets under `data/teacher_prepared/`.

### 4. Train and evaluate the student model

```bash
python src/train.py \
  --train data/teacher_prepared/train.jsonl \
  --validation data/teacher_prepared/validation.jsonl \
  --output-dir artifacts/student_teacher_v2 \
  --epochs 8 --batch-size 32

python src/evaluate_student.py \
  --model-dir artifacts/student_teacher_v2
```

To fine-tune the encoder instead of keeping it frozen, add `--unfreeze-encoder` and set
an appropriate encoder learning rate.

## Leakage evaluation

The attacker is trained on SST-2 examples that do not overlap with the teacher set.

```bash
python src/train_sst2_attacker.py

python src/evaluate_leakage.py \
  --student-dir artifacts/student_teacher_v2 \
  --attacker-dir artifacts/sst2_attacker_bert_tiny \
  --mode mask
```

Leakage evaluation compares the original text, teacher redactions, student redactions,
and random masks with matched redaction budgets. Use `--mode delete` to remove selected
words instead of replacing them with the attacker's mask token.

## Legacy 100-example validation files

The tracked `teacher/requests.jsonl`, `teacher/chatgpt_responses.jsonl`, and
`teacher/chatgpt_validated.jsonl` files contain an earlier 100-request/10-response pilot.
They are retained as small examples and are not the 1,000-example training dataset.

To test the generic response validator with the included sample:

```bash
cp teacher/responses.example.jsonl teacher/responses.jsonl
python src/validate_teacher_responses.py
```

Generated `*_rejected.jsonl` files are ignored by Git.

## Project layout

```text
prompts/    Teacher annotation instructions
src/        Data preparation, training, prediction, and evaluation scripts
teacher/    Tracked teacher inputs and small pilot examples
tests/      Unit tests
data/       Generated or locally reviewed datasets; ignored by Git
artifacts/  Trained models and evaluation outputs; ignored by Git
```

## Important limitations

- The heuristic labels are weak labels for pipeline testing, not research ground truth.
- SST-2 contains movie-review sentiment, not personally identifiable information.
- Here, "redaction" means hiding tokens that reveal the task label. It is not a general
  privacy or PII-redaction system.
- Only public or synthetic text should be submitted to an external teacher model.
- `data/` and `artifacts/` are intentionally excluded from Git and must be backed up
  separately if they contain results you need to preserve.
