import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "robustness"))

from build_pairs import (  # noqa: E402
    TRANSFORMS,
    apostrophe_variant,
    dosage_variant,
    labels_from_char_mask,
    space_variant,
)
from evaluate import shared_target_robustness  # noqa: E402


def row(text, words, labels):
    return {"id": "x", "text": text, "words": words, "labels": labels}


def test_double_space_preserves_gold_span():
    source = row(
        "She has blurred vision today.",
        ["She", "has", "blurred", "vision", "today", "."],
        [0, 0, 1, 1, 0, 0],
    )
    variant = space_variant(source, "  ", "double_space")
    assert variant is not None
    assert "blurred  vision" in variant.text
    words, _, labels = labels_from_char_mask(variant.text, variant.char_mask)
    assert [
        word for word, label in zip(words, labels) if label
    ] == ["blurred", "vision"]


def test_dosage_join_preserves_number_and_unit():
    source = row(
        "Paxil 25 mg daily.",
        ["Paxil", "25", "mg", "daily", "."],
        [1, 1, 1, 0, 0],
    )
    variant = dosage_variant(source, "", "dosage_join")
    assert variant is not None
    assert variant.text == "Paxil 25mg daily."
    words, _, labels = labels_from_char_mask(variant.text, variant.char_mask)
    assert [
        word for word, label in zip(words, labels) if label
    ] == ["Paxil", "25mg"]


def test_apostrophe_replacement_stays_sensitive():
    source = row(
        "Andrew's chart.",
        ["Andrew's", "chart", "."],
        [1, 0, 0],
    )
    variant = apostrophe_variant(source, "\x92\x92", "c1")
    assert variant is not None
    words, _, labels = labels_from_char_mask(variant.text, variant.char_mask)
    assert all(label for word, label in zip(words, labels) if "Andrew" in word)

def test_c1_training_transform_uses_one_control_character():
    source = row(
        "Andrew's chart.",
        ["Andrew's", "chart", "."],
        [1, 0, 0],
    )
    transform = next(
        fn for name, _, fn in TRANSFORMS if name == "c1_apostrophe"
    )
    variant = transform(source)
    assert variant is not None
    assert variant.text.count("\x92") == 1
    assert len(variant.text) == len(source["text"])


def test_shared_clean_correct_span_survival_is_paired():
    pair = {
        "clean_text": "Andrew chart",
        "clean_words": ["Andrew", "chart"],
        "clean_labels": [1, 0],
        "clean_target": [0, 6],
        "text": "Andrew chart",
        "words": ["Andrew", "chart"],
        "labels": [1, 0],
        "noisy_target": [0, 6],
    }
    result = shared_target_robustness(
        [pair],
        rule_clean=[[1, 0]],
        rule_noisy=[[0, 0]],
        student_clean=[[1, 0]],
        student_noisy=[[1, 0]],
        repeats=20,
        seed=42,
    )
    assert result["eligible_shared_clean_targets"] == 1
    assert result["rule_span_survival_rate"] == 0.0
    assert result["student_span_survival_rate"] == 1.0
    assert result["student_minus_rule"] == 1.0
    assert result["ci95"] == [1.0, 1.0]
