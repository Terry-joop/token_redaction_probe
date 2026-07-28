import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from evaluate_medical_redactors import score_labels
from medical_common import labels_to_spans, word_offsets


def test_word_offsets_and_spans():
    words, offsets = word_offsets("I took Lexapro for anxiety.")
    assert words == ["I", "took", "Lexapro", "for", "anxiety", "."]
    assert offsets[2] == (7, 14)
    assert labels_to_spans([0, 0, 1, 0, 1, 0]) == [(2, 3), (4, 5)]


def test_redactor_metrics_do_not_hide_class_imbalance():
    result = score_labels({"x": [0, 0, 1, 1]}, {"x": [0, 0, 1, 0]})
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
    assert result["residual_sensitive_rate"] == 0.5
    assert result["accuracy_secondary"] == 0.75
