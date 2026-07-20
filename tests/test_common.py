import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from common import render_redaction, word_tokenize
from make_pseudo_labels import heuristic_labels


def test_tokenization_and_redaction():
    words = word_tokenize("It's absolutely wonderful!")
    assert words == ["It's", "absolutely", "wonderful", "!"]
    labels = heuristic_labels(words)
    assert labels == [0, 1, 1, 0]
    assert render_redaction(words, labels) == "It's [REDACTED] [REDACTED]!"

