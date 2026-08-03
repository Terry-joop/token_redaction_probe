import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "src" / "robustness"))

from build_pairs import (  # noqa: E402
    TRANSFORMS,
    apostrophe_variant,
    dosage_variant,
    labels_from_char_mask,
    limit_reached,
    space_variant,
    transforms_for_group,
)
from evaluate import (  # noqa: E402
    absolute_target_robustness,
    bootstrap_cluster_sums,
    shared_target_robustness,
    source_cluster_totals,
)


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


def test_zero_per_noise_means_unlimited():
    assert not limit_reached(0, 0)
    assert not limit_reached(10_000, 0)
    assert not limit_reached(2, 3)
    assert limit_reached(3, 3)


def test_strict_noise_groups_are_disjoint_five_and_seven():
    seen = transforms_for_group("seen")
    unseen = transforms_for_group("unseen")
    assert len(seen) == 5
    assert len(unseen) == 7
    assert {name for name, _, _ in seen}.isdisjoint(
        {name for name, _, _ in unseen}
    )
    assert transforms_for_group("all") == TRANSFORMS


def test_vectorized_source_bootstrap_matches_literal_resampling():
    pairs = [
        {"source_id": source_id}
        for source_id in ["a", "a", "b", "c", "c", "c"]
    ]
    values = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]
    )
    clusters = [[0, 1], [2], [3, 4, 5]]
    literal = []
    rng = np.random.default_rng(42)
    for _ in range(17):
        chosen = rng.integers(0, len(clusters), size=len(clusters))
        indices = [index for cluster in chosen for index in clusters[cluster]]
        literal.append(values[indices].mean(axis=0))

    totals = source_cluster_totals(
        pairs, np.column_stack([values, np.ones(len(pairs))])
    )
    sampled = bootstrap_cluster_sums(
        totals, 17, np.random.default_rng(42), batch_size=32
    )
    vectorized = sampled[:, :2] / sampled[:, 2, None]
    np.testing.assert_allclose(literal, vectorized)


def test_shared_clean_correct_span_survival_is_paired():
    pair = {
        "source_id": "x",
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



def test_absolute_target_robustness_uses_all_fixed_targets():
    pairs = []
    for source_id in ["a", "a", "b"]:
        pairs.append(
            {
                "source_id": source_id,
                "clean_text": "Andrew chart",
                "clean_words": ["Andrew", "chart"],
                "clean_labels": [1, 0],
                "clean_target": [0, 6],
                "text": "Andrew chart",
                "words": ["Andrew", "chart"],
                "labels": [1, 0],
                "noisy_target": [0, 6],
            }
        )
    result = absolute_target_robustness(
        pairs,
        rule_clean=[[1, 0], [1, 0], [1, 0]],
        rule_noisy=[[0, 0], [0, 0], [1, 0]],
        student_clean=[[1, 0], [1, 0], [0, 0]],
        student_noisy=[[1, 0], [0, 0], [1, 0]],
        repeats=20,
        seed=42,
    )
    assert result["target_pairs"] == 3
    assert result["unique_source_rows"] == 2
    assert result["rule_clean_target_detection"] == 1.0
    assert result["rule_noisy_target_detection"] == 1 / 3
    assert result["student_clean_target_detection"] == 2 / 3
    assert result["student_noisy_target_detection"] == 2 / 3
    assert result["student_minus_rule_noisy"] == 1 / 3


def test_perturbation_catalog_matches_runtime_and_documents_counts():
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from build_perturbation_catalog import CATALOG, build_html

    assert [(item["name"], item["group"]) for item in CATALOG] == [
        (name, group) for name, group, _ in TRANSFORMS
    ]
    assert sum(item["group"] == "seen" for item in CATALOG) == 5
    assert sum(item["group"] == "unseen" for item in CATALOG) == 7
    page = build_html()
    assert "10개 데이터셋 모두" in page
    assert "C1 artifact 불일치" not in page
    assert "U+0092</code> 한 문자" in page
    assert "학습에서 본 교란과 test에서 처음 본 교란" in page
    assert "종류별 최대 100쌍" in page
