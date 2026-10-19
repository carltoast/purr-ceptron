"""Tests for evaluation metrics (pure NumPy, no torch needed)."""

import numpy as np
import pytest

from plant_toxicity_checker.metrics import (
    average_k_accuracy,
    macro_top_k_accuracy,
    per_class_accuracy,
    top_k_accuracy,
    toxic_confusion_pairs,
    toxic_confusion_rate,
)

# 3 samples, 4 classes.
# sample 0: true=0, rank 1 (top prob) - correct at every k.
# sample 1: true=2, rank 3 - correct from k=3 on, wrong at k=1.
# sample 2: true=3, rank 4 (lowest prob) - only correct once k covers all
#           4 classes.
_PROBAS = np.array([
    [0.90, 0.05, 0.03, 0.02],
    [0.50, 0.30, 0.15, 0.05],
    [0.40, 0.35, 0.20, 0.05],
])
_LABELS = np.array([0, 2, 3])


def test_top_1_accuracy():
    assert top_k_accuracy(_PROBAS, _LABELS, 1) == 1 / 3


def test_top_3_accuracy():
    assert top_k_accuracy(_PROBAS, _LABELS, 3) == 2 / 3


def test_top_k_clamps_to_the_number_of_classes():
    # k=10 > 4 classes: every sample's true label is trivially "in the
    # top 10" once there are only 4 classes to rank at all.
    assert top_k_accuracy(_PROBAS, _LABELS, 10) == 1.0


def test_average_k_accuracy_worked_example():
    # lmbda for k=1: flatten [0.9,0.1,0.6,0.4], sorted desc [0.9,0.6,0.4,0.1];
    # n_val*k=2 -> 0.5*(sorted[1]+sorted[2]) = 0.5*(0.6+0.4) = 0.5.
    val_probas = np.array([[0.9, 0.1], [0.6, 0.4]])
    val_labels = np.array([0, 0])  # unused by the lmbda calibration itself

    test_probas = np.array([[0.7, 0.3], [0.2, 0.8], [0.4, 0.6]])
    test_labels = np.array([0, 1, 0])
    # true-class probas: 0.7 (>=0.5, correct), 0.8 (>=0.5, correct),
    # 0.4 (<0.5, wrong) -> 2/3.
    acc = average_k_accuracy(
        val_probas, val_labels, test_probas, test_labels, k=1,
    )
    assert acc == 2 / 3


# per_class_accuracy: 5 samples, 6 classes (needs >5 classes for a real
# top-1/top-5 distinction - top_k_accuracy's 4-class fixture above
# trivially puts every class in a "top 5").
#   sample 0: true=0, top1 correct, top5 correct.
#   sample 1: true=0, top1 WRONG (argmax=1), top5 correct (rank 3).
#   sample 2: true=1, top1 correct, top5 correct.
#   sample 3: true=2, top1 WRONG, top5 WRONG (class 2 is the unique min).
#   sample 4: true=0, top1 correct, top5 correct.
_CLASS_PROBAS = np.array([
    [0.90, 0.02, 0.02, 0.02, 0.02, 0.02],
    [0.10, 0.50, 0.08, 0.08, 0.08, 0.16],
    [0.02, 0.90, 0.02, 0.02, 0.02, 0.02],
    [0.30, 0.30, 0.01, 0.13, 0.13, 0.13],
    [0.80, 0.04, 0.04, 0.04, 0.04, 0.04],
])
_CLASS_LABELS = np.array([0, 0, 1, 2, 0])


def test_per_class_accuracy_counts_and_correctness_per_class():
    stats = per_class_accuracy(_CLASS_PROBAS, _CLASS_LABELS)

    assert stats[0] == {"n": 3, "top1_correct": 2, "top5_correct": 3}
    assert stats[1] == {"n": 1, "top1_correct": 1, "top5_correct": 1}
    assert stats[2] == {"n": 1, "top1_correct": 0, "top5_correct": 0}


def test_per_class_accuracy_omits_classes_absent_from_labels():
    stats = per_class_accuracy(_CLASS_PROBAS, _CLASS_LABELS)
    assert set(stats) == {0, 1, 2}


# macro_top_k_accuracy: the point of the metric is that it weights each
# CLASS equally, not each SAMPLE, so a common class getting everything
# right cannot hide a rare class getting everything wrong.
_MACRO_PROBAS = np.array([
    [0.9, 0.1],   # class 0, correct
    [0.9, 0.1],   # class 0, correct
    [0.9, 0.1],   # class 0, correct
    [0.9, 0.1],   # class 1, WRONG
])
_MACRO_LABELS = np.array([0, 0, 0, 1])


def test_macro_top_k_weights_classes_equally_not_samples():
    # plain top-1 is 3/4 = 75%, dominated by the 3-sample class;
    # macro-average is mean(100%, 0%) = 50%.
    assert top_k_accuracy(_MACRO_PROBAS, _MACRO_LABELS, 1) == 0.75
    assert macro_top_k_accuracy(_MACRO_PROBAS, _MACRO_LABELS, 1) == 0.5


def test_macro_top_k_matches_plain_when_classes_are_balanced():
    probas = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]])
    labels = np.array([0, 0, 1, 1])  # 2 samples each, one class half wrong
    assert macro_top_k_accuracy(probas, labels, 1) == top_k_accuracy(
        probas, labels, 1
    )


def test_macro_top_k_rejects_unsupported_k():
    with pytest.raises(ValueError):
        macro_top_k_accuracy(_MACRO_PROBAS, _MACRO_LABELS, 3)


# toxic_confusion_rate: 4 classes, sides = [toxic, non_toxic, unknown, toxic].
#   sample 0: true=0 (toxic),   pred=0 (toxic)   -> same side
#   sample 1: true=0 (toxic),   pred=1 (non_tox) -> CROSS-SIDE CONFUSION
#   sample 2: true=1 (non_tox), pred=2 (unknown) -> AMBIGUOUS
#   sample 3: true=2 (unknown) -> excluded entirely (no known true side)
#   sample 4: true=3 (toxic),   pred=3 (toxic)   -> same side
_SIDE_PROBAS = np.array([
    [0.70, 0.10, 0.10, 0.10],
    [0.20, 0.60, 0.10, 0.10],
    [0.10, 0.20, 0.60, 0.10],
    [0.25, 0.25, 0.25, 0.25],
    [0.10, 0.10, 0.10, 0.70],
])
_SIDE_LABELS = np.array([0, 0, 1, 2, 3])
_SPECIES_SIDES = [1, 0, -1, 1]


def test_toxic_confusion_rate_excludes_unknown_true_side():
    result = toxic_confusion_rate(_SIDE_PROBAS, _SIDE_LABELS, _SPECIES_SIDES)
    assert result["n_known_side"] == 4


def test_toxic_confusion_rate_counts_cross_side_predictions():
    result = toxic_confusion_rate(_SIDE_PROBAS, _SIDE_LABELS, _SPECIES_SIDES)
    assert result["confusion_rate"] == 1 / 4


def test_toxic_confusion_rate_counts_ambiguous_separately():
    result = toxic_confusion_rate(_SIDE_PROBAS, _SIDE_LABELS, _SPECIES_SIDES)
    assert result["ambiguous_rate"] == 1 / 4


def test_toxic_confusion_rate_none_when_no_known_true_side():
    all_unknown_sides = [-1, -1, -1, -1]
    result = toxic_confusion_rate(
        _SIDE_PROBAS, _SIDE_LABELS, all_unknown_sides,
    )
    assert result == {
        "confusion_rate": None, "ambiguous_rate": None, "n_known_side": 0,
    }


def test_toxic_confusion_pairs_finds_the_cross_side_pair():
    # Same fixture as toxic_confusion_rate's tests above: only sample 1
    # (true=0, pred=1) is a confirmed cross-side confusion.
    pairs = toxic_confusion_pairs(_SIDE_PROBAS, _SIDE_LABELS, _SPECIES_SIDES)
    assert pairs == [(0, 1, 1)]


def test_toxic_confusion_pairs_counts_repeats_and_sorts_by_count():
    # classes 0 (toxic) / 1 (non_toxic): two (true=0, pred=1)
    # confusions, one (true=1, pred=0), one correct prediction.
    probas = np.array([
        [0.1, 0.9],
        [0.2, 0.8],
        [0.9, 0.1],
        [0.6, 0.4],
    ])
    labels = np.array([0, 0, 1, 0])
    sides = [1, 0]
    pairs = toxic_confusion_pairs(probas, labels, sides)
    assert pairs == [(0, 1, 2), (1, 0, 1)]
