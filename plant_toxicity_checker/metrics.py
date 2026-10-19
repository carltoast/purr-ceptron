"""Evaluation metrics matching the PlantNet-300K paper's own reproduction.

Pure NumPy, no torch import - a probability matrix and a label array are
plain arrays by the time they get here, so this module (and its tests)
never needs torch installed.
"""

import collections

import numpy as np


def top_k_accuracy(probas, labels, k):
    """Fraction of samples whose true label is in the top ``k`` classes
    by predicted probability.

    ``probas`` is (n_samples, n_classes); ``labels`` is (n_samples,) of
    integer class indices.
    """
    probas = np.asarray(probas)
    labels = np.asarray(labels)
    k = min(k, probas.shape[1])
    top_k = np.argpartition(-probas, kth=k - 1, axis=1)[:, :k]
    hits = (top_k == labels[:, None]).any(axis=1)
    return float(hits.mean())


def per_class_accuracy(probas, labels):
    """Per-class sample count and top-1/top-5 correctness.

    ``probas``/``labels`` as in ``top_k_accuracy``. Returns
    ``{class_index: {"n", "top1_correct", "top5_correct"}}`` for every
    class index present at least once in ``labels`` - a class with zero
    eval examples is omitted, not reported as 0/0. Lets callers check
    whether accuracy is uneven across classes, e.g. with PlantNet-300K's
    long-tailed species distribution.
    """
    probas = np.asarray(probas)
    labels = np.asarray(labels)
    top1_hits = probas.argmax(axis=1) == labels
    k5 = min(5, probas.shape[1])
    top5 = np.argpartition(-probas, kth=k5 - 1, axis=1)[:, :k5]
    top5_hits = (top5 == labels[:, None]).any(axis=1)

    stats = {}
    for label, top1_hit, top5_hit in zip(labels, top1_hits, top5_hits):
        label = int(label)
        entry = stats.setdefault(
            label, {"n": 0, "top1_correct": 0, "top5_correct": 0}
        )
        entry["n"] += 1
        entry["top1_correct"] += int(top1_hit)
        entry["top5_correct"] += int(top5_hit)
    return stats


def macro_top_k_accuracy(probas, labels, k):
    """Top-k accuracy computed per class, then averaged over classes
    with equal weight - the PlantNet-300K paper's headline metric
    (Section 4.1, "macro-average top-k accuracy").

    Plain (micro) top-k is dominated by the handful of species that
    hold most of the images, which is why the paper calls a high plain
    top-1 "misleading": a model can score ~80% while barely
    recognising the hundreds of species with a few images each. Macro
    -average weights every species the same, so it actually reflects
    performance across the long tail. For reference, the paper reports
    that no architecture it benchmarks exceeds ~40% macro-average
    top-1 on the v1 dataset.

    Classes absent from ``labels`` are skipped rather than counted as
    zero, matching ``per_class_accuracy``.
    """
    stats = per_class_accuracy(probas, labels)
    if not stats:
        return None
    field = "top1_correct" if k == 1 else "top5_correct"
    if k not in (1, 5):
        raise ValueError(f"macro_top_k_accuracy supports k=1 or k=5, got {k}")
    return float(
        np.mean([entry[field] / entry["n"] for entry in stats.values()])
    )


def toxic_confusion_rate(probas, labels, species_sides):
    """How often a confident top-1 prediction crosses the toxic/
    non-toxic boundary - the metric a toxicity-aware training
    objective (Phase 3) specifically targets, since top-1/top-5
    accuracy alone wouldn't necessarily show it moving.

    ``probas``/``labels`` as in ``top_k_accuracy``. ``species_sides``
    is a per-class-index array (``species_match.species_sides``'
    output) of ``1`` (toxic) / ``0`` (non-toxic) / ``-1`` (unknown).

    Only samples whose TRUE class has a known side are scored at all -
    a species with no toxicity signal can't have its confusions judged
    by a side it doesn't have. Among those, a top-1 prediction that
    itself lands on an unknown-side class is neither a confirmed same-
    side nor cross-side outcome - counted separately as
    ``ambiguous_rate``, not silently folded into either.

    Returns ``{"confusion_rate", "ambiguous_rate", "n_known_side"}``;
    the two rates are fractions of ``n_known_side`` (true-side-known
    samples), not of the whole dataset. Both rates are ``None`` if no
    sample has a known true side at all.
    """
    probas = np.asarray(probas)
    labels = np.asarray(labels)
    sides = np.asarray(species_sides)

    true_sides = sides[labels]
    known_mask = true_sides != -1
    n_known_side = int(known_mask.sum())
    if n_known_side == 0:
        return {
            "confusion_rate": None, "ambiguous_rate": None,
            "n_known_side": 0,
        }

    pred_sides = sides[probas.argmax(axis=1)]
    known_true_sides = true_sides[known_mask]
    known_pred_sides = pred_sides[known_mask]

    ambiguous = known_pred_sides == -1
    confusion = ~ambiguous & (known_pred_sides != known_true_sides)

    return {
        "confusion_rate": float(confusion.mean()),
        "ambiguous_rate": float(ambiguous.mean()),
        "n_known_side": n_known_side,
    }


def toxic_confusion_pairs(probas, labels, species_sides):
    """Per-pair detail behind ``toxic_confusion_rate``'s aggregate
    number: for samples whose true class has a known side, every
    (true class index, predicted class index) pair where the top-1
    prediction crosses to the opposite known side. A prediction
    landing on an unknown-side class is excluded here too (ambiguous,
    not a confirmed cross-side confusion - same convention as
    ``toxic_confusion_rate``).

    Repeat confusions (the same species pair crossed by multiple eval
    images) collapse into one row. Returns a list of ``(true_index,
    pred_index, count)``, sorted by count descending - the species
    names behind each index are a caller concern (this module stays
    index-only, matching every other function here).
    """
    probas = np.asarray(probas)
    labels = np.asarray(labels)
    sides = np.asarray(species_sides)

    true_sides = sides[labels]
    preds = probas.argmax(axis=1)
    pred_sides = sides[preds]

    confused = (
        (true_sides != -1) & (pred_sides != -1) & (pred_sides != true_sides)
    )
    pairs = zip(labels[confused].tolist(), preds[confused].tolist())
    counts = collections.Counter(pairs)
    return sorted(
        ((t, p, c) for (t, p), c in counts.items()),
        key=lambda row: row[2],
        reverse=True,
    )


def _lmbda_for_k(val_probas, k):
    """The global probability threshold making the average predicted-set
    size over ``val_probas`` equal ``k``.

    Matches the official PlantNet-300K reproduction exactly: flatten
    every (sample, class) probability, sort descending, and average the
    two values either side of the ``n_val * k``-th largest - see
    ``plantnet/PlantNet-300K``'s ``epoch.py`` (``val_epoch``).
    """
    val_probas = np.asarray(val_probas)
    n_val = val_probas.shape[0]
    flat = np.sort(val_probas.reshape(-1))[::-1]
    idx = n_val * k
    return 0.5 * (flat[idx - 1] + flat[idx])


def average_k_accuracy(val_probas, val_labels, test_probas, test_labels, k):
    """Average-k accuracy at ``k``, calibrated on validation, scored on
    test.

    Unlike ``top_k_accuracy`` (a fixed per-sample rank cutoff), this
    thresholds every sample's true-class probability against one global
    ``lmbda`` chosen so the *average* predicted-set size on the
    validation set equals ``k`` - some test samples may have 0 classes
    above threshold, others many, depending how peaked their prediction
    is. Returns the fraction of test samples where the true class's
    probability is >= that threshold.
    """
    val_probas = np.asarray(val_probas)
    test_probas = np.asarray(test_probas)
    test_labels = np.asarray(test_labels)
    lmbda = _lmbda_for_k(val_probas, k)
    true_class_probas = test_probas[
        np.arange(len(test_labels)), test_labels
    ]
    return float((true_class_probas >= lmbda).mean())
