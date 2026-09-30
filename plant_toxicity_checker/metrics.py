"""Evaluation metrics matching the PlantNet-300K paper's own reproduction.

Pure NumPy, no torch import - a probability matrix and a label array are
plain arrays by the time they get here, so this module (and its tests)
never needs torch installed.
"""

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
