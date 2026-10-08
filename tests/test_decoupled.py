"""Tests for decoupled.py's composable pieces.

``load_config``/``features_path`` are plain JSON/path logic - no torch,
run in the default suite. The loss functions build small synthetic
tensors (no real data/checkpoint needed) and are marked ``slow``,
matching this project's convention that the default ``pytest`` run
never requires torch to be installed.
"""

import json

import pytest

from plant_toxicity_checker.decoupled import features_path, load_config
from plant_toxicity_checker.train import ConfigError

BASE_CONFIG = {"loss": "ce", "epochs": 10}


def write_config(tmp_path, **overrides):
    data = {**BASE_CONFIG, **overrides}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_config_reads_a_valid_file(tmp_path):
    path = write_config(tmp_path, loss="cb", cb_beta=0.99)
    config = load_config(path)
    assert config["loss"] == "cb"
    assert config["cb_beta"] == 0.99


def test_load_config_rejects_missing_required_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"loss": "ce"}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


def test_load_config_rejects_unknown_loss(tmp_path):
    path = write_config(tmp_path, loss="not_a_real_loss")
    with pytest.raises(ConfigError):
        load_config(path)


def test_features_path_is_keyed_by_backbone_run_and_split():
    from plant_toxicity_checker.train import OUTPUTS_DIR

    path = features_path("plantnet-baseline", "val")
    assert path == OUTPUTS_DIR / "plantnet-baseline" / "features_val.npz"


@pytest.mark.slow
def test_class_balanced_weights_matches_the_formula():
    from plant_toxicity_checker.decoupled import class_balanced_weights

    # 3 classes, counts [1, 10, 100], beta=0.9 for easy hand-checking.
    counts = [1, 10, 100]
    beta = 0.9
    weights = class_balanced_weights(counts, beta=beta)

    raw = [(1 - beta) / (1 - beta**n) for n in counts]
    expected = [r / (sum(raw) / len(raw)) for r in raw]
    assert weights.tolist() == pytest.approx(expected, rel=1e-5)


@pytest.mark.slow
def test_class_balanced_weights_gives_more_weight_to_rare_classes():
    from plant_toxicity_checker.decoupled import class_balanced_weights

    weights = class_balanced_weights([1, 1000], beta=0.9999)
    assert weights[0] > weights[1]


@pytest.mark.slow
def test_class_balanced_weights_zeroes_out_absent_classes():
    from plant_toxicity_checker.decoupled import class_balanced_weights

    weights = class_balanced_weights([5, 0, 5], beta=0.9999)
    assert weights[1] == 0.0


@pytest.mark.slow
def test_balanced_softmax_logits_adds_log_class_ratio():
    import torch

    from plant_toxicity_checker.decoupled import balanced_softmax_logits

    logits = torch.zeros((1, 3))
    counts = [10, 20, 70]
    adjusted = balanced_softmax_logits(logits, counts)

    total = sum(counts)
    expected = [torch.log(torch.tensor(c / total)).item() for c in counts]
    assert adjusted.tolist()[0] == pytest.approx(expected, rel=1e-5)


@pytest.mark.slow
def test_balanced_softmax_logits_tau_scales_the_adjustment():
    import torch

    from plant_toxicity_checker.decoupled import balanced_softmax_logits

    logits = torch.zeros((1, 3))
    counts = [10, 20, 70]

    baseline = balanced_softmax_logits(logits, counts, tau=1.0)
    doubled = balanced_softmax_logits(logits, counts, tau=2.0)
    disabled = balanced_softmax_logits(logits, counts, tau=0.0)

    assert doubled.tolist()[0] == pytest.approx(
        (2.0 * baseline).tolist()[0], rel=1e-5
    )
    assert disabled.tolist()[0] == pytest.approx([0.0, 0.0, 0.0])


@pytest.mark.slow
def test_balanced_softmax_logits_handles_zero_count_classes():
    import torch

    from plant_toxicity_checker.decoupled import balanced_softmax_logits

    logits = torch.zeros((1, 2))
    adjusted = balanced_softmax_logits(logits, [5, 0])
    assert torch.isfinite(adjusted).all()
    # The zero-count class's adjustment is strongly negative (clamped
    # to count 1 before the log, not -inf).
    assert adjusted[0, 1] < adjusted[0, 0]


@pytest.mark.slow
def test_toxicity_cost_term_is_zero_with_no_known_side_samples():
    import torch

    from plant_toxicity_checker.decoupled import toxicity_cost_term

    probas = torch.tensor([[0.5, 0.5]])
    labels = torch.tensor([0])
    sides = [-1, -1]
    term = toxicity_cost_term(probas, labels, sides)
    assert term.item() == pytest.approx(0.0)


@pytest.mark.slow
def test_toxicity_cost_term_rises_with_wrong_side_probability_mass():
    import torch

    from plant_toxicity_checker.decoupled import toxicity_cost_term

    # class 0 = toxic, class 1 = non_toxic; true label is 0 for both
    # samples, but sample B puts far more mass on the wrong side.
    sides = [1, 0]
    labels = torch.tensor([0, 0])

    mostly_right = torch.tensor([[0.9, 0.1], [0.9, 0.1]])
    mostly_wrong = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

    low_term = toxicity_cost_term(mostly_right, labels, sides)
    high_term = toxicity_cost_term(mostly_wrong, labels, sides)
    assert high_term.item() > low_term.item()


@pytest.mark.slow
def test_toxicity_cost_term_ignores_unknown_side_true_labels():
    import torch

    from plant_toxicity_checker.decoupled import toxicity_cost_term

    # class 1's true side is unknown - excluded regardless of where
    # its probability mass lands.
    sides = [1, -1]
    labels = torch.tensor([1])
    probas = torch.tensor([[0.9, 0.1]])
    term = toxicity_cost_term(probas, labels, sides)
    assert term.item() == pytest.approx(0.0)
