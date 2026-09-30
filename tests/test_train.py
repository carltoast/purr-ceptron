"""Tests for train.py's composable pieces.

``load_config`` is plain JSON parsing - no torch, runs in the default
suite. Everything else here touches torch (even just to build a tiny
dummy model/optimizer, no GPU or real data needed) and is marked
``slow``, matching this project's convention that the default ``pytest``
run never requires torch to be installed.
"""

import json

import pytest

from plant_toxicity_checker.train import ConfigError, load_config

BASE_CONFIG = {
    "lr": 0.01,
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 0.0001,
    "batch_size": 32,
    "epochs": 30,
    "lr_decay_epochs": [20, 25],
    "lr_decay_factor": 0.1,
}


def write_config(tmp_path, **overrides):
    data = {**BASE_CONFIG, **overrides}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_config_reads_a_valid_file(tmp_path):
    config = load_config(write_config(tmp_path))
    assert config["lr"] == 0.01
    assert config["epochs"] == 30


def test_load_config_rejects_missing_keys(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"lr": 0.01}), encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(path)


@pytest.mark.slow
def test_checkpoint_round_trips_model_optimizer_and_scheduler(
    tmp_path, monkeypatch,
):
    import torch
    from torch import nn

    from plant_toxicity_checker import train

    monkeypatch.setattr(train, "CHECKPOINTS_DIR", tmp_path)

    model = nn.Linear(4, 2)
    config = dict(BASE_CONFIG)
    optimizer = train.build_optimizer(model, config)
    scheduler = train.build_scheduler(optimizer, config)

    train.save_checkpoint(
        "run1", epoch=3, model=model, optimizer=optimizer,
        scheduler=scheduler, config=config,
    )

    other_model = nn.Linear(4, 2)
    other_optimizer = train.build_optimizer(other_model, config)
    other_scheduler = train.build_scheduler(other_optimizer, config)
    finished_epoch = train.load_checkpoint(
        "run1", other_model, other_optimizer, other_scheduler,
    )

    assert finished_epoch == 3
    for p1, p2 in zip(model.parameters(), other_model.parameters()):
        assert torch.equal(p1, p2)


@pytest.mark.slow
def test_build_optimizer_uses_the_configs_hyperparameters():
    from torch import nn

    from plant_toxicity_checker import train

    model = nn.Linear(4, 2)
    optimizer = train.build_optimizer(model, BASE_CONFIG)
    group = optimizer.param_groups[0]
    assert group["lr"] == BASE_CONFIG["lr"]
    assert group["momentum"] == BASE_CONFIG["momentum"]
    assert group["nesterov"] == BASE_CONFIG["nesterov"]
    assert group["weight_decay"] == BASE_CONFIG["weight_decay"]


@pytest.mark.slow
def test_build_scheduler_uses_the_configs_milestones():
    from torch import nn

    from plant_toxicity_checker import train

    model = nn.Linear(4, 2)
    optimizer = train.build_optimizer(model, BASE_CONFIG)
    scheduler = train.build_scheduler(optimizer, BASE_CONFIG)
    assert sorted(scheduler.milestones.elements()) == BASE_CONFIG[
        "lr_decay_epochs"
    ]
    assert scheduler.gamma == BASE_CONFIG["lr_decay_factor"]


@pytest.mark.slow
def test_make_loader_uses_the_standard_performance_knobs():
    from plant_toxicity_checker import train

    loader = train.make_loader(list(range(4)), batch_size=2, shuffle=False)
    assert loader.num_workers == train.NUM_WORKERS
    assert loader.persistent_workers is True
    assert loader.pin_memory is True
    assert loader.batch_size == 2
