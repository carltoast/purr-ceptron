"""Tests for PlantNetClassifier - torch-touching, so ``@pytest.mark.slow``.

Uses a tiny fake backbone/head pair (never the real ~93MB checkpoint or
the real 1000-species metadata file) so this runs fast and offline.
"""

import csv

import pytest


def _write_fake_species_csv(path, names):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["species_id", "full_species"])
        writer.writeheader()
        for i, name in enumerate(names):
            writer.writerow({"species_id": str(i), "full_species": name})


@pytest.fixture
def fake_checkpoints(tmp_path, monkeypatch):
    """Writes a tiny fake backbone ``latest.pt`` + classifier head
    ``head.pt`` under a monkeypatched ``CHECKPOINTS_DIR``, and a tiny
    fake species CSV under a monkeypatched ``SPECIES_METADATA_PATH`` -
    everything ``load_combined_model`` needs, none of it real.
    """
    import torch
    from torch import nn

    from plant_toxicity_checker import plantnet_labels, train

    monkeypatch.setattr(train, "CHECKPOINTS_DIR", tmp_path)

    species_csv = tmp_path / "species_metadata.csv"
    names = ["Fake species A", "Fake species B", "Fake species C"]
    _write_fake_species_csv(species_csv, names)
    monkeypatch.setattr(plantnet_labels, "SPECIES_METADATA_PATH", species_csv)

    class TinyBackbone(nn.Module):
        """Stands in for ``train.build_model``'s real ResNet-18 -
        adaptive pooling handles any input size, so this never needs
        to match ``build_transforms``' exact crop dimensions."""

        def __init__(self, num_classes, model="resnet18", pretrained=True):
            # Mirrors build_model's signature, including the arguments
            # it ignores here, so a caller passing them still works.
            super().__init__()
            self.squeeze = nn.Conv2d(3, 4, kernel_size=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(4, num_classes)

        def forward(self, x):
            x = self.pool(self.squeeze(x)).flatten(1)
            return self.fc(x)

    monkeypatch.setattr(train, "build_model", TinyBackbone)

    backbone = TinyBackbone(num_classes=3)
    backbone_dir = tmp_path / "fake-backbone"
    backbone_dir.mkdir()
    torch.save(
        {"model_state_dict": backbone.state_dict()},
        backbone_dir / "latest.pt",
    )

    head = nn.Linear(4, 3)
    head_dir = tmp_path / "fake-head"
    head_dir.mkdir()
    torch.save(
        {
            "state_dict": head.state_dict(),
            "backbone_run": "fake-backbone",
            "config": {"loss": "ce", "epochs": 1},
        },
        head_dir / "head.pt",
    )

    return names


@pytest.mark.slow
def test_predict_returns_one_prediction_per_class_when_top_k_is_none(
    fake_checkpoints,
):
    from PIL import Image

    from plant_toxicity_checker.plantnet_classifier import PlantNetClassifier

    classifier = PlantNetClassifier("fake-backbone", "fake-head")
    predictions = classifier.predict(Image.new("RGB", (64, 64)), top_k=None)

    assert len(predictions) == 3
    assert {p.species for p in predictions} == set(fake_checkpoints)


@pytest.mark.slow
def test_predict_confidences_sum_to_one(fake_checkpoints):
    from PIL import Image

    from plant_toxicity_checker.plantnet_classifier import PlantNetClassifier

    classifier = PlantNetClassifier("fake-backbone", "fake-head")
    predictions = classifier.predict(Image.new("RGB", (64, 64)))

    assert sum(p.confidence for p in predictions) == pytest.approx(1.0)


@pytest.mark.slow
def test_predict_top_k_slices_to_the_highest_confidence_predictions(
    fake_checkpoints,
):
    from PIL import Image

    from plant_toxicity_checker.plantnet_classifier import PlantNetClassifier

    classifier = PlantNetClassifier("fake-backbone", "fake-head")
    full = classifier.predict(Image.new("RGB", (64, 64)), top_k=None)
    top_2 = classifier.predict(Image.new("RGB", (64, 64)), top_k=2)

    assert len(top_2) == 2
    assert top_2 == full[:2]
    assert list(top_2) == sorted(
        top_2, key=lambda p: p.confidence, reverse=True,
    )


@pytest.mark.slow
def test_load_combined_model_rejects_mismatched_backbone_run(
    fake_checkpoints, tmp_path,
):
    import torch

    from plant_toxicity_checker import decoupled, train

    # A second backbone checkpoint under a different run name, so the
    # head (trained against "fake-backbone") is passed a mismatched one.
    other_dir = tmp_path / "other-backbone"
    other_dir.mkdir()
    backbone = train.build_model(3)  # the monkeypatched TinyBackbone
    torch.save(
        {"model_state_dict": backbone.state_dict()}, other_dir / "latest.pt",
    )

    with pytest.raises(ValueError, match="other-backbone"):
        decoupled.load_combined_model("other-backbone", "fake-head")
