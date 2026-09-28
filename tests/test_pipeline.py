"""Tests for the pipeline (image -> species -> verdict) with a fake
classifier, and for the CLI's verdict-formatting helpers.

These are CLI-command-independent on purpose: ``check_image``/``load_image``
and ``verdict_to_dict``/``format_verdict`` are already species-space-
agnostic and will back whichever species-ID command (Oxford-102's removed
``check``, or its PlantNet-trained replacement) sits in front of them next.
"""

import json

import pytest
from conftest import FakeClassifier
from PIL import Image

from bouquet_checker import cli
from bouquet_checker.pipeline import ImageLoadError, check_image, load_image
from bouquet_checker.verdict import SCOPE_NOTE, Status


def test_check_image_accepts_pil_image(table):
    classifier = FakeClassifier([("tiger lily", 0.9)])
    verdict = check_image(Image.new("RGB", (8, 8)), classifier, table)
    assert verdict.status is Status.TOXIC


def test_check_image_accepts_path_and_converts_to_rgb(tmp_path, table):
    path = tmp_path / "flower.png"
    Image.new("RGBA", (8, 8)).save(path)
    assert load_image(path).mode == "RGB"
    classifier = FakeClassifier([("rose", 0.99), ("sunflower", 0.01)])
    assert check_image(path, classifier, table).status is Status.LIKELY_SAFE


def test_check_image_always_uses_the_full_distribution(table):
    # Regression: a toxic alternative must never be dropped just because
    # the caller wanted fewer candidates displayed.
    seen = []

    class Recording(FakeClassifier):
        def predict(self, image, top_k=None):
            seen.append(top_k)
            return super().predict(image, top_k)

    classifier = Recording([("rose", 0.85), ("tiger lily", 0.12)])
    verdict = check_image(Image.new("RGB", (8, 8)), classifier, table)
    assert seen == [None]
    assert verdict.status is Status.CAUTION


def test_unreadable_file_raises_image_load_error(tmp_path):
    junk = tmp_path / "notes.txt"
    junk.write_text("not an image")
    with pytest.raises(ImageLoadError):
        load_image(junk)


def test_missing_file_raises_image_load_error(tmp_path):
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "missing.jpg")


def test_heic_photos_can_be_loaded(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    path = tmp_path / "photo.heic"
    pillow_heif.from_pillow(Image.new("RGB", (16, 16), "red")).save(path)
    assert load_image(path).size == (16, 16)


def test_verdict_serialises_to_json(table):
    classifier = FakeClassifier([("tiger lily", 0.9), ("rose", 0.05)])
    verdict = check_image(Image.new("RGB", (8, 8)), classifier, table)
    data = json.loads(json.dumps(cli.verdict_to_dict(verdict)))
    assert data["status"] == "toxic"
    assert data["toxicity"]["severity"] == "life_threatening"
    assert data["scope_note"] == SCOPE_NOTE


def test_top_k_only_trims_the_displayed_predictions(table):
    classifier = FakeClassifier(
        [("rose", 0.6), ("sunflower", 0.3), ("tiger lily", 0.1)]
    )
    verdict = check_image(Image.new("RGB", (8, 8)), classifier, table)
    data = cli.verdict_to_dict(verdict, top_k=1)
    assert len(data["predictions"]) == 1
    shown = cli.format_verdict(verdict, top_k=2)
    assert "10.0%  tiger lily" not in shown  # trimmed from the list...
    assert "alternative 'tiger lily'" in shown  # ...but still explained
    assert data["status"] == verdict.status.value


def test_format_verdict_says_only_main_flower_was_assessed(table):
    classifier = FakeClassifier([("rose", 0.99), ("sunflower", 0.01)])
    verdict = check_image(Image.new("RGB", (8, 8)), classifier, table)
    text = cli.format_verdict(verdict)
    assert "LIKELY SAFE for the main flower only" in text
    assert "rose" in text
    assert SCOPE_NOTE in text
