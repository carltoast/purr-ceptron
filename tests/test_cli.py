"""Tests for cli.py's `check` subcommand wiring.

Stubs the real classifier/table loading (both lazily imported inside
`_run_check`, so monkeypatching the source modules' attributes is what
actually takes effect) so this never needs real checkpoints or the
real toxicity table.
"""

import json

from conftest import FakeClassifier
from PIL import Image

from plant_toxicity_checker import cli


def test_check_parses_arguments_with_defaults():
    args = cli.build_parser().parse_args(["check", "--image", "photo.jpg"])
    assert args.image == "photo.jpg"
    assert args.run  # the shipped end-to-end model, not a backbone+head
    assert args.head_run is None
    assert args.epoch is None
    assert args.ema is False
    assert args.top_k is None
    assert args.json is False


def _stub_classifier_and_table(monkeypatch, table, predictions):
    import plant_toxicity_checker.plantnet_classifier as classifier_module
    import plant_toxicity_checker.toxicity as toxicity_module

    monkeypatch.setattr(
        classifier_module, "PlantNetClassifier",
        lambda run, **kwargs: FakeClassifier(predictions),
    )
    monkeypatch.setattr(toxicity_module, "load_table", lambda path: table)


def test_check_runs_the_pipeline_and_prints_a_text_report(
    tmp_path, monkeypatch, table, capsys,
):
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8)).save(image_path)
    _stub_classifier_and_table(
        monkeypatch, table, [("Lilium tigrinum", 0.95)],
    )

    args = cli.build_parser().parse_args(
        ["check", "--image", str(image_path)]
    )
    assert args.func(args) == 0

    out = capsys.readouterr().out
    assert "TOXIC" in out
    assert "Lilium tigrinum" in out


def test_check_json_flag_prints_machine_readable_output(
    tmp_path, monkeypatch, table, capsys,
):
    image_path = tmp_path / "photo.jpg"
    Image.new("RGB", (8, 8)).save(image_path)
    _stub_classifier_and_table(
        monkeypatch, table,
        [("Rosa rugosa", 0.99), ("Helianthus annuus", 0.01)],
    )

    args = cli.build_parser().parse_args(
        ["check", "--image", str(image_path), "--json"]
    )
    assert args.func(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["species"] == "Rosa rugosa"
    assert payload["status"] == "likely_safe"


def test_check_reports_a_bad_image_path_as_a_clean_error(
    tmp_path, monkeypatch, table,
):
    _stub_classifier_and_table(monkeypatch, table, [("Rosa rugosa", 0.99)])

    missing_path = str(tmp_path / "missing.jpg")
    assert cli.main(["check", "--image", missing_path]) == 1
