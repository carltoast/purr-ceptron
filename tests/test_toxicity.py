"""Tests for toxicity-table loading and validation."""

import json

import pytest

from plant_toxicity_checker.toxicity import ToxicityTableError, load_table


def write_table(tmp_path, *species):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"species": list(species)}))
    return path


def entry(**overrides):
    base = {"species": "rose", "toxic_to_cats": "unknown"}
    base.update(overrides)
    return base


def test_minimal_entry_loads(tmp_path):
    table = load_table(write_table(tmp_path, entry()))
    assert table.get("rose").toxic_to_cats == "unknown"
    assert table.get("tulip") is None


@pytest.mark.parametrize("bad", [
    entry(toxic_to_cats="maybe"),
    entry(severity="deadly"),
    entry(verified=True),
    entry(verified=True, toxic_to_cats="no"),
    entry(verified=True, toxic_to_cats="yes", sources=["x"]),
    # verified must be a real boolean: bool("false") is True
    entry(verified="false"),
    entry(verified="no", toxic_to_cats="no", sources=["x"]),
    entry(verified=1, toxic_to_cats="no", sources=["x"]),
    # contradictory fields
    entry(toxic_to_cats="no", severity="life_threatening"),
    entry(toxic_to_cats="no", symptoms=["vomiting"]),
    entry(toxic_to_cats="yes", severity="none"),
    entry(toxic_to_cats="unknown", severity="mild"),
    # wrong types
    entry(symptoms="vomiting"),
    entry(sources=[1]),
    entry(notes=None),
    entry(scientific_name=5),
])
def test_invalid_entries_are_rejected(tmp_path, bad):
    with pytest.raises(ToxicityTableError):
        load_table(write_table(tmp_path, bad))


def test_missing_required_field_is_rejected(tmp_path):
    with pytest.raises(ToxicityTableError):
        load_table(write_table(tmp_path, {"species": "rose"}))


def test_duplicate_species_is_rejected(tmp_path):
    with pytest.raises(ToxicityTableError):
        load_table(write_table(tmp_path, entry(), entry()))


def test_verified_toxic_with_severity_and_source_is_valid(tmp_path):
    good = entry(
        species="tiger lily", toxic_to_cats="yes", verified=True,
        severity="life_threatening", sources=["https://aspca.org"],
    )
    assert load_table(write_table(tmp_path, good)).get("tiger lily").verified


def test_missing_species_list_is_rejected(tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps({"plants": []}))
    with pytest.raises(ToxicityTableError):
        load_table(path)


def test_invalid_json_is_rejected(tmp_path):
    path = tmp_path / "t.json"
    path.write_text("{not json")
    with pytest.raises(ToxicityTableError):
        load_table(path)


def test_verified_non_toxic_entry_is_valid(tmp_path):
    good = entry(
        toxic_to_cats="no", verified=True, severity="none", sources=["x"]
    )
    assert load_table(write_table(tmp_path, good)).get("rose").verified


# --- valid_species: the table's schema is reused across species spaces ------
# (PlantNet, ...); nothing here hardcodes which one is in use.


def test_without_valid_species_any_name_is_accepted(tmp_path):
    table = load_table(write_table(tmp_path, entry(species="Rosa canina")))
    assert table.get("Rosa canina").toxic_to_cats == "unknown"


def test_valid_species_rejects_a_name_outside_it(tmp_path):
    path = write_table(tmp_path, entry(species="tulip"))
    with pytest.raises(ToxicityTableError):
        load_table(path, valid_species={"rose"})


def test_valid_species_accepts_a_name_inside_it(tmp_path):
    path = write_table(tmp_path, entry())
    table = load_table(path, valid_species={"rose", "tulip"})
    assert table.get("rose") is not None


def test_missing_species_lists_names_not_covered(tmp_path):
    path = write_table(tmp_path, entry(species="rose"))
    table = load_table(path)
    assert table.missing_species(["rose", "tulip"]) == ["tulip"]
