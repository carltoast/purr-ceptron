"""Tests for seeding data/toxicity.json from the ASPCA scrape."""

import json

import pytest

from plant_toxicity_checker.toxicity import load_table
from plant_toxicity_checker.toxicity_seed import (
    SeedError,
    seed_entries,
    write_seed_table,
)


def plant(name, scientific_name, toxic_to=(), non_toxic_to=(), **overrides):
    base = {
        "name": name,
        "url": f"https://example.org/{name.lower().replace(' ', '-')}",
        "scientific_name": scientific_name,
        "toxic_to": list(toxic_to),
        "non_toxic_to": list(non_toxic_to),
        "toxic_principles": "",
        "clinical_signs": "",
    }
    base.update(overrides)
    return base


def test_seed_entries_maps_toxic_yes():
    entries = seed_entries(
        [plant("Lily", "Lilium longiflorum", toxic_to=["cats"])]
    )
    assert len(entries) == 1
    assert entries[0]["toxic_to_cats"] == "yes"
    assert entries[0]["species"] == "Lilium longiflorum"
    assert entries[0]["verified"] is False


def test_seed_entries_maps_toxic_no():
    entries = seed_entries(
        [plant("Rose", "Rosa canina", non_toxic_to=["cats"])]
    )
    assert entries[0]["toxic_to_cats"] == "no"


def test_seed_entries_maps_unknown_when_cats_not_addressed():
    entries = seed_entries(
        [plant("Foxglove", "Digitalis purpurea", toxic_to=["dogs"])]
    )
    assert entries[0]["toxic_to_cats"] == "unknown"


def test_seed_entries_every_entry_is_unverified():
    entries = seed_entries([
        plant("A", "Rosa canina", toxic_to=["cats"]),
        plant("B", "Digitalis purpurea", non_toxic_to=["cats"]),
    ])
    assert all(entry["verified"] is False for entry in entries)


def test_seed_entries_skips_plants_with_no_scientific_name():
    entries = seed_entries([plant("Unknown plant", "")])
    assert entries == []


def test_seed_entries_groups_agreeing_pages_by_scientific_name():
    entries = seed_entries([
        plant("Foxtail Fern cv A", "Asparagus densiflorus",
              toxic_to=["cats"]),
        plant("Foxtail Fern cv B", "Asparagus densiflorus",
              toxic_to=["cats"]),
    ])
    assert len(entries) == 1
    assert entries[0]["toxic_to_cats"] == "yes"
    assert len(entries[0]["sources"]) == 2
    assert "Foxtail Fern cv A" in entries[0]["notes"]
    assert "Foxtail Fern cv B" in entries[0]["notes"]


def test_seed_entries_raises_on_disagreeing_pages():
    with pytest.raises(SeedError):
        seed_entries([
            plant("A", "Rosa canina", toxic_to=["cats"]),
            plant("B", "Rosa canina", non_toxic_to=["cats"]),
        ])


def test_seed_entries_sorted_by_species():
    entries = seed_entries([
        plant("Z", "Zinnia elegans", toxic_to=["cats"]),
        plant("A", "Abies alba", non_toxic_to=["cats"]),
    ])
    assert [e["species"] for e in entries] == [
        "Abies alba", "Zinnia elegans",
    ]


def test_seed_entries_have_no_severity_or_symptoms():
    # Deliberately left for a human to fill in on verification - never
    # transcribed automatically from clinical_signs/toxic_principles.
    entries = seed_entries([
        plant("A", "Rosa canina", toxic_to=["cats"],
              toxic_principles="x", clinical_signs="y"),
    ])
    assert entries[0]["severity"] is None
    assert entries[0]["symptoms"] == []


def test_write_seed_table_produces_a_table_loadable_by_load_table(
    tmp_path,
):
    out_path = tmp_path / "toxicity.json"
    write_seed_table(
        [
            plant("A", "Rosa canina", toxic_to=["cats"]),
            plant("B", "Digitalis purpurea", non_toxic_to=["cats"]),
        ],
        out_path,
    )
    table = load_table(out_path)
    assert len(table) == 2
    assert table.get("Rosa canina").toxic_to_cats == "yes"
    assert table.get("Rosa canina").verified is False


def test_write_seed_table_writes_valid_json(tmp_path):
    out_path = tmp_path / "toxicity.json"
    write_seed_table([plant("A", "Rosa canina", toxic_to=["cats"])], out_path)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "species" in data
