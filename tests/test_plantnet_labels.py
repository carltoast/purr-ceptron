"""Tests for PlantNet-300K's canonical species-name list/order and the
species_id <-> name <-> index mapping."""

import csv

import pytest

from plant_toxicity_checker.plantnet_labels import SpeciesLabels, load_species


def write_species_csv(tmp_path, *rows):
    """``rows`` is (species_id, full_species) pairs. ``species`` (the
    bare binomial) is derived by dropping any author citation, so a
    fixture can deliberately reuse the same bare binomial for two
    species_ids, mirroring PlantNet-300K's real data (e.g. "Ophrys
    sphegodes" covers 6 distinct species_ids)."""
    path = tmp_path / "species_metadata.csv"
    fieldnames = ["species_id", "full_species", "species", "genus"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for species_id, full_species in rows:
            bare = " ".join(full_species.split()[:2])
            writer.writerow({
                "species_id": species_id,
                "full_species": full_species,
                "species": bare,
                "genus": bare.split()[0],
            })
    return path


def test_load_species_returns_file_order(tmp_path):
    path = write_species_csv(
        tmp_path, ("0", "Rosa canina L."), ("1", "Lilium bulbiferum L."),
    )
    assert load_species(path) == ("Rosa canina L.", "Lilium bulbiferum L.")


def test_species_labels_index_and_name_round_trip(tmp_path):
    path = write_species_csv(
        tmp_path, ("0", "Rosa canina L."), ("1", "Lilium bulbiferum L."),
    )
    labels = SpeciesLabels.from_metadata(path)
    assert len(labels) == 2
    assert labels.index_of("Rosa canina L.") == 0
    assert labels.index_of("Lilium bulbiferum L.") == 1
    assert labels.name_of(0) == "Rosa canina L."
    assert labels.name_of(1) == "Lilium bulbiferum L."


def test_index_of_species_id_does_not_assume_dense_ids(tmp_path):
    # species_id values are sparse/non-contiguous on purpose here, to
    # make sure nothing assumes species_id == class index.
    path = write_species_csv(
        tmp_path, ("42", "Rosa canina L."), ("7", "Lilium bulbiferum L."),
    )
    labels = SpeciesLabels.from_metadata(path)
    assert labels.index_of_species_id("42") == 0
    assert labels.index_of_species_id("7") == 1
    # int input works too - image metadata rows may hand this back as
    # either a string or an int depending on how they were parsed.
    assert labels.index_of_species_id(7) == 1


def test_species_id_of_is_the_inverse_of_index_of_species_id(tmp_path):
    path = write_species_csv(
        tmp_path, ("42", "Rosa canina L."), ("7", "Lilium bulbiferum L."),
    )
    labels = SpeciesLabels.from_metadata(path)
    assert labels.species_id_of(0) == "42"
    assert labels.species_id_of(1) == "7"


def test_species_id_of_out_of_range_raises_index_error(tmp_path):
    path = write_species_csv(tmp_path, ("0", "Rosa canina L."))
    labels = SpeciesLabels.from_metadata(path)
    with pytest.raises(IndexError):
        labels.species_id_of(5)


def test_index_of_unknown_name_raises_key_error(tmp_path):
    path = write_species_csv(tmp_path, ("0", "Rosa canina L."))
    labels = SpeciesLabels.from_metadata(path)
    with pytest.raises(KeyError):
        labels.index_of("Tulipa gesneriana L.")


def test_index_of_unknown_species_id_raises_key_error(tmp_path):
    path = write_species_csv(tmp_path, ("0", "Rosa canina L."))
    labels = SpeciesLabels.from_metadata(path)
    with pytest.raises(KeyError):
        labels.index_of_species_id("999")


def test_name_of_out_of_range_raises_index_error(tmp_path):
    path = write_species_csv(tmp_path, ("0", "Rosa canina L."))
    labels = SpeciesLabels.from_metadata(path)
    with pytest.raises(IndexError):
        labels.name_of(5)


def test_duplicate_full_species_values_are_rejected(tmp_path):
    path = write_species_csv(
        tmp_path, ("0", "Rosa canina L."), ("1", "Rosa canina L."),
    )
    with pytest.raises(ValueError):
        SpeciesLabels.from_metadata(path)


def test_species_ids_sharing_a_bare_binomial_get_distinct_classes(tmp_path):
    # Regression: real PlantNet-300K data has 2 bare binomials
    # ("Ophrys fusca", "Ophrys sphegodes") each covering multiple
    # species_ids with different author citations - full_species (used
    # as the class key) disambiguates them; "species" (bare binomial)
    # would not.
    path = write_species_csv(
        tmp_path,
        ("163", "Ophrys fusca Link"),
        ("454", "Ophrys fusca (Viv.) Arcang."),
    )
    labels = SpeciesLabels.from_metadata(path)
    assert len(labels) == 2
    assert labels.index_of_species_id("163") != labels.index_of_species_id(
        "454"
    )
