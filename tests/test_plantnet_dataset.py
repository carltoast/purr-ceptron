"""Tests for the pure, torch-free pieces of the PlantNet-300K dataset
loading: split filtering and image-path construction. No torch, no PIL,
no real images."""

from pathlib import Path

from plant_toxicity_checker.plantnet_dataset import (
    cap_images_per_species,
    filter_species_subset,
    image_path,
    rows_for_split,
    species_dir,
)


def rows():
    return [
        {"species_id": "7", "split": "train", "PN_hash": "abc"},
        {"species_id": "12", "split": "val", "PN_hash": "def"},
        {"species_id": "999", "split": "train", "PN_hash": "ghi"},
    ]


def multi_image_rows():
    return [
        {"species_id": "7", "split": "train", "PN_hash": "a"},
        {"species_id": "7", "split": "train", "PN_hash": "b"},
        {"species_id": "7", "split": "train", "PN_hash": "c"},
        {"species_id": "12", "split": "train", "PN_hash": "d"},
        {"species_id": "12", "split": "train", "PN_hash": "e"},
    ]


def test_rows_for_split_filters_correctly():
    train_rows = rows_for_split(rows(), "train")
    assert [r["PN_hash"] for r in train_rows] == ["abc", "ghi"]
    assert rows_for_split(rows(), "test") == []


def test_species_dir_zero_pads_to_four_digits():
    assert species_dir("7") == "0007"
    assert species_dir(7) == "0007"
    assert species_dir("999") == "0999"
    assert species_dir("0") == "0000"


def test_image_path_joins_split_species_dir_and_hash():
    row = {"species_id": "7", "split": "train", "PN_hash": "abc123"}
    path = image_path(row, images_dir=Path("/data/images"))
    assert path == Path("/data/images/train/0007/abc123.jpg")


def test_filter_species_subset_keeps_lowest_ids():
    kept = filter_species_subset(rows(), max_species=2)
    assert {r["species_id"] for r in kept} == {"7", "12"}


def test_filter_species_subset_none_is_a_no_op():
    assert filter_species_subset(rows(), max_species=None) == rows()


def test_filter_species_subset_is_deterministic_regardless_of_order():
    shuffled = list(reversed(rows()))
    kept_a = filter_species_subset(rows(), max_species=2)
    kept_b = filter_species_subset(shuffled, max_species=2)
    assert {r["species_id"] for r in kept_a} == {
        r["species_id"] for r in kept_b
    }


def test_cap_images_per_species_keeps_first_n_per_species():
    kept = cap_images_per_species(
        multi_image_rows(), max_images_per_species=2
    )
    assert [r["PN_hash"] for r in kept] == ["a", "b", "d", "e"]


def test_cap_images_per_species_none_is_a_no_op():
    assert cap_images_per_species(
        multi_image_rows(), max_images_per_species=None
    ) == multi_image_rows()


def test_cap_images_per_species_leaves_species_under_the_cap_alone():
    kept = cap_images_per_species(
        multi_image_rows(), max_images_per_species=10
    )
    assert kept == multi_image_rows()
