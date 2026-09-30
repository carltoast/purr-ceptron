"""Tests for the pure, torch-free pieces of the PlantNet-300K dataset
loading: split filtering and image-path construction. No torch, no PIL,
no real images."""

from pathlib import Path

from plant_toxicity_checker.plantnet_dataset import (
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
