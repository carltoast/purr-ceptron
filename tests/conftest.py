"""Shared fixtures: small in-memory toxicity tables and a fake classifier."""

import os

import pytest

# Plots in tests must never need a display or a Tcl/Tk install.
os.environ.setdefault("MPLBACKEND", "Agg")

from plant_toxicity_checker.toxicity import ToxicityEntry, ToxicityTable
from plant_toxicity_checker.verdict import Prediction


def make_entry(species, toxic="unknown", verified=False, severity=None):
    return ToxicityEntry(
        species=species,
        scientific_name=None,
        toxic_to_cats=toxic,
        severity=severity,
        symptoms=("vomiting",) if toxic == "yes" else (),
        sources=("aspca",) if verified else (),
        verified=verified,
        notes="",
    )


@pytest.fixture
def table():
    # Two-word, genus+epithet-shaped names on purpose: matching is by
    # parsed scientific name (species_match.py), and a bare single
    # word parses as a genus with no epithet - i.e. a genus
    # PLACEHOLDER, not a specific species. A single-word name here
    # would self-match as its own genus-level entry instead of an
    # exact one, which isn't what these fixtures are meant to test.
    return ToxicityTable([
        make_entry("Lilium tigrinum", "yes", True, "life_threatening"),
        make_entry("Rosa rugosa", "no", True, "none"),
        make_entry("Helianthus annuus", "no", True, "none"),
        make_entry("Narcissus pseudonarcissus", "yes", False),
        make_entry("Petunia hybrida", "no", False),
        make_entry("Dianthus caryophyllus", "unknown"),
    ])


class FakeClassifier:
    def __init__(self, predictions):
        self._predictions = [Prediction(s, c) for s, c in predictions]

    def predict(self, image, top_k=None):
        return self._predictions[:top_k]
