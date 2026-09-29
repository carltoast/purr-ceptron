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
    return ToxicityTable([
        make_entry("tiger lily", "yes", True, "life_threatening"),
        make_entry("rose", "no", True, "none"),
        make_entry("sunflower", "no", True, "none"),
        make_entry("daffodil", "yes", False),
        make_entry("petunia", "no", False),
        make_entry("carnation", "unknown"),
    ])


class FakeClassifier:
    def __init__(self, predictions):
        self._predictions = [Prediction(s, c) for s, c in predictions]

    def predict(self, image, top_k=None):
        return self._predictions[:top_k]
