"""PlantNet-300K's canonical species-name list and name<->index mapping.

Mirrors the role the old (removed) ``labels.py`` played for Oxford-102:
model output indices must never be used or compared without going through
this name mapping, since ``species_id`` in PlantNet's own CSVs is a
numeric code, not a stable name, and nothing guarantees a model's output
order matches it (see CLAUDE.md: "everything is keyed by species name,
never class index").

The image-level metadata (``plantnet300K_metadata.csv``) only has
``species_id``, not the species name itself - that join lives in
``species_metadata.csv`` (``species_id`` + ``species``/``full_species``),
so this module carries a ``species_id -> index`` mapping too, built
directly from that same file, rather than assuming ``species_id`` is a
dense 0..999 index.

Canonical name is ``full_species`` (with author citation), not ``species``
(the bare binomial): checked directly against the real data, 2 of
PlantNet-300K's 1000 species_ids collapse to the same bare binomial (e.g.
"Ophrys sphegodes" covers 6 distinct species_ids - different
infraspecific taxa the ``species`` column doesn't distinguish) while
``full_species`` is confirmed unique across all 1000 rows. Using the
non-unique column as a class key would silently merge distinct classes.
"""

import csv

from plant_toxicity_checker.config import PLANTNET_DIR

SPECIES_METADATA_PATH = PLANTNET_DIR / "species_metadata.csv"
NAME_FIELD = "full_species"


def load_species_rows(path=None):
    """All rows of ``species_metadata.csv`` as a list of dicts, file
    order preserved."""
    path = path or SPECIES_METADATA_PATH
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_species(path=None):
    """Ordered, unique species names (``full_species``) from
    ``species_metadata.csv``.

    File order becomes the canonical class-index order for anything
    trained against it.
    """
    return tuple(row[NAME_FIELD] for row in load_species_rows(path))


class SpeciesLabels:
    """Name/``species_id`` <-> class-index mapping, built once from
    ``species_metadata.csv``'s rows (file order = canonical index
    order)."""

    def __init__(self, rows):
        self.names = tuple(row[NAME_FIELD] for row in rows)
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"{NAME_FIELD!r} values must be unique")
        self._index_of_name = {
            name: i for i, name in enumerate(self.names)
        }
        self._index_of_species_id = {
            row["species_id"]: i for i, row in enumerate(rows)
        }

    @classmethod
    def from_metadata(cls, path=None):
        """Build from ``species_metadata.csv`` (or ``path`` if given)."""
        return cls(load_species_rows(path))

    def __len__(self):
        return len(self.names)

    def index_of(self, name):
        """Class index for a species ``name``; raises ``KeyError`` if
        unknown."""
        return self._index_of_name[name]

    def index_of_species_id(self, species_id):
        """Class index for a raw ``species_id`` (e.g. from
        ``plantnet300K_metadata.csv``'s per-image rows, as a string or
        int); raises ``KeyError`` if unknown.
        """
        return self._index_of_species_id[str(species_id)]

    def name_of(self, index):
        """Species name for ``index``; raises ``IndexError`` if out of
        range."""
        return self.names[index]
