"""Match species across datasets by scientific name - never by common name.

Two levels only, the same distinction used throughout this project's ASPCA
work: an **exact** match (same genus and specific epithet) or a **genus**
match (same genus, different or unknown species). A common name is never
used to decide identity - the "lily problem": *Lilium*, *Hemerocallis*,
*Hippeastrum* and *Spathiphyllum* are all called "lily" in English and are
four different genera with different toxicity.

``SpeciesIndex`` is symmetric: build one from ASPCA's entries and match
PlantNet/PlantCLEF/iNaturalist species against it (toxicity lookup for a
classified species), or build one from a photo dataset's species table and
match ASPCA entries against it (dataset-coverage analysis, as used here) -
same code either way.
"""

import collections
import re
from dataclasses import dataclass

_NOT_AN_EPITHET = {"spp", "sp", "cv", "x", "var", "ssp"}


def parse_scientific_name(text):
    """(genus, epithet) in lower case from a scientific name.

    ``epithet`` is ``None`` for a genus-only name ("Rosa spp.") or when the
    second word looks like a qualifier ("cv", "var", ...) rather than a
    species epithet. Only the text before the first comma is used - ASPCA
    entries occasionally list alternate/related names after a comma, and
    the first is its primary one. Returns ``None`` if nothing genus-like is
    found at all (an empty or unusable string).
    """
    if not text:
        return None
    words = re.findall(r"[A-Za-z\-]+", text.split(",")[0])
    if not words:
        return None
    genus = words[0].lower()
    if len(words) < 2 or words[1].lower() in _NOT_AN_EPITHET:
        return (genus, None)
    return (genus, words[1].lower())


@dataclass(frozen=True)
class MatchResult:
    """The outcome of matching one scientific name against a
    ``SpeciesIndex``."""

    status: str              # "exact" / "genus_only" / "none"
    exact_row: object = None
    genus_candidates: tuple = ()


class SpeciesIndex:
    """A dataset's species, keyed for matching by (genus, epithet) and by
    genus alone. Build with ``from_rows``, then call ``match()``.
    """

    def __init__(self):
        self.by_binomial = {}
        self.by_genus = collections.defaultdict(list)

    @classmethod
    def from_rows(cls, rows, genus_of, epithet_of):
        """Build an index from any iterable of rows (dicts, namedtuples, ...).

        ``genus_of`` / ``epithet_of`` extract the genus / specific epithet
        from one row (already lower-cased is fine; this lower-cases again
        defensively). A row with no genus is skipped.
        """
        index = cls()
        for row in rows:
            genus = genus_of(row)
            if not genus:
                continue
            genus = genus.strip().lower()
            index.by_genus[genus].append(row)
            epithet = epithet_of(row)
            if epithet:
                index.by_binomial[(genus, epithet.strip().lower())] = row
        return index

    def match(self, scientific_name):
        """``MatchResult`` for one scientific name against this index."""
        parsed = parse_scientific_name(scientific_name)
        if parsed is None:
            return MatchResult("none")
        genus, epithet = parsed
        if epithet and (genus, epithet) in self.by_binomial:
            return MatchResult(
                "exact", self.by_binomial[(genus, epithet)],
                tuple(self.by_genus[genus]),
            )
        if genus in self.by_genus:
            return MatchResult(
                "genus_only", None, tuple(self.by_genus[genus])
            )
        return MatchResult("none")


def aspca_matched_species_ids(plantnet_rows, aspca_plants):
    """``species_id`` values (as ints) of ``plantnet_rows`` whose
    ``genus``/``epithet`` columns (``species_metadata.csv``) are an
    EXACT match against an ASPCA-scraped plant's own ``scientific_name``
    (the ``"plants"`` array of ``cats_plant_details.json``).

    Name matching only - this says "ASPCA has a page for this species",
    nothing about toxicity, and never infers anything about a species
    from a genus-only match. Used so a training-subset selection (see
    ``plantnet_dataset.filter_species_subset``'s ``priority_species_ids``)
    can keep the species this project actually cares about identifying,
    instead of an arbitrary cut that might exclude all of them.
    """
    aspca_rows = []
    for plant in aspca_plants:
        parsed = parse_scientific_name(plant.get("scientific_name"))
        if parsed is None:
            continue
        genus, epithet = parsed
        aspca_rows.append({"genus": genus, "epithet": epithet})
    index = SpeciesIndex.from_rows(
        aspca_rows,
        genus_of=lambda row: row["genus"],
        epithet_of=lambda row: row["epithet"],
    )

    matched_ids = set()
    for row in plantnet_rows:
        genus = row.get("genus")
        if not genus:
            continue
        epithet = row.get("epithet") or ""
        result = index.match(f"{genus} {epithet}".strip())
        if result.status == "exact":
            matched_ids.add(int(row["species_id"]))
    return matched_ids
