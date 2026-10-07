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


def match_against_toxicity_table(plantnet_rows, toxicity_entries):
    """Match every row in ``plantnet_rows`` against the toxicity table,
    by genus+epithet - the one place this project decides how PlantNet
    species relate to it (never by common name, per this module's
    "lily problem").

    ``plantnet_rows`` is ``species_metadata.csv``-shaped (needs
    ``species_id``, ``genus``, ``epithet``). ``toxicity_entries`` is
    any iterable of objects with a ``.species`` attribute holding a
    scientific name - meant to be ``toxicity.load_table(...)``'s
    ``ToxicityTable`` (or its entries directly), i.e. the real, hand
    -curated ``data/toxicity.json`` - not the raw ASPCA scrape. Once
    that table exists, it's the only thing PlantNet species get matched
    against, so there is exactly one place this project decides "how do
    these two datasets relate", not a second path that re-reads the
    scrape with its own matching logic.

    Returns ``{species_id (int): MatchResult}``, with ``exact_row``/
    ``genus_candidates`` holding the matched ``toxicity_entries``
    objects directly - callers read ``.toxic_to_cats``/``.verified``
    straight off them for whatever view they need (dev-subset priority,
    category stats, which entries to verify first, ...), rather than
    this function picking one particular view itself.
    """
    wrapped = []
    for entry in toxicity_entries:
        parsed = parse_scientific_name(entry.species)
        if parsed is None:
            continue
        genus, epithet = parsed
        wrapped.append({"genus": genus, "epithet": epithet, "entry": entry})
    index = SpeciesIndex.from_rows(
        wrapped,
        genus_of=lambda row: row["genus"],
        epithet_of=lambda row: row["epithet"],
    )

    matches = {}
    for row in plantnet_rows:
        species_id = int(row["species_id"])
        genus = row.get("genus")
        if not genus:
            matches[species_id] = MatchResult("none")
            continue
        epithet = row.get("epithet") or ""
        result = index.match(f"{genus} {epithet}".strip())
        exact_entry = (
            result.exact_row["entry"] if result.exact_row else None
        )
        genus_entries = tuple(c["entry"] for c in result.genus_candidates)
        matches[species_id] = MatchResult(
            result.status, exact_entry, genus_entries
        )
    return matches


_CATS_LABEL = {"yes": "toxic", "no": "non-toxic", "unknown": "unknown"}


def toxicity_category(result):
    """A human-readable category label for one ``MatchResult`` from
    ``match_against_toxicity_table``: ``"exact - toxic"``/``"exact -
    non-toxic"``/``"exact - unknown"``, ``"genus - toxic"``/``"genus -
    non-toxic"``/``"genus - mixed"``/``"genus - unknown"``, or ``"no
    match"``.

    A genus-only match's label is the UNION of every candidate entry's
    ``toxic_to_cats`` - "genus - mixed" means some relatives are
    cat-toxic and some aren't. This never infers the species' own
    toxicity from a relative - it's a label for illustrative analysis
    (e.g. checking how model accuracy varies by table-relevance), not
    the project's real toxicity-verdict path.
    """
    if result.status == "exact":
        return f"exact - {_CATS_LABEL[result.exact_row.toxic_to_cats]}"
    if result.status == "genus_only":
        signals = {entry.toxic_to_cats for entry in result.genus_candidates}
        if "yes" in signals and "no" in signals:
            return "genus - mixed"
        if "yes" in signals:
            return "genus - toxic"
        if "no" in signals:
            return "genus - non-toxic"
        return "genus - unknown"
    return "no match"
