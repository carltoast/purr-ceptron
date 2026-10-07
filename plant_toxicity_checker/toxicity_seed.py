"""Seed ``data/toxicity.json``'s starting entries from the ASPCA
cat-toxicity scrape (``aspca_detail.py``'s output).

Every entry is written ``"verified": false``, never ``true`` - this
scrape is a first draft (see ``aspca_detail.py``'s own docstring), and
CLAUDE.md is explicit that it must never set ``verified`` itself. A
human must review and flip individual entries to ``verified: true``
(correcting anything wrong along the way) by hand afterward; nothing
here is part of the safety path.

ASPCA documents some species under more than one page - different
common names or cultivars sharing one scientific binomial (e.g. several
distinct ``Asparagus densiflorus`` cultivar pages). This groups by
``scientific_name`` (the identity granularity used throughout this
project - see ``species_match.py``'s genus+epithet matching) and merges
matches into one entry, carrying every page as a separate source.
Grouped pages that disagree on cat toxicity are a real data problem,
not noise to silently resolve - ``seed_entries`` refuses those instead
of guessing.
"""

import json
from pathlib import Path


class SeedError(ValueError):
    """ASPCA pages sharing a scientific name disagree on cat toxicity."""


def _toxic_to_cats(plant):
    """``"yes"``/``"no"``/``"unknown"`` from one ASPCA plant's
    ``toxic_to``/``non_toxic_to``, for cats specifically."""
    if "cats" in plant.get("toxic_to", []):
        return "yes"
    if "cats" in plant.get("non_toxic_to", []):
        return "no"
    return "unknown"


def _group_by_scientific_name(aspca_plants):
    groups = {}
    for plant in aspca_plants:
        scientific_name = (plant.get("scientific_name") or "").strip()
        if not scientific_name:
            continue
        groups.setdefault(scientific_name, []).append(plant)
    return groups


def _notes_for(group):
    common_names = ", ".join(plant["name"] for plant in group)
    lines = [f"ASPCA common name(s): {common_names}"]
    for plant in group:
        principles = plant.get("toxic_principles") or "not listed"
        signs = plant.get("clinical_signs") or "not listed"
        lines.append(
            f"[{plant['name']}] toxic principles: {principles} | "
            f"clinical signs: {signs}"
        )
    return "\n".join(lines)


def seed_entries(aspca_plants):
    """One unverified ``toxicity.json``-schema entry per distinct
    scientific name in ``aspca_plants`` (``cats_plant_details.json``'s
    ``"plants"`` list), sorted by ``species``.

    Raises ``SeedError`` if ASPCA pages sharing a scientific name
    disagree on cat toxicity - hasn't happened in the real scrape (all
    199 multi-page groups agree), but must never be silently resolved
    one way if it ever did; that judgment call belongs to whoever
    verifies the table, not to an automatic seed.
    """
    entries = []
    for scientific_name, group in _group_by_scientific_name(
        aspca_plants
    ).items():
        signals = {_toxic_to_cats(plant) for plant in group}
        if len(signals) > 1:
            pages = ", ".join(
                f"{plant['name']!r} ({_toxic_to_cats(plant)})"
                for plant in group
            )
            raise SeedError(
                f"{scientific_name!r}: ASPCA pages disagree on cat "
                f"toxicity - {pages}"
            )
        toxic_to_cats = signals.pop()
        sources = [plant["url"] for plant in group if plant.get("url")]
        entries.append({
            "species": scientific_name,
            "scientific_name": scientific_name,
            "toxic_to_cats": toxic_to_cats,
            "severity": None,
            "symptoms": [],
            "sources": sources,
            "verified": False,
            "notes": _notes_for(group),
        })
    entries.sort(key=lambda entry: entry["species"])
    return entries


def write_seed_table(aspca_plants, out_path):
    """Write a fresh ``data/toxicity.json`` seeded from ``aspca_plants``
    - every entry unverified.

    Overwrites ``out_path``: a one-time initial seed, not an
    incremental merge - running this again after entries have been hand
    -verified would discard that work. See ``toxicity.load_table`` for
    reading the result back.
    """
    document = {"species": seed_entries(aspca_plants)}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return document
