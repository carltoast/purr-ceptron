"""Stage 2: the curated species -> toxicity lookup table (not ML)."""

import json
from dataclasses import dataclass
from pathlib import Path

TOXIC_VALUES = ("yes", "no", "unknown")
SEVERITIES = ("none", "mild", "moderate", "severe", "life_threatening")


class ToxicityTableError(ValueError):
    """The toxicity table is malformed or internally inconsistent."""


@dataclass(frozen=True)
class ToxicityEntry:
    """One species' cat-toxicity record.

    ``verified`` means a human cross-checked it against ASPCA (and any other
    listed source). Only verified entries may ever produce a "likely safe"
    verdict.
    """

    species: str
    scientific_name: str | None
    toxic_to_cats: str
    severity: str | None
    symptoms: tuple[str, ...]
    sources: tuple[str, ...]
    verified: bool
    notes: str


def _str_list(name, raw, field):
    value = raw.get(field, [])
    ok = isinstance(value, list) and all(
        isinstance(v, str) and v.strip() for v in value
    )
    if not ok:
        raise ToxicityTableError(
            f"{name!r}: {field} must be a list of non-empty strings"
        )
    return tuple(value)


def _parse_entry(raw, valid_species=None):
    if not isinstance(raw, dict) or not isinstance(raw.get("species"), str):
        raise ToxicityTableError(f"Entry needs a string 'species': {raw}")
    name = raw["species"]
    if "toxic_to_cats" not in raw:
        raise ToxicityTableError(f"{name!r}: missing 'toxic_to_cats'")

    # Strict on purpose: bool("false") is True, and a hand-edit typo must
    # never be able to turn an entry into a verified one.
    verified = raw.get("verified", False)
    if not isinstance(verified, bool):
        raise ToxicityTableError(
            f"{name!r}: verified must be true or false, got {verified!r}"
        )
    scientific_name = raw.get("scientific_name")
    if scientific_name is not None and not isinstance(scientific_name, str):
        raise ToxicityTableError(
            f"{name!r}: scientific_name must be a string or null"
        )
    notes = raw.get("notes", "")
    if not isinstance(notes, str):
        raise ToxicityTableError(f"{name!r}: notes must be a string")

    entry = ToxicityEntry(
        species=name,
        scientific_name=scientific_name,
        toxic_to_cats=raw["toxic_to_cats"],
        severity=raw.get("severity"),
        symptoms=_str_list(name, raw, "symptoms"),
        sources=_str_list(name, raw, "sources"),
        verified=verified,
        notes=notes,
    )
    _validate_entry(entry, valid_species)
    return entry


def _validate_entry(entry, valid_species=None):
    """Check one entry; ``valid_species``, if given, restricts the name.

    ``valid_species`` is deliberately optional and caller-supplied rather
    than a module-level constant: this table's schema is reused across
    different species universes (Oxford-102, PlantNet, ...), and nothing
    here should silently assume which one it is - see ``labels``-style
    modules for a real name list to pass in.
    """
    name = entry.species
    if valid_species is not None and entry.species not in valid_species:
        raise ToxicityTableError(f"{name!r}: not a known species")
    if entry.toxic_to_cats not in TOXIC_VALUES:
        raise ToxicityTableError(
            f"{name!r}: toxic_to_cats must be one of {TOXIC_VALUES}"
        )
    if entry.severity is not None and entry.severity not in SEVERITIES:
        raise ToxicityTableError(
            f"{name!r}: severity must be null or one of {SEVERITIES}"
        )
    _validate_consistency(entry)
    if entry.verified:
        if entry.toxic_to_cats == "unknown":
            raise ToxicityTableError(
                f"{name!r}: verified entries cannot be 'unknown'"
            )
        if not entry.sources:
            raise ToxicityTableError(
                f"{name!r}: verified entries need at least one source"
            )
        if entry.toxic_to_cats == "yes" and entry.severity is None:
            raise ToxicityTableError(
                f"{name!r}: verified toxic entries need a severity"
            )


def _validate_consistency(entry):
    """Reject entries whose fields contradict each other."""
    name = entry.species
    if entry.toxic_to_cats == "yes" and entry.severity == "none":
        raise ToxicityTableError(
            f"{name!r}: toxic entry cannot have severity 'none'"
        )
    if entry.toxic_to_cats == "no":
        if entry.severity not in (None, "none"):
            raise ToxicityTableError(
                f"{name!r}: non-toxic entry cannot have severity "
                f"{entry.severity!r}"
            )
        if entry.symptoms:
            raise ToxicityTableError(
                f"{name!r}: non-toxic entry cannot list symptoms"
            )
    if entry.toxic_to_cats == "unknown" and entry.severity is not None:
        raise ToxicityTableError(
            f"{name!r}: 'unknown' entry cannot have a severity"
        )


class ToxicityTable:
    """Species-name -> ``ToxicityEntry`` lookup."""

    def __init__(self, entries):
        self._entries = {}
        for entry in entries:
            if entry.species in self._entries:
                raise ToxicityTableError(
                    f"Duplicate species {entry.species!r}"
                )
            self._entries[entry.species] = entry

    def __len__(self):
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())

    def get(self, species):
        """Return the entry for ``species``, or None if it has none."""
        return self._entries.get(species)

    def missing_species(self, valid_species):
        """Species in ``valid_species`` that have no entry at all."""
        return [s for s in valid_species if s not in self._entries]


def load_table(path, valid_species=None):
    """Load and validate the JSON table at ``path``.

    ``valid_species``, if given, is checked against every entry's
    ``species`` (see ``_validate_entry``).
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise ToxicityTableError(f"{path} is not valid JSON: {err}") from err
    if not isinstance(data, dict) or not isinstance(
        data.get("species"), list
    ):
        raise ToxicityTableError(f"{path} needs a top-level 'species' list")
    return ToxicityTable(
        _parse_entry(raw, valid_species) for raw in data["species"]
    )
