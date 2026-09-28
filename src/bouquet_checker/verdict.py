"""Combine classifier output with the toxicity table into a verdict.

Safety rule: the ONLY way to get ``LIKELY_SAFE`` is for the top match to be
confident AND every plausible candidate to have a *verified* "not toxic"
entry. Anything unknown, unverified, missing or low-confidence falls back to
``CAUTION``; a toxic top match is ``TOXIC``. A false alarm is acceptable, a
false "it's fine" is not.

Defence in depth: callers cannot weaken the rule. Thresholds passed to
``assess`` can only be made stricter than the ``config`` defaults, and a
truncated prediction list that could be hiding a plausible toxic candidate
is never accepted as safe.
"""

from dataclasses import dataclass, field
from enum import Enum

from bouquet_checker.config import CANDIDATE_FLOOR, MIN_CONFIDENCE

SCOPE_NOTE = (
    "Only the single most prominent flower in the photo was assessed. "
    "Other flowers in a bouquet are NOT checked."
)

# Float slack when comparing accumulated probability mass to the floor.
_EPS = 1e-6


class Status(str, Enum):
    """Overall outcome, ordered from most to least alarming."""

    TOXIC = "toxic"
    CAUTION = "caution"
    LIKELY_SAFE = "likely_safe"


@dataclass(frozen=True)
class Prediction:
    """One classifier guess: canonical species name and probability."""

    species: str
    confidence: float


@dataclass(frozen=True)
class Verdict:
    """The final answer plus the evidence behind it."""

    status: Status
    species: str | None
    confidence: float | None
    entry: object | None
    reasons: tuple[str, ...]
    predictions: tuple[Prediction, ...] = field(default_factory=tuple)


def _is_toxic(entry):
    # Unverified "yes" still counts: erring towards alarm is always safe.
    return entry is not None and entry.toxic_to_cats == "yes"


def _is_verified_safe(entry):
    return (
        entry is not None
        and entry.toxic_to_cats == "no"
        and entry.verified
    )


def _may_be_truncated(predictions, floor):
    """True if unseen classes could still hold a candidate >= ``floor``.

    Any class missing from the list has probability at most the smallest
    listed one, and at most the unaccounted-for mass. If both can reach the
    floor, we cannot rule out a plausible candidate we were never shown.
    """
    unseen = 1.0 - sum(p.confidence for p in predictions)
    return (
        predictions[-1].confidence >= floor - _EPS
        and unseen >= floor - _EPS
    )


def assess(
    predictions,
    table,
    min_confidence=MIN_CONFIDENCE,
    candidate_floor=CANDIDATE_FLOOR,
):
    """Return a ``Verdict`` for ``predictions`` (a full distribution).

    ``min_confidence`` can only be raised above, and ``candidate_floor``
    only lowered below, the ``config`` defaults.
    """
    min_confidence = max(min_confidence, MIN_CONFIDENCE)
    candidate_floor = min(candidate_floor, CANDIDATE_FLOOR)

    predictions = tuple(
        sorted(predictions, key=lambda p: p.confidence, reverse=True)
    )
    if not predictions:
        return Verdict(
            Status.CAUTION, None, None, None,
            ("No prediction was produced; treat as toxic.",),
        )

    top = predictions[0]
    top_entry = table.get(top.species)
    reasons = []

    if _is_toxic(top_entry):
        reasons.append(f"Best match '{top.species}' is toxic to cats.")
        if top.confidence < min_confidence:
            reasons.append("Confidence is low, so it may be another flower.")
        return Verdict(
            Status.TOXIC, top.species, top.confidence, top_entry,
            tuple(reasons), predictions,
        )

    candidates = [p for p in predictions if p.confidence >= candidate_floor]
    if top not in candidates:
        candidates.insert(0, top)

    if top.confidence < min_confidence:
        reasons.append(
            f"Confidence {top.confidence:.0%} is below "
            f"{min_confidence:.0%}; treat as toxic."
        )
    if _may_be_truncated(predictions, candidate_floor):
        reasons.append(
            "The prediction list is incomplete, so a toxic alternative "
            "cannot be ruled out; treat as toxic."
        )
    for cand in candidates:
        entry = table.get(cand.species)
        if _is_toxic(entry):
            reasons.append(
                f"Plausible alternative '{cand.species}' "
                f"({cand.confidence:.0%}) is toxic to cats."
            )
        elif not _is_verified_safe(entry):
            reasons.append(
                f"No verified non-toxic entry for '{cand.species}'; "
                "treat as toxic."
            )

    status = Status.CAUTION if reasons else Status.LIKELY_SAFE
    if not reasons:
        reasons.append(
            f"'{top.species}' is verified non-toxic to cats and the match "
            "is confident. Still check with a vet if a cat ate any."
        )
    return Verdict(
        status, top.species, top.confidence, top_entry,
        tuple(reasons), predictions,
    )
