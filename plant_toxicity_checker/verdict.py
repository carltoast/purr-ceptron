"""Combine classifier output with the toxicity table into a verdict.

Safety rule: the ONLY way to get ``LIKELY_SAFE`` is for the top match to be
confident AND every plausible candidate to have a *verified*, EXACT "not
toxic" entry. Anything unknown, unverified, missing or low-confidence falls
back to ``CAUTION``; a toxic top match is ``TOXIC``. A false alarm is
acceptable, a false "it's fine" is not.

A species with no exact table entry falls back to its genus
(``species_match.lookup_toxicity``/``toxicity_category``): a genus-level
PLACEHOLDER entry (ASPCA's own genus-wide claim, e.g. ``"Senecio
species"``) is confident enough for a full ``TOXIC`` verdict, same as an
exact match; a specific, different relative being toxic is weaker (we're
the ones generalizing there) and only ever reaches ``CAUTION``. Either way,
a genus-level match - however confident - can NEVER produce
``LIKELY_SAFE``: that still requires an exact, verified non-toxic entry for
the species itself, not an inference from a relative.

Defence in depth: callers cannot weaken the rule. Thresholds passed to
``assess`` can only be made stricter than the ``config`` defaults, and a
truncated prediction list that could be hiding a plausible toxic candidate
is never accepted as safe.
"""

from dataclasses import dataclass, field
from enum import Enum

from plant_toxicity_checker import species_match
from plant_toxicity_checker.config import CANDIDATE_FLOOR, MIN_CONFIDENCE

# species_match.toxicity_category's labels that are confident/weak
# enough to count as a toxic signal here. "exact - toxic": the species
# itself is in the table. "genus - toxic (direct)": a genus-level
# PLACEHOLDER entry makes an explicit, genus-wide claim - ASPCA's own
# generalization, confident enough for a full TOXIC verdict. "genus -
# toxic (inferred)": only a specific, different relative says toxic -
# WE'RE the ones generalizing there, so it only ever reaches CAUTION.
# Every other category (including both "non-toxic" ones) carries no
# toxic signal at all here - a non-toxic relative is never promoted to
# meaning this species is safe, only an exact verified entry can do
# that (see ``_is_verified_safe``).
_CONFIDENT_TOXIC_CATEGORIES = {"exact - toxic", "genus - toxic (direct)"}
_WEAK_TOXIC_CATEGORIES = {"genus - toxic (inferred)"}


def _toxic_tier(result):
    """``"confident"`` / ``"weak"`` / ``None`` for one
    ``species_match.MatchResult``, per the categories above."""
    category = species_match.toxicity_category(result)
    if category in _CONFIDENT_TOXIC_CATEGORIES:
        return "confident"
    if category in _WEAK_TOXIC_CATEGORIES:
        return "weak"
    return None

SCOPE_NOTE = (
    "Only the single most prominent plant in the photo was assessed. "
    "Other plants in the photo are NOT checked."
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


def _is_verified_safe(result):
    # Only an EXACT, verified non-toxic entry counts - never a genus
    # match, direct or inferred: we never infer "safe" for a species
    # from a relative, no matter how confident the relative's own
    # entry is.
    return (
        result.status == "exact"
        and result.exact_row.toxic_to_cats == "no"
        and result.exact_row.verified
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
    index = species_match.build_toxicity_index(table)
    top_result = species_match.lookup_toxicity(top.species, index)
    top_tier = _toxic_tier(top_result)
    reasons = []

    if top_tier == "confident":
        if top_result.status == "exact":
            reasons.append(f"Best match '{top.species}' is toxic to cats.")
        else:
            reasons.append(
                f"Best match '{top.species}' is in a genus ASPCA lists "
                "as toxic to cats."
            )
        if top.confidence < min_confidence:
            reasons.append("Confidence is low, so it may be another flower.")
        return Verdict(
            Status.TOXIC, top.species, top.confidence, top_result.exact_row,
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
    if top_tier == "weak":
        relative = next(
            e for e in top_result.genus_candidates if e.toxic_to_cats == "yes"
        )
        reasons.append(
            f"'{top.species}' is not itself in the toxicity table, but a "
            f"relative, '{relative.species}', is toxic to cats; treat as "
            "toxic."
        )

    for cand in candidates:
        cand_result = (
            top_result if cand is top
            else species_match.lookup_toxicity(cand.species, index)
        )
        if cand is top:
            # top's own toxicity (incl. "weak") was already handled
            # above; only its verified-safe eligibility is still open,
            # and only worth a separate reason if "weak" didn't already
            # explain why it's not verified-safe.
            if top_tier is None and not _is_verified_safe(cand_result):
                reasons.append(
                    f"No verified non-toxic entry for '{cand.species}'; "
                    "treat as toxic."
                )
            continue
        cand_tier = _toxic_tier(cand_result)
        if cand_tier == "confident":
            reasons.append(
                f"Plausible alternative '{cand.species}' "
                f"({cand.confidence:.0%}) is toxic to cats."
            )
        elif cand_tier == "weak":
            reasons.append(
                f"Plausible alternative '{cand.species}' "
                f"({cand.confidence:.0%}) has a toxic relative in the "
                "toxicity table; treat as toxic."
            )
        elif not _is_verified_safe(cand_result):
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
        status, top.species, top.confidence, top_result.exact_row,
        tuple(reasons), predictions,
    )
