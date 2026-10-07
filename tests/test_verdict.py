"""Tests for the safety rules that turn predictions into verdicts."""

import pytest

from plant_toxicity_checker.verdict import Prediction, Status, assess


def preds(*pairs):
    return [Prediction(s, c) for s, c in pairs]


def test_confident_verified_safe_is_likely_safe(table):
    verdict = assess(
        preds(("Rosa rugosa", 0.97), ("Helianthus annuus", 0.01)), table,
    )
    assert verdict.status is Status.LIKELY_SAFE


def test_confident_toxic_is_toxic(table):
    verdict = assess(preds(("Lilium tigrinum", 0.95)), table)
    assert verdict.status is Status.TOXIC


def test_low_confidence_toxic_is_still_toxic(table):
    verdict = assess(
        preds(("Lilium tigrinum", 0.30), ("Rosa rugosa", 0.25)), table,
    )
    assert verdict.status is Status.TOXIC


def test_low_confidence_safe_is_caution(table):
    verdict = assess(
        preds(("Rosa rugosa", 0.55), ("Helianthus annuus", 0.30)), table,
    )
    assert verdict.status is Status.CAUTION


def test_unknown_species_is_caution(table):
    verdict = assess(preds(("Dianthus caryophyllus", 0.99)), table)
    assert verdict.status is Status.CAUTION


def test_species_missing_from_table_is_caution(table):
    verdict = assess(preds(("Magnolia grandiflora", 0.99)), table)
    assert verdict.status is Status.CAUTION


def test_unverified_non_toxic_is_caution(table):
    verdict = assess(preds(("Petunia hybrida", 0.99)), table)
    assert verdict.status is Status.CAUTION


def test_unverified_toxic_is_still_toxic(table):
    verdict = assess(preds(("Narcissus pseudonarcissus", 0.99)), table)
    assert verdict.status is Status.TOXIC


def test_confident_safe_but_plausible_toxic_alternative_is_caution(table):
    verdict = assess(
        preds(("Rosa rugosa", 0.85), ("Lilium tigrinum", 0.10)), table,
    )
    assert verdict.status is Status.CAUTION
    assert any("Lilium tigrinum" in r for r in verdict.reasons)


def test_negligible_toxic_alternative_below_floor_is_ignored(table):
    verdict = assess(
        preds(("Rosa rugosa", 0.97), ("Lilium tigrinum", 0.01)), table,
    )
    assert verdict.status is Status.LIKELY_SAFE


def test_unknown_plausible_alternative_blocks_safe(table):
    verdict = assess(
        preds(("Rosa rugosa", 0.85), ("Dianthus caryophyllus", 0.10)), table,
    )
    assert verdict.status is Status.CAUTION


def test_no_predictions_is_caution(table):
    assert assess([], table).status is Status.CAUTION


@pytest.mark.parametrize("confidence", [0.79, 0.6, 0.5])
def test_threshold_boundary_below_is_not_safe(table, confidence):
    verdict = assess(
        preds(("Rosa rugosa", confidence),
              ("Helianthus annuus", 1 - confidence)),
        table,
    )
    assert verdict.status is Status.CAUTION


def test_threshold_boundary_at_threshold_is_safe(table):
    verdict = assess(
        preds(("Rosa rugosa", 0.80), ("Helianthus annuus", 0.20)), table,
    )
    assert verdict.status is Status.LIKELY_SAFE


# --- callers must not be able to weaken the safety rule -----------------


def test_truncated_list_hiding_possible_toxic_alternative_is_caution(table):
    # Only the top guess was passed in: 15% of the mass is unaccounted for
    # and could be a toxic species we were never shown.
    verdict = assess(preds(("Rosa rugosa", 0.85)), table)
    assert verdict.status is Status.CAUTION
    assert any("incomplete" in r for r in verdict.reasons)


def test_complete_distribution_is_not_flagged_truncated(table):
    verdict = assess(
        preds(
            ("Rosa rugosa", 0.96), ("Helianthus annuus", 0.03),
            ("Magnolia grandiflora", 0.01),
        ),
        table,
    )
    assert verdict.status is Status.LIKELY_SAFE


def test_min_confidence_can_only_be_raised(table):
    weak = assess(
        preds(("Rosa rugosa", 0.30), ("Helianthus annuus", 0.70)), table,
        min_confidence=0,
    )
    assert weak.status is Status.CAUTION
    strict = assess(
        preds(("Rosa rugosa", 0.85), ("Helianthus annuus", 0.15)), table,
        min_confidence=0.9,
    )
    assert strict.status is Status.CAUTION


def test_candidate_floor_can_only_be_lowered(table):
    verdict = assess(
        preds(
            ("Rosa rugosa", 0.85), ("Lilium tigrinum", 0.10),
            ("Helianthus annuus", 0.05),
        ),
        table, candidate_floor=0.5,
    )
    assert verdict.status is Status.CAUTION


def test_unsorted_predictions_are_ranked_first(table):
    verdict = assess(
        preds(("Helianthus annuus", 0.02), ("Rosa rugosa", 0.98)), table,
    )
    assert verdict.species == "Rosa rugosa"
    assert verdict.status is Status.LIKELY_SAFE


# --- genus-fallback matching ---------------------------------------------
# ("Lilium tigrinum" is toxic and exact; "Rosa rugosa" is non-toxic and
# exact - see the `table` fixture in conftest.py. These add genus-only
# relatives of those two genera that aren't themselves in the table.)


def test_genus_placeholder_toxic_top_match_is_toxic(table):
    # No exact entry for "Lilium longiflorum", but a genus-level
    # PLACEHOLDER ("Lilium spp.") says toxic - ASPCA's own genus-wide
    # claim, confident enough for a full TOXIC verdict.
    from plant_toxicity_checker.toxicity import ToxicityEntry, ToxicityTable

    placeholder = ToxicityEntry(
        species="Lilium spp.", scientific_name=None, toxic_to_cats="yes",
        severity="life_threatening", symptoms=(), sources=("aspca",),
        verified=False, notes="",
    )
    extended = ToxicityTable(list(table) + [placeholder])
    verdict = assess(preds(("Lilium longiflorum", 0.95)), extended)
    assert verdict.status is Status.TOXIC
    assert any("genus" in r for r in verdict.reasons)


def test_genus_sibling_toxic_top_match_is_only_caution(table):
    # No exact entry for "Lilium longiflorum", and no genus placeholder
    # either - only a SPECIFIC, different relative ("Lilium tigrinum")
    # happens to be toxic. That's inference on our part, not ASPCA's
    # own claim, so it's capped at CAUTION, never TOXIC.
    verdict = assess(preds(("Lilium longiflorum", 0.95)), table)
    assert verdict.status is Status.CAUTION
    assert any("Lilium tigrinum" in r for r in verdict.reasons)


def test_genus_match_never_produces_likely_safe(table):
    # Even a confident genus-level match can't promote to LIKELY_SAFE -
    # only an exact, verified non-toxic entry for the species ITSELF
    # can. "Helianthus debilis" has no exact entry; its genus relative
    # "Helianthus annuus" is verified non-toxic, but that must never
    # count as this species being verified safe.
    verdict = assess(preds(("Helianthus debilis", 0.95)), table)
    assert verdict.status is Status.CAUTION


def test_genus_toxic_alternative_blocks_likely_safe(table):
    # A plausible alternative with only a genus-level (not exact) toxic
    # signal must still block LIKELY_SAFE for the top match.
    verdict = assess(
        preds(("Rosa rugosa", 0.85), ("Lilium longiflorum", 0.10)), table,
    )
    assert verdict.status is Status.CAUTION
