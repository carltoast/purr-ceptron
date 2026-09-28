"""Tests for the safety rules that turn predictions into verdicts."""

import pytest

from bouquet_checker.verdict import Prediction, Status, assess


def preds(*pairs):
    return [Prediction(s, c) for s, c in pairs]


def test_confident_verified_safe_is_likely_safe(table):
    verdict = assess(preds(("rose", 0.97), ("sunflower", 0.01)), table)
    assert verdict.status is Status.LIKELY_SAFE


def test_confident_toxic_is_toxic(table):
    verdict = assess(preds(("tiger lily", 0.95)), table)
    assert verdict.status is Status.TOXIC


def test_low_confidence_toxic_is_still_toxic(table):
    verdict = assess(preds(("tiger lily", 0.30), ("rose", 0.25)), table)
    assert verdict.status is Status.TOXIC


def test_low_confidence_safe_is_caution(table):
    verdict = assess(preds(("rose", 0.55), ("sunflower", 0.30)), table)
    assert verdict.status is Status.CAUTION


def test_unknown_species_is_caution(table):
    verdict = assess(preds(("carnation", 0.99)), table)
    assert verdict.status is Status.CAUTION


def test_species_missing_from_table_is_caution(table):
    verdict = assess(preds(("magnolia", 0.99)), table)
    assert verdict.status is Status.CAUTION


def test_unverified_non_toxic_is_caution(table):
    verdict = assess(preds(("petunia", 0.99)), table)
    assert verdict.status is Status.CAUTION


def test_unverified_toxic_is_still_toxic(table):
    verdict = assess(preds(("daffodil", 0.99)), table)
    assert verdict.status is Status.TOXIC


def test_confident_safe_but_plausible_toxic_alternative_is_caution(table):
    verdict = assess(preds(("rose", 0.85), ("tiger lily", 0.10)), table)
    assert verdict.status is Status.CAUTION
    assert any("tiger lily" in r for r in verdict.reasons)


def test_negligible_toxic_alternative_below_floor_is_ignored(table):
    verdict = assess(preds(("rose", 0.97), ("tiger lily", 0.01)), table)
    assert verdict.status is Status.LIKELY_SAFE


def test_unknown_plausible_alternative_blocks_safe(table):
    verdict = assess(preds(("rose", 0.85), ("carnation", 0.10)), table)
    assert verdict.status is Status.CAUTION


def test_no_predictions_is_caution(table):
    assert assess([], table).status is Status.CAUTION


@pytest.mark.parametrize("confidence", [0.79, 0.6, 0.5])
def test_threshold_boundary_below_is_not_safe(table, confidence):
    verdict = assess(
        preds(("rose", confidence), ("sunflower", 1 - confidence)), table
    )
    assert verdict.status is Status.CAUTION


def test_threshold_boundary_at_threshold_is_safe(table):
    verdict = assess(preds(("rose", 0.80), ("sunflower", 0.20)), table)
    assert verdict.status is Status.LIKELY_SAFE


# --- callers must not be able to weaken the safety rule -----------------


def test_truncated_list_hiding_possible_toxic_alternative_is_caution(table):
    # Only the top guess was passed in: 15% of the mass is unaccounted for
    # and could be a toxic species we were never shown.
    verdict = assess(preds(("rose", 0.85)), table)
    assert verdict.status is Status.CAUTION
    assert any("incomplete" in r for r in verdict.reasons)


def test_complete_distribution_is_not_flagged_truncated(table):
    verdict = assess(
        preds(("rose", 0.96), ("sunflower", 0.03), ("magnolia", 0.01)),
        table,
    )
    assert verdict.status is Status.LIKELY_SAFE


def test_min_confidence_can_only_be_raised(table):
    weak = assess(
        preds(("rose", 0.30), ("sunflower", 0.70)), table, min_confidence=0
    )
    assert weak.status is Status.CAUTION
    strict = assess(
        preds(("rose", 0.85), ("sunflower", 0.15)), table,
        min_confidence=0.9,
    )
    assert strict.status is Status.CAUTION


def test_candidate_floor_can_only_be_lowered(table):
    verdict = assess(
        preds(("rose", 0.85), ("tiger lily", 0.10), ("sunflower", 0.05)),
        table, candidate_floor=0.5,
    )
    assert verdict.status is Status.CAUTION


def test_unsorted_predictions_are_ranked_first(table):
    verdict = assess(preds(("sunflower", 0.02), ("rose", 0.98)), table)
    assert verdict.species == "rose"
    assert verdict.status is Status.LIKELY_SAFE
