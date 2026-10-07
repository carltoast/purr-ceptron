"""Tests for scientific-name matching across datasets."""

from plant_toxicity_checker.species_match import (
    MatchResult,
    SpeciesIndex,
    match_against_toxicity_table,
    parse_scientific_name,
    toxicity_category,
)
from plant_toxicity_checker.toxicity import ToxicityEntry


def toxicity_entry(species, toxic_to_cats):
    return ToxicityEntry(
        species=species, scientific_name=species,
        toxic_to_cats=toxic_to_cats, severity=None, symptoms=(),
        sources=(), verified=False, notes="",
    )


def test_parses_a_plain_binomial():
    assert parse_scientific_name("Rosa canina") == ("rosa", "canina")


def test_parses_a_genus_only_name():
    assert parse_scientific_name("Rosa spp.") == ("rosa", None)
    assert parse_scientific_name("Rosa spp") == ("rosa", None)
    assert parse_scientific_name("Rosa") == ("rosa", None)


def test_a_qualifier_is_not_mistaken_for_an_epithet():
    assert parse_scientific_name("Asparagus densiflorus cv sprengeri") == (
        "asparagus", "densiflorus",
    )
    assert parse_scientific_name("Cucurbita maxima var. banana") == (
        "cucurbita", "maxima",
    )


def test_only_the_first_comma_clause_is_used():
    assert parse_scientific_name("Scindapsus, Philodendron spp") == (
        "scindapsus", None,
    )


def test_case_and_hyphens_are_handled():
    assert parse_scientific_name("SOLANUM Lycopersicum") == (
        "solanum", "lycopersicum",
    )
    assert parse_scientific_name("Chrysanthemum x morifolium") == (
        "chrysanthemum", None,  # "x" is a hybrid marker, not an epithet
    )


def test_empty_or_unusable_text_returns_none():
    assert parse_scientific_name("") is None
    assert parse_scientific_name(None) is None
    assert parse_scientific_name("???") is None


# --- SpeciesIndex ------------------------------------------------------------


def rows():
    return [
        {"genus": "Rosa", "epithet": "canina", "id": "r1"},
        {"genus": "Rosa", "epithet": "gallica", "id": "r2"},
        {"genus": "Tulipa", "epithet": "", "id": "t1"},   # no epithet given
        {"genus": "", "epithet": "x", "id": "bad"},        # no genus: skipped
    ]


def build_index():
    return SpeciesIndex.from_rows(
        rows(), genus_of=lambda r: r["genus"],
        epithet_of=lambda r: r["epithet"],
    )


def test_exact_match_also_returns_genus_siblings():
    index = build_index()
    result = index.match("Rosa canina")
    assert result.status == "exact"
    assert result.exact_row["id"] == "r1"
    assert {r["id"] for r in result.genus_candidates} == {"r1", "r2"}


def test_genus_only_match_when_no_exact_species():
    index = build_index()
    result = index.match("Rosa rugosa")           # genus known, species not
    assert result.status == "genus_only"
    assert result.exact_row is None
    assert {r["id"] for r in result.genus_candidates} == {"r1", "r2"}


def test_genus_only_match_for_a_bare_genus_query():
    index = build_index()
    assert index.match("Rosa spp.").status == "genus_only"


def test_a_row_with_no_epithet_is_still_findable_by_genus():
    index = build_index()
    result = index.match("Tulipa gesneriana")
    assert result.status == "genus_only"
    assert result.genus_candidates[0]["id"] == "t1"


def test_no_match_for_an_unknown_genus():
    index = build_index()
    assert index.match("Lilium longiflorum") == MatchResult("none")


def test_no_match_for_unparseable_input():
    index = build_index()
    assert index.match("") == MatchResult("none")


def test_rows_without_a_genus_are_skipped_entirely():
    index = build_index()
    assert "" not in index.by_genus


def test_matching_is_symmetric_aspca_index_or_dataset_index():
    # The same SpeciesIndex works built from either "direction": here it is
    # built from ASPCA-shaped rows and matched against a photo-dataset name.
    aspca_rows = [{"scientific_name": "Rosa canina", "toxic": False}]
    index = SpeciesIndex.from_rows(
        aspca_rows,
        genus_of=lambda r: parse_scientific_name(r["scientific_name"])[0],
        epithet_of=lambda r: parse_scientific_name(r["scientific_name"])[1],
    )
    result = index.match("Rosa gallica")
    assert result.status == "genus_only"
    assert result.genus_candidates[0]["toxic"] is False


# --- match_against_toxicity_table ---------------------------------------


def test_match_against_toxicity_table_exact_match():
    plantnet_rows = [
        {"species_id": "1", "genus": "Rosa", "epithet": "canina"},
        {"species_id": "2", "genus": "Rosa", "epithet": "rugosa"},  # genus
        {"species_id": "3", "genus": "Lilium", "epithet": "longiflorum"},
    ]
    toxicity_entries = [toxicity_entry("Rosa canina", "yes")]
    matches = match_against_toxicity_table(plantnet_rows, toxicity_entries)

    assert matches[1].status == "exact"
    assert matches[1].exact_row.species == "Rosa canina"
    assert matches[2].status == "genus_only"
    assert matches[2].genus_candidates[0].species == "Rosa canina"
    assert matches[3].status == "none"


def test_match_against_toxicity_table_skips_rows_with_no_genus():
    plantnet_rows = [{"species_id": "1", "genus": "", "epithet": "canina"}]
    toxicity_entries = [toxicity_entry("Rosa canina", "yes")]
    matches = match_against_toxicity_table(plantnet_rows, toxicity_entries)
    assert matches[1] == MatchResult("none")


def test_match_against_toxicity_table_skips_unparseable_entries():
    plantnet_rows = [{"species_id": "1", "genus": "Rosa", "epithet": "canina"}]
    toxicity_entries = [toxicity_entry("", "yes")]
    matches = match_against_toxicity_table(plantnet_rows, toxicity_entries)
    assert matches[1] == MatchResult("none")


# --- toxicity_category ----------------------------------------------------


def test_toxicity_category_exact_toxic():
    result = MatchResult("exact", toxicity_entry("Rosa canina", "yes"), ())
    assert toxicity_category(result) == "exact - toxic"


def test_toxicity_category_exact_non_toxic():
    result = MatchResult("exact", toxicity_entry("Rosa canina", "no"), ())
    assert toxicity_category(result) == "exact - non-toxic"


def test_toxicity_category_exact_unknown():
    result = MatchResult(
        "exact", toxicity_entry("Rosa canina", "unknown"), (),
    )
    assert toxicity_category(result) == "exact - unknown"


def test_toxicity_category_genus_only_toxic():
    result = MatchResult(
        "genus_only", None, (toxicity_entry("Rosa canina", "yes"),),
    )
    assert toxicity_category(result) == "genus - toxic"


def test_toxicity_category_genus_only_mixed():
    # Two relatives of the same genus disagree on cat toxicity.
    candidates = (
        toxicity_entry("Rosa canina", "yes"),
        toxicity_entry("Rosa gallica", "no"),
    )
    result = MatchResult("genus_only", None, candidates)
    assert toxicity_category(result) == "genus - mixed"


def test_toxicity_category_genus_only_unknown():
    result = MatchResult(
        "genus_only", None, (toxicity_entry("Rosa canina", "unknown"),),
    )
    assert toxicity_category(result) == "genus - unknown"


def test_toxicity_category_no_match():
    assert toxicity_category(MatchResult("none")) == "no match"
