"""Tests for scientific-name matching across datasets."""

from plant_toxicity_checker.species_match import (
    MatchResult,
    SpeciesIndex,
    build_toxicity_index,
    is_genus_placeholder,
    lookup_toxicity,
    match_against_toxicity_table,
    parse_scientific_name,
    species_sides,
    toxicity_category,
    toxicity_side,
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
    # "species" (not just "spp"/"sp") is also a real ASPCA-style genus
    # placeholder, e.g. "Senecio species" in the real scrape.
    assert parse_scientific_name("Rosa species") == ("rosa", None)


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


# --- build_toxicity_index / lookup_toxicity -------------------------------


def test_lookup_toxicity_exact_match():
    index = build_toxicity_index([toxicity_entry("Rosa canina", "yes")])
    result = lookup_toxicity("Rosa canina", index)
    assert result.status == "exact"
    assert result.exact_row.species == "Rosa canina"


def test_lookup_toxicity_genus_only_match():
    index = build_toxicity_index([toxicity_entry("Rosa canina", "yes")])
    result = lookup_toxicity("Rosa rugosa", index)
    assert result.status == "genus_only"
    assert result.genus_candidates[0].species == "Rosa canina"


def test_lookup_toxicity_no_match():
    index = build_toxicity_index([toxicity_entry("Rosa canina", "yes")])
    assert lookup_toxicity("Lilium longiflorum", index) == MatchResult("none")


def test_build_toxicity_index_skips_unparseable_entries():
    index = build_toxicity_index([toxicity_entry("", "yes")])
    assert lookup_toxicity("Rosa canina", index) == MatchResult("none")


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


# --- is_genus_placeholder -------------------------------------------------


def test_is_genus_placeholder_true_for_a_genus_only_name():
    assert is_genus_placeholder(toxicity_entry("Rosa spp.", "yes"))
    assert is_genus_placeholder(toxicity_entry("Rosa species", "yes"))


def test_is_genus_placeholder_false_for_a_specific_species():
    assert not is_genus_placeholder(toxicity_entry("Rosa canina", "yes"))


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


def test_toxicity_category_genus_direct_toxic():
    # A genus-level PLACEHOLDER entry itself says toxic (ASPCA's own
    # genus-wide claim) - "direct", parent -> child.
    result = MatchResult(
        "genus_only", None, (toxicity_entry("Rosa spp.", "yes"),),
    )
    assert toxicity_category(result) == "genus - toxic (direct)"


def test_toxicity_category_genus_inferred_toxic():
    # Only a SPECIFIC sibling species says toxic, no genus placeholder
    # at all - "inferred", sibling <-> sibling, a weaker basis.
    result = MatchResult(
        "genus_only", None, (toxicity_entry("Rosa canina", "yes"),),
    )
    assert toxicity_category(result) == "genus - toxic (inferred)"


def test_toxicity_category_genus_direct_non_toxic():
    result = MatchResult(
        "genus_only", None, (toxicity_entry("Rosa spp.", "no"),),
    )
    assert toxicity_category(result) == "genus - non-toxic (direct)"


def test_toxicity_category_genus_inferred_non_toxic():
    result = MatchResult(
        "genus_only", None, (toxicity_entry("Rosa canina", "no"),),
    )
    assert toxicity_category(result) == "genus - non-toxic (inferred)"


def test_toxicity_category_toxic_always_wins_over_non_toxic():
    # A non-toxic sibling must never cancel out a toxic one, regardless
    # of which kind of candidate carries which signal.
    candidates = (
        toxicity_entry("Rosa canina", "no"),
        toxicity_entry("Rosa gallica", "yes"),
    )
    result = MatchResult("genus_only", None, candidates)
    assert toxicity_category(result) == "genus - toxic (inferred)"


def test_toxicity_category_direct_wins_over_inferred_for_the_same_signal():
    # A placeholder AND a sibling both say toxic - still "direct",
    # since that's the stronger of the two bases for the same signal.
    candidates = (
        toxicity_entry("Rosa spp.", "yes"),
        toxicity_entry("Rosa canina", "no"),
    )
    result = MatchResult("genus_only", None, candidates)
    assert toxicity_category(result) == "genus - toxic (direct)"


def test_toxicity_category_genus_only_unknown():
    result = MatchResult(
        "genus_only", None, (toxicity_entry("Rosa canina", "unknown"),),
    )
    assert toxicity_category(result) == "genus - unknown"


def test_toxicity_category_no_match():
    assert toxicity_category(MatchResult("none")) == "no match"


# --- toxicity_side / species_sides -----------------------------------------


def test_toxicity_side_toxic_for_any_toxic_category():
    result = MatchResult("exact", toxicity_entry("Rosa canina", "yes"), ())
    assert toxicity_side(result) == "toxic"

    genus_direct = MatchResult(
        "genus_only", None, (toxicity_entry("Rosa spp.", "yes"),),
    )
    assert toxicity_side(genus_direct) == "toxic"

    genus_inferred = MatchResult(
        "genus_only", None, (toxicity_entry("Rosa canina", "yes"),),
    )
    assert toxicity_side(genus_inferred) == "toxic"


def test_toxicity_side_non_toxic_for_any_non_toxic_category():
    result = MatchResult("exact", toxicity_entry("Rosa canina", "no"), ())
    assert toxicity_side(result) == "non_toxic"


def test_toxicity_side_none_for_no_match():
    assert toxicity_side(MatchResult("none")) is None


def test_toxicity_side_none_for_unknown():
    result = MatchResult(
        "exact", toxicity_entry("Rosa canina", "unknown"), (),
    )
    assert toxicity_side(result) is None


class _FakeLabels:
    """Minimal duck-typed stand-in for plantnet_labels.SpeciesLabels -
    just the two members species_sides actually uses."""

    def __init__(self, species_ids_in_index_order):
        self._ids = [str(s) for s in species_ids_in_index_order]

    def __len__(self):
        return len(self._ids)

    def index_of_species_id(self, species_id):
        return self._ids.index(str(species_id))


def test_species_sides_builds_dense_array_by_class_index():
    plantnet_rows = [
        {"species_id": "1", "genus": "Rosa", "epithet": "canina"},
        {"species_id": "2", "genus": "Lilium", "epithet": "longiflorum"},
        {"species_id": "3", "genus": "Tulipa", "epithet": "gesneriana"},
    ]
    toxicity_entries = [
        toxicity_entry("Rosa canina", "yes"),
        toxicity_entry("Lilium longiflorum", "no"),
    ]
    # class index 0 -> species_id 3, 1 -> species_id 1, 2 -> species_id 2
    labels = _FakeLabels(["3", "1", "2"])

    sides = species_sides(plantnet_rows, toxicity_entries, labels)

    assert sides == [-1, 1, 0]


def test_species_sides_length_matches_label_space_not_row_count():
    plantnet_rows = [{"species_id": "1", "genus": "Rosa", "epithet": "canina"}]
    labels = _FakeLabels(["1", "2", "3"])
    sides = species_sides(plantnet_rows, [], labels)
    assert len(sides) == 3
