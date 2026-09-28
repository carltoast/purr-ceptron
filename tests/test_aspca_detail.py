"""Tests for the ASPCA per-plant detail-page scraper (offline; faked net)."""

import json
import urllib.request

import pytest

from bouquet_checker import aspca_detail
from bouquet_checker.aspca_detail import (
    AspcaDetailError,
    PlantRef,
    parse_detail_page,
    parse_species_list,
)


def field(name, value, label=None):
    if not value:
        return ""
    return (
        f'<div class="field field-name-field-{name} field-label-hidden">'
        '<div class="field-items"><div class="field-item even">'
        f'<span class="label-inline-format-label">{label or name}:</span>'
        f'<span class="values">{value}</span>'
        "</div></div></div>"
    )


def page(name="Adam-and-Eve", common_names="Arum, Wake Robin",
         scientific="Arum maculatum", family="Araceae",
         toxicity="Toxic to Dogs, Toxic to Cats, Toxic to Horses",
         non_toxicity="", principles="Insoluble calcium oxalates",
         signs="Oral irritation, vomiting"):
    names_field = field(
        "additional-common-names", common_names, "Additional Common Names"
    )
    return (
        "<html><body>"
        f"<h1>{name}</h1>"
        f"{names_field}"
        f'{field("scientific-name", scientific, "Scientific Name")}'
        f'{field("family", family, "Family")}'
        f'{field("toxicity", toxicity, "Toxicity")}'
        f'{field("non-toxicity", non_toxicity, "Non-Toxicity")}'
        f'{field("toxic-principles", principles, "Toxic Principles")}'
        f'{field("clinical-signs", signs, "Clinical Signs")}'
        "</body></html>"
    )


URL = ("https://www.aspca.org/pet-care/aspca-poison-control/"
      "toxic-and-non-toxic-plants/adam-and-eve")


# --- parsing one page ------------------------------------------------------


def test_a_toxic_page_is_parsed_fully():
    detail = parse_detail_page(page(), URL)
    assert detail.name == "Adam-and-Eve"
    assert detail.additional_common_names == ("Arum", "Wake Robin")
    assert detail.scientific_name == "Arum maculatum"
    assert detail.family == "Araceae"
    assert detail.toxic_principles == "Insoluble calcium oxalates"
    assert detail.clinical_signs == "Oral irritation, vomiting"
    assert detail.toxic_to == ("dogs", "cats", "horses")
    assert detail.non_toxic_to == ()
    assert detail.slug == "adam-and-eve"
    assert detail.url == URL


def test_a_non_toxic_only_page_has_no_toxicity_fields():
    non_toxic = "Non-Toxic to Dogs, Non-Toxic to Cats, Non-Toxic to Horses"
    html = page(
        name="African Violet", common_names="", scientific="Saintpaulia spp.",
        family="Gesneriaceae", toxicity="", principles="", signs="",
        non_toxicity=non_toxic,
    )
    detail = parse_detail_page(html, URL)
    assert detail.toxic_to == ()
    assert detail.non_toxic_to == ("dogs", "cats", "horses")
    assert detail.toxic_principles == ""
    assert detail.clinical_signs == ""
    assert detail.additional_common_names == ()


def test_a_page_can_have_both_toxicity_and_non_toxicity():
    # The genuinely asymmetric case: toxic to some species, confirmed
    # non-toxic to another.
    html = page(toxicity="Toxic to Cats", non_toxicity="Non-Toxic to Dogs")
    detail = parse_detail_page(html, URL)
    assert detail.toxic_to_species("cats") == "yes"
    assert detail.toxic_to_species("dogs") == "no"
    assert detail.toxic_to_species("horses") == "unknown"


def test_species_named_in_neither_field_is_unknown_not_guessed():
    detail = parse_detail_page(page(toxicity="Toxic to Cats"), URL)
    assert detail.toxic_to_species("cats") == "yes"
    assert detail.toxic_to_species("dogs") == "unknown"
    assert detail.toxic_to_species("horses") == "unknown"


def test_missing_h1_is_rejected():
    html = page().replace("<h1>Adam-and-Eve</h1>", "")
    with pytest.raises(AspcaDetailError, match="No <h1>"):
        parse_detail_page(html, URL)


def test_missing_scientific_name_is_rejected():
    html = page(scientific="")
    with pytest.raises(AspcaDetailError, match="No scientific name"):
        parse_detail_page(html, URL)


def test_bytes_input_is_decoded_as_utf8():
    html = page(name="Café Lily").encode("utf-8")
    assert parse_detail_page(html, URL).name == "Café Lily"


# --- parse_species_list ----------------------------------------------------


def test_parse_species_list_splits_and_lowercases():
    assert parse_species_list("Toxic to Dogs, Toxic to Cats") == (
        "dogs", "cats",
    )
    assert parse_species_list("Non-Toxic to Horses") == ("horses",)
    assert parse_species_list("") == ()


def test_parse_species_list_rejects_unexpected_wording():
    with pytest.raises(AspcaDetailError, match="Unexpected toxicity phrase"):
        parse_species_list("Mildly toxic to Cats")


# --- fetch_detail_html: caching and crawl-delay -----------------------------


class FakeResponse:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(aspca_detail.time, "sleep", calls.append)
    return calls


def test_cached_page_is_served_without_touching_the_network(
    tmp_path, monkeypatch
):
    cache_dir = tmp_path / "details"
    cache_dir.mkdir()
    (cache_dir / "adam-and-eve.html").write_bytes(b"cached")

    def boom(*args, **kwargs):
        raise AssertionError("network must not be used when cached")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    body = aspca_detail.fetch_detail_html(
        URL, "adam-and-eve", delay=10, cache_dir=cache_dir
    )
    assert body == b"cached"


def test_uncached_page_sleeps_the_delay_then_fetches_and_caches(
    tmp_path, monkeypatch, sleeps
):
    def fake_urlopen(request, timeout=None):
        assert request.get_header("User-agent") == aspca_detail.USER_AGENT
        return FakeResponse(page().encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    cache_dir = tmp_path / "details"
    body = aspca_detail.fetch_detail_html(
        URL, "adam-and-eve", delay=10, cache_dir=cache_dir
    )
    assert sleeps == [10]
    assert (cache_dir / "adam-and-eve.html").read_bytes() == body


def test_refresh_redownloads_even_when_cached(tmp_path, monkeypatch, sleeps):
    cache_dir = tmp_path / "details"
    cache_dir.mkdir()
    (cache_dir / "adam-and-eve.html").write_bytes(b"old")

    monkeypatch.setattr(
        urllib.request, "urlopen",
        lambda request, timeout=None: FakeResponse(b"new"),
    )
    body = aspca_detail.fetch_detail_html(
        URL, "adam-and-eve", delay=0, cache_dir=cache_dir, refresh=True
    )
    assert body == b"new"
    assert (cache_dir / "adam-and-eve.html").read_bytes() == b"new"


# --- scrape_all: robots-once, per-page failures, incremental writes --------


def entry(slug):
    return PlantRef(
        url=f"https://www.aspca.org/pet-care/aspca-poison-control/"
            f"toxic-and-non-toxic-plants/{slug}",
        slug=slug,
    )


def test_robots_is_checked_once_for_the_whole_batch(monkeypatch, tmp_path):
    calls = []

    def fake_robots(url):
        calls.append(url)
        return True, 0

    monkeypatch.setattr(aspca_detail, "_robots", fake_robots)
    monkeypatch.setattr(
        aspca_detail, "fetch_detail_html",
        lambda url, slug, delay, cache_dir, refresh: page().encode(),
    )
    entries = [entry("a"), entry("b"), entry("c")]
    aspca_detail.scrape_all(entries, out=tmp_path / "out.json")
    assert len(calls) == 1


def test_robots_disallow_raises_before_fetching_anything(monkeypatch):
    monkeypatch.setattr(aspca_detail, "_robots", lambda url: (False, 0))

    def boom(*a, **k):
        raise AssertionError("must not fetch when robots disallows")

    monkeypatch.setattr(aspca_detail, "fetch_detail_html", boom)
    with pytest.raises(PermissionError):
        aspca_detail.scrape_all([entry("a")])


def test_a_bad_page_is_recorded_not_raised_and_the_rest_still_run(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(aspca_detail, "_robots", lambda url: (True, 0))

    def fetch(url, slug, delay, cache_dir, refresh):
        if slug == "bad":
            return b"<html><body>not a plant page</body></html>"
        return page(name=slug).encode()

    monkeypatch.setattr(aspca_detail, "fetch_detail_html", fetch)
    entries = [entry("good1"), entry("bad"), entry("good2")]
    doc = aspca_detail.scrape_all(entries, out=tmp_path / "out.json")

    assert doc["counts"] == {"ok": 2, "failed": 1, "total": 3}
    assert doc["failed"][0]["slug"] == "bad"
    assert {p["slug"] for p in doc["plants"]} == {"good1", "good2"}


def test_output_is_written_incrementally(monkeypatch, tmp_path):
    monkeypatch.setattr(aspca_detail, "_robots", lambda url: (True, 0))
    monkeypatch.setattr(
        aspca_detail, "fetch_detail_html",
        lambda url, slug, delay, cache_dir, refresh: page(name=slug).encode(),
    )
    out = tmp_path / "out.json"
    entries = [entry(f"p{i}") for i in range(5)]
    aspca_detail.scrape_all(entries, out=out, save_every=2)
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["counts"]["ok"] == 5
    assert "Not verified" in saved["note"]


def test_limit_only_scrapes_the_first_n(monkeypatch, tmp_path):
    monkeypatch.setattr(aspca_detail, "_robots", lambda url: (True, 0))
    seen = []

    def fetch(url, slug, delay, cache_dir, refresh):
        seen.append(slug)
        return page(name=slug).encode()

    monkeypatch.setattr(aspca_detail, "fetch_detail_html", fetch)
    entries = [entry(f"p{i}") for i in range(10)]
    aspca_detail.scrape_all(entries, out=tmp_path / "out.json", limit=3)
    assert seen == ["p0", "p1", "p2"]


def test_empty_entry_list_writes_an_empty_document(tmp_path):
    doc = aspca_detail.scrape_all([], out=tmp_path / "out.json")
    assert doc["counts"] == {"ok": 0, "failed": 0, "total": 0}


# --- load_entries: bootstrapping from the cache directory ------------------


def test_load_entries_lists_every_cached_page_sorted_by_slug(tmp_path):
    for slug in ("zinnia", "adam-and-eve", "rose"):
        (tmp_path / f"{slug}.html").write_bytes(b"<html></html>")
    entries = aspca_detail.load_entries(tmp_path)
    assert [e.slug for e in entries] == ["adam-and-eve", "rose", "zinnia"]
    assert entries[0].url == (
        "https://www.aspca.org/pet-care/aspca-poison-control/"
        "toxic-and-non-toxic-plants/adam-and-eve"
    )


def test_load_entries_ignores_non_html_files(tmp_path):
    (tmp_path / "rose.html").write_bytes(b"<html></html>")
    (tmp_path / "notes.txt").write_text("not a page")
    assert [e.slug for e in aspca_detail.load_entries(tmp_path)] == ["rose"]


def test_load_entries_survives_a_limit_run_that_truncated_the_output(
    tmp_path
):
    # The bug this guards against: --limit must never be able to make the
    # plant list itself forgettable, only the output file for that run.
    for slug in ("a", "b", "c"):
        (tmp_path / f"{slug}.html").write_bytes(b"<html></html>")
    out = tmp_path / "out.json"
    out.write_text(json.dumps({
        "plants": [{"url": "https://x/a", "slug": "a"}],  # a --limit 1 run
        "failed": [],
    }))
    assert len(aspca_detail.load_entries(tmp_path)) == 3


def test_load_entries_on_an_empty_or_missing_directory_is_empty(tmp_path):
    assert aspca_detail.load_entries(tmp_path / "does-not-exist") == []
    assert aspca_detail.load_entries(tmp_path) == []
