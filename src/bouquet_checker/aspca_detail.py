"""Scrape each ASPCA plant's own detail page.

One page per plant: `additional_common_names`, `scientific_name`, `family`,
`toxic_principles`, `clinical_signs`, and toxicity to dogs/cats/horses
(separately - see ``AspcaDetail.toxic_to_species()``).

983 pages at ASPCA's own ``robots.txt`` ``Crawl-delay`` (10s at last check)
is roughly 2.7 hours, so this module is built to survive being interrupted
and to never lose progress on one bad page:

* ``robots.txt`` is checked **once** for the whole run, not once per page;
* every plant's page is cached individually (by slug), so an interrupted
  run resumes without re-fetching anything already on disk;
* a per-plant fetch/parse failure is recorded and the run keeps going,
  instead of raising and losing everything, and the output JSON is written
  incrementally so a crash loses at most ``save_every`` plants' worth.

``load_entries()`` bootstraps the list of plants to (re-)scrape from the
**cache directory** (one cached page per plant, ever fetched), not from the
output JSON: the cache only ever grows, so it survives a ``--limit`` test
run untouched, unlike the output file, which a limited run legitimately
truncates. The cache directory is therefore the only durable record of the
983 ASPCA URLs this project uses - it is git-ignored, so it lives only on
disk, not in version control.

Each plant's page has a "Toxicity:" field (species it *is* toxic to) and,
separately, a "Non-Toxicity:" field (species it is confirmed *not* toxic
to) - ASPCA states non-toxicity explicitly, it is not silence. A species
named in neither field is genuinely unaddressed by ASPCA for that plant, so
``AspcaDetail.toxic_to_species()`` returns "unknown" for it rather than
guessing either way.

FIRST DRAFT ONLY: nothing here is verified, and nothing here may be copied
into ``data/toxicity.json`` without a human cross-check.
"""

import json
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from bouquet_checker.config import (
    ASPCA_DETAIL_CACHE_DIR,
    ASPCA_DETAILS_JSON_PATH,
)

USER_AGENT = (
    "bouquet-checker/0.1 (personal research; one cached request per plant)"
)
DETAIL_URL_TEMPLATE = (
    "https://www.aspca.org/pet-care/aspca-poison-control/"
    "toxic-and-non-toxic-plants/<slug>"
)
SPECIES = ("dogs", "cats", "horses")

_DRAFT_NOTE = (
    "First-draft scrape of each ASPCA plant's own detail page (toxic "
    "principles, clinical signs, and toxicity split by dogs/cats/horses). "
    "Not verified. toxic_to_dogs/cats/horses is 'yes' if that species is "
    "named in the page's Toxicity field, 'no' if named in its Non-Toxicity "
    "field, and 'unknown' if named in neither (ASPCA does not address that "
    "species for this plant) - never inferred from silence."
)


class AspcaDetailError(ValueError):
    """A detail page did not look like we expected; refuse to guess."""


@dataclass(frozen=True)
class PlantRef:
    """Just enough to (re-)fetch one plant: where its page is."""

    url: str
    slug: str


@dataclass(frozen=True)
class AspcaDetail:
    """One plant's own ASPCA detail page."""

    name: str
    url: str
    additional_common_names: tuple[str, ...]
    scientific_name: str
    family: str
    toxic_principles: str
    clinical_signs: str
    toxic_to: tuple[str, ...]      # species named in the "Toxicity:" field
    non_toxic_to: tuple[str, ...]  # species named in "Non-Toxicity:"

    @property
    def slug(self):
        """Last URL path segment, e.g. ``adam-and-eve``."""
        return urlparse(self.url).path.rstrip("/").rsplit("/", 1)[-1]

    def toxic_to_species(self, species):
        """"yes" / "no" / "unknown" tri-state for one of ``SPECIES``."""
        if species in self.toxic_to:
            return "yes"
        if species in self.non_toxic_to:
            return "no"
        return "unknown"


def _split_top_level(text):
    """Split on commas that are not inside parentheses."""
    parts, depth, current = [], 0, []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return tuple(p.strip() for p in parts if p.strip())


def parse_species_list(raw):
    """``"Toxic to Dogs, Toxic to Cats"`` -> ``("dogs", "cats")``.

    Each item must start with "Toxic to " or "Non-Toxic to "; anything else
    means the page's wording has changed under us.
    """
    species = []
    for phrase in _split_top_level(raw):
        for prefix in ("Non-Toxic to ", "Toxic to "):
            if phrase.startswith(prefix):
                species.append(phrase[len(prefix):].strip().lower())
                break
        else:
            raise AspcaDetailError(f"Unexpected toxicity phrase: {phrase!r}")
    return tuple(species)


def _field_value(soup, field_name):
    """Text of ``div.field-name-field-<field_name> .values``, or ""."""
    div = soup.find("div", class_=f"field-name-field-{field_name}")
    if div is None:
        return ""
    value = div.select_one(".values")
    return " ".join(value.get_text(" ", strip=True).split()) if value else ""


def parse_detail_page(html, url):
    """Parse one plant's detail page HTML into an ``AspcaDetail``."""
    if isinstance(html, bytes):
        html = html.decode("utf-8")
    soup = BeautifulSoup(html, "html.parser")

    h1 = soup.find("h1")
    if h1 is None:
        raise AspcaDetailError(f"No <h1> title found on {url}")
    name = " ".join(h1.get_text().split())

    scientific_name = _field_value(soup, "scientific-name")
    if not scientific_name:
        raise AspcaDetailError(
            f"No scientific name field on {url}: page layout may have "
            "changed, or this is not a real plant page"
        )

    return AspcaDetail(
        name=name,
        url=url,
        additional_common_names=_split_top_level(
            _field_value(soup, "additional-common-names")
        ),
        scientific_name=scientific_name,
        family=_field_value(soup, "family"),
        toxic_principles=_field_value(soup, "toxic-principles"),
        clinical_signs=_field_value(soup, "clinical-signs"),
        toxic_to=parse_species_list(_field_value(soup, "toxicity")),
        non_toxic_to=parse_species_list(_field_value(soup, "non-toxicity")),
    )


def _robots(url):
    """Return (allowed, crawl_delay_seconds) for ``url`` per robots.txt."""
    root = "{0.scheme}://{0.netloc}".format(urlparse(url))
    request = urllib.request.Request(
        root + "/robots.txt", headers={"User-Agent": USER_AGENT}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        lines = response.read().decode("utf-8", "replace").splitlines()
    parser = robotparser.RobotFileParser()
    parser.parse(lines)
    return parser.can_fetch(USER_AGENT, url), parser.crawl_delay(USER_AGENT)


def fetch_detail_html(url, slug, delay, cache_dir=ASPCA_DETAIL_CACHE_DIR,
                      refresh=False):
    """Return one plant page's bytes, downloading once and caching by slug."""
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{slug}.html"
    if cache_path.exists() and not refresh:
        return cache_path.read_bytes()
    time.sleep(delay or 0)  # honour the site's requested Crawl-delay
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        body = response.read()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(body)
    return body


def load_entries(cache_dir=ASPCA_DETAIL_CACHE_DIR):
    """The plants to (re-)scrape: one per cached page under ``cache_dir``.

    The cache directory, not the output JSON, is the stable record of which
    plants exist: it only ever grows (a page is cached the moment it is
    fetched, whether or not it goes on to parse), so it survives a
    ``--limit`` test run untouched - unlike the output JSON, which a
    ``--limit`` run legitimately (and previously, dangerously) truncates.
    """
    cache_dir = Path(cache_dir)
    slugs = sorted(p.stem for p in cache_dir.glob("*.html"))
    return [
        PlantRef(url=DETAIL_URL_TEMPLATE.replace("<slug>", slug), slug=slug)
        for slug in slugs
    ]


def write_json(details, failed, out=ASPCA_DETAILS_JSON_PATH,
               retrieved_at=None):
    """Write the scraped details plus provenance and failures to ``out``."""
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    )
    document = {
        "note": _DRAFT_NOTE,
        "source": DETAIL_URL_TEMPLATE,
        "retrieved_at": retrieved_at,
        "counts": {
            "ok": len(details),
            "failed": len(failed),
            "total": len(details) + len(failed),
        },
        "failed": list(failed),
        "plants": [{**asdict(d), "slug": d.slug} for d in details],
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return document


def scrape_all(entries, out=ASPCA_DETAILS_JSON_PATH,
               cache_dir=ASPCA_DETAIL_CACHE_DIR, refresh=False, limit=None,
               save_every=25, on_progress=None):
    """Fetch and parse each entry's own page; write ``out`` as it goes.

    ``entries`` is any sequence of objects with ``.url``/``.slug`` (see
    ``load_entries`` / ``PlantRef``). A per-plant fetch/parse failure is
    recorded in the output's ``failed`` list rather than raised, so one bad
    page cannot lose the rest of a multi-hour run. Returns the final
    written document.
    """
    if limit is not None:
        entries = list(entries)[:limit]
    if not entries:
        return write_json([], [], out)

    allowed, delay = _robots(entries[0].url)
    if not allowed:
        raise PermissionError(
            f"robots.txt disallows fetching {entries[0].url}"
        )

    details, failed, document = [], [], None
    for n, entry in enumerate(entries, 1):
        try:
            body = fetch_detail_html(
                entry.url, entry.slug, delay, cache_dir, refresh
            )
            details.append(parse_detail_page(body, entry.url))
        except Exception as err:  # noqa: BLE001 - record, never abort the run
            failed.append({
                "url": entry.url, "slug": entry.slug,
                "error": f"{type(err).__name__}: {err}",
            })
        if on_progress:
            on_progress(n, len(entries), len(details), len(failed))
        if n % save_every == 0 or n == len(entries):
            document = write_json(details, failed, out)
    return document
