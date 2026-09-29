# Plant Toxicity Checker

Take a photo of a plant, identify the species, and say whether it is toxic
to **cats**.

> Working title. The Python package is `plant_toxicity_checker`; renaming
> is a one-line change in `pyproject.toml` plus the `src/` folder.

## Why

Some very common houseplants and cut flowers (true lilies and daylilies
especially) can cause fatal kidney failure in cats from tiny amounts of
contact, and this often isn't obvious. This is a safety tool for a home
with two cats that chew plants, and a from-scratch look at species
classification + class imbalance + a toxicity-aware training objective,
still respecting a hard rule: never collapse the problem into a single
"toxic vs non-toxic" model.

## Status

Species ID is being trained on **PlantNet-300K** (1,000 species, 306k
images), with a cat-toxicity-aware training objective built from a
hand-scraped **ASPCA** dataset (982 plants, per-species dogs/cats/horses
toxicity), so that species confusions that cross the toxic/non-toxic line
are penalized more than same-side ones. `toxicity.py` and `verdict.py` are
already fully species-space-agnostic and ready to plug a classifier into;
none exists yet, so there is currently no `check`/`demo` CLI command -
`cli.py` only has `scrape-aspca-details`.

An earlier, larger exploratory track (open-vocabulary detection,
segmentation, video frame-picking - investigating multi-plant
localization in a photo) and an Oxford-102 flower classifier were both
built, evaluated, and then removed as abandoned experiments. Nothing here
depends on them.

## Architecture: two separate stages

```
photo ──► [1] species classifier (ML) ──► species + confidence
                                             │
                                             ▼
                        [2] toxicity lookup (curated table, NOT ML)
                                             │
                                             ▼
                          verdict: TOXIC / CAUTION / LIKELY SAFE
```

1. **Species identification** - image -> species name + confidence
   distribution. Currently unimplemented: the PlantNet-300K classifier is
   Phase 1 of the Roadmap below. Whatever model lands here must predict
   *species*, never toxic/non-toxic directly.
2. **Toxicity lookup** (`toxicity.py`) - species -> toxic / non-toxic /
   unknown + severity + symptoms, from a hand-curated JSON table.
   Species-space-agnostic: `load_table(path, valid_species=...)` accepts an
   optional species universe to validate against.
3. **Verdict** (`verdict.py`) - combines the two under the safety rule
   below.

We deliberately do **not** train a single "toxic vs non-toxic" classifier:
toxicity is a property of species/genus, not a visual pattern, so an
end-to-end model would not generalise. A cost-sensitive *training
objective* (Phase 3) still only ever predicts species - it just penalizes
some species-confusion mistakes more than others.

## Critical safety rule

**Unknown species or low confidence means "assume toxic", never "assume
safe."** A false alarm costs a bit of caution; a false "it's fine" can kill a
cat. Concretely, `LIKELY_SAFE` is returned only if **all** hold:

- the top match has confidence ≥ 80 % (`config.MIN_CONFIDENCE`), and
- the top match *and every plausible alternative* (≥ 5 %,
  `config.CANDIDATE_FLOOR`) have an entry with `toxic_to_cats = "no"` **and**
  `verified = true`.

Everything else is `CAUTION`; a toxic top match is `TOXIC` even at low
confidence. Unverified or missing entries never count as safe (an unverified
"toxic" still counts as toxic). These rules are covered by
`tests/test_verdict.py`.

The rule cannot be weakened from the outside:

- `assess()`'s `min_confidence` can only *raise* the 80 % threshold and the
  candidate floor can only be *lowered*; to change the defaults, edit
  `config.py`.
- The safety check always sees the classifier's **full** distribution; any
  future `--top-k` only controls how many candidates are *displayed*.
- A truncated prediction list that could hide a plausible toxic candidate is
  never accepted as safe.
- The verdict is labelled "likely safe **for the main plant only**", and
  every report carries the note that other plants in the photo were not
  checked.

### Known limitations (read these)

- **No trained species classifier yet.** PlantNet-300K training is Phase 1
  of the Roadmap, not yet built. Until it lands, there's no `check`/`demo`
  command - `pipeline.check_image()` and `verdict.assess()` are fully
  implemented and tested against a fake classifier, ready to plug a real
  one into.
- **No shipped toxicity table yet either.** The real table
  (`data/plantnet_toxicity.json`, per the Roadmap) is a hand-verified
  subset of the scraped ASPCA data - see "Scraping each ASPCA plant's own
  detail page" below for the (unverified) raw material.
- **Photo of a single dominant plant only** is the intended v1 scope. A
  photo with multiple plants in frame would only have the dominant one
  assessed.
- Not veterinary advice. If a cat has chewed any lily, call a vet
  immediately.

## Data & models

| What | Source | Notes |
|---|---|---|
| Species images | [PlantNet-300K-v2](https://zenodo.org/records/10419064) | 1,000 species, 306,087 images, 39.2 GB extracted, `data/raw/plantnet/` (git-ignored) |
| Toxicity | your own scrape of [ASPCA's per-plant pages](https://www.aspca.org/pet-care/animal-poison-control/cats-plant-list) | `data/raw/aspca/cats_plant_details.json` (git-ignored, first draft, unverified) - see below |

**Label order gotcha:** whatever species-name list Phase 1's classifier
ends up using, everything downstream must be keyed by species *name*,
never by class index - checkpoint/dataset label orders are not guaranteed
to match each other. `plantnet_labels.py` (Phase 1, not built yet) will
build the canonical name/order from `data/raw/plantnet/species_metadata.csv`.

## Toxicity table schema

`toxicity.py`'s schema is species-space-agnostic:
`load_table(path, valid_species=None)` validates every entry, and
optionally rejects any `species` not in a caller-supplied `valid_species`
collection. No table is currently shipped; the example below is
illustrative only.

```json
{
  "species": "Lilium lancifolium",
  "scientific_name": "Lilium lancifolium",
  "toxic_to_cats": "yes",
  "severity": "life_threatening",
  "symptoms": ["vomiting", "lethargy", "acute kidney failure"],
  "sources": [
    "https://www.aspca.org/pet-care/aspca-poison-control/toxic-and-non-toxic-plants/tiger-lily"
  ],
  "verified": true,
  "notes": "Any part; even pollen/vase water."
}
```

| Field | Values |
|---|---|
| `species` | A species name; checked against `valid_species` if one is passed to `load_table` |
| `toxic_to_cats` | `"yes"` / `"no"` / `"unknown"` |
| `severity` | `null`, `"none"`, `"mild"`, `"moderate"`, `"severe"`, `"life_threatening"` |
| `symptoms` | list of strings (must be empty for `"no"`) |
| `sources` | list of URLs / citations. Cross-check a second source |
| `verified` | a real JSON `true`/`false` (strings like `"false"` are rejected). `true` only after a human cross-check; requires a known `toxic_to_cats`, ≥ 1 source, and (if toxic) a severity |
| `notes` | free text, e.g. common name vs. genus caveats |

`severity` is **your own judgement**, not ASPCA data: ASPCA gives toxic /
non-toxic and clinical signs but no severity grades. Record what you based it
on in `notes`.

`load_table()` enforces the rules above and rejects contradictory entries
(`"no"` with a severity or symptoms, `"yes"` with severity `"none"`,
`"unknown"` with any severity), wrong types, and (if `valid_species` is
given) unknown species.

### Scraping each ASPCA plant's own detail page (first draft only)

`python -m plant_toxicity_checker scrape-aspca-details` visits each ASPCA
plant's own page (one page per plant) and writes
`data/raw/aspca/cats_plant_details.json` (git-ignored):
`additional_common_names`, `scientific_name`, `family`, `toxic_principles`,
`clinical_signs`, and `toxic_to` / `non_toxic_to` - the species named in
that page's own separate "Toxicity:" and "Non-Toxicity:" fields. ASPCA
states non-toxicity per species explicitly; it is not silence, so
`AspcaDetail.toxic_to_species("cats")` returns `"yes"`/`"no"`/`"unknown"`
("unknown" only when a species is named in neither field) rather than
guessing from an absence. `--limit N` scrapes only the first N plants, for a
quick check.

The list of plants to (re-)scrape comes from `load_entries()`, which scans
the **cache directory** `data/raw/aspca/details/` (git-ignored, one cached
HTML page per plant slug, kept forever once fetched) rather than the output
JSON: the cache only ever grows, so it survives a `--limit` test run
untouched, unlike the output file, which a limited run legitimately
truncates. That cache directory is therefore the only durable record of
which ASPCA plant pages this project knows about - it lives only on disk,
not in git, so losing it means starting over.

983 pages at ASPCA's own `robots.txt` `Crawl-delay` (10s at last check) is
about 2.7 hours. `robots.txt` is checked once for the whole run, not once
per page; a plant whose page is already cached is never re-fetched
(`--refresh` forces it), so an interrupted or `--limit`-ed run resumes
cheaply. A plant that fetched fine but failed to *parse* is cached too, and
will fail the same way again until the page itself changes; a bad page is
recorded in the output's `failed` list rather than aborting the whole run.
First real run: 982 of 983 plants ok, 1 genuine failure - "Medicine Plant" is
another ASPCA listing for the already-captured *Aloe vera* (aliases: Aloe,
True Aloe, Barbados Aloe) whose page is simply missing its scientific-name
field; the parser refuses to guess one.

The output is **unverified**: nothing here may be copied into a toxicity
table as `verified: true` without a human check.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
pip install -e .
```

### Run

```bash
python -m plant_toxicity_checker scrape-aspca-details   # each ASPCA plant's own page (first draft, ~2.7h)
```

Species-ID commands (`train-plantnet`, `check`, ...) land with Phase 1 of
the Roadmap.

### Notebooks

```bash
pip install -r requirements-notebook.txt   # matplotlib, pandas, networkx, ipykernel, ...
```

Open a notebook and pick the project's `.venv` as the kernel. The heavy
dependencies are kept out of `requirements.txt` so the Docker image and CI
stay small.

`notebooks/01_plantnet_class_distribution.ipynb` looks at whether
PlantNet-300K (`data/raw/plantnet/`, 1000 species, 306,087 images) is
actually a good foundation to train a species classifier on, by matching
every one of its species against `data/raw/aspca/cats_plant_details.json`
(982 plants) using `src/plant_toxicity_checker/species_match.py`:
scientific-name-only matching (`SpeciesIndex`), exact (same genus +
species) or genus-only (a relative is known, not necessarily the same
species) - never by common name, to avoid the "lily problem" (*Lilium*,
*Hemerocallis*, *Hippeastrum* and *Spathiphyllum* are all called "lily" in
English and are four different genera with different toxicity).

Result: PlantNet's class-size distribution is exactly as imbalanced as its
own README warns (mean 306 images/species, median 28; 43% of species have
fewer than 20 images at all), and the 34 species that happen to have an
**exact** ASPCA match are not a random sample of that distribution - they
cluster at about 5x the mean, because the plants ASPCA catalogues *and*
PlantNet's citizen scientists heavily photograph are the same kind of
common, easily-recognised weed or garden escapee. Genus-only matches show
no such bias. Even so, 40% of the ASPCA-matched species are themselves
under the 20-image floor. A family -> genus taxonomy graph (`networkx`) and
a worked genus-fallback toxicity lookup are also in the notebook - the
lookup is the same `SpeciesIndex` built the other way round (from ASPCA,
matched against), directly reusable for a real verdict function later:
`Lilium bulbiferum` (no exact ASPCA entry) correctly resolves to "could be
toxic" from its toxic *Lilium* relatives, while `Rosa canina` (a genus known
only via non-toxic relatives) correctly refuses to say safe. This notebook
is the direct precedent for Phase 1/3's `plantnet_aspca_sides.csv`.

### Test & lint

```bash
pytest               # fast, offline
ruff check .          # line length 79
```

### Docker

```bash
docker build -t plant-toxicity-checker .
docker run --rm plant-toxicity-checker         # runs pytest
docker compose run --rm test
docker compose run --rm cli --help
```

The image uses CPU-only PyTorch. Put photos in `./photos`; the data and
model caches are mounted from the host so they survive rebuilds.

CI (`.github/workflows/ci.yml`) runs `ruff` and builds the Docker image and
runs `pytest` inside it, on every push and PR.

## Remote training

Training runs on a home PC's GPU over SSH, not on this laptop - see
`scripts/remote/` and CLAUDE.md's "Remote training" section.

## Repo layout

```
src/plant_toxicity_checker/   toxicity table, verdict, pipeline, CLI,
                               species matching, ASPCA scraper
data/raw/                      downloaded/scraped datasets (git-ignored):
                               aspca/ (scraped toxicity), plantnet/ (species images)
data/processed/                (planned, Phase 1) small committed derived tables,
                               e.g. plantnet_aspca_sides.csv
configs/                       (planned, Phase 1) one JSON file per training run
models/                        model caches/checkpoints (git-ignored)
scripts/remote/                remote GPU training tooling (see CLAUDE.md)
notebooks/                     exploratory notebooks (see above)
tests/                         pytest suite, offline by default
```

## Roadmap

1. **PlantNet-300K baseline.** A real PyTorch training pipeline
   (`plantnet_labels.py`, `plantnet_dataset.py`, `metrics.py`, `train.py`),
   replicating the official paper's ResNet-18 baseline setup (GPU:
   GTX 1070, via `scripts/remote/`), with per-epoch resumable
   checkpointing. `train-plantnet` / `evaluate-plantnet` CLI commands.
2. **Imbalance-aware variants.** Class-Balanced Loss / Balanced Softmax /
   decoupled retraining as additional configs through the same training
   script, compared side by side.
3. **Toxicity-aware training.** A cost-sensitive loss term built from
   `species_match.SpeciesIndex` matching PlantNet's full 1,000-species
   label space against the scraped ASPCA data (`data/processed/
   plantnet_aspca_sides.csv`), penalizing toxic <-> non-toxic species
   confusions more than same-side ones - still only ever predicting
   species. A hand-verified subset of ASPCA data (starting from the ~34
   exact-match, well-photographed species already identified in notebook
   01) becomes the first real `verified: true` entries.
4. **Results.** A `README.md` results table and programmatically selected
   correct-catch and failure-case examples (same-genus confusion, sparse
   training class, wrong plant organ in frame), not hand-picked. `cli.py
   check` then becomes a real command.
