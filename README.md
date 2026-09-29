# Plant Toxicity Checker

Identify a plant species from a photo and say whether it's toxic to cats.

Some common houseplants and cut flowers (true lilies and daylilies
especially) can cause fatal kidney failure in cats from minimal contact.
This project pairs a species classifier with a hand-curated toxicity
table to flag that risk from a photo.

## How it works

```
photo ──► [1] species classifier (ML) ──► species + confidence
                                             │
                                             ▼
                        [2] toxicity lookup (curated table, NOT ML)
                                             │
                                             ▼
                          verdict: TOXIC / CAUTION / LIKELY SAFE
```

Two separate stages, deliberately never merged into one "toxic vs
non-toxic" model — toxicity is a property of species/genus, not a visual
pattern.

1. **Species ID** — image → species name + confidence distribution.
2. **Toxicity lookup** (`toxicity.py`) — species → toxic / non-toxic /
   unknown + severity + symptoms, from a curated JSON table.
3. **Verdict** (`verdict.py`) — combines the two under the safety rule
   below.

**Safety rule:** unknown species or low confidence means assume toxic,
never assume safe. `LIKELY_SAFE` requires a confident top match *and*
every plausible alternative to have a verified non-toxic entry; thresholds
can only be tightened by a caller, never loosened. See
`tests/test_verdict.py` for the full rule set.

## Status

Species ID is being trained on **PlantNet-300K** (1,000 species, 306k
images), with a cat-toxicity-aware training objective built from a
hand-scraped **ASPCA** dataset (982 plants, per-species dogs/cats/horses
toxicity) — species confusions that cross the toxic/non-toxic line are
penalized more than same-side ones, while the model still only ever
predicts species. `toxicity.py` and `verdict.py` are ready to plug a
classifier into; none exists yet, so there's no `check` command yet —
`cli.py` currently only has `scrape-aspca-details`.

## Data

| What | Source |
|---|---|
| Species images | [PlantNet-300K-v2](https://zenodo.org/records/10419064) — 1,000 species, 306k images, `data/raw/plantnet/` (git-ignored) |
| Toxicity | your own scrape of [ASPCA's per-plant pages](https://www.aspca.org/pet-care/animal-poison-control/cats-plant-list) — `data/raw/aspca/` (git-ignored, unverified first draft) |

`python -m plant_toxicity_checker scrape-aspca-details` scrapes each ASPCA
plant's own page (toxic/non-toxic per species, principles, clinical
signs). Resumable and rate-limited to ASPCA's `robots.txt` crawl-delay
(~2.7h for all 983 pages). The output is unverified — never copied into a
shipped toxicity table as `verified: true` without a human check.

## Toxicity table schema

`toxicity.py` is species-space-agnostic: `load_table(path,
valid_species=None)` validates entries and optionally checks `species`
against a caller-supplied set. No table is shipped yet.

```json
{
  "species": "Lilium lancifolium",
  "scientific_name": "Lilium lancifolium",
  "toxic_to_cats": "yes",
  "severity": "life_threatening",
  "symptoms": ["vomiting", "lethargy", "acute kidney failure"],
  "sources": ["https://www.aspca.org/..."],
  "verified": true,
  "notes": "Any part; even pollen/vase water."
}
```

`toxic_to_cats` is `yes`/`no`/`unknown`; `verified` is only `true` after a
human cross-check and requires a source and (if toxic) a severity.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
pip install -e .
```

```bash
python -m plant_toxicity_checker scrape-aspca-details
pytest                            # offline
ruff check .
docker build -t plant-toxicity-checker . && docker run --rm plant-toxicity-checker
```

Training runs on a home PC's GPU over SSH, not locally — see
`scripts/remote/`.

### Notebooks

```bash
pip install -r requirements-notebook.txt
```

`notebooks/01_plantnet_class_distribution.ipynb` matches PlantNet-300K's
1,000 species against the scraped ASPCA data by scientific name (exact
genus+species, or genus-only — never by common name, to avoid the "lily
problem": *Lilium*, *Hemerocallis*, *Hippeastrum* and *Spathiphyllum* are
all "lily" in English, four different genera with different toxicity).
Findings: PlantNet is strongly imbalanced (mean 306 images/species, median
28), and its 34 exact ASPCA matches skew toward well-photographed species
rather than a random sample. Also builds a genus-fallback toxicity lookup,
the precedent for Phase 3 below.

## Repo layout

```
plant_toxicity_checker/   toxicity table, verdict, pipeline, CLI,
                           species matching, ASPCA scraper
data/raw/                 downloaded/scraped datasets (git-ignored)
data/processed/           (planned) small committed derived tables
configs/                  (planned) one JSON file per training run
models/                   model caches/checkpoints (git-ignored)
scripts/remote/           remote GPU training tooling
notebooks/                exploratory notebooks
tests/                    pytest suite, offline by default
```

## Roadmap

1. **PlantNet-300K baseline** — a real training pipeline replicating the
   official paper's ResNet-18 setup, trained remotely (GTX 1070).
2. **Imbalance-aware variants** — Class-Balanced Loss / Balanced Softmax /
   decoupled retraining, compared side by side.
3. **Toxicity-aware training** — a cost-sensitive loss penalizing
   toxic↔non-toxic species confusions more than same-side ones, over the
   full label space, still only ever predicting species. A hand-verified
   ASPCA subset (starting from the ~34 exact matches in notebook 01)
   becomes the first real `verified: true` entries.
4. **Results** — a results table and programmatically selected
   correct-catch and failure-case examples, not hand-picked. `check`
   becomes a real command.
