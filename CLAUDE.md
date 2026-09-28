# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

Bouquet Toxicity Checker: photo -> plant species -> cat toxicity
(hand-curated JSON table). Species ID is moving to a classifier trained on
PlantNet-300K (1,000 species), with a cat-toxicity-aware training objective
built from a hand-scraped ASPCA dataset (see README.md's Roadmap); the
previous Oxford-102 classifier/label list/dataset were removed rather than
kept as a parallel path. See README.md for status, plan, limitations and
the table schema.

## Non-negotiable safety rules

- Stage 1 (species, ML) and stage 2 (toxicity, curated table) stay
  separate. Never train a direct "toxic vs non-toxic" model. A
  cost-sensitive *training objective* (weighting some species-confusion
  mistakes more than others, e.g. by ASPCA toxic/non-toxic side) is fine;
  the model's output must still be a species, never toxic/non-toxic.
- Unknown / unverified / missing / low-confidence must resolve to
  caution or toxic, never safe. `LIKELY_SAFE` requires a confident top match
  AND every plausible alternative having a *verified* `"no"` entry
  (`src/bouquet_checker/verdict.py`). Any change to that logic needs tests
  in `tests/test_verdict.py`.
- Callers must not be able to weaken the safety rule: `assess()` clamps
  thresholds to the `config` defaults (stricter only), `check_image()` always
  requests the classifier's full distribution, and `verified` in any
  toxicity table must be a real boolean. Keep tests for each.
- `toxicity.py`'s schema is species-space-agnostic (`load_table(path,
  valid_species=None)`); do not hardcode a specific species list back into
  it (e.g. a global constant like the old `OXFORD_102_CLASSES`) - pass
  `valid_species` in from whatever `labels`-style module is current.
- Do not fill in or "improve" a toxicity table from memory or a single
  source. It must be curated by hand from ASPCA (or another real source);
  entries you did not verify stay `unknown` / `verified: false`.
- `aspca_detail.py`'s scrape (`data/raw/aspca/cats_plant_details.json`) is a
  first draft, never an input to the safety path directly: it must not
  write a shipped toxicity table or set `verified` itself. It may inform a
  training-time cost/loss weighting or offline analysis, but that is a
  different, non-safety-path use from being `verified: true` in a table
  `verdict.assess()` consumes. Never infer "safe" for a species from a
  related one.
- Downloaded videos (`data/raw/videos/`) belong to their creators: keep them
  git-ignored, never commit or redistribute them or frames from them, and
  keep the provenance sidecar next to each one.
- Ultralytics (YOLO26 / YOLOE / FastSAM / MobileSAM loading) is AGPL-3.0: keep it notebook-only, never add
  it to `requirements.txt` or the Docker image without a licensing decision.
  Install it with `--no-deps` (see README) and keep its weights in
  `models/ultralytics/`. SAM 3 is gated on Hugging Face; never work around it.
- Everything is keyed by species **name**, never class index (checkpoint,
  dataset and torchvision label orders are not guaranteed to match each
  other). PlantNet's `species_id` is a numeric code, not a stable name -
  any label-order module for it (e.g. `plantnet_labels.py`) must key by
  name the same way the removed `labels.py` did for Oxford-102.

## Commands

```bash
pip install -r requirements.txt && pip install -e .
python -m bouquet_checker scrape-aspca-details
pytest                  # offline
ruff check .            # line length 79, D100 module docstrings required
docker build -t bouquet-checker . && docker run --rm bouquet-checker
```

Species-ID / training commands (`train-plantnet`, `evaluate-plantnet`,
`check`, ...) land with the PlantNet training pipeline (README Roadmap
Phase 1+); they do not exist yet.

CI runs `ruff` and builds the Docker image and runs `pytest` inside it.

## Code style

79-char lines. Module docstrings are required (`D100`); per-function
docstrings in tests are not (descriptive test names instead). torch /
transformers are imported lazily so the table, verdict logic and tests run
without loading them.
