"""Paths and default settings."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("PLANT_DATA_DIR", REPO_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("PLANT_MODELS_DIR", REPO_ROOT / "models"))

DATASET_DIR = DATA_DIR / "raw"
ASPCA_DIR = DATASET_DIR / "aspca"
ASPCA_DETAILS_JSON_PATH = ASPCA_DIR / "cats_plant_details.json"
ASPCA_DETAIL_CACHE_DIR = ASPCA_DIR / "details"
PLANTNET_DIR = DATASET_DIR / "plantnet"
PROCESSED_DIR = DATA_DIR / "processed"
PLANTNET_ASPCA_SIDES_PATH = PROCESSED_DIR / "plantnet_aspca_sides.csv"

# The real, hand-curated toxicity table (toxicity.load_table's schema) -
# seeded from the ASPCA scrape (toxicity_seed.py) but the only thing
# anything should match PlantNet species against from here on; see
# species_match.match_against_toxicity_table.
TOXICITY_TABLE_PATH = DATA_DIR / "toxicity.json"

# Safety thresholds (see verdict.py). Erring high is deliberate: a low
# threshold only ever risks a false "it's fine".
MIN_CONFIDENCE = 0.80
CANDIDATE_FLOOR = 0.05
