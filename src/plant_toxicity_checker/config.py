"""Paths and default settings."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("PLANT_DATA_DIR", REPO_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("PLANT_MODELS_DIR", REPO_ROOT / "models"))

DATASET_DIR = DATA_DIR / "raw"
ASPCA_DIR = DATASET_DIR / "aspca"
ASPCA_DETAILS_JSON_PATH = ASPCA_DIR / "cats_plant_details.json"
ASPCA_DETAIL_CACHE_DIR = ASPCA_DIR / "details"
PLANTNET_DIR = DATASET_DIR / "plantnet"
PROCESSED_DIR = DATA_DIR / "processed"
PLANTNET_ASPCA_SIDES_PATH = PROCESSED_DIR / "plantnet_aspca_sides.csv"

# Safety thresholds (see verdict.py). Erring high is deliberate: a low
# threshold only ever risks a false "it's fine".
MIN_CONFIDENCE = 0.80
CANDIDATE_FLOOR = 0.05
