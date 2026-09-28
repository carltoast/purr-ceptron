"""Paths and default settings."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("BOUQUET_DATA_DIR", REPO_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("BOUQUET_MODELS_DIR", REPO_ROOT / "models"))

DATASET_DIR = DATA_DIR / "raw"
HF_CACHE_DIR = MODELS_DIR / "hf"
ULTRALYTICS_DIR = MODELS_DIR / "ultralytics"
ASPCA_DIR = DATASET_DIR / "aspca"
VIDEO_DIR = DATASET_DIR / "videos"
PICKS_DIR = DATASET_DIR / "picks"
ASPCA_DETAILS_JSON_PATH = ASPCA_DIR / "cats_plant_details.json"
ASPCA_DETAIL_CACHE_DIR = ASPCA_DIR / "details"
PLANTNET_DIR = DATASET_DIR / "plantnet"
PROCESSED_DIR = DATA_DIR / "processed"
PLANTNET_ASPCA_SIDES_PATH = PROCESSED_DIR / "plantnet_aspca_sides.csv"

# Safety thresholds (see verdict.py). Erring high is deliberate: a low
# threshold only ever risks a false "it's fine".
MIN_CONFIDENCE = 0.80
CANDIDATE_FLOOR = 0.05
