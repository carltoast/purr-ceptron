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

# The shipped species-ID model: the ConvNeXt V2-Tiny run, trained with
# square-root-frequency sampling for the class imbalance and the
# toxicity-aware cost term (see notebooks/05_convnextv2_training_
# progress.ipynb). Epoch 30's EMA weights, chosen on val and reported
# on test: 81.9% top-1, 42.6% macro-average top-1, 0.27% toxic/
# non-toxic confusion. One place this decision lives, so picking a
# different run later is a one-line change here, not a hunt through
# callers.
DEFAULT_RUN = "plantnet_convnextv2_tiny_toxicity_v2"

# The Phase 1 backbone plus a separately trained classifier head, from
# the decoupled-training experiments (notebooks/04_decoupled_training_
# comparison.ipynb). Not what ships - kept so those notebooks and the
# `check --head-run` path still have a named default to point at.
DEFAULT_BACKBONE_RUN = "plantnet-baseline"
DEFAULT_HEAD_RUN = "decoupled_ce_toxicity_w20.0"
