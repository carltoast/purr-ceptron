# Bouquet Toxicity Checker

Take a photo of a flower, identify the species, and say whether it is toxic
to **cats**.

> Working title. Alternatives: *PawSafe Bouquet*, *CatSafe Blooms*,
> *Petal Patrol*. The Python package is `bouquet_checker`; renaming is a
> one-line change in `pyproject.toml` plus the `src/` folder.

## Status

This project started narrower - a classifier fine-tuned on the Oxford-102
garden-flower set - and has moved to a broader, more interesting problem:
species ID over **PlantNet-300K** (1,000 species, 306k images), trained
with a cat-toxicity-aware objective built from a hand-scraped **ASPCA**
dataset, so that species confusions that cross the toxic/non-toxic line are
penalized more than same-side ones. The old Oxford-102 classifier,
label list and dataset were removed rather than kept as a parallel legacy
path (see Roadmap): `toxicity.py` and `verdict.py` were already, or are now,
fully species-space-agnostic, so the stage-2 safety logic below carries over
unchanged onto the new species space. `cli.py` currently only has
`scrape-aspca-details`; the species-ID commands land with Phase 1 below.

The detection/segmentation/video notebooks (02-06) are a separate,
**paused** exploratory track investigating multi-flower localization in
bouquet photos - interesting computer-vision work, but not blocking or
gating the species-classification + toxicity results described in the
Roadmap.

## Why

Some very common florist flowers (true lilies and daylilies especially) can
cause fatal kidney failure in cats from tiny amounts of contact, and florists
often don't know. This is a safety tool for a home with two cats that chew
bouquets, and a from-scratch look at species classification + imbalance +
a toxicity-aware training objective, still respecting a hard rule: never
collapse the problem into a single "toxic vs non-toxic" model.

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
   optional species universe to validate against, so the same schema serves
   whatever species list stage 1 currently uses.
3. **Verdict** (`verdict.py`) - combines the two under the safety rule
   below. Also species-space-agnostic; unchanged by the move to PlantNet.

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
- The verdict is labelled "likely safe **for the main flower only**", and
  every report carries the note that other flowers in a bouquet were not
  checked.

### Known limitations (read these)

- **No trained species classifier yet.** The Oxford-102 classifier was
  removed; PlantNet-300K training is Phase 1 of the Roadmap, not yet built.
  Until it lands, there's no `check`/`demo` command - `pipeline.check_image()`
  and `verdict.assess()` are fully implemented and tested against a fake
  classifier, ready to plug a real one into.
- **No shipped toxicity table yet either.** The old 102-species placeholder
  (`data/toxicity.json`) was removed along with the Oxford-102 pipeline it
  was keyed to. The real table for the new species space
  (`data/plantnet_toxicity.json`, per the Roadmap) is a hand-verified subset
  of the scraped ASPCA data - see "Scraping each ASPCA plant's own detail
  page" below for the (unverified) raw material.
- **Photo of a single dominant flower only** is the intended v1 scope. In a
  mixed bouquet only the dominant flower would be assessed. Multi-flower
  detection is the paused stretch-goal track (notebooks 02-06).
- Not veterinary advice. If a cat has chewed any lily, call a vet
  immediately.

## Data & models

| What | Source | Notes |
|---|---|---|
| Species images | [PlantNet-300K-v2](https://zenodo.org/records/10419064) | 1,000 species, 306,087 images, 39.2 GB extracted, `data/raw/plantnet/` (git-ignored) |
| Toxicity | your own scrape of [ASPCA's per-plant pages](https://www.aspca.org/pet-care/animal-poison-control/cats-plant-list) | `data/raw/aspca/cats_plant_details.json` (git-ignored, first draft, unverified) - see below |
| (paused/reference) Oxford-102 flowers | [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | removed from the pipeline this session; `notebooks/01_explore_oxford102.ipynb`'s DINOv2+kNN exploration was retired with it, not kept as reference |

**Label order gotcha (still applies):** whatever species-name list Phase 1's
classifier ends up using, everything downstream must be keyed by species
*name*, never by class index - checkpoint/dataset label orders are not
guaranteed to match each other. `labels.py` filled this role for Oxford-102
and was removed with it; its replacement (`plantnet_labels.py`, building the
canonical name/order from `data/raw/plantnet/species_metadata.csv`) is part
of Phase 1.

## Toxicity table schema

`toxicity.py`'s schema is unchanged and species-space-agnostic:
`load_table(path, valid_species=None)` validates every entry, and
optionally rejects any `species` not in a caller-supplied `valid_species`
collection (so the same code validates an Oxford-102 table, a PlantNet
table, or an unrestricted one - see `tests/test_toxicity.py`). No table is
currently shipped; the example below is illustrative only.

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

`python -m bouquet_checker scrape-aspca-details` visits each ASPCA plant's
own page (one page per plant) and writes
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
untouched, unlike the output file, which a limited run legitimately (and
once, before this was fixed, dangerously) truncates. That cache directory is
therefore the only durable record of which ASPCA plant pages this project
knows about - it lives only on disk, not in git, so losing it means starting
over.

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
python -m bouquet_checker scrape-aspca-details   # each ASPCA plant's own page (first draft, ~2.7h)
```

Species-ID commands (`train-plantnet`, `check`, ...) land with Phase 1 of
the Roadmap.

### Notebooks

```bash
pip install -r requirements-notebook.txt   # matplotlib, pandas, scikit-learn, umap-learn, plotly, ipykernel, ...
```

Open a notebook and pick the project's `.venv` as the kernel. The heavy
dependencies are kept out of `requirements.txt` so the Docker image and CI
stay small.

`notebooks/01_explore_oxford102.ipynb` (Oxford-102 exploration: DINOv2
embeddings, kNN separability, near-duplicates) was retired with the
Oxford-102 pipeline it supported.

### Video and open-vocabulary detection (paused stretch-goal track)

`notebooks/02_video_open_vocab_detection.ipynb` takes a real flower-shop
vlog and asks whether a detector that was **never trained on flowers** can
find them from the text prompt "flower". It compares Grounding DINO tiny,
MM-Grounding-DINO tiny and OWLv2 base **on CPU**, one at a time: speed,
boxes per frame, false alarms on street scenes, prompt wording, thresholds and
box sizes, then runs the best trade-off over a clip and writes an annotated
mp4.

Reusable code: `src/bouquet_checker/video.py` (download once with yt-dlp,
read frames with OpenCV) and `detection.py` (one wrapper over the three
model families, plus NMS and box drawing). All three models are Apache-2.0
and run through `transformers`; nothing needs a GPU or ffmpeg.

Headline result on this laptop (CPU only, no GPU): Grounding DINO tiny with
the prompt "flower head" isolates individual flowers best but is noisiest on
non-flower scenes; OWLv2 base with "flower" is quieter but tends to box whole
bunches; MM-Grounding-DINO tiny found almost nothing. The best prompt differs
per model. Even in the clearest second of the video (about 20-30 flower heads
in view, checked frame by frame in section 8) they box only 1-6 things per
frame, so recall on individual flowers is low. Speed is 10-30 s per frame and
varies 2-3x with machine state: nothing is real time, and misses are silent.
Details and pictures are in the notebook.

### Picking sample frames

`notebooks/04_pick_frames.ipynb` is a small interactive viewer for choosing
frames by hand: buttons jump -1000 / -100 / -10 / -1 / +1 / +10 / +100 / +1000
frames, a button picks the current frame, and the picks are saved to
`data/raw/picks/<video>_frames.json` (git-ignored) so other notebooks can reuse
the same frames. Code in `src/bouquet_checker/framepicker.py` (plain, tested
logic) on top of `video.FrameReader` (random access by frame index; verified
pixel-exact against sequential decoding on the real video). Needs `ipywidgets`
(in `requirements-notebook.txt`) and a live Jupyter or VS Code session.

### Segmentation (no annotation)

`notebooks/03_video_segmentation.ipynb` asks whether pretrained
*segmentation* can replace hand-drawn boxes, on the same one-second window as
notebook 02: YOLO26 (instance, COCO: a negative control; semantic, ADE20K),
SegFormer, YOLOE (text, prompt-free and visual prompts), SAM 2.1 plus a
"flower pixels" filter, and SAM 3 if you have been granted access. Code in
`src/bouquet_checker/segmentation.py`.

Result: most methods give only *where the bouquet is* (semantic masks cover
bouquet, wrapping and foliage together; YOLOE gives one bouquet mask; YOLO26's
COCO model is blind to flowers). **SAM 3** is the exception: with the text
"flower" it returns about 45 individual, well-outlined masks per frame, but it
also labels big eucalyptus leaves as flowers, cannot reliably name species,
keeps only about 40% of its masks between frames 0.1 s apart, and takes about
43 s per frame per prompt on this CPU. The promising route is SAM 3 as an
**auto-labeller** (a person rejects wrong masks instead of drawing boxes) to
train a fast segmenter, with a separate classifier naming each mask.

Install Ultralytics *without* dependencies (its normal install pulls a second
OpenCV build, and pip requirements files cannot express that):

```bash
pip install ultralytics --no-deps polars ultralytics-thop psutil pyyaml requests
```

Ultralytics is AGPL-3.0 and stays notebook-only (not in `requirements.txt` or
the Docker image). Its analytics are switched off by the wrapper and its
weights go to `models/ultralytics/`. **SAM 3** is gated: request access at
<https://huggingface.co/facebook/sam3> and run `hf auth login`; until then its
cell prints how to get access and is skipped.

### SAM 3 on the hand-picked frames

`notebooks/05_sam3_picked_frames.ipynb` runs SAM 3 on the 18 frames saved by
notebook 04 with 11 text prompts (`flower`, `bouquet`, `leaf`, `rose`,
`carnation`, `gerbera`, `hydrangea`, `chrysanthemum`, `tulip`, `lily`,
`alstroemeria`). The image is encoded once and re-used for every prompt
(`Sam3TextSegmenter.segment_many`, 3.2x faster than separate calls), and the
masks are cached per frame under `data/raw/cache/sam3_picks/`
(`segmentation.save_results` / `load_results`). It takes about 74-106 s per
frame for all 11 prompts.

Result (judged by eye, there is no ground truth): `flower` is reliable and
`rose` is somewhat meaningful, but the other species words mostly outline the
*same* blobs (carnation / hydrangea / chrysanthemum overlap 35-67%), and `lily`
missed what look like lilies in the wide frame. SAM 3 is a good flower
*finder* and a poor flower *namer*: a prompt that does not fire must never be
read as "this flower is absent", and prompt words must not be used as species
labels. Naming has to come from a classifier on each mask's crop.

**Cheaper alternatives (same notebook, sections 9-16).** 15 methods with 2-4
parameter settings each (45 settings) are compared with SAM 3's `flower` masks
on the same 18 frames: classical ones in `src/bouquet_checker/classical.py`
(HSV threshold, k-means, mean-shift, Felzenszwalb, SLIC, quick shift, watershed,
MSER, Hough circles, DoG blobs, Canny regions, GrabCut; OpenCV, scipy and
scikit-image) and small deep models (`FastSamSegmenter`, `MobileSamSegmenter`
in `segmentation.py`, plus SAM 2.1 tiny). One shared, simple filter
(`classical.flowerlike`: not green, not grey, plausible size, compact) turns
every method's segments into flower-head candidates, and `classical.agreement`
scores them against SAM 3. That is *agreement with SAM 3, not accuracy*.

Result: FastSAM-s is the only method that comes close (recall 0.42, precision
0.43, about 1 s per frame, roughly 50x faster than SAM 3) and it also boxes
wrapping paper and background; SAM 2.1 tiny and MobileSAM are slow and find few
heads; the classical methods do not separate heads (best instance F1 0.24), but
a colour-mask watershed finds *where* the flowers are at about 20 fps. Nothing
that needs no training gives both speed and good heads, which is the case for a
trained detector. Results are cached under `data/raw/cache/small_seg/`. The
executed notebook is about 40 MB because of its many large figures.

### FastSAM-s over time: sequences and tracking

`notebooks/06_fastsam_sequences.ipynb` takes FastSAM-s from notebook 05 and
looks at **video**: a 5-second window around each of the 18 picks (windows
that overlap merge into one sequence: 9 sequences, 1611 frames). Detecting on
every frame would cost about 27 minutes, so FastSAM-s only runs on a stride
of keyframes (every 5th frame, 331 keyframes, 67 s total), and the frames in
between are filled two ways for comparison: **hold** (repeat the last
detection) and **flow** (warp it forward with
`classical.dense_flow` / `classical.warp_mask`, cheap dense optical flow).
Two different things are called "tracking" here: giving a consistent id to
objects already detected (`segmentation.link_tracks`, built on
`match_instances`, does not save any compute) versus predicting where an
object went so the heavy model does not have to run on every frame (the flow
step, which does). `segmentation.track_color` / `overlay_tracks` colour each
id consistently across a video, unlike `overlay_instances`'s colour-by-order.
The `flowerlike` filter's "not grey" check is switched off here (`min_sat=0`).

Result: optical flow clearly beats holding still when a track survives (mean
IoU with the next real detection 0.78 vs 0.66, over 4047 transitions), but
most tracks do not survive: 88% of the 14096 ids created appear in only one
keyframe. That is consistent with notebook 03's finding that SAM 3 keeps only
about 40% of its masks 0.1 s apart, not a new problem. A hard scene cut inside
one "5-second window" (sequence 6) also shows neither method can bridge a cut,
only continuous motion. Rendered comparison videos and detection caches are
under `data/raw/cache/fastsam_track/` (git-ignored).

### Reference photos for the Swedish species list

`notebooks/07_sweden_species_images.ipynb` is a visual sanity check: for
each of the 61 species in `data/sweden_species.json` (a hand-curated,
Sweden-focused toxicity table - cut flowers, foliage/filler and potted
plants commonly sold there, gathered from Swedish florist sources and
cross-checked against ASPCA), it looks up a representative photo via
Wikipedia's public REST API and displays them grouped by `type`. Tries the
(cleaned) `scientific_name` first, then "(plant)"/"(genus)" variants, then
`english_name`, with retries on transient failures (an early run without
retries mis-cached real photos as "missing" after a run of timeouts) and a
check that skips SVG-sourced thumbnails (diagrams/icons, not photos - one
genus page's lead image was the Chinese characters for the flower's name,
not the flower). Images are cached under
`data/raw/cache/sweden_species_images/` (git-ignored) and shown **for
personal reference only**: Wikimedia Commons images carry their own licence,
so do not reuse or publish any of them without checking that image's licence
page first.

### PlantNet-300K's class distribution, with ASPCA overlaid

`notebooks/08_plantnet_class_distribution.ipynb` looks at whether
PlantNet-300K (`data/raw/plantnet/`, 1000 species, 306,087 images - see
"Other datasets" below) is actually a good foundation to train a species
classifier on, by matching every one of its species against
`data/raw/aspca/cats_plant_details.json` (982 plants) using
`src/bouquet_checker/species_match.py`: scientific-name-only matching
(`SpeciesIndex`), exact (same genus + species) or genus-only (a relative is
known, not necessarily the same species) - never by common name, to avoid
the "lily problem" (*Lilium*, *Hemerocallis*, *Hippeastrum* and
*Spathiphyllum* are all called "lily" in English and are four different
genera with different toxicity).

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

### Other datasets checked against ASPCA (not yet downloaded as images)

Two much larger species-labelled plant datasets were also matched against
ASPCA's 982 plants the same way (species-level CSVs cached under
`data/raw/plantnet/` and `data/raw/other_datasets/`, not yet as full image
sets): **PlantCLEF2024** (7,806 species, ~1.4M images, a superset of
PlantNet's own data plus an iNaturalist pull - 191 exact / 270 genus-only /
521 no match against ASPCA's 982; 23/26/12 against the 61-species Sweden
list) and **iNaturalist 2021's Plantae supercategory** (4,271 species, ~1.15M
images - 260/323/399 against ASPCA; 19/26/16 against Sweden). Both cover the
actual bouquet/houseplant genera (rose, tulip, lily, chrysanthemum, peony,
azalea, poinsettia, ...) at genus level far better than PlantNet-300K does,
since neither excludes cultivated/garden plants as systematically as
PlantNet's collection guidelines do. PlantCLEF2024's full images are a
~158.6-281.4 GiB download depending on resolution, not yet fetched.

**About downloaded videos.** A YouTube video keeps its creator's licence (the
one used here is the standard YouTube licence). It is fetched once for
private analysis into the git-ignored `data/raw/videos/`, with a sidecar JSON
recording the source and licence. Never commit or redistribute the video or
frames taken from it. yt-dlp is also dropping Python 3.10 support, so it may
need a newer Python later.

### Test & lint

```bash
pytest               # fast, offline
ruff check .          # line length 79
```

### Docker

```bash
docker build -t bouquet-checker .
docker run --rm bouquet-checker                # runs pytest
docker compose run --rm test
docker compose run --rm cli --help
```

The image uses CPU-only PyTorch. Put photos in `./photos`; the data and
model caches are mounted from the host so they survive rebuilds.

CI (`.github/workflows/ci.yml`) runs `ruff` and builds the Docker image and
runs `pytest` inside it, on every push and PR.

## Repo layout

```
src/bouquet_checker/   toxicity table, verdict, pipeline, CLI, species
                        matching, ASPCA scraper, video/detection/
                        segmentation stretch-goal modules
data/raw/               downloaded/scraped datasets (git-ignored):
                        aspca/ (scraped toxicity), plantnet/ (species images),
                        other_datasets/ (PlantCLEF2024/iNat2021 metadata),
                        videos/, picks/, cache/
data/sweden_species.json  hand-curated 61-species toxicity table (Sweden)
data/processed/         (planned, Phase 1) small committed derived tables,
                        e.g. plantnet_aspca_sides.csv
configs/                (planned, Phase 1) one JSON file per training run
models/hf/               HF model cache (git-ignored)
models/ultralytics/      Ultralytics weights, AGPL-3.0, notebook-only (git-ignored)
models/runs/            (planned, Phase 1) per-run checkpoints + metrics (git-ignored)
reports/                (planned, Phase 1) light, tracked run artifacts + results
notebooks/               exploratory notebooks (see above)
tests/                   pytest suite, offline by default
```

## Roadmap

Phases below replace the old, now-stale roadmap (hand-curate a 102-species
table, add florist species to Oxford-102, ...), which assumed the
Oxford-102 pipeline that this session removed.

1. **PlantNet-300K baseline.** A real PyTorch training pipeline
   (`plantnet_labels.py`, `plantnet_dataset.py`, `metrics.py`, `train.py`),
   replicating the official paper's ResNet-18 baseline setup (GPU:
   GTX 1070), with per-epoch resumable checkpointing. `train-plantnet` /
   `evaluate-plantnet` CLI commands.
2. **Imbalance-aware variants.** Class-Balanced Loss / Balanced Softmax /
   decoupled retraining as additional configs through the same training
   script, compared side by side via `reports/experiments/`.
3. **Toxicity-aware training.** A cost-sensitive loss term built from
   `species_match.SpeciesIndex` matching PlantNet's full 1,000-species
   label space against the scraped ASPCA data (`data/processed/
   plantnet_aspca_sides.csv`), penalizing toxic <-> non-toxic species
   confusions more than same-side ones - warm-started from Phase 2's best
   checkpoint, same training script, still only ever predicting species.
   A hand-verified subset of ASPCA data (starting from the ~34 exact-match,
   well-photographed species already identified in notebook 08) becomes the
   first real `verified: true` entries.
4. **Results.** A `README.md` results table, a results notebook, and
   `reports/results/` - programmatically selected correct-catch and
   failure-case examples (same-genus confusion, sparse training class,
   wrong plant organ in frame), not hand-picked. `cli.py check` then
   defaults to the PlantNet+toxicity pipeline.

The detection/segmentation/video stretch-goal track (notebooks 02-06) is
paused, not abandoned, while phases 1-4 are built.
