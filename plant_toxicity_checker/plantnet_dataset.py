"""PlantNet-300K image dataset: metadata -> image paths -> a PyTorch
``Dataset``.

The path-construction and split-filtering logic below is kept as plain,
torch-free functions on purpose, separate from the ``Dataset`` class
itself (which needs PIL/torchvision) - so it's unit-testable without
torch, a GPU, or the real ~39GB of images.
"""

import csv

from plant_toxicity_checker.config import PLANTNET_DIR

METADATA_PATH = PLANTNET_DIR / "plantnet300K_metadata.csv"
IMAGES_DIR = PLANTNET_DIR / "images"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_metadata(path=None):
    """All rows of ``plantnet300K_metadata.csv`` as a list of dicts."""
    path = path or METADATA_PATH
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rows_for_split(rows, split):
    """Rows whose ``split`` column matches ``split`` (the paper's own
    train/val/test assignment - never re-split)."""
    return [row for row in rows if row["split"] == split]


def filter_species_subset(rows, max_species, priority_species_ids=None):
    """Rows belonging only to a ``max_species``-sized subset of the
    ``species_id`` values present in ``rows``.

    Any ID in ``priority_species_ids`` that's actually present in
    ``rows`` is kept first (sorted, for determinism); remaining slots are
    filled with the lowest remaining ``species_id`` values. Without
    ``priority_species_ids`` this is just "keep the ``max_species``
    lowest IDs". Either way the choice is fixed, not random, so a dev
    run's species list is reproducible without persisting it separately,
    and calling this independently for the train/val/test splits still
    lands on the same subset for all three (the selection only depends
    on the set of IDs present, not on ``split``).

    If ``priority_species_ids`` alone exceeds ``max_species``, only the
    lowest-ID ``max_species`` of them are kept - the cap is never
    silently exceeded.
    """
    if max_species is None:
        return rows
    all_ids = sorted({int(row["species_id"]) for row in rows})
    priority = set(priority_species_ids or ())
    priority_ids = [i for i in all_ids if i in priority][:max_species]
    remaining_slots = max_species - len(priority_ids)
    rest_ids = [i for i in all_ids if i not in priority][:remaining_slots]
    keep_ids = set(priority_ids) | set(rest_ids)
    return [row for row in rows if int(row["species_id"]) in keep_ids]


def cap_images_per_species(rows, max_images_per_species):
    """At most ``max_images_per_species`` rows per ``species_id``, taking
    the first N encountered in ``rows``' existing order.

    Deterministic (no sampling) for the same reason as
    ``filter_species_subset`` - a dev run's image selection shouldn't
    depend on a seed to be reproducible.
    """
    if max_images_per_species is None:
        return rows
    counts = {}
    kept = []
    for row in rows:
        species_id = row["species_id"]
        seen = counts.get(species_id, 0)
        if seen < max_images_per_species:
            kept.append(row)
            counts[species_id] = seen + 1
    return kept


def species_dir(species_id):
    """The zero-padded 4-digit directory name for a ``species_id``.

    Image directories are ``0000``-``0999``; the metadata CSV's
    ``species_id`` column is a plain int (e.g. ``"7"``), so this join has
    to happen explicitly rather than using the string as-is.
    """
    return f"{int(species_id):04d}"


def image_path(row, images_dir=None):
    """The on-disk path for one metadata row."""
    images_dir = images_dir or IMAGES_DIR
    return (
        images_dir
        / row["split"]
        / species_dir(row["species_id"])
        / f"{row['PN_hash']}.jpg"
    )


def build_transforms(train):
    """The paper's transforms: Resize(256) then RandomCrop(224) (train)
    or CenterCrop(224) (val/test), then ImageNet normalization.

    ``torchvision`` is imported here, not at module level, so the rest of
    this module (path construction, split filtering) stays importable
    without torch installed.
    """
    from torchvision import transforms

    crop = transforms.RandomCrop(224) if train else transforms.CenterCrop(224)
    return transforms.Compose([
        transforms.Resize(256),
        crop,
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class PlantNetDataset:
    """A map-style dataset over one split of PlantNet-300K.

    Duck-types ``torch.utils.data.Dataset`` (``__len__``/``__getitem__``)
    rather than importing/subclassing it, so this module stays importable
    without torch installed - ``DataLoader`` only needs those two dunder
    methods, not the base class itself.

    Labels come from ``plantnet_labels.SpeciesLabels.index_of_species_id``
    - never the raw CSV ``species_id`` used as an index directly, since
    nothing guarantees it is a dense, contiguous 0..999 index matching a
    model's output order.
    """

    def __init__(
        self, split, labels, rows=None, images_dir=None, transform=None,
        max_species=None, max_images_per_species=None,
        priority_species_ids=None,
    ):
        self.split = split
        self.labels = labels
        self.images_dir = images_dir or IMAGES_DIR
        self.transform = transform or build_transforms(train=split == "train")
        all_rows = rows if rows is not None else load_metadata()
        all_rows = filter_species_subset(
            all_rows, max_species, priority_species_ids
        )
        self.rows = rows_for_split(all_rows, split)
        self.rows = cap_images_per_species(
            self.rows, max_images_per_species
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        from PIL import Image

        row = self.rows[index]
        image = Image.open(
            image_path(row, self.images_dir)
        ).convert("RGB")
        image = self.transform(image)
        label = self.labels.index_of_species_id(row["species_id"])
        return image, label
