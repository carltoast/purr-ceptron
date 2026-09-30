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
    ):
        self.split = split
        self.labels = labels
        self.images_dir = images_dir or IMAGES_DIR
        self.transform = transform or build_transforms(train=split == "train")
        all_rows = rows if rows is not None else load_metadata()
        self.rows = rows_for_split(all_rows, split)

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
