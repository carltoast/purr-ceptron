"""End-to-end: image -> species -> toxicity verdict."""

from pathlib import Path

from PIL import Image, ImageOps

from plant_toxicity_checker.config import CANDIDATE_FLOOR, MIN_CONFIDENCE
from plant_toxicity_checker.verdict import assess

try:  # optional: lets PIL open iPhone .heic photos
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


class ImageLoadError(ValueError):
    """The input could not be read as an image."""


def load_image(image):
    """Accept a path or PIL image; return an upright RGB PIL image."""
    if isinstance(image, (str, Path)):
        try:
            image = Image.open(image)
            image.load()
        except (OSError, Image.DecompressionBombError) as err:
            raise ImageLoadError(
                f"Cannot read {image} as an image: {err}"
            ) from err
    # Phone photos store rotation in EXIF; without this they can be sideways.
    return ImageOps.exif_transpose(image).convert("RGB")


def check_image(
    image,
    classifier,
    table,
    min_confidence=MIN_CONFIDENCE,
    candidate_floor=CANDIDATE_FLOOR,
):
    """Classify ``image`` and return the safety ``Verdict``.

    The classifier is always asked for its full distribution: the safety
    check must see every plausible alternative, however few the caller
    wants to display.
    """
    predictions = classifier.predict(load_image(image), top_k=None)
    return assess(predictions, table, min_confidence, candidate_floor)
