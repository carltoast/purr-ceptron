"""Segmentation: pixel masks instead of boxes.

Two kinds, and the difference matters for bouquets:

- **Semantic** segmentation labels every pixel with a class ("flower",
  "plant", "vase", ...). It says *where* flowers are but not *which* flower
  is which: touching heads merge into one blob.
- **Instance** segmentation returns one mask per object, so two overlapping
  flower heads stay two masks.

Every method here needs no labels of our own, which is the point: they are
pretrained (YOLO26 on COCO / ADE20K, SegFormer on ADE20K, YOLOE and SAM for
"anything"). Heavy libraries (``ultralytics``, ``transformers``) are imported
lazily; ``ultralytics`` is AGPL-3.0 and notebook-only
(``requirements-notebook.txt``).
"""

import contextlib
import os
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from bouquet_checker.config import HF_CACHE_DIR, ULTRALYTICS_DIR

SAM3_HELP = (
    "SAM 3 is a gated model. Request access at "
    "https://huggingface.co/facebook/sam3 (approval is manual), then run "
    "`hf auth login` with a read token and try again."
)


class Sam3AccessError(RuntimeError):
    """SAM 3 could not be loaded (no token, or access not granted yet)."""


@dataclass(eq=False)
class Instance:
    """One segmented object: a boolean ``(H, W)`` mask, score and label."""

    mask: np.ndarray
    score: float
    label: str

    @property
    def area(self):
        """Number of mask pixels."""
        return int(self.mask.sum())

    @property
    def box(self):
        """Tight ``(x0, y0, x1, y1)`` around the mask, or None if empty."""
        return mask_box(self.mask)


# --- mask geometry -----------------------------------------------------------


def mask_box(mask):
    """Tight ``(x0, y0, x1, y1)`` around the True pixels (None if none)."""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)


def mask_iou(a, b):
    """Intersection over union of two boolean masks (0 to 1)."""
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def match_instances(previous, current, iou_threshold=0.5):
    """Pair instances of two consecutive frames by mask overlap.

    Greedy and one-to-one, like ``detection.match_boxes`` but on pixels:
    returns ``[(i, j, iou)]`` with ``i`` into ``previous``, ``j`` into
    ``current``.
    """
    pairs = sorted(
        (
            (mask_iou(a.mask, b.mask), i, j)
            for i, a in enumerate(previous)
            for j, b in enumerate(current)
        ),
        reverse=True,
    )
    used_prev, used_cur, matches = set(), set(), []
    for iou, i, j in pairs:
        if iou < iou_threshold:
            break
        if i in used_prev or j in used_cur:
            continue
        used_prev.add(i)
        used_cur.add(j)
        matches.append((i, j, iou))
    return matches


def find_class_ids(names, wanted):
    """Ids in an ``{id: name}`` mapping whose name equals ``wanted``."""
    return [int(i) for i, n in names.items() if str(n).lower() == wanted]


def class_mask(class_map, class_ids):
    """Boolean mask of the pixels whose class is one of ``class_ids``."""
    return np.isin(class_map, list(class_ids))


def keep_by_semantic(instances, semantic_mask, min_fraction=0.5,
                     min_area_frac=0.0005, max_area_frac=0.5):
    """Keep instances that mostly lie on a semantic class and are plausibly
    sized (this is how "segment everything" becomes "flower candidates").

    ``min_fraction`` is the share of an instance's pixels that must be inside
    ``semantic_mask``; the area limits are fractions of the whole frame. The
    returned copies carry that share as their ``score``.
    """
    total = semantic_mask.size
    kept = []
    for inst in instances:
        area = inst.area
        if not (min_area_frac * total <= area <= max_area_frac * total):
            continue
        fraction = float(np.logical_and(inst.mask, semantic_mask).sum() / area)
        if fraction >= min_fraction:
            kept.append(replace(inst, score=fraction))
    return kept


def remove_overlapping(instances, exclude, min_overlap=0.5):
    """Drop instances whose pixels are mostly covered by ``exclude``.

    Typical use: SAM 3 "flower" masks that are really eucalyptus leaves are
    removed by excluding the SAM 3 "leaf" masks. ``min_overlap`` is the share
    of an instance's own pixels that must lie inside the excluded masks.
    """
    if not exclude:
        return list(instances)
    covered = np.logical_or.reduce([e.mask for e in exclude])
    kept = []
    for inst in instances:
        area = inst.area
        inside = np.logical_and(inst.mask, covered).sum()
        overlap = inside / area if area else 0.0
        if overlap < min_overlap:
            kept.append(inst)
    return kept


def link_tracks(frame_instances, iou_threshold=0.5, start_id=0):
    """Assign persistent integer ids to instances across a sequence.

    ``frame_instances`` is a list of instance lists, one per frame (they need
    not be consecutive video frames, e.g. sampled keyframes). Consecutive
    frames are linked with ``match_instances`` (mask IoU, greedy, one-to-one);
    a matched instance keeps its id, an unmatched one starts a new id, and a
    previous instance with no match simply has no successor (its track ends).

    Returns ``(tracks, next_id)``: ``tracks`` is a list of ``{id: Instance}``
    dicts, one per frame, and ``next_id`` is the first id not yet used.
    """
    tracks = []
    ids_by_index = {}
    next_id = start_id
    previous = []
    for current in frame_instances:
        if previous:
            pairs = match_instances(previous, current, iou_threshold)
            matched = {j: ids_by_index[i] for i, j, _ in pairs}
        else:
            matched = {}
        new_ids = dict(matched)
        for j in range(len(current)):
            if j not in new_ids:
                new_ids[j] = next_id
                next_id += 1
        tracks.append({new_ids[j]: inst for j, inst in enumerate(current)})
        ids_by_index = new_ids
        previous = current
    return tracks, next_id


def track_color(track_id, seed=0.0):
    """Deterministic, well-separated RGB colour (0-255) for a track id.

    Same id always gives the same colour (golden-ratio hue steps, so nearby
    ids still look different), which is the point: unlike ``overlay_instances``
    (colour by draw order), this lets an object keep its colour across the
    frames of a video.
    """
    import colorsys

    hue = (seed + track_id * 0.6180339887498949) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (int(red * 255), int(green * 255), int(blue * 255))


def overlay_tracks(image, id_to_mask, alpha=0.5, outline=True):
    """Like ``overlay_instances`` but coloured by ``track_color(id)``, so the
    same object keeps the same colour across a video's frames."""
    canvas = np.asarray(image.convert("RGB")).astype(np.float32)
    for track_id, mask in id_to_mask.items():
        color = np.array(track_color(track_id), dtype=np.float32)
        canvas[mask] = (1 - alpha) * canvas[mask] + alpha * color
        if outline:
            edge = mask & ~_erode(_erode(mask))
            canvas[edge] = color
    return Image.fromarray(canvas.astype(np.uint8))


def save_results(path, results):
    """Save ``{prompt: [Instance]}`` to a compressed ``.npz`` file.

    Masks are stored as they are (downscale them first if memory matters),
    so a saved frame can be extended with more prompts later.
    """
    arrays = {"prompts": np.array(list(results))}
    for n, (prompt, found) in enumerate(results.items()):
        arrays[f"scores_{n}"] = np.array(
            [i.score for i in found], dtype=np.float32)
        arrays[f"masks_{n}"] = (
            np.stack([i.mask for i in found]) if found
            else np.zeros((0, 1, 1), dtype=bool)
        )
    np.savez_compressed(path, **arrays)


def load_results(path):
    """Read a file written by ``save_results`` back to ``{prompt: [...]}``."""
    data = np.load(path)
    return {
        str(prompt): [
            Instance(mask, float(score), str(prompt))
            for mask, score in zip(data[f"masks_{n}"], data[f"scores_{n}"])
        ]
        for n, prompt in enumerate(data["prompts"])
    }


# --- drawing -----------------------------------------------------------------


def _erode(mask):
    inner = mask.copy()
    inner[1:] &= mask[:-1]
    inner[:-1] &= mask[1:]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return inner


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """A copy of ``image`` with one boolean ``mask`` tinted ``color``."""
    canvas = np.asarray(image.convert("RGB")).astype(np.float32)
    canvas[mask] = (1 - alpha) * canvas[mask] + alpha * np.array(color)
    return Image.fromarray(canvas.astype(np.uint8))


def overlay_instances(image, instances, alpha=0.5, seed=0, outline=True):
    """A copy of ``image`` with every instance tinted a different colour
    (and outlined, so touching masks stay distinguishable)."""
    canvas = np.asarray(image.convert("RGB")).astype(np.float32)
    rng = np.random.default_rng(seed)
    for inst in instances:
        color = rng.integers(40, 256, size=3).astype(np.float32)
        mask = inst.mask
        canvas[mask] = (1 - alpha) * canvas[mask] + alpha * color
        if outline:
            edge = mask & ~_erode(_erode(mask))
            canvas[edge] = color
    return Image.fromarray(canvas.astype(np.uint8))


# --- YOLO (Ultralytics) -------------------------------------------------


def _ultralytics():
    """Import ultralytics with its weights kept under ``models/`` and its
    usage analytics switched off."""
    try:
        import ultralytics
    except ImportError as err:
        raise ImportError(
            "ultralytics is not installed. Install it WITHOUT dependencies "
            "(its normal install pulls a second OpenCV build): "
            "pip install ultralytics --no-deps polars ultralytics-thop "
            "psutil pyyaml requests"
        ) from err
    ULTRALYTICS_DIR.mkdir(parents=True, exist_ok=True)
    ultralytics.settings.update(
        {"weights_dir": str(ULTRALYTICS_DIR), "sync": False}
    )
    return ultralytics


@contextlib.contextmanager
def _working_dir(path):
    """Run a block with ``path`` as the cwd. YOLOE downloads its ~240 MB text
    encoder into the working directory, so we point that at ``models/``."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _instances_from_result(result):
    """Convert an Ultralytics segmentation ``Results`` to ``Instance``s."""
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return []
    masks = result.masks.data.cpu().numpy().astype(bool)
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    found = [
        Instance(mask, float(score), str(result.names[int(cls)]))
        for mask, score, cls in zip(masks, scores, classes)
    ]
    return sorted(found, key=lambda inst: -inst.score)


class YoloSegmenter:
    """Instance segmentation with an Ultralytics model.

    ``yolo26s-seg.pt`` is COCO-trained (80 classes, no flower). A ``yoloe``
    weight takes text ``prompts`` at run time (open-vocabulary).
    """

    def __init__(self, weights, prompts=None):
        self.weights = weights
        self._model = self._load(weights, prompts)

    @staticmethod
    def _load(weights, prompts):
        ul = _ultralytics()
        path = str(ULTRALYTICS_DIR / weights)
        if "yoloe" not in weights:
            return ul.YOLO(path)
        model = ul.YOLOE(path)
        if prompts:
            with _working_dir(ULTRALYTICS_DIR):
                model.set_classes(list(prompts))
        return model

    def set_prompts(self, prompts):
        """Change the text prompts of a YOLOE model (cheap, no reload)."""
        with _working_dir(ULTRALYTICS_DIR):
            self._model.set_classes(list(prompts))

    def segment(self, image, conf=0.25):
        """Instances in a PIL ``image`` above ``conf``, best first."""
        result = self._model.predict(
            image, retina_masks=True, conf=conf, verbose=False
        )[0]
        return _instances_from_result(result)


class YoloeVisualSegmenter:
    """YOLOE prompted by example boxes on a reference image (no text, no
    training). Scores are much lower than with text prompts, so use a low
    ``conf``. Kept separate from ``YoloSegmenter`` because Ultralytics caches
    the visual-prompt predictor on the model."""

    def __init__(self, weights="yoloe-26s-seg.pt"):
        self._model = self._load(weights)

    @staticmethod
    def _load(weights):
        ul = _ultralytics()
        return ul.YOLOE(str(ULTRALYTICS_DIR / weights))

    def segment(self, image, reference, boxes, conf=0.03):
        """Find things like the ``boxes`` (xyxy, drawn on ``reference``)."""
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        boxes = np.asarray(boxes, dtype=float)
        prompts = {"bboxes": boxes, "cls": np.zeros(len(boxes), dtype=int)}
        result = self._model.predict(
            image, refer_image=reference, visual_prompts=prompts,
            predictor=YOLOEVPSegPredictor, retina_masks=True, conf=conf,
            verbose=False,
        )[0]
        return _instances_from_result(result)


def _instances_from_polygons(result, downscale, label):
    """``Instance``s from the polygons of an Ultralytics ``Results``,
    rasterised at 1/``downscale`` of the original image size (the "segment
    everything" models return letterboxed masks; polygons are in original
    pixels)."""
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return []
    import cv2

    height, width = result.orig_shape
    out_h, out_w = -(-height // downscale), -(-width // downscale)
    scores = result.boxes.conf.cpu().numpy()
    found = []
    for polygon, score in zip(result.masks.xy, scores):
        if len(polygon) < 3:
            continue
        canvas = np.zeros((out_h, out_w), dtype=np.uint8)
        points = np.round(np.asarray(polygon) / downscale).astype(np.int32)
        cv2.fillPoly(canvas, [points], 1)
        if canvas.any():
            found.append(Instance(canvas.astype(bool), float(score), label))
    return sorted(found, key=lambda inst: -inst.score)


class FastSamSegmenter:
    """FastSAM-s ("segment everything", a YOLO-based SAM look-alike, 23 MB).

    About a second per frame on this CPU. Masks have no meaning: filter them
    (see ``classical.flowerlike``)."""

    def __init__(self, weights="FastSAM-s.pt"):
        self._model = self._load(weights)

    @staticmethod
    def _load(weights):
        ul = _ultralytics()
        with _working_dir(ULTRALYTICS_DIR):
            return ul.FastSAM(str(ULTRALYTICS_DIR / weights))

    def segment(self, image, imgsz=1024, conf=0.4, iou=0.9, downscale=3):
        """Instances for one PIL image, best first, masks at 1/``downscale``.
        ``conf`` and ``iou`` are the confidence and NMS thresholds."""
        result = self._model(
            image, imgsz=imgsz, conf=conf, iou=iou, device="cpu",
            verbose=False,
        )[0]
        return _instances_from_polygons(result, downscale, "fastsam")


class MobileSamSegmenter:
    """MobileSAM (SAM with a tiny image encoder, 39 MB) prompted with a
    regular grid of points, one object per point ("segment everything" by
    hand: the built-in mode uses 32 x 32 points and takes minutes on this
    CPU). Neighbouring points often find the same object, so expect
    duplicates."""

    def __init__(self, weights="mobile_sam.pt"):
        self._model = self._load(weights)

    @staticmethod
    def _load(weights):
        ul = _ultralytics()
        with _working_dir(ULTRALYTICS_DIR):
            return ul.SAM(str(ULTRALYTICS_DIR / weights))

    def segment(self, image, grid=8, downscale=3):
        """Instances for one PIL image, best first, masks at 1/``downscale``.
        ``grid`` is the side of the point grid (8 means 64 prompts)."""
        width, height = image.size
        xs = (np.arange(grid) + 0.5) * width / grid
        ys = (np.arange(grid) + 0.5) * height / grid
        points = [[float(x), float(y)] for y in ys for x in xs]
        result = self._model(
            image, points=points, labels=[1] * len(points), device="cpu",
            verbose=False,
        )[0]
        return _instances_from_polygons(result, downscale, "mobilesam")


class YoloSemanticSegmenter:
    """Semantic segmentation with ``yolo26?-sem-ade20k.pt`` (150 ADE20K
    classes, including "flower")."""

    def __init__(self, weights="yolo26s-sem-ade20k.pt"):
        self.weights = weights
        self._model = self._load(weights)

    @staticmethod
    def _load(weights):
        return _ultralytics().YOLO(str(ULTRALYTICS_DIR / weights))

    def segment(self, image):
        """``(class_map, names)``: an ``(H, W)`` array of class ids at the
        image's resolution, and the ``{id: name}`` mapping."""
        result = self._model.predict(image, verbose=False)[0]
        class_map = result.semantic_mask.data.cpu().numpy()
        return class_map, dict(result.names)


# --- Hugging Face (SegFormer, SAM 2, SAM 3) -----------------------------


class SegformerSemanticSegmenter:
    """Semantic segmentation with SegFormer trained on ADE20K."""

    def __init__(self, model_id="nvidia/segformer-b2-finetuned-ade-512-512"):
        self.model_id = model_id
        self._processor, self._model = self._load(model_id)

    @staticmethod
    def _load(model_id):
        from transformers import (
            AutoImageProcessor,
            SegformerForSemanticSegmentation,
        )

        processor = AutoImageProcessor.from_pretrained(
            model_id, cache_dir=HF_CACHE_DIR
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_id, cache_dir=HF_CACHE_DIR
        ).eval()
        return processor, model

    def segment(self, image):
        """``(class_map, names)`` as for ``YoloSemanticSegmenter``."""
        import torch

        inputs = self._processor(images=image, return_tensors="pt")
        with torch.inference_mode():
            outputs = self._model(**inputs)
        class_map = self._processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(image.height, image.width)]
        )[0]
        names = {int(i): n for i, n in self._model.config.id2label.items()}
        return class_map.cpu().numpy(), names


class Sam2AutoMasker:
    """SAM 2.1 "segment everything": masks for whatever it finds, with no
    idea what they are. Slow on CPU (tens of seconds per frame), so use it on
    a few frames, then filter (``keep_by_semantic``)."""

    def __init__(self, model_id="facebook/sam2.1-hiera-tiny"):
        self._pipeline = self._load(model_id)

    @staticmethod
    def _load(model_id):
        from transformers import pipeline

        return pipeline(
            "mask-generation", model=model_id, device="cpu",
            model_kwargs={"cache_dir": str(HF_CACHE_DIR)},
        )

    def segment(self, image, points_per_crop=16, points_per_batch=64,
                pred_iou_thresh=0.8, stability_score_thresh=0.9):
        """Instances for one PIL image, best first. ``points_per_crop`` is
        the side of the prompt grid (16 means 16 x 16 = 256 points)."""
        result = self._pipeline(
            image, points_per_batch=points_per_batch,
            points_per_crop=points_per_crop,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
        )
        found = [
            Instance(mask.cpu().numpy().astype(bool), float(score), "object")
            for mask, score in zip(result["masks"], result["scores"])
        ]
        return sorted(found, key=lambda inst: -inst.score)


class Sam3TextSegmenter:
    """SAM 3: every instance of a text concept ("flower"), in one pass.

    Gated on Hugging Face (manual approval) and ~3.4 GB, so construction
    raises ``Sam3AccessError`` with instructions until access is granted.
    """

    def __init__(self, model_id="facebook/sam3"):
        from huggingface_hub import get_token

        if not get_token():
            raise Sam3AccessError(SAM3_HELP)
        try:
            self._processor, self._model = self._load(model_id)
        except OSError as err:  # gated repo without access
            raise Sam3AccessError(SAM3_HELP) from err

    @staticmethod
    def _load(model_id):
        from transformers import Sam3Model, Sam3Processor

        processor = Sam3Processor.from_pretrained(
            model_id, cache_dir=HF_CACHE_DIR
        )
        model = Sam3Model.from_pretrained(
            model_id, cache_dir=HF_CACHE_DIR
        ).eval()
        return processor, model

    def segment(self, image, prompt, threshold=0.5, mask_threshold=0.5):
        """Instances matching the text ``prompt``, best first."""
        import torch

        inputs = self._processor(images=image, text=prompt,
                                 return_tensors="pt")
        with torch.inference_mode():
            outputs = self._model(**inputs)
        result = self._processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        found = [
            Instance(mask.cpu().numpy().astype(bool), float(score), prompt)
            for mask, score in zip(result["masks"], result["scores"])
        ]
        return sorted(found, key=lambda inst: -inst.score)

    def segment_many(self, image, prompts, threshold=0.5, mask_threshold=0.5,
                     downscale=1):
        """``{prompt: instances}`` for several prompts on one image.

        The image is encoded ONCE and the encoding reused for every prompt,
        which is where most of the time goes, so N prompts cost far less
        than N calls to ``segment``. ``downscale`` keeps every n-th pixel of
        each mask (2 = half resolution), which saves a lot of memory when
        there are many masks.
        """
        import torch

        image_inputs = self._processor(images=image, return_tensors="pt")
        sizes = image_inputs.get("original_sizes").tolist()
        results = {}
        with torch.inference_mode():
            vision = self._model.get_vision_features(
                pixel_values=image_inputs["pixel_values"]
            )
            for prompt in prompts:
                text = self._processor(text=prompt, return_tensors="pt")
                outputs = self._model(
                    vision_embeds=vision,
                    input_ids=text["input_ids"],
                    attention_mask=text.get("attention_mask"),
                )
                result = self._processor.post_process_instance_segmentation(
                    outputs, threshold=threshold,
                    mask_threshold=mask_threshold, target_sizes=sizes,
                )[0]
                found = [
                    Instance(
                        mask.cpu().numpy().astype(bool)[::downscale,
                                                        ::downscale],
                        float(score), prompt)
                    for mask, score in zip(result["masks"], result["scores"])
                ]
                results[prompt] = sorted(found, key=lambda i: -i.score)
        return results
