"""Open-vocabulary object detection: find things from a text prompt.

"Open-vocabulary" means the class list is text you type ("flower"), not a
fixed set the model was trained on, so it can find flowers with no flower
training. One wrapper covers the families available in ``transformers``:

- Grounding DINO and MM-Grounding-DINO: text and image fused in the
  detector; the most accurate of the openly runnable ones, slower.
- OWLv2 (and OWL-ViT): CLIP-style; simpler and permissively licensed.

All return ``Detection`` boxes in the pixel coordinates of the image passed
in. ``transformers`` and torch are imported lazily.
"""

from dataclasses import dataclass

from PIL import ImageDraw, ImageFont

from bouquet_checker.config import HF_CACHE_DIR


@dataclass(frozen=True)
class Detection:
    """One box: pixel corners ``(x0, y0, x1, y1)``, score and matched text."""

    box: tuple[float, float, float, float]
    score: float
    label: str

    @property
    def width(self):
        """Box width in pixels."""
        return self.box[2] - self.box[0]

    @property
    def height(self):
        """Box height in pixels."""
        return self.box[3] - self.box[1]

    @property
    def area(self):
        """Box area in square pixels."""
        return self.width * self.height


def clip_box(box, width, height):
    """Clamp a box to the image. Some detectors pad frames to a square, so
    their boxes can spill outside the real image."""
    x0, y0, x1, y1 = box
    return (
        min(max(x0, 0.0), width), min(max(y0, 0.0), height),
        min(max(x1, 0.0), width), min(max(y1, 0.0), height),
    )


def non_max_suppression(detections, iou_threshold=0.5):
    """Drop boxes that mostly overlap a higher-scoring one (any label).

    Open-vocabulary detectors return many near-duplicates, and several
    prompts ("flower", "rose") can fire on the same object.
    """
    if len(detections) < 2:
        return list(detections)
    import torch
    from torchvision.ops import nms

    boxes = torch.tensor([d.box for d in detections], dtype=torch.float32)
    scores = torch.tensor([d.score for d in detections], dtype=torch.float32)
    keep = nms(boxes, scores, iou_threshold).tolist()
    return [detections[i] for i in sorted(keep, key=lambda i: -scores[i])]


def box_iou(a, b):
    """Intersection over union of two ``(x0, y0, x1, y1)`` boxes (0 to 1)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def match_boxes(previous, current, iou_threshold=0.5):
    """Pair boxes of two consecutive frames by overlap, one-to-one.

    Greedy: the most overlapping pair is matched first. Returns
    ``[(i, j, iou)]`` with ``i`` indexing ``previous`` and ``j`` indexing
    ``current``. Both are lists of ``(x0, y0, x1, y1)`` boxes. This is the
    simplest building block of tracking: a box that keeps being matched is
    the same object from frame to frame.
    """
    pairs = sorted(
        (
            (box_iou(a, b), i, j)
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


def grounding_prompt(prompts):
    """Grounding DINO wants lower-case phrases, each ending with a period."""
    phrases = [p.strip().lower().rstrip(".").strip() for p in prompts]
    if not phrases or not all(phrases):
        raise ValueError("Give at least one non-empty text prompt")
    return " ".join(f"{p}." for p in phrases)


class OpenVocabDetector:
    """Detect objects named by text prompts with a Hugging Face model."""

    def __init__(self, model_id, device="cpu"):
        self.model_id = model_id
        self.device = device
        self._processor, self._model, self.family = self._load(
            model_id, device
        )

    @staticmethod
    def _load(model_id, device):
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )

        processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=HF_CACHE_DIR
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, cache_dir=HF_CACHE_DIR
        ).to(device).eval()
        model_type = model.config.model_type
        if "grounding-dino" in model_type:
            family = "grounding"
        elif model_type.startswith("owl"):
            family = "owl"
        else:
            raise ValueError(f"Unsupported detector type: {model_type!r}")
        return processor, model, family

    def detect(self, image, prompts, threshold=0.25, text_threshold=None,
               nms_iou=0.5):
        """Boxes for ``prompts`` in a PIL ``image``, best first.

        ``threshold`` is the box score cut-off. ``text_threshold`` (Grounding
        DINO only) is how strongly a box must match the words; it defaults to
        ``threshold``. Overlapping boxes are merged with NMS (``nms_iou``).
        """
        prompts = [p.strip() for p in prompts]
        if not prompts or not all(prompts):
            raise ValueError("Give at least one non-empty text prompt")
        if text_threshold is None:
            text_threshold = threshold
        if self.family == "grounding":
            raw = self._detect_grounding(
                image, prompts, threshold, text_threshold
            )
        else:
            raw = self._detect_owl(image, prompts, threshold)
        width, height = image.size
        detections = []
        for box, score, label in raw:
            clipped = clip_box(box, width, height)
            if clipped[2] - clipped[0] >= 1 and clipped[3] - clipped[1] >= 1:
                detections.append(Detection(clipped, float(score), label))
        detections.sort(key=lambda d: -d.score)
        return non_max_suppression(detections, nms_iou)

    def _run(self, inputs):
        import torch

        with torch.inference_mode():
            return self._model(**inputs)

    def _detect_grounding(self, image, prompts, threshold, text_threshold):
        inputs = self._processor(
            images=image, text=grounding_prompt(prompts),
            return_tensors="pt",
        ).to(self.device)
        outputs = self._run(inputs)
        result = self._processor.post_process_grounded_object_detection(
            outputs, inputs["input_ids"], threshold=threshold,
            text_threshold=text_threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        labels = result.get("text_labels")
        if labels is None:
            labels = result["labels"]
        return [
            (box.tolist(), score.item(), str(label) or prompts[0])
            for box, score, label in zip(
                result["boxes"], result["scores"], labels
            )
        ]

    def _detect_owl(self, image, prompts, threshold):
        inputs = self._processor(
            text=[prompts], images=image, return_tensors="pt"
        ).to(self.device)
        outputs = self._run(inputs)
        # OWLv2 pads to a square; the processor scales boxes to the longer
        # side, and detect() clips them back to the real image. Recent
        # transformers versions name this method "..._grounded_...".
        post_process = getattr(
            self._processor, "post_process_grounded_object_detection",
            None,
        ) or self._processor.post_process_object_detection
        result = post_process(
            outputs, threshold=threshold,
            target_sizes=[(image.height, image.width)],
        )[0]
        return [
            (box.tolist(), score.item(), prompts[int(label)])
            for box, score, label in zip(
                result["boxes"], result["scores"], result["labels"]
            )
        ]


def draw_detections(image, detections, color="lime", width=3,
                    show_score=True):
    """A copy of ``image`` with each detection boxed and captioned."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=max(12, image.height // 45))
    for det in detections:
        x0, y0, x1, y1 = det.box
        draw.rectangle((x0, y0, x1, y1), outline=color, width=width)
        caption = f"{det.label} {det.score:.2f}" if show_score else det.label
        left, top, right, bottom = draw.textbbox((x0, y0), caption, font=font)
        draw.rectangle((left, top, right + 2, bottom + 2), fill=color)
        draw.text((x0 + 1, y0), caption, fill="black", font=font)
    return canvas
