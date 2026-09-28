"""Tests for the open-vocabulary detection wrapper (models are faked)."""

import pytest
import torch
from PIL import Image

from bouquet_checker import detection
from bouquet_checker.detection import (
    Detection,
    OpenVocabDetector,
    box_iou,
    clip_box,
    draw_detections,
    grounding_prompt,
    match_boxes,
    non_max_suppression,
)


class Batch(dict):
    """Minimal stand-in for a processor's BatchEncoding."""

    __getattr__ = dict.__getitem__

    def to(self, device):
        return self


class FakeGroundingProcessor:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, images, text, return_tensors):
        self.calls.append(("call", text))
        return Batch(input_ids=torch.tensor([[1, 2, 3]]), pixel_values=0)

    def post_process_grounded_object_detection(
        self, outputs, input_ids, threshold, text_threshold, target_sizes
    ):
        self.calls.append(("post", threshold, text_threshold, target_sizes))
        return [self.result]


class FakeOwlProcessor:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, text, images, return_tensors):
        self.calls.append(("call", text))
        return Batch(pixel_values=0)

    # Recent transformers: OWL processors expose the "grounded" name.
    def post_process_grounded_object_detection(
        self, outputs, threshold, target_sizes
    ):
        self.calls.append(("post", threshold, target_sizes))
        return [self.result]


class FakeModel:
    def __call__(self, **inputs):
        return object()


def make_detector(monkeypatch, processor, family):
    loaded = (processor, FakeModel(), family)
    monkeypatch.setattr(
        OpenVocabDetector, "_load", staticmethod(lambda *args: loaded)
    )
    return OpenVocabDetector("fake/model")


IMAGE = Image.new("RGB", (200, 100))


# --- pure helpers ---------------------------------------------------------


def test_detection_geometry():
    det = Detection((10, 20, 40, 60), 0.9, "flower")
    assert (det.width, det.height, det.area) == (30, 40, 1200)


@pytest.mark.parametrize("box, expected", [
    ((10, 10, 50, 50), (10, 10, 50, 50)),
    ((-5, -5, 300, 200), (0, 0, 200, 100)),           # spills on every side
    ((150, 60, 260, 180), (150, 60, 200, 100)),
])
def test_clip_box(box, expected):
    assert clip_box(box, 200, 100) == expected


def test_grounding_prompt_format():
    assert grounding_prompt(["Flower", " a rose. "]) == "flower. a rose."


@pytest.mark.parametrize("bad", [[], [""], ["flower", "  "]])
def test_grounding_prompt_rejects_empty_prompts(bad):
    with pytest.raises(ValueError):
        grounding_prompt(bad)


def test_nms_keeps_the_best_of_overlapping_boxes_across_labels():
    dets = [
        Detection((0, 0, 100, 100), 0.6, "flower"),
        Detection((5, 5, 100, 100), 0.9, "rose"),           # overlaps, better
        Detection((200, 200, 260, 260), 0.5, "flower"),     # separate
    ]
    kept = non_max_suppression(dets, iou_threshold=0.5)
    assert [(d.label, d.score) for d in kept] == [("rose", 0.9),
                                                  ("flower", 0.5)]


def test_nms_handles_zero_or_one_box():
    assert non_max_suppression([]) == []
    one = [Detection((0, 0, 5, 5), 0.5, "x")]
    assert non_max_suppression(one) == one


def test_draw_detections_returns_a_marked_copy():
    image = Image.new("RGB", (120, 80), "black")
    out = draw_detections(image, [Detection((10, 10, 60, 50), 0.8, "flower")])
    assert out.size == image.size and out is not image
    assert image.getpixel((10, 30)) == (0, 0, 0)          # original untouched
    assert out.getpixel((10, 30)) != (0, 0, 0)            # box edge drawn


# --- backends (processor and model are faked) --------------------------------


def test_grounding_backend_formats_the_prompt_and_maps_results(monkeypatch):
    result = {
        "boxes": torch.tensor([[10.0, 10.0, 60.0, 50.0]]),
        "scores": torch.tensor([0.7]),
        "text_labels": ["flower"],
    }
    processor = FakeGroundingProcessor(result)
    detector = make_detector(monkeypatch, processor, "grounding")
    found = detector.detect(IMAGE, ["Flower", "rose"], threshold=0.3)
    assert processor.calls[0] == ("call", "flower. rose.")
    assert processor.calls[1] == ("post", 0.3, 0.3, [(100, 200)])
    assert [(d.label, round(d.score, 2)) for d in found] == [("flower", 0.7)]
    assert found[0].box == (10.0, 10.0, 60.0, 50.0)


def test_grounding_text_threshold_is_passed_separately(monkeypatch):
    processor = FakeGroundingProcessor({
        "boxes": torch.zeros(0, 4), "scores": torch.zeros(0),
        "text_labels": [],
    })
    detector = make_detector(monkeypatch, processor, "grounding")
    assert detector.detect(IMAGE, ["flower"], 0.4, text_threshold=0.2) == []
    assert processor.calls[1][1:3] == (0.4, 0.2)


def test_owl_backend_maps_label_indices_and_clips_padded_boxes(monkeypatch):
    # A 200x100 image is padded to 200x200, so boxes can extend below it.
    result = {
        "boxes": torch.tensor([[20.0, 30.0, 90.0, 150.0],     # spills down
                               [0.0, 0.0, 0.4, 0.4]]),        # < 1 px: dropped
        "scores": torch.tensor([0.8, 0.9]),
        "labels": torch.tensor([1, 0]),
    }
    processor = FakeOwlProcessor(result)
    detector = make_detector(monkeypatch, processor, "owl")
    found = detector.detect(IMAGE, ["flower", "rose"], threshold=0.2)
    assert processor.calls[0] == ("call", [["flower", "rose"]])
    assert processor.calls[1] == ("post", 0.2, [(100, 200)])
    assert len(found) == 1
    assert found[0].label == "rose"
    assert found[0].box == (20.0, 30.0, 90.0, 100.0)      # clipped to height


def test_detections_come_back_best_first_and_deduplicated(monkeypatch):
    result = {
        "boxes": torch.tensor([[0.0, 0.0, 50.0, 50.0],
                               [1.0, 1.0, 50.0, 50.0],
                               [100.0, 10.0, 150.0, 60.0]]),
        "scores": torch.tensor([0.5, 0.8, 0.6]),
        "labels": torch.tensor([0, 0, 0]),
    }
    detector = make_detector(monkeypatch, FakeOwlProcessor(result), "owl")
    found = detector.detect(IMAGE, ["flower"])
    assert [round(d.score, 1) for d in found] == [0.8, 0.6]


@pytest.mark.parametrize("prompts", [[], [""], ["  "]])
def test_detect_rejects_empty_prompts(monkeypatch, prompts):
    detector = make_detector(monkeypatch, FakeOwlProcessor({}), "owl")
    with pytest.raises(ValueError):
        detector.detect(IMAGE, prompts)


def test_unsupported_model_types_are_rejected(monkeypatch):
    class Config:
        model_type = "detr"

    class Model:
        config = Config()

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_transformers = type("T", (), {})()
    fake_transformers.AutoProcessor = type(
        "P", (), {"from_pretrained": staticmethod(lambda *a, **k: object())}
    )
    fake_transformers.AutoModelForZeroShotObjectDetection = type(
        "M", (), {"from_pretrained": staticmethod(lambda *a, **k: Model())}
    )
    monkeypatch.setitem(__import__("sys").modules, "transformers",
                        fake_transformers)
    with pytest.raises(ValueError, match="Unsupported"):
        detection.OpenVocabDetector._load("x/y", "cpu")


def test_owl_backend_falls_back_to_the_older_method_name(monkeypatch):
    class OldProcessor(FakeOwlProcessor):
        post_process_grounded_object_detection = None

        def post_process_object_detection(
            self, outputs, threshold, target_sizes
        ):
            self.calls.append(("post-old", threshold))
            return [self.result]

    result = {
        "boxes": torch.tensor([[10.0, 10.0, 60.0, 60.0]]),
        "scores": torch.tensor([0.7]),
        "labels": torch.tensor([0]),
    }
    processor = OldProcessor(result)
    detector = make_detector(monkeypatch, processor, "owl")
    assert len(detector.detect(IMAGE, ["flower"], threshold=0.3)) == 1
    assert ("post-old", 0.3) in processor.calls


# --- box overlap and matching (the seed of tracking) -------------------------


@pytest.mark.parametrize("a, b, expected", [
    ((0, 0, 10, 10), (0, 0, 10, 10), 1.0),
    ((0, 0, 10, 10), (20, 20, 30, 30), 0.0),               # disjoint
    ((0, 0, 10, 10), (5, 0, 15, 10), 50 / 150),            # half overlap
    ((0, 0, 10, 10), (0, 0, 5, 10), 0.5),                  # contained
    ((0, 0, 0, 0), (0, 0, 0, 0), 0.0),                     # degenerate
])
def test_box_iou(a, b, expected):
    assert box_iou(a, b) == pytest.approx(expected)


def test_match_boxes_pairs_the_same_object_across_frames():
    previous = [(0, 0, 100, 100), (200, 200, 300, 300)]
    current = [(205, 198, 305, 302), (2, 3, 102, 101)]     # order swapped
    matches = match_boxes(previous, current)
    assert sorted((i, j) for i, j, _ in matches) == [(0, 1), (1, 0)]


def test_match_boxes_is_one_to_one_and_prefers_the_best_overlap():
    previous = [(0, 0, 100, 100)]
    current = [(30, 0, 130, 100), (5, 0, 105, 100)]        # second is closer
    matches = match_boxes(previous, current, iou_threshold=0.3)
    assert [(i, j) for i, j, _ in matches] == [(0, 1)]


def test_match_boxes_respects_the_threshold_and_empty_input():
    previous = [(0, 0, 100, 100)]
    assert match_boxes(previous, [(60, 0, 160, 100)], 0.5) == []
    assert match_boxes([], previous) == []
    assert match_boxes(previous, []) == []
