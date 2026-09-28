"""Tests for the segmentation helpers and wrappers (models are faked)."""

import os
import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

from bouquet_checker import segmentation
from bouquet_checker.segmentation import (
    Instance,
    Sam2AutoMasker,
    Sam3AccessError,
    Sam3TextSegmenter,
    SegformerSemanticSegmenter,
    YoloSegmenter,
    YoloSemanticSegmenter,
    class_mask,
    find_class_ids,
    keep_by_semantic,
    load_results,
    mask_box,
    mask_iou,
    match_instances,
    overlay_instances,
    overlay_mask,
    remove_overlapping,
    save_results,
)


def rect(h, w, y0, y1, x0, x1):
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


# --- geometry -------------------------------------------------------------


def test_mask_box_is_tight_and_none_when_empty():
    assert mask_box(rect(20, 30, 5, 9, 10, 25)) == (10, 5, 25, 9)
    assert mask_box(np.zeros((4, 4), dtype=bool)) is None


def test_instance_area_and_box():
    inst = Instance(rect(20, 30, 5, 9, 10, 25), 0.9, "flower")
    assert inst.area == 4 * 15
    assert inst.box == (10, 5, 25, 9)


@pytest.mark.parametrize("a, b, expected", [
    (rect(10, 10, 0, 5, 0, 5), rect(10, 10, 0, 5, 0, 5), 1.0),
    (rect(10, 10, 0, 5, 0, 5), rect(10, 10, 6, 9, 6, 9), 0.0),
    (rect(10, 10, 0, 4, 0, 4), rect(10, 10, 0, 4, 2, 6), 8 / 24),
    (np.zeros((5, 5), dtype=bool), np.zeros((5, 5), dtype=bool), 0.0),
])
def test_mask_iou(a, b, expected):
    assert mask_iou(a, b) == pytest.approx(expected)


def test_match_instances_pairs_the_same_object_and_is_one_to_one():
    prev = [Instance(rect(50, 50, 0, 20, 0, 20), 1, "a"),
            Instance(rect(50, 50, 30, 50, 30, 50), 1, "b")]
    cur = [Instance(rect(50, 50, 31, 50, 31, 50), 1, "b"),     # swapped order
           Instance(rect(50, 50, 1, 20, 1, 20), 1, "a")]
    matches = match_instances(prev, cur, iou_threshold=0.5)
    assert sorted((i, j) for i, j, _ in matches) == [(0, 1), (1, 0)]
    assert match_instances(prev, [], 0.5) == []
    far = [Instance(rect(50, 50, 22, 28, 22, 28), 1, "c")]
    assert match_instances(prev, far, 0.5) == []


def test_find_class_ids_and_class_mask():
    names = {0: "wall", 66: "flower", 17: "plant", 5: "Flower"}
    assert find_class_ids(names, "flower") == [66, 5]        # case-insensitive
    class_map = np.array([[0, 66], [17, 66]])
    assert class_mask(class_map, [66]).tolist() == [[False, True],
                                                    [False, True]]
    assert class_mask(class_map, []).sum() == 0


def test_keep_by_semantic_filters_on_overlap_and_size():
    h = w = 100
    flower = rect(h, w, 0, 100, 0, 50)                 # left half is "flower"
    on = Instance(rect(h, w, 10, 30, 10, 30), 0.9, "object")      # inside
    off = Instance(rect(h, w, 10, 30, 60, 80), 0.9, "object")     # outside
    half = Instance(rect(h, w, 10, 30, 40, 60), 0.9, "object")    # 50% in
    tiny = Instance(rect(h, w, 1, 2, 1, 2), 0.9, "object")        # too small
    huge = Instance(rect(h, w, 0, 100, 0, 100), 0.9, "object")    # too big
    kept = keep_by_semantic([on, off, half, tiny, huge], flower,
                            min_fraction=0.5, min_area_frac=0.001,
                            max_area_frac=0.5)
    assert [k.area for k in kept] == [on.area, half.area]
    assert [round(k.score, 2) for k in kept] == [1.0, 0.5]
    assert on.score == 0.9                                   # originals intact


def test_remove_overlapping_drops_instances_covered_by_the_excluded_masks():
    h = w = 100
    flower = Instance(rect(h, w, 10, 30, 10, 30), 0.9, "flower")
    leaf_like = Instance(rect(h, w, 50, 70, 50, 70), 0.9, "flower")
    partly = Instance(rect(h, w, 10, 30, 25, 45), 0.9, "flower")     # 25% in
    leaf = Instance(rect(h, w, 45, 75, 45, 75), 0.9, "leaf")
    kept = remove_overlapping([flower, leaf_like, partly], [leaf])
    assert kept == [flower, partly]
    leaf_over_partly = Instance(rect(h, w, 10, 30, 30, 50), 0.9, "leaf")
    assert remove_overlapping([partly], [leaf_over_partly], 0.5) == []
    assert remove_overlapping([partly], [leaf_over_partly], 0.9) == [partly]


def test_remove_overlapping_without_exclusions_keeps_everything():
    inst = Instance(rect(10, 10, 0, 5, 0, 5), 0.9, "flower")
    assert remove_overlapping([inst], []) == [inst]


# --- drawing --------------------------------------------------------------


def test_overlay_mask_tints_only_the_mask_and_copies():
    image = Image.new("RGB", (20, 10), (100, 100, 100))
    out = overlay_mask(image, rect(10, 20, 2, 6, 5, 15), (255, 0, 0), 0.5)
    assert image.getpixel((8, 4)) == (100, 100, 100)          # original intact
    assert out.getpixel((8, 4)) == (177, 50, 50)              # tinted
    assert out.getpixel((1, 1)) == (100, 100, 100)            # outside


def test_overlay_instances_colours_each_instance_and_outlines():
    image = Image.new("RGB", (40, 40), (0, 0, 0))
    a = Instance(rect(40, 40, 2, 20, 2, 20), 1, "a")
    b = Instance(rect(40, 40, 22, 38, 22, 38), 1, "b")
    out = overlay_instances(image, [a, b], alpha=0.5, seed=1)
    assert out.size == image.size and out is not image
    assert out.getpixel((10, 10)) != out.getpixel((30, 30))   # different tints
    assert out.getpixel((0, 39)) == (0, 0, 0)                 # untouched
    empty = overlay_instances(image, [], alpha=0.5)
    assert empty.getpixel((5, 5)) == (0, 0, 0)


# --- YOLO wrappers ----------------------------------------------------------


class FakeTensor:
    """Just enough of a torch tensor: ``.cpu().numpy()``."""

    def __init__(self, array):
        self.array = np.asarray(array)

    def cpu(self):
        return self

    def numpy(self):
        return self.array


class FakeBoxes:
    def __init__(self, n):
        self.conf = FakeTensor(np.linspace(0.3, 0.9, n))
        self.cls = FakeTensor(np.arange(n) % 2)
        self._n = n

    def __len__(self):
        return self._n


def fake_result(n=2, h=8, w=10, names=None):
    masks = np.zeros((n, h, w), dtype=np.uint8)
    for i in range(n):
        masks[i, i:i + 3, i:i + 3] = 1
    return types.SimpleNamespace(
        masks=types.SimpleNamespace(data=FakeTensor(masks)),
        boxes=FakeBoxes(n),
        names=names or {0: "vase", 1: "potted plant"},
    )


class FakeYolo:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, image, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def test_yolo_segmenter_converts_results_best_first(monkeypatch):
    model = FakeYolo(fake_result(n=3))
    monkeypatch.setattr(
        YoloSegmenter, "_load", staticmethod(lambda weights, prompts: model)
    )
    found = YoloSegmenter("yolo26s-seg.pt").segment(Image.new("RGB", (10, 8)),
                                                    conf=0.2)
    assert [round(f.score, 1) for f in found] == [0.9, 0.6, 0.3]
    assert found[0].label in {"vase", "potted plant"}
    assert found[0].mask.dtype == bool and found[0].mask.shape == (8, 10)
    assert model.calls[0]["conf"] == 0.2
    assert model.calls[0]["retina_masks"] is True         # full-size masks


def test_yolo_segmenter_returns_nothing_when_no_masks(monkeypatch):
    empty = types.SimpleNamespace(masks=None, boxes=None, names={})
    monkeypatch.setattr(
        YoloSegmenter, "_load",
        staticmethod(lambda weights, prompts: FakeYolo(empty)),
    )
    assert YoloSegmenter("x.pt").segment(Image.new("RGB", (4, 4))) == []


def test_yolo_semantic_segmenter_returns_class_map_and_names(monkeypatch):
    class_map = np.array([[0, 66], [17, 66]], dtype=np.uint8)
    result = types.SimpleNamespace(
        semantic_mask=types.SimpleNamespace(data=FakeTensor(class_map)),
        names={0: "wall", 17: "plant", 66: "flower"},
    )
    monkeypatch.setattr(
        YoloSemanticSegmenter, "_load",
        staticmethod(lambda weights: FakeYolo(result)),
    )
    seg, names = YoloSemanticSegmenter().segment(Image.new("RGB", (2, 2)))
    assert seg.tolist() == class_map.tolist()
    assert find_class_ids(names, "flower") == [66]


def test_ultralytics_helper_explains_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "ultralytics", None)      # not importable
    with pytest.raises(ImportError, match="--no-deps"):
        segmentation._ultralytics()


def test_working_dir_is_restored_even_on_error(tmp_path):
    before = os.getcwd()
    with pytest.raises(RuntimeError):
        with segmentation._working_dir(tmp_path):
            assert os.getcwd() == str(tmp_path)
            raise RuntimeError("boom")
    assert os.getcwd() == before


def test_yoloe_prompts_are_set_from_the_models_directory(monkeypatch,
                                                         tmp_path):
    seen = {}

    class FakeYoloe:
        def __init__(self, path):
            seen["path"] = path

        def set_classes(self, names):
            seen["names"] = names
            seen["cwd"] = os.getcwd()

    fake = types.SimpleNamespace(YOLOE=FakeYoloe, YOLO=None,
                                 settings=types.SimpleNamespace(
                                     update=lambda d: None))
    monkeypatch.setattr(segmentation, "ULTRALYTICS_DIR", tmp_path)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    YoloSegmenter("yoloe-26s-seg.pt", prompts=("flower",))
    assert seen["names"] == ["flower"]
    assert seen["cwd"] == str(tmp_path)              # text encoder lands here
    assert seen["path"] == str(tmp_path / "yoloe-26s-seg.pt")


# --- Hugging Face wrappers ---------------------------------------------


def test_segformer_returns_class_map_and_int_keyed_names(monkeypatch):
    class Processor:
        def __call__(self, images, return_tensors):
            return {"pixel_values": torch.zeros(1, 3, 4, 4)}

        def post_process_semantic_segmentation(self, outputs, target_sizes):
            assert target_sizes == [(6, 8)]
            return [torch.full((6, 8), 66)]

    class Model:
        config = types.SimpleNamespace(id2label={"66": "flower", 0: "wall"})

        def __call__(self, pixel_values):
            return object()

    monkeypatch.setattr(
        SegformerSemanticSegmenter, "_load",
        staticmethod(lambda model_id: (Processor(), Model())),
    )
    class_map, names = SegformerSemanticSegmenter().segment(
        Image.new("RGB", (8, 6)))
    assert class_map.shape == (6, 8) and (class_map == 66).all()
    assert names == {66: "flower", 0: "wall"}


def test_sam2_auto_masker_converts_and_sorts(monkeypatch):
    masks = [torch.zeros(6, 8, dtype=torch.bool) for _ in range(3)]
    for i, mask in enumerate(masks):
        mask[i:i + 2, i:i + 2] = True
    calls = {}

    def pipe(image, **kwargs):
        calls.update(kwargs)
        return {"masks": masks, "scores": torch.tensor([0.5, 0.9, 0.7])}

    monkeypatch.setattr(Sam2AutoMasker, "_load", staticmethod(lambda m: pipe))
    found = Sam2AutoMasker().segment(Image.new("RGB", (8, 6)),
                                     points_per_crop=8)
    assert [round(f.score, 1) for f in found] == [0.9, 0.7, 0.5]
    assert found[0].mask.dtype == bool and found[0].label == "object"
    assert calls["points_per_crop"] == 8


def test_sam3_without_a_token_explains_how_to_get_access(monkeypatch):
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    with pytest.raises(Sam3AccessError, match="huggingface.co/facebook/sam3"):
        Sam3TextSegmenter()


def test_sam3_with_a_token_but_no_approval_gives_the_same_help(monkeypatch):
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_x")

    def gated(model_id):
        raise OSError("You are trying to access a gated repo.")

    monkeypatch.setattr(Sam3TextSegmenter, "_load", staticmethod(gated))
    with pytest.raises(Sam3AccessError, match="approval"):
        Sam3TextSegmenter()


def test_sam3_segments_a_text_concept(monkeypatch):
    class Processor:
        def __call__(self, images, text, return_tensors):
            self.text = text
            return {"original_sizes": torch.tensor([[6, 8]])}

        def post_process_instance_segmentation(
            self, outputs, threshold, mask_threshold, target_sizes
        ):
            assert target_sizes == [[6, 8]]
            return [{"masks": torch.ones(2, 6, 8, dtype=torch.bool),
                     "scores": torch.tensor([0.6, 0.8])}]

    class Model:
        def __call__(self, **inputs):
            return object()

    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_x")
    processor = Processor()
    monkeypatch.setattr(
        Sam3TextSegmenter, "_load",
        staticmethod(lambda model_id: (processor, Model())),
    )
    found = Sam3TextSegmenter().segment(Image.new("RGB", (8, 6)), "flower")
    assert processor.text == "flower"
    assert [f.label for f in found] == ["flower", "flower"]
    assert [round(f.score, 1) for f in found] == [0.8, 0.6]


def test_sam3_segment_many_encodes_the_image_once_for_all_prompts(
    monkeypatch,
):
    calls = {"vision": 0, "decode": []}

    class Processor:
        def __call__(self, images=None, text=None, return_tensors=None):
            if images is not None:
                return {"pixel_values": torch.zeros(1, 3, 4, 4),
                        "original_sizes": torch.tensor([[6, 8]])}
            return {"input_ids": torch.tensor([[len(text)]]),
                    "attention_mask": torch.ones(1, 1)}

        def post_process_instance_segmentation(
            self, outputs, threshold, mask_threshold, target_sizes
        ):
            assert target_sizes == [[6, 8]]
            n = outputs["n"]
            return [{"masks": torch.ones(n, 6, 8, dtype=torch.bool),
                     "scores": torch.linspace(0.9, 0.5, n)}]

    class Model:
        def get_vision_features(self, pixel_values):
            calls["vision"] += 1
            return "VISION"

        def __call__(self, vision_embeds, input_ids, attention_mask):
            assert vision_embeds == "VISION"
            calls["decode"].append(int(input_ids[0, 0]))
            return {"n": int(input_ids[0, 0])}

    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_x")
    monkeypatch.setattr(
        Sam3TextSegmenter, "_load",
        staticmethod(lambda model_id: (Processor(), Model())),
    )
    found = Sam3TextSegmenter().segment_many(
        Image.new("RGB", (8, 6)), ["ab", "abc"])
    assert calls["vision"] == 1                       # encoded once
    assert calls["decode"] == [2, 3]                  # one decode per prompt
    assert [len(found["ab"]), len(found["abc"])] == [2, 3]
    assert [round(i.score, 1) for i in found["abc"]] == [0.9, 0.7, 0.5]
    assert found["ab"][0].label == "ab"


def test_results_cache_round_trip_including_empty_prompts(tmp_path):
    mask = rect(6, 8, 1, 4, 2, 6)
    results = {
        "flower": [Instance(mask, 0.9, "flower"),
                   Instance(mask.copy(), 0.4, "flower")],
        "lily": [],                                        # nothing found
        "chrysanthemum": [Instance(mask, 0.7, "chrysanthemum")],
    }
    path = tmp_path / "frame.npz"
    save_results(path, results)
    back = load_results(path)
    assert list(back) == ["flower", "lily", "chrysanthemum"]
    assert [round(i.score, 1) for i in back["flower"]] == [0.9, 0.4]
    assert back["lily"] == []
    assert back["chrysanthemum"][0].label == "chrysanthemum"
    assert (back["flower"][0].mask == mask).all()
    assert back["flower"][0].mask.dtype == bool


def test_sam3_segment_many_can_downscale_masks(monkeypatch):
    class Processor:
        def __call__(self, images=None, text=None, return_tensors=None):
            if images is not None:
                return {"pixel_values": torch.zeros(1, 3, 4, 4),
                        "original_sizes": torch.tensor([[6, 8]])}
            return {"input_ids": torch.tensor([[1]])}

        def post_process_instance_segmentation(self, outputs, **kwargs):
            return [{"masks": torch.ones(2, 6, 8, dtype=torch.bool),
                     "scores": torch.tensor([0.9, 0.6])}]

    class Model:
        def get_vision_features(self, pixel_values):
            return "V"

        def __call__(self, **kwargs):
            return {}

    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_x")
    monkeypatch.setattr(
        Sam3TextSegmenter, "_load",
        staticmethod(lambda model_id: (Processor(), Model())),
    )
    found = Sam3TextSegmenter().segment_many(
        Image.new("RGB", (8, 6)), ["x"], downscale=2)
    assert found["x"][0].mask.shape == (3, 4)


# --- "segment everything" wrappers (FastSAM, MobileSAM) --------------------


def polygon_result(polygons, height=30, width=60, conf=None):
    conf = np.linspace(0.3, 0.9, len(polygons)) if conf is None else conf
    return types.SimpleNamespace(
        orig_shape=(height, width),
        masks=types.SimpleNamespace(xy=[np.array(p, float) for p in polygons]),
        boxes=types.SimpleNamespace(conf=FakeTensor(conf),
                                    __len__=lambda: len(polygons)),
    )


class CountedBoxes:
    def __init__(self, conf):
        self.conf = FakeTensor(conf)

    def __len__(self):
        return len(self.conf.array)


def test_polygons_become_downscaled_masks_best_first():
    pytest.importorskip("cv2")
    square = [(6, 6), (24, 6), (24, 18), (6, 18)]
    other = [(30, 3), (45, 3), (45, 12)]
    result = polygon_result([square, other])
    result.boxes = CountedBoxes([0.4, 0.8])
    found = segmentation._instances_from_polygons(result, 3, "demo")
    assert [round(f.score, 1) for f in found] == [0.8, 0.4]
    assert found[0].mask.shape == (10, 20)                # 30x60 / 3
    assert found[0].label == "demo" and found[0].mask.dtype == bool
    x0, y0, x1, y1 = found[1].box                          # the square
    assert (x0, y0) == (2, 2) and 8 <= x1 <= 9 and 6 <= y1 <= 7


def test_polygons_skip_degenerate_shapes_and_empty_results():
    pytest.importorskip("cv2")
    result = polygon_result([[(1, 1), (2, 2)]])            # a line: skipped
    result.boxes = CountedBoxes([0.5])
    assert segmentation._instances_from_polygons(result, 1, "x") == []
    none = types.SimpleNamespace(orig_shape=(4, 4), masks=None, boxes=None)
    assert segmentation._instances_from_polygons(none, 1, "x") == []


class FakeEverything:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


def test_fastsam_passes_thresholds_and_returns_instances(monkeypatch):
    pytest.importorskip("cv2")
    result = polygon_result([[(6, 6), (24, 6), (24, 18), (6, 18)]])
    result.boxes = CountedBoxes([0.7])
    model = FakeEverything(result)
    monkeypatch.setattr(segmentation.FastSamSegmenter, "_load",
                        staticmethod(lambda weights: model))
    found = segmentation.FastSamSegmenter().segment(
        Image.new("RGB", (60, 30)), imgsz=640, conf=0.2, iou=0.7)
    assert len(found) == 1 and found[0].label == "fastsam"
    assert model.calls[0]["imgsz"] == 640
    assert model.calls[0]["conf"] == 0.2 and model.calls[0]["iou"] == 0.7
    assert model.calls[0]["device"] == "cpu"


def test_mobilesam_prompts_a_regular_grid_of_points(monkeypatch):
    pytest.importorskip("cv2")
    result = polygon_result([[(6, 6), (24, 6), (24, 18), (6, 18)]])
    result.boxes = CountedBoxes([0.7])
    model = FakeEverything(result)
    monkeypatch.setattr(segmentation.MobileSamSegmenter, "_load",
                        staticmethod(lambda weights: model))
    found = segmentation.MobileSamSegmenter().segment(
        Image.new("RGB", (60, 30)), grid=4)
    call = model.calls[0]
    assert len(call["points"]) == 16 and call["labels"] == [1] * 16
    xs = sorted({p[0] for p in call["points"]})
    ys = sorted({p[1] for p in call["points"]})
    assert xs == [7.5, 22.5, 37.5, 52.5] and ys == [3.75, 11.25, 18.75, 26.25]
    assert found[0].label == "mobilesam"


# --- tracking: linking ids across frames, colouring by id -------------------


def box_inst(x0, y0, x1, y1, score=1.0, label="x", shape=(20, 30)):
    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return Instance(mask, score, label)


def test_link_tracks_keeps_ids_across_matches_and_assigns_new_ones():
    frame1 = [box_inst(0, 0, 5, 5), box_inst(20, 10, 25, 15)]
    frame2 = [box_inst(1, 0, 6, 5),                    # matches frame1[0]
             box_inst(20, 10, 25, 15),                 # matches frame1[1]
             box_inst(10, 10, 14, 14)]                 # brand new
    tracks, next_id = segmentation.link_tracks([frame1, frame2])
    assert list(tracks[0]) == [0, 1]
    assert set(tracks[1]) == {0, 1, 2}                       # ids 0, 1 kept
    assert next_id == 3
    assert tracks[1][0] is frame2[0] and tracks[1][1] is frame2[1]
    assert tracks[1][2] is frame2[2]


def test_link_tracks_ends_a_track_with_no_match():
    frame1 = [box_inst(0, 0, 5, 5), box_inst(20, 10, 25, 15)]
    frame2 = [box_inst(20, 10, 25, 15)]                # frame1[0] is gone
    tracks, next_id = segmentation.link_tracks([frame1, frame2])
    assert set(tracks[1]) == {1}
    assert next_id == 2


def test_link_tracks_respects_the_iou_threshold_and_start_id():
    frame1 = [box_inst(0, 0, 5, 5)]
    frame2 = [box_inst(3, 0, 8, 5)]                          # partial overlap
    loose, _ = segmentation.link_tracks([frame1, frame2], iou_threshold=0.2)
    strict, next_id = segmentation.link_tracks([frame1, frame2],
                                                iou_threshold=0.9, start_id=5)
    assert list(loose[1]) == [0]                       # matched: id kept
    assert list(strict[0]) == [5]                      # start_id honoured
    assert list(strict[1]) == [6]                      # no match: new id
    assert next_id == 7


def test_link_tracks_handles_an_empty_sequence():
    tracks, next_id = segmentation.link_tracks([])
    assert tracks == [] and next_id == 0
    tracks, next_id = segmentation.link_tracks([[]])
    assert tracks == [{}] and next_id == 0


def test_track_color_is_deterministic_and_varies_with_id():
    assert segmentation.track_color(3) == segmentation.track_color(3)
    colors = {segmentation.track_color(i) for i in range(6)}
    assert len(colors) == 6
    for channel in segmentation.track_color(1):
        assert 0 <= channel <= 255


def test_overlay_tracks_paints_each_id_its_own_colour():
    image = Image.new("RGB", (20, 20), (0, 0, 0))
    left = np.zeros((20, 20), dtype=bool)
    left[5:10, 2:6] = True
    right = np.zeros((20, 20), dtype=bool)
    right[5:10, 14:18] = True
    out = np.asarray(segmentation.overlay_tracks(image, {0: left, 1: right},
                                                 alpha=1.0, outline=False))
    assert tuple(out[7, 4]) == segmentation.track_color(0)
    assert tuple(out[7, 16]) == segmentation.track_color(1)
    assert tuple(out[0, 0]) == (0, 0, 0)                     # untouched


def test_overlay_tracks_is_consistent_across_two_calls():
    image = Image.new("RGB", (10, 10))
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    a = np.asarray(segmentation.overlay_tracks(image, {7: mask}))
    b = np.asarray(segmentation.overlay_tracks(image, {7: mask}))
    assert (a == b).all()
