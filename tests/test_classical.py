"""Tests for the classical segmentation methods (synthetic flower scenes)."""

import numpy as np
import pytest
from PIL import Image

from bouquet_checker import classical as cl
from bouquet_checker.segmentation import Instance

pytest.importorskip("cv2")

H, W = 120, 200


def disc(cx, cy, r, shape=(H, W)):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def scene(discs, colour=(230, 40, 140), background=(30, 140, 40)):
    """Coloured discs on a green background."""
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas[:] = background
    for cx, cy, r in discs:
        canvas[disc(cx, cy, r)] = colour
    return Image.fromarray(canvas)


def bright(discs):
    """White discs on black: strong grey-level and saturation-free contrast
    for the intensity-based methods (MSER, edges, blobs)."""
    return scene(discs, colour=(255, 255, 255), background=(20, 20, 20))


TWO = [(40, 40, 14), (110, 40, 14)]
TOUCHING = [(50, 90, 13), (72, 90, 13)]


def inst(mask, label="x"):
    return Instance(mask, 1.0, label)


def centres(instances):
    out = []
    for i in instances:
        box = i.box
        out.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
    return out


def covers(instances, x, y, r=14):
    """True if some instance lies mostly inside the disc at (x, y)."""
    target = disc(x, y, r)
    return any(np.logical_and(i.mask, target).sum() >= 0.6 * i.area
               for i in instances)


def near(instances, x, y, tol=5):
    return any(abs(cx - x) < tol and abs(cy - y) < tol
               for cx, cy in centres(instances))


# --- colour ------------------------------------------------------------------


def test_colour_mask_keeps_pink_and_drops_green():
    mask = cl.colour_mask(scene(TWO))
    assert mask[40, 40] and mask[40, 110]
    assert not mask[5, 5]                        # green background


def test_colour_mask_white_is_optional():
    canvas = np.full((20, 20, 3), 250, dtype=np.uint8)
    image = Image.fromarray(canvas)
    assert not cl.colour_mask(image).any()
    assert cl.colour_mask(image, include_white=True).all()


def test_colour_blobs_one_per_flower_and_min_area():
    found = cl.colour_blobs(scene(TWO))
    assert len(found) == 2
    assert near(found, 40, 40) and near(found, 110, 40)
    assert cl.colour_blobs(scene(TWO), min_area=10 ** 6) == []


def test_kmeans_and_meanshift_find_the_flowers():
    for method in (cl.kmeans_segments, cl.meanshift_segments):
        found = method(scene(TWO))
        assert near(found, 40, 40) and near(found, 110, 40), method


def test_kmeans_is_deterministic():
    a = cl.kmeans_segments(scene(TWO), k=4)
    b = cl.kmeans_segments(scene(TWO), k=4)
    assert len(a) == len(b)
    assert all((x.mask == y.mask).all() for x, y in zip(a, b))


# --- scikit-image methods ----------------------------------------------------


@pytest.mark.parametrize("method", [
    cl.felzenszwalb_segments, cl.slic_segments, cl.quickshift_segments,
])
def test_superpixel_methods_find_the_flowers(method):
    pytest.importorskip("skimage")
    found = method(scene(TWO))
    assert covers(found, 40, 40) and covers(found, 110, 40)


def test_slic_merge_reduces_the_number_of_regions():
    pytest.importorskip("skimage")
    image = scene(TWO)
    plain = cl.slic_segments(image, n_segments=100)
    merged = cl.slic_segments(image, n_segments=100, merge=40.0)
    assert len(merged) <= len(plain)


def test_watershed_splits_two_touching_flowers():
    pytest.importorskip("skimage")
    image = scene(TOUCHING)
    assert len(cl.colour_blobs(image)) == 1               # one merged blob
    found = cl.watershed_segments(image, min_distance=5)
    assert len(found) == 2
    assert near(found, 50, 90, 8) and near(found, 72, 90, 8)


def test_watershed_on_an_empty_scene_is_empty():
    pytest.importorskip("skimage")
    assert cl.watershed_segments(scene([])) == []


def test_dog_blobs_locate_the_flowers():
    pytest.importorskip("skimage")
    found = cl.dog_blobs(bright(TWO), min_sigma=4, max_sigma=12,
                         threshold=0.1, channel="grey")
    assert near(found, 40, 40, 6) and near(found, 110, 40, 6)


# --- shapes ------------------------------------------------------------------


def test_disc_masks_are_circles():
    mask = cl._disc((30, 30), 15, 15, 5)
    assert mask[15, 15] and mask[15, 20] and not mask[15, 21]
    assert 70 < mask.sum() < 90                            # ~ pi * 25


def test_hough_finds_a_clear_circle_and_handles_no_circles():
    image = scene([(100, 60, 20)], colour=(255, 255, 255),
                  background=(0, 0, 0))
    found = cl.hough_circles(image, param2=15, min_radius=10, max_radius=30)
    assert near(found, 100, 60, 4)
    assert cl.hough_circles(scene([])) == []


def test_mser_returns_deduplicated_regions():
    yy, xx = np.mgrid[:H, :W]                        # two smooth bright blobs
    grey = np.full((H, W), 20.0)
    for cx in (40, 110):
        grey += 200 * np.exp(-((xx - cx) ** 2 + (yy - 40) ** 2) / 128)
    grey = np.clip(grey, 0, 255).astype(np.uint8)
    image = Image.fromarray(np.stack([grey] * 3, axis=2))
    found = cl.mser_segments(image, delta=5)
    assert near(found, 40, 40) and near(found, 110, 40)
    assert len(found) == 2                                 # nested ones merged


def test_dedupe_drops_near_identical_masks_but_keeps_distinct_ones():
    a = inst(disc(30, 30, 10))
    b = inst(disc(31, 30, 10))                             # almost the same
    c = inst(disc(90, 30, 10))
    kept = cl.dedupe([a, b, c], iou=0.7)
    assert len(kept) == 2
    assert cl.dedupe([inst(np.zeros((H, W), bool))]) == []


def test_edge_contours_returns_regions():
    found = cl.edge_contours(bright(TWO), low=30, high=90)
    assert covers(found, 40, 40, 16) and covers(found, 110, 40, 16)


def test_grabcut_keeps_the_count_and_covers_the_flower():
    image = scene(TWO)
    seeds = cl.colour_blobs(image)
    refined = cl.grabcut_refine(image, seeds)
    assert len(refined) == len(seeds) == 2
    assert all(r.area > 0 for r in refined)
    assert near(refined, 40, 40) and near(refined, 110, 40)


def test_grabcut_skips_instances_it_cannot_refine():
    image = scene(TWO)
    tiny = inst(disc(100, 100, 1))
    assert cl.grabcut_refine(image, [tiny])[0] is tiny
    assert cl.grabcut_refine(image, [inst(np.zeros((H, W), bool))]) == []


# --- the shared filter -------------------------------------------------------


def test_flowerlike_keeps_a_compact_coloured_blob():
    image = scene(TWO)
    good = inst(disc(40, 40, 14))
    assert cl.flowerlike([good], image) == [good]


def test_flowerlike_drops_green_tiny_huge_grey_and_ragged():
    image = scene(TWO)
    green = inst(disc(150, 90, 10))                         # on the green bg
    tiny = inst(disc(40, 40, 1))
    huge = inst(np.ones((H, W), bool))
    ring = disc(40, 40, 14) & ~disc(40, 40, 9)         # hollow ring
    ragged = inst(ring)
    assert cl.flowerlike([green, tiny, huge, ragged], image) == []
    grey_image = Image.fromarray(np.full((H, W, 3), 128, dtype=np.uint8))
    assert cl.flowerlike([inst(disc(40, 40, 14))], grey_image) == []


def test_flowerlike_thresholds_can_be_tightened():
    image = scene(TWO)
    good = inst(disc(40, 40, 14))
    assert cl.flowerlike([good], image, max_area_frac=0.001) == []


# --- agreement ---------------------------------------------------------------


def test_agreement_perfect_partial_and_empty():
    ref = [inst(disc(40, 40, 14)), inst(disc(110, 40, 14))]
    perfect = cl.agreement(ref, [inst(disc(40, 40, 14)),
                                 inst(disc(110, 40, 14))])
    assert perfect == {"recall": 1.0, "precision": 1.0, "pixel_f1": 1.0,
                       "matched": 2}
    partial = cl.agreement(ref, [inst(disc(40, 40, 14)),
                                 inst(disc(60, 100, 5))])
    assert partial["recall"] == 0.5 and partial["precision"] == 0.5
    assert partial["matched"] == 1
    assert 0 < partial["pixel_f1"] < 1
    empty = cl.agreement(ref, [])
    assert empty == {"recall": 0.0, "precision": 0.0, "pixel_f1": 0.0,
                     "matched": 0}
    assert cl.agreement([], ref)["recall"] == 0.0


def test_agreement_is_one_to_one_and_respects_the_iou_threshold():
    ref = [inst(disc(40, 40, 14))]
    twins = [inst(disc(40, 40, 14)), inst(disc(40, 40, 14))]
    result = cl.agreement(ref, twins)
    assert result["recall"] == 1.0 and result["precision"] == 0.5
    shifted = [inst(disc(52, 40, 14))]                      # IoU well below .8
    assert cl.agreement(ref, shifted, iou=0.8)["matched"] == 0
    assert cl.agreement(ref, shifted, iou=0.2)["matched"] == 1


# --- optical flow -----------------------------------------------------------


def test_grayscale_shape_and_dtype():
    pytest.importorskip("cv2")
    grey = cl.grayscale(scene(TWO))
    assert grey.shape == (H, W) and grey.dtype == np.uint8


def test_dense_flow_shape_and_zero_for_identical_frames():
    pytest.importorskip("cv2")
    grey = cl.grayscale(scene(TWO))
    flow = cl.dense_flow(grey, grey)
    assert flow.shape == (H, W, 2) and flow.dtype == np.float32
    assert np.abs(flow).max() < 1e-3                       # no motion at all


def test_dense_flow_detects_a_clear_shift():
    cv2 = pytest.importorskip("cv2")
    # A flat-coloured disc gives no flow signal in its interior (the
    # aperture problem), so use a smooth, non-repeating gradient instead,
    # shifted by an exact affine warp.
    yy, xx = np.mgrid[:H, :W]
    ramp = ((xx * 1.3 + yy * 0.7) % 256).astype(np.uint8)  # no flat area
    a = cv2.GaussianBlur(ramp, (0, 0), 2)
    shifted = cv2.warpAffine(a, np.float32([[1, 0, 6], [0, 1, 0]]), (W, H),
                             borderMode=cv2.BORDER_REFLECT)
    region = np.zeros((H, W), bool)
    region[20:100, 20:180] = True                    # away from the border
    flow = cl.dense_flow(a, shifted)
    assert flow[region, 0].mean() > 0.2               # dx clearly positive
    assert abs(flow[region, 1].mean()) < 0.2          # little vertical drift


def test_warp_mask_with_zero_flow_is_unchanged():
    pytest.importorskip("cv2")
    mask = disc(40, 40, 14)
    flow = np.zeros((H, W, 2), dtype=np.float32)
    assert (cl.warp_mask(mask, flow) == mask).all()


def test_warp_mask_follows_a_constant_shift():
    pytest.importorskip("cv2")
    mask = disc(40, 40, 10)
    flow = np.zeros((H, W, 2), dtype=np.float32)
    flow[..., 0] = 20.0                                     # shift 20px right
    warped = cl.warp_mask(mask, flow)
    box = Instance(warped, 1.0, "x").box
    assert box is not None
    cx = (box[0] + box[2]) / 2
    assert 58 < cx < 62                              # 40 + 20, some slop
