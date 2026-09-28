"""Classical (non-deep) segmentation of flower-head candidates.

Cheap methods that could run close to real time: colour thresholds, k-means,
mean-shift, graph-based / superpixel / watershed segmentation, MSER, Hough
circles, blob detectors, contours and GrabCut. None of them knows what a flower
is: each one returns *segments* (``segmentation.Instance``, label = method
name), and ``flowerlike`` is the one shared, deliberately simple filter that
turns segments into flower-head candidates (not green, not tiny, not huge,
compact). ``agreement`` compares candidates with another set of masks, for
example SAM 3's, which is a reference to compare with, not the truth.

OpenCV, scipy and scikit-image are imported lazily; the base install needs
none of them.
"""

import numpy as np
from PIL import Image

from bouquet_checker.segmentation import Instance

# OpenCV hue runs 0-179; leaves and stems sit around 35-85.
GREEN_HUES = (30, 90)


def _rgb(image):
    """``(H, W, 3)`` uint8 RGB array of a PIL image (or array)."""
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
    return np.ascontiguousarray(np.asarray(image, dtype=np.uint8))


def _hsv(rgb):
    import cv2

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def _components(mask, min_area, label, score=1.0, connectivity=8):
    """One ``Instance`` per connected blob of a boolean mask."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=connectivity
    )
    return [
        Instance(labels == i, float(score), label)
        for i in range(1, count)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]


def _from_labels(labels, min_area, label, skip=None):
    """Instances from an integer label map (each label split into blobs)."""
    found = []
    for value in np.unique(labels):
        if skip is not None and value == skip:
            continue
        found.extend(_components(labels == value, min_area, label))
    return found


def colour_mask(image, sat_min=80, val_min=60, exclude_green=True,
                include_white=False):
    """Boolean "coloured, not green" mask in HSV.

    Pixels count if saturation >= ``sat_min`` and value >= ``val_min`` and (if
    ``exclude_green``) the hue is not green. ``include_white`` adds bright,
    unsaturated pixels (white flowers, but also white wrapping paper).
    """
    hsv = _hsv(_rgb(image))
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (sat >= sat_min) & (val >= val_min)
    if exclude_green:
        mask &= ~((hue >= GREEN_HUES[0]) & (hue <= GREEN_HUES[1]))
    if include_white:
        mask |= (sat < 40) & (val > 215)
    return mask


def grayscale(image):
    """``(H, W)`` uint8 grey-level array of a PIL image (or array)."""
    import cv2

    return cv2.cvtColor(_rgb(image), cv2.COLOR_RGB2GRAY)


def _open(mask, size):
    import cv2

    if size <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN,
                            kernel).astype(bool)


# --- colour --------------------------------------------------------------


def colour_blobs(image, sat_min=80, val_min=60, exclude_green=True,
                 include_white=False, open_size=3, min_area=20):
    """Threshold in HSV, clean up, split into connected blobs."""
    mask = colour_mask(image, sat_min, val_min, exclude_green, include_white)
    mask = _open(mask, open_size)
    return _components(mask, min_area, "colour blob")


def kmeans_segments(image, k=6, min_area=20, seed=0):
    """K-means on Lab colour, each cluster split into connected regions."""
    import cv2

    lab = cv2.cvtColor(_rgb(image), cv2.COLOR_RGB2LAB)
    data = lab.reshape(-1, 3).astype(np.float32)
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, _ = cv2.kmeans(data, k, None, criteria, 1,
                              cv2.KMEANS_PP_CENTERS)
    return _from_labels(labels.reshape(lab.shape[:2]), min_area, "k-means")


def meanshift_segments(image, sp=8, sr=16, quant=24, min_area=20):
    """Mean-shift filter (edge-preserving smoothing), then group pixels whose
    smoothed Lab colour falls in the same ``quant``-wide bin."""
    import cv2

    bgr = cv2.cvtColor(_rgb(image), cv2.COLOR_RGB2BGR)
    smooth = cv2.pyrMeanShiftFiltering(bgr, sp, sr, maxLevel=1)
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB).astype(np.int32)
    cells = lab // quant
    key = (cells[..., 0] * 64 + cells[..., 1]) * 64 + cells[..., 2]
    return _from_labels(key, min_area, "mean-shift")


# --- graph, superpixels, watershed (scikit-image) --------------------------


def felzenszwalb_segments(image, scale=100, sigma=0.8, min_size=20):
    """Felzenszwalb graph-based segmentation (larger ``scale``: bigger
    regions)."""
    from skimage.segmentation import felzenszwalb

    labels = felzenszwalb(_rgb(image), scale=scale, sigma=sigma,
                          min_size=min_size)
    return _from_labels(labels, min_size, "felzenszwalb")


def slic_segments(image, n_segments=200, compactness=10.0, merge=0.0,
                  min_area=20):
    """SLIC superpixels; with ``merge`` > 0 neighbouring superpixels whose
    mean colours are within ``merge`` are merged (region-adjacency graph)."""
    from skimage.segmentation import slic

    rgb = _rgb(image)
    labels = slic(rgb, n_segments=n_segments, compactness=compactness,
                  start_label=1)
    if merge > 0:
        from skimage import graph

        rag = graph.rag_mean_color(rgb, labels)
        labels = graph.cut_threshold(labels, rag, merge)
    return _from_labels(labels, min_area, "slic")


def quickshift_segments(image, kernel_size=3, max_dist=6.0, ratio=0.5,
                        min_area=20):
    """Quick shift: mode-seeking in the joint colour + position space."""
    from skimage.segmentation import quickshift

    labels = quickshift(_rgb(image), kernel_size=kernel_size,
                        max_dist=max_dist, ratio=ratio)
    return _from_labels(labels, min_area, "quickshift")


def watershed_segments(image, sat_min=80, min_distance=4, include_white=False,
                       open_size=3, min_area=20):
    """Split touching flowers: colour mask -> distance transform -> one marker
    per local maximum (at least ``min_distance`` px apart) -> watershed."""
    import cv2
    from scipy import ndimage
    from skimage.segmentation import watershed

    mask = _open(colour_mask(image, sat_min, include_white=include_white),
                 open_size)
    if not mask.any():
        return []
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    peaks = (dist == ndimage.maximum_filter(
        dist, footprint=np.ones((2 * min_distance + 1,) * 2)
    )) & (dist >= 1.5)
    markers, _ = ndimage.label(peaks)
    labels = watershed(-dist, markers, mask=mask)
    return _from_labels(labels, min_area, "watershed", skip=0)


# --- shapes ---------------------------------------------------------------


def _disc(shape, x, y, radius):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2


def dedupe(instances, iou=0.7):
    """Drop instances that overlap an earlier (larger) one by more than
    ``iou``: MSER and friends return many nested duplicates."""
    kept = []
    for inst in sorted(instances, key=lambda i: -i.area):
        box = inst.box
        if box is None:
            continue
        duplicate = False
        for other in kept:
            ob = other.box
            if (box[2] <= ob[0] or ob[2] <= box[0]
                    or box[3] <= ob[1] or ob[3] <= box[1]):
                continue
            inter = np.logical_and(inst.mask, other.mask).sum()
            union = inst.area + other.area - inter
            if union and inter / union > iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(inst)
    return kept


def mser_segments(image, delta=5, min_area=30, max_area_frac=0.05,
                  dedupe_iou=0.7):
    """MSER: maximally stable extremal regions of the grey image."""
    import cv2

    gray = cv2.cvtColor(_rgb(image), cv2.COLOR_RGB2GRAY)
    height, width = gray.shape
    mser = cv2.MSER_create(
        delta=delta, min_area=min_area,
        max_area=int(max_area_frac * height * width),
    )
    regions, _ = mser.detectRegions(gray)
    found = []
    for points in regions:
        mask = np.zeros(gray.shape, dtype=bool)
        mask[points[:, 1], points[:, 0]] = True
        found.append(Instance(mask, 1.0, "mser"))
    return dedupe(found, dedupe_iou)


def hough_circles(image, param2=25, min_radius=4, max_radius=30, dp=1.2,
                  min_dist=6, param1=100):
    """Hough circle transform on the blurred grey image: each circle is a
    disc-shaped candidate (``param2`` is the accumulator threshold: lower
    finds more, and more false, circles)."""
    import cv2

    rgb = _rgb(image)
    gray = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 5)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist, param1=param1,
        param2=param2, minRadius=min_radius, maxRadius=max_radius,
    )
    if circles is None:
        return []
    found = [
        Instance(_disc(gray.shape, x, y, r), 1.0, "hough")
        for x, y, r in circles[0]
    ]
    return found


def dog_blobs(image, min_sigma=2.0, max_sigma=10.0, threshold=0.05,
              channel="saturation"):
    """Difference-of-Gaussians blob detector on the saturation (or grey)
    channel; each blob becomes a disc of radius sigma * sqrt(2)."""
    from skimage.feature import blob_dog

    rgb = _rgb(image)
    if channel == "saturation":
        plane = _hsv(rgb)[..., 1].astype(np.float32) / 255.0
    else:
        plane = rgb.astype(np.float32).mean(axis=2) / 255.0
    blobs = blob_dog(plane, min_sigma=min_sigma, max_sigma=max_sigma,
                     threshold=threshold)
    return [
        Instance(_disc(plane.shape, x, y, sigma * np.sqrt(2)), 1.0, "dog")
        for y, x, sigma in blobs
    ]


def edge_contours(image, low=40, high=120, close_size=3, min_area=25):
    """Canny edges, closed, then the regions those edges enclose."""
    import cv2

    rgb = _rgb(image)
    gray = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    edges = cv2.Canny(gray, low, high)
    kernel = np.ones((close_size, close_size), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # Regions enclosed by edges = 4-connected components of the non-edge
    # area (8-connectivity would leak through diagonal gaps in the edge).
    return _components(edges == 0, min_area, "edge region", connectivity=4)


def grabcut_refine(image, instances, margin=0.4, iterations=2):
    """Refine each instance with GrabCut inside its box (grown by ``margin``):
    the instance is "probably foreground", the rest of the box "probably
    background". A per-object refinement workflow, not a detector."""
    import cv2

    bgr = cv2.cvtColor(_rgb(image), cv2.COLOR_RGB2BGR)
    height, width = bgr.shape[:2]
    refined = []
    for inst in instances:
        box = inst.box
        if box is None:
            continue
        x0, y0, x1, y1 = box
        pad = int(margin * max(x1 - x0, y1 - y0)) + 2
        cx0, cy0 = max(x0 - pad, 0), max(y0 - pad, 0)
        cx1, cy1 = min(x1 + pad, width), min(y1 + pad, height)
        crop = np.ascontiguousarray(bgr[cy0:cy1, cx0:cx1])
        seed = inst.mask[cy0:cy1, cx0:cx1]
        if seed.sum() < 8 or (~seed).sum() < 8:
            refined.append(inst)
            continue
        mask = np.where(seed, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        core = cv2.erode(seed.astype(np.uint8), np.ones((3, 3))).astype(bool)
        mask[core] = cv2.GC_FGD
        try:
            cv2.grabCut(crop, mask, None, np.zeros((1, 65)),
                        np.zeros((1, 65)), iterations, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            refined.append(inst)
            continue
        full = np.zeros((height, width), dtype=bool)
        full[cy0:cy1, cx0:cx1] = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
        refined.append(
            Instance(full if full.any() else inst.mask, inst.score,
                     inst.label)
        )
    return refined


# --- optical flow (cheap motion propagation between detections) ------------


def dense_flow(prev_gray, next_gray, pyr_scale=0.5, levels=3, winsize=15,
               iterations=3, poly_n=5, poly_sigma=1.2):
    """Farneback dense optical flow from ``prev_gray`` to ``next_gray`` (both
    2-D grey-level arrays, same shape). Returns a ``(H, W, 2)`` float32 field:
    ``flow[y, x]`` is the ``(dx, dy)`` a pixel at ``(x, y)`` in ``prev_gray``
    moved by. Cheap (tens of ms at video-frame size): the point is to move a
    mask forward between two detections, not to detect anything new."""
    import cv2

    return cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None, pyr_scale, levels, winsize, iterations,
        poly_n, poly_sigma, 0,
    )


def warp_mask(mask, flow):
    """Move a boolean mask forward by a dense flow field (nearest-neighbour
    sampling: a mask stays binary, never fades). An approximation of where an
    object went, not a new detection, so it drifts and should be re-anchored
    to a real mask every so often."""
    import cv2

    # cv2.remap is a backward warp (dst(x,y) = src(map_x, map_y)), but ``flow``
    # is forward (prev(x,y) moves to next(x + dx, y + dy)); approximating the
    # inverse with -flow at the output location is the standard trick.
    height, width = mask.shape
    grid_x, grid_y = np.meshgrid(np.arange(width), np.arange(height))
    map_x = (grid_x - flow[..., 0]).astype(np.float32)
    map_y = (grid_y - flow[..., 1]).astype(np.float32)
    warped = cv2.remap(mask.astype(np.uint8), map_x, map_y,
                       cv2.INTER_NEAREST, borderValue=0)
    return warped.astype(bool)


# --- filtering and comparison ------------------------------------------------


def _solidity(mask):
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    points = np.concatenate(contours)
    hull = cv2.contourArea(cv2.convexHull(points))
    area = float(mask.sum())
    return min(area / hull, 1.0) if hull > 0 else 0.0


def flowerlike(instances, image, max_green=0.4, min_area_frac=0.0004,
               max_area_frac=0.05, min_solidity=0.7, min_sat=40):
    """The shared, deliberately simple "could this be a flower head" filter.

    Keeps segments that are not mostly green (``max_green``: share of green
    pixels), have a plausible size (fraction of the image), are compact
    (``min_solidity``: area / convex-hull area) and are not grey (mean
    saturation >= ``min_sat``, so white flowers are lost: a known trade-off).
    """
    hsv = _hsv(_rgb(image))
    hue, sat = hsv[..., 0], hsv[..., 1]
    green = (hue >= GREEN_HUES[0]) & (hue <= GREEN_HUES[1]) & (sat > 40)
    total = hsv.shape[0] * hsv.shape[1]
    kept = []
    for inst in instances:
        area = inst.area
        if not min_area_frac * total <= area <= max_area_frac * total:
            continue
        if green[inst.mask].mean() > max_green:
            continue
        if sat[inst.mask].mean() < min_sat:
            continue
        if _solidity(inst.mask) < min_solidity:
            continue
        kept.append(inst)
    return kept


def agreement(reference, candidates, iou=0.5):
    """How well ``candidates`` agree with ``reference`` masks.

    ``recall``: share of reference masks with a candidate at mask IoU >=
    ``iou`` (greedy one-to-one). ``precision``: share of candidates that
    have one. ``pixel_f1``: F1 of the union of all candidate pixels against
    the union of all reference pixels (region-level, ignores splitting into
    instances).
    Empty inputs give 0.0 rather than an error.
    """
    ref_boxes = [i.box for i in reference]
    pairs = []
    for c, cand in enumerate(candidates):
        cb = cand.box
        if cb is None:
            continue
        for r, ref in enumerate(reference):
            rb = ref_boxes[r]
            if (rb is None or cb[2] <= rb[0] or rb[2] <= cb[0]
                    or cb[3] <= rb[1] or rb[3] <= cb[1]):
                continue
            inter = np.logical_and(cand.mask, ref.mask).sum()
            union = cand.area + ref.area - inter
            if union and inter / union >= iou:
                pairs.append((inter / union, r, c))
    used_r, used_c = set(), set()
    for _, r, c in sorted(pairs, reverse=True):
        if r not in used_r and c not in used_c:
            used_r.add(r)
            used_c.add(c)
    ref_union = _union(reference)
    cand_union = _union(candidates)
    if ref_union is None or cand_union is None:
        pixel_f1 = 0.0
    else:
        inter = np.logical_and(ref_union, cand_union).sum()
        total = ref_union.sum() + cand_union.sum()
        pixel_f1 = float(2 * inter / total) if total else 0.0
    return {
        "recall": len(used_r) / len(reference) if reference else 0.0,
        "precision": len(used_c) / len(candidates) if candidates else 0.0,
        "pixel_f1": pixel_f1,
        "matched": len(used_r),
    }


def _union(instances):
    if not instances:
        return None
    union = np.zeros_like(instances[0].mask, dtype=bool)
    for inst in instances:
        union |= inst.mask
    return union
