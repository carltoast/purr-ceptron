"""Get a video onto disk and read frames from it.

Two jobs, both lazy about their heavy dependencies (``yt-dlp`` and OpenCV
are notebook-only, see ``requirements-notebook.txt``):

- ``download_video`` fetches a YouTube video ONCE for private research, keeps
  it under the git-ignored ``data/raw/videos/`` and writes a sidecar JSON
  with where it came from and under which licence. Re-running reuses the
  file, so it never re-downloads. Only the video stream is fetched (no
  audio), so no ffmpeg merge step is needed.
- ``sample_frames`` reads frames at a chosen interval as PIL RGB images.

A downloaded video stays the property of its creator: it is kept for local
analysis, never committed, never redistributed.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from bouquet_checker.config import VIDEO_DIR

_YOUTUBE_ID = re.compile(
    r"(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})"
)


@dataclass(frozen=True)
class VideoInfo:
    """Basic properties of a video file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int

    @property
    def duration_s(self):
        """Length in seconds."""
        return self.frame_count / self.fps if self.fps else 0.0


@dataclass(frozen=True)
class Frame:
    """One sampled frame: its index, time in seconds and RGB image."""

    index: int
    time_s: float
    image: Image.Image


def video_id(url):
    """The 11-character YouTube id in ``url`` (raises if there is none)."""
    match = _YOUTUBE_ID.search(url)
    if match is None:
        raise ValueError(f"Not a recognised YouTube URL: {url!r}")
    return match.group(1)


def _format_selector(max_height):
    # Video-only, H.264 in an mp4 container: OpenCV decodes it everywhere
    # and there is nothing to merge, so ffmpeg is not needed.
    h = f"[height<={max_height}]"
    return (
        f"bestvideo{h}[ext=mp4][vcodec^=avc1]"
        f"/bestvideo{h}[ext=mp4]/best{h}[ext=mp4]"
    )


def download_video(url, out_dir=VIDEO_DIR, max_height=720, force=False):
    """Download ``url`` once and return the local ``Path``.

    Writes ``<id>.json`` next to the video (source URL, title, uploader,
    licence, when it was fetched). If the file already exists it is reused
    and no network access happens unless ``force`` is set.
    """
    out_dir = Path(out_dir)
    vid = video_id(url)
    existing = sorted(out_dir.glob(f"{vid}.*"))
    existing = [p for p in existing if p.suffix != ".json"]
    if existing and not force:
        return existing[0]

    import yt_dlp

    out_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": _format_selector(max_height),
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "overwrites": bool(force),
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info))

    sidecar = {
        "source_url": url,
        "id": vid,
        "title": info.get("title"),
        "uploader": info.get("uploader"),
        "upload_date": info.get("upload_date"),
        "duration_s": info.get("duration"),
        # yt-dlp only reports a licence for Creative Commons uploads.
        "license": info.get("license") or "Standard YouTube License",
        "downloaded_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "note": "Private research copy. Do not commit or redistribute.",
    }
    path.with_suffix(".json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def read_provenance(path):
    """The sidecar JSON of a downloaded video, or ``None`` if absent."""
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.exists():
        return None
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _open(path):
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"Cannot open video: {path}")
    return capture


def video_info(path):
    """Read width, height, fps and frame count from the file header."""
    import cv2

    capture = _open(path)
    try:
        return VideoInfo(
            path=Path(path),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        capture.release()


def read_frame(path, time_s):
    """The frame nearest ``time_s`` (seek-based, so it is quick anywhere)."""
    import cv2

    info = video_info(path)
    last = max(info.frame_count - 1, 0)
    index = min(max(int(round(time_s * info.fps)), 0), last)
    capture = _open(path)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = capture.read()
        if not ok:
            raise OSError(f"Cannot read the frame at {time_s}s of {path}")
    finally:
        capture.release()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Frame(index, index / info.fps, Image.fromarray(rgb))


class FrameReader:
    """Random access to frames by index, keeping the file open.

    Stepping forward one frame reads sequentially (no seek), so it is fast;
    any other jump seeks. Indices are clamped to the video, so ``read(-5)``
    is frame 0 and ``read(10**9)`` is the last frame.
    """

    def __init__(self, path):
        self.info = video_info(path)
        self._capture = _open(path)
        self._next = 0                 # index the capture will return next

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        """Release the underlying file."""
        self._capture.release()

    def clamp(self, index):
        """``index`` limited to the valid range of this video."""
        return min(max(int(index), 0), max(self.info.frame_count - 1, 0))

    def read(self, index):
        """The ``Frame`` at ``index`` (clamped)."""
        import cv2

        index = self.clamp(index)
        if index != self._next:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, bgr = self._capture.read()
        if not ok:
            self._next = -1            # position unknown: force a seek
            raise OSError(f"Cannot read frame {index} of {self.info.path}")
        self._next = index + 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Frame(index, index / self.info.fps, Image.fromarray(rgb))


def sample_frames(path, every_s=1.0, start_s=0.0, end_s=None,
                  max_frames=None):
    """Yield a ``Frame`` roughly every ``every_s`` seconds.

    Frames are read sequentially (seeking in compressed video is slow and
    inexact), so the cost is one decode pass. ``start_s`` / ``end_s`` bound
    the span; ``max_frames`` caps how many are yielded.
    """
    import cv2

    if every_s <= 0:
        raise ValueError("every_s must be positive")
    info = video_info(path)
    if info.fps <= 0:
        raise OSError(f"Video reports no frame rate: {path}")
    step = max(1, round(every_s * info.fps))
    first = int(start_s * info.fps)
    last = info.frame_count if end_s is None else int(end_s * info.fps)

    capture = _open(path)
    try:
        yielded = 0
        for index in range(last):
            if not capture.grab():
                break
            if index < first or (index - first) % step:
                continue
            ok, bgr = capture.retrieve()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            yield Frame(index, index / info.fps, Image.fromarray(rgb))
            yielded += 1
            if max_frames is not None and yielded >= max_frames:
                break
    finally:
        capture.release()


def write_video(images, path, fps):
    """Write PIL images (all the same size) to an mp4 at ``fps``."""
    import cv2
    import numpy as np

    writer = None
    try:
        for image in images:
            rgb = np.asarray(image.convert("RGB"))
            if writer is None:
                height, width = rgb.shape[:2]
                writer = cv2.VideoWriter(
                    str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                    (width, height),
                )
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        if writer is not None:
            writer.release()
    return Path(path)
