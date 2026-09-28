"""Tests for video download and frame sampling (offline, tiny videos)."""

import json
import sys
import types

import pytest
from PIL import Image

from bouquet_checker import video


def make_video(path, seconds=4, fps=10, size=(64, 48)):
    """A video whose frame at second s is a flat colour keyed on s."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    for second in range(seconds):
        bgr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        bgr[:, :, 2] = 60 * (second + 1)          # red channel (BGR order)
        for _ in range(fps):
            writer.write(bgr)
    writer.release()
    return path


# --- ids & provenance ---------------------------------------------------


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=QSVyyrKea14",
    "https://www.youtube.com/watch?v=QSVyyrKea14&t=30s",
    "https://youtu.be/QSVyyrKea14",
    "https://www.youtube.com/shorts/QSVyyrKea14",
])
def test_video_id_from_the_common_url_shapes(url):
    assert video.video_id(url) == "QSVyyrKea14"


def test_video_id_rejects_other_urls():
    with pytest.raises(ValueError, match="YouTube"):
        video.video_id("https://example.com/video.mp4")


def test_format_selector_caps_height_and_avoids_audio_merging():
    selector = video._format_selector(720)
    assert "height<=720" in selector
    assert "bestaudio" not in selector and "+" not in selector


# --- download (yt-dlp is faked) ----------------------------------------------


class FakeYDL:
    calls = []

    def __init__(self, options):
        self.options = options

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download):
        FakeYDL.calls.append((url, download, self.options))
        info = {
            "id": "QSVyyrKea14", "ext": "mp4", "title": "A title \U0001f339",
            "uploader": "Someone", "upload_date": "20250214",
            "duration": 399, "license": None,
        }
        out = self.prepare_filename(info)
        with open(out, "wb") as handle:
            handle.write(b"fake video bytes")
        return info

    def prepare_filename(self, info):
        template = self.options["outtmpl"]
        return template.replace("%(id)s", info["id"]).replace(
            "%(ext)s", info["ext"]
        )


@pytest.fixture
def fake_yt_dlp(monkeypatch):
    FakeYDL.calls = []
    monkeypatch.setitem(
        sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL)
    )
    return FakeYDL


URL = "https://www.youtube.com/watch?v=QSVyyrKea14"


def test_download_writes_the_file_and_a_provenance_sidecar(
    tmp_path, fake_yt_dlp
):
    path = video.download_video(URL, out_dir=tmp_path, max_height=480)
    assert path == tmp_path / "QSVyyrKea14.mp4" and path.exists()
    (url, download, options), = fake_yt_dlp.calls
    assert url == URL and download is True
    assert "height<=480" in options["format"]

    sidecar = video.read_provenance(path)
    assert sidecar["source_url"] == URL
    assert sidecar["title"].startswith("A title")
    assert sidecar["license"] == "Standard YouTube License"
    assert "Do not commit" in sidecar["note"]
    assert json.loads(path.with_suffix(".json").read_text("utf-8")) == sidecar


def test_download_is_idempotent_and_does_not_touch_the_network(
    tmp_path, fake_yt_dlp
):
    first = video.download_video(URL, out_dir=tmp_path)
    second = video.download_video(URL, out_dir=tmp_path)
    assert first == second
    assert len(fake_yt_dlp.calls) == 1


def test_force_downloads_again(tmp_path, fake_yt_dlp):
    video.download_video(URL, out_dir=tmp_path)
    video.download_video(URL, out_dir=tmp_path, force=True)
    assert len(fake_yt_dlp.calls) == 2
    assert fake_yt_dlp.calls[1][2]["overwrites"] is True


def test_provenance_is_none_without_a_sidecar(tmp_path):
    assert video.read_provenance(tmp_path / "missing.mp4") is None


# --- reading frames ---------------------------------------------------------


def test_video_info_reports_size_fps_and_length(tmp_path):
    path = make_video(tmp_path / "v.mp4", seconds=4, fps=10, size=(64, 48))
    info = video.video_info(path)
    assert (info.width, info.height) == (64, 48)
    assert info.fps == pytest.approx(10, abs=0.5)
    assert info.frame_count == 40
    assert info.duration_s == pytest.approx(4, abs=0.2)


def test_sample_frames_every_second_in_rgb_order(tmp_path):
    path = make_video(tmp_path / "v.mp4", seconds=4, fps=10)
    frames = list(video.sample_frames(path, every_s=1.0))
    assert [f.index for f in frames] == [0, 10, 20, 30]
    assert [round(f.time_s) for f in frames] == [0, 1, 2, 3]
    assert all(f.image.mode == "RGB" for f in frames)
    reds = [f.image.getpixel((5, 5))[0] for f in frames]
    assert reds == pytest.approx([60, 120, 180, 240], abs=12)   # not blue
    assert all(f.image.getpixel((5, 5))[2] < 15 for f in frames)


def test_sample_frames_respects_the_span_and_the_cap(tmp_path):
    path = make_video(tmp_path / "v.mp4", seconds=4, fps=10)
    span = list(video.sample_frames(path, every_s=1.0, start_s=1, end_s=3))
    assert [f.index for f in span] == [10, 20]
    capped = list(video.sample_frames(path, every_s=0.5, max_frames=3))
    assert [f.index for f in capped] == [0, 5, 10]


def test_sample_frames_validates_its_input(tmp_path):
    path = make_video(tmp_path / "v.mp4", seconds=1)
    with pytest.raises(ValueError):
        list(video.sample_frames(path, every_s=0))
    with pytest.raises(OSError, match="Cannot open"):
        list(video.sample_frames(tmp_path / "missing.mp4"))


def test_write_video_round_trips_frame_count_and_size(tmp_path):
    pytest.importorskip("cv2")
    images = [Image.new("RGB", (64, 48), (i * 20, 0, 0)) for i in range(12)]
    out = video.write_video(images, tmp_path / "out.mp4", fps=6)
    info = video.video_info(out)
    assert (info.width, info.height) == (64, 48)
    assert info.frame_count == 12


def test_read_frame_seeks_to_the_requested_time(tmp_path):
    path = make_video(tmp_path / "v.mp4", seconds=4, fps=10)
    for second, red in [(0, 60), (2, 180), (3, 240)]:
        frame = video.read_frame(path, second + 0.2)
        assert frame.index == second * 10 + 2
        assert frame.image.getpixel((5, 5))[0] == pytest.approx(red, abs=12)


def test_read_frame_clamps_to_the_video(tmp_path):
    path = make_video(tmp_path / "v.mp4", seconds=2, fps=10)
    assert video.read_frame(path, -5).index == 0
    assert video.read_frame(path, 999).index == 19


def test_frame_reader_matches_sequential_and_seeked_reads(tmp_path):
    path = make_video(tmp_path / "v.mp4", seconds=4, fps=10)
    expected = {f.index: f.image.getpixel((5, 5))[0]
                for f in video.sample_frames(path, every_s=0.1)}
    with video.FrameReader(path) as reader:
        order = [0, 1, 2, 3, 20, 21, 5, 39, 38, 0]     # steps and jumps
        for index in order:
            frame = reader.read(index)
            assert frame.index == index
            assert frame.time_s == pytest.approx(index / 10)
            assert frame.image.getpixel((5, 5))[0] == pytest.approx(
                expected[index], abs=1)


def test_frame_reader_clamps_and_reports_the_video(tmp_path):
    path = make_video(tmp_path / "v.mp4", seconds=2, fps=10)
    with video.FrameReader(path) as reader:
        assert reader.info.frame_count == 20
        assert reader.clamp(-7) == 0 and reader.clamp(10**9) == 19
        assert reader.read(-3).index == 0
        assert reader.read(10**6).index == 19


def test_frame_reader_rejects_a_missing_file(tmp_path):
    pytest.importorskip("cv2")
    with pytest.raises(OSError, match="Cannot open"):
        video.FrameReader(tmp_path / "missing.mp4")


def make_ramp_video(path, frames=40, fps=10, size=(64, 48)):
    """Every frame is unique: its red level is 5 x its index."""
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    for i in range(frames):
        bgr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        bgr[:, :, 2] = 5 * i
        writer.write(bgr)
    writer.release()
    return path


def test_frame_reader_lands_on_exactly_the_requested_frame(tmp_path):
    # The codec is lossy, so compare with what a plain sequential decode of
    # the same file produces (the ground truth), not with the encoder input.
    path = make_ramp_video(tmp_path / "ramp.mp4")
    truth = {f.index: f.image.getpixel((5, 5))[0]
             for f in video.sample_frames(path, every_s=0.1)}
    assert len(set(truth.values())) == len(truth)        # every frame differs
    with video.FrameReader(path) as reader:
        for index in [0, 1, 2, 30, 31, 7, 39, 38, 12, 13, 14, 0, 25]:
            red = reader.read(index).image.getpixel((5, 5))[0]
            assert red == truth[index], index
