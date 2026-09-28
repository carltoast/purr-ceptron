"""Tests for the frame picker (logic, saving and the widget buttons)."""

import json

import pytest

from bouquet_checker.framepicker import FramePicker

FRAMES, FPS = 2500, 30


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    path = tmp_path_factory.mktemp("video") / "clip.mp4"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48)
    )
    for i in range(FRAMES):
        bgr = np.zeros((48, 64, 3), dtype=np.uint8)
        bgr[:, :, 2] = i % 250
        writer.write(bgr)
    writer.release()
    return path


@pytest.fixture
def picker(clip, tmp_path):
    with FramePicker(clip, start=1500, picks_path=tmp_path / "p.json") as p:
        yield p


# --- moving around ---------------------------------------------------------


def test_starts_where_asked_and_reports_time(picker):
    assert picker.index == 1500
    assert picker.time_s == pytest.approx(50.0)
    assert picker.last_index == FRAMES - 1


@pytest.mark.parametrize("delta, expected", [
    (1, 1501), (10, 1510), (100, 1600), (1000, 2499),      # clamped at the end
    (-1, 1499), (-10, 1490), (-100, 1400), (-1000, 500),
])
def test_jump_by_1_10_100_1000_frames_either_way(picker, delta, expected):
    assert picker.jump(delta) == expected
    assert picker.index == expected


def test_jumps_never_leave_the_video(picker):
    assert picker.jump(-10**6) == 0
    assert picker.jump(10**6) == FRAMES - 1
    assert picker.goto(-5) == 0 and picker.goto(10**9) == FRAMES - 1


def test_goto_time_and_frame_content(picker):
    assert picker.goto_time(10.0) == 300
    frame = picker.frame()
    assert frame.index == 300 and frame.time_s == pytest.approx(10.0)
    assert frame.image.size == (64, 48)


# --- picking ------------------------------------------------------------


def test_picks_are_sorted_and_deduplicated(picker):
    for index in (900, 100, 900, 500):
        picker.goto(index)
        picker.pick()
    assert picker.picks == [100, 500, 900]
    picker.goto(500)
    assert picker.pick() is False                     # already picked


def test_unpick_toggle_and_clear(picker):
    picker.goto(10)
    assert picker.toggle() is True and picker.picks == [10]
    assert picker.toggle() is False and picker.picks == []
    picker.pick()
    picker.goto(20)
    picker.pick()
    assert picker.unpick(10) is True and picker.unpick(10) is False
    assert picker.picks == [20]
    picker.clear()
    assert picker.picks == [] and picker.is_picked() is False


def test_frames_returns_the_picks_without_moving_the_cursor(picker):
    for index in (200, 100):
        picker.goto(index)
        picker.pick()
    picker.goto(777)
    assert [f.index for f in picker.frames()] == [100, 200]
    assert picker.index == 777


def test_contact_sheet_is_none_when_nothing_is_picked_else_sized(picker):
    assert picker.contact_sheet() is None
    for index in range(0, 500, 100):                    # 5 picks
        picker.goto(index)
        picker.pick()
    sheet = picker.contact_sheet(cols=4, thumb=(320, 180))
    assert sheet.size == (4 * 320, 2 * 180)


# --- saving and loading -----------------------------------------------------


def test_save_and_load_round_trip(picker, tmp_path):
    for index in (30, 3, 900):
        picker.goto(index)
        picker.pick()
    path = picker.save()
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["video"] == "clip.mp4"
    assert document["fps"] == pytest.approx(FPS)
    assert [p["index"] for p in document["picks"]] == [3, 30, 900]
    assert document["picks"][1]["time_s"] == pytest.approx(1.0)

    picker.clear()
    assert picker.load() == [3, 30, 900]


def test_load_can_merge_or_replace(picker):
    picker.goto(5)
    picker.pick()
    picker.save()
    picker.clear()
    picker.goto(9)
    picker.pick()
    assert picker.load(merge=True) == [5, 9]
    assert picker.load(merge=False) == [5]


def test_load_refuses_another_video_or_out_of_range_picks(picker, tmp_path):
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"video": "not-this.mp4", "picks": []}))
    with pytest.raises(ValueError, match="not-this.mp4"):
        picker.load(other)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(
        {"video": "clip.mp4", "picks": [{"index": FRAMES + 5}]}))
    with pytest.raises(ValueError, match="outside"):
        picker.load(bad)


# --- the buttons (simulated clicks) ------------------------------------------


@pytest.fixture
def ui(picker):
    pytest.importorskip("ipywidgets")
    picker.ui(width=160)
    return picker.widgets


def test_all_eight_jump_buttons_exist_and_move_the_cursor(picker, ui):
    assert list(ui.jump) == ["-1000", "-100", "-10", "-1",
                             "+1", "+10", "+100", "+1000"]
    expected = {"+1": 1501, "+10": 1511, "+100": 1611, "+1000": 2499,
                "-1000": 1499, "-100": 1399, "-10": 1389, "-1": 1388}
    for label, index in expected.items():                # cumulative clicks
        ui.jump[label].click()
        assert picker.index == index, label


def test_the_frame_image_and_status_follow_the_cursor(picker, ui):
    assert ui.image.value[:2] == b"\xff\xd8"              # a JPEG
    before = ui.image.value
    ui.jump["+100"].click()
    assert ui.image.value != before
    assert "1,600" in ui.status.value and "picked" not in ui.status.value


def test_pick_button_toggles_and_lists_the_pick(picker, ui):
    ui.pick.click()
    assert picker.picks == [1500]
    assert ui.pick.description == "Unpick this frame"
    assert "picked" in ui.status.value
    assert [v for _, v in ui.picks.options] == [1500]
    ui.pick.click()
    assert picker.picks == [] and ui.pick.description == "Pick this frame"
    assert ui.picks.options == ()


def test_go_to_frame_and_time_controls(picker, ui):
    ui.goto_index.value = 1234
    ui.go.click()
    assert picker.index == 1234
    ui.goto_time.value = 10.0
    ui.go_time.click()
    assert picker.index == 300


def test_show_remove_clear_and_save_buttons(picker, ui):
    for label in ("+1", "+100", "+100"):
        ui.jump[label].click()
        ui.pick.click()
    assert picker.picks == [1501, 1601, 1701]
    ui.jump["-1000"].click()
    ui.picks.value = 1601
    ui.show_selected.click()
    assert picker.index == 1601
    ui.picks.value = 1701
    ui.remove_selected.click()
    assert picker.picks == [1501, 1601]
    ui.save.click()
    assert "saved 2 pick(s)" in ui.message.value
    assert picker.picks_path.exists()
    ui.clear.click()
    assert picker.picks == []
