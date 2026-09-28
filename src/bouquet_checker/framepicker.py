"""Hand-pick sample frames from a video by jumping around in it.

``FramePicker`` holds the state (where you are, which frames you picked) and
knows nothing about widgets, so it is plain and testable. ``FramePicker.ui``
wraps it in an ``ipywidgets`` interface for notebooks: buttons that jump
-1000 / -100 / -10 / -1 / +1 / +10 / +100 / +1000 frames, a pick button, the
list of picks, and saving them to a JSON file (git-ignored) that later
notebooks can load. ``ipywidgets`` is notebook-only and imported lazily.
"""

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from PIL import Image, ImageDraw

from bouquet_checker.config import PICKS_DIR
from bouquet_checker.video import FrameReader

JUMPS = (1000, 100, 10, 1)


class FramePicker:
    """A cursor over the frames of one video, plus a set of picked frames."""

    def __init__(self, video_path, start=0, picks_path=None):
        self.path = Path(video_path)
        self.reader = FrameReader(self.path)
        self.info = self.reader.info
        self.index = self.reader.clamp(start)
        self.picks = []
        self.picks_path = (
            Path(picks_path) if picks_path
            else PICKS_DIR / f"{self.path.stem}_frames.json"
        )
        self.widgets = None               # filled in by ``ui``

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def close(self):
        """Release the video file."""
        self.reader.close()

    # --- moving around -------------------------------------------------------

    @property
    def last_index(self):
        """Index of the last frame."""
        return max(self.info.frame_count - 1, 0)

    @property
    def time_s(self):
        """Time of the current frame in seconds."""
        return self.index / self.info.fps

    def goto(self, index):
        """Move to ``index`` (clamped to the video); returns the new index."""
        self.index = self.reader.clamp(index)
        return self.index

    def jump(self, delta):
        """Move by ``delta`` frames (negative = backwards), clamped."""
        return self.goto(self.index + delta)

    def goto_time(self, seconds):
        """Move to the frame nearest ``seconds``."""
        return self.goto(round(seconds * self.info.fps))

    def frame(self):
        """The current ``Frame``."""
        return self.reader.read(self.index)

    # --- picking -------------------------------------------------------------

    def is_picked(self, index=None):
        """Whether ``index`` (default: the current frame) is picked."""
        return (self.index if index is None else index) in self.picks

    def pick(self):
        """Pick the current frame; returns False if it already was."""
        if self.index in self.picks:
            return False
        self.picks = sorted([*self.picks, self.index])
        return True

    def unpick(self, index=None):
        """Remove a pick (default: the current frame); False if absent."""
        index = self.index if index is None else index
        if index not in self.picks:
            return False
        self.picks.remove(index)
        return True

    def toggle(self):
        """Pick the current frame, or unpick it if it is picked."""
        if not self.pick():
            self.unpick()
        return self.is_picked()

    def clear(self):
        """Remove every pick."""
        self.picks = []

    def frames(self):
        """The picked ``Frame``s, in order (does not move the cursor)."""
        return [self.reader.read(i) for i in self.picks]

    def contact_sheet(self, cols=4, thumb=(320, 180)):
        """One image with a labelled thumbnail per pick (None if none)."""
        if not self.picks:
            return None
        rows = -(-len(self.picks) // cols)
        sheet = Image.new("RGB", (cols * thumb[0], rows * thumb[1]), "white")
        for n, frame in enumerate(self.frames()):
            tile = frame.image.resize(thumb)
            draw = ImageDraw.Draw(tile)
            draw.rectangle((0, 0, 190, 16), fill="black")
            draw.text(
                (4, 2), f"frame {frame.index}  {frame.time_s:.2f} s",
                fill="yellow",
            )
            sheet.paste(tile, ((n % cols) * thumb[0], (n // cols) * thumb[1]))
        return sheet

    # --- saving --------------------------------------------------------------

    def save(self, path=None):
        """Write the picks to JSON; returns the path."""
        path = Path(path) if path else self.picks_path
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "video": self.path.name,
            "fps": self.info.fps,
            "frame_count": self.info.frame_count,
            "saved_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "picks": [
                {"index": i, "time_s": round(i / self.info.fps, 3)}
                for i in self.picks
            ],
        }
        path.write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        return path

    def load(self, path=None, merge=False):
        """Read picks from a saved JSON file (replacing, or merging with, the
        current ones). Refuses a file written for a different video."""
        path = Path(path) if path else self.picks_path
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("video") != self.path.name:
            raise ValueError(
                f"{path} was saved for {document.get('video')!r}, "
                f"not {self.path.name!r}"
            )
        loaded = [int(p["index"]) for p in document["picks"]]
        bad = [i for i in loaded if not 0 <= i <= self.last_index]
        if bad:
            raise ValueError(f"Picks outside this video: {bad}")
        self.picks = sorted(set(loaded) | set(self.picks if merge else ()))
        return self.picks

    # --- the notebook interface -------------------------------------------

    def ui(self, width=720):
        """Build (and return) the interactive widget. Display it with
        ``display(picker.ui())`` or by ending a cell with ``picker.ui()``."""
        import ipywidgets as w

        fps = self.info.fps
        image = w.Image(format="jpeg", width=width)
        status = w.HTML()
        message = w.HTML()
        pick_button = w.Button(description="Pick this frame",
                               button_style="success",
                               layout=w.Layout(width="190px"))
        picks_list = w.Select(options=[], rows=8,
                              layout=w.Layout(width="260px"))
        goto_index = w.BoundedIntText(
            value=self.index, min=0, max=self.last_index,
            description="Frame", layout=w.Layout(width="170px"))
        goto_time = w.BoundedFloatText(
            value=self.time_s, min=0, max=self.last_index / fps, step=0.1,
            description="Time (s)", layout=w.Layout(width="190px"))

        def refresh():
            frame = self.frame()
            ratio = self.info.height / self.info.width
            small = frame.image.resize((width, round(width * ratio)))
            buffer = io.BytesIO()
            small.save(buffer, "JPEG", quality=85)
            image.value = buffer.getvalue()
            picked = self.is_picked()
            status.value = (
                f"<b>frame {self.index:,}</b> of {self.last_index:,}"
                f" &nbsp;&middot;&nbsp; {self.time_s:.3f} s"
                + (" &nbsp;&#9733; <b>picked</b>" if picked else "")
            )
            pick_button.description = (
                "Unpick this frame" if picked else "Pick this frame")
            pick_button.button_style = "warning" if picked else "success"
            goto_index.value = self.index
            goto_time.value = round(self.time_s, 3)
            picks_list.options = [
                (f"frame {i:,}  ·  {i / fps:.2f} s", i) for i in self.picks]
            picks_list.value = self.index if picked else None

        def move(delta):
            self.jump(delta)
            refresh()

        jump_buttons = {}
        for delta in [-d for d in JUMPS] + [d for d in reversed(JUMPS)]:
            button = w.Button(
                description=f"{delta:+d}", layout=w.Layout(width="72px"),
                tooltip=f"{delta:+d} frames ({delta / fps:+.2f} s)")
            button.on_click(lambda _, d=delta: move(d))
            jump_buttons[button.description] = button

        def on_go(_):
            self.goto(goto_index.value)
            refresh()

        def on_go_time(_):
            self.goto_time(goto_time.value)
            refresh()

        def on_pick(_):
            self.toggle()
            message.value = ""
            refresh()

        def on_go_selected(_):
            if picks_list.value is not None:
                self.goto(picks_list.value)
                refresh()

        def on_remove_selected(_):
            if picks_list.value is not None:
                self.unpick(picks_list.value)
                refresh()

        def on_clear(_):
            self.clear()
            refresh()

        def on_save(_):
            where = self.save()
            message.value = (
                f"saved {len(self.picks)} pick(s) to <code>{where}</code>")

        go_button = w.Button(description="Go to frame")
        go_button.on_click(on_go)
        go_time_button = w.Button(description="Go to time")
        go_time_button.on_click(on_go_time)
        pick_button.on_click(on_pick)
        goto_selected = w.Button(description="Show selected")
        goto_selected.on_click(on_go_selected)
        remove_selected = w.Button(description="Remove selected")
        remove_selected.on_click(on_remove_selected)
        clear_button = w.Button(description="Clear all picks")
        clear_button.on_click(on_clear)
        save_button = w.Button(description="Save picks",
                               button_style="info")
        save_button.on_click(on_save)

        self.widgets = SimpleNamespace(
            jump=jump_buttons, pick=pick_button, image=image,
            status=status, message=message, picks=picks_list,
            goto_index=goto_index, goto_time=goto_time, go=go_button,
            go_time=go_time_button, save=save_button, clear=clear_button,
            show_selected=goto_selected, remove_selected=remove_selected,
            refresh=refresh,
        )
        refresh()
        return w.VBox([
            w.HBox(list(jump_buttons.values())),
            image,
            status,
            w.HBox([goto_index, go_button, goto_time, go_time_button]),
            w.HBox([pick_button, save_button, clear_button]),
            w.HBox([picks_list, w.VBox([goto_selected, remove_selected])]),
            message,
        ])
