"""The panel layout: the chat on the left, the character on the right, input below.

A full-screen prompt_toolkit application wrapped around the same Repl: everything the
Repl prints (replies as they stream, notes, command output) goes into the chat area,
so every command works unchanged. Questions (wizards, pickers, confirmations) are
asked in the input box itself, so the screen never has to switch back and forth.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import get_app_session
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.data_structures import Point
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, StyleAndTextTuples, to_formatted_text
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import (
    BufferControl,
    ConditionalContainer,
    Dimension,
    Float,
    FloatContainer,
    FormattedTextControl,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput, ConditionalProcessor, PasswordProcessor
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from chatclient.core.chat import ChatError
from chatclient.panel_data import panel_data
from chatclient.providers.base import ProviderError
from chatclient.ui import sixel

if TYPE_CHECKING:
    from chatclient.ui.picker import PickItem
    from chatclient.ui.repl import Repl

KEEP_CHARS = 250_000  # the chat area keeps this much text (older scrolls away)
RENDER_LINES = 600  # lines drawn at once: plenty for a screen, cheap while streaming
MIN_TOTAL_WIDTH = 80  # narrower terminals: the chat alone, no side panel
MIN_SIDE_WIDTH = 20  # the side panel is never narrower…
MIN_CHAT_WIDTH = 40  # …and the chat column keeps at least this much
RESIZE_STEP = 2  # columns per Alt+, / Alt+.
SCROLL_STEP = 10
WHEEL_STEP = 3
SIDE_STEP = 5  # Alt+PageUp / Alt+PageDown in the side panel
OLDER_MESSAGES = 10  # loaded at a time when scrolling past the top
# The terminal control the Repl writes: carriage return and clear-line (the "typing…"
# line rewrites itself); pictures (Sixel) and other cursor/screen codes are dropped.
_CONTROL = re.compile(r"\x1bP.*?\x1b\\|\r\n|\r|\x1b\[2K|\x1b\[[0-9;?]*[A-HJKfhlns]", re.DOTALL)


class ChatLog:
    """A file-like sink for everything the Repl prints, kept as ANSI text."""

    encoding = "utf-8"

    def __init__(self, on_change: Any = None):
        self.text = ""
        self.version = 0
        self.on_change = on_change
        self._cr = False

    def isatty(self) -> bool:
        return True  # colours and the typing indicator, as in a terminal

    def flush(self) -> None:
        pass

    def _line_start(self) -> int:
        return self.text.rfind("\n") + 1

    def write(self, data: str) -> int:
        pos = 0
        for m in _CONTROL.finditer(data):
            self._plain(data[pos : m.start()])
            token = m.group()
            if token == "\r\n":  # a line ending, not "erase this line"
                self._plain("\n")
            elif token == "\r":
                self._cr = True
            elif token == "\x1b[2K":
                self.text = self.text[: self._line_start()]
            pos = m.end()
        self._plain(data[pos:])
        if len(self.text) > KEEP_CHARS:
            cut = self.text.find("\n", len(self.text) - KEEP_CHARS)
            self.text = self.text[cut + 1 :]
        self.version += 1
        if self.on_change is not None:
            self.on_change()
        return len(data)

    def _plain(self, s: str) -> None:
        if not s:
            return
        if self._cr:  # after \r, what follows replaces the current line
            if not s.startswith("\n"):  # (a \r and \n split over two writes: a line ending)
                self.text = self.text[: self._line_start()]
            self._cr = False
        self.text += s

    def clear(self) -> None:
        self.text = ""
        self.version += 1
        if self.on_change is not None:
            self.on_change()


def thumbnail(path: Path, width: int, max_rows: int) -> list[StyleAndTextTuples]:
    """A picture as coloured half-blocks: each cell shows two pixels (▀, the upper one in
    front, the lower one behind). A preview that fits the panel, not the picture."""
    from PIL import Image

    with Image.open(path) as src:
        img = src.convert("RGB")
    rows = max(1, round(img.height / img.width * width / 2))
    if rows > max_rows:  # a tall picture: narrower, so the rest of the panel still fits
        rows = max_rows
        width = max(4, round(rows * 2 * img.width / img.height))
    img = img.resize((width, rows * 2), Image.Resampling.LANCZOS)
    px = img.load()
    lines: list[StyleAndTextTuples] = []
    for y in range(rows):
        line: StyleAndTextTuples = []
        for x in range(width):
            top, low = px[x, y * 2], px[x, y * 2 + 1]
            style = f"fg:#{top[0]:02x}{top[1]:02x}{top[2]:02x} bg:#{low[0]:02x}{low[1]:02x}{low[2]:02x}"  # fmt: skip
            line.append((style, "▀"))
        lines.append(line)
    return lines


def fit(size: tuple[int, int], max_w: int, max_h: int) -> tuple[int, int]:
    """A picture's size scaled down (never up) to fit max_w × max_h, as sixel.encode does."""
    w, h = size
    scale = min(1.0, max_w / w, max_h / h)
    return (max(1, int(w * scale)), max(1, int(h * scale))) if scale < 1.0 else (w, h)


Spot = tuple[int, int]  # (line of the chat log, character in that line)


class Selection:
    """Text selected in the chat area with the mouse, in lines of the log (so it stays
    put while the chat scrolls or grows). Both ends are inclusive."""

    def __init__(self) -> None:
        self.anchor: Spot | None = None
        self.end: Spot | None = None
        self.dragging = False
        self.rev = 0  # changes whenever what is highlighted changes
        self.first = 0  # the log line the drawn part starts with (set by _visible)
        self.note = ""  # a short message for the status line ("Copied …")
        self.timer: asyncio.TimerHandle | None = None

    def span(self) -> tuple[Spot, Spot] | None:
        if self.anchor is None or self.end is None:
            return None
        return (min(self.anchor, self.end), max(self.anchor, self.end))

    def clear(self) -> None:
        if self.anchor is not None:
            self.rev += 1
        self.anchor = self.end = None
        self.dragging = False


def highlight(fragments: StyleAndTextTuples, start: Spot, end: Spot) -> StyleAndTextTuples:
    """The fragments with the characters from start to end (row, column; inclusive) in
    reverse video."""
    out: StyleAndTextTuples = []
    row = col = 0
    for style, text, *_ in fragments:
        run, on = "", False
        for ch in text:
            chosen = False if ch == "\n" else start <= (row, col) <= end
            if run and chosen != on:
                out.append((f"{style} reverse" if on else style, run))
                run = ""
            run, on = run + ch, chosen
            if ch == "\n":
                row, col = row + 1, 0
            else:
                col += 1
        if run:
            out.append((f"{style} reverse" if on else style, run))
    return out


class _TerminalWriter:
    """Where the clipboard fallback (OSC 52) writes while stdout is the chat log: the
    terminal itself, through the app's output."""

    def __init__(self, output: Any):
        self.output = output

    def write(self, data: str) -> int:
        self.output.write_raw(data)
        return len(data)

    def flush(self) -> None:
        self.output.flush()


class ScrollableText(FormattedTextControl):
    """The chat area: the mouse wheel scrolls the conversation (not the input history);
    dragging with the left button selects text and copies it on release."""

    def __init__(self, view: PanelView, **kw: Any):
        super().__init__(**kw)
        self.view = view

    def mouse_handler(self, mouse_event: MouseEvent) -> Any:
        kind, view = mouse_event.event_type, self.view
        if kind == MouseEventType.SCROLL_UP:
            view.scroll_by(WHEEL_STEP)
        elif kind == MouseEventType.SCROLL_DOWN:
            view.scroll_by(-WHEEL_STEP)
        elif kind == MouseEventType.MOUSE_DOWN and mouse_event.button == MouseButton.LEFT:
            view.select_start(mouse_event.position)
        elif view.selection.dragging and kind == MouseEventType.MOUSE_MOVE:
            if mouse_event.button == MouseButton.NONE:  # the release was missed
                view.select_end(mouse_event.position)
            else:
                view.select_drag(mouse_event.position)
        elif view.selection.dragging and kind == MouseEventType.MOUSE_UP:
            view.select_end(mouse_event.position)
        else:
            return NotImplemented
        return None


class SideText(FormattedTextControl):
    """The side panel: the mouse wheel over it scrolls it, not the chat."""

    def __init__(self, view: PanelView, **kw: Any):
        super().__init__(**kw)
        self.view = view

    def mouse_handler(self, mouse_event: MouseEvent) -> Any:
        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.view.scroll_side(-WHEEL_STEP)
            return None
        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.view.scroll_side(WHEEL_STEP)
            return None
        return NotImplemented


class DragText(FormattedTextControl):
    """Text whose mouse events go to one handler (the separator, and the drag catcher)."""

    def __init__(self, text: Any, handler: Any):
        super().__init__(text, focusable=False)
        self.handler = handler

    def mouse_handler(self, mouse_event: MouseEvent) -> Any:
        return self.handler(mouse_event)


class PanelView:
    def __init__(self, repl: Repl, completer: Completer):
        self.repl = repl
        self.log = ChatLog(self.invalidate)
        self.side: StyleAndTextTuples = []
        self.scroll = 0  # lines scrolled back from the bottom
        self.question = ""  # a question asked in the input box (wizards, pickers…)
        self._answer: asyncio.Future[str] | None = None
        self._password = False
        self.busy: asyncio.Future[None] | None = None  # the line being handled
        self._thumb: tuple[str, int, int, list[StyleAndTextTuples]] | None = None
        self._view: tuple[Any, StyleAndTextTuples, int] = (None, [], 0)
        self.selection = Selection()  # text selected with the mouse
        self.hidden_before = 0  # messages of this chat above what the chat area holds
        self.shown_session: int | None = None  # the chat the chat area belongs to
        # The picture, drawn by the terminal itself (Sixel) over blank panel lines:
        # (path, width px, height px) wanted, what is on screen now, and the encoding of
        # the wanted one (made in refresh(), so drawing never waits for it).
        self._pic: tuple[str, int, int] | None = None
        self._drawn: tuple[Any, ...] | None = None
        self._encoded: tuple[tuple[str, int, int], str] | None = None
        self._last_size: tuple[int, int] | None = None
        # The side panel scrolls on its own: lines hidden above, the lines the picture
        # takes (it starts on the first), the chat the offset belongs to, its window.
        self.side_scroll = 0
        self._pic_rows = 0
        self._side_session: int | None = None
        self._side_window: Window | None = None
        self._live_width: int | None = None  # while the separator is dragged (saved on release)
        self._dragging = False
        self.input = Buffer(
            multiline=True,
            completer=completer,
            complete_while_typing=True,
            history=FileHistory(str(repl.paths.history_file)),
        )
        self.app: Application[None] = self._build()

    # ── sizes ───────────────────────────────────────────────────────────────

    def size(self) -> tuple[int, int]:
        try:
            s = self.app.output.get_size()
            return s.columns, s.rows
        except Exception:
            return 100, 30

    def side_width(self) -> int:
        columns = self.size()[0]
        if columns < MIN_TOTAL_WIDTH:
            return 0
        want = self._live_width if self._live_width is not None else self.repl.panel_width()
        return self.clamp_width(want, columns)

    def clamp_width(self, width: int, columns: int | None = None) -> int:
        """At least MIN_SIDE_WIDTH, and the chat keeps at least MIN_CHAT_WIDTH."""
        columns = self.size()[0] if columns is None else columns
        return max(MIN_SIDE_WIDTH, min(width, columns - MIN_CHAT_WIDTH - 2))

    def resize(self, width: int) -> None:
        """Set (and save) the side panel's width; everything that depends on it is rebuilt."""
        self._live_width = None
        self.repl.set_panel_width(self.clamp_width(width))
        self.refresh()

    def _separator_event(self, event: MouseEvent) -> Any:
        """Drag the │ between chat and panel: the width follows the pointer live; it is
        saved on release. While dragging, an invisible full-screen window catches the
        pointer (it may leave the 1-column separator)."""
        kind = event.event_type
        if kind == MouseEventType.MOUSE_DOWN and not self._dragging:
            self._dragging = True
            self.invalidate()  # shows the catcher
        elif self._dragging and kind == MouseEventType.MOUSE_MOVE:
            if event.button == MouseButton.NONE:  # the release was missed
                self._drag_end()
            else:
                self._drag_to(event.position.x)
        elif self._dragging and kind == MouseEventType.MOUSE_UP:
            self._drag_to(event.position.x)
            self._drag_end()
        return None

    def _drag_to(self, x: int) -> None:
        """The separator is at column x, the panel takes what is right of it."""
        width = self.clamp_width(self.size()[0] - x - 1)
        if width != self.side_width():
            self._live_width = width
            self.refresh()

    def _drag_end(self) -> None:
        self._dragging = False
        if self._live_width is not None:
            self.resize(self._live_width)
        self.invalidate()

    def chat_width(self) -> int:
        side = self.side_width()
        return max(20, self.size()[0] - (side + 2 if side else 1))

    # ── the chat area ───────────────────────────────────────────────────────

    def _visible(self) -> tuple[StyleAndTextTuples, int]:
        """The lines up to the scroll position (the window shows their end)."""
        sel = self.selection
        key = (self.log.version, self.scroll, sel.rev)
        if self._view[0] != key:
            lines = self.log.text.split("\n")
            self.scroll = min(self.scroll, max(0, len(lines) - 1))
            end = len(lines) - self.scroll
            sel.first = max(0, end - RENDER_LINES)
            part = lines[sel.first : end]
            fragments = to_formatted_text(ANSI("\n".join(part)))
            span = sel.span()
            if span is not None and span[1][0] >= len(lines):
                sel.clear()  # the text it was on is gone (/clear)
                span = None
            if span is not None:
                fragments = highlight(
                    fragments,
                    (span[0][0] - sel.first, span[0][1]),
                    (span[1][0] - sel.first, span[1][1]),
                )
            self._view = (key, fragments, len(part) - 1)
        return self._view[1], self._view[2]

    # ── selecting and copying text ──────────────────────────────────────────

    def _spot(self, pos: Point) -> Spot:
        """The mouse position (a line of what is drawn and a character in it) as a spot
        in the log; wrapped screen rows are already folded into their line."""
        return (self.selection.first + pos.y, pos.x)

    def select_start(self, pos: Point) -> None:
        sel = self.selection
        sel.anchor = sel.end = self._spot(pos)
        sel.dragging = True
        sel.rev += 1
        self.invalidate()

    def select_drag(self, pos: Point) -> None:
        sel = self.selection
        spot = self._spot(pos)
        if spot != sel.end:
            sel.end = spot
            sel.rev += 1
            self.invalidate()

    def select_end(self, pos: Point) -> None:
        """Mouse up: a click only clears; a drag copies what was selected."""
        sel = self.selection
        self.select_drag(pos)
        sel.dragging = False
        if sel.anchor == sel.end:
            sel.clear()
            self.invalidate()
            return
        text = self.selected_text()
        if text.strip():
            self.copy(text)

    def selected_text(self) -> str:
        """The selected characters as plain text, line by line as in the log."""
        span = self.selection.span()
        if span is None:
            return ""
        (r0, c0), (r1, c1) = span
        raw = self.log.text.split("\n")[r0 : r1 + 1]
        plain = "".join(f[1] for f in to_formatted_text(ANSI("\n".join(raw)))).split("\n")
        if not plain:
            return ""
        plain[-1] = plain[-1][: c1 + 1]
        plain[0] = plain[0][c0:]
        return "\n".join(line.rstrip() for line in plain).strip("\n")

    def copy(self, text: str) -> None:
        """To the clipboard (as /copy does) and a short note in the status line."""
        import sys

        from chatclient.ui import clipboard

        real, sys.stdout = sys.stdout, _TerminalWriter(self.app.output)  # type: ignore[assignment]
        try:
            how = clipboard.copy_text(text)
            key = "panel_copied" if how == "system" else "panel_copied_terminal"
        except Exception:
            key = "panel_copy_failed"
        finally:
            sys.stdout = real
        self.note(self.repl.t(key, n=len(text)))

    def note(self, text: str, seconds: float = 3.0) -> None:
        """A message in the status line that goes away by itself."""
        sel = self.selection
        sel.note = text
        if sel.timer is not None:
            sel.timer.cancel()
        with contextlib.suppress(RuntimeError):  # (no running loop: it stays until replaced)
            sel.timer = asyncio.get_running_loop().call_later(seconds, self._note_gone)
        self.invalidate()

    def _note_gone(self) -> None:
        self.selection.note = ""
        self.invalidate()

    def toolbar(self) -> Any:
        note = self.selection.note
        return [("class:accent", f" {note} ")] if note else self.repl.toolbar()

    def scroll_by(self, lines: int) -> None:
        """Positive: back in time. Past the top, older messages are brought in."""
        self.scroll = max(0, self.scroll + lines)
        total = self.log.text.count("\n") + 1
        if lines > 0 and self.hidden_before and self.scroll + self.size()[1] >= total:
            self.load_older()
        self.invalidate()

    def load_older(self) -> None:
        """Put the previous messages of this chat above what is shown."""
        session = self.repl.chat.session
        if session is None or session.id != self.shown_session:
            self.hidden_before = 0  # the chat area belongs to another conversation
            return
        history = self.repl.chat.history()
        end = min(self.hidden_before, len(history))
        older = history[max(0, end - OLDER_MESSAGES) : end]
        self.hidden_before = max(0, end - OLDER_MESSAGES)
        if not older:
            return
        page = ChatLog()
        import sys

        real, sys.stdout = sys.stdout, page  # type: ignore[assignment]
        try:
            self.repl.write_messages(older)
        finally:
            sys.stdout = real
        note = f"\x1b[2m  ↑ {self.repl.t('panel_older')}\x1b[0m\n\n" if self.hidden_before else ""
        self.log.text = note + page.text + self.log.text
        self.log.version += 1
        sel = self.selection
        if sel.anchor is not None and sel.end is not None:  # it stays on the same text
            added = (note + page.text).count("\n")
            sel.anchor, sel.end = (
                (sel.anchor[0] + added, sel.anchor[1]),
                (sel.end[0] + added, sel.end[1]),
            )
            sel.rev += 1

    def _scroll_note(self) -> StyleAndTextTuples:
        return [("class:panel.dim", f" ↓ {self.repl.t('panel_scrolled')} ")] if self.scroll else []

    # ── layout ──────────────────────────────────────────────────────────────

    def _build(self) -> Application[None]:
        chat = Window(
            ScrollableText(
                self,
                text=lambda: self._visible()[0],
                get_cursor_position=lambda: Point(0, self._visible()[1]),
            ),
            wrap_lines=True,
        )
        self._side_window = Window(
            SideText(self, text=self._side_view),
            width=lambda: self.side_width(),
            wrap_lines=False,  # _side() wraps the lines itself, so rows can be counted
            style="class:panel",
        )

        def column(char: str) -> StyleAndTextTuples:  # one line per screen row (mouse hit area)
            return [("class:panel.border", (char + "\n") * self.size()[1])]

        separator = Window(
            DragText(lambda: column("│"), self._separator_event),
            width=1,
            style="class:panel.border",
        )
        side = ConditionalContainer(
            VSplit([separator, self._side_window]),
            filter=Condition(lambda: self.side_width() > 0),
        )
        # catches the pointer while the separator is dragged (see _separator_event)
        catcher = ConditionalContainer(
            # (blank text over the whole screen: the pointer's column is read from the text)
            Window(
                DragText(
                    lambda: [("", (" " * self.size()[0] + "\n") * self.size()[1])],
                    self._separator_event,
                )
            ),
            filter=Condition(lambda: self._dragging),
        )
        question = ConditionalContainer(
            Window(
                FormattedTextControl(lambda: [("class:accent", " " + self.question)]),
                height=Dimension(min=1, max=3),
                wrap_lines=True,
            ),
            filter=Condition(lambda: bool(self.question)),
        )
        entry = Window(
            BufferControl(
                self.input,
                input_processors=[
                    BeforeInput(lambda: [("class:prompt-arrow", "› ")]),
                    ConditionalProcessor(PasswordProcessor(), Condition(lambda: self._password)),
                ],
            ),
            height=Dimension(min=1, max=6),
            wrap_lines=True,
        )
        rule = VSplit([
            Window(height=1, char="─", style="class:panel.border"),
            Window(FormattedTextControl(self._scroll_note), height=1, dont_extend_width=True),
        ])  # fmt: skip
        status = Window(FormattedTextControl(self.toolbar), height=1,
                        style="class:bottom-toolbar")  # fmt: skip
        body = HSplit([VSplit([chat, side]), rule, question, entry, status])
        root = FloatContainer(
            body,
            floats=[
                Float(xcursor=True, ycursor=True, content=CompletionsMenu(max_height=8)),
                Float(content=catcher, left=0, top=0, right=0, bottom=0),
            ],
        )
        session = get_app_session()
        app: Application[None] = Application(
            layout=Layout(root, focused_element=entry),
            key_bindings=self._keys(),
            style=self.repl.ptk_style(),
            full_screen=True,
            mouse_support=True,  # the wheel scrolls the chat; dragging selects and copies
            output=session.output,
            input=session.input,
        )
        app.after_render += self._check_size
        app.after_render += self._draw_picture
        return app

    def _check_size(self, app: Application[None]) -> None:
        """The terminal was resized: the side panel, picture and text width follow."""
        now = self.size()
        if now != self._last_size:
            first = self._last_size is None
            self._last_size = now
            if not first:
                self.refresh()

    # ── scrolling the side panel ────────────────────────────────────────────

    def side_height(self) -> int:
        """Rows the side panel shows (known after the first render)."""
        info = self._side_window.render_info if self._side_window is not None else None
        return info.window_height if info is not None else max(1, self.size()[1] - 3)

    def _side_lines(self) -> list[StyleAndTextTuples]:
        """self.side cut into its lines."""
        lines: list[StyleAndTextTuples] = [[]]
        for frag in self.side:
            for i, part in enumerate(frag[1].split("\n")):
                if i:
                    lines.append([])
                if part:
                    lines[-1].append((frag[0], part))
        if not lines[-1]:
            lines.pop()
        return lines

    def scroll_side(self, lines: int) -> None:
        """Positive: further down the side panel."""
        most = max(0, len(self._side_lines()) - self.side_height())
        self.side_scroll = min(max(0, self.side_scroll + lines), most)
        self.invalidate()

    def _side_view(self) -> StyleAndTextTuples:
        """The side panel from the scroll offset on, with a hint where more is hidden."""
        lines, height = self._side_lines(), self.side_height()
        self.side_scroll = off = min(self.side_scroll, max(0, len(lines) - height))
        shown = lines[off : off + height]
        t = self.repl.t
        if off > 0 and shown:
            shown[0] = [("class:panel.dim", f" ↑ {t('panel_side_above')}")]
        if off + height < len(lines) and shown:
            shown[-1] = [("class:panel.dim", f" ↓ {t('panel_side_below')}")]
        out: StyleAndTextTuples = []
        for i, line in enumerate(shown):
            if i:
                out.append(("", "\n"))
            out.extend(line)
        return out

    def _picture_row(self) -> int | None:
        """The screen row (1 = top) of the picture, or None while any of its lines is
        scrolled out or under the "more below" hint."""
        total, height = len(self._side_lines()), self.side_height()
        top = -self.side_scroll  # the picture starts on the first line
        last = height - (1 if self.side_scroll + height < total else 0)
        return top + 1 if top >= 0 and top + self._pic_rows <= last else None

    def _prepare_picture(self) -> None:
        """Encode the wanted picture now, and keep only that one."""
        if self._pic is None:
            self._encoded = None
        elif self._encoded is None or self._encoded[0] != self._pic:
            try:
                path, w, h = self._pic
                self._encoded = (self._pic, sixel.encode(Path(path), w, h))
            except Exception:
                self._pic, self._encoded = None, None

    def _draw_picture(self, app: Application[None]) -> None:
        """After the screen is drawn: the picture over its blank lines in the panel.
        The screen is only redrawn where something changed, so it stays there; when
        it must go (another picture, a resize), the whole screen is drawn again."""
        want: tuple[Any, ...] | None = None
        side = self.side_width()
        row = self._picture_row() if self._pic is not None and side else None
        if row is not None and self._pic is not None:
            columns, rows = self.size()
            want = (*self._pic, columns - side + 1, columns, rows, row)
        if want == self._drawn:
            return
        if self._drawn is not None:
            self._drawn = None
            app.renderer.clear()  # wipe the old one; the next render draws the new one
            app.invalidate()
            return
        if want is None:
            return
        x = want[3]
        if self._encoded is None or self._encoded[0] != self._pic:
            return  # not encoded yet (refresh() does it); never encode while drawing
        data = self._encoded[1]
        # save the cursor, draw at the picture's place in the panel, put the cursor back
        app.output.write_raw(f"\x1b7\x1b[{want[6]};{x + 1}H{data}\x1b8")
        app.output.flush()
        self._drawn = want

    def _keys(self) -> KeyBindings:
        kb = KeyBindings()
        empty = Condition(lambda: not self.input.text and not self.question)

        @kb.add("enter")
        def _enter(event: Any) -> None:
            state = self.input.complete_state
            if state is not None and state.current_completion is not None:
                self.input.apply_completion(state.current_completion)
                return
            text = self.input.text
            if self._answer is not None and not self._answer.done():
                self.input.reset()
                self._answer.set_result(text)
            elif not self.is_busy():  # one thing at a time; otherwise the text waits
                self.input.append_to_history()
                self.input.reset()
                self.start(text)

        @kb.add("escape", "enter")
        def _newline(event: Any) -> None:
            self.input.insert_text("\n")

        @kb.add("c-c")
        def _stop(event: Any) -> None:
            repl = self.repl
            if self._answer is not None and not self._answer.done():
                self.input.reset()
                self._answer.set_exception(KeyboardInterrupt())
            elif repl.stream_task is not None and not repl.stream_task.done():
                repl.stream_task.cancel()  # stop the reply; what was written stays
            elif self.busy is not None and not self.busy.done():
                self.busy.cancel()
            else:
                self.input.reset()
                repl.save_draft("")  # reset() fires no change event; Ctrl+C discards the draft

        @kb.add("c-d")
        def _quit(event: Any) -> None:
            if not self.input.text and not self.is_busy():
                self.repl.running = False
                event.app.exit()

        @kb.add("pageup")
        def _up(event: Any) -> None:
            self.scroll_by(SCROLL_STEP)

        @kb.add("pagedown")
        def _down(event: Any) -> None:
            self.scroll_by(-SCROLL_STEP)

        @kb.add("escape", "left", filter=empty)
        @kb.add("c-left", filter=empty)
        def _prev(event: Any) -> None:
            if not self.is_busy():
                self.start("/prev")

        @kb.add("escape", "right", filter=empty)
        @kb.add("c-right", filter=empty)
        def _next(event: Any) -> None:
            if not self.is_busy():
                self.start("/next")

        @kb.add("c-l")
        def _redraw(event: Any) -> None:
            self._drawn = None  # the screen is wiped: the picture must be drawn again
            event.app.renderer.clear()
            event.app.invalidate()

        @kb.add("escape", "pageup")
        def _side_up(event: Any) -> None:
            self.scroll_side(-SIDE_STEP)

        @kb.add("escape", "pagedown")
        def _side_down(event: Any) -> None:
            self.scroll_side(SIDE_STEP)

        @kb.add("escape", ",")
        def _narrower(event: Any) -> None:
            if self.side_width():
                self.resize(self.side_width() - RESIZE_STEP)

        @kb.add("escape", ".")
        def _wider(event: Any) -> None:
            if self.side_width():
                self.resize(self.side_width() + RESIZE_STEP)

        @kb.add("c-p")
        def _palette(event: Any) -> None:
            if not self.is_busy():
                self.busy = asyncio.ensure_future(self._run(self._palette()))

        return kb

    # ── running lines ───────────────────────────────────────────────────────

    def is_busy(self) -> bool:
        return self.busy is not None and not self.busy.done()

    def start(self, line: str) -> None:
        self.scroll = 0
        self.busy = asyncio.ensure_future(self._run(self.repl.process_line(line)))

    async def _run(self, job: Any) -> None:
        try:
            await job
        except (asyncio.CancelledError, KeyboardInterrupt):  # Ctrl+C in a question too
            self.repl.info(self.repl.t("aborted"))
        except (ChatError, ValueError) as e:
            self.repl.warn(str(e))
        except ProviderError as e:
            self.repl.error(str(e))
        except Exception as e:  # the screen must survive anything a command does
            self.repl.error(f"{type(e).__name__}: {e}")
        self.refresh()
        if not self.repl.running or self.repl.layout_changed:
            self.app.exit()

    async def _palette(self) -> None:
        prefill = await self.repl.palette() or ""
        if prefill:
            self.input.text, self.input.cursor_position = prefill, len(prefill)

    def invalidate(self) -> None:
        with contextlib.suppress(Exception):
            self.app.invalidate()

    # ── asking in the input box ─────────────────────────────────────────────

    async def ask(self, question: str, default: str = "", password: bool = False) -> str:
        """Ctrl+C raises KeyboardInterrupt, as in a normal prompt."""
        self._answer = asyncio.get_running_loop().create_future()
        typed = self.input.text  # what the user was writing meanwhile comes back after
        self.question, self._password = question, password
        self.input.text, self.input.cursor_position = default, len(default)
        self.scroll = 0
        self.invalidate()
        try:
            return await self._answer
        finally:
            self.question, self._password, self._answer = "", False, None
            self.input.reset(document=Document(typed, len(typed)))  # no change events
            self.invalidate()

    async def pick(self, title: str, items: list[PickItem], current: Any = None) -> Any:
        """A numbered list in the chat area; the answer is a number or part of a name."""
        repl = self.repl
        repl.heading(title)
        for i, item in enumerate(items, 1):
            mark = "▸" if item.value == current else " "
            hint = f"  · {item.hint}" if item.hint else ""
            repl.console.print(f" {mark}{i:>3}) {item.label}{hint}", markup=False)
        while True:
            try:
                answer = (await self.ask(repl.t("panel_pick", title=title))).strip()
            except KeyboardInterrupt:
                return None
            if not answer:
                return None
            if answer.isdigit() and 1 <= int(answer) <= len(items):
                return items[int(answer) - 1].value
            hits = [it for it in items if answer.lower() in it.label.lower()]
            if len(hits) == 1:
                return hits[0].value
            repl.warn(repl.t("panel_pick_again", n=len(hits)))

    # ── the side panel ──────────────────────────────────────────────────────

    def refresh(self) -> None:
        """Rebuild the side panel (after a reply, a command, a new picture…)."""
        self._pic, self._pic_rows = None, 0  # _side() sets them when there is a picture
        sid = self.repl.chat.session.id if self.repl.chat.session else None
        if sid != self._side_session:  # another chat: its panel starts at the top
            self._side_session, self.side_scroll = sid, 0
        try:
            self.side = self._side()
        except Exception:  # the panel must never break the chat
            self.side, self._pic = [], None
        self._prepare_picture()
        self.repl.console.width = self.chat_width()
        self.invalidate()

    def _side(self) -> StyleAndTextTuples:
        repl, chat = self.repl, self.repl.chat
        width = self.side_width()
        if not width or chat.session is None:
            return []
        t = repl.t
        out: StyleAndTextTuples = []

        def line(text: str, style: str = "class:panel") -> None:
            for part in textwrap.wrap(text, max(8, width - 2)) or [""]:  # rows are counted
                out.append((style, f" {part}\n"))

        def gap() -> None:
            out.append(("", "\n"))

        char = chat.character()
        picture = repl.last_picture() if char is not None else None
        data = panel_data(chat, picture)
        rows = max(6, self.size()[1] // 2 - 2)
        if picture is not None and sixel.supported(repl.inline_setting()):
            # the real picture, drawn by the terminal; here only the room for it
            from PIL import Image

            cw, ch = sixel.cell_size()
            with Image.open(picture) as img:
                w, h = fit(img.size, (width - 2) * cw, rows * ch)
            self._pic = (str(picture), w, h)
            self._pic_rows = -(-h // ch)
            out.append(("", "\n" * self._pic_rows))
            gap()
        elif picture is not None:  # no Sixel: a preview in coloured blocks
            key = (str(picture), width - 2, rows)
            if self._thumb is None or self._thumb[:3] != key:
                self._thumb = (*key, thumbnail(picture, width - 2, rows))
            for row in self._thumb[3]:
                out.append(("", " "))
                out.extend(row)
                out.append(("", "\n"))
            gap()
        if char is not None:
            color = repl.assistant_color()
            age = data.age
            line(f"{char.name}{f' · {age}' if age else ''}",
                 f"bold fg:{color}" if color != "default" else "bold")  # fmt: skip
            flags = [data.content] + (["private"] if data.private else [])
            line(" · ".join(flags), "class:panel.dim")
            if data.personality:
                gap()
                line(t("panel_personality"), "class:accent")
                for trait in data.personality:
                    line(trait)
        else:
            line(data.title or t("untitled"), "bold")
        if data.scene:
            gap()
            line(t("panel_scene"), "class:accent")
            for key, value in data.scene.items():
                line(f"{t('panel_' + key)}: {value}")
        if data.memories:
            gap()
            line(t("panel_memories"), "class:accent")
            for m in data.memories:
                line(f"{'*' if m['pinned'] else '·'} {m['content']}")
        gap()
        line((data.model or "—").split(":", 1)[-1], "class:panel.dim")
        return out
