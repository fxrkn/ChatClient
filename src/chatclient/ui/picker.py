"""Inline arrow-key picker: ↑/↓ to move, type to filter, Enter to choose, Esc to cancel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import BaseStyle

from chatclient.text import normalize_for_search


@dataclass(frozen=True)
class PickItem:
    value: Any
    label: str
    hint: str = ""


class Picker:
    def __init__(
        self,
        title: str,
        items: Sequence[PickItem],
        *,
        current: Any = None,
        footer: str = "",
        max_rows: int = 10,
    ):
        self.title = title
        self.items = list(items)
        self.footer = footer
        self.max_rows = max_rows
        self.filter = ""
        self.index = next((i for i, it in enumerate(self.items) if it.value == current), 0)

    def visible(self) -> list[PickItem]:
        if not self.filter:
            return self.items
        needle = normalize_for_search(self.filter)
        return [it for it in self.items if needle in normalize_for_search(f"{it.label} {it.hint}")]

    def move(self, step: int) -> None:
        n = len(self.visible())
        if n:
            self.index = (self.index + step) % n

    def type(self, ch: str) -> None:
        self.filter += ch
        self.index = 0

    def backspace(self) -> None:
        self.filter = self.filter[:-1]
        self.index = 0

    def selected(self) -> PickItem | None:
        items = self.visible()
        return items[self.index] if items and 0 <= self.index < len(items) else None

    def render(self) -> StyleAndTextTuples:
        items = self.visible()
        out: StyleAndTextTuples = [("class:picker.title", f"{self.title}\n")]
        if self.filter:
            out.append(("class:picker.hint", f"  ⌕ {self.filter}\n"))
        start = max(0, min(self.index - self.max_rows // 2, len(items) - self.max_rows))
        for i, it in enumerate(items[start : start + self.max_rows], start):
            current = i == self.index
            out.append(
                ("class:picker.selected" if current else "class:picker.item",
                 f"  {'▸' if current else ' '} {it.label}")
            )  # fmt: skip
            if it.hint:
                out.append(("class:picker.hint", f"  {it.hint}"))
            out.append(("", "\n"))
        if not items:
            out.append(("class:picker.hint", "  —\n"))
        if len(items) > self.max_rows:
            out.append(("class:picker.hint", f"  ({len(items)})\n"))
        if self.footer:
            out.append(("class:picker.hint", f"  {self.footer}"))
        return out


async def pick(
    title: str,
    items: Sequence[PickItem],
    *,
    current: Any = None,
    footer: str = "",
    style: BaseStyle | None = None,
) -> Any | None:
    """Show the picker under the prompt; returns the chosen value, or None if cancelled."""
    if not items:
        return None
    picker = Picker(title, items, current=current, footer=footer)
    kb = KeyBindings()

    @kb.add("up")
    @kb.add("c-p")
    def _up(event: KeyPressEvent) -> None:
        picker.move(-1)

    @kb.add("down")
    @kb.add("c-n")
    def _down(event: KeyPressEvent) -> None:
        picker.move(1)

    @kb.add("pageup")
    def _pgup(event: KeyPressEvent) -> None:
        picker.move(-picker.max_rows)

    @kb.add("pagedown")
    def _pgdn(event: KeyPressEvent) -> None:
        picker.move(picker.max_rows)

    @kb.add("enter")
    def _enter(event: KeyPressEvent) -> None:
        chosen = picker.selected()
        event.app.exit(result=chosen.value if chosen else None)

    @kb.add("escape", eager=True)
    @kb.add("c-c")
    def _cancel(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    @kb.add("backspace")
    def _back(event: KeyPressEvent) -> None:
        picker.backspace()

    @kb.add("<any>")
    def _type(event: KeyPressEvent) -> None:
        if event.data and event.data.isprintable():
            picker.type(event.data)

    control = FormattedTextControl(picker.render, focusable=True, show_cursor=False)
    app: Application[Any] = Application(
        layout=Layout(Window(control, dont_extend_height=True)),
        key_bindings=kb,
        style=style,
        full_screen=False,
        erase_when_done=True,
    )
    return await app.run_async()
