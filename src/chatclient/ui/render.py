"""Low-level terminal output: word-wrapping streamed text and Windows VT setup."""

from __future__ import annotations

import os
import shutil
import sys
from typing import TextIO

from chatclient.theme import BOLD, ITALIC, NO_ITALIC, RESET

__all__ = ["BOLD", "RESET", "StreamWriter", "enable_windows_vt", "reading_width"]


def enable_windows_vt() -> None:
    """Turn on ANSI escape processing for the Windows console (no-op elsewhere)."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(
                handle, mode.value | 0x0004
            )  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def reading_width(max_width: int | None = None, margin: int = 1) -> int:
    term = shutil.get_terminal_size((100, 24)).columns - margin
    return max(20, min(term, max_width) if max_width else term)


class StreamWriter:
    """Writes streamed text with word wrapping, so words aren't split at the edge.

    Tracks the cursor column itself; text arrives in arbitrary chunks, so a partial
    word is held back until whitespace (or ``finish``) shows where it ends.

    Optional styling: ``indent`` prefixes every line, ``base`` is the text colour
    escape, and with ``style_actions`` roleplay ``*actions*`` are drawn in italics
    with the ``action`` colour and the asterisks are hidden. Escape codes never
    count toward the line width.
    """

    def __init__(
        self,
        out: TextIO | None = None,
        width: int | None = None,
        margin: int = 1,
        *,
        indent: str = "",
        base: str = "",
        action: str = "",
        style_actions: bool = False,
    ):
        self.out = out or sys.stdout
        self.width = max(20, (width or shutil.get_terminal_size((100, 24)).columns) - margin)
        self.indent = indent
        self.base = base
        self.action = action
        self.style_actions = style_actions
        self.col = 0
        self.word = ""  # visible characters of the pending word
        self.raw = ""  # the pending word including escape codes
        self.wrote_any = False
        self.in_action = False
        self.out.write(base)

    def start_at(self, col: int) -> None:
        self.col = col

    def _newline(self) -> None:
        self.out.write("\n")
        self.col = 0

    def _ensure_indent(self) -> None:
        if self.col == 0 and self.indent:
            self.out.write(self.indent)
            self.col = len(self.indent)

    def write(self, text: str) -> None:
        for ch in text:
            if ch == "\n":
                self._flush_word()
                self._newline()
            elif ch in " \t":
                self._flush_word()
                if self.col <= len(self.indent) and self.wrote_any:
                    continue  # don't start a (wrapped) line with a space
                if self.col + 1 >= self.width:
                    self._newline()
                    continue
                self._ensure_indent()
                self.out.write(ch)
                self.col += 1
            elif ch == "*" and self.style_actions:
                self.in_action = not self.in_action
                self.raw += (ITALIC + self.action) if self.in_action else (NO_ITALIC + self.base)
            else:
                self.word += ch
                self.raw += ch
                if len(self.word) >= self.width - len(self.indent):  # one huge token
                    self._flush_word()
        self.out.flush()

    def _flush_word(self) -> None:
        if not self.raw:
            return
        if self.word and self.col > len(self.indent) and self.col + len(self.word) > self.width:
            self._newline()
        if self.word:
            self._ensure_indent()
        self.out.write(self.raw)
        self.col += len(self.word)
        if self.word:
            self.wrote_any = True
        self.word = self.raw = ""

    def finish(self) -> None:
        self._flush_word()
        self.out.write(RESET)
        self.out.flush()
