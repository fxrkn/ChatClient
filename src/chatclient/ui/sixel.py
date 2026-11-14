"""Pictures inside the terminal (Sixel: Windows Terminal 1.22+, WezTerm, foot, mlterm…).

The image is reduced to a palette, then written six pixel rows at a time: for each
colour in a band, one character per column says which of the six pixels have it.
numpy finds the runs, so even a 512×512 picture encodes in a fraction of a second.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

COLORS = 128


_asked: bool | None = None  # what the terminal answered at start-up (None: not asked)


def supported(setting: str = "auto") -> bool:
    """``auto``: the terminal said so when asked, or is known to. ``sixel`` forces it,
    ``off`` never."""
    if setting == "sixel":
        return True
    if setting != "auto":
        return False
    if _asked is not None:
        return _asked
    term = os.environ.get("TERM", "").lower()
    program = os.environ.get("TERM_PROGRAM", "")
    return bool(
        os.environ.get("WT_SESSION")  # Windows Terminal (Sixel since 1.22)
        or program in ("WezTerm", "iTerm.app", "mintty")
        or any(name in term for name in ("sixel", "foot", "mlterm", "contour"))
    )


def has_sixel(da1_reply: str) -> bool:
    """The terminal's answer to "who are you?" (DA1, ``ESC [ c``) lists its features;
    4 means Sixel graphics. E.g. Windows Terminal 1.22+: ``ESC[?61;4;6;7;…c``."""
    start, end = da1_reply.find("[?"), da1_reply.rfind("c")
    if start < 0 or end < start:
        return False
    return "4" in da1_reply[start + 2 : end].split(";")


def ask_terminal(timeout: float = 0.4) -> None:
    """Ask the terminal once, before the prompt starts reading input. Whatever goes
    wrong (not a terminal, no answer) leaves the old guess in place."""
    global _asked
    import sys

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return
    try:
        reply = _read_da1_windows(timeout) if os.name == "nt" else _read_da1_posix(timeout)
    except Exception:
        return
    if reply:
        _asked = has_sixel(reply)


def _read_da1_windows(timeout: float) -> str:
    import ctypes
    import msvcrt
    import sys
    import time
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.GetStdHandle(-10)  # stdin
    mode = wintypes.DWORD()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return ""
    # VT input on (the answer arrives as input), line input and echo off
    kernel32.SetConsoleMode(handle, (mode.value | 0x0200) & ~0x0006)
    try:
        sys.stdout.write("\x1b[c")
        sys.stdout.flush()
        reply, deadline = "", time.monotonic() + timeout
        while time.monotonic() < deadline:
            if msvcrt.kbhit():
                reply += msvcrt.getwch()
                if reply.endswith("c") and "[?" in reply:
                    break
            else:
                time.sleep(0.01)
        return reply
    finally:
        kernel32.SetConsoleMode(handle, mode.value)


def _read_da1_posix(timeout: float) -> str:
    import select
    import sys
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        sys.stdout.write("\x1b[c")
        sys.stdout.flush()
        reply = ""
        while select.select([fd], [], [], timeout)[0]:
            reply += os.read(fd, 64).decode("ascii", "replace")
            if reply.endswith("c") and "[?" in reply:
                break
        return reply
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def encode(path: Path, max_width: int = 384, max_height: int = 512) -> str:
    from PIL import Image

    with Image.open(path) as src:
        img = src.convert("RGB")
    scale = min(1.0, max_width / img.width, max_height / img.height)
    if scale < 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    q = img.quantize(colors=COLORS, method=Image.Quantize.FASTOCTREE)
    pixels = np.asarray(q, dtype=np.uint8)
    h, w = pixels.shape
    palette = q.getpalette() or []
    out = ["\x1bPq", f'"1;1;{w};{h}']
    for i in range(min(COLORS, len(palette) // 3)):
        r, g, b = (round(v * 100 / 255) for v in palette[i * 3 : i * 3 + 3])
        out.append(f"#{i};2;{r};{g};{b}")
    for top in range(0, h, 6):
        band = pixels[top : top + 6]
        weights = (1 << np.arange(band.shape[0], dtype=np.int32))[:, None]
        layers = []
        for color in np.unique(band):
            bits = ((band == color) * weights).sum(axis=0)
            layers.append(f"#{color}{_runs(bits)}")
        out.append("$".join(layers) + "-")
    out.append("\x1b\\")
    return "".join(out)


def _runs(bits: np.ndarray) -> str:
    """Run-length coded sixel characters for one colour of one band."""
    starts = np.concatenate(([0], np.flatnonzero(np.diff(bits)) + 1))
    lengths = np.diff(np.concatenate((starts, [len(bits)])))
    values = bits[starts]
    if values[-1] == 0:  # nothing to draw at the end of the line
        starts, lengths, values = starts[:-1], lengths[:-1], values[:-1]
    parts = []
    for value, n in zip(values.tolist(), lengths.tolist(), strict=True):
        ch = chr(63 + value)
        parts.append(f"!{n}{ch}" if n > 3 else ch * n)
    return "".join(parts)
