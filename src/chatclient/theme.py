"""Colour themes for the terminal UI.

Fonts are chosen by the terminal app (e.g. Windows Terminal settings), not by
programs running inside it; everything colour-related is configurable here.
Colours are "#rrggbb" / "#rgb" hex values, or "default" for the terminal's own colour.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields, replace

COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


@dataclass(frozen=True)
class Theme:
    name: str
    text: str  # reply text
    accent: str  # character name, prompt arrow, headings
    user: str  # your name
    action: str  # *actions* in roleplay replies (shown in italics)
    dim: str  # info lines, stats
    warn: str
    error: str
    bar_bg: str  # bottom toolbar background
    bar_fg: str  # bottom toolbar text
    speech: str  # "quoted speech" in replies, a touch brighter than narration


COLOR_KEYS = tuple(f.name for f in fields(Theme) if f.name != "name")

PRESETS: dict[str, Theme] = {
    # Low-contrast warm greys with a soft blue accent: easy on the eyes for long sessions.
    "soft": Theme("soft", "#d6d3d1", "#93c5fd", "#a7c4a0", "#a8a29e", "#78716c",
                  "#e6c07b", "#e57373", "#262626", "#a8a29e", "#f0e6d6"),
    # Sepia / Gruvbox-like: warm, paper-ish; good at night.
    "warm": Theme("warm", "#ebdbb2", "#d8a657", "#a9b665", "#a89984", "#7c6f64",
                  "#e78a4e", "#ea6962", "#32302f", "#a89984", "#fbf1c7"),
    # Cool arctic blues (Nord-like).
    "nord": Theme("nord", "#d8dee9", "#88c0d0", "#a3be8c", "#81a1c1", "#616e88",
                  "#ebcb8b", "#bf616a", "#2e3440", "#81a1c1", "#eceff4"),
    # Muted rose and lavender (Rosé Pine-like).
    "rose": Theme("rose", "#e0def4", "#ebbcba", "#9ccfd8", "#908caa", "#6e6a86",
                  "#f6c177", "#eb6f92", "#1f1d2e", "#908caa", "#f2e9e1"),
    # For light terminal backgrounds.
    "light": Theme("light", "#3f3f46", "#2563eb", "#15803d", "#71717a", "#a1a1aa",
                   "#b45309", "#b91c1c", "#e4e4e7", "#52525b", "#18181b"),
    # The terminal's own colours, nothing imposed.
    "classic": Theme("classic", "default", "default", "default", "default", "default",
                     "default", "default", "default", "default", "default"),
}  # fmt: skip

# Automatic per-character name colours: soft pastels for dark themes, deeper tones for light.
CHARACTER_PALETTE_DARK = (
    "#93c5fd", "#f9a8d4", "#86efac", "#fcd34d", "#c4b5fd", "#fdba74", "#67e8f9", "#fca5a5",
)  # fmt: skip
CHARACTER_PALETTE_LIGHT = (
    "#1d4ed8", "#be185d", "#15803d", "#a16207", "#6d28d9", "#c2410c", "#0e7490", "#b91c1c",
)  # fmt: skip


def is_light(theme: Theme) -> bool:
    """A theme with dark text is meant for a light background."""
    if theme.text == "default":
        return False
    r, g, b = _rgb(theme.text)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 128


def character_color(name: str, theme: Theme, override: str | None = None) -> str:
    """The colour for a character's name: the card's own colour, else a stable pick
    from the palette (same name → same colour), else the accent on the classic theme."""
    if override and valid_color(override):
        return override
    if theme.accent == "default":
        return theme.accent
    palette = CHARACTER_PALETTE_LIGHT if is_light(theme) else CHARACTER_PALETTE_DARK
    digest = sum((i + 1) * ord(c) for i, c in enumerate(name.casefold()))
    return palette[digest % len(palette)]


DEFAULT_THEME = "soft"


def valid_color(value: str) -> bool:
    return value == "default" or bool(COLOR_RE.match(value))


def resolve_theme(name: str, overrides: dict[str, str] | None = None) -> Theme:
    base = PRESETS.get(name) or PRESETS[DEFAULT_THEME]
    changes = {k: v for k, v in (overrides or {}).items() if k in COLOR_KEYS and valid_color(v)}
    return replace(base, **changes) if changes else base


def _rgb(color: str) -> tuple[int, int, int]:
    h = color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# ── renderers for the three output paths (raw ANSI, Rich, prompt_toolkit) ──

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
ITALIC = "\x1b[3m"
NO_ITALIC = "\x1b[23m"
DEFAULT_FG = "\x1b[39m"


def color_mode() -> str:
    """ "none" (NO_COLOR set), "truecolor" or "256", from the usual environment hints."""
    if os.environ.get("NO_COLOR"):
        return "none"
    if os.name == "nt":
        return "truecolor"  # Windows 10+ consoles and Windows Terminal
    if os.environ.get("COLORTERM", "").lower() in ("truecolor", "24bit"):
        return "truecolor"
    term = os.environ.get("TERM", "").lower()
    if "direct" in term or os.environ.get("TERM_PROGRAM") in ("iTerm.app", "WezTerm", "vscode"):
        return "truecolor"
    return "256"


def _to_256(r: int, g: int, b: int) -> int:
    """Nearest xterm-256 colour (6×6×6 cube or the grey ramp)."""
    levels = (0, 95, 135, 175, 215, 255)

    def nearest(v: int) -> int:
        return min(range(6), key=lambda i: abs(levels[i] - v))

    ri, gi, bi = nearest(r), nearest(g), nearest(b)
    cube = 16 + 36 * ri + 6 * gi + bi
    cube_err = (levels[ri] - r) ** 2 + (levels[gi] - g) ** 2 + (levels[bi] - b) ** 2
    grey_i = max(0, min(23, round(((r + g + b) / 3 - 8) / 10)))
    grey_v = 8 + 10 * grey_i
    grey_err = (grey_v - r) ** 2 + (grey_v - g) ** 2 + (grey_v - b) ** 2
    return 232 + grey_i if grey_err < cube_err else cube


def ansi(color: str, mode: str | None = None) -> str:
    """Foreground escape: 24-bit where supported, otherwise the 256-colour palette."""
    mode = mode or color_mode()
    if mode == "none":
        return ""
    if color == "default":
        return DEFAULT_FG
    r, g, b = _rgb(color)
    if mode == "truecolor":
        return f"\x1b[38;2;{r};{g};{b}m"
    return f"\x1b[38;5;{_to_256(r, g, b)}m"


def rich_style(color: str, extra: str = "") -> str:
    base = "" if color == "default" else color
    return f"{extra} {base}".strip() or "none"


def ptk_color(color: str, bg: bool = False) -> str:
    if color == "default":
        return ""
    return f"bg:{color}" if bg else color
