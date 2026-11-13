import io
import re

import pytest

from chatclient.config import ConfigError, load_config
from chatclient.theme import PRESETS, ansi, resolve_theme, rich_style, valid_color
from chatclient.ui.render import StreamWriter

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def visible(s: str) -> str:
    return ANSI_RE.sub("", s)


def test_presets_are_complete_and_valid():
    assert {"soft", "warm", "nord", "rose", "light", "classic"} <= set(PRESETS)
    for th in PRESETS.values():
        for key, value in vars(th).items():
            if key != "name":
                assert valid_color(value), (th.name, key, value)


def test_resolve_theme_overrides_and_fallback():
    th = resolve_theme("nord", {"accent": "#ff0000", "text": "bogus", "nope": "#000"})
    assert th.accent == "#ff0000"
    assert th.text == PRESETS["nord"].text  # invalid override ignored
    assert resolve_theme("does-not-exist").name == "soft"


def test_color_renderers():
    assert ansi("#ff8000", "truecolor") == "\x1b[38;2;255;128;0m"
    assert ansi("#f80", "truecolor") == "\x1b[38;2;255;136;0m"
    assert ansi("default", "truecolor") == "\x1b[39m"
    # terminals without 24-bit colour get the nearest 256-palette entry
    assert ansi("#ff0000", "256") == "\x1b[38;5;196m"
    assert ansi("#808080", "256") == "\x1b[38;5;244m"  # grey ramp
    assert ansi("#ff0000", "none") == ""
    assert rich_style("default", "bold") == "bold"
    assert rich_style("#123456", "italic") == "italic #123456"


def test_color_mode_detection(monkeypatch):
    import chatclient.theme as theme

    for var in ("NO_COLOR", "COLORTERM", "TERM", "TERM_PROGRAM"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(theme.os, "name", "posix")
    assert theme.color_mode() == "256"
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert theme.color_mode() == "truecolor"
    monkeypatch.setenv("NO_COLOR", "1")
    assert theme.color_mode() == "none"


def test_config_validates_theme_and_colors(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[ui]\ntheme = "warm"\n[ui.colors]\naccent = "#abc"\n', encoding="utf-8")
    cfg = load_config(path)
    assert cfg.ui.theme == "warm" and cfg.ui.colors == {"accent": "#abc"}
    path.write_text('[ui]\ntheme = "neon"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown theme"):
        load_config(path)
    path.write_text('[ui.colors]\naccent = "blue"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="#rrggbb"):
        load_config(path)
    path.write_text('[ui.colors]\nsparkle = "#fff"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown colour"):
        load_config(path)


def test_writer_styles_actions_and_hides_asterisks():
    buf = io.StringIO()
    w = StreamWriter(buf, width=80, margin=0, base="<B>", action="<A>", style_actions=True)
    for chunk in ["*Gülüm", "süyor.* Merhaba ", "*başını sallar*"]:
        w.write(chunk)
    w.finish()
    out = buf.getvalue()
    assert "*" not in out
    assert "\x1b[3m<A>Gülümsüyor.\x1b[23m<B> Merhaba" in out


def test_writer_keeps_asterisks_when_styling_is_off():
    buf = io.StringIO()
    w = StreamWriter(buf, width=80, margin=0, style_actions=False)
    w.write("*nods*")
    w.finish()
    assert visible(buf.getvalue()) == "*nods*"


def test_writer_indents_every_line_and_ignores_escape_width():
    buf = io.StringIO()
    w = StreamWriter(
        buf, width=24, margin=0, indent="  ", base="\x1b[38;2;1;2;3m", action="\x1b[38;2;9;9;9m",
        style_actions=True,
    )  # fmt: skip
    text = "*Kapı açılıyor.* Yağmur hâlâ yağıyor ve rüzgâr camları titretiyor.\n\nİkinci paragraf."
    for i in range(0, len(text), 5):  # arbitrary chunking
        w.write(text[i : i + 5])
    w.finish()
    lines = visible(buf.getvalue()).split("\n")
    for line in lines:
        assert len(line) <= 24
        assert line == "" or line.startswith("  ")
    words = " ".join(lines).split()
    assert words == [
        "Kapı",
        "açılıyor.",
        "Yağmur",
        "hâlâ",
        "yağıyor",
        "ve",
        "rüzgâr",
        "camları",
        "titretiyor.",
        "İkinci",
        "paragraf.",
    ]
