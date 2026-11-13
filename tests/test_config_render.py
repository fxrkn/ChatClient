import io
import re

import pytest

from chatclient.config import DEFAULT_CONFIG_TOML, ConfigError, load_config
from chatclient.i18n import Translator, help_text
from chatclient.ui.render import StreamWriter


def test_default_template_is_valid(tmp_path):
    path = tmp_path / "config.toml"
    cfg = load_config(path)  # writes the template
    assert path.read_text(encoding="utf-8") == DEFAULT_CONFIG_TOML
    assert {"ollama", "lmstudio", "openrouter", "anthropic", "gemini"} <= set(cfg.providers)


def test_typos_are_reported(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[ui]\nlanguag = "tr"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="languag"):
        load_config(path)


def test_config_with_utf8_bom(tmp_path):
    path = tmp_path / "config.toml"
    path.write_bytes('﻿[ui]\nlanguage = "tr"\n'.encode())
    assert load_config(path).ui.language == "tr"


def test_translator_falls_back_to_english():
    en, tr = help_text(Translator("en")), help_text(Translator("tr"))
    assert en.startswith("Commands") and "/continue" in en and "Ctrl+P" in en
    assert tr.startswith("Komutlar") and "karakter dışı" in tr
    assert Translator("tr")("missing-key") == "missing-key"


def test_stream_writer_wraps_on_word_boundaries():
    buf = io.StringIO()
    w = StreamWriter(buf, width=21, margin=1)  # 20 usable columns
    for chunk in ["Bir zamanlar uzak", " bir kraliyette ya", "şayan bir ejderha"]:
        w.write(chunk)
    w.finish()
    lines = re.sub(r"\x1b\[[0-9;]*m", "", buf.getvalue()).split("\n")  # count visible chars
    assert all(len(line) <= 20 for line in lines)
    assert " ".join(lines).split() == [
        "Bir",
        "zamanlar",
        "uzak",
        "bir",
        "kraliyette",
        "yaşayan",
        "bir",
        "ejderha",
    ]
