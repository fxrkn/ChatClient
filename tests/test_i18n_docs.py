"""docs/DOCUMENTATION.md covers every command the app knows."""

from pathlib import Path

from chatclient.i18n import COMMAND_DOCS

DOC = Path(__file__).resolve().parents[1] / "docs" / "DOCUMENTATION.md"


def test_every_command_is_documented():
    text = DOC.read_text(encoding="utf-8")
    missing = [d.command for d in COMMAND_DOCS if f"`{d.command}" not in text]
    assert missing == [], f"add these to docs/DOCUMENTATION.md: {missing}"
