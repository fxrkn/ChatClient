"""Language-aware text helpers: search normalisation and token estimation."""

from __future__ import annotations

import re
import unicodedata

# All i-variants collapse to plain "i" so "İstanbul", "ISTANBUL", "ıstanbul" and
# "istanbul" match each other regardless of Turkish vs. English casing rules.
_I_VARIANTS = str.maketrans({"İ": "i", "I": "i", "ı": "i"})
# Characters with no decomposition that people commonly type without diacritics.
_EXTRA_FOLD = str.maketrans({"ß": "ss", "æ": "ae", "ø": "o", "đ": "d", "ł": "l", "œ": "oe"})
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'", "´": "'"})
_WS = re.compile(r"\s+")


def normalize_for_search(text: str) -> str:
    """Fold text for matching: case-, diacritic- and i/ı-insensitive.

    "Kütüphane'deki İz" → "kutuphane'deki iz". Searching "kutuphane" therefore finds
    "kütüphanedeki", which matters for agglutinative languages like Turkish and for
    users typing without Turkish characters.
    """
    t = unicodedata.normalize("NFKC", text).translate(_I_VARIANTS).translate(_APOSTROPHES)
    t = t.lower()
    t = unicodedata.normalize("NFD", t)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = t.translate(_EXTRA_FOLD)
    return _WS.sub(" ", t).strip()


def estimate_tokens(text: str) -> int:
    """Conservative token estimate for when a provider can't count exactly.

    English averages ~4 chars/token, Turkish and most other languages fewer, so
    we use ~3 chars/token to avoid overflowing context windows.
    """
    if not text:
        return 0
    return len(text) // 3 + 1


MESSAGE_OVERHEAD_TOKENS = 4  # role markers / separators per message


def short_title(text: str, limit: int = 48) -> str:
    """A one-line session title derived from the first user message."""
    line = _WS.sub(" ", text).strip()
    if len(line) <= limit:
        return line
    cut = line[:limit].rsplit(" ", 1)[0]
    return (cut or line[:limit]) + "…"
