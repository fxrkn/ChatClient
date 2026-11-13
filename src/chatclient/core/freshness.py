"""Keeping replies fresh: measure repetition instead of guessing at it.

Small models drift into loops: the same opening, the same sentences, sometimes the
previous reply copied whole. Two tools, both language-independent:

- ``repetition``: the share of a reply's word triples that already appeared in recent
  replies (0 = all new, 1 = a copy). The app rewrites a reply that scores high.
- ``fresh_note``: a short note for the next turn built from what was actually written
  last time (its opening), instead of the same static text every turn, which small
  models end up parroting.
"""

from __future__ import annotations

from collections.abc import Sequence

from chatclient.text import normalize_for_search

REPEAT_THRESHOLD = 0.45  # above this a reply is mostly recycled text
RECENT_REPLIES = 4
OPENING_WORDS = 6


def _words(text: str) -> list[str]:
    return normalize_for_search(text).split()


def _triples(words: list[str]) -> set[tuple[str, ...]]:
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}


def repetition(reply: str, previous: Sequence[str]) -> float:
    """How much of ``reply`` was already written in ``previous`` replies (0..1)."""
    new = _triples(_words(reply))
    if len(new) < 5:  # too short to judge
        return 0.0
    seen: set[tuple[str, ...]] = set()
    for p in previous:
        seen |= _triples(_words(p))
    return len(new & seen) / len(new)


def is_repetitive(reply: str, previous: Sequence[str]) -> bool:
    return repetition(reply, previous) >= REPEAT_THRESHOLD


def opening(text: str, words: int = OPENING_WORDS) -> str:
    plain = " ".join(text.replace("*", " ").split())
    parts = plain.split(" ")
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def fresh_note(previous_reply: str, user: str) -> str:
    """For the next turn: don't echo the last reply; answer what the user just did."""
    if not previous_reply.strip():
        return ""
    return (
        f'Your previous reply began "{opening(previous_reply)}". Begin differently and reuse '
        f"no sentences or images from earlier replies. Respond to what {user} just wrote, "
        "staying on their topic."
    )


def rewrite_note(repeated: str, user: str) -> str:
    """Stronger: the draft just written was recycled; ask for a genuinely new one."""
    return (
        f'Your draft repeated earlier replies (it began "{opening(repeated)}"). Write a '
        f"completely new reply in new words that responds directly to {user}'s last message."
    )
