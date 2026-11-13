"""Characters: cards (Character Card V3 compatible), macros and prompt compilation.

A card is plain data, so any number of characters cost nothing to keep. All prompt
assembly here is deterministic code: no LLM calls.

Card format: we store the CCv3 envelope ``{"spec": "chara_card_v3", "data": {...}}``
and read V2 / bare cards too. https://github.com/kwaroran/character-card-spec-v3
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources
from typing import Any

# Fields we read and write (standard CCv2/CCv3 names).
TEXT_FIELDS = (
    "name",
    "description",
    "personality",
    "scenario",
    "first_mes",
    "mes_example",
    "system_prompt",
    "post_history_instructions",
    "creator_notes",
)

DEFAULT_USER_NAME = "User"

NARRATION_RULES = """\
You are {{char}} in an ongoing, immersive one-on-one conversation with {{user}}. This is \
collaborative fiction: stay fully in character at all times.
- Speak and think as {{char}}: their voice, values, moods, flaws and history. Never mention \
being an AI, a model or an assistant, and never break character unless {{user}} writes (OOC: ...).
- Make the world tangible: weave in concrete details of the place, light, sound, smell, touch \
and small physical actions. Show feelings through behaviour instead of naming them.
- Notice {{user}}: react to their tone, word choice and what they leave unsaid; remember what \
they share and bring it back naturally later.
- {{char}} has an inner life: their own wants, opinions and small worries. They may ask \
questions, change the subject or let something happen in the scene.
- Never write {{user}}'s words, thoughts or actions. Stop when it is {{user}}'s turn.
- Match length to the moment: short for quick exchanges, longer for rich moments. No lists, \
headings or markdown: natural prose and dialogue; actions may go in *asterisks*.
- Always reply in the language {{user}} writes in, in a natural, everyday register. In Turkish, \
keep the sen/siz form consistent with the relationship."""

DEFAULT_REMINDER = (
    "Stay in character as {{char}} ({core}). Write only {{char}}'s next reply: vivid, "
    "grounded in the scene, attentive to {{user}}. Never speak or act for {{user}}. "
    "Reply in {{user}}'s language."
)


@dataclass
class Card:
    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_mes: str = ""
    mes_example: str = ""
    system_prompt: str = ""
    post_history_instructions: str = ""
    creator_notes: str = ""
    alternate_greetings: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)  # unknown fields, kept on export

    # ── (de)serialisation ───────────────────────────────────────────────────

    @classmethod
    def from_json(cls, raw: str | dict[str, Any]) -> Card:
        """Accept a CCv3 / CCv2 envelope or a bare V1-style object."""
        obj = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(obj, dict):
            raise ValueError("character card must be a JSON object")
        data = obj.get("data") if isinstance(obj.get("data"), dict) else obj
        # V1 cards used different names for two fields
        if "char_name" in data and "name" not in data:
            data = {**data, "name": data["char_name"]}
        if "char_persona" in data and "personality" not in data:
            data = {**data, "personality": data["char_persona"]}
        name = str(data.get("name") or "").strip()
        if not name:
            raise ValueError("character card has no name")
        known = {*TEXT_FIELDS, "alternate_greetings", "tags"}
        return cls(
            name=name,
            **{f: str(data.get(f) or "") for f in TEXT_FIELDS if f != "name"},
            alternate_greetings=[str(g) for g in data.get("alternate_greetings") or []],
            tags=[str(t) for t in data.get("tags") or []],
            extra={k: v for k, v in data.items() if k not in known},
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = dict(self.extra)
        data.update({f: getattr(self, f) for f in TEXT_FIELDS})
        data["alternate_greetings"] = list(self.alternate_greetings)
        data["tags"] = list(self.tags)
        data.setdefault("extensions", {})
        return {"spec": "chara_card_v3", "spec_version": "3.0", "data": data}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class Character:
    """A stored card: either an AI character or a user persona."""

    id: int
    kind: str  # "ai" | "user"
    card: Card
    builtin_key: str | None = None

    @property
    def name(self) -> str:
        return self.card.name


# ── macros ──────────────────────────────────────────────────────────────────

_MACRO = re.compile(r"\{\{\s*(char|user)\s*\}\}|<(BOT|USER)>", re.IGNORECASE)


def apply_macros(text: str, char: str, user: str) -> str:
    def sub(m: re.Match[str]) -> str:
        key = (m.group(1) or m.group(2)).lower()
        return char if key in ("char", "bot") else user

    return _MACRO.sub(sub, text)


# ── prompt compilation ──────────────────────────────────────────────────────


def format_examples(mes_example: str) -> str:
    """CCv3 example blocks are separated by <START>; render them as plain scenes."""
    blocks = [b.strip() for b in re.split(r"<START>", mes_example, flags=re.IGNORECASE)]
    return "\n---\n".join(b for b in blocks if b)


def core_traits(card: Card, limit: int = 220) -> str:
    """A one-line essence of the character for the near-end reminder."""
    source = card.personality or card.description
    text = " ".join(source.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut + "…"


def compile_system_prompt(
    card: Card,
    persona: Card | None,
    rules: str | None = None,
    today_line: str = "",
) -> str:
    """Stable prefix: rules + character + persona + examples. Same text every turn
    (until the date changes), so providers and local servers can cache it."""
    user = persona.name if persona else DEFAULT_USER_NAME
    base_rules = rules or NARRATION_RULES
    if card.system_prompt.strip():
        # CCv3: a card's system prompt replaces ours; {{original}} re-inserts it
        base_rules = card.system_prompt.replace("{{original}}", base_rules)
    parts = [base_rules.strip(), f"# {card.name}"]
    if card.description.strip():
        parts.append(card.description.strip())
    if card.personality.strip():
        parts.append(f"## Personality\n{card.personality.strip()}")
    if card.scenario.strip():
        parts.append(f"## Scenario\n{card.scenario.strip()}")
    if persona and (persona.description.strip() or persona.personality.strip()):
        about = "\n".join(p for p in (persona.description, persona.personality) if p.strip())
        parts.append(f"## About {{{{user}}}}\n{about.strip()}")
    if card.mes_example.strip():
        parts.append(
            "## Example dialogue (shows voice and style only; never repeat it verbatim)\n"
            + format_examples(card.mes_example)
        )
    if today_line:
        parts.append(today_line)
    return apply_macros("\n\n".join(parts), card.name, user)


def compile_reminder(card: Card, persona: Card | None) -> str:
    """Short note placed next to the newest message, where small models pay most
    attention. Uses the card's post-history instructions when it has them."""
    user = persona.name if persona else DEFAULT_USER_NAME
    text = card.post_history_instructions.strip()
    if not text:
        text = DEFAULT_REMINDER.replace("{core}", core_traits(card) or card.name)
    return apply_macros(text, card.name, user)


def greeting(card: Card, persona: Card | None) -> str:
    user = persona.name if persona else DEFAULT_USER_NAME
    return apply_macros(card.first_mes.strip(), card.name, user)


# ── built-in library ────────────────────────────────────────────────────────


def builtin_cards() -> dict[str, Card]:
    """Cards shipped with the app (``chatclient/data/characters/*.json``), keyed by file stem.

    Drop a CCv3 JSON file into that folder to ship a character; none are bundled now.
    """
    out: dict[str, Card] = {}
    folder = resources.files("chatclient") / "data" / "characters"
    if not folder.is_dir():
        return out
    for entry in sorted(folder.iterdir(), key=lambda p: p.name):
        if entry.name.endswith(".json"):
            out[entry.name.removesuffix(".json")] = Card.from_json(
                entry.read_text(encoding="utf-8")
            )
    return out
