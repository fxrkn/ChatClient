"""Prompts and parsing for background memory work, done by the "utility" model.

Each task is one short, focused request, so small local models handle them well.
Output parsing is forgiving: code fences, chatter around the JSON, trailing commas,
importance as words or numbers… whatever small models produce, we try to use it.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field, field_validator

from chatclient.providers.base import ChatMessage, GenParams, Provider, ThinkFilter

log = logging.getLogger(__name__)

SCENE_KEYS = ("location", "time", "mood", "activity", "relationship", "threads")


async def complete(
    provider: Provider,
    model: str,
    messages: Sequence[ChatMessage],
    *,
    max_tokens: int = 700,
    temperature: float = 0.2,
    schema: dict[str, Any] | None = None,
) -> str:
    # No hidden reasoning: with a fixed budget a reasoning model would think it all
    # away and return nothing (seen with DeepSeek V4 Pro on OpenRouter).
    params = GenParams(
        temperature=temperature, max_tokens=max_tokens, json_schema=schema, reasoning=False
    )
    think = ThinkFilter()
    parts: list[str] = []
    async for ev in provider.stream_chat(model, list(messages), params):
        if ev.text:
            parts.append(think.feed(ev.text))
    parts.append(think.flush())
    return "".join(parts).strip()


def parse_json(text: str) -> dict[str, Any] | None:
    """The first JSON object in ``text``, tolerating fences and trailing commas."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    blob = text[start : end + 1]
    for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else None
        except ValueError:
            continue
    return None


# ── output shapes ───────────────────────────────────────────────────────────
# Given to servers that can enforce them, small models stop writing broken JSON: the
# shape is fixed, the content is still entirely the model's. The parsers below stay
# lenient for servers that can't.

_STR = {"type": "string"}

PASS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "content": _STR,
                    "kind": {"type": "string", "enum": ["event", "fact", "preference",
                                                        "relationship", "plan", "place"]},
                    "importance": {"type": "integer", "minimum": 1, "maximum": 5},
                    "people": {"type": "array", "items": _STR},
                    "place": _STR,
                    "story_time": _STR,
                    "said_by": _STR,
                    "fact": {"type": "object", "properties": {"key": _STR, "value": _STR},
                             "required": ["key", "value"]},
                },
                "required": ["content", "kind", "importance"],
            },
        },
        "scene": {
            "type": "object",
            "properties": {
                "location": _STR, "time": _STR, "mood": _STR, "activity": _STR,
                "relationship": _STR, "threads": {"type": "array", "items": _STR},
            },
        },
    },
    "required": ["memories", "scene"],
}  # fmt: skip

BRIEF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "about_past": {"type": "boolean"},
        "dates": {
            "type": "array",
            "items": {"type": "object", "properties": {"from": _STR, "to": _STR},
                      "required": ["from", "to"]},
        },
        "relevant": {"type": "array", "items": _STR},
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": _STR,
                    "status": {"type": "string",
                               "enum": ["supported", "contradicted", "unsupported"]},
                    "evidence": {"type": "array", "items": _STR},
                },
                "required": ["claim", "status"],
            },
        },
        "note": _STR,
        "direction": _STR,
    },
    "required": ["about_past", "dates", "relevant", "claims", "note", "direction"],
}  # fmt: skip


CONTRADICTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"contradict": {"type": "boolean"}},
    "required": ["contradict"],
}


def contradiction_prompt(old: str, new: str) -> list[ChatMessage]:
    task = (
        "Two notes from the memory of an ongoing story:\n"
        f'A (earlier): "{old}"\nB (later): "{new}"\n\n'
        "Do they contradict each other, so that both cannot be true? Two descriptions of "
        "the same moment, or a detail added later, are NOT a contradiction; a fact that "
        'changed or was stated differently is (e.g. "lives in Ankara" vs "lives in Izmir").\n'
        'Return ONLY JSON: {"contradict": true or false}'
    )
    return [ChatMessage("system", SYSTEM), ChatMessage("user", task)]


def parse_contradiction(text: str) -> bool | None:
    data = parse_json(text)
    value = data.get("contradict") if data else None
    return value if isinstance(value, bool) else None


class MemoryItem(BaseModel):
    content: str
    kind: str = "fact"
    importance: float = 0.5
    people: list[str] = Field(default_factory=list)
    place: str | None = None
    story_time: str | None = None
    # a checkable fact: subject's attribute is value ("Ali's cat" / "name" / "Pamuk")
    subject: str | None = None
    attribute: str | None = None
    value: str | None = None
    key: str | None = None  # stable fact key chosen by the model ("Ali.cat.name")
    said_by: str | None = None  # a name, "user" or "char"; mapped by the engine

    @field_validator("importance", mode="before")
    @classmethod
    def _importance(cls, v: Any) -> float:
        words = {"low": 0.3, "medium": 0.5, "high": 0.8, "critical": 1.0}
        if isinstance(v, str):
            v = words.get(v.strip().lower(), v)
        try:
            n = float(v)
        except (TypeError, ValueError):
            return 0.5
        return n / 5 if n > 1 else n  # the prompt asks for 1–5

    @field_validator("people", mode="before")
    @classmethod
    def _people(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = [p.strip() for p in v.split(",")]
        return [str(p).strip() for p in v or [] if str(p).strip()]

    @field_validator(
        "place", "story_time", "subject", "attribute", "value", "said_by", mode="before"
    )
    @classmethod
    def _optional(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return None if s.lower() in ("", "null", "none", "n/a", "unknown", "-") else s


class PassResult(BaseModel):
    memories: list[MemoryItem] = Field(default_factory=list)
    scene: dict[str, Any] = Field(default_factory=dict)


def parse_pass(text: str) -> PassResult | None:
    data = parse_json(text)
    if data is None:
        return None
    items = []
    for raw in data.get("memories") or []:
        if isinstance(raw, str):
            raw = {"content": raw}
        if isinstance(raw, dict) and isinstance(raw.get("fact"), dict):
            # {"fact": {"key": …, "value": …}} → flat fields
            fact = raw["fact"]
            raw = {**raw, **{k: fact.get(k) for k in ("key", "subject", "attribute", "value")}}
        if isinstance(raw, dict) and str(raw.get("content", "")).strip():
            try:
                items.append(MemoryItem.model_validate(raw))
            except ValueError as e:
                log.debug("skipping malformed memory %r: %s", raw, e)
    scene_raw = data.get("scene") if isinstance(data.get("scene"), dict) else {}
    scene: dict[str, Any] = {}
    for key in SCENE_KEYS:
        val = scene_raw.get(key)
        if key == "threads":
            if isinstance(val, str):
                val = [val]
            vals = [str(x).strip() for x in val or [] if str(x).strip()]
            if vals:
                scene[key] = vals[:5]
        elif (
            val is not None
            and str(val).strip()
            and str(val).strip().lower() not in ("null", "none", "unknown")
        ):
            scene[key] = str(val).strip()
    return PassResult(memories=items[:8], scene=scene)


# ── prompts ─────────────────────────────────────────────────────────────────

SYSTEM = (
    "You maintain the long-term memory of an ongoing chat or roleplay. "
    "Be precise and brief. Write in the same language as the conversation. "
    "Use names, never pronouns or 'the user'."
)


def transcript(lines: Sequence[tuple[str, str, str]]) -> str:
    """(day, speaker, text) → '[2026-11-12] Elif: …' lines."""
    return "\n".join(f"[{day}] {who}: {text.strip()}" for day, who, text in lines)


def memory_pass_prompt(
    char: str,
    user: str,
    scene: dict[str, Any],
    known: Sequence[str],
    lines: Sequence[tuple[str, str, str]],
    fact_keys: Sequence[tuple[str, str]] = (),
) -> list[ChatMessage]:
    known_txt = "\n".join(f"- {k}" for k in known) or "(none yet)"
    keys_txt = "\n".join(f"- {k} = {v}" for k, v in fact_keys) or "(none yet)"
    scene_txt = json.dumps(scene, ensure_ascii=False) if scene else "(unknown)"
    task = f"""Characters: {char} (the AI) and {user}.

Already remembered (do NOT repeat these):
{known_txt}

Known fact keys (key = current value):
{keys_txt}

Current scene: {scene_txt}

New conversation:
{transcript(lines)}

Return ONLY a JSON object:
{{"memories": [{{"content": "one self-contained sentence", "kind": "event|fact|preference|relationship|plan|place",
   "importance": 1-5, "people": ["names involved"], "place": "place name or null",
   "story_time": "time inside the story or null", "said_by": "{user} or {char}",
   "fact": {{"key": "stable key", "value": "its value"}} or null}}],
  "scene": {{"location": "...", "time": "...", "mood": "{char}'s mood", "activity": "what is happening",
   "relationship": "how {char} and {user} relate now", "threads": ["open questions or plans"]}}}}

Rules: at most 6 memories; only lasting, new information worth remembering weeks later
(facts about {user}, promises, events, discoveries, feelings, names, places); skip small talk.
Record what happened or is true, in plain words. Never record how something was described:
no imagery, metaphors or style ("described X as a storm" is not a memory). If {char}
brings up a shared past event that is neither in this conversation nor already remembered,
it was probably made up: skip it.
Write each memory so it makes sense on its own.
For a fact that has one value at a time, fill "fact" with a short key in English such as
"{user}.cat.name", "{user}.job", "{user}.sister.city". If the fact is already in the known
fact keys, reuse that exact key, even when the new value is different (a changed or
contradicting value must be recorded under the same key).
The scene describes the situation right now, in full: "threads" holds only plans or questions
still open now; leave out what is finished or no longer being talked about."""
    return [ChatMessage("system", SYSTEM), ChatMessage("user", task)]


# ── the per-turn brief: understanding the message (model), not rules (code) ──


class BriefClaim(BaseModel):
    claim: str
    status: str = "unsupported"  # supported | contradicted | unsupported
    evidence: list[str] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def _status(cls, v: Any) -> str:
        s = str(v or "").strip().lower()
        return s if s in ("supported", "contradicted", "unsupported") else "unsupported"

    @field_validator("evidence", mode="before")
    @classmethod
    def _evidence(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = re.findall(r"[MQD]\d+", v)
        return [str(x).strip().strip("[]") for x in v or [] if str(x).strip()]


class Brief(BaseModel):
    about_past: bool = False
    dates: list[tuple[str, str]] = Field(default_factory=list)  # (from, to) ISO days
    relevant: list[str] = Field(default_factory=list)  # candidate ids: M12, Q34, D3
    claims: list[BriefClaim] = Field(default_factory=list)
    note: str = ""
    direction: str = ""  # private prose for the character's next reply (not memory)

    @field_validator("relevant", mode="before")
    @classmethod
    def _ids(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            v = re.findall(r"[MQD]\d+", v)
        return [str(x).strip().strip("[]") for x in v or [] if str(x).strip()]

    @field_validator("dates", mode="before")
    @classmethod
    def _dates(cls, v: Any) -> list[tuple[str, str]]:
        out = []
        for d in v or []:
            if isinstance(d, str):
                d = {"from": d, "to": d}
            if isinstance(d, dict):
                a, b = str(d.get("from") or ""), str(d.get("to") or d.get("from") or "")
                if re.fullmatch(r"\d{4}-\d{2}-\d{2}", a) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", b):
                    out.append((a, b) if a <= b else (b, a))
        return out[:3]

    @field_validator("note", "direction", mode="before")
    @classmethod
    def _note(cls, v: Any) -> str:
        return "" if v is None else str(v).strip()


def parse_brief(text: str) -> Brief | None:
    data = parse_json(text)
    if data is None:
        return None
    claims = [c if isinstance(c, dict) else {"claim": str(c)} for c in data.get("claims") or []]
    data["claims"] = [c for c in claims if str(c.get("claim", "")).strip()]
    try:
        return Brief.model_validate(data)
    except ValueError as e:
        log.debug("unusable brief %r: %s", text[:300], e)
        return None


def brief_prompt(
    char: str,
    user: str,
    today: str,
    recent: Sequence[tuple[str, str, str]],
    message: str,
    candidates: Sequence[tuple[str, str]],
    profile: str = "",
    initiative: str = "",
    scene: str = "",
    content: str = "",
) -> list[ChatMessage]:
    """``profile`` (who the character is), ``initiative`` (a hint), ``scene`` and
    ``content`` let the brief also write ``direction``; without a profile it is left empty."""
    cands = "\n".join(f"[{cid}] {text}" for cid, text in candidates) or "(none found)"
    if profile:
        about = f"\nAbout {char} (for the direction):\n{profile}\n"
        if initiative:
            about += f"Initiative hint (weigh it, it is not a rule): {initiative}\n"
        if scene:
            about += f"Current scene: {scene}\n"
        if content:
            about += f"Content setting: {content}\n"
        direction_rule = (
            f"- direction: a private note, 2-4 sentences of plain prose about THIS moment, for "
            f"{char}'s next reply. Say what is happening now; what {user} seems to want right "
            f"now (quick banter, an answer, a scene); how {char}, being who they are and with "
            f"their own wants, would naturally respond (they may disagree, refuse, change the "
            f"subject or not care); roughly how long the reply should be; and the one or two "
            f"details from the story that matter now and must not be lost. Never write {char}'s "
            f"reply or {user}'s words, and no stock phrases."
        )
    else:
        about = ""
        direction_rule = "- direction: leave empty."
    task = f"""You prepare a memory brief for {char}'s next reply in a chat with {user}.
{about}Calendar (newest first):
{today}

Recent conversation:
{transcript(recent) or "(just started)"}

{user}'s new message, written today (the first calendar line):
"{message}"

Possibly related memories (M), past quotes (Q) and day summaries (D) found by search.
They may be irrelevant; together with the recent conversation they are ALL that is known
about the shared past:
{cands}

Return ONLY a JSON object:
{{"about_past": true or false,
  "dates": [{{"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}}],
  "relevant": ["M12", "Q34"],
  "claims": [{{"claim": "...", "status": "supported|contradicted|unsupported", "evidence": ["Q34"]}}],
  "note": "one short correction about the shared past, or empty",
  "direction": "private note for the next reply, or empty"}}

- about_past: does the message ask about or take for granted something from the shared past?
  (true even if that thing never actually happened)
- dates: days or periods the message refers to ("yesterday", "last week", "12 October"),
  counted from today, the first calendar line (yesterday = the second line); [] if none.
- relevant: only items that really help {char} answer this message.
- claims: what the message states or presupposes about the shared past or known facts.
  supported = backed by an item or the recent conversation; contradicted = an item says
  otherwise; unsupported = nothing known backs it (then it probably did not happen).
- note: only when the message gets the shared past wrong: one short fact that sets it
  straight, e.g. "That trip never happened; say so gently." Never about what {char}
  wants, prefers or should do next, and never steer the story: memory only. Usually empty.
{direction_rule}"""
    return [ChatMessage("system", SYSTEM), ChatMessage("user", task)]


def date_prompt(today: str, expression: str) -> list[ChatMessage]:
    task = f"""Calendar (newest first):
{today}

Which calendar days does this refer to: "{expression}"?
Return ONLY JSON: {{"dates": [{{"from": "YYYY-MM-DD", "to": "YYYY-MM-DD"}}]}}"""
    return [ChatMessage("system", SYSTEM), ChatMessage("user", task)]


def segment_prompt(
    char: str, user: str, lines: Sequence[tuple[str, str, str]]
) -> list[ChatMessage]:
    task = f"""Summarise this part of the conversation between {char} and {user} in 3-6 sentences:
what happened, where, who was involved, what changed between them, and any open threads.
Past tense. Output only the summary.

{transcript(lines)}"""
    return [ChatMessage("system", SYSTEM), ChatMessage("user", task)]


def synopsis_prompt(
    char: str, user: str, previous: str, segments: Sequence[str]
) -> list[ChatMessage]:
    parts = "\n\n".join(segments)
    task = f"""Write the "story so far" between {char} and {user} in at most 200 words,
merging the previous synopsis with the newer parts. Keep important names, places,
promises and how the relationship developed. Output only the synopsis.

Previous synopsis:
{previous or "(none)"}

Newer parts:
{parts}"""
    return [ChatMessage("system", SYSTEM), ChatMessage("user", task)]


def reflect_prompt(char: str, user: str, memories: Sequence[str]) -> list[ChatMessage]:
    listed = "\n".join(f"- {m}" for m in memories)
    task = f"""From these memories, what has {char} learned about {user} and about their
relationship? Give 1-3 short insights that go beyond the single facts (patterns, feelings,
what {user} cares about, how {user} reacts to things). One per line, no numbering, no preface.

{listed}"""
    return [ChatMessage("system", SYSTEM), ChatMessage("user", task)]


def day_prompt(
    char: str, user: str, day: str, lines: Sequence[tuple[str, str, str]]
) -> list[ChatMessage]:
    task = f"""Summarise what {char} and {user} talked about and did on {day}, in 3-6 sentences.
Mention concrete things (places, people, decisions, feelings). Output only the summary.

{transcript(lines)}"""
    return [ChatMessage("system", SYSTEM), ChatMessage("user", task)]
