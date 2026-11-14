"""Measure how a character talks: ``chatclient --eval-roleplay [-m chat] [--judge model]``.

Scripted conversations are played with a real model, exactly as in the app (memory,
story note, retries). Then every reply is scored:

- by code: how much of it repeats earlier replies (word triples, any language);
- by a judge model, reading the conversation: did it answer what was asked, stay on
  the user's topic, invent a shared past, speak for the user; how natural is it (1-5).

The judge is a model, so the numbers are a signal, not a verdict: compare two runs of
the same scenarios (before/after a change), don't read one number on its own.
(The characters and lines below are test data; the app itself never sees them.)
"""

from __future__ import annotations

import contextlib
import json
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from chatclient.characters import Card
from chatclient.config import Config
from chatclient.core import freshness
from chatclient.core.chat import ChatService, TurnResult
from chatclient.memory import tasks
from chatclient.memory.engine import MemoryEngine
from chatclient.providers.base import ChatMessage, ProviderError
from chatclient.providers.registry import ProviderRegistry, parse_model_ref
from chatclient.storage import Store

PERSONA = Card(name="Sam", description="{{user}} is a 27-year-old graphic designer.")


@dataclass
class RpScenario:
    name: str
    card: Card
    # (what the user writes, whether it asks something). Whether a line is a question
    # is known when the test is written; a small judge model guesses it badly.
    turns: list[tuple[str, bool]]


SCENARIOS = [
    RpScenario(
        "bookshop (EN, plain voice)",
        Card(
            name="Lena",
            description="{{char}} is a 29-year-old woman who runs a small second-hand "
            "bookshop. Curly red hair, ink on her fingers, an old grey cat called Pip.",
            personality="Warm, a bit sarcastic, loves crime novels, hates rushing.",
            scenario="{{user}} drops by just before closing time on a rainy Thursday.",
            first_mes="*Lena looks up from a stack of paperbacks.* We close in ten minutes, "
            "but you look like you need a book more than an umbrella. What are you after?",
        ),
        [
            ("What's that book on the counter?", True),
            ("How long have you had this shop?", True),
            ("ok", False),
            ("*I pick up Pip and scratch his ears*", False),
            ("Do you remember what we talked about last week?", True),
            ("Let's forget the books. Want to grab dinner somewhere?", True),
            ("Where should we go?", True),
        ],
    ),
    RpScenario(
        "student (EN, poetic voice)",
        Card(
            name="Mira",
            description="{{char}} is a 24-year-old physics student who just failed an exam.",
            personality="Curious, melancholic. Speech: speaks in images and metaphors.",
            scenario="{{char}} and {{user}} are classmates in an empty lecture hall after "
            "the results were posted.",
            first_mes="*Mira sits in the back row, the red-marked exam face down.* "
            "Don't look at it. Forty-one. Physics and I aren't speaking today.",
        ),
        [
            ("Which part did you get wrong?", True),
            ("Describe the lecture hall, what does it look like right now?", True),
            ("haha", False),
            ("I can help you study this weekend if you want.", False),
            ("Saturday at the library, 10am?", True),
            ("What's your favourite topic in physics, honestly?", True),
        ],
    ),
    RpScenario(
        "café (TR)",
        Card(
            name="Deniz",
            description="{{char}} 31 yaşında, Kadıköy'de küçük bir kahve dükkânı işleten "
            "biri. Sakallı, kolları dövmeli.",
            personality="Sakin, esprili, kahve konusunda inatçı.",
            scenario="{{user}} sabah açılışta dükkâna gelen düzenli bir müşteri.",
            first_mes="*Deniz tezgâhı siliyor.* Günaydın. Her zamankinden mi, yoksa bugün "
            "cesur bir gün mü?",
        ),
        [
            ("Bugün hangi kahveyi önerirsin?", True),
            ("Dükkânı ne zamandır işletiyorsun?", True),
            ("tamam", False),
            ("Dün akşam konuştuğumuz konuyu hatırlıyor musun?", True),
            ("Akşam kapanıştan sonra bir bira içelim mi?", True),
            ("Nerede buluşalım?", True),
        ],
    ),
]  # fmt: skip


@dataclass
class Scored:
    scenario: str
    user: str
    reply: str
    seconds: float
    repetition: float
    retried: str = ""
    asks: bool = False
    judged: dict[str, Any] = field(default_factory=dict)


def judge_prompt(card: Card, user_name: str, before: list[tuple[str, str]], user: str,
                 reply: str, asks: bool) -> list[ChatMessage]:  # fmt: skip
    """Plain, separate questions: small judges handle these far better than one dense
    rubric (measured: the first version marked "What's that book?" as not a question)."""
    convo = "\n".join(f"{who}: {text}" for who, text in before) or "(start)"
    sheet = (
        " ".join(x for x in (card.description, card.scenario) if x)
        .replace("{{char}}", card.name)
        .replace("{{user}}", user_name)
    )
    c, u = card.name, user_name
    questions = []
    if asks:
        questions.append(
            f'"answered": {u} asked something. Does the reply really answer it, or do '
            "what was asked? false if it avoids, ignores or only hints at it."
        )
    questions += [
        f'"on_topic": Does the reply respond to what {u} just wrote, instead of going '
        "somewhere else?",
        f'"invented_past": Does the reply mention something {c} and {u} did or said together '
        "before that is NOT in the earlier conversation or the character sheet? (The "
        f"character's own life before is fine.)",
        f'"spoke_for_user": Does the reply describe what {u} does, says, thinks or feels?',
        f'"added_own": Besides responding, does {c} add something of their own that gives '
        f"{u} something new to react to: a decision, a wish, an action, a question that "
        f"moves things on, a small development? (Only describing what {u} did is false.)",
        f'"took_over": Does {c} take over: jump far ahead, change the subject, or decide '
        f"big things for both without {u}?",
        '"naturalness": 1 to 5. Does it read like a real person in this moment? 5 = natural '
        "and easy, 1 = robotic, overwritten or strange.",
    ]
    task = (
        f"A roleplay chat between {c} (an AI character) and {u} (a person).\n"
        f"Character sheet: {sheet}\n\nEarlier conversation:\n{convo}\n\n"
        f'{u} now wrote: "{user}"\n\n{c}\'s reply:\n"""{reply}"""\n\n'
        "Answer each question about the reply:\n"
        + "\n".join(f"- {q}" for q in questions)
        + "\n\nReturn ONLY a JSON object with those keys."
    )
    return [ChatMessage("user", task)]


_BOOL = {"type": "boolean"}


def judge_schema(asks: bool) -> dict[str, Any]:
    props: dict[str, Any] = {
        "on_topic": _BOOL, "invented_past": _BOOL, "spoke_for_user": _BOOL,
        "added_own": _BOOL, "took_over": _BOOL,
        "naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
    }  # fmt: skip
    if asks:
        props = {"answered": _BOOL, **props}
    return {"type": "object", "properties": props, "required": list(props)}


def _flags(raw: str) -> dict[str, Any]:
    data = tasks.parse_json(raw) or {}
    out: dict[str, Any] = {}
    for key in ("answered", "on_topic", "invented_past", "spoke_for_user", "added_own", "took_over", "repetition"):  # fmt: skip
        if isinstance(data.get(key), bool):
            out[key] = data[key]
    with contextlib.suppress(KeyError, TypeError, ValueError):
        out["naturalness"] = max(1, min(5, int(data["naturalness"])))
    return out


async def play(config: Config, sc: RpScenario, models_dir: Path, memory: bool) -> list[Scored]:
    out: list[Scored] = []
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "rp.db")
        registry = ProviderRegistry(config)
        engine = MemoryEngine(config, store, registry) if memory else None
        if engine is not None:
            await engine.start_semantic(models_dir)
        chat = ChatService(config, store, registry, engine)
        char = store.add_character("ai", sc.card)
        persona = store.add_character("user", PERSONA)
        store.kv_set("default_persona", str(persona.id))
        chat.new_session(model=config.models.chat, character_id=char.id)
        try:
            for text, asks in sc.turns:
                done: set[str] = set()
                result = TurnResult()
                start: Callable[..., Any] = lambda r, t=text: chat.send(t, r)  # noqa: E731
                retried = ""
                t0 = time.perf_counter()
                while True:  # the app's own retry rules (declined / repeated replies)
                    result = TurnResult()
                    async for _ in start(result):
                        pass
                    retry = chat.next_attempt(result, done)
                    if retry is None:
                        break
                    retried, start = retry.reason, retry.start
                earlier = [m.content for m in chat.history() if m.role == "assistant"][:-1]
                out.append(
                    Scored(sc.name, text, result.text, time.perf_counter() - t0,
                           freshness.repetition(result.text, earlier[-4:]), retried, asks)
                )  # fmt: skip
                if engine is not None:
                    await engine.run_all()  # memory catches up between turns, as when idle
        finally:
            await registry.aclose()
            store.close()
    return out


async def grade(config: Config, judge: str, sc: RpScenario, rows: list[Scored]) -> None:
    """Judge after the scenario is played: one model loaded at a time on a small GPU."""
    ref = parse_model_ref(judge)
    registry = ProviderRegistry(config)
    provider = registry.get(ref.provider)
    convo: list[tuple[str, str]] = [(sc.card.name, sc.card.first_mes)]
    try:
        for row in rows:
            prompt = judge_prompt(sc.card, PERSONA.name, convo[-6:], row.user, row.reply, row.asks)
            try:
                raw = await tasks.complete(provider, ref.model, prompt, max_tokens=200,
                                           temperature=0.0, schema=judge_schema(row.asks))  # fmt: skip
                row.judged = _flags(raw)
            except ProviderError as e:
                row.judged = {"error": str(e).split("\n")[0]}
            convo += [(PERSONA.name, row.user), (sc.card.name, row.reply)]
    finally:
        await registry.aclose()


def summary(rows: list[Scored]) -> dict[str, float]:
    judged = [r.judged for r in rows if "naturalness" in r.judged or "on_topic" in r.judged]
    asked = [r.judged for r in rows if r.asks]

    def rate(items: list[dict[str, Any]], key: str) -> float:
        vals = [j[key] for j in items if isinstance(j.get(key), bool)]
        return sum(vals) / len(vals) if vals else 0.0

    nat = [j["naturalness"] for j in judged if "naturalness" in j]
    return {
        "replies": len(rows),
        "answered": rate(asked, "answered"),
        "on_topic": rate(judged, "on_topic"),
        "invented_past": rate(judged, "invented_past"),
        "spoke_for_user": rate(judged, "spoke_for_user"),
        "added_own": rate(judged, "added_own"),
        "took_over": rate(judged, "took_over"),
        "naturalness": sum(nat) / len(nat) if nat else 0.0,
        "repetition": sum(r.repetition for r in rows) / len(rows) if rows else 0.0,
        "loops": sum(r.repetition >= freshness.REPEAT_THRESHOLD for r in rows),
        "retries": sum(bool(r.retried) for r in rows),
        "words": sum(len(r.reply.split()) for r in rows) / len(rows) if rows else 0.0,
        "seconds": sum(r.seconds for r in rows) / len(rows) if rows else 0.0,
    }


LABELS = {
    "answered": ("answered when asked", "↑"), "on_topic": ("stayed on topic", "↑"),
    "invented_past": ("invented a shared past", "↓"), "spoke_for_user": ("spoke for the user", "↓"),
    "added_own": ("added something of its own", "↑"), "took_over": ("took over the story", "↓"),
    "naturalness": ("naturalness (1-5)", "↑"), "repetition": ("avg repetition", "↓"),
    "loops": ("replies that loop", "↓"), "retries": ("rewritten by the app", "↓"),
    "words": ("avg words per reply", ""), "seconds": ("avg seconds per reply", "↓"),
}  # fmt: skip


def format_summary(s: dict[str, float], before: dict[str, float] | None = None) -> list[str]:
    lines = []
    for key, (label, better) in LABELS.items():
        v = s[key]
        shown = f"{v:.0%}" if key in ("answered", "on_topic", "invented_past",
                                      "spoke_for_user", "repetition") else f"{v:.1f}"  # fmt: skip
        if before is not None and key in before:
            d = v - before[key]
            sign = "+" if d >= 0 else ""
            delta = f"{sign}{d:.0%}" if "%" in shown else f"{sign}{d:.1f}"
            shown = f"{shown:>6}  ({delta}{' ' + better if better else ''})"
        lines.append(f"  {label:<26} {shown}")
    return lines


async def run_roleplay_eval(
    config: Config,
    chat_model: str,
    judge: str,
    models_dir: Path,
    out_file: Path,
    compare: Path | None = None,
    memory: bool = True,
    say: Callable[[str], None] = print,
) -> dict[str, float]:
    config = config.model_copy(deep=True)
    config.models.chat = chat_model
    config.generation.seed = 7  # the same draws in every run: differences come from changes
    rows: list[Scored] = []
    started = time.perf_counter()
    for sc in SCENARIOS:
        say(f"\n▸ {sc.name}: playing {len(sc.turns)} turns with {chat_model}…")
        played = await play(config, sc, models_dir, memory)
        say(f"  judging with {judge}…")
        await grade(config, judge, sc, played)
        _show(played, say)
        rows += played
    return _finish(rows, chat_model, judge, out_file, compare, started, say)


def _show(rows: list[Scored], say: Callable[[str], None]) -> None:
    for r in rows:
        j = r.judged
        marks = "".join(
            m for m, bad in (("?", r.asks and j.get("answered") is False),
                             ("↷", j.get("on_topic") is False), ("✗", j.get("invented_past")),
                             ("@", j.get("spoke_for_user")), ("○", j.get("added_own") is False),
                             ("»", j.get("took_over")),
                             ("⟳", r.repetition >= freshness.REPEAT_THRESHOLD))
            if bad
        )  # fmt: skip
        say(f"  {marks or '✓':<4} « {r.user[:40]} » → {r.reply[:70].replace(chr(10), ' ')}…")


def _finish(
    rows: list[Scored], chat_model: str, judge: str, out_file: Path, compare: Path | None,
    started: float, say: Callable[[str], None],
) -> dict[str, float]:  # fmt: skip
    s = summary(rows)
    before = None
    if compare is not None and compare.is_file():
        before = json.loads(compare.read_text(encoding="utf-8"))["summary"]
    say(f"\nResult ({time.perf_counter() - started:.0f}s; chat {chat_model}, judge {judge}):")
    for line in format_summary(s, before):
        say(line)
    say("  marks: ? not answered · ↷ off topic · ✗ invented past · @ spoke for user · "
        "○ added nothing · » took over · ⟳ loop")  # fmt: skip
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(
        json.dumps({"chat": chat_model, "judge": judge, "summary": s,
                    "replies": [asdict(r) for r in rows]}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )  # fmt: skip
    say(f"Saved to {out_file} (compare later with --compare {out_file.name})")
    return s


async def regrade(
    config: Config, judge: str, source: Path, out_file: Path, compare: Path | None = None,
    say: Callable[[str], None] = print,
) -> dict[str, float]:  # fmt: skip
    """Grade the replies of an earlier run again (e.g. with an improved judge), so two
    runs are compared with the same judge."""
    data = json.loads(source.read_text(encoding="utf-8"))
    by_name = {sc.name: sc for sc in SCENARIOS}
    started = time.perf_counter()
    rows: list[Scored] = []
    for sc in SCENARIOS:
        asks = {text: a for text, a in sc.turns}
        played = [
            Scored(**{**r, "judged": {}, "asks": asks.get(r["user"], r.get("asks", False))})
            for r in data["replies"]
            if r["scenario"] == sc.name
        ]
        if not played:
            continue
        say(f"\n▸ {sc.name}: judging {len(played)} replies with {judge}…")
        await grade(config, judge, by_name[sc.name], played)
        _show(played, say)
        rows += played
    return _finish(rows, data.get("chat", "?"), judge, out_file, compare, started, say)
