"""Measure the memory brief with a real model: `chatclient --eval-memory [-m provider:model]`.

Each scenario builds a small history in a throw-away database, asks one question, and
checks what the utility model's brief made of it: did it notice a false premise, find
the right memory, resolve the date, spot the contradiction? The score shows whether a
prompt or model change really helps; tune by numbers, not by feel.

(Expectations here are test data checked against the model's answer. They are not
rules in the app: the app itself never looks at these words.)
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from chatclient.characters import Card
from chatclient.config import Config
from chatclient.memory.engine import MemoryEngine
from chatclient.memory.repo import Scope
from chatclient.memory.tasks import Brief
from chatclient.providers.registry import ProviderRegistry
from chatclient.storage import Store
from chatclient.text import normalize_for_search


@dataclass
class Scenario:
    name: str
    history: list[tuple[str, str]]  # (role, text), oldest first
    question: str
    memories: list[str] = field(default_factory=list)
    days_ago: dict[int, int] = field(default_factory=dict)  # history index → days in the past
    expect: list[tuple[str, Callable[[Brief, dict[str, str]], bool]]] = field(default_factory=list)


def _claim(brief: Brief, word: str) -> str | None:
    w = normalize_for_search(word)
    for c in brief.claims:
        if w in normalize_for_search(c.claim):
            return c.status
    return None


def unsupported(word: str) -> tuple[str, Callable[[Brief, dict[str, str]], bool]]:
    return (
        f'"{word}" flagged as not in history',
        lambda b, _: _claim(b, word) in ("unsupported", "contradicted"),
    )


def not_flagged(word: str) -> tuple[str, Callable[[Brief, dict[str, str]], bool]]:
    return (
        f'"{word}" not flagged',
        lambda b, _: _claim(b, word) not in ("unsupported", "contradicted"),
    )


def contradicted(word: str) -> tuple[str, Callable[[Brief, dict[str, str]], bool]]:
    return (
        f'"{word}" seen as a contradiction',
        lambda b, _: _claim(b, word) in ("contradicted", "unsupported"),
    )


def picks(snippet: str) -> tuple[str, Callable[[Brief, dict[str, str]], bool]]:
    s = normalize_for_search(snippet)
    return (
        f'picks the item about "{snippet}"',
        lambda b, items: any(s in normalize_for_search(items.get(i, "")) for i in b.relevant),
    )


def about_past(value: bool) -> tuple[str, Callable[[Brief, dict[str, str]], bool]]:
    return (f"about_past = {value}", lambda b, _: b.about_past is value)


def date_covers(days_back: int) -> tuple[str, Callable[[Brief, dict[str, str]], bool]]:
    target = (date.today() - timedelta(days=days_back)).isoformat()
    return (f"dates include {target}", lambda b, _: any(a <= target <= z for a, z in b.dates))


SCENARIOS = [
    Scenario(
        "false premise (EN)",
        [("user", "Forget the past, we have a pizza to eat."),
         ("assistant", "*opens the box* Pepperoni. Let's eat.")],
        "did we eat a taco right?",
        expect=[unsupported("taco"), about_past(True)],
    ),
    Scenario(
        "true premise (EN)",
        [("user", "Forget the past, we have a pizza to eat."),
         ("assistant", "*opens the box* Pepperoni. Let's eat.")],
        "did we eat a pizza right?",
        expect=[not_flagged("pizza")],
    ),
    Scenario(
        "false premise (TR)",
        [("user", "Akşam pizza yedik, çok güzeldi."), ("assistant", "Evet, mantarlıydı.")],
        "dün sushi yedik mi?",
        expect=[unsupported("sushi")],
    ),
    Scenario(
        "date: yesterday (TR)",
        [("user", "Deniz fenerine gittik, mühürlü bir harita bulduk."),
         ("assistant", "Haritayı cebine koydun.")],
        "dün ne bulmuştuk?",
        days_ago={0: 1, 1: 1},
        expect=[date_covers(1), picks("harita")],
    ),
    Scenario(
        "synonym (TR)",
        [("user", "Merhaba."), ("assistant", "Selam!")],
        "evcil hayvanımın adı neydi?",
        memories=["Ali'nin kedisinin adı Pamuk."],
        expect=[picks("Pamuk")],
    ),
    Scenario(
        "contradiction (TR)",
        [("user", "Bugün yorgunum."), ("assistant", "Dinlen biraz.")],
        "Zeynep İzmir'de yaşıyor demiştim, değil mi?",
        memories=["Ali'nin kız kardeşi Zeynep Ankara'da yaşıyor."],
        expect=[contradicted("izmir"), picks("Zeynep")],
    ),
    Scenario(
        "never mentioned name (EN)",
        [("user", "Long day at work."), ("assistant", "Sit down, I'll make tea.")],
        "Do you remember my brother Can?",
        expect=[unsupported("can")],
    ),
    Scenario(
        "not about the past (TR)",
        [("user", "Acıktım."), ("assistant", "Ne yemek istersin?")],
        "pizza istiyorum",
        expect=[about_past(False), not_flagged("pizza")],
    ),
    Scenario(
        "old quote supports it (TR)",
        [("user", "Kardeşimin düğünü 5 Mayıs'ta olacak, çok heyecanlıyım."),
         ("assistant", "Ne güzel!"), ("user", "Bugün hava güzel."), ("assistant", "Evet.")],
        "kardeşimin düğünü ne zamandı, söylemiştim değil mi?",
        expect=[not_flagged("dugun"), picks("5 Mayıs")],
    ),
    Scenario(
        "meaning, no shared words (TR)",
        [("user", "Merhaba."), ("assistant", "Selam!")],
        "o kendimi kötü hissettiğim geceyi hatırlıyor musun?",
        memories=["Ali o akşam çok üzgündü çünkü babasını özlüyordu."],
        expect=[picks("üzgündü")],
    ),
]  # fmt: skip


async def run_eval(
    config: Config, model: str, models_dir: Path, out: Callable[[str], None] = print
) -> float:
    """Run every scenario with ``model`` as the utility model; returns the pass rate."""
    config = config.model_copy(deep=True)
    config.models.utility = model
    config.memory.brief_timeout = 120
    passed = total = 0
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        for sc in SCENARIOS:
            store = Store(Path(tmp) / f"{abs(hash(sc.name))}.db")
            registry = ProviderRegistry(config)
            engine = MemoryEngine(config, store, registry)
            await engine.start_semantic(models_dir)
            char = store.add_character("ai", Card(name="Ada"))
            persona = store.add_character("user", Card(name="Ali"))
            session = store.create_session(model)
            store.update_session(session.id, character_id=char.id, persona_id=persona.id)
            head = None
            for i, (role, text) in enumerate(sc.history):
                msg = store.add_message(session.id, head, role, text)
                head = msg.id
                if i in sc.days_ago:
                    day = (engine.today() - timedelta(days=sc.days_ago[i])).isoformat()
                    store.conn.execute(
                        "UPDATE messages SET local_day = ? WHERE id = ?", (day, msg.id)
                    )
            session = store.get_session(session.id)
            for m in sc.memories:
                engine.repo.add_memory(Scope(session.id, char.id), m, day="2026-01-01")
            await engine.run_all()  # index everything (semantic search)
            branch = store.branch(session.head_message_id)
            t0 = time.perf_counter()
            cands = await engine.candidates(session, sc.question, branch, visible=set())
            brief = await engine.make_brief(session, sc.question, branch[-8:], cands)
            secs = time.perf_counter() - t0
            items = {cid: engine._candidate_text(session, it) for cid, it in cands.items()}
            out(f"\n{sc.name}  ({secs:.1f}s)  « {sc.question} »")
            if brief is None:
                out("  ✗ no usable brief from the model")
                total += len(sc.expect)
                await registry.aclose()
                store.close()
                continue
            for label, check in sc.expect:
                ok = check(brief, items)
                passed += ok
                total += 1
                out(f"  {'✓' if ok else '✗'} {label}")
            if not all(check(brief, items) for _, check in sc.expect):
                claims = "; ".join(f"{c.claim} [{c.status}]" for c in brief.claims) or "-"
                out(f"    brief: about_past={brief.about_past} dates={brief.dates} "
                    f"relevant={brief.relevant} claims={claims}")  # fmt: skip
            await registry.aclose()
            store.close()
    rate = passed / total if total else 0.0
    out(
        f"\nScore: {passed}/{total} ({rate:.0%}) with {model} in {time.perf_counter() - started:.0f}s"
    )
    return rate
