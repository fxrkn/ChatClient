"""The brief: the model understands the message (no rules in the code).

The fake utility model below plays the part of a decent model, deciding from the
candidates it is shown, so these tests check the plumbing around it: what the model
sees, how its answer turns into the prompt, prefetching while typing, fallbacks."""

import asyncio
import json
import re

import pytest

from chatclient.characters import Card
from chatclient.config import Config, MemoryConfig, ProviderConfig
from chatclient.core.chat import ChatService, TurnResult
from chatclient.memory.engine import MemoryEngine
from chatclient.providers.base import StreamEvent
from chatclient.providers.echo import EchoProvider
from chatclient.providers.registry import ProviderRegistry


def judge(prompt: str) -> dict:
    """A stand-in "model": a claim is supported if its key word appears in the known
    history shown in the prompt (recent conversation + candidates)."""
    message = re.search(r'new message[^\n]*:\n"(.*)"', prompt).group(1).lower()
    recent = prompt.split("new message")[0]
    candidates = prompt.split("found by search.")[-1].split("Return ONLY")[0]
    known = (recent + candidates).lower()
    claims = []
    for food in ("taco", "pizza", "sushi"):
        if food in message:
            claims.append(
                {
                    "claim": f"we ate {food}",
                    "status": "supported" if food in known else "unsupported",
                }
            )
    ids = re.findall(r"\[([MQD]\d+)\]", prompt)
    return {"about_past": bool(claims), "relevant": ids, "claims": claims,
            "note": "We only had pizza." if any(c["status"] == "unsupported" for c in claims) else ""}  # fmt: skip


@pytest.fixture
def env(store, monkeypatch):
    calls: list[str] = []
    briefs: list[str] = []

    async def model(self, model, messages, params):
        prompt = messages[-1].content
        calls.append("\n".join(m.content for m in messages))
        if "memory brief" in prompt:
            briefs.append(prompt)
            await asyncio.sleep(0)
            text = json.dumps(judge(prompt))
        elif "Return ONLY a JSON object" in prompt:
            text = '{"memories": []}'
        else:
            text = "ok"
        yield StreamEvent(text=text)

    monkeypatch.setattr(EchoProvider, "stream_chat", model)
    config = Config(
        memory=MemoryConfig(),
        providers={"local": ProviderConfig(type="openai_compat", base_url="http://localhost:1/v1")},
    )
    registry = ProviderRegistry(config)
    engine = MemoryEngine(config, store, registry)
    chat = ChatService(config, store, registry, engine)
    char = store.add_character("ai", Card(name="Mert"))
    chat.new_session(model="echo:echo", character_id=char.id)
    return chat, engine, calls, briefs


async def say(chat, text):
    r = TurnResult()
    async for _ in chat.send(text, r):
        pass
    return r


async def test_false_premise_is_flagged_by_the_model_not_by_rules(env):
    chat, _engine, calls, briefs = env
    await say(chat, "Let's forget the past, we have a pizza to eat.")
    await say(chat, "did we eat a taco right?")
    assert "pizza" in briefs[-1]  # the recent conversation is shown to the model
    prompt = calls[-1]
    assert 'Fact check: "we ate taco" is not in your shared history' in prompt
    assert "Memory note: We only had pizza." in prompt
    assert chat.last_user_message().content == "did we eat a taco right?"  # stored clean


async def test_true_premise_gets_no_fact_check(env):
    chat, _engine, calls, _briefs = env
    await say(chat, "Let's forget the past, we have a pizza to eat.")
    await say(chat, "did we eat a pizza right?")
    assert "Fact check" not in calls[-1]


async def test_it_works_the_same_in_any_language(env):
    chat, _engine, calls, _briefs = env
    await say(chat, "Akşam pizza yedik, çok güzeldi.")
    await say(chat, "dün sushi yedik mi?")
    assert 'Fact check: "we ate sushi"' in calls[-1]


async def test_prefetched_brief_is_reused_on_enter(env):
    chat, engine, _calls, briefs = env
    await say(chat, "Akşam pizza yedik.")
    before = len(briefs)
    await chat.prefetch_memory("  taco yedik mi?  ")  # while typing
    assert len(briefs) == before + 1
    assert engine.prefetched_for(chat.session.id, "taco yedik mi?") is not None
    await say(chat, "taco yedik mi?")
    assert len(briefs) == before + 1  # no second brief: Enter didn't wait for the model
    assert engine.prefetched_for(chat.session.id, "taco yedik mi?") is None  # used up


async def test_a_changed_draft_gets_a_fresh_brief(env):
    chat, _engine, _calls, briefs = env
    await chat.prefetch_memory("taco yedik mi")
    before = len(briefs)
    await say(chat, "taco yedik mi, yoksa sushi mi?")
    assert len(briefs) == before + 1


async def test_brief_off_falls_back_to_search(env):
    chat, engine, calls, briefs = env
    engine.cfg.brief = False
    await say(chat, "Akşam pizza yedik.")
    await say(chat, "taco yedik mi?")
    assert briefs == [] and "Fact check" not in calls[-1]


async def test_unreadable_or_slow_brief_never_blocks_the_reply(env, monkeypatch):
    chat, engine, calls, _briefs = env
    engine.cfg.brief_timeout = 0.05

    async def slow(self, model, messages, params):
        if "memory brief" in messages[-1].content:
            await asyncio.sleep(1)
        calls.append("reply")
        yield StreamEvent(text="ok")

    monkeypatch.setattr(EchoProvider, "stream_chat", slow)
    r = await say(chat, "taco yedik mi?")
    assert r.text == "ok"


def test_word_start_matching_is_generic():
    from chatclient.memory.repo import shares_a_word_start

    assert shares_a_word_start("haritayı nerede bulduk", "Bir harita bulundu.")
    assert shares_a_word_start("kedimin adı", "Ali'nin kedisinin adı Pamuk.")
    assert shares_a_word_start("the lighthouses", "We climbed the lighthouse.")
    assert not shares_a_word_start("hatırlıyor musun", "İstanbul'da çalışıyor.")  # ending only


async def test_typing_prepares_the_brief_before_enter(env, tmp_path):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from chatclient.i18n import Translator
    from chatclient.paths import resolve_paths
    from chatclient.ui.repl import Repl

    chat, _engine, calls, briefs = env
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text("did we eat a taco right?")  # typed…
        task = asyncio.ensure_future(repl.run([]))
        await asyncio.sleep(1.2)  # …a pause: the brief is prepared now
        assert len(briefs) == 1 and repl.prefetch_text == "did we eat a taco right?"
        inp.send_text("\r/quit\r")  # Enter
        await task
    assert len(briefs) == 1  # reused, not asked again
    assert "Fact check" in next(c for c in reversed(calls) if "did we eat a taco" in c)


async def test_slash_commands_are_not_prefetched(env):
    chat, _engine, _calls, briefs = env
    await chat.prefetch_memory("/memory status")
    assert briefs == []
