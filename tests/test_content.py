"""Mature content setting, private sessions, and declined replies."""

from __future__ import annotations

import pytest
from tests.conftest import ALPHA

from chatclient.config import Config, MemoryConfig, ProviderConfig
from chatclient.core import refusal
from chatclient.core.chat import ChatError, ChatService, TurnResult
from chatclient.memory.engine import MemoryEngine
from chatclient.memory.repo import Scope
from chatclient.providers.base import ProviderError, StreamEvent
from chatclient.providers.echo import EchoProvider
from chatclient.providers.registry import ProviderRegistry


async def say(chat, text):
    r = TurnResult()
    async for _ in chat.send(text, r):
        pass
    return r


# ── content setting ─────────────────────────────────────────────────────────


def test_mature_rules_are_added_only_when_chosen(chat):
    assert chat.content_mode() == "standard"
    assert "Content setting: mature" not in chat.effective_system_prompt()
    chat.set_content_mode("mature")
    prompt = chat.effective_system_prompt()
    assert "Content setting: mature" in prompt and "{{user}}" not in prompt
    assert "Every character in sexual content is an adult" in prompt


def test_mature_applies_to_character_chats_too(chat):
    char = chat.store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=char.id)
    chat.set_content_mode("mature")
    prompt = chat.effective_system_prompt()
    assert prompt.startswith("You are Alpha") and "Content setting: mature" in prompt


def test_config_default_and_bad_mode(config, store):
    config.roleplay.content = "mature"
    chat = ChatService(config, store, ProviderRegistry(config))
    chat.new_session(model="echo:echo")
    assert chat.content_mode() == "mature"
    with pytest.raises(ChatError):
        chat.set_content_mode("anything")


# ── private sessions ────────────────────────────────────────────────────────


@pytest.fixture
def mem(store, monkeypatch):
    seen: list[str] = []

    async def model(self, model, messages, params):
        seen.append(messages[-1].content)
        text = '{"memories": []}' if "Return ONLY a JSON object" in messages[-1].content else "ok"
        yield StreamEvent(text=text)

    monkeypatch.setattr(EchoProvider, "stream_chat", model)
    config = Config(
        memory=MemoryConfig(embeddings="off"),
        providers={"local": ProviderConfig(type="openai_compat", base_url="http://localhost:1/v1")},
    )
    registry = ProviderRegistry(config)
    engine = MemoryEngine(config, store, registry)
    chat = ChatService(config, store, registry, engine)
    char = store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=char.id)
    return chat, engine, seen


async def test_private_session_does_no_memory_work(mem):
    chat, engine, seen = mem
    chat.set_private(True)
    await say(chat, "a secret")
    assert engine.repo.next_job() is None  # nothing queued
    assert not any("memory brief" in s for s in seen)  # text went to no utility model
    assert chat.refresh().private


async def test_going_private_forgets_what_memory_took(mem):
    chat, engine, _seen = mem
    s = chat.require_session()
    scope = engine.scope(s)
    engine.repo.add_memory(scope, "Alpha likes rain.", day="2026-01-01")
    engine.repo.add_summary(
        s.id, "segment", "They talked.", start_message_id=None, end_message_id=None
    )
    chat.set_private(True)
    assert engine.repo.list_memories(scope, 10) == []
    assert engine.repo.summaries(s.id, "segment") == []


async def test_private_sessions_are_not_searched_or_indexed(mem):
    chat, engine, _seen = mem
    secret = chat.require_session()
    await say(chat, "lighthouse secret")
    chat.set_private(True)
    other = chat.new_session(model="echo:echo", character_id=secret.character_id)
    scope = Scope(other.id, secret.character_id)
    assert secret.id not in engine.repo.scope_sessions(scope)
    assert all(sid != secret.id for _t, _id, sid, _x in engine.repo.unindexed("m", 100))


# ── declined replies ────────────────────────────────────────────────────────


def test_provider_stop_reasons():
    assert refusal.stopped_as_refusal("refusal")
    assert refusal.stopped_as_refusal("CONTENT_FILTER")
    assert not refusal.stopped_as_refusal("stop")
    assert refusal.parse_verdict(" Yes.") is True
    assert refusal.parse_verdict("no") is False
    assert refusal.parse_verdict("maybe") is None


def fake_model(monkeypatch, *, finish=None, judge="no"):
    calls: list[tuple[str, str]] = []

    async def model(self, model, messages, params):
        prompt = messages[-1].content
        calls.append((model, prompt))
        if "Answer with one word: yes or no" in prompt:
            yield StreamEvent(text=judge)
        elif model == "other":
            yield StreamEvent(text="*Alpha goes on with the story.*")
        else:
            yield StreamEvent(text="I can't continue this.", finish_reason=finish)

    monkeypatch.setattr(EchoProvider, "stream_chat", model)
    return calls


async def test_a_provider_refusal_is_marked_and_skipped_by_memory(chat, monkeypatch):
    fake_model(monkeypatch, finish="content_filter")
    r = await say(chat, "hello")
    assert r.refused and r.message.meta["refused"] is True
    # memory never reads a declined reply as part of the story
    engine = MemoryEngine(chat.config, chat.store, chat.registry)
    lines = engine._lines(chat.require_session(), chat.history())
    assert [text for *_, text in lines] == ["hello"]


async def test_soft_refusal_is_judged_by_a_model_only_with_a_fallback(chat, monkeypatch):
    calls = fake_model(monkeypatch, judge="yes")
    r = await say(chat, "hello")
    assert not r.refused and len(calls) == 1  # no fallback set: no extra model call
    chat.set_refusal_fallback("echo:other")
    r = await say(chat, "hello again")
    assert r.refused and "Answer with one word" in calls[-1][1]


async def test_long_replies_are_not_checked(chat, monkeypatch):
    calls = fake_model(monkeypatch, judge="yes")
    chat.set_refusal_fallback("echo:other")

    async def long_reply(self, model, messages, params):
        calls.append((model, messages[-1].content))
        yield StreamEvent(text="word " * 400)

    monkeypatch.setattr(EchoProvider, "stream_chat", long_reply)
    r = await say(chat, "hello")
    assert not r.refused and len(calls) == 1


async def test_repl_hands_a_declined_turn_to_the_fallback(chat, monkeypatch, tmp_path):
    from chatclient.i18n import Translator
    from chatclient.paths import resolve_paths
    from chatclient.ui.repl import Repl

    fake_model(monkeypatch, finish="refusal")
    chat.set_refusal_fallback("echo:other")
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths)
    await repl.reply(lambda r: chat.send("hello", r))
    head = chat.last_reply()
    assert head.model == "echo:other" and "goes on" in head.content
    sibs, _ = chat.alternatives()
    assert [m.meta.get("refused", False) for m in sibs] == [True, False]  # kept, see /prev
    assert chat.require_session().model == "echo:echo"  # the session keeps its model


def test_fallback_setting(chat):
    assert chat.refusal_fallback() is None
    chat.set_refusal_fallback("echo:other")
    assert chat.refusal_fallback() == "echo:other"
    chat.set_refusal_fallback(None)
    assert chat.refusal_fallback() is None
    with pytest.raises(ProviderError):
        chat.set_refusal_fallback("nosuchprovider:x")


def test_mature_also_goes_into_the_story_note(chat):
    char = chat.store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=char.id)
    assert "Mature content is on" not in chat.reminder()
    chat.set_content_mode("mature")
    assert "Mature content is on" in chat.reminder()


def test_brief_note_cannot_steer_a_turn_that_is_not_about_the_past(mem):
    from chatclient.memory.engine import Recall
    from chatclient.memory.tasks import Brief, BriefClaim

    chat, engine, _seen = mem
    s = chat.require_session()
    steer = Brief(about_past=False, note="Alpha prefers silence; finish the pizza first.")
    assert "Memory note" not in engine.format_note(s, Recall(brief=steer))
    fix = Brief(
        about_past=True,
        claims=[BriefClaim(claim="we ate taco", status="unsupported")],
        note="Only pizza was eaten.",
    )
    assert "Memory note: Only pizza was eaten." in engine.format_note(s, Recall(brief=fix))


def test_a_character_keeps_its_own_content_setting(chat):
    from chatclient.characters import Card, set_card_content

    card = Card(name="Aras", description="adult")
    set_card_content(card, "mature")
    aras = chat.store.add_character("ai", card)
    plain = chat.store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=aras.id)
    assert chat.content_mode() == "mature"  # from the card, default is standard
    chat.new_session(model="echo:echo", character_id=plain.id)
    assert chat.content_mode() == "standard"
    assert chat.set_content_mode("mature") == "character"  # /content in a char chat
    assert chat.content_mode() == "mature"
    chat.new_session(model="echo:echo")
    assert chat.content_mode() == "standard"  # the default did not change
