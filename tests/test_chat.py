import asyncio

import pytest

from chatclient.config import Config
from chatclient.core.chat import ChatError, TurnResult
from chatclient.core.context import build_context
from chatclient.providers.base import ProviderError, StreamEvent
from chatclient.providers.echo import EchoProvider
from chatclient.storage import Message


async def drain(agen):
    return [ev async for ev in agen]


async def test_send_persists_both_sides(chat):
    r = TurnResult()
    await drain(chat.send("merhaba dünya", r))
    msgs = chat.history()
    assert [m.role for m in msgs] == ["user", "assistant"]
    assert "merhaba dünya" in msgs[1].content
    assert r.message is not None and r.message.id == msgs[1].id
    assert chat.session.title == "merhaba dünya"
    assert r.usage and r.usage.completion_tokens


async def test_regenerate_creates_sibling(chat):
    await drain(chat.send("bir", TurnResult()))
    first_reply = chat.history()[-1]
    await drain(chat.regenerate(TurnResult()))
    msgs = chat.history()
    assert len(msgs) == 2
    assert msgs[-1].id != first_reply.id
    assert msgs[-1].parent_id == first_reply.parent_id


async def test_undo_removes_last_exchange(chat):
    await drain(chat.send("bir", TurnResult()))
    await drain(chat.send("iki", TurnResult()))
    assert chat.undo() == 2
    assert [m.content for m in chat.history() if m.role == "user"] == ["bir"]
    assert chat.undo() == 2
    assert chat.history() == []
    with pytest.raises(ChatError):
        chat.undo()


async def test_cancel_keeps_partial_reply(chat, monkeypatch):
    monkeypatch.setattr(EchoProvider, "delay", 0.05)
    r = TurnResult()

    async def consume():
        async for _ in chat.send("bu uzun bir mesaj olacak ve iptal edilecek", r):
            pass

    task = asyncio.create_task(consume())
    while not r.text:
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert r.cancelled
    last = chat.history()[-1]
    assert last.role == "assistant" and last.meta.get("truncated") is True


async def test_provider_error_leaves_user_message_for_retry(chat, monkeypatch):
    async def boom(self, model, messages, params):
        raise ProviderError("down")
        yield  # pragma: no cover

    monkeypatch.setattr(EchoProvider, "stream_chat", boom)
    with pytest.raises(ProviderError):
        await drain(chat.send("merhaba", TurnResult()))
    assert [m.role for m in chat.history()] == ["user"]
    monkeypatch.undo()
    monkeypatch.setattr(EchoProvider, "delay", 0.0)
    await drain(chat.regenerate(TurnResult()))
    assert [m.role for m in chat.history()] == ["user", "assistant"]


async def test_inline_think_blocks_are_not_shown_or_saved(chat, monkeypatch):
    async def thinker(self, model, messages, params):
        for piece in ["<think>Let me", " plan this.</think>", "\n\nMerhaba", "!"]:
            yield StreamEvent(text=piece)
        yield StreamEvent(finish_reason="stop")

    monkeypatch.setattr(EchoProvider, "stream_chat", thinker)
    r = TurnResult()
    shown = "".join(ev.text for ev in await drain(chat.send("selam", r)))
    assert shown == "Merhaba!"
    assert chat.history()[-1].content == "Merhaba!"


def test_set_param_and_reset(chat):
    assert chat.set_param("temperature", "0.3") == 0.3
    assert chat.gen_params().temperature == 0.3
    assert chat.set_param("stop", "###, User:") == ["###", "User:"]
    assert chat.set_param("temperature", "reset") is None
    assert chat.gen_params().temperature == Config().generation.temperature
    with pytest.raises(ChatError):
        chat.set_param("nope", "1")
    with pytest.raises(ChatError):
        chat.set_param("top_k", "abc")


def test_set_model_validates_provider(chat):
    with pytest.raises(ProviderError):
        chat.set_model("missing:model")
    with pytest.raises(ValueError):
        chat.set_model("no-colon")
    assert str(chat.set_model("local:qwen2.5:7b")) == "local:qwen2.5:7b"
    assert chat.default_model() == "local:qwen2.5:7b"


def _msg(i, role, content):
    return Message(i, 1, None, role, content, "", "")


def test_context_keeps_newest_within_budget():
    history = [_msg(i, "user" if i % 2 else "assistant", "x" * 300) for i in range(1, 21)]
    ctx = build_context("system", history, budget_tokens=600)
    assert ctx.messages[0].role == "system"
    assert ctx.dropped > 0
    assert ctx.included + ctx.dropped == 20
    # the newest message is always kept, and order is preserved
    assert ctx.messages[-1].content == history[-1].content
    assert ctx.estimated_tokens <= 600


def test_context_always_includes_latest_even_if_huge():
    ctx = build_context("s", [_msg(1, "user", "y" * 10_000)], budget_tokens=100)
    assert ctx.included == 1
