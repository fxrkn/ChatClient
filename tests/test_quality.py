"""Prompt diet, structured output for small models, time awareness, the roleplay eval."""

from __future__ import annotations

import json

import pytest
from tests.conftest import ALPHA

from chatclient.config import Config, MemoryConfig, ProviderConfig
from chatclient.core.chat import ChatService, TurnResult
from chatclient.memory import tasks
from chatclient.memory.engine import MemoryEngine
from chatclient.providers.base import GenParams, ProviderError, StreamEvent
from chatclient.providers.echo import EchoProvider
from chatclient.providers.registry import ProviderRegistry


async def say(chat, text):
    r = TurnResult()
    async for _ in chat.send(text, r):
        pass
    return r


@pytest.fixture
def mem(store, monkeypatch):
    seen: list[tuple[str, GenParams]] = []
    replies: dict[str, str] = {}

    async def model(self, model, messages, params):
        prompt = messages[-1].content
        seen.append((prompt, params))
        if "memory brief" in prompt:
            text = replies.get("brief", '{"about_past": false, "relevant": [], "claims": []}')
        elif "Return ONLY a JSON object" in prompt:
            text = replies.get("pass", '{"memories": [], "scene": {}}')
        else:
            text = "ok"
        yield StreamEvent(text=text)

    monkeypatch.setattr(EchoProvider, "stream_chat", model)
    config = Config(
        memory=MemoryConfig(embeddings="off"),
        providers={"local": ProviderConfig(type="openai_compat", base_url="http://x/v1")},
    )
    registry = ProviderRegistry(config)
    engine = MemoryEngine(config, store, registry)
    chat = ChatService(config, store, registry, engine)
    char = store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=char.id)
    return chat, engine, seen, replies


# ── prompt diet ─────────────────────────────────────────────────────────────


async def test_the_scene_is_replaced_not_piled_up(mem):
    chat, engine, _seen, replies = mem
    s = chat.require_session()
    engine.repo.set_scene(s.id, {"location": "the bar", "threads": ["finish the pizza"],
                                 "activity": "eating pizza"})  # fmt: skip
    replies["pass"] = json.dumps(
        {"memories": [], "scene": {"mood": "relaxed", "activity": "walking"}}
    )
    await say(chat, "let's go for a walk")
    await engine.run_all()
    scene = chat.refresh().scene
    assert "threads" not in scene  # finished: the model left it out, so it is gone
    assert scene["activity"] == "walking"
    assert scene["location"] == "the bar"  # a lasting fact the model forgot is kept


async def test_only_scene_facts_go_into_every_turn(mem):
    chat, engine, seen, _replies = mem
    engine.repo.set_scene(
        chat.require_session().id,
        {"location": "the bar", "time": "night", "relationship": "old friends",
         "mood": "tense", "activity": "eating pizza", "threads": ["finish the pizza"]},
    )  # fmt: skip
    await say(chat, "hi")
    prompt = seen[-1][0]
    assert "location: the bar · time: night · relationship: old friends" in prompt
    for steering in ("tense", "eating pizza", "finish the pizza"):
        assert steering not in prompt


# ── structured output ───────────────────────────────────────────────────────


async def test_memory_tasks_ask_for_their_json_shape(mem):
    chat, engine, seen, _replies = mem
    await say(chat, "hello")
    await engine.run_all()
    schemas = {id(p.json_schema) for _, p in seen if p.json_schema}
    assert id(tasks.BRIEF_SCHEMA) in schemas and id(tasks.PASS_SCHEMA) in schemas
    assert all(p.json_schema is None for prompt, p in seen if prompt == "hello")  # not the reply


def test_ollama_enforces_the_shape():
    from chatclient.config import ProviderConfig
    from chatclient.providers.ollama import OllamaProvider

    p = OllamaProvider("ollama", ProviderConfig(type="ollama", base_url="http://x"))
    body = p.build_body("m", [], GenParams(json_schema={"type": "object"}))
    assert body["format"] == {"type": "object"}
    assert "format" not in p.build_body("m", [], GenParams())


async def test_failing_memory_work_is_noticed_once(mem):
    chat, engine, _seen, replies = mem
    replies["pass"] = "sorry, I can't do JSON"
    for i in range(3):
        await say(chat, f"message {i}")
        engine.repo.conn.execute("UPDATE jobs SET run_after = NULL")  # don't wait for retries
        await engine.run_all()
    assert engine.struggling and "unreadable" in engine.last_failure
    replies["pass"] = '{"memories": [], "scene": {}}'
    await say(chat, "again")
    engine.repo.conn.execute("UPDATE jobs SET run_after = NULL")
    await engine.run_all()
    assert not engine.struggling  # one success clears it


# ── time awareness ──────────────────────────────────────────────────────────


async def test_a_long_break_is_told_once(mem):
    chat, _engine, seen, _replies = mem
    await say(chat, "see you")
    last = chat.last_reply()
    chat.store.conn.execute(
        "UPDATE messages SET created_at = '2026-01-01T20:00:00+00:00' WHERE id = ?", (last.id,)
    )
    await say(chat, "I'm back!")
    assert "Time has passed" in seen[-1][0] and "ago" in seen[-1][0]
    await say(chat, "so, where were we?")
    assert "Time has passed" not in seen[-1][0]  # only on the first message after the gap


def test_durations_read_naturally():
    from datetime import timedelta

    from chatclient.core.chat import _duration

    assert _duration(timedelta(hours=7)) == "7 hours"
    assert _duration(timedelta(days=3)) == "3 days"
    assert _duration(timedelta(days=21)) == "3 weeks"
    assert _duration(timedelta(days=95)) == "3 months"


# ── the roleplay eval itself (stand-in models) ──────────────────────────────


async def test_roleplay_eval_scores_and_compares(tmp_path, monkeypatch):
    from chatclient.evaluate_roleplay import SCENARIOS, run_roleplay_eval

    async def model(self, model, messages, params):
        prompt = messages[-1].content
        if "Answer each question about the reply" in prompt:
            text = json.dumps({"answered": True,
                               "on_topic": True, "invented_past": False,
                               "spoke_for_user": False, "naturalness": 4})  # fmt: skip
        elif "Return ONLY" in prompt:
            text = '{"memories": [], "scene": {}}'
        else:
            text = f"*smiles* Sure, about {prompt[-30:]}"
        yield StreamEvent(text=text)

    monkeypatch.setattr(EchoProvider, "stream_chat", model)
    cfg = Config(memory=MemoryConfig(embeddings="off"))
    lines: list[str] = []
    out = tmp_path / "a.json"
    s = await run_roleplay_eval(cfg, "echo:echo", "echo:echo", tmp_path, out, say=lines.append)
    assert s["replies"] == sum(len(sc.turns) for sc in SCENARIOS)
    assert s["naturalness"] == 4 and s["spoke_for_user"] == 0
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["summary"] == s and len(saved["replies"]) == s["replies"]
    lines.clear()
    await run_roleplay_eval(cfg, "echo:echo", "echo:echo", tmp_path, tmp_path / "b.json", out,
                            say=lines.append)  # fmt: skip
    assert any("(+0.0" in x or "(+0%" in x for x in lines)  # compared with the first run


def test_cloud_tasks_ask_for_json_and_no_reasoning():
    from chatclient.providers.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider(
        "openrouter", ProviderConfig(type="openai_compat", base_url="https://openrouter.ai/api/v1")
    )
    task = GenParams(json_schema={"type": "object"}, reasoning=False)
    body = p.build_body("m", [], task, usage=False)
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert body["reasoning"] == {"enabled": False}
    chat_body = p.build_body("m", [], GenParams(), usage=False)  # a normal reply: untouched
    assert "response_format" not in chat_body and "reasoning" not in chat_body


async def test_a_server_that_rejects_task_options_is_asked_plainly():
    import httpx
    import respx

    from chatclient.providers.openai_compat import OpenAICompatProvider

    p = OpenAICompatProvider(
        "local", ProviderConfig(type="openai_compat", base_url="http://localhost:9/v1")
    )
    bodies: list[dict] = []

    def reply(request):
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, text='{"error": "unknown field response_format"}')
        data = json.dumps({"choices": [{"delta": {"content": "{}"}, "finish_reason": "stop"}]})
        return httpx.Response(200, text=f"data: {data}\n\ndata: [DONE]\n\n",
                              headers={"content-type": "text/event-stream"})  # fmt: skip

    with respx.mock:
        respx.post("http://localhost:9/v1/chat/completions").mock(side_effect=reply)
        text = await tasks.complete(p, "m", [], schema={"type": "object"})
    assert text == "{}" and "response_format" not in bodies[-1]
    assert "m" in p._no_extras  # remembered: not tried again for this model
    await p.aclose()


async def test_utility_model_can_be_chosen_in_the_app(mem, tmp_path):
    chat, engine, _seen, _replies = mem
    chat.config.models.utility = "echo:from-config"
    assert chat.utility_model() == "echo:from-config"
    chat.set_utility_model("echo:fast")
    assert chat.utility_model() == "echo:fast"
    assert str(engine.utility_ref(chat.require_session())) == "echo:fast"  # memory uses it
    chat.set_utility_model("chat")
    assert chat.utility_model() == "echo:echo"  # the chat model, despite the config
    chat.set_utility_model(None)
    assert chat.utility_model() == "echo:from-config"  # back to the config file
    with pytest.raises(ProviderError):
        chat.set_utility_model("nosuchprovider:x")

    from tests.test_repl import run_script

    await run_script(chat, tmp_path, "/utility echo:fast\r/quit\r")
    assert chat.utility_model() == "echo:fast"
