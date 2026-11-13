import json

import httpx
import pytest
import respx
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from chatclient.config import Config, ProviderConfig
from chatclient.core.chat import ChatService
from chatclient.core.free_models import (
    drop_working,
    find_working,
    free_candidates,
    is_free_model,
    load_working,
    save_working,
)
from chatclient.i18n import Translator
from chatclient.paths import resolve_paths
from chatclient.providers.registry import ProviderRegistry
from chatclient.ui.repl import Repl

BASE = "https://openrouter.ai/api/v1"
MODELS = {
    "data": [
        {"id": "paid/model", "created": 300},
        {"id": "old/c:free", "created": 100},
        {"id": "new/b:free", "created": 200},
        {"id": "busy/a:free", "created": 150},
    ]
}


def rate_limited():
    return httpx.Response(
        429,
        json={
            "error": {
                "message": "Provider returned error",
                "metadata": {"raw": "rate-limited upstream", "provider_name": "X"},
            }
        },
    )


def ok_stream(text="Merhaba!"):
    body = f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\ndata: [DONE]\n\n"
    return httpx.Response(200, content=body.encode())


def handler(working: set[str], calls: list[str]):
    def respond(request: httpx.Request) -> httpx.Response:
        model = json.loads(request.content)["model"]
        calls.append(model)
        return ok_stream() if model in working else rate_limited()

    return respond


@pytest.fixture
def or_chat(store):
    config = Config(
        providers={"openrouter": ProviderConfig(type="openai_compat", base_url=BASE, api_key="k")}
    )
    service = ChatService(config, store, ProviderRegistry(config))
    service.new_session(model="openrouter:busy/a:free")
    return service


async def run_keys(chat, tmp_path, keys):
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths, show_stats=False)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(keys)
        await repl.run([])
    await chat.registry.aclose()


def test_is_free_model():
    assert is_free_model("google/gemma:free")
    assert is_free_model("openrouter/free")
    assert not is_free_model("openai/gpt-mini-latest")


@respx.mock
async def test_candidates_are_free_only_and_newest_first(or_chat):
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json=MODELS))
    p = or_chat.registry.get("openrouter")
    assert await free_candidates(p) == ["new/b:free", "busy/a:free", "old/c:free"]


@respx.mock
async def test_find_working_stops_early(or_chat):
    calls: list[str] = []
    respx.post(f"{BASE}/chat/completions").mock(side_effect=handler({"m1", "m2"}, calls))
    p = or_chat.registry.get("openrouter")
    models = ["m1", "m2", "m3", "m4", "m5", "m6"]
    results = await find_working(p, models, want=2, concurrency=2)
    assert [r.model for r in results if r.ok] == ["m1", "m2"]
    assert calls == ["m1", "m2"]  # stopped after the first batch: no quota wasted


def test_working_cache_roundtrip(store):
    from chatclient.core.free_models import ProbeResult

    save_working(
        store,
        "or",
        [
            ProbeResult("slow", True, 3.0),
            ProbeResult("fast", True, 0.5),
            ProbeResult("bad", False, error="x"),
        ],
    )
    assert load_working(store, "or") == ["fast", "slow"]
    drop_working(store, "or", "fast")
    assert load_working(store, "or") == ["slow"]


@respx.mock
async def test_busy_free_model_falls_back_automatically(or_chat, tmp_path, capsys):
    calls: list[str] = []
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json=MODELS))
    respx.post(f"{BASE}/chat/completions").mock(side_effect=handler({"new/b:free"}, calls))
    await run_keys(or_chat, tmp_path, "merhaba\r/quit\r")
    out = capsys.readouterr().out
    assert "busy/a:free is busy" in out
    assert "Merhaba!" in out
    assert or_chat.require_session().model == "openrouter:new/b:free"
    assert [m.role for m in or_chat.history()] == ["user", "assistant"]
    assert calls.count("busy/a:free") == 1  # a 429 is not blindly retried
    assert load_working(or_chat.store, "openrouter") == ["new/b:free"]


@respx.mock
async def test_free_command_selects_fastest_working(or_chat, tmp_path, capsys):
    calls: list[str] = []
    respx.get(f"{BASE}/models").mock(return_value=httpx.Response(200, json=MODELS))
    respx.post(f"{BASE}/chat/completions").mock(
        side_effect=handler({"new/b:free", "old/c:free"}, calls)
    )
    await run_keys(or_chat, tmp_path, "/free\r/free list\r/quit\r")
    out = capsys.readouterr().out
    assert "✓ new/b:free" in out and "✗ busy/a:free" in out
    assert or_chat.require_session().model in ("openrouter:new/b:free", "openrouter:old/c:free")
    assert "paid/model" not in calls
    assert set(load_working(or_chat.store, "openrouter")) == {"new/b:free", "old/c:free"}


@respx.mock
async def test_no_fallback_for_paid_models(or_chat, tmp_path, capsys):
    or_chat.set_model("openrouter:paid/model")
    calls: list[str] = []
    respx.post(f"{BASE}/chat/completions").mock(side_effect=handler(set(), calls))
    await run_keys(or_chat, tmp_path, "merhaba\r/quit\r")
    assert "HTTP 429" in capsys.readouterr().out
    assert or_chat.require_session().model == "openrouter:paid/model"
    assert calls == ["paid/model"]
