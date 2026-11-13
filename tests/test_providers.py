import json
from types import SimpleNamespace

import httpx
import pytest
import respx

from chatclient.config import ProviderConfig
from chatclient.providers.anthropic import AnthropicProvider, to_anthropic_messages
from chatclient.providers.base import ChatMessage, GenParams, ProviderError, ThinkFilter
from chatclient.providers.gemini import GeminiProvider
from chatclient.providers.ollama import OllamaProvider
from chatclient.providers.openai_compat import OpenAICompatProvider
from chatclient.providers.registry import parse_model_ref

MSGS = [ChatMessage("system", "be nice"), ChatMessage("user", "merhaba")]


async def collect(agen):
    return [ev async for ev in agen]


def sse(*payloads: object) -> bytes:
    out = []
    for p in payloads:
        data = p if isinstance(p, str) else json.dumps(p)
        out.append(f"data: {data}\n\n")
    return "".join(out).encode()


def test_parse_model_ref_splits_on_first_colon():
    ref = parse_model_ref("ollama:llama3.1:8b")
    assert (ref.provider, ref.model) == ("ollama", "llama3.1:8b")
    with pytest.raises(ValueError):
        parse_model_ref("llama3")


# ── OpenAI-compatible ────────────────────────────────────────────────────────


@respx.mock
async def test_openai_compat_streams_text_and_usage():
    route = respx.post("http://srv/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse(
                {"choices": [{"delta": {"role": "assistant"}}]},
                {"choices": [{"delta": {"content": "Mer"}}]},
                ": keep-alive comment",
                {"choices": [{"delta": {"content": "haba"}, "finish_reason": "stop"}]},
                {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 3}},
                "[DONE]",
            ),
        )
    )
    p = OpenAICompatProvider(
        "t", ProviderConfig(type="openai_compat", base_url="http://srv/v1"), "sk-test"
    )
    events = await collect(p.stream_chat("m", MSGS, GenParams(temperature=0.7, top_k=40)))
    assert "".join(e.text for e in events) == "Merhaba"
    assert events[-1].usage.prompt_tokens == 12
    body = json.loads(route.calls[0].request.content)
    assert body["temperature"] == 0.7 and body["top_k"] == 40
    assert "top_p" not in body  # unset params are not sent
    assert route.calls[0].request.headers["authorization"] == "Bearer sk-test"
    await p.aclose()


def test_openai_compat_reply_limit_defaults():
    cloud = OpenAICompatProvider(
        "openrouter",
        ProviderConfig(type="openai_compat", base_url="https://openrouter.ai/api/v1"),
    )
    local = OpenAICompatProvider(
        "lmstudio", ProviderConfig(type="openai_compat", base_url="http://localhost:1234/v1")
    )
    capped = OpenAICompatProvider(
        "x", ProviderConfig(type="openai_compat", base_url="https://x.ai/v1", max_tokens=800)
    )
    # cloud APIs get a cap so they don't reserve credit for the model's maximum output
    assert cloud.build_body("m", MSGS, GenParams(), usage=False)["max_tokens"] == 4096
    assert "max_tokens" not in local.build_body("m", MSGS, GenParams(), usage=False)
    assert capped.build_body("m", MSGS, GenParams(), usage=False)["max_tokens"] == 800
    # an explicit session value always wins
    assert cloud.build_body("m", MSGS, GenParams(max_tokens=300), usage=False)["max_tokens"] == 300


@respx.mock
async def test_openrouter_upstream_error_details_are_shown():
    body = {
        "error": {
            "message": "Provider returned error",
            "code": 429,
            "metadata": {
                "raw": "google/gemma:free is temporarily rate-limited upstream. Please retry shortly.",
                "provider_name": "Google AI Studio",
            },
        }
    }
    respx.post("http://srv/v1/chat/completions").mock(return_value=httpx.Response(429, json=body))
    p = OpenAICompatProvider("t", ProviderConfig(type="openai_compat", base_url="http://srv/v1"))
    with pytest.raises(ProviderError) as info:
        await collect(p.stream_chat("m", MSGS, GenParams()))
    msg = str(info.value)
    assert "Google AI Studio: google/gemma:free is temporarily rate-limited" in msg
    assert "/regen" in msg
    assert info.value.retryable
    await p.aclose()


@respx.mock
async def test_openai_compat_402_gives_hint():
    respx.post("http://srv/v1/chat/completions").mock(
        return_value=httpx.Response(402, json={"error": {"message": "requires more credits"}})
    )
    p = OpenAICompatProvider("t", ProviderConfig(type="openai_compat", base_url="http://srv/v1"))
    with pytest.raises(ProviderError, match="/set max_tokens"):
        await collect(p.stream_chat("m", MSGS, GenParams()))
    await p.aclose()


@respx.mock
async def test_openai_compat_retries_without_stream_options():
    route = respx.post("http://srv/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(400, json={"error": {"message": "unknown field stream_options"}}),
            httpx.Response(200, content=sse({"choices": [{"delta": {"content": "ok"}}]}, "[DONE]")),
        ]
    )
    p = OpenAICompatProvider("t", ProviderConfig(type="openai_compat", base_url="http://srv/v1"))
    events = await collect(p.stream_chat("m", MSGS, GenParams()))
    assert "".join(e.text for e in events) == "ok"
    assert "stream_options" not in json.loads(route.calls[1].request.content)
    await p.aclose()


@respx.mock
async def test_openai_compat_error_message_is_readable():
    respx.post("http://srv/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
    )
    p = OpenAICompatProvider("t", ProviderConfig(type="openai_compat", base_url="http://srv/v1"))
    with pytest.raises(ProviderError, match="Invalid API key"):
        await collect(p.stream_chat("m", MSGS, GenParams()))
    await p.aclose()


async def test_openai_compat_connection_error_hint():
    p = OpenAICompatProvider(
        "local", ProviderConfig(type="openai_compat", base_url="http://127.0.0.1:9/v1")
    )
    with pytest.raises(ProviderError, match="server running"):
        await collect(p.stream_chat("m", MSGS, GenParams()))
    await p.aclose()


@respx.mock
async def test_openai_compat_list_models():
    respx.get("http://srv/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "b"}, {"id": "a"}]})
    )
    p = OpenAICompatProvider("t", ProviderConfig(type="openai_compat", base_url="http://srv/v1"))
    assert await p.list_models() == ["a", "b"]
    await p.aclose()


# ── Ollama ───────────────────────────────────────────────────────────────────


@respx.mock
async def test_ollama_native_stream_and_options():
    lines = [
        {"message": {"role": "assistant", "content": "Selam"}, "done": False},
        {"message": {"role": "assistant", "content": "!"}, "done": False},
        {
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 20,
            "eval_count": 2,
        },
    ]
    route = respx.post("http://ol/api/chat").mock(
        return_value=httpx.Response(200, content="\n".join(json.dumps(x) for x in lines).encode())
    )
    p = OllamaProvider("ollama", ProviderConfig(type="ollama", base_url="http://ol", num_ctx=8192))
    events = await collect(p.stream_chat("llama3", MSGS, GenParams(max_tokens=100, min_p=0.05)))
    assert "".join(e.text for e in events) == "Selam!"
    assert events[-1].usage.completion_tokens == 2
    body = json.loads(route.calls[0].request.content)
    assert body["options"] == {"num_predict": 100, "min_p": 0.05, "num_ctx": 8192}
    assert body["keep_alive"] == "30m"
    await p.aclose()


@respx.mock
async def test_ollama_disables_thinking_and_reports_load_time():
    done = {"message": {"content": ""}, "done": True, "load_duration": 3_200_000_000}
    route = respx.post("http://ol/api/chat").mock(
        return_value=httpx.Response(
            200,
            content=(
                json.dumps({"message": {"thinking": "hmm", "content": ""}, "done": False})
                + "\n"
                + json.dumps({"message": {"content": "Selam"}, "done": False})
                + "\n"
                + json.dumps(done)
            ).encode(),
        )
    )
    p = OllamaProvider("ollama", ProviderConfig(type="ollama", base_url="http://ol"))
    events = await collect(p.stream_chat("qwen3.5:4b", MSGS, GenParams()))
    assert json.loads(route.calls[0].request.content)["think"] is False
    assert "".join(e.text for e in events) == "Selam"  # thinking never shown
    assert events[-1].usage.load_seconds == pytest.approx(3.2)
    await p.aclose()


@respx.mock
async def test_ollama_retries_without_think_when_rejected():
    route = respx.post("http://ol/api/chat").mock(
        side_effect=[
            httpx.Response(400, json={"error": '"llama3" does not support thinking'}),
            httpx.Response(
                200, content=json.dumps({"message": {"content": "ok"}, "done": True}).encode()
            ),
        ]
    )
    p = OllamaProvider("ollama", ProviderConfig(type="ollama", base_url="http://ol"))
    events = await collect(p.stream_chat("llama3", MSGS, GenParams()))
    assert "".join(e.text for e in events) == "ok"
    assert "think" not in json.loads(route.calls[1].request.content)
    # remembered: the next request skips it straight away
    assert "think" not in p.build_body("llama3", MSGS, GenParams())
    await p.aclose()


def test_think_filter_handles_split_tags():
    f = ThinkFilter()
    chunks = ["<thi", "nk>\nplanning the reply", "...</th", "ink>\n\nMer", "haba <", "b>!"]
    out = "".join(f.feed(c) for c in chunks) + f.flush()
    assert out == "Merhaba <b>!"
    g = ThinkFilter()
    assert g.feed("plain text") + g.flush() == "plain text"
    h = ThinkFilter()
    assert h.feed("<think>never closed") + h.flush() == ""


@respx.mock
async def test_ollama_missing_model_hint():
    respx.post("http://ol/api/chat").mock(
        return_value=httpx.Response(404, json={"error": "model not found"})
    )
    p = OllamaProvider("ollama", ProviderConfig(type="ollama", base_url="http://ol"))
    with pytest.raises(ProviderError, match="ollama pull"):
        await collect(p.stream_chat("nope", MSGS, GenParams()))
    await p.aclose()


# ── Gemini ───────────────────────────────────────────────────────────────────


@respx.mock
async def test_gemini_stream_skips_thoughts():
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/g1:streamGenerateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            content=sse(
                {"candidates": [{"content": {"parts": [{"text": "hmm", "thought": True}]}}]},
                {"candidates": [{"content": {"parts": [{"text": "Mer"}]}}]},
                {
                    "candidates": [
                        {"content": {"parts": [{"text": "haba"}]}, "finishReason": "STOP"}
                    ],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 2},
                },
            ),
        )
    )
    p = GeminiProvider("gemini", ProviderConfig(type="gemini"), "key")
    events = await collect(p.stream_chat("g1", MSGS, GenParams(temperature=0.5)))
    assert "".join(e.text for e in events) == "Merhaba"
    assert events[-1].finish_reason == "stop"
    body = json.loads(route.calls[0].request.content)
    assert body["systemInstruction"]["parts"][0]["text"] == "be nice"
    assert body["generationConfig"] == {"temperature": 0.5}
    assert route.calls[0].request.headers["x-goog-api-key"] == "key"
    await p.aclose()


async def test_gemini_requires_key():
    p = GeminiProvider("gemini", ProviderConfig(type="gemini", api_key_env="NOPE_NOT_SET"))
    with pytest.raises(ProviderError, match="no API key"):
        await collect(p.stream_chat("g1", MSGS, GenParams()))


# ── Anthropic ────────────────────────────────────────────────────────────────


def test_anthropic_message_conversion():
    system, msgs = to_anthropic_messages(
        [
            ChatMessage("system", "sys"),
            ChatMessage("assistant", "greeting"),
            ChatMessage("user", "a"),
            ChatMessage("user", "b"),
        ]
    )
    assert system == "sys"
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[-1]["content"] == "a\n\nb"


def test_anthropic_request_omits_sampling_and_adds_features():
    p = AnthropicProvider("anthropic", ProviderConfig(type="anthropic", effort="low"), "k")
    req = p.build_request("claude-opus-5-5", MSGS, GenParams(temperature=0.9), extras=True)
    assert "temperature" not in req
    assert req["max_tokens"] >= 1024
    assert req["output_config"] == {"effort": "low"}
    assert req["fallbacks"] == "default" and req["betas"]
    assert req["cache_control"] == {"type": "ephemeral"}
    plain = p.build_request("claude-haiku-4-5", MSGS, GenParams(), extras=True)
    assert "fallbacks" not in plain


class _FakeStream:
    def __init__(self, events):
        self._events = events
        self.closed = False

    def __aiter__(self):
        async def gen():
            for e in self._events:
                yield e

        return gen()

    async def close(self):
        self.closed = True


async def test_anthropic_stream_event_handling():
    ns = SimpleNamespace
    events = [
        ns(
            type="message_start",
            message=ns(
                usage=ns(input_tokens=10, cache_read_input_tokens=90, cache_creation_input_tokens=0)
            ),
        ),
        ns(type="content_block_start"),
        ns(type="content_block_delta", delta=ns(type="thinking_delta", thinking="")),
        ns(type="content_block_delta", delta=ns(type="text_delta", text="Mer")),
        ns(type="content_block_delta", delta=ns(type="text_delta", text="haba")),
        ns(type="message_delta", delta=ns(stop_reason="end_turn"), usage=ns(output_tokens=4)),
        ns(type="message_stop"),
    ]
    fake = _FakeStream(events)
    captured = {}

    async def create(**req):
        captured.update(req)
        return fake

    p = AnthropicProvider("anthropic", ProviderConfig(type="anthropic"), "k")
    p._client = ns(beta=ns(messages=ns(create=create)))
    out = await collect(p.stream_chat("claude-opus-5-5", MSGS, GenParams()))
    assert "".join(e.text for e in out) == "Merhaba"
    assert out[-1].usage.prompt_tokens == 100 and out[-1].usage.completion_tokens == 4
    assert out[-1].finish_reason == "end_turn"
    assert fake.closed
    assert captured["system"] == "be nice"
