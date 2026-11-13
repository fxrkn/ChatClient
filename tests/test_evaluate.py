"""The evaluation harness itself (with stand-in models; real runs use a real one)."""

import json
import re

from chatclient.config import Config, MemoryConfig
from chatclient.memory.evaluate import SCENARIOS, run_eval
from chatclient.providers.base import StreamEvent
from chatclient.providers.echo import EchoProvider


async def test_unusable_model_scores_zero(monkeypatch, tmp_path):
    async def garbage(self, model, messages, params):
        yield StreamEvent(text="no idea")

    monkeypatch.setattr(EchoProvider, "stream_chat", garbage)
    lines: list[str] = []
    rate = await run_eval(
        Config(memory=MemoryConfig(embeddings="off")), "echo:echo", tmp_path, lines.append
    )
    assert rate == 0.0
    assert sum("no usable brief" in x for x in lines) == len(SCENARIOS)


async def test_a_sensible_model_scores_well(monkeypatch, tmp_path):
    async def sensible(self, model, messages, params):
        prompt = messages[-1].content
        message = re.search(r'new message[^\n]*:\n"(.*)"', prompt).group(1)
        known = (
            prompt.split("new message")[0]
            + prompt.split("found by search.")[-1].split("Return ONLY")[0]
        ).lower()
        ids = re.findall(r"\[([MQD]\d+)\]", prompt)
        words = [w for w in re.findall(r"\w+", message.lower()) if len(w) > 2]
        claims = [
            {"claim": w, "status": "supported" if w in known else "unsupported"} for w in words
        ]
        brief = {"about_past": "?" in message, "relevant": ids, "claims": claims}
        yield StreamEvent(text=json.dumps(brief))

    monkeypatch.setattr(EchoProvider, "stream_chat", sensible)
    lines: list[str] = []
    rate = await run_eval(
        Config(memory=MemoryConfig(embeddings="off")), "echo:echo", tmp_path, lines.append
    )
    assert 0.3 < rate <= 1.0
    assert lines[-1].strip().startswith("Score:")
