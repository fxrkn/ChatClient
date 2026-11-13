"""Offline built-in provider for trying the app and for tests.

``echo:echo`` repeats your message back, word by word, like a streaming model.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence

from chatclient.providers.base import ChatMessage, GenParams, Provider, StreamEvent, Usage
from chatclient.text import estimate_tokens


class EchoProvider(Provider):
    delay: float = 0.02

    async def stream_chat(
        self, model: str, messages: Sequence[ChatMessage], params: GenParams
    ) -> AsyncIterator[StreamEvent]:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
        turns = sum(1 for m in messages if m.role != "system")
        reply = f"(echo · {turns} messages in context) {last_user}"
        words = reply.split(" ")
        for i, w in enumerate(words):
            await asyncio.sleep(self.delay)
            yield StreamEvent(text=w if i == 0 else " " + w)
        prompt = sum(estimate_tokens(m.content) for m in messages)
        yield StreamEvent(usage=Usage(prompt, estimate_tokens(reply)), finish_reason="stop")

    async def list_models(self) -> list[str]:
        return ["echo"]

    def describe(self) -> str:
        return "built-in, offline"
