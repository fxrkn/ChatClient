"""OpenAI-compatible Chat Completions adapter.

Covers OpenAI, OpenRouter, Groq, Together, DeepInfra, Mistral, xAI, DeepSeek and
local servers: LM Studio, llama.cpp ``llama-server``, vLLM, KoboldCpp, Jan, and
Ollama's ``/v1`` endpoint.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from chatclient.providers.base import (
    ChatMessage,
    GenParams,
    HttpProvider,
    ProviderError,
    StreamEvent,
    Usage,
)
from chatclient.providers.registry import is_local
from chatclient.providers.sse import iter_sse

log = logging.getLogger(__name__)

CLOUD_DEFAULT_MAX_TOKENS = 4096


class OpenAICompatProvider(HttpProvider):
    default_base_url = "https://api.openai.com/v1"

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def reply_limit(self, params: GenParams) -> int | None:
        """Session value → provider config → 4096 for cloud APIs (none for local servers).

        Without a cap, services like OpenRouter assume the model's maximum output
        (often 65k tokens) and reject the request if the balance can't cover it.
        """
        if params.max_tokens:
            return params.max_tokens
        if self.config.max_tokens:
            return self.config.max_tokens
        return None if is_local(self.config) else CLOUD_DEFAULT_MAX_TOKENS

    def build_body(
        self, model: str, messages: Sequence[ChatMessage], params: GenParams, *, usage: bool
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
        }
        if usage:
            body["stream_options"] = {"include_usage": True}
        # Standard OpenAI parameters
        for key in ("temperature", "top_p", "seed", "stop"):
            val = getattr(params, key)
            if val is not None:
                body[key] = val
        if max_tokens := self.reply_limit(params):
            body["max_tokens"] = max_tokens
        # Widely supported extensions (OpenRouter, vLLM, llama.cpp, LM Studio...).
        # Only sent when the user sets them, since strict servers reject unknown fields.
        for key in ("top_k", "min_p", "repetition_penalty"):
            val = getattr(params, key)
            if val is not None:
                body[key] = val
        return body

    async def stream_chat(
        self, model: str, messages: Sequence[ChatMessage], params: GenParams
    ) -> AsyncIterator[StreamEvent]:
        use_usage = self.config.stream_usage
        try:
            for attempt in range(2):
                body = self.build_body(model, messages, params, usage=use_usage)
                async with self.client.stream(
                    "POST", f"{self.base_url}/chat/completions", json=body
                ) as resp:
                    if resp.status_code == 400 and use_usage and attempt == 0:
                        # Some older servers reject stream_options; retry without it.
                        text = (await resp.aread()).decode("utf-8", "replace")
                        if "stream_options" in text or "include_usage" in text:
                            log.info("%s: retrying without stream_options", self.name)
                            use_usage = False
                            continue
                        raise ProviderError(f"{self.name}: HTTP 400: {text[:500]}", status=400)
                    await self.raise_for_status(resp)
                    async for ev in self._parse(resp):
                        yield ev
                    return
        except httpx.HTTPError as e:
            raise self.connection_error(e) from e

    async def _parse(self, resp: httpx.Response) -> AsyncIterator[StreamEvent]:
        async for sse in iter_sse(resp):
            if sse.data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(sse.data)
            except ValueError:
                log.debug("%s: skipping non-JSON SSE data: %r", self.name, sse.data[:200])
                continue
            if "error" in chunk:
                err = chunk["error"]
                msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                raise ProviderError(f"{self.name}: {msg}")
            usage = None
            if u := chunk.get("usage"):
                usage = Usage(u.get("prompt_tokens"), u.get("completion_tokens"))
            text, finish = "", None
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                text += delta.get("content") or ""
                finish = choice.get("finish_reason") or finish
            if text or usage or finish:
                yield StreamEvent(text=text, usage=usage, finish_reason=finish)

    async def verify_key(self) -> None:
        # OpenRouter lists models without auth, so check the key endpoint instead.
        if "openrouter.ai" in self.base_url:
            try:
                resp = await self.client.get(f"{self.base_url}/key", timeout=10.0)
            except httpx.HTTPError as e:
                raise self.connection_error(e) from e
            await self.raise_for_status(resp)
            return
        await self.list_models()

    async def list_model_info(self) -> list[dict[str, Any]]:
        """Raw model entries (OpenRouter includes pricing, context length, created...)."""
        try:
            resp = await self.client.get(f"{self.base_url}/models", timeout=10.0)
        except httpx.HTTPError as e:
            raise self.connection_error(e) from e
        await self.raise_for_status(resp)
        data = resp.json()
        items = data.get("data", data) if isinstance(data, dict) else data
        return [m for m in items if isinstance(m, dict) and "id" in m]

    async def list_models(self) -> list[str]:
        return sorted(str(m["id"]) for m in await self.list_model_info())
