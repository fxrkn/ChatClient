"""Native Ollama adapter (``/api/chat``).

Used instead of Ollama's OpenAI-compatible endpoint because only the native API
lets us set ``num_ctx`` (Ollama's default context window is small, which makes
long conversations silently lose their beginning) and ``keep_alive``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from chatclient.config import ProviderConfig
from chatclient.providers.base import (
    ChatMessage,
    GenParams,
    HttpProvider,
    ProviderError,
    StreamEvent,
    Usage,
)

log = logging.getLogger(__name__)


def _ns(value: object) -> float | None:
    """Ollama reports durations in nanoseconds."""
    return value / 1e9 if isinstance(value, int | float) else None


class OllamaProvider(HttpProvider):
    default_base_url = "http://localhost:11434"

    def __init__(self, name: str, config: ProviderConfig, api_key: str | None = None):
        super().__init__(name, config, api_key)
        self._no_think: set[str] = set()  # models that rejected the "think" parameter

    def auth_headers(self) -> dict[str, str]:
        # Remote/proxied Ollama deployments sometimes sit behind a bearer token.
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def build_body(
        self, model: str, messages: Sequence[ChatMessage], params: GenParams
    ) -> dict[str, Any]:
        options: dict[str, Any] = {}
        mapping = {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
            "max_tokens": "num_predict",
            "repetition_penalty": "repeat_penalty",
            "seed": "seed",
            "stop": "stop",
        }
        for ours, theirs in mapping.items():
            val = getattr(params, ours)
            if val is not None:
                options[theirs] = val
        if self.config.num_ctx:
            options["num_ctx"] = self.config.num_ctx
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": options,
        }
        if self.config.keep_alive:
            body["keep_alive"] = self.config.keep_alive
        think = False if params.reasoning is False else self.config.think
        if think is not None and model not in self._no_think:
            body["think"] = think
        if params.json_schema:
            body["format"] = params.json_schema  # decoding is constrained to valid JSON
        return body

    async def stream_chat(
        self, model: str, messages: Sequence[ChatMessage], params: GenParams
    ) -> AsyncIterator[StreamEvent]:
        try:
            for attempt in range(2):
                body = self.build_body(model, messages, params)
                async with self.client.stream(
                    "POST", f"{self.base_url}/api/chat", json=body
                ) as resp:
                    if resp.status_code == 404:
                        raise ProviderError(
                            f"{self.name}: model '{model}' not found — run `ollama pull {model}`",
                            status=404,
                        )
                    if resp.status_code == 400 and "think" in body and attempt == 0:
                        text = (await resp.aread()).decode("utf-8", "replace")
                        if "think" in text.lower():
                            # Older Ollama, or a model without reasoning support.
                            log.info("%s: %s rejected 'think'; retrying without", self.name, model)
                            self._no_think.add(model)
                            continue
                        raise ProviderError(f"{self.name}: HTTP 400: {text[:300]}", status=400)
                    await self.raise_for_status(resp)
                    async for ev in self._parse(resp):
                        yield ev
                    return
        except httpx.HTTPError as e:
            raise self.connection_error(e) from e

    async def _parse(self, resp: httpx.Response) -> AsyncIterator[StreamEvent]:
        async for line in resp.aiter_lines():
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            if err := chunk.get("error"):
                raise ProviderError(f"{self.name}: {err}")
            # message.thinking (reasoning) is deliberately not shown
            text = (chunk.get("message") or {}).get("content") or ""
            if chunk.get("done"):
                usage = Usage(
                    chunk.get("prompt_eval_count"),
                    chunk.get("eval_count"),
                    load_seconds=_ns(chunk.get("load_duration")),
                )
                log.info(
                    "%s timings: load=%.2fs prompt=%.2fs (%s tok) gen=%.2fs (%s tok)",
                    self.name,
                    _ns(chunk.get("load_duration")) or 0,
                    _ns(chunk.get("prompt_eval_duration")) or 0,
                    chunk.get("prompt_eval_count"),
                    _ns(chunk.get("eval_duration")) or 0,
                    chunk.get("eval_count"),
                )
                yield StreamEvent(
                    text=text, usage=usage, finish_reason=chunk.get("done_reason", "stop")
                )
                return
            if text:
                yield StreamEvent(text=text)

    async def list_models(self) -> list[str]:
        try:
            resp = await self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
        except httpx.HTTPError as e:
            raise self.connection_error(e) from e
        await self.raise_for_status(resp)
        return sorted(m["name"] for m in resp.json().get("models", []))
