"""Anthropic (Claude) adapter using the official SDK.

Notes for current Claude models:
- Sampling parameters (temperature/top_p/top_k) are rejected by current models, so
  they are not sent. Response depth/speed is controlled with ``effort`` instead.
- Thinking is adaptive and hidden by default; only the visible text is streamed.
- Top-level automatic prompt caching is enabled: long, stable chat histories are
  re-read from cache, which cuts both cost and time-to-first-token.
- Server-side refusal fallbacks are enabled for models that support them.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic

from chatclient.config import ProviderConfig
from chatclient.providers.base import (
    ChatMessage,
    GenParams,
    Provider,
    ProviderError,
    StreamEvent,
    Usage,
    require_key,
)

log = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 16000  # thinking tokens count toward this, so keep it generous
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
}


def to_anthropic_messages(
    messages: Sequence[ChatMessage],
) -> tuple[str, list[dict[str, Any]]]:
    """Split out system text and coerce history into strict user/assistant alternation."""
    system_parts = [m.content for m in messages if m.role == "system" and m.content.strip()]
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system" or not m.content:
            continue
        if out and out[-1]["role"] == m.role:
            out[-1]["content"] += "\n\n" + m.content
        else:
            out.append({"role": m.role, "content": m.content})
    if out and out[0]["role"] == "assistant":
        # e.g. a character greeting opened the chat; the API requires a user turn first
        out.insert(0, {"role": "user", "content": "(The conversation begins.)"})
    return "\n\n".join(system_parts), out


class AnthropicProvider(Provider):
    def __init__(self, name: str, config: ProviderConfig, api_key: str | None = None):
        super().__init__(name, config, api_key)
        self._client: anthropic.AsyncAnthropic | None = None

    @property
    def client(self) -> anthropic.AsyncAnthropic:
        if self._client is None:
            self._client = anthropic.AsyncAnthropic(
                api_key=require_key(self),
                base_url=self.config.base_url or None,
                timeout=self.config.timeout,
                max_retries=2,
                default_headers=self.config.headers or None,
            )
        return self._client

    def describe(self) -> str:
        return self.config.base_url or "api.anthropic.com"

    def build_request(
        self, model: str, messages: Sequence[ChatMessage], params: GenParams, *, extras: bool
    ) -> dict[str, Any]:
        system, msgs = to_anthropic_messages(messages)
        req: dict[str, Any] = {
            "model": model,
            "max_tokens": params.max_tokens or self.config.max_tokens or DEFAULT_MAX_TOKENS,
            "messages": msgs,
            "cache_control": {"type": "ephemeral"},
            "stream": True,
        }
        if system:
            req["system"] = system
        if params.stop:
            req["stop_sequences"] = params.stop
        if extras:
            if self.config.effort:
                req["output_config"] = {"effort": self.config.effort}
            if self.config.refusal_fallback and model in FALLBACK_MODELS:
                req["betas"] = [FALLBACK_BETA]
                req["fallbacks"] = "default"
        return req

    async def stream_chat(
        self, model: str, messages: Sequence[ChatMessage], params: GenParams
    ) -> AsyncIterator[StreamEvent]:
        extras = True
        for attempt in range(2):
            req = self.build_request(model, messages, params, extras=extras)
            try:
                stream = await self.client.beta.messages.create(**req)
            except anthropic.BadRequestError as e:
                # An optional feature (effort / fallbacks) may not be supported by this
                # model; retry once without them before surfacing the error.
                has_extras = "output_config" in req or "fallbacks" in req
                if attempt == 0 and has_extras:
                    log.info("%s: retrying without optional features: %s", self.name, e.message)
                    extras = False
                    continue
                raise ProviderError(f"{self.name}: {e.message}", status=400) from e
            except anthropic.APIStatusError as e:
                raise ProviderError(
                    f"{self.name}: HTTP {e.status_code}: {e.message}",
                    status=e.status_code,
                    retryable=e.status_code in (408, 429) or e.status_code >= 500,
                ) from e
            except anthropic.APIConnectionError as e:
                raise ProviderError(f"{self.name}: connection failed: {e}", retryable=True) from e
            break

        usage = Usage()
        try:
            async for event in stream:
                etype = getattr(event, "type", "")
                if etype == "message_start":
                    u = event.message.usage
                    usage.prompt_tokens = (
                        (u.input_tokens or 0)
                        + (getattr(u, "cache_read_input_tokens", None) or 0)
                        + (getattr(u, "cache_creation_input_tokens", None) or 0)
                    )
                elif etype == "content_block_delta":
                    if getattr(event.delta, "type", "") == "text_delta":
                        yield StreamEvent(text=event.delta.text)
                elif etype == "message_delta":
                    if event.usage is not None and event.usage.output_tokens is not None:
                        usage.completion_tokens = event.usage.output_tokens
                    if event.delta.stop_reason:
                        yield StreamEvent(usage=usage, finish_reason=event.delta.stop_reason)
                elif etype == "error":
                    err = getattr(event, "error", None)
                    raise ProviderError(f"{self.name}: {getattr(err, 'message', err)}")
        except anthropic.APIError as e:
            raise ProviderError(f"{self.name}: {e}", retryable=True) from e
        finally:
            await stream.close()

    async def list_models(self) -> list[str]:
        try:
            return sorted([m.id async for m in self.client.models.list(limit=100)])
        except anthropic.APIStatusError as e:
            raise ProviderError(f"{self.name}: HTTP {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError(f"{self.name}: connection failed: {e}") from e

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
