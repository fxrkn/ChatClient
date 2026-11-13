"""Google Gemini adapter (Generative Language API, native SSE streaming)."""

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
    require_key,
)
from chatclient.providers.sse import iter_sse

log = logging.getLogger(__name__)


def to_gemini_contents(messages: Sequence[ChatMessage]) -> tuple[str, list[dict[str, Any]]]:
    system = "\n\n".join(m.content for m in messages if m.role == "system" and m.content.strip())
    contents: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "system" or not m.content:
            continue
        role = "model" if m.role == "assistant" else "user"
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].append({"text": m.content})
        else:
            contents.append({"role": role, "parts": [{"text": m.content}]})
    return system, contents


class GeminiProvider(HttpProvider):
    default_base_url = "https://generativelanguage.googleapis.com/v1beta"

    def auth_headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key} if self.api_key else {}

    def build_body(self, messages: Sequence[ChatMessage], params: GenParams) -> dict[str, Any]:
        system, contents = to_gemini_contents(messages)
        gen: dict[str, Any] = {}
        mapping = {
            "temperature": "temperature",
            "top_p": "topP",
            "top_k": "topK",
            "max_tokens": "maxOutputTokens",
            "seed": "seed",
            "stop": "stopSequences",
        }
        for ours, theirs in mapping.items():
            val = getattr(params, ours)
            if val is not None:
                gen[theirs] = val
        body: dict[str, Any] = {"contents": contents}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if gen:
            body["generationConfig"] = gen
        return body

    async def stream_chat(
        self, model: str, messages: Sequence[ChatMessage], params: GenParams
    ) -> AsyncIterator[StreamEvent]:
        require_key(self)
        model_path = model if model.startswith("models/") else f"models/{model}"
        url = f"{self.base_url}/{model_path}:streamGenerateContent?alt=sse"
        body = self.build_body(messages, params)
        try:
            async with self.client.stream("POST", url, json=body) as resp:
                await self.raise_for_status(resp)
                usage: Usage | None = None
                finish: str | None = None
                async for sse in iter_sse(resp):
                    try:
                        chunk = json.loads(sse.data)
                    except ValueError:
                        continue
                    if "error" in chunk:
                        raise ProviderError(f"{self.name}: {chunk['error'].get('message')}")
                    if fb := chunk.get("promptFeedback", {}).get("blockReason"):
                        raise ProviderError(f"{self.name}: prompt blocked ({fb})")
                    text = ""
                    for cand in chunk.get("candidates") or []:
                        for part in (cand.get("content") or {}).get("parts") or []:
                            if part.get("thought"):
                                continue  # hidden reasoning
                            text += part.get("text") or ""
                        finish = cand.get("finishReason") or finish
                    if um := chunk.get("usageMetadata"):
                        usage = Usage(um.get("promptTokenCount"), um.get("candidatesTokenCount"))
                    if text:
                        yield StreamEvent(text=text)
                yield StreamEvent(usage=usage, finish_reason=(finish or "STOP").lower())
        except httpx.HTTPError as e:
            raise self.connection_error(e) from e

    async def list_models(self) -> list[str]:
        require_key(self)
        try:
            resp = await self.client.get(
                f"{self.base_url}/models", params={"pageSize": 1000}, timeout=10.0
            )
        except httpx.HTTPError as e:
            raise self.connection_error(e) from e
        await self.raise_for_status(resp)
        out = []
        for m in resp.json().get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                out.append(m["name"].removeprefix("models/"))
        return sorted(out)
