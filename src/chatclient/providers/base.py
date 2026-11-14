"""Provider abstraction shared by every LLM backend."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal

import httpx

from chatclient.config import ProviderConfig

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass
class GenParams:
    """Normalised generation parameters. ``None`` means "use the provider default"."""

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    stop: list[str] | None = None
    # A JSON schema the reply must follow (background tasks). Servers that can enforce
    # it (Ollama) do; others ignore it and the reply is parsed leniently as before.
    json_schema: dict[str, Any] | None = None
    # False: answer directly, no hidden reasoning (background tasks with a token budget:
    # a reasoning model would spend it all thinking and return nothing). None: default.
    reasoning: bool | None = None

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def merged(self, overrides: dict[str, object]) -> GenParams:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        for k, v in overrides.items():
            if k in data:
                data[k] = v
        return GenParams(**data)  # type: ignore[arg-type]


@dataclass
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    load_seconds: float | None = None  # local servers: time spent loading the model


class ThinkFilter:
    """Removes ``<think>…</think>`` reasoning blocks from a streamed reply.

    Some local servers put a reasoning model's hidden thoughts inline in the reply.
    Tags may be split across chunks, so a possible partial tag is held back.
    """

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self) -> None:
        self.buf = ""
        self.inside = False
        self.started = False  # any visible text emitted yet

    @staticmethod
    def _partial(s: str, tag: str) -> int:
        for k in range(min(len(tag) - 1, len(s)), 0, -1):
            if s.endswith(tag[:k]):
                return k
        return 0

    def feed(self, text: str) -> str:
        self.buf += text
        out: list[str] = []
        while self.buf:
            if self.inside:
                i = self.buf.find(self.CLOSE)
                if i < 0:
                    keep = self._partial(self.buf, self.CLOSE)
                    self.buf = self.buf[len(self.buf) - keep :] if keep else ""
                    break
                self.buf = self.buf[i + len(self.CLOSE) :]
                self.inside = False
            else:
                i = self.buf.find(self.OPEN)
                if i < 0:
                    keep = self._partial(self.buf, self.OPEN)
                    out.append(self.buf[: len(self.buf) - keep])
                    self.buf = self.buf[len(self.buf) - keep :]
                    break
                out.append(self.buf[:i])
                self.buf = self.buf[i + len(self.OPEN) :]
                self.inside = True
        return self._visible("".join(out))

    def flush(self) -> str:
        rest, self.buf = ("" if self.inside else self.buf), ""
        return self._visible(rest)

    def _visible(self, text: str) -> str:
        if not self.started:
            text = text.lstrip()  # drop blank lines left where a think block was
            if text:
                self.started = True
        return text


@dataclass
class StreamEvent:
    """One piece of a streamed reply: text, and/or final metadata."""

    text: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None


class ProviderError(Exception):
    """A user-presentable provider failure."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class Provider(ABC):
    def __init__(self, name: str, config: ProviderConfig, api_key: str | None = None):
        self.name = name
        self.config = config
        self.api_key = api_key

    @abstractmethod
    def stream_chat(
        self, model: str, messages: Sequence[ChatMessage], params: GenParams
    ) -> AsyncIterator[StreamEvent]:
        """Stream a reply. Cancelling the consuming task aborts the request."""

    @abstractmethod
    async def list_models(self) -> list[str]: ...

    async def verify_key(self) -> None:
        """Raise ProviderError if the API key is rejected. Listing models needs auth
        on most services; providers where it doesn't override this."""
        await self.list_models()

    async def aclose(self) -> None:  # noqa: B027 - optional hook
        pass

    def describe(self) -> str:
        return self.config.base_url or ""


class HttpProvider(Provider):
    """Base for providers that talk HTTP via a shared httpx client."""

    default_base_url: str = ""

    def __init__(self, name: str, config: ProviderConfig, api_key: str | None = None):
        super().__init__(name, config, api_key)
        self.base_url = (config.base_url or self.default_base_url).rstrip("/")
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0, read=self.config.timeout, write=30.0, pool=10.0
                ),
                headers={**self.auth_headers(), **self.config.headers},
            )
        return self._client

    def auth_headers(self) -> dict[str, str]:
        return {}

    def describe(self) -> str:
        return self.base_url

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def raise_for_status(self, resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        body = (await resp.aread()).decode("utf-8", "replace")
        hint = ""
        if resp.status_code == 402:
            hint = "\n(hint: lower the reply limit, e.g. /set max_tokens 2048, or add credits)"
        elif resp.status_code == 429:
            hint = "\n(hint: rate limited — wait a moment and /regen, or pick another model)"
        elif resp.status_code in (401, 403):
            hint = f"\n(hint: the API key is missing or was rejected — /key set {self.name})"
        raise ProviderError(
            f"{self.name}: HTTP {resp.status_code}: {extract_error_message(body)}{hint}",
            status=resp.status_code,
            retryable=resp.status_code in (408, 429, 500, 502, 503, 504),
        )

    def connection_error(self, e: Exception) -> ProviderError:
        if isinstance(e, httpx.ConnectError):
            return ProviderError(
                f"{self.name}: cannot connect to {self.base_url} — is the server running?",
                retryable=True,
            )
        if isinstance(e, httpx.TimeoutException):
            return ProviderError(f"{self.name}: request timed out", retryable=True)
        return ProviderError(f"{self.name}: {type(e).__name__}: {e}", retryable=True)


def extract_error_message(body: str) -> str:
    """Pull a readable message out of the many JSON error shapes providers use."""
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:500] or "(empty response)"
    if isinstance(data, list) and data:
        data = data[0]
    if isinstance(data, dict):
        err = data.get("error", data)
        if isinstance(err, dict):
            msg = err.get("message") or err.get("msg") or err.get("detail")
            if msg:
                return str(msg) + _upstream_detail(err)
        elif isinstance(err, str):
            return err
        if "message" in data:
            return str(data["message"])
    return body.strip()[:500]


def _upstream_detail(err: dict[str, Any]) -> str:
    """Routers like OpenRouter wrap the real upstream error in ``metadata.raw``."""
    meta = err.get("metadata")
    if not isinstance(meta, dict):
        return ""
    raw = meta.get("raw")
    if isinstance(raw, str) and raw.strip():
        inner = extract_error_message(raw)  # raw is often JSON itself
    elif isinstance(raw, dict):
        inner = extract_error_message(json.dumps(raw))
    else:
        inner = ""
    who = meta.get("provider_name")
    if not inner and not who:
        return ""
    inner = " ".join(inner.split())[:300]
    return f" — {who}: {inner}" if who and inner else f" — {who or inner}"


def require_key(provider: Provider) -> str:
    if not provider.api_key:
        env = provider.config.api_key_env
        hint = f"set {env}" if env else "add an API key"
        raise ProviderError(
            f"{provider.name}: no API key ({hint}, or run `chatclient --set-key {provider.name}`)"
        )
    return provider.api_key
