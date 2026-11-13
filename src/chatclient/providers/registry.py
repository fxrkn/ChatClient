"""Provider registry: builds providers from config and resolves "provider:model"."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from chatclient.config import Config, ProviderConfig, resolve_api_key
from chatclient.providers.base import Provider, ProviderError

log = logging.getLogger(__name__)

ECHO = "echo"


@dataclass(frozen=True)
class ModelRef:
    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def parse_model_ref(spec: str) -> ModelRef:
    """Split on the first colon only: "ollama:llama3.1:8b" → ("ollama", "llama3.1:8b")."""
    provider, sep, model = spec.strip().partition(":")
    if not sep or not provider or not model:
        raise ValueError(f"expected 'provider:model', got {spec!r}")
    return ModelRef(provider, model)


def _build(name: str, pc: ProviderConfig) -> Provider:
    key = resolve_api_key(name, pc) if pc.type != "echo" else None
    match pc.type:
        case "openai_compat":
            from chatclient.providers.openai_compat import OpenAICompatProvider

            return OpenAICompatProvider(name, pc, key)
        case "ollama":
            from chatclient.providers.ollama import OllamaProvider

            return OllamaProvider(name, pc, key)
        case "anthropic":
            from chatclient.providers.anthropic import AnthropicProvider

            return AnthropicProvider(name, pc, key)
        case "gemini":
            from chatclient.providers.gemini import GeminiProvider

            return GeminiProvider(name, pc, key)
        case "echo":
            from chatclient.providers.echo import EchoProvider

            return EchoProvider(name, pc, None)
    raise ValueError(f"unknown provider type {pc.type!r}")


def is_local(pc: ProviderConfig) -> bool:
    if pc.type == "ollama" and not pc.base_url:
        return True
    host = urlparse(pc.base_url or "").hostname or ""
    return host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local")


class ProviderRegistry:
    def __init__(self, config: Config):
        self.configs: dict[str, ProviderConfig] = {
            **config.providers,
            ECHO: ProviderConfig(type="echo"),
        }
        self._instances: dict[str, Provider] = {}

    def names(self) -> list[str]:
        return list(self.configs)

    def get(self, name: str) -> Provider:
        if name not in self.configs:
            known = ", ".join(self.configs)
            raise ProviderError(f"unknown provider '{name}' (configured: {known})")
        if name not in self._instances:
            self._instances[name] = _build(name, self.configs[name])
        return self._instances[name]

    def context_tokens(self, provider: str) -> int | None:
        pc = self.configs.get(provider)
        return pc.context_tokens if pc else None

    async def list_models(
        self, names: list[str] | None = None, timeout: float = 8.0
    ) -> dict[str, list[str] | ProviderError]:
        """List models for several providers concurrently; failures are returned, not raised."""
        names = names or self.names()

        async def one(n: str) -> list[str] | ProviderError:
            try:
                return await asyncio.wait_for(self.get(n).list_models(), timeout)
            except ProviderError as e:
                return e
            except TimeoutError:
                return ProviderError(f"{n}: timed out")
            except Exception as e:
                log.exception("listing models for %s failed", n)
                return ProviderError(f"{n}: {type(e).__name__}: {e}")

        results = await asyncio.gather(*(one(n) for n in names))
        return dict(zip(names, results, strict=True))

    async def autodetect(self) -> ModelRef | None:
        """Find a running local server with at least one model (Ollama preferred)."""
        local = [n for n, pc in self.configs.items() if pc.type != "echo" and is_local(pc)]
        local.sort(key=lambda n: self.configs[n].type != "ollama")
        found = await self.list_models(local, timeout=2.5)
        for n in local:
            models = found[n]
            if isinstance(models, list) and models:
                chat_models = [m for m in models if "embed" not in m.lower()] or models
                return ModelRef(n, chat_models[0])
        return None

    async def aclose(self) -> None:
        for p in self._instances.values():
            try:
                await p.aclose()
            except Exception:
                log.debug("closing %s failed", p.name, exc_info=True)
        self._instances.clear()
