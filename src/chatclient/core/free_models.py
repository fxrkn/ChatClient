"""Find working free models on OpenRouter and fall back between them.

Free models are often rate-limited upstream, and OpenRouter caps free usage per day
(about 50 requests/day for accounts with under $10 of credits). So probing stops
as soon as a few working models are found, instead of testing every model.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from chatclient.providers.base import ChatMessage, GenParams, Provider, ProviderError
from chatclient.providers.registry import ProviderRegistry
from chatclient.storage import Store
from chatclient.timeutil import to_iso, utcnow

log = logging.getLogger(__name__)

FREE_ROUTER = "openrouter/free"  # OpenRouter's own router across free models
CACHE_KEY = "free_models:{provider}"
PROBE_PROMPT = [ChatMessage("user", "Reply with the single word: OK")]


def is_free_model(model: str) -> bool:
    return model.endswith(":free") or model == FREE_ROUTER


def find_openrouter(registry: ProviderRegistry) -> str | None:
    for name, pc in registry.configs.items():
        if pc.type == "openai_compat" and "openrouter.ai" in (pc.base_url or ""):
            return name
    return None


@dataclass
class ProbeResult:
    model: str
    ok: bool
    latency: float | None = None  # seconds to first streamed event
    error: str | None = None


def _rank_key(entry: dict[str, Any]) -> tuple[int, int]:
    # Newer and longer-context models first: a rough proxy for quality.
    return (-int(entry.get("created") or 0), -int(entry.get("context_length") or 0))


async def free_candidates(provider: Provider) -> list[str]:
    list_info = getattr(provider, "list_model_info", None)
    if list_info is None:
        return []
    entries = [e for e in await list_info() if str(e["id"]).endswith(":free")]
    entries.sort(key=_rank_key)
    return [str(e["id"]) for e in entries]


async def probe(provider: Provider, model: str, timeout: float = 25.0) -> ProbeResult:
    """Send a tiny request; success = the stream starts without an error."""
    start = time.perf_counter()

    async def run() -> float:
        first: float | None = None
        async for _ in provider.stream_chat(model, PROBE_PROMPT, GenParams(max_tokens=16)):
            if first is None:
                first = time.perf_counter() - start
        return first if first is not None else time.perf_counter() - start

    try:
        latency = await asyncio.wait_for(run(), timeout)
        return ProbeResult(model, True, latency)
    except TimeoutError:
        return ProbeResult(model, False, error="timeout")
    except ProviderError as e:
        msg = str(e).split("\n")[0]
        if e.status:
            msg = f"HTTP {e.status}: " + msg.split(": ", 2)[-1]
        return ProbeResult(model, False, error=msg[:160])


async def find_working(
    provider: Provider,
    models: Sequence[str],
    *,
    want: int = 3,
    max_probes: int = 8,
    concurrency: int = 4,
    on_result: Callable[[ProbeResult], None] | None = None,
) -> list[ProbeResult]:
    """Probe in small batches until ``want`` models work or ``max_probes`` are spent."""
    results: list[ProbeResult] = []
    queue = list(models)[:max_probes]
    while queue and sum(r.ok for r in results) < want:
        batch, queue = queue[:concurrency], queue[concurrency:]
        for r in await asyncio.gather(*(probe(provider, m) for m in batch)):
            results.append(r)
            if on_result:
                on_result(r)
    return results


def save_working(store: Store, provider: str, results: Sequence[ProbeResult]) -> list[str]:
    ok = sorted((r for r in results if r.ok), key=lambda r: r.latency or 999)
    payload = {"checked_at": to_iso(utcnow()), "models": [asdict(r) for r in ok]}
    store.kv_set(CACHE_KEY.format(provider=provider), json.dumps(payload))
    return [r.model for r in ok]


def drop_working(store: Store, provider: str, model: str) -> None:
    key = CACHE_KEY.format(provider=provider)
    raw = store.kv_get(key)
    if not raw:
        return
    try:
        payload = json.loads(raw)
        payload["models"] = [m for m in payload["models"] if m.get("model") != model]
        store.kv_set(key, json.dumps(payload))
    except (ValueError, KeyError, TypeError, AttributeError):
        log.debug("free model cache unreadable; leaving it")


def load_working(store: Store, provider: str) -> list[str]:
    raw = store.kv_get(CACHE_KEY.format(provider=provider))
    if not raw:
        return []
    try:
        return [m["model"] for m in json.loads(raw)["models"]]
    except (ValueError, KeyError, TypeError):
        return []
