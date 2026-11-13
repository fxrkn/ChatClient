"""Minimal Server-Sent Events parser over an httpx streaming response."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx


@dataclass
class SSEEvent:
    event: str
    data: str


async def iter_sse(resp: httpx.Response) -> AsyncIterator[SSEEvent]:
    """Yield events; multi-line ``data:`` fields are joined with newlines."""
    event = ""
    data: list[str] = []
    async for line in resp.aiter_lines():
        if line == "":
            if data:
                yield SSEEvent(event or "message", "\n".join(data))
            event, data = "", []
            continue
        if line.startswith(":"):
            continue  # comment / keep-alive (e.g. OpenRouter ": PROCESSING")
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "data":
            data.append(value)
        elif field == "event":
            event = value
    if data:
        yield SSEEvent(event or "message", "\n".join(data))
