"""Prompt assembly under a token budget.

Today: system prompt + as much recent history as fits, plus an optional in-character
note on the newest turn. This is the seam where summaries and recalled memories will
be inserted later, so callers already go through ``build_context``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from chatclient.providers.base import ChatMessage
from chatclient.storage import Message
from chatclient.text import MESSAGE_OVERHEAD_TOKENS, estimate_tokens


@dataclass
class BuiltContext:
    messages: list[ChatMessage]
    estimated_tokens: int
    included: int  # history messages included
    dropped: int  # older history messages that did not fit


def system_text(base_prompt: str, today: date | None) -> str:
    # Only the date (not the time) so the prompt prefix stays cache-friendly all day.
    if today is None:
        return base_prompt.strip()
    return f"{base_prompt.strip()}\n\nCurrent date: {today.isoformat()} ({today.strftime('%A')})."


def with_reminder(user_text: str, reminder: str) -> str:
    """Attach the hidden story note to the newest user turn.

    It sits *before* the user's words, so they stay the last thing the model reads,
    yet it is far closer to the reply than the system prompt; small models drift out
    of character much less this way. Sent as part of the user turn (not a mid-history
    system message) because many local chat templates reject system messages there.
    """
    return f"[Story note, not written by the user; follow it silently: {reminder}]\n\n{user_text}"


def build_context(
    system_prompt: str,
    history: Sequence[Message],
    budget_tokens: int,
    today: date | None = None,
    reminder: str | None = None,
) -> BuiltContext:
    """Keep the newest messages that fit; the latest message is always included."""
    system = system_text(system_prompt, today)
    used = estimate_tokens(system) + MESSAGE_OVERHEAD_TOKENS
    if reminder:
        used += estimate_tokens(reminder) + MESSAGE_OVERHEAD_TOKENS
    picked: list[Message] = []
    for i, msg in enumerate(reversed(history)):
        cost = estimate_tokens(msg.content) + MESSAGE_OVERHEAD_TOKENS
        if i > 0 and used + cost > budget_tokens:
            break
        picked.append(msg)
        used += cost
    picked.reverse()
    msgs = [ChatMessage("system", system)] if system else []
    msgs += [ChatMessage(m.role, m.content) for m in picked if m.role in ("user", "assistant")]  # type: ignore[arg-type]
    if reminder and msgs and msgs[-1].role == "user":
        msgs[-1] = ChatMessage("user", with_reminder(msgs[-1].content, reminder))
    return BuiltContext(msgs, used, len(picked), len(history) - len(picked))
