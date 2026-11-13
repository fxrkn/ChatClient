"""Prompt assembly under a token budget.

Layout of every request:
  system:  rules + character card + persona            (stable → cache-friendly)
           "Story so far" summaries of older messages  (changes only occasionally)
           current date
  history: as many recent messages as fit, verbatim
  newest user turn: a hidden note (character reminder, scene, recalled memories)
           followed by the user's own words
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from chatclient.providers.base import ChatMessage
from chatclient.storage import Message
from chatclient.text import MESSAGE_OVERHEAD_TOKENS, estimate_tokens


@dataclass(frozen=True)
class StoryPart:
    """A summary covering messages ``start_id``..``end_id``."""

    start_id: int
    end_id: int
    text: str


@dataclass
class BuiltContext:
    messages: list[ChatMessage]
    estimated_tokens: int
    included: int  # history messages included
    dropped: int  # older history messages that did not fit
    story_parts: int = 0  # summaries standing in for older messages


def system_text(base_prompt: str, today: date | None, story: Sequence[StoryPart] = ()) -> str:
    text = base_prompt.strip()
    if story:
        text += "\n\n## Story so far (summary of earlier conversation)\n" + "\n\n".join(
            p.text.strip() for p in story
        )
    # Only the date (not the time) so the prompt prefix stays cache-friendly all day.
    if today is not None:
        text += f"\n\nCurrent date: {today.isoformat()} ({today.strftime('%A')})."
    return text


def with_reminder(user_text: str, reminder: str) -> str:
    """Attach the hidden note to the newest user turn.

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
    story: Sequence[StoryPart] = (),
) -> BuiltContext:
    """Keep the newest messages that fit; the latest message is always included.

    ``story`` summaries are only used for messages that did *not* fit verbatim, so the
    model never reads the same events twice."""
    base_cost = estimate_tokens(system_text(system_prompt, today)) + MESSAGE_OVERHEAD_TOKENS
    story_cost = sum(estimate_tokens(p.text) + 2 for p in story)
    used = base_cost + story_cost
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

    first_visible = picked[0].id if picked else 0
    kept_story = [p for p in story if p.end_id < first_visible]
    used -= story_cost - sum(estimate_tokens(p.text) + 2 for p in kept_story)

    system = system_text(system_prompt, today, kept_story)
    msgs = [ChatMessage("system", system)] if system else []
    msgs += [ChatMessage(m.role, m.content) for m in picked if m.role in ("user", "assistant")]  # type: ignore[arg-type]
    if reminder and msgs and msgs[-1].role == "user":
        msgs[-1] = ChatMessage("user", with_reminder(msgs[-1].content, reminder))
    return BuiltContext(msgs, used, len(picked), len(history) - len(picked), len(kept_story))
