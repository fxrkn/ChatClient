"""Noticing a declined reply, so the app can hand the turn to another model.

Two signals, neither of them word lists:
- the provider says so (a stop reason such as "refusal", "content_filter", "safety");
- otherwise, for short replies only (long ones are rarely a refusal), the utility
  model is asked whether the reply declined. The model judges; the code only acts.
"""

from __future__ import annotations

import asyncio
import logging

from chatclient.memory import tasks
from chatclient.providers.base import ChatMessage, ProviderError
from chatclient.text import estimate_tokens

log = logging.getLogger(__name__)

# Stop reasons providers use when they block or decline a reply (lower-cased).
REFUSAL_STOPS = {"refusal", "content_filter", "safety", "prohibited_content", "blocklist", "spii"}
SHORT_REPLY_TOKENS = 200  # longer replies are not checked (a refusal is almost always short)
CHECK_TIMEOUT = 15.0


def stopped_as_refusal(finish_reason: str | None) -> bool:
    return (finish_reason or "").lower() in REFUSAL_STOPS


def worth_checking(reply: str) -> bool:
    return bool(reply.strip()) and estimate_tokens(reply) <= SHORT_REPLY_TOKENS


def refusal_prompt(user_text: str, reply: str) -> list[ChatMessage]:
    return [
        ChatMessage(
            "user",
            "An AI wrote the reply below in a chat or a piece of collaborative fiction.\n"
            "Did it decline what the user wrote or asked, instead of going along with it? "
            "That includes apologising that it can't, stepping out of the story to object, "
            "or lecturing/redirecting instead of continuing. A character who refuses "
            "something *inside* the story is NOT a decline.\n\n"
            f'User:\n"""{user_text[-1500:]}"""\n\nReply:\n"""{reply[:1500]}"""\n\n'
            "Answer with one word: yes or no.",
        )
    ]


def parse_verdict(text: str) -> bool | None:
    words = text.strip().strip("*\"'`.").lower().split()
    if not words:
        return None
    first = words[0].strip(".,!:;*\"'`")
    if first in ("yes", "evet", "true"):
        return True
    if first in ("no", "hayır", "hayir", "false"):
        return False
    return None


async def model_says_refused(provider, model: str, user_text: str, reply: str) -> bool:
    """Ask a model; on any problem assume it was not a refusal (never block chatting)."""
    try:
        raw = await asyncio.wait_for(
            tasks.complete(provider, model, refusal_prompt(user_text, reply), max_tokens=5),
            CHECK_TIMEOUT,
        )
    except (TimeoutError, ProviderError) as e:
        log.info("refusal check unavailable: %s", e)
        return False
    return parse_verdict(raw) is True
