"""Which model does the background work (memory, summaries, refusal checks)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chatclient.config import Config
    from chatclient.storage import Store

UTILITY_KEY = "utility_model"
SAME_AS_CHAT = "chat"


def utility_spec(store: Store, config: Config, chat_model: str | None) -> str | None:
    """/utility (saved in the app) wins over ``[models] utility``; else the chat model.
    ``/utility chat`` means "the chat model", even if the config names another."""
    saved = store.kv_get(UTILITY_KEY)
    if saved == SAME_AS_CHAT:
        return chat_model
    return saved or config.models.utility or chat_model
