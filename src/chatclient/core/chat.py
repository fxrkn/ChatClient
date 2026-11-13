"""Chat orchestration: sessions, sending, streaming, regenerate/undo.

UI-independent. The REPL (or any future front end) drives it and renders the
``StreamEvent``s it yields.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from chatclient.characters import (
    DEFAULT_USER_NAME,
    Card,
    Character,
    builtin_cards,
    compile_reminder,
    compile_system_prompt,
    greeting,
)
from chatclient.config import Config
from chatclient.core.context import BuiltContext, build_context
from chatclient.providers.base import GenParams, ProviderError, StreamEvent, ThinkFilter, Usage
from chatclient.providers.registry import ModelRef, ProviderRegistry, parse_model_ref
from chatclient.storage import Message, Session, Store
from chatclient.text import short_title
from chatclient.timeutil import local_day, utcnow

log = logging.getLogger(__name__)

LAST_MODEL_KEY = "last_model"
LAST_SESSION_KEY = "last_session"
DEFAULT_PERSONA_KEY = "default_persona"
REMOVED_BUILTINS_KEY = "removed_builtin_characters"

# name → parser for /set
PARAM_TYPES: dict[str, type] = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "min_p": float,
    "max_tokens": int,
    "repetition_penalty": float,
    "seed": int,
    "stop": list,
    "context_tokens": int,
}


class ChatError(Exception):
    """A user-facing problem that is not a provider failure (no model, nothing to undo...)."""


@dataclass
class TurnResult:
    """Filled in while a reply streams; complete once the generator finishes."""

    text: str = ""
    usage: Usage | None = None
    finish_reason: str | None = None
    message: Message | None = None
    model: str = ""
    started: float = field(default_factory=time.perf_counter)
    first_token_at: float | None = None
    ended: float | None = None
    cancelled: bool = False
    context: BuiltContext | None = None

    @property
    def elapsed(self) -> float:
        return (self.ended or time.perf_counter()) - self.started

    @property
    def time_to_first_token(self) -> float | None:
        return None if self.first_token_at is None else self.first_token_at - self.started


class ChatService:
    def __init__(self, config: Config, store: Store, registry: ProviderRegistry):
        self.config = config
        self.store = store
        self.registry = registry
        self.session: Session | None = None

    # ── sessions ────────────────────────────────────────────────────────────

    def require_session(self) -> Session:
        if self.session is None:
            raise ChatError("no active session")
        return self.session

    def refresh(self) -> Session:
        s = self.store.get_session(self.require_session().id)
        if s is None:
            raise ChatError("session no longer exists")
        self.session = s
        return s

    def default_model(self) -> str | None:
        return self.store.kv_get(LAST_MODEL_KEY) or self.config.models.chat or None

    def new_session(
        self, model: str | None = None, title: str = "", character_id: int | None = None
    ) -> Session:
        char = self.store.get_character(character_id)
        s = self.store.create_session(
            model or self.default_model(), title or (char.name if char else "")
        )
        persona = self.default_persona()
        if char or persona:
            self.store.update_session(
                s.id,
                character_id=char.id if char else None,
                persona_id=persona.id if persona else None,
            )
        self.session = self.store.get_session(s.id)
        self.store.kv_set(LAST_SESSION_KEY, str(s.id))
        if char and char.card.first_mes.strip():
            text = greeting(char.card, persona.card if persona else None)
            self.store.add_message(s.id, None, "assistant", text, meta={"greeting": True})
        return self.refresh()

    # ── characters & persona ────────────────────────────────────────────────

    def character(self) -> Character | None:
        return self.store.get_character(self.require_session().character_id)

    def persona(self) -> Character | None:
        return self.store.get_character(self.require_session().persona_id)

    def user_name(self) -> str:
        p = self.persona()
        return p.name if p else DEFAULT_USER_NAME

    def default_persona(self) -> Character | None:
        raw = self.store.kv_get(DEFAULT_PERSONA_KEY)
        return self.store.get_character(int(raw)) if raw and raw.isdigit() else None

    def set_persona(self, card: Card) -> Character:
        """Create or update the user's persona; it becomes the default and applies here."""
        current = self.default_persona()
        if current:
            self.store.update_character(current.id, card)
            persona = self.store.get_character(current.id)
        else:
            persona = self.store.add_character("user", card)
        assert persona is not None
        self.store.kv_set(DEFAULT_PERSONA_KEY, str(persona.id))
        self.store.update_session(self.require_session().id, persona_id=persona.id)
        self.refresh()
        return persona

    def seed_builtin_characters(self) -> int:
        """Sync shipped characters: add missing ones (unless the user deleted them), and
        retire ones no longer shipped: deleted if unused, otherwise kept as the user's own."""
        shipped = builtin_cards()
        have = self.store.builtin_keys()
        for key in have - set(shipped):
            for char in self.store.list_characters("ai"):
                if char.builtin_key != key:
                    continue
                if self.store.sessions_using(char.id):
                    self.store.clear_builtin_key(char.id)
                else:
                    self.store.delete_character(char.id)
        removed = set(json.loads(self.store.kv_get(REMOVED_BUILTINS_KEY) or "[]"))
        added = 0
        for key, card in shipped.items():
            if key not in have and key not in removed:
                self.store.add_character("ai", card, builtin_key=key)
                added += 1
        return added

    def delete_character(self, char: Character) -> None:
        if char.builtin_key:
            removed = set(json.loads(self.store.kv_get(REMOVED_BUILTINS_KEY) or "[]"))
            removed.add(char.builtin_key)
            self.store.kv_set(REMOVED_BUILTINS_KEY, json.dumps(sorted(removed)))
        if self.store.kv_get(DEFAULT_PERSONA_KEY) == str(char.id):
            self.store.kv_set(DEFAULT_PERSONA_KEY, "")
        self.store.delete_character(char.id)
        self.refresh()

    def clear_persona(self) -> None:
        self.store.kv_set(DEFAULT_PERSONA_KEY, "")
        self.store.update_session(self.require_session().id, persona_id=None)
        self.refresh()

    def open_session(self, session_id: int) -> Session:
        s = self.store.get_session(session_id)
        if s is None:
            raise ChatError(f"session {session_id} not found")
        self.session = s
        self.store.kv_set(LAST_SESSION_KEY, str(s.id))
        return s

    def resume_or_create(self) -> Session:
        last = self.store.kv_get(LAST_SESSION_KEY)
        s = self.store.get_session(int(last)) if last and last.isdigit() else None
        if s is None or s.archived:
            s = self.store.latest_session()
        if s is None:
            return self.new_session()
        return self.open_session(s.id)

    def history(self, limit: int | None = None) -> list[Message]:
        s = self.refresh()
        return self.store.branch(s.head_message_id, limit)

    # ── model & settings ────────────────────────────────────────────────────

    def model_ref(self) -> ModelRef | None:
        spec = self.require_session().model
        return parse_model_ref(spec) if spec else None

    def set_model(self, spec: str) -> ModelRef:
        ref = parse_model_ref(spec)
        self.registry.get(ref.provider)  # validates the provider name
        self.store.update_session(self.require_session().id, model=str(ref))
        self.store.kv_set(LAST_MODEL_KEY, str(ref))
        self.refresh()
        return ref

    def gen_params(self) -> GenParams:
        base = GenParams(**self.config.generation.model_dump())
        params = base.merged(self.require_session().params)
        if params.stop is None and self.config.roleplay.stop_at_user_turn and self.character():
            # Small models often continue the script by writing "User: ..." lines.
            params.stop = [f"\n{self.user_name()}:"]
        return params

    def effective_system_prompt(self) -> str:
        s = self.require_session()
        char = self.character()
        if char is None:
            return s.system_prompt or self.config.system_prompt
        persona = self.persona()
        # In character chats, /system (or [roleplay].rules) replaces the narration rules only.
        rules = s.system_prompt or self.config.roleplay.rules
        return compile_system_prompt(char.card, persona.card if persona else None, rules)

    def reminder(self) -> str | None:
        char = self.character()
        if char is None or not self.config.roleplay.reminder:
            return None
        persona = self.persona()
        return compile_reminder(char.card, persona.card if persona else None)

    def set_system_prompt(self, text: str | None) -> None:
        self.store.update_session(self.require_session().id, system_prompt=text or None)
        self.refresh()

    def set_param(self, name: str, raw: str) -> Any:
        """Set (or with raw 'reset' clear) a per-session parameter. Returns the parsed value."""
        if name not in PARAM_TYPES:
            raise ChatError(f"unknown parameter '{name}' (known: {', '.join(PARAM_TYPES)})")
        params = dict(self.require_session().params)
        value: Any
        if raw.strip().lower() in ("reset", "default", "none", ""):
            params.pop(name, None)
            value = None
        else:
            kind = PARAM_TYPES[name]
            try:
                if kind is list:
                    value = [s for s in (p.strip() for p in raw.split(",")) if s]
                else:
                    value = kind(raw)
            except ValueError as e:
                raise ChatError(f"invalid value for {name}: {raw!r}") from e
            params[name] = value
        self.store.update_session(self.require_session().id, params=params)
        self.refresh()
        return value

    def context_budget(self, ref: ModelRef, params: GenParams) -> int:
        session_value = self.require_session().params.get("context_tokens")
        budget = int(
            session_value
            or self.registry.context_tokens(ref.provider)
            or self.config.context.max_tokens
        )
        pc = self.registry.configs.get(ref.provider)
        if pc and pc.type == "ollama" and pc.num_ctx:
            # Ollama's num_ctx covers prompt + reply; leave room for the reply.
            budget = min(budget, pc.num_ctx - (params.max_tokens or 1024))
        return max(budget, 512)

    # ── conversation ────────────────────────────────────────────────────────

    async def send(self, text: str, result: TurnResult) -> AsyncIterator[StreamEvent]:
        """Persist the user's message, then stream the assistant's reply."""
        text = text.strip()
        if not text:
            raise ChatError("empty message")
        s = self.require_session()
        if self.model_ref() is None:
            raise ChatError("no model selected")
        user_msg = self.store.add_message(s.id, s.head_message_id, "user", text)
        if not s.title:
            self.store.update_session(s.id, title=short_title(text))
        self.refresh()
        async for ev in self._generate(user_msg, result):
            yield ev

    async def regenerate(self, result: TurnResult) -> AsyncIterator[StreamEvent]:
        """Produce a new reply to the latest user message (old reply stays as a branch)."""
        s = self.refresh()
        head = self.store.get_message(s.head_message_id) if s.head_message_id else None
        if head is None:
            raise ChatError("nothing to regenerate")
        if head.role == "assistant":
            parent = self.store.get_message(head.parent_id) if head.parent_id else None
        else:
            parent = head  # the last attempt failed, so the user message is the head
        if parent is None or parent.role != "user":
            raise ChatError("nothing to regenerate")
        self.store.update_session(s.id, head_message_id=parent.id)
        self.refresh()
        async for ev in self._generate(parent, result):
            yield ev

    def undo(self) -> int:
        """Step back over the last exchange. Returns the number of messages hidden."""
        s = self.refresh()
        msgs = self.store.branch(s.head_message_id, limit=2)
        if not msgs:
            raise ChatError("nothing to undo")
        hidden = 0
        head = msgs[-1]
        if (head.meta or {}).get("greeting"):
            raise ChatError("nothing to undo")  # the character's opening line stays
        if head.role == "assistant":
            hidden += 1
            head = msgs[-2] if len(msgs) == 2 else None  # type: ignore[assignment]
        if head is not None and head.role == "user":
            hidden += 1
            new_head = head.parent_id
        else:
            new_head = head.id if head else None
        self.store.update_session(s.id, head_message_id=new_head)
        self.refresh()
        return hidden

    async def _generate(self, user_msg: Message, result: TurnResult) -> AsyncIterator[StreamEvent]:
        ref = self.model_ref()
        assert ref is not None
        provider = self.registry.get(ref.provider)
        params = self.gen_params()
        budget = self.context_budget(ref, params)
        history = self.store.branch(user_msg.id, limit=500)
        today = local_day(utcnow(), self.store.tz, self.store.day_boundary)
        ctx = build_context(
            self.effective_system_prompt(), history, budget, today, reminder=self.reminder()
        )
        result.context = ctx
        result.model = str(ref)
        result.started = time.perf_counter()
        log.info(
            "generate model=%s msgs=%d est_tokens=%d dropped=%d",
            ref,
            len(ctx.messages),
            ctx.estimated_tokens,
            ctx.dropped,
        )

        parts: list[str] = []

        def take(text: str) -> None:
            if result.first_token_at is None:
                result.first_token_at = time.perf_counter()
            parts.append(text)
            result.text = "".join(parts)

        try:
            for attempt in range(2):
                think = ThinkFilter()  # hides inline <think>…</think> reasoning
                try:
                    async for ev in provider.stream_chat(ref.model, ctx.messages, params):
                        visible = think.feed(ev.text) if ev.text else ""
                        if visible:
                            take(visible)
                        if ev.usage:
                            result.usage = ev.usage
                        if ev.finish_reason:
                            result.finish_reason = ev.finish_reason
                        if visible or ev.usage or ev.finish_reason:
                            yield StreamEvent(visible, ev.usage, ev.finish_reason)
                    if tail := think.flush():
                        take(tail)
                        yield StreamEvent(tail)
                    break
                except ProviderError as e:
                    # Retry once on transient failures, but never after text was shown.
                    # 429s are not retried here: an immediate retry rarely helps and
                    # spends rate-limited quota (the REPL may switch models instead).
                    if attempt == 0 and e.retryable and e.status != 429 and not parts:
                        log.warning("retrying after transient error: %s", e)
                        await asyncio.sleep(1.0)
                        continue
                    raise
        except asyncio.CancelledError:
            result.cancelled = True
            if parts:
                self._save_reply(user_msg, result, ref, truncated=True)
            raise
        finally:
            result.ended = time.perf_counter()

        if not parts:
            raise ProviderError(
                f"{ref.provider}: empty reply"
                + (f" (stop reason: {result.finish_reason})" if result.finish_reason else "")
            )
        self._save_reply(user_msg, result, ref, truncated=False)

    def _save_reply(
        self, user_msg: Message, result: TurnResult, ref: ModelRef, *, truncated: bool
    ) -> None:
        meta: dict[str, Any] = {}
        if truncated:
            meta["truncated"] = True
        if result.finish_reason:
            meta["finish_reason"] = result.finish_reason
        usage = result.usage or Usage()
        result.message = self.store.add_message(
            user_msg.session_id,
            user_msg.id,
            "assistant",
            result.text,
            model=str(ref),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            meta=meta,
        )
        self.refresh()
