"""Chat orchestration: sessions, sending, streaming, regenerate/undo.

UI-independent. The REPL (or any future front end) drives it and renders the
``StreamEvent``s it yields.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Any

from chatclient.characters import (
    DEFAULT_USER_NAME,
    INITIATIVE_RULES,
    MATURE_RULES,
    Card,
    Character,
    apply_macros,
    builtin_cards,
    card_content,
    card_initiative,
    compile_reminder,
    compile_system_prompt,
    greeting,
    set_card_content,
)
from chatclient.config import Config
from chatclient.core import freshness, refusal
from chatclient.core.context import BuiltContext, StoryPart, build_context
from chatclient.providers.base import (
    ChatMessage,
    GenParams,
    ProviderError,
    StreamEvent,
    ThinkFilter,
    Usage,
)
from chatclient.providers.registry import ModelRef, ProviderRegistry, parse_model_ref
from chatclient.storage import Message, Session, Store
from chatclient.text import MESSAGE_OVERHEAD_TOKENS, estimate_tokens, short_title
from chatclient.timeutil import from_iso, local_day, to_local, utcnow
from chatclient.utility import SAME_AS_CHAT, UTILITY_KEY, utility_spec

if TYPE_CHECKING:
    from chatclient.memory.engine import MemoryEngine

log = logging.getLogger(__name__)

LAST_MODEL_KEY = "last_model"
LAST_SESSION_KEY = "last_session"
DEFAULT_PERSONA_KEY = "default_persona"
REMOVED_BUILTINS_KEY = "removed_builtin_characters"
CONTENT_KEY = "content_mode"
CONTENT_MODES = ("standard", "mature")
FALLBACK_KEY = "refusal_fallback"

CONTINUE_NOTE = (
    "[Continue your previous reply exactly where it stopped. Do not repeat or summarise "
    "what you already wrote, and do not add any preface; write only the continuation.]"
)

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


TIME_GAP = timedelta(hours=6)  # a break long enough for a person to notice


def _duration(d: timedelta) -> str:
    hours = d.total_seconds() / 3600
    if hours < 48:
        return f"{round(hours)} hours"
    days = round(hours / 24)
    if days < 14:
        return f"{days} days"
    if days < 60:
        return f"{round(days / 7)} weeks"
    return f"{round(days / 30)} months"


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
    recalled: int = 0  # memories / quotes / day summaries brought into this turn
    conflicts: int = 0  # contradictions the character was asked to bring up
    refused: bool = False  # the model declined (provider signal or the utility model's call)
    repetition: float = 0.0  # share of the reply recycled from recent replies (0..1)

    @property
    def repeated(self) -> bool:
        return self.repetition >= freshness.REPEAT_THRESHOLD

    @property
    def elapsed(self) -> float:
        return (self.ended or time.perf_counter()) - self.started

    @property
    def time_to_first_token(self) -> float | None:
        return None if self.first_token_at is None else self.first_token_at - self.started


@dataclass
class Retry:
    reason: str  # "refusal" | "repeat"
    model: str  # who writes the new version
    start: Callable[[TurnResult], AsyncIterator[StreamEvent]]


class ChatService:
    def __init__(
        self,
        config: Config,
        store: Store,
        registry: ProviderRegistry,
        memory: MemoryEngine | None = None,
    ):
        self.config = config
        self.store = store
        self.registry = registry
        self.memory = memory
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
        persona = self.default_persona()
        s = self.store.create_session(
            model or self.default_model(),
            title or (char.name if char else ""),
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

    def content_mode(self) -> str:
        """Either "standard" or "mature": the character's own setting (chosen when it was
        created, or with /content in its chat) wins; otherwise /content, then config."""
        char = self.character() if self.session is not None else None
        own = card_content(char.card) if char else None
        if own:
            return own
        saved = self.store.kv_get(CONTENT_KEY)
        return saved if saved in CONTENT_MODES else self.config.roleplay.content

    def set_content_mode(self, mode: str) -> str:
        """In a character chat this sets the character's setting, otherwise the default.
        Returns "character" or "default"."""
        if mode not in CONTENT_MODES:
            raise ChatError(f"content mode must be one of: {', '.join(CONTENT_MODES)}")
        char = self.character() if self.session is not None else None
        if char is not None:
            set_card_content(char.card, mode)
            self.store.update_character(char.id, char.card)
            return "character"
        self.store.kv_set(CONTENT_KEY, mode)
        return "default"

    def effective_system_prompt(self) -> str:
        s = self.require_session()
        char = self.character()
        if char is None:
            prompt = s.system_prompt or self.config.system_prompt
        else:
            persona = self.persona()
            # In character chats, /system (or [roleplay].rules) replaces the narration rules.
            rules = s.system_prompt or self.config.roleplay.rules
            prompt = compile_system_prompt(char.card, persona.card if persona else None, rules)
            if lead := INITIATIVE_RULES.get(card_initiative(char.card)):
                prompt += "\n\n" + apply_macros(lead, char.name, self.user_name())
        if self.content_mode() == "mature":
            mature = apply_macros(MATURE_RULES, char.name if char else "AI", self.user_name())
            prompt = f"{prompt}\n\n{mature}" if prompt.strip() else mature
        return prompt

    def set_private(self, private: bool) -> None:
        """Private: nothing from this session is remembered, indexed or recalled, and
        whatever memory already took from it is dropped."""
        s = self.require_session()
        self.store.update_session(s.id, private=int(private))
        if private:
            self.store.forget_session_memory(s.id)
            if self.memory is not None:
                self.memory.forget_session(s.id)
            char = self.character()
            if self.store.library is not None and char is not None:
                self.store.library.write_memory(self.store, char)  # drop it from the file too
        self.refresh()

    def reminder(self) -> str | None:
        char = self.character()
        if char is None or not self.config.roleplay.reminder:
            return None
        persona = self.persona()
        note = compile_reminder(char.card, persona.card if persona else None)
        if self.content_mode() == "mature":
            # Small models follow this note more than the system prompt.
            note += f" Mature content is on: go where {self.user_name()} takes the story, with no refusing, moralising or changing the subject."
        return note

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

    async def regenerate(
        self, result: TurnResult, model: str | None = None, fresh: bool = False
    ) -> AsyncIterator[StreamEvent]:
        """Produce a new reply to the latest user message (old reply stays as a branch).

        ``model`` ("provider:model") writes this one version with another model, without
        changing the session's model, e.g. to compare how two models handle a scene.
        ``fresh``: the current reply recycled earlier ones; ask for something new.
        """
        if model:
            self.registry.get(parse_model_ref(model).provider)  # fail before touching history
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
        rewrite_of = head.content if fresh and head.role == "assistant" else None
        self.store.update_session(s.id, head_message_id=parent.id)
        self.refresh()
        async for ev in self._generate(parent, result, model=model, rewrite_of=rewrite_of):
            yield ev

    def alternatives(self) -> tuple[list[Message], int]:
        """Versions of the latest reply (from /regen) and which one is shown."""
        s = self.refresh()
        head = self.store.get_message(s.head_message_id) if s.head_message_id else None
        if head is None or head.role != "assistant":
            return [], -1
        sibs = self.store.siblings(head)
        index = next((i for i, m in enumerate(sibs) if m.id == head.id), -1)
        return sibs, index

    def switch_alternative(self, step: int) -> Message | None:
        """Show the previous/next version of the latest reply.

        Returns None when stepping past the newest version: the caller should then
        generate a fresh one (like swiping right in other roleplay apps).
        """
        sibs, index = self.alternatives()
        if index < 0:
            raise ChatError("nothing to switch")
        target = index + step
        if target < 0:
            raise ChatError("this is the first version")
        if target >= len(sibs):
            if (sibs[index].meta or {}).get("greeting"):
                raise ChatError("the opening line has no other versions")
            return None
        self.store.update_session(self.require_session().id, head_message_id=sibs[target].id)
        self.refresh()
        return sibs[target]

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

    # ── editing & continuing ────────────────────────────────────────────────

    def last_user_message(self) -> Message | None:
        for msg in reversed(self.history(limit=4)):
            if msg.role == "user":
                return msg
        return None

    def last_reply(self) -> Message | None:
        s = self.refresh()
        head = self.store.get_message(s.head_message_id) if s.head_message_id else None
        return head if head is not None and head.role == "assistant" else None

    async def edit_and_resend(self, text: str, result: TurnResult) -> AsyncIterator[StreamEvent]:
        """Replace your last message with an edited version and get a fresh reply.
        The original stays in history as a branch."""
        text = text.strip()
        old = self.last_user_message()
        if old is None:
            raise ChatError("no message to edit")
        if not text:
            raise ChatError("empty message")
        new = self.store.add_message(
            old.session_id, old.parent_id, "user", text, meta={"edited_from": old.id}
        )
        self.refresh()
        async for ev in self._generate(new, result):
            yield ev

    def replace_reply(self, text: str) -> Message:
        """Store a hand-edited version of the last reply (as a new alternative)."""
        old = self.last_reply()
        if old is None:
            raise ChatError("no reply to edit")
        if not text.strip():
            raise ChatError("empty message")
        meta = {**(old.meta or {}), "edited_from": old.id}
        new = self.store.add_message(
            old.session_id, old.parent_id, "assistant", text.strip(), model=old.model, meta=meta
        )
        self.refresh()
        return new

    async def continue_reply(self, result: TurnResult) -> AsyncIterator[StreamEvent]:
        """Ask the model to carry on from where its last reply stopped."""
        head = self.last_reply()
        if head is None or (head.meta or {}).get("greeting"):
            raise ChatError("no reply to continue")
        ref = self.model_ref()
        if ref is None:
            raise ChatError("no model selected")
        params = self.gen_params()
        ctx = await self._context(head, ref, params, reminder=False)
        # Not every API allows prefilling the assistant's turn, so ask with a note instead.
        ctx.messages.append(ChatMessage("user", CONTINUE_NOTE))
        before = head.content

        def save(truncated: bool) -> None:
            joiner = "" if not before or before[-1].isspace() else " "
            if result.text[:1] in ",.;:!?)»”" or result.text[:1].isspace():
                joiner = ""
            self.store.update_message_content(head.id, before + joiner + result.text.lstrip())
            result.message = self.store.get_message(head.id)
            self._after_reply()

        async for ev in self._stream(ref, ctx.messages, params, result, save):
            yield ev

    # ── generation ──────────────────────────────────────────────────────────

    async def _context(
        self,
        last: Message,
        ref: ModelRef,
        params: GenParams,
        *,
        reminder: bool = True,
        result: TurnResult | None = None,
        extra_note: str | None = None,
    ) -> BuiltContext:
        budget = self.context_budget(ref, params)
        today = local_day(utcnow(), self.store.tz, self.store.day_boundary)
        note = self.reminder() if reminder else None
        if note is not None and last.role == "user":
            # Changes every turn (built from the last reply), so it isn't parroted back.
            previous = next(
                (m.content for m in reversed(self.store.branch(last.id, limit=4))
                 if m.role == "assistant"),
                "",
            )  # fmt: skip
            fresh = extra_note or freshness.fresh_note(previous, self.user_name())
            note = f"{note} {fresh}".strip()
            if gap := self._time_gap_note(last):
                note = f"{note}\n\n{gap}"
        system = self.effective_system_prompt()
        story: list[StoryPart] = []
        if self.memory is not None:
            branch = self.store.branch(last.id)  # full branch: memory checks what is current
            query = last.content if last.role == "user" else ""
            visible = self._likely_visible(branch, budget, system)
            # refresh: background memory work may have updated the scene since
            mem = await self.memory.prepare(self.refresh(), query, branch, visible)
            story = mem.story
            if mem.note:
                note = f"{note}\n\n{mem.note}" if note else mem.note
            if result is not None:
                result.recalled = mem.recall.count
                result.conflicts = len(mem.conflicts)
            history = branch[-500:]
        else:
            history = self.store.branch(last.id, limit=500)
        return build_context(system, history, budget, today, reminder=note, story=story)

    def _time_gap_note(self, user_msg: Message) -> str:
        """A fact for the character when the user comes back after a while ("it has been
        3 days"); how to react is the model's call. Only on the first message after the
        gap, so it is never repeated."""
        before = self.store.get_message(user_msg.parent_id) if user_msg.parent_id else None
        if before is None or self.character() is None:
            return ""
        try:
            gap = from_iso(user_msg.created_at) - from_iso(before.created_at)
        except ValueError:
            return ""
        if gap < TIME_GAP:
            return ""
        when = to_local(from_iso(before.created_at), self.store.tz)
        return (
            f"Time has passed: the previous message in this conversation was "
            f"{_duration(gap)} ago ({when:%A %d %B, %H:%M}). {self.user_name()} is writing "
            "now; the scene may have moved on."
        )

    async def prefetch_memory(self, draft: str) -> None:
        """Prepare memory for a message that is still being typed, so that pressing
        Enter doesn't wait for it."""
        if self.memory is None or not draft.strip() or draft.lstrip().startswith("/"):
            return
        s = self.refresh()
        ref = self.model_ref()
        if ref is None:
            return
        budget = self.context_budget(ref, self.gen_params())
        branch = self.store.branch(s.head_message_id) if s.head_message_id else []
        visible = self._likely_visible(branch, budget, self.effective_system_prompt())
        await self.memory.prefetch(s, draft, branch, visible)

    def _likely_visible(self, branch: list[Message], budget: int, system: str) -> set[int]:
        """Messages that will be in the prompt verbatim (so recall doesn't repeat them).
        Same newest-first packing as build_context, leaving room for memory blocks."""
        assert self.memory is not None
        cfg = self.memory.cfg
        room = budget - estimate_tokens(system) - cfg.summary_tokens - cfg.recall_tokens - 200
        out: set[int] = set()
        for i, m in enumerate(reversed(branch)):
            room -= estimate_tokens(m.content) + MESSAGE_OVERHEAD_TOKENS
            if i > 0 and room < 0:
                break
            out.add(m.id)
        return out

    def _after_reply(self) -> None:
        if self.memory is not None:
            try:
                self.memory.after_reply(self.refresh())
            except Exception:  # memory upkeep must never break chatting
                log.exception("scheduling memory work failed")

    def _earlier_replies(self, user_msg: Message) -> list[str]:
        """The character's latest replies before this turn, newest first."""
        branch = self.store.branch(user_msg.id, limit=freshness.RECENT_REPLIES * 2 + 1)
        return [m.content for m in reversed(branch) if m.role == "assistant"][
            : freshness.RECENT_REPLIES
        ]

    async def _generate(
        self,
        user_msg: Message,
        result: TurnResult,
        model: str | None = None,
        rewrite_of: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        ref = parse_model_ref(model) if model else self.model_ref()
        if ref is None:
            raise ChatError("no model selected")
        self.registry.get(ref.provider)  # validates a one-off model's provider
        params = self.gen_params()
        earlier = self._earlier_replies(user_msg)
        extra = None
        if rewrite_of is not None:
            # A rewrite after a recycled reply: say so, and loosen sampling a little.
            extra = freshness.rewrite_note(rewrite_of, self.user_name())
            params.temperature = min((params.temperature or 0.8) + 0.2, 1.3)
            params.repetition_penalty = max(params.repetition_penalty or 1.0, 1.15)
        ctx = await self._context(user_msg, ref, params, result=result, extra_note=extra)
        result.context = ctx

        def save(truncated: bool) -> None:
            self._save_reply(user_msg, result, ref, truncated=truncated)
            if truncated:
                self._after_reply()

        async for ev in self._stream(ref, ctx.messages, params, result, save):
            yield ev
        result.repetition = freshness.repetition(result.text, earlier)
        await self._check_refusal(user_msg, result)
        self._after_reply()  # after the check: memory skips a declined reply

    async def _check_refusal(self, user_msg: Message, result: TurnResult) -> None:
        if refusal.stopped_as_refusal(result.finish_reason):
            result.refused = True
        elif self.refusal_fallback() and refusal.worth_checking(result.text):
            # Only worth a model call when there is somewhere to send the turn. A private
            # session's text goes to no model but the one already chatting.
            try:
                ref = self.model_ref() if self.require_session().private else self.utility_ref()
                if ref is None:
                    return
                provider = self.registry.get(ref.provider)
            except Exception:
                return
            result.refused = await refusal.model_says_refused(
                provider, ref.model, user_msg.content, result.text
            )
        if result.refused and result.message is not None:
            meta = {**(result.message.meta or {}), "refused": True}
            self.store.set_message_meta(result.message.id, meta)
            result.message = self.store.get_message(result.message.id)

    def utility_ref(self) -> ModelRef:
        spec = utility_spec(self.store, self.config, self.require_session().model)
        if not spec:
            raise ChatError("no model selected")
        return parse_model_ref(spec)

    def next_attempt(self, result: TurnResult, done: set[str]) -> Retry | None:
        """After a reply: should this turn be written again, and how? (Same rules for
        every front end.) ``done`` collects the reasons already tried this turn."""
        if result.cancelled or not result.text:
            return None
        fallback = self.refusal_fallback()
        if result.refused and fallback and fallback != result.model and "refusal" not in done:
            # The model declined: the fallback writes this turn instead (the declined
            # version stays as an alternative).
            done.add("refusal")
            return Retry("refusal", fallback, lambda r: self.regenerate(r, model=fallback))
        rewrite = self.config.roleplay.rewrite_repeats and self.character() is not None
        if result.repeated and not result.refused and rewrite and "repeat" not in done:
            # Mostly recycled from earlier replies: ask once for a new one. Only in
            # stories: in a plain chat the same question may rightly get the same answer.
            done.add("repeat")
            one_off = result.model if result.model != self.require_session().model else None
            return Retry(
                "repeat",
                result.model,
                lambda r: self.regenerate(r, model=one_off, fresh=True),
            )
        return None

    def utility_model(self) -> str | None:
        """The model doing memory work for this session (see chatclient.utility)."""
        return utility_spec(self.store, self.config, self.require_session().model)

    def set_utility_model(self, spec: str | None) -> None:
        """A model ("provider:model"), "chat" (always the chat model), or None to go
        back to the config file's [models] utility."""
        if spec and spec != SAME_AS_CHAT:
            ref = parse_model_ref(spec)
            self.registry.get(ref.provider)  # validates the provider
            spec = str(ref)
        self.store.kv_set(UTILITY_KEY, spec or "")

    def refusal_fallback(self) -> str | None:
        """The model to rewrite a declined reply with: /fallback, else [models]."""
        saved = self.store.kv_get(FALLBACK_KEY)
        if saved is not None:
            return saved or None
        return self.config.models.refusal_fallback or None

    def set_refusal_fallback(self, spec: str | None) -> None:
        if spec:
            self.registry.get(parse_model_ref(spec).provider)  # validates the provider
            spec = str(parse_model_ref(spec))
        self.store.kv_set(FALLBACK_KEY, spec or "")

    async def _stream(
        self,
        ref: ModelRef,
        messages: list[ChatMessage],
        params: GenParams,
        result: TurnResult,
        save: Callable[[bool], None],
    ) -> AsyncIterator[StreamEvent]:
        """Stream from the provider, hiding reasoning; ``save(truncated)`` stores the text."""
        provider = self.registry.get(ref.provider)
        result.model = str(ref)
        result.started = time.perf_counter()
        log.info("generate model=%s msgs=%d", ref, len(messages))
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
                    async for ev in provider.stream_chat(ref.model, messages, params):
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
                save(True)
            raise
        finally:
            result.ended = time.perf_counter()

        if not parts:
            raise ProviderError(
                f"{ref.provider}: empty reply"
                + (f" (stop reason: {result.finish_reason})" if result.finish_reason else "")
            )
        save(False)

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
