"""The memory engine: background "compiling" of the conversation, and recall per turn.

Principle: **the model understands, the code executes.** No language rules live in
the code (no regexes, word lists or stemming): understanding a message (is it about
the past? which dates? is a claim true?) is the utility model's job, in one structured
"brief" per message, prepared while the user is still typing. The code only stores,
indexes, searches, budgets and compares.

Background pipeline (paused while a reply streams):
  0. index        every message/memory/summary → a vector            (local, no LLM)
  1. memory_pass  facts with stable keys, events, scene; conflicts      (utility LLM)
  2. segment/day/synopsis summaries, reflection insights               (utility LLM)

Per turn:
  candidates  meaning + fuzzy text + names, over memories, raw old messages and
              day summaries                                            (code, ms)
  brief       which candidates matter, dates, claims supported or not (utility LLM;
              usually prepared before Enter is pressed)
  note        what the character gets next to the user's message      (code)
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from chatclient.characters import INITIATIVE_HINTS, apply_macros, card_initiative
from chatclient.config import Config
from chatclient.core.context import StoryPart
from chatclient.memory import tasks
from chatclient.memory.embed import Embedder, VectorIndex, pick_embedder, to_blob
from chatclient.memory.repo import (
    Conflict,
    Job,
    Memory,
    MemoryRepo,
    Scope,
    Summary,
    same_value,
    search_text,
)
from chatclient.memory.tasks import Brief, BriefClaim
from chatclient.providers.base import ProviderError
from chatclient.providers.registry import ModelRef, ProviderRegistry, parse_model_ref
from chatclient.storage import Message, Session, Store
from chatclient.text import MESSAGE_OVERHEAD_TOKENS, estimate_tokens
from chatclient.timeutil import local_day, to_iso, utcnow
from chatclient.utility import utility_spec

log = logging.getLogger(__name__)

SCENE_FACTS = ("location", "time", "relationship")  # sent with every turn; the rest is not
STRUGGLING_AFTER = 3  # failed memory jobs in a row before the user is told
LAST_OK_KEY = "memory_last_ok"  # when a memory job last succeeded
CURSOR_KEY = "memory_cursor:{session}"
REFLECT_KEY = "reflect_cursor:{scope}"
BRIEF_STRUGGLING_AFTER = 2  # failed briefs in a row before the user is told
MAX_ATTEMPTS = 3
MAX_PASS_MESSAGES = 30  # a long backlog is condensed from its most recent part
MAX_LINE_CHARS = 1200  # very long messages are shortened in prompts
QUOTE_CHARS = 240  # raw-message quotes in prompts
INDEX_BATCH = 64
REFLECT_EVERY = 12  # new memories before an insight pass
BRIEF_RECENT = 8  # recent messages the brief sees
DIRECTOR_CARD_CHARS = 700  # per card field, in the brief's view of the character
CANDIDATES = {"memory": 12, "quote": 8, "day": 3}
FALLBACK_PICK = {"memory": 6, "quote": 3, "day": 1}  # when no brief is available
# Semantic hits must be close in absolute terms and relative to the best hit; without a
# floor a nearest neighbour always exists. (MiniLM scores related pairs ~0.35-0.55.)
MIN_SIMILARITY = 0.3
RELATIVE_SIMILARITY = 0.7


def _close_enough(hits: list[tuple[int, float]]) -> list[int]:
    if not hits:
        return []
    floor = max(MIN_SIMILARITY, hits[0][1] * RELATIVE_SIMILARITY)
    return [i for i, s in hits if s >= floor]


def _fuse(lists: Sequence[Sequence[int]]) -> list[int]:
    """Reciprocal-rank fusion: items found by several searches, or ranked high, win."""
    score: dict[int, float] = defaultdict(float)
    for lst in lists:
        for rank, item in enumerate(lst):
            score[item] += 1.0 / (10 + rank)
    return sorted(score, key=lambda i: (-score[i], -i))


class MemoryUnavailable(Exception):
    """No model to do memory work with (nothing configured or selected)."""


@dataclass
class Candidates:
    memories: list[Memory] = field(default_factory=list)
    quotes: list[Message] = field(default_factory=list)
    days: list[Summary] = field(default_factory=list)

    def items(self) -> list[tuple[str, Any]]:
        return (
            [(f"M{m.id}", m) for m in self.memories]
            + [(f"Q{q.id}", q) for q in self.quotes]
            + [(f"D{d.id}", d) for d in self.days]
        )


@dataclass
class Recall:
    memories: list[Memory] = field(default_factory=list)
    quotes: list[Message] = field(default_factory=list)  # raw old messages
    days: list[Summary] = field(default_factory=list)
    brief: Brief | None = None

    @property
    def count(self) -> int:
        return len(self.memories) + len(self.quotes) + len(self.days)


@dataclass
class MemoryContext:
    story: list[StoryPart] = field(default_factory=list)  # for the system prompt
    note: str = ""  # scene, conflicts, recall, fact checks: next to the newest message
    direction: str = ""  # the brief's private direction for this one reply
    recall: Recall = field(default_factory=Recall)
    conflicts: list[Conflict] = field(default_factory=list)


@dataclass
class _Prefetched:
    text: str
    candidates: Candidates
    brief: Brief | None
    failure: str = ""  # why the prefetched brief failed, if it did


class MemoryEngine:
    def __init__(self, config: Config, store: Store, registry: ProviderRegistry):
        self.config = config
        self.cfg = config.memory
        self.store = store
        self.repo = MemoryRepo(store)
        self.registry = registry
        self.embedder: Embedder | None = None
        self.semantic = "off"  # off | waiting | loading | ready | failed
        self.index = VectorIndex()
        self._prefetched: dict[int, _Prefetched] = {}
        self.failures = 0  # memory jobs failed in a row (the model can't do the task)
        self.last_failure = ""
        self.brief_failures = 0  # briefs failed in a row (no direction, no smart recall)
        self.last_brief_error = ""
        self._brief_error_session: int | None = None  # the session it happened in
        self.repo.reset_stale_jobs()

    # ── semantic search setup ───────────────────────────────────────────────

    async def start_semantic(self, models_dir: Path) -> None:
        """Choose the embedding backend. Loading/downloading the model happens later, in
        the first background "index" job, never while the user waits for a reply."""
        ollama = next(
            (
                pc.base_url or "http://localhost:11434"
                for pc in self.registry.configs.values()
                if pc.type == "ollama"
            ),
            None,
        )
        try:
            self.embedder = await pick_embedder(self.cfg.embeddings, models_dir, ollama)
        except Exception:
            log.exception("choosing an embedding backend failed")
            self.embedder = None
        self.semantic = "waiting" if self.embedder else "off"
        if self.embedder:
            self.repo.enqueue(None, "index")

    def _ensure_index(self) -> None:
        if self.embedder and self.index.loaded_model != self.embedder.name:
            self.index.load(self.embedder.name, self.repo.all_vectors(self.embedder.name))

    async def _embed_query(self, text: str) -> list[float] | None:
        if self.embedder is None or self.semantic != "ready" or not text.strip():
            return None
        try:
            return (await self.embedder.embed([text[:2000]]))[0]
        except Exception:
            log.warning("query embedding failed", exc_info=True)
            return None

    # ── helpers ─────────────────────────────────────────────────────────────

    def scope(self, session: Session) -> Scope:
        if self.cfg.scope == "character" and session.character_id is not None:
            return Scope(session.id, session.character_id)
        return Scope(session.id, None)

    def today(self) -> date:
        return local_day(utcnow(), self.store.tz, self.store.day_boundary)

    def names(self, session: Session) -> tuple[str, str]:
        char = self.store.get_character(session.character_id)
        persona = self.store.get_character(session.persona_id)
        return (char.name if char else "Assistant"), (persona.name if persona else "User")

    def _lines(self, session: Session, msgs: Sequence[Message]) -> list[tuple[str, str, str]]:
        char, user = self.names(session)
        out = []
        for m in msgs:
            if (m.meta or {}).get("refused"):
                continue  # a declined reply is not part of the story
            text = (
                m.content if len(m.content) <= MAX_LINE_CHARS else m.content[:MAX_LINE_CHARS] + "…"
            )
            out.append((m.local_day, user if m.role == "user" else char, text))
        return out

    def utility_ref(self, session: Session) -> ModelRef:
        spec = utility_spec(self.store, self.config, session.model)
        if not spec:
            raise MemoryUnavailable("no model for memory work")
        return parse_model_ref(spec)

    def _kv_int(self, key: str) -> int:
        raw = self.store.kv_get(key)
        return int(raw) if raw and raw.isdigit() else 0

    def _valid_summaries(self, session: Session, kind: str, branch_ids: set[int]) -> list[Summary]:
        """Summaries of the active branch only (after /undo or edits, others are stale)."""
        return [
            s
            for s in self.repo.summaries(session.id, kind)
            if s.end_message_id is None or s.end_message_id in branch_ids
        ]

    def _unsummarised(self, session: Session, branch: list[Message]) -> list[Message]:
        """Older messages (outside the recent window) not yet covered by a segment summary."""
        ids = {m.id for m in branch}
        covered = max(
            (s.end_message_id or 0 for s in self._valid_summaries(session, "segment", ids)),
            default=0,
        )
        older = branch[: max(0, len(branch) - self.cfg.keep_recent_messages)]
        return [m for m in older if m.id > covered]

    async def _complete(
        self,
        ref: ModelRef,
        messages: list[Any],
        max_tokens: int = 700,
        schema: dict[str, Any] | None = None,
    ) -> str:
        provider = self.registry.get(ref.provider)
        return await tasks.complete(
            provider, ref.model, messages, max_tokens=max_tokens, schema=schema
        )

    # ── scheduling (after each reply) ───────────────────────────────────────

    def after_reply(self, session: Session) -> list[str]:
        """Queue whatever is due. Cheap: only counts and DB lookups."""
        if not self.cfg.enabled or session.private or session.head_message_id is None:
            return []
        self._prefetched.pop(session.id, None)
        queued: list[str] = []
        if self.embedder and self.repo.enqueue(None, "index"):
            queued.append("index")
        branch = self.store.branch(session.head_message_id)
        ids = {m.id for m in branch}
        cursor = self._kv_int(CURSOR_KEY.format(session=session.id))
        new_replies = sum(1 for m in branch if m.id > cursor and m.role == "assistant")
        if new_replies >= self.cfg.every_exchanges and self.repo.enqueue(session.id, "memory_pass"):
            queued.append("memory_pass")
        pending = self._unsummarised(session, branch)
        pending_tokens = sum(estimate_tokens(m.content) for m in pending)
        if pending_tokens >= self.cfg.summarize_after_tokens and self.repo.enqueue(
            session.id, "segment"
        ):
            queued.append("segment")
        segments = self._valid_summaries(session, "segment", ids)
        synopsis = self._valid_summaries(session, "synopsis", ids)
        since = (synopsis[-1].end_message_id or 0) if synopsis else 0
        newer_segments = sum(1 for s in segments if (s.end_message_id or 0) > since)
        if newer_segments >= 4 and self.repo.enqueue(session.id, "synopsis"):
            queued.append("synopsis")
        have_days = {s.day for s in self.repo.summaries(session.id, "day")}
        for day in self.repo.days_with_messages(
            session.id, before=self.today().isoformat(), limit=5
        ):
            if day not in have_days and self.repo.enqueue(session.id, "day", {"day": day}):
                queued.append("day")
        scope = self.scope(session)
        reflect_key = REFLECT_KEY.format(scope=f"{scope.session_id}:{scope.character_id}")
        newest = self.repo.list_memories(scope, 1)
        due = bool(newest) and newest[0].id - self._kv_int(reflect_key) >= REFLECT_EVERY
        if due and self.repo.enqueue(session.id, "reflect"):
            queued.append("reflect")
        return queued

    # ── running jobs ────────────────────────────────────────────────────────

    async def run_next(self) -> bool:
        """Run one queued job. Returns False when the queue is empty."""
        job = self.repo.next_job()
        if job is None:
            return False
        try:
            await self._run(job)
        except asyncio.CancelledError:
            self.repo.release_job(job.id)  # interrupted by the user: retry later, no penalty
            raise
        except (ProviderError, MemoryUnavailable, ValueError) as e:
            give_up = job.attempts + 1 >= MAX_ATTEMPTS or isinstance(e, MemoryUnavailable)
            delay = to_iso(utcnow() + timedelta(minutes=2 * (job.attempts + 1)))
            log.warning("memory job %s (%s) failed: %s", job.id, job.kind, e)
            self.repo.fail_job(job.id, str(e).split("\n")[0], delay, give_up)
            if job.kind != "index":
                self.failures += 1
                self.last_failure = str(e).split("\n")[0]
        except Exception as e:
            log.exception("memory job %s crashed", job.id)
            if job.kind == "index":
                self.semantic = "failed"
            self.repo.fail_job(job.id, f"{type(e).__name__}: {e}", None, True)
        else:
            self.repo.finish_job(job.id)
            if job.kind != "index":
                self.failures = 0
                self.store.kv_set(LAST_OK_KEY, to_iso(utcnow()))  # older errors are stale
            if job.kind != "index" and self.embedder is not None:
                self.repo.enqueue(None, "index")  # new memories / summaries need vectors too
            if job.kind != "index":
                self.save_snapshot(job.session_id)
        return True

    @property
    def struggling(self) -> bool:
        """Memory work keeps failing: worth telling the user once (they can pick a
        utility model that copes better)."""
        return self.failures >= STRUGGLING_AFTER

    def save_snapshot(self, session_id: int | None) -> None:
        """Refresh the character's memory.jsonl (its folder keeps what memory learned)."""
        lib = self.store.library
        s = self.store.get_session(session_id) if session_id else None
        char = self.store.get_character(s.character_id) if s else None
        if lib is None or char is None or s is None or s.private:
            return
        try:
            lib.write_memory(self.store, char)
        except OSError:
            log.exception("could not write the memory snapshot of %s", char.name)

    async def run_all(self, limit: int = 100) -> int:
        n = 0
        while n < limit and await self.run_next():
            n += 1
        return n

    async def _run(self, job: Job) -> None:
        if job.kind == "index":
            await self._index()
            return
        session = self.store.get_session(job.session_id) if job.session_id else None
        if session is None or session.private or session.head_message_id is None:
            return
        ref = self.utility_ref(session)
        if job.kind == "memory_pass":
            await self._memory_pass(session, ref)
        elif job.kind == "segment":
            await self._segment(session, ref)
        elif job.kind == "synopsis":
            await self._synopsis(session, ref)
        elif job.kind == "day":
            await self._day(session, ref, str(job.payload.get("day", "")))
        elif job.kind == "reflect":
            await self._reflect(session, ref)

    async def _index(self) -> None:
        """Stage 0: vectors for everything new (local model, no LLM)."""
        if self.embedder is None:
            return
        if self.semantic in ("waiting", "failed"):
            self.semantic = "loading"
        self._ensure_index()
        batch = self.repo.unindexed(self.embedder.name, INDEX_BATCH)
        if batch:
            vectors = await self.embedder.embed([text[:2000] for *_, text in batch])
            for (owner_type, owner_id, session_id, _text), vec in zip(batch, vectors, strict=True):
                blob = to_blob(vec)
                self.repo.save_vector(owner_type, owner_id, session_id, self.embedder.name, blob)
                self.index.add(owner_type, owner_id, session_id, blob)
        self.semantic = "ready"
        if len(batch) >= INDEX_BATCH:
            self.repo.enqueue(None, "index")  # more to do: continue in the next job

    async def _memory_pass(self, session: Session, ref: ModelRef) -> None:
        assert session.head_message_id is not None
        branch = self.store.branch(session.head_message_id)
        cursor_key = CURSOR_KEY.format(session=session.id)
        new = [m for m in branch if m.id > self._kv_int(cursor_key)][-MAX_PASS_MESSAGES:]
        if not new:
            return
        scope = self.scope(session)
        char, user = self.names(session)
        prompt = tasks.memory_pass_prompt(
            char, user, session.scene, self.repo.recent_contents(scope),
            self._lines(session, new), self.repo.fact_keys(scope),
        )  # fmt: skip
        raw = await self._complete(ref, prompt, max_tokens=1000, schema=tasks.PASS_SCHEMA)
        if not raw.strip():
            raise ValueError(
                "the model returned no text (a reasoning model may have spent its budget "
                "thinking; a non-reasoning [models] utility avoids that)"
            )
        result = tasks.parse_pass(raw)
        if result is None:
            raise ValueError("the model's memory notes were unreadable")
        last = new[-1]
        for item in result.memories:
            said_by = self._speaker(item.said_by, char, user)
            dup = self.repo.find_duplicate(scope, item.content)
            if dup is not None and same_value(dup.value, item.value):
                self.repo.bump_importance(dup.id, item.importance)
                continue
            memory = self.repo.add_memory(
                scope, item.content, kind=item.kind, importance=item.importance,
                day=last.local_day, story_time=item.story_time, place=item.place,
                people=item.people, source_message_id=last.id, subject=item.subject,
                attribute=item.attribute, value=item.value, said_by=said_by, key=item.key,
            )  # fmt: skip
            old = self.repo.conflicting(scope, memory)
            if old is not None and await self._contradicts(ref, old.content, memory.content):
                # Same fact key, different value: the newer statement wins, the older one
                # is kept as history, and the character will bring it up.
                self.repo.supersede(old.id, memory.id)
                self.repo.add_conflict(session.id, old.id, memory.id)
        if result.scene:
            # The model describes the scene as it is now; what it leaves out has ended
            # (a finished activity, a closed thread). Only the lasting facts are kept
            # when a small model forgets to repeat them.
            new = dict(result.scene)
            for k in SCENE_FACTS:
                if not new.get(k) and session.scene.get(k):
                    new[k] = session.scene[k]
            self.repo.set_scene(session.id, new)
        self.store.kv_set(cursor_key, str(last.id))

    async def _contradicts(self, ref: ModelRef, old: str, new: str) -> bool:
        """Same fact key, different wording: ask the model whether they really clash.
        A false alarm makes the character say "but you told me…" for nothing, so when
        the model can't tell, it is not treated as a contradiction."""
        try:
            raw = await self._complete(
                ref, tasks.contradiction_prompt(old, new), max_tokens=20,
                schema=tasks.CONTRADICTION_SCHEMA,
            )  # fmt: skip
        except ProviderError as e:
            log.info("contradiction check unavailable: %s", e)
            return False
        return tasks.parse_contradiction(raw) is True

    @staticmethod
    def _speaker(raw: str | None, char: str, user: str) -> str | None:
        """Who said it, from whatever the model wrote (a name, "user", "char"…)."""
        if not raw:
            return None
        r = search_text(raw)
        if r == search_text(user) or r in ("user", "kullanici"):
            return "user"
        if r == search_text(char) or r in ("char", "character", "assistant", "ai"):
            return "char"
        return None

    async def _segment(self, session: Session, ref: ModelRef) -> None:
        assert session.head_message_id is not None
        pending = self._unsummarised(session, self.store.branch(session.head_message_id))
        chunk: list[Message] = []
        used = 0
        for m in pending:
            chunk.append(m)
            used += estimate_tokens(m.content)
            if used >= self.cfg.summarize_after_tokens * 2:
                break
        if not chunk:
            return
        char, user = self.names(session)
        text = await self._complete(
            ref, tasks.segment_prompt(char, user, self._lines(session, chunk))
        )
        if not text:
            raise ValueError("empty summary")
        self.repo.add_summary(
            session.id, "segment", text, start_message_id=chunk[0].id, end_message_id=chunk[-1].id
        )
        if len(chunk) < len(pending):  # old chats: work through the backlog chunk by chunk
            self.repo.enqueue(session.id, "segment")

    async def _synopsis(self, session: Session, ref: ModelRef) -> None:
        assert session.head_message_id is not None
        ids = {m.id for m in self.store.branch(session.head_message_id)}
        previous = self._valid_summaries(session, "synopsis", ids)
        since = (previous[-1].end_message_id or 0) if previous else 0
        newer = [
            s
            for s in self._valid_summaries(session, "segment", ids)
            if (s.end_message_id or 0) > since
        ]
        if not newer:
            return
        char, user = self.names(session)
        prev_text = previous[-1].content if previous else ""
        text = await self._complete(
            ref, tasks.synopsis_prompt(char, user, prev_text, [s.content for s in newer])
        )
        if not text:
            raise ValueError("empty synopsis")
        start = previous[-1].start_message_id if previous else newer[0].start_message_id
        self.repo.add_summary(
            session.id, "synopsis", text, start_message_id=start,
            end_message_id=newer[-1].end_message_id,
        )  # fmt: skip

    async def _day(self, session: Session, ref: ModelRef, day: str) -> None:
        rows = self.store.conn.execute(
            "SELECT id FROM messages WHERE session_id = ? AND local_day = ? ORDER BY id",
            (session.id, day),
        ).fetchall()
        msgs = [m for r in rows if (m := self.store.get_message(r[0]))][-80:]
        if len(msgs) < 2:
            return
        char, user = self.names(session)
        text = await self._complete(
            ref, tasks.day_prompt(char, user, day, self._lines(session, msgs))
        )
        if not text:
            raise ValueError("empty day summary")
        self.repo.add_summary(session.id, "day", text, day=day)

    async def _reflect(self, session: Session, ref: ModelRef) -> None:
        """Stage 2: insights beyond single facts ("Ali goes quiet when he's tired")."""
        scope = self.scope(session)
        recent = [
            m for m in self.repo.list_memories(scope, 30) if not m.outdated and m.kind != "insight"
        ]
        if len(recent) < 5:
            return
        char, user = self.names(session)
        text = await self._complete(
            ref, tasks.reflect_prompt(char, user, [m.content for m in recent])
        )
        for line in text.splitlines():
            insight = line.strip(" -•*\t")
            if len(insight) > 12 and self.repo.find_duplicate(scope, insight) is None:
                self.repo.add_memory(
                    scope, insight, kind="insight", importance=0.7, day=self.today().isoformat()
                )
        key = REFLECT_KEY.format(scope=f"{scope.session_id}:{scope.character_id}")
        self.store.kv_set(key, str(recent[0].id))

    # ── candidates (code, every turn, milliseconds) ─────────────────────────

    async def candidates(
        self,
        session: Session,
        text: str,
        branch: list[Message],
        visible: set[int],
        context: str = "",
    ) -> Candidates:
        """Things that may matter for ``text``: found by meaning, fuzzy text and names,
        among memories, raw old messages and day summaries. Messages still on screen,
        and memories from an abandoned branch, are left out."""
        scope = self.scope(session)
        sessions = self.repo.scope_sessions(scope)
        branch_ids = {m.id for m in branch}
        qvec = await self._embed_query(f"{text}\n{context}".strip())
        if qvec is not None:
            self._ensure_index()

        mlists: list[list[int]] = [self.repo.keyword_search(scope, text)]
        norm = search_text(text)
        names = [
            eid
            for eid, _name, n, _kind in self.repo.entities(scope)
            if len(n) >= 3 and re.search(rf"(?<!\w){re.escape(n)}", norm)
        ]
        if names:
            mlists.append(self.repo.memories_for_entities(names))
        qlists: list[list[int]] = [self.repo.message_keyword_search(sessions, text)]
        dlists: list[list[int]] = []
        if qvec is not None:
            owners = self._active_memory_ids(scope)
            mlists.append(
                _close_enough(self.index.search(qvec, owner_type="memory", owners=owners))
            )
            qlists.append(
                _close_enough(
                    self.index.search(
                        qvec, owner_type="message", sessions=sessions, exclude=visible
                    )
                )
            )
            day_ids = {d.id for d in self.repo.day_summaries_in(sessions)}
            dlists.append(
                _close_enough(self.index.search(qvec, owner_type="summary", owners=day_ids))
            )

        memories = []
        for mid in _fuse(mlists):
            m = self.repo.get_memory(mid)
            if m is None or m.outdated:
                continue
            src = m.source_message_id
            if (
                m.session_id == session.id
                and src is not None
                and (src not in branch_ids or src in visible)
            ):
                continue  # from an undone branch, or still on screen anyway
            memories.append(m)
            if len(memories) >= CANDIDATES["memory"]:
                break
        quotes = []
        for qid in _fuse(qlists):
            if qid in visible:
                continue
            msg = self.store.get_message(qid)
            if msg is None or (msg.session_id == session.id and qid not in branch_ids):
                continue
            quotes.append(msg)
            if len(quotes) >= CANDIDATES["quote"]:
                break
        days = [d for i in _fuse(dlists)[: CANDIDATES["day"]] if (d := self.repo.get_summary(i))]
        return Candidates(memories, quotes, days)

    def _active_memory_ids(self, scope: Scope) -> set[int]:
        where, args = scope.where()
        rows = self.store.conn.execute(
            f"SELECT id FROM memories WHERE {where} AND superseded_by IS NULL", args
        ).fetchall()
        return {r[0] for r in rows}

    # ── the brief (utility model, usually before Enter) ─────────────────────

    def _candidate_text(self, session: Session, item: Any) -> str:
        char, user = self.names(session)
        if isinstance(item, Memory):
            tag = " · ".join(x for x in (item.day, item.place, item.story_time) if x)
            return f"({tag}) {item.content}" if tag else item.content
        if isinstance(item, Message):
            who = user if item.role == "user" else char
            body = " ".join(item.content.split())[:QUOTE_CHARS]
            return f'({item.local_day}) {who}: "{body}"'
        return f"({item.day}) {item.content}"

    def _director_view(self, session: Session, content: str) -> dict[str, str]:
        """What the brief needs to also write the direction: who the character is (name,
        description, personality, trimmed), the initiative setting as a hint, the current
        scene and the content mode. Empty without a character."""
        ch = self.store.get_character(session.character_id)
        if ch is None:
            return {}
        char, user = self.names(session)
        card = ch.card
        parts = [
            apply_macros(t.strip()[:DIRECTOR_CARD_CHARS], char, user)
            for t in (card.description, card.personality)
            if t.strip()
        ]
        scene = " · ".join(
            f"{k}: {v}" for k, v in (session.scene or {}).items() if v and k != "threads"
        )
        threads = (session.scene or {}).get("threads")
        if threads:
            scene += " · open: " + "; ".join(str(t) for t in threads)
        hint = INITIATIVE_HINTS.get(card_initiative(card), "")
        return {
            "profile": "\n".join(parts) or char,
            "initiative": apply_macros(hint, char, user),
            "scene": scene,
            "content": content,
        }

    async def make_brief(
        self,
        session: Session,
        text: str,
        recent: Sequence[Message],
        cands: Candidates,
        content: str = "",
    ) -> Brief | None:
        """One brief for a message that is being sent (a failure counts toward the
        warning). Keystroke prefetch uses ``_attempt_brief`` and counts at send time."""
        brief, failure = await self._attempt_brief(session, text, recent, cands, content)
        self._note_brief(session.id, brief, failure, count=True)
        return brief

    def _note_brief(
        self, session_id: int, brief: Brief | None, failure: str, *, count: bool
    ) -> None:
        """Remember how a brief went. ``count`` moves the in-a-row counter (once per
        sent message); the reason is kept either way, so /direction can show it."""
        if failure:
            self.last_brief_error = failure
            self._brief_error_session = session_id
            if count:
                self.brief_failures += 1
        elif brief is not None:
            self.last_brief_error = ""  # an empty direction is not an error
            if count:
                self.brief_failures = 0

    def brief_error_for(self, session_id: int | None) -> str:
        """The last brief failure, only if it happened in this session."""
        return self.last_brief_error if session_id == self._brief_error_session else ""

    async def _attempt_brief(
        self,
        session: Session,
        text: str,
        recent: Sequence[Message],
        cands: Candidates,
        content: str = "",
    ) -> tuple[Brief | None, str]:
        """(brief, "") or (None, why it failed); (None, "") when the brief is off."""
        if not self.cfg.brief:
            return None, ""
        try:
            ref = self.utility_ref(session)
        except MemoryUnavailable as e:
            return None, str(e) or "no model for memory work"
        char, user = self.names(session)
        prompt = tasks.brief_prompt(
            char, user, calendar(self.today()), self._lines(session, recent), text,
            [(cid, self._candidate_text(session, item)) for cid, item in cands.items()],
            **self._director_view(session, content),
        )  # fmt: skip
        try:
            raw = await asyncio.wait_for(
                self._complete(ref, prompt, max_tokens=800, schema=tasks.BRIEF_SCHEMA),
                self.cfg.brief_timeout,
            )
        except (TimeoutError, ProviderError) as e:
            log.info("brief unavailable: %s", e)
            if isinstance(e, TimeoutError):
                return None, f"no answer within {self.cfg.brief_timeout:g}s"
            lines = str(e).strip().splitlines()
            return None, f"{ref.provider}: {lines[0] if lines else type(e).__name__}"
        brief = tasks.parse_brief(raw)
        if brief is None:
            log.info("brief unreadable: %r", raw[:200] or "(no text: the model only reasoned)")
            return None, f"{ref.model}: " + ("unreadable answer" if raw.strip() else "empty answer")
        return brief, ""

    @property
    def brief_struggling(self) -> bool:
        """The brief (and so the direction) keeps failing: the user should be told once."""
        return self.brief_failures >= BRIEF_STRUGGLING_AFTER

    async def prefetch(
        self,
        session: Session,
        text: str,
        branch: list[Message],
        visible: set[int],
        content: str = "",
    ) -> None:
        """Prepare the brief for a draft while the user is still typing."""
        text = text.strip()
        if not self.cfg.enabled or not text:
            return
        if session.private:  # direction only: no search, nothing stored
            cands = Candidates()
        else:
            previous = next((m.content for m in reversed(branch) if m.role == "assistant"), "")
            cands = await self.candidates(session, text, branch, visible, context=previous[-600:])
        brief, failure = await self._attempt_brief(
            session, text, branch[-BRIEF_RECENT:], cands, content
        )
        # drafts come and go: only the reason is kept now; the send counts the failure
        self._note_brief(session.id, brief, failure, count=False)
        self._prefetched[session.id] = _Prefetched(text, cands, brief, failure)

    def forget_session(self, session_id: int) -> None:
        """Drop in-memory state for a session (its vectors leave the search index)."""
        self._prefetched.pop(session_id, None)
        self.index = VectorIndex()  # reloaded from the database on next use

    def prefetched_for(self, session_id: int, text: str) -> _Prefetched | None:
        hit = self._prefetched.get(session_id)
        return hit if hit is not None and hit.text == text.strip() else None

    # ── assembling the turn ─────────────────────────────────────────────────

    async def prepare(
        self,
        session: Session,
        text: str,
        branch: list[Message],
        visible: set[int] | None = None,
        content: str = "",
    ) -> MemoryContext:
        """Everything memory adds to this turn's prompt. ``branch`` ends with the user's
        new message (when there is one).

        A private session gets no memory: no search, no recall from elsewhere, nothing
        stored. It still gets a direction (the brief runs with no candidates)."""
        if not self.cfg.enabled:
            return MemoryContext()
        if session.private:
            if not text:
                return MemoryContext()
            earlier = branch[:-1] if branch and branch[-1].role == "user" else branch
            ready = self.prefetched_for(session.id, text)
            if ready is not None:
                brief = ready.brief
                self._note_brief(session.id, brief, ready.failure, count=True)
            else:
                brief = await self.make_brief(
                    session, text, earlier[-BRIEF_RECENT:], Candidates(), content
                )
            self._prefetched.pop(session.id, None)
            return MemoryContext(direction=brief.direction if brief is not None else "")
        visible = visible or set()
        ids = {m.id for m in branch}
        story = self._story(session, ids)

        rec = Recall()
        brief: Brief | None = None  # a continuation (no text) has no brief
        if text:
            earlier = branch[:-1] if branch and branch[-1].role == "user" else branch
            ready = self.prefetched_for(session.id, text)
            if ready is not None:
                cands, brief = ready.candidates, ready.brief
                self._note_brief(session.id, brief, ready.failure, count=True)
            else:
                previous = next((m.content for m in reversed(earlier) if m.role == "assistant"), "")
                cands = await self.candidates(
                    session, text, branch, visible, context=previous[-600:]
                )
                brief = await self.make_brief(
                    session, text, earlier[-BRIEF_RECENT:], cands, content
                )
            rec = self._select(session, cands, brief, visible)
        self._prefetched.pop(session.id, None)
        self.repo.mark_used([m.id for m in rec.memories])
        conflicts = self.repo.conflicts(session.id, "new")
        for c in conflicts:
            self.repo.set_conflict_status(c.id, "mentioned")  # bring each up once
        return MemoryContext(
            story=story,
            note=self.format_note(session, rec, conflicts),
            direction=brief.direction if brief is not None else "",
            recall=rec,
            conflicts=conflicts,
        )

    def _story(self, session: Session, ids: set[int]) -> list[StoryPart]:
        synopsis = self._valid_summaries(session, "synopsis", ids)
        since = (synopsis[-1].end_message_id or 0) if synopsis else 0
        parts: list[StoryPart] = []
        if synopsis:
            s = synopsis[-1]
            parts.append(StoryPart(s.start_message_id or 0, s.end_message_id or 0, s.content))
        parts += [
            StoryPart(s.start_message_id or 0, s.end_message_id or 0, s.content)
            for s in self._valid_summaries(session, "segment", ids)
            if (s.end_message_id or 0) > since
        ]
        trimmed: list[StoryPart] = []
        used = 0
        for part in reversed(parts):  # keep the newest parts within the budget
            cost = estimate_tokens(part.text) + MESSAGE_OVERHEAD_TOKENS
            if used + cost > self.cfg.summary_tokens:
                break
            trimmed.insert(0, part)
            used += cost
        return trimmed

    def _select(
        self, session: Session, cands: Candidates, brief: Brief | None, visible: set[int]
    ) -> Recall:
        """The brief picks what matters (and adds what its dates point to); without a
        brief, the best-ranked candidates are used."""
        scope = self.scope(session)
        by_id = dict(cands.items())
        if brief is not None:
            picked = [by_id[i] for i in brief.relevant if i in by_id]
            memories = [x for x in picked if isinstance(x, Memory)]
            quotes = [x for x in picked if isinstance(x, Message)]
            days = [x for x in picked if isinstance(x, Summary)]
            dated = sorted({d.isoformat() for a, b in brief.dates for d in _days(a, b)})
            if dated:
                days += [d for d in self.repo.day_summaries(scope, dated) if d not in days]
                seen = {m.id for m in memories}
                for mid in self.repo.memories_on_days(scope, dated, limit=6):
                    if mid not in seen and (m := self.repo.get_memory(mid)):
                        memories.append(m)
                if not days:  # no digest yet for those days: quote the day itself
                    sessions = self.repo.scope_sessions(scope)
                    for qid in self.repo.messages_on_days(sessions, dated, limit=4):
                        if (
                            qid not in visible
                            and (q := self.store.get_message(qid))
                            and q not in quotes
                        ):
                            quotes.append(q)
        else:
            memories = cands.memories[: FALLBACK_PICK["memory"]]
            quotes = cands.quotes[: FALLBACK_PICK["quote"]]
            days = cands.days[: FALLBACK_PICK["day"]]
        # Quote only the user's own words. The character's old prose, quoted back, is
        # copied by small models (same images, same sentences) and feeds a loop; what it
        # said that matters is already in the memories. (The brief still saw those quotes
        # when checking claims.)
        quotes = [q for q in quotes if q.role == "user"]
        pinned = [m for i in self.repo.pinned_ids(scope) if (m := self.repo.get_memory(i))]
        memories = pinned + [m for m in memories if m.id not in {p.id for p in pinned}]

        # fit the budget: pinned and chosen memories first, then day summaries, then quotes
        out = Recall(brief=brief)
        used = 0
        for kind, items in (("m", memories), ("d", days[:3]), ("q", quotes)):
            for item in items:
                cost = estimate_tokens(self._candidate_text(session, item)) + 8
                if used + cost > self.cfg.recall_tokens:
                    break
                used += cost
                {"m": out.memories, "d": out.days, "q": out.quotes}[kind].append(item)
        return out

    def format_note(self, session: Session, rec: Recall, conflicts: Sequence[Conflict] = ()) -> str:
        char, user = self.names(session)
        lines: list[str] = []
        scene = session.scene or {}
        # Only facts that keep the story consistent. Mood, "what is happening" and open
        # threads, repeated next to every message, pull small models back to the same
        # spot (finish the pizza, the ocean…); the recent conversation already shows
        # them, and plans that matter are memories, recalled when relevant.
        bits = [f"{k}: {scene[k]}" for k in SCENE_FACTS if scene.get(k)]
        if bits:
            lines.append("Current scene — " + " · ".join(bits))
        for c in conflicts:
            when = f" ({c.old.day})" if c.old.day else ""
            if c.new.said_by == "char":
                lines.append(
                    f"Consistency: earlier it was established that {c.old.content}{when}, but you "
                    f"({char}) later said: {c.new.content}. If that was a mistake, correct it naturally."
                )
            else:
                lines.append(
                    f"Contradiction: {user} said earlier that {c.old.content}{when}. Now: "
                    f"{c.new.content}. Notice this and ask about it naturally, in character "
                    f'(e.g. "but you told me …").'
                )
        if rec.memories:
            lines.append(f"Things {char} remembers (use naturally, only if relevant):")
            lines += [f"- {self._candidate_text(session, m)}" for m in rec.memories]
        if rec.quotes:
            lines.append("Earlier words, quoted from past conversations:")
            lines += [f"- {self._candidate_text(session, q)}" for q in rec.quotes]
        for d in rec.days:
            lines.append(f"What happened on {d.day}: {d.content}")
        if rec.brief is not None:
            lines += self._claim_lines(rec.brief.claims, rec.brief.about_past)
            # The note corrects the past; on other turns it could only steer the story.
            disputed = any(c.status != "supported" for c in rec.brief.claims)
            if rec.brief.note and rec.brief.about_past and disputed:
                lines.append(f"Memory note: {rec.brief.note}")
        return "\n".join(lines)

    @staticmethod
    def _claim_lines(claims: Sequence[BriefClaim], about_past: bool) -> list[str]:
        # Claims are, by the brief's own definition, about the shared past or known
        # facts, so they count even when the model's about_past flag disagrees (small
        # models get the flag wrong more often than the claim itself; see --eval-memory).
        out = []
        for c in claims:
            if c.status == "contradicted":
                out.append(
                    f'Fact check: "{c.claim}" contradicts what you know. Point it out naturally, '
                    "in character."
                )
            elif c.status == "unsupported":
                out.append(
                    f'Fact check: "{c.claim}" is not in your shared history. Do not invent it or '
                    "agree that it happened; say, in character, that it didn't happen or you "
                    "don't remember it."
                )
        return out

    # ── dates for /day (model, not rules) ───────────────────────────────────

    async def resolve_dates(self, session: Session, expression: str) -> list[str]:
        """ "yesterday", "12 October", "last week" → ISO days, via the utility model."""
        expression = expression.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", expression):
            return [expression]
        ref = self.utility_ref(session)
        raw = await self._complete(ref, tasks.date_prompt(calendar(self.today()), expression), 120)
        data = tasks.parse_json(raw) or {}
        brief = Brief.model_validate({"dates": data.get("dates") or []})
        return sorted({d.isoformat() for a, b in brief.dates for d in _days(a, b)})


def calendar(today: date, days: int = 14) -> str:
    """Today plus the previous two weeks, spelled out. Small models are unreliable at
    date arithmetic ("yesterday" came out one day off in --eval-memory); given the
    calendar as data, they only need to look the day up."""
    lines = [f"{today.isoformat()} {today.strftime('%A')} (today)"]
    for i in range(1, days):
        d = today - timedelta(days=i)
        lines.append(f"{d.isoformat()} {d.strftime('%A')}" + (" (yesterday)" if i == 1 else ""))
    return "\n".join(lines)


def _days(start: str, end: str, limit: int = 31) -> list[date]:
    try:
        a, b = date.fromisoformat(start), date.fromisoformat(end)
    except ValueError:
        return []
    n = min((b - a).days, limit - 1)
    return [a + timedelta(days=i) for i in range(max(0, n) + 1)]
