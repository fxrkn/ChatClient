"""Database access for the memory layer (tables from storage migration 3)."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from chatclient.storage import Store
from chatclient.text import normalize_for_search
from chatclient.timeutil import to_iso, utcnow

KINDS = ("event", "fact", "preference", "relationship", "plan", "place", "insight")
MOMENTS = ("event", "insight")  # descriptions of a moment, never a one-value fact

# Lower runs first: free local work, then the memory pass, then rarer summaries.
_INDEXED = (("memory", "memories"), ("summary", "summaries"), ("message", "messages"))


def _unindexed_where(owner_type: str) -> str:
    """Rows (alias m) still needing a vector; params: owner_type, model. Private sessions
    and declined replies are never indexed."""
    sql = (
        "LEFT JOIN embeddings e ON e.owner_type = ? AND e.owner_id = m.id AND e.model = ? "
        "WHERE e.owner_id IS NULL "
        "AND COALESCE((SELECT s.private FROM sessions s WHERE s.id = m.session_id), 0) = 0"
    )
    if owner_type == "message":
        sql += " AND COALESCE(m.meta_json, '') NOT LIKE '%\"refused\"%'"
    return sql


JOB_PRIORITY = {"index": 0, "memory_pass": 1, "segment": 2, "day": 2, "synopsis": 3, "reflect": 4}


@dataclass
class Memory:
    id: int
    session_id: int | None
    character_id: int | None
    kind: str
    content: str
    importance: float
    pinned: bool
    day: str | None
    story_time: str | None
    place: str | None
    people: list[str] = field(default_factory=list)
    source_message_id: int | None = None
    created_at: str = ""
    use_count: int = 0
    # structured fact (for contradiction checks): subject / attribute = value
    subject: str | None = None
    attribute: str | None = None
    value: str | None = None
    said_by: str | None = None  # "user" | "char"
    superseded_by: int | None = None  # a newer memory replaced this one
    slot: str | None = None  # normalised fact key ("ali cat name")

    @property
    def outdated(self) -> bool:
        return self.superseded_by is not None


@dataclass
class Conflict:
    id: int
    session_id: int
    old: Memory
    new: Memory


@dataclass
class Summary:
    id: int
    session_id: int
    kind: str  # segment | synopsis | day
    start_message_id: int | None
    end_message_id: int | None
    day: str | None
    content: str
    updated_at: str


@dataclass
class Job:
    id: int
    session_id: int | None
    kind: str
    payload: dict[str, Any]
    attempts: int


@dataclass(frozen=True)
class Scope:
    """Whose memories: one session, or everything with one character (all sessions)."""

    session_id: int
    character_id: int | None = None

    def where(self, alias: str = "") -> tuple[str, tuple[Any, ...]]:
        p = f"{alias}." if alias else ""
        if self.character_id is not None:
            return f"{p}character_id = ?", (self.character_id,)
        return f"{p}session_id = ?", (self.session_id,)


def _memory(r: sqlite3.Row) -> Memory:
    return Memory(
        id=r["id"],
        session_id=r["session_id"],
        character_id=r["character_id"],
        kind=r["kind"],
        content=r["content"],
        importance=r["importance"],
        pinned=bool(r["pinned"]),
        day=r["day"],
        story_time=r["story_time"],
        place=r["place"],
        people=json.loads(r["people_json"] or "[]"),
        source_message_id=r["source_message_id"],
        created_at=r["created_at"],
        use_count=r["use_count"],
        subject=r["subject"],
        attribute=r["attribute"],
        value=r["value"],
        said_by=r["said_by"],
        superseded_by=r["superseded_by"],
        slot=r["slot"],
    )


def _summary(r: sqlite3.Row) -> Summary:
    return Summary(
        r["id"], r["session_id"], r["kind"], r["start_message_id"], r["end_message_id"],
        r["day"], r["content"], r["updated_at"],
    )  # fmt: skip


_WORD = re.compile(r"\w+", re.UNICODE)


def search_text(text: str) -> str:
    """Normalised for matching; apostrophes joined ("Bodrum'da" → "bodrumda")."""
    return normalize_for_search(text).replace("'", "")


def word_set(text: str) -> set[str]:
    return {w for w in _WORD.findall(search_text(text)) if len(w) > 2}


def normalize_key(key: str | None) -> str | None:
    """A fact key chosen by the model ("Ali.cat.name") in comparable form. The model is
    shown the keys that already exist and reuses them, so no language rules are needed
    to see that two facts are about the same thing."""
    if not key or not key.strip():
        return None
    return " ".join(search_text(key).replace(".", " ").replace("_", " ").split()) or None


def same_value(a: str | None, b: str | None) -> bool:
    if not a or not b:
        return True  # unknown: can't call it a contradiction
    na, nb = search_text(a).strip(" ."), search_text(b).strip(" .")
    return na == nb or na in nb or nb in na or similarity(a, b) >= 0.5


def _grams(word: str) -> list[str]:
    return [word[i : i + 3] for i in range(len(word) - 2)]


def shares_a_word_start(query: str, doc: str) -> bool:
    """True if the beginning of some query word appears in ``doc``.

    Word endings vary (suffixes, plurals, tenses) while beginnings carry the meaning,
    so a hit must share at least 2 of the first 4 letter-triples of a query word:
    "haritayı" ~ "harita", "kedimin" ~ "kedisinin", but "hatırlıyor" ≁ "çalışıyor"
    (they only share their ending). Generic: no word lists, no language rules."""
    doc_grams = {g for w in _WORD.findall(search_text(doc)) for g in _grams(w)}
    for word in _WORD.findall(search_text(query)):
        head = _grams(word)[:4]
        if len(head) >= 2 and sum(g in doc_grams for g in head) >= max(2, len(head) // 2):
            return True
    return False


def trigram_query(text: str, limit: int = 64) -> str | None:
    """A fuzzy full-text query: every 3-letter piece of every word, OR-ed together.

    Ranking (bm25) does the rest: a stored "harita" shares har/ari/rit/ita with the
    query word "haritayı", so it ranks high, while pieces common to many texts count
    for little. Works the same for any language; nothing to maintain."""
    grams: list[str] = []
    for word in _WORD.findall(search_text(text)):
        for i in range(len(word) - 2):
            g = word[i : i + 3]
            if g not in grams:
                grams.append(g)
    if not grams:
        return None
    return " OR ".join('"' + g.replace('"', '""') + '"' for g in grams[:limit])


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of words: cheap duplicate detection for memories."""
    wa, wb = word_set(a), word_set(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


class MemoryRepo:
    def __init__(self, store: Store):
        self.store = store
        self.conn = store.conn

    # ── scene ───────────────────────────────────────────────────────────────

    def set_scene(self, session_id: int, scene: dict[str, Any]) -> None:
        self.store.set_scene(session_id, scene)

    # ── memories ────────────────────────────────────────────────────────────

    def add_memory(
        self,
        scope: Scope,
        content: str,
        *,
        kind: str = "fact",
        importance: float = 0.5,
        pinned: bool = False,
        day: str | None = None,
        story_time: str | None = None,
        place: str | None = None,
        people: list[str] | None = None,
        source_message_id: int | None = None,
        subject: str | None = None,
        attribute: str | None = None,
        value: str | None = None,
        said_by: str | None = None,
        key: str | None = None,
    ) -> Memory:
        people = [p for p in (people or []) if p]
        slot = normalize_key(key) or (
            normalize_key(f"{subject} {attribute}") if subject and attribute else None
        )
        with self.store.tx():
            cur = self.conn.execute(
                """INSERT INTO memories(session_id, character_id, kind, content, importance,
                   pinned, day, story_time, place, people_json, source_message_id, created_at,
                   subject, attribute, value, slot, said_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    scope.session_id, scope.character_id, kind if kind in KINDS else "fact",
                    content.strip(), max(0.0, min(1.0, importance)), int(pinned), day,
                    story_time or None, place or None, json.dumps(people, ensure_ascii=False),
                    source_message_id, to_iso(utcnow()), subject or None, attribute or None,
                    value or None, slot,
                    said_by if said_by in ("user", "char") else None,
                ),
            )  # fmt: skip
            mid = cur.lastrowid or 0
            self.conn.execute(
                "INSERT INTO memories_fts(rowid, norm_text) VALUES (?, ?)",
                (mid, search_text(" ".join([content, place or "", *people]))),
            )
            for name in people:
                self._link(mid, self.upsert_entity(scope, name, "person"))
            if place:
                self._link(mid, self.upsert_entity(scope, place, "place"))
        memory = self.get_memory(mid)
        assert memory is not None
        return memory

    def get_memory(self, memory_id: int) -> Memory | None:
        row = self.conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return _memory(row) if row else None

    def list_memories(self, scope: Scope, limit: int = 50, query: str = "") -> list[Memory]:
        where, args = scope.where("m")
        if query.strip():
            ids = self.keyword_search(scope, query, limit)
            return [m for i in ids if (m := self.get_memory(i))]
        rows = self.conn.execute(
            f"SELECT m.* FROM memories m WHERE {where} ORDER BY m.pinned DESC, m.id DESC LIMIT ?",
            (*args, limit),
        ).fetchall()
        return [_memory(r) for r in rows]

    def count_memories(self, scope: Scope) -> int:
        where, args = scope.where()
        return int(
            self.conn.execute(f"SELECT COUNT(*) FROM memories WHERE {where}", args).fetchone()[0]
        )

    def recent_contents(self, scope: Scope, limit: int = 15) -> list[str]:
        return [m.content for m in self.list_memories(scope, limit)]

    def find_duplicate(self, scope: Scope, content: str, threshold: float = 0.6) -> Memory | None:
        for mid in self.keyword_search(scope, content, 10):
            m = self.get_memory(mid)
            if m and similarity(m.content, content) >= threshold:
                return m
        return None

    def set_pinned(self, memory_id: int, pinned: bool) -> bool:
        cur = self.conn.execute(
            "UPDATE memories SET pinned = ? WHERE id = ?", (int(pinned), memory_id)
        )
        return cur.rowcount > 0

    def bump_importance(self, memory_id: int, importance: float) -> None:
        self.conn.execute(
            "UPDATE memories SET importance = MAX(importance, ?) WHERE id = ?",
            (importance, memory_id),
        )

    def delete_memory(self, memory_id: int) -> bool:
        with self.store.tx():
            self.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (memory_id,))
            self.conn.execute(
                "DELETE FROM embeddings WHERE owner_type = 'memory' AND owner_id = ?", (memory_id,)
            )
            self.conn.execute(
                "UPDATE memories SET superseded_by = NULL WHERE superseded_by = ?", (memory_id,)
            )
            cur = self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cur.rowcount > 0

    # ── contradictions ──────────────────────────────────────────────────────

    def fact_keys(self, scope: Scope, limit: int = 40) -> list[tuple[str, str]]:
        """(key, current value) of known facts, shown to the model so it reuses keys."""
        where, args = scope.where()
        rows = self.conn.execute(
            f"SELECT slot, value FROM memories WHERE {where} AND slot IS NOT NULL "
            "AND superseded_by IS NULL ORDER BY id DESC LIMIT ?",
            (*args, limit),
        ).fetchall()
        return [(r[0], r[1] or "") for r in rows]

    def conflicting(self, scope: Scope, memory: Memory) -> Memory | None:
        """An earlier, still-valid memory with the same fact key and a different value.
        Events and insights describe moments, not one-value facts: two of them can't
        contradict each other (the model labels which is which)."""
        slot = memory.slot
        if slot is None or memory.kind in MOMENTS:
            return None
        where, args = scope.where()
        rows = self.conn.execute(
            f"SELECT * FROM memories WHERE {where} AND slot = ? AND id != ? "
            "AND superseded_by IS NULL ORDER BY id DESC",
            (*args, slot, memory.id),
        ).fetchall()
        for r in rows:
            old = _memory(r)
            if old.kind not in MOMENTS and not same_value(old.value, memory.value):
                return old
        return None

    def supersede(self, old_id: int, new_id: int) -> None:
        self.conn.execute("UPDATE memories SET superseded_by = ? WHERE id = ?", (new_id, old_id))

    def add_conflict(self, session_id: int, old_id: int, new_id: int) -> None:
        self.conn.execute(
            "INSERT INTO conflicts(session_id, old_memory_id, new_memory_id, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, old_id, new_id, to_iso(utcnow())),
        )

    def conflicts(self, session_id: int, status: str | None = "new") -> list[Conflict]:
        sql = "SELECT * FROM conflicts WHERE session_id = ?"
        args: tuple[Any, ...] = (session_id,)
        if status:
            sql += " AND status = ?"
            args += (status,)
        out = []
        for r in self.conn.execute(sql + " ORDER BY id", args).fetchall():
            old, new = self.get_memory(r["old_memory_id"]), self.get_memory(r["new_memory_id"])
            if old and new:
                out.append(Conflict(r["id"], r["session_id"], old, new))
        return out

    def set_conflict_status(self, conflict_id: int, status: str) -> None:
        self.conn.execute("UPDATE conflicts SET status = ? WHERE id = ?", (status, conflict_id))

    # ── vectors ─────────────────────────────────────────────────────────────

    def save_vector(
        self, owner_type: str, owner_id: int, session_id: int | None, model: str, blob: bytes
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO embeddings(owner_type, owner_id, session_id, model, vector) "
            "VALUES (?, ?, ?, ?, ?)",
            (owner_type, owner_id, session_id, model, blob),
        )

    def all_vectors(self, model: str) -> list[tuple[str, int, int | None, bytes]]:
        rows = self.conn.execute(
            "SELECT owner_type, owner_id, session_id, vector FROM embeddings WHERE model = ?",
            (model,),
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    def unindexed(self, model: str, limit: int = 64) -> list[tuple[str, int, int | None, str]]:
        """Newest first: messages, memories and summaries without a vector for ``model``."""
        out: list[tuple[str, int, int | None, str]] = []
        for owner_type, table in _INDEXED:
            rows = self.conn.execute(
                f"SELECT m.id, m.session_id, m.content FROM {table} m "
                f"{_unindexed_where(owner_type)} ORDER BY m.id DESC LIMIT ?",
                (owner_type, model, limit - len(out)),
            ).fetchall()
            out += [(owner_type, r[0], r[1], r[2]) for r in rows if (r[2] or "").strip()]
            if len(out) >= limit:
                break
        return out

    def count_unindexed(self, model: str) -> int:
        total = 0
        for owner_type, table in _INDEXED:
            total += self.conn.execute(
                f"SELECT COUNT(*) FROM {table} m {_unindexed_where(owner_type)}",
                (owner_type, model),
            ).fetchone()[0]
        return int(total)

    # ── raw messages (recall even when the extractor missed something) ──────

    def scope_sessions(self, scope: Scope) -> set[int]:
        if scope.character_id is None:
            return {scope.session_id}
        rows = self.conn.execute(
            "SELECT id FROM sessions WHERE character_id = ? AND private = 0",
            (scope.character_id,),
        ).fetchall()
        return {r[0] for r in rows} | {scope.session_id}

    def message_keyword_search(self, sessions: set[int], text: str, limit: int = 20) -> list[int]:
        query = trigram_query(text)
        if not query or not sessions:
            return []
        marks = ",".join("?" * len(sessions))
        try:
            rows = self.conn.execute(
                f"""SELECT m.id, m.content FROM messages_fts f JOIN messages m ON m.id = f.rowid
                    WHERE messages_fts MATCH ? AND m.session_id IN ({marks})
                    ORDER BY bm25(messages_fts) LIMIT ?""",
                (query, *sessions, limit * 3),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0] for r in rows if shares_a_word_start(text, r[1])][:limit]

    def messages_on_days(self, sessions: set[int], days: list[str], limit: int = 20) -> list[int]:
        if not sessions or not days:
            return []
        s_marks, d_marks = ",".join("?" * len(sessions)), ",".join("?" * len(days))
        rows = self.conn.execute(
            f"SELECT id FROM messages WHERE session_id IN ({s_marks}) AND local_day IN ({d_marks}) "
            "ORDER BY length(content) DESC LIMIT ?",
            (*sessions, *days, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def mark_used(self, ids: list[int]) -> None:
        if ids:
            now = to_iso(utcnow())
            self.conn.executemany(
                "UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id = ?",
                [(now, i) for i in ids],
            )

    # ── search primitives (used by recall) ──────────────────────────────────

    def keyword_search(self, scope: Scope, text: str, limit: int = 20) -> list[int]:
        query = trigram_query(text)
        if not query:
            return []
        where, args = scope.where("m")
        try:
            rows = self.conn.execute(
                f"""SELECT m.id, m.content, m.place, m.people_json FROM memories_fts f
                    JOIN memories m ON m.id = f.rowid
                    WHERE memories_fts MATCH ? AND {where} AND m.superseded_by IS NULL
                    ORDER BY bm25(memories_fts) LIMIT ?""",
                (query, *args, limit * 3),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0] for r in rows if shares_a_word_start(text, f"{r[1]} {r[2] or ''} {r[3]}")][
            :limit
        ]

    def memories_on_days(self, scope: Scope, days: list[str], limit: int = 30) -> list[int]:
        if not days:
            return []
        where, args = scope.where()
        marks = ",".join("?" * len(days))
        rows = self.conn.execute(
            f"SELECT id FROM memories WHERE {where} AND day IN ({marks}) "
            "AND superseded_by IS NULL ORDER BY importance DESC, id DESC LIMIT ?",
            (*args, *days, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def memories_for_entities(self, entity_ids: list[int], limit: int = 30) -> list[int]:
        if not entity_ids:
            return []
        marks = ",".join("?" * len(entity_ids))
        rows = self.conn.execute(
            f"""SELECT me.memory_id, COUNT(*) AS hits FROM memory_entities me
                JOIN memories m ON m.id = me.memory_id
                WHERE me.entity_id IN ({marks}) AND m.superseded_by IS NULL
                GROUP BY me.memory_id ORDER BY hits DESC, m.importance DESC, m.id DESC LIMIT ?""",
            (*entity_ids, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def pinned_ids(self, scope: Scope) -> list[int]:
        where, args = scope.where()
        rows = self.conn.execute(
            f"SELECT id FROM memories WHERE {where} AND pinned = 1 "
            "AND superseded_by IS NULL ORDER BY id",
            args,
        ).fetchall()
        return [r[0] for r in rows]

    # ── entities ────────────────────────────────────────────────────────────

    def upsert_entity(self, scope: Scope, name: str, kind: str) -> int:
        norm = search_text(name)
        where, args = scope.where()
        row = self.conn.execute(
            f"SELECT id FROM entities WHERE {where} AND norm = ?", (*args, norm)
        ).fetchone()
        now = to_iso(utcnow())
        if row:
            self.conn.execute("UPDATE entities SET updated_at = ? WHERE id = ?", (now, row[0]))
            return int(row[0])
        cur = self.conn.execute(
            "INSERT INTO entities(session_id, character_id, name, norm, kind, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (scope.session_id, scope.character_id, name.strip(), norm, kind, now),
        )
        return int(cur.lastrowid or 0)

    def _link(self, memory_id: int, entity_id: int) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id) VALUES (?, ?)",
            (memory_id, entity_id),
        )

    def entities(self, scope: Scope) -> list[tuple[int, str, str, str]]:
        """(id, name, norm, kind) for name matching."""
        where, args = scope.where()
        rows = self.conn.execute(
            f"SELECT id, name, norm, kind FROM entities WHERE {where}", args
        ).fetchall()
        return [(r[0], r[1], r[2], r[3]) for r in rows]

    # ── summaries ───────────────────────────────────────────────────────────

    def add_summary(
        self,
        session_id: int,
        kind: str,
        content: str,
        *,
        start_message_id: int | None = None,
        end_message_id: int | None = None,
        day: str | None = None,
    ) -> Summary:
        now = to_iso(utcnow())
        if kind == "day":
            # one digest per day: replace an older one
            self.conn.execute(
                "DELETE FROM summaries WHERE session_id = ? AND kind = 'day' AND day = ?",
                (session_id, day),
            )
        cur = self.conn.execute(
            """INSERT INTO summaries(session_id, kind, start_message_id, end_message_id, day,
               content, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, kind, start_message_id, end_message_id, day, content.strip(), now, now),
        )
        row = self.conn.execute("SELECT * FROM summaries WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _summary(row)

    def summaries(self, session_id: int, kind: str) -> list[Summary]:
        rows = self.conn.execute(
            "SELECT * FROM summaries WHERE session_id = ? AND kind = ? "
            "ORDER BY COALESCE(end_message_id, 0), day, id",
            (session_id, kind),
        ).fetchall()
        return [_summary(r) for r in rows]

    def day_summaries(self, scope: Scope, days: list[str]) -> list[Summary]:
        if not days:
            return []
        marks = ",".join("?" * len(days))
        if scope.character_id is not None:
            sql = (
                f"SELECT s.* FROM summaries s JOIN sessions x ON x.id = s.session_id "
                f"WHERE x.character_id = ? AND s.kind = 'day' AND s.day IN ({marks}) "
                "ORDER BY s.day, s.id"
            )
            args: tuple[Any, ...] = (scope.character_id, *days)
        else:
            sql = (
                f"SELECT * FROM summaries WHERE session_id = ? AND kind = 'day' "
                f"AND day IN ({marks}) ORDER BY day, id"
            )
            args = (scope.session_id, *days)
        return [_summary(r) for r in self.conn.execute(sql, args).fetchall()]

    def get_summary(self, summary_id: int) -> Summary | None:
        row = self.conn.execute("SELECT * FROM summaries WHERE id = ?", (summary_id,)).fetchone()
        return _summary(row) if row else None

    def day_summaries_in(self, sessions: set[int]) -> list[Summary]:
        if not sessions:
            return []
        marks = ",".join("?" * len(sessions))
        rows = self.conn.execute(
            f"SELECT * FROM summaries WHERE kind = 'day' AND session_id IN ({marks})",
            tuple(sessions),
        ).fetchall()
        return [_summary(r) for r in rows]

    def days_with_messages(self, session_id: int, before: str, limit: int = 30) -> list[str]:
        rows = self.conn.execute(
            "SELECT local_day, COUNT(*) FROM messages WHERE session_id = ? AND local_day < ? "
            "GROUP BY local_day HAVING COUNT(*) >= 2 ORDER BY local_day DESC LIMIT ?",
            (session_id, before, limit),
        ).fetchall()
        return [r[0] for r in rows]

    # ── job queue ───────────────────────────────────────────────────────────

    def enqueue(
        self, session_id: int | None, kind: str, payload: dict[str, Any] | None = None
    ) -> bool:
        """Add a job unless the same one is already waiting. Returns True if added."""
        body = json.dumps(payload or {}, sort_keys=True, ensure_ascii=False)
        exists = self.conn.execute(
            "SELECT 1 FROM jobs WHERE session_id IS ? AND kind = ? AND payload_json = ? "
            "AND status IN ('pending', 'running')",
            (session_id, kind, body),
        ).fetchone()
        if exists:
            return False
        now = to_iso(utcnow())
        self.conn.execute(
            "INSERT INTO jobs(session_id, kind, payload_json, priority, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, kind, body, JOB_PRIORITY.get(kind, 5), now, now),
        )
        return True

    def next_job(self) -> Job | None:
        """Cheapest work first (indexing), then memory passes, then summaries."""
        now = to_iso(utcnow())
        row = self.conn.execute(
            "SELECT * FROM jobs WHERE status = 'pending' AND (run_after IS NULL OR run_after <= ?) "
            "ORDER BY priority, id LIMIT 1",
            (now,),
        ).fetchone()
        if row is None:
            return None
        self.conn.execute(
            "UPDATE jobs SET status = 'running', updated_at = ? WHERE id = ?", (now, row["id"])
        )
        return Job(
            row["id"],
            row["session_id"],
            row["kind"],
            json.loads(row["payload_json"]),
            row["attempts"],
        )

    def finish_job(self, job_id: int) -> None:
        self.conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))

    def release_job(self, job_id: int) -> None:
        """Interrupted (e.g. the user started typing): back to the queue, no penalty."""
        self.conn.execute("UPDATE jobs SET status = 'pending' WHERE id = ?", (job_id,))

    def fail_job(self, job_id: int, error: str, retry_after: str | None, give_up: bool) -> None:
        self.conn.execute(
            "UPDATE jobs SET status = ?, attempts = attempts + 1, error = ?, run_after = ?, "
            "updated_at = ? WHERE id = ?",
            (
                "failed" if give_up else "pending",
                error[:500],
                retry_after,
                to_iso(utcnow()),
                job_id,
            ),
        )

    def reset_stale_jobs(self) -> None:
        """Jobs left 'running' by a previous run that was closed mid-job."""
        self.conn.execute("UPDATE jobs SET status = 'pending' WHERE status = 'running'")

    def job_stats(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status").fetchall()
        return {r[0]: r[1] for r in rows}

    def last_job_error(self, since: str | None = None) -> str | None:
        """The latest job error, only if it happened after ``since`` (the last success):
        an old, already-fixed problem is not worth showing."""
        row = self.conn.execute(
            "SELECT error FROM jobs WHERE error IS NOT NULL AND updated_at > ? "
            "ORDER BY updated_at DESC LIMIT 1",
            (since or "",),
        ).fetchone()
        return row[0] if row else None
