"""SQLite persistence: sessions, a branching message tree, and keyword search.

The database is a single file in WAL mode. Messages form a tree (``parent_id``);
each session's ``head_message_id`` marks the tip of the active branch, so
regenerating or undoing never destroys history.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import tzinfo
from pathlib import Path
from typing import Any

from chatclient.characters import Card, Character
from chatclient.text import normalize_for_search
from chatclient.timeutil import local_day, to_iso, utcnow

log = logging.getLogger(__name__)


def _run_script(conn: sqlite3.Connection, script: str) -> None:
    # Not executescript(): it implicitly COMMITs, which would break the migration txn.
    for stmt in script.split(";"):
        if stmt.strip():
            conn.execute(stmt)


def _m001_init(conn: sqlite3.Connection) -> None:
    _run_script(
        conn,
        """
        CREATE TABLE sessions (
          id INTEGER PRIMARY KEY,
          title TEXT NOT NULL DEFAULT '',
          model TEXT,
          system_prompt TEXT,
          params_json TEXT NOT NULL DEFAULT '{}',
          head_message_id INTEGER,
          archived INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_sessions_updated ON sessions(archived, updated_at);

        CREATE TABLE messages (
          id INTEGER PRIMARY KEY,
          session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
          parent_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
          role TEXT NOT NULL CHECK (role IN ('system','user','assistant')),
          content TEXT NOT NULL,
          created_at TEXT NOT NULL,
          local_day TEXT NOT NULL,
          model TEXT,
          prompt_tokens INTEGER,
          completion_tokens INTEGER,
          meta_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE INDEX idx_messages_session_time ON messages(session_id, created_at);
        CREATE INDEX idx_messages_day ON messages(local_day, session_id);
        CREATE INDEX idx_messages_parent ON messages(parent_id);

        CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """,
    )
    # Trigram tokenizer: substring matching that works for any language and is
    # tolerant of suffixes ("kutuphane" finds "kütüphanedeki"). Needs SQLite >= 3.34.
    try:
        conn.execute("CREATE VIRTUAL TABLE messages_fts USING fts5(norm_text, tokenize='trigram')")
    except sqlite3.OperationalError:
        log.warning("SQLite trigram tokenizer unavailable; falling back to unicode61")
        conn.execute(
            "CREATE VIRTUAL TABLE messages_fts USING fts5(norm_text, tokenize='unicode61')"
        )


def _m002_characters(conn: sqlite3.Connection) -> None:
    _run_script(
        conn,
        """
        CREATE TABLE characters (
          id INTEGER PRIMARY KEY,
          kind TEXT NOT NULL CHECK (kind IN ('ai','user')),
          name TEXT NOT NULL,
          card_json TEXT NOT NULL,
          builtin_key TEXT UNIQUE,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_characters_kind ON characters(kind, name);
        ALTER TABLE sessions ADD COLUMN character_id INTEGER
          REFERENCES characters(id) ON DELETE SET NULL;
        ALTER TABLE sessions ADD COLUMN persona_id INTEGER
          REFERENCES characters(id) ON DELETE SET NULL
        """,
    )


MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [_m001_init, _m002_characters]


@dataclass
class Session:
    id: int
    title: str
    model: str | None
    system_prompt: str | None
    params: dict[str, Any]
    head_message_id: int | None
    archived: bool
    created_at: str
    updated_at: str
    character_id: int | None = None
    persona_id: int | None = None


@dataclass
class Message:
    id: int
    session_id: int
    parent_id: int | None
    role: str
    content: str
    created_at: str
    local_day: str
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    meta: dict[str, Any] | None = None


@dataclass
class SearchHit:
    message: Message
    session_title: str
    snippet: str


@dataclass
class SessionSummary:
    session: Session
    message_count: int


def _session(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        title=row["title"],
        model=row["model"],
        system_prompt=row["system_prompt"],
        params=json.loads(row["params_json"] or "{}"),
        head_message_id=row["head_message_id"],
        archived=bool(row["archived"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        character_id=row["character_id"],
        persona_id=row["persona_id"],
    )


def _character(row: sqlite3.Row) -> Character:
    return Character(
        id=row["id"],
        kind=row["kind"],
        card=Card.from_json(row["card_json"]),
        builtin_key=row["builtin_key"],
    )


def _message(row: sqlite3.Row) -> Message:
    return Message(
        id=row["id"],
        session_id=row["session_id"],
        parent_id=row["parent_id"],
        role=row["role"],
        content=row["content"],
        created_at=row["created_at"],
        local_day=row["local_day"],
        model=row["model"],
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        meta=json.loads(row["meta_json"] or "{}"),
    )


class Store:
    def __init__(
        self,
        path: Path | str,
        *,
        tz: tzinfo | None = None,
        day_boundary: str = "04:00",
    ):
        self.path = str(path)
        self.tz = tz
        self.day_boundary = day_boundary
        self.conn = sqlite3.connect(self.path, isolation_level=None)  # autocommit; explicit txns
        self.conn.row_factory = sqlite3.Row
        for pragma in (
            "journal_mode=WAL",
            "foreign_keys=ON",
            "synchronous=NORMAL",
            "busy_timeout=5000",
        ):
            self.conn.execute(f"PRAGMA {pragma}")
        self._migrate()

    # ── infrastructure ──────────────────────────────────────────────────────

    def _migrate(self) -> None:
        version = self.conn.execute("PRAGMA user_version").fetchone()[0]
        for i, migration in enumerate(MIGRATIONS[version:], start=version + 1):
            log.info("applying migration %d", i)
            with self.tx():
                migration(self.conn)
                self.conn.execute(f"PRAGMA user_version = {i}")

    def tx(self) -> _Tx:
        return _Tx(self.conn)

    def close(self) -> None:
        self.conn.close()

    # ── key/value ───────────────────────────────────────────────────────────

    def kv_get(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def kv_set(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO kv(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    # ── characters & personas ───────────────────────────────────────────────

    def add_character(self, kind: str, card: Card, builtin_key: str | None = None) -> Character:
        now = to_iso(utcnow())
        cur = self.conn.execute(
            "INSERT INTO characters(kind, name, card_json, builtin_key, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (kind, card.name, card.to_json(), builtin_key, now, now),
        )
        char = self.get_character(cur.lastrowid or 0)
        assert char is not None
        return char

    def get_character(self, character_id: int | None) -> Character | None:
        if character_id is None:
            return None
        row = self.conn.execute("SELECT * FROM characters WHERE id = ?", (character_id,)).fetchone()
        return _character(row) if row else None

    def list_characters(self, kind: str = "ai") -> list[Character]:
        rows = self.conn.execute(
            "SELECT * FROM characters WHERE kind = ? ORDER BY name COLLATE NOCASE, id", (kind,)
        ).fetchall()
        return [_character(r) for r in rows]

    def find_character(self, ref: str, kind: str = "ai") -> Character | None:
        """By id ("3", "#3") or by name: exact (case/diacritic-insensitive) first, then prefix."""
        ref = ref.strip().lstrip("#")
        if ref.isdigit():
            c = self.get_character(int(ref))
            return c if c and c.kind == kind else None
        want = normalize_for_search(ref)
        chars = self.list_characters(kind)
        for c in chars:
            if normalize_for_search(c.name) == want:
                return c
        prefixed = [c for c in chars if normalize_for_search(c.name).startswith(want)]
        return prefixed[0] if len(prefixed) == 1 else None

    def update_character(self, character_id: int, card: Card) -> None:
        self.conn.execute(
            "UPDATE characters SET name = ?, card_json = ?, updated_at = ? WHERE id = ?",
            (card.name, card.to_json(), to_iso(utcnow()), character_id),
        )

    def delete_character(self, character_id: int) -> None:
        self.conn.execute("DELETE FROM characters WHERE id = ?", (character_id,))

    def clear_builtin_key(self, character_id: int) -> None:
        self.conn.execute("UPDATE characters SET builtin_key = NULL WHERE id = ?", (character_id,))

    def builtin_keys(self) -> set[str]:
        rows = self.conn.execute(
            "SELECT builtin_key FROM characters WHERE builtin_key IS NOT NULL"
        ).fetchall()
        return {r[0] for r in rows}

    def sessions_using(self, character_id: int) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE character_id = ? OR persona_id = ?",
            (character_id, character_id),
        ).fetchone()
        return int(row[0])

    # ── sessions ────────────────────────────────────────────────────────────

    def create_session(self, model: str | None, title: str = "") -> Session:
        now = to_iso(utcnow())
        cur = self.conn.execute(
            "INSERT INTO sessions(title, model, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (title, model, now, now),
        )
        session = self.get_session(cur.lastrowid or 0)
        assert session is not None
        return session

    def get_session(self, session_id: int) -> Session | None:
        row = self.conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return _session(row) if row else None

    def latest_session(self) -> Session | None:
        row = self.conn.execute(
            "SELECT * FROM sessions WHERE archived = 0 ORDER BY updated_at DESC, id DESC LIMIT 1"
        ).fetchone()
        return _session(row) if row else None

    def list_sessions(
        self, limit: int = 50, include_archived: bool = False
    ) -> list[SessionSummary]:
        rows = self.conn.execute(
            f"""
            SELECT s.*, (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS n
            FROM sessions s
            {"" if include_archived else "WHERE s.archived = 0"}
            ORDER BY s.updated_at DESC, s.id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [SessionSummary(_session(r), r["n"]) for r in rows]

    def update_session(self, session_id: int, **fields: Any) -> None:
        allowed = {
            "title",
            "model",
            "system_prompt",
            "params",
            "head_message_id",
            "archived",
            "character_id",
            "persona_id",
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown session fields: {unknown}")
        if "params" in fields:
            fields["params_json"] = json.dumps(fields.pop("params"), ensure_ascii=False)
        fields["updated_at"] = to_iso(utcnow())
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE sessions SET {cols} WHERE id = ?", (*fields.values(), session_id)
        )

    def delete_session(self, session_id: int) -> None:
        with self.tx():
            self.conn.execute(
                "DELETE FROM messages_fts WHERE rowid IN "
                "(SELECT id FROM messages WHERE session_id = ?)",
                (session_id,),
            )
            # Break the head pointer first, then delete messages and the session.
            self.conn.execute(
                "UPDATE sessions SET head_message_id = NULL WHERE id = ?", (session_id,)
            )
            self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))

    # ── messages ────────────────────────────────────────────────────────────

    def add_message(
        self,
        session_id: int,
        parent_id: int | None,
        role: str,
        content: str,
        *,
        model: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        meta: dict[str, Any] | None = None,
        set_head: bool = True,
    ) -> Message:
        now = utcnow()
        day = local_day(now, self.tz, self.day_boundary).isoformat()
        with self.tx():
            cur = self.conn.execute(
                """
                INSERT INTO messages(session_id, parent_id, role, content, created_at, local_day,
                                     model, prompt_tokens, completion_tokens, meta_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    parent_id,
                    role,
                    content,
                    to_iso(now),
                    day,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    json.dumps(meta or {}, ensure_ascii=False),
                ),
            )
            mid = cur.lastrowid
            self.conn.execute(
                "INSERT INTO messages_fts(rowid, norm_text) VALUES (?, ?)",
                (mid, normalize_for_search(content)),
            )
            if set_head:
                self.conn.execute(
                    "UPDATE sessions SET head_message_id = ?, updated_at = ? WHERE id = ?",
                    (mid, to_iso(now), session_id),
                )
        msg = self.get_message(mid or 0)
        assert msg is not None
        return msg

    def get_message(self, message_id: int) -> Message | None:
        row = self.conn.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return _message(row) if row else None

    def branch(self, head_id: int | None, limit: int | None = None) -> list[Message]:
        """Messages from the root to ``head_id`` (the active branch), oldest first.

        With ``limit``, only the most recent ``limit`` messages are returned.
        """
        if head_id is None:
            return []
        rows = self.conn.execute(
            """
            WITH RECURSIVE chain(id, parent_id, depth) AS (
              SELECT id, parent_id, 0 FROM messages WHERE id = ?
              UNION ALL
              SELECT m.id, m.parent_id, c.depth + 1
              FROM messages m JOIN chain c ON m.id = c.parent_id
              WHERE (? IS NULL OR c.depth + 1 < ?)
            )
            SELECT m.* FROM chain c JOIN messages m ON m.id = c.id ORDER BY c.depth DESC
            """,
            (head_id, limit, limit),
        ).fetchall()
        return [_message(r) for r in rows]

    def search(self, query: str, session_id: int | None = None, limit: int = 20) -> list[SearchHit]:
        norm = normalize_for_search(query)
        if len(norm) < 3:
            return []
        phrase = '"' + norm.replace('"', '""') + '"'
        rows = self.conn.execute(
            f"""
            SELECT m.*, s.title AS session_title
            FROM messages_fts f
            JOIN messages m ON m.id = f.rowid
            JOIN sessions s ON s.id = m.session_id
            WHERE messages_fts MATCH ?
            {"AND m.session_id = ?" if session_id is not None else ""}
            ORDER BY m.created_at DESC
            LIMIT ?
            """,
            (phrase, session_id, limit) if session_id is not None else (phrase, limit),
        ).fetchall()
        hits = []
        for r in rows:
            msg = _message(r)
            hits.append(SearchHit(msg, r["session_title"], _snippet(msg.content, norm)))
        return hits


def _snippet(content: str, norm_query: str, width: int = 120) -> str:
    """Cut a window around the first match (positions align for most scripts)."""
    norm = normalize_for_search(content)
    flat = " ".join(content.split())
    pos = norm.find(norm_query)
    if pos < 0 or len(flat) <= width:
        return flat[:width] + ("…" if len(flat) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(flat), start + width)
    return ("…" if start else "") + flat[start:end] + ("…" if end < len(flat) else "")


class _Tx:
    """Nestable transaction context (BEGIN IMMEDIATE / savepoints)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self.nested = False

    def __enter__(self) -> sqlite3.Connection:
        self.nested = self.conn.in_transaction
        self.conn.execute("SAVEPOINT sp" if self.nested else "BEGIN IMMEDIATE")
        return self.conn

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.conn.execute("RELEASE sp" if self.nested else "COMMIT")
        else:
            self.conn.execute("ROLLBACK TO sp" if self.nested else "ROLLBACK")
            if self.nested:
                self.conn.execute("RELEASE sp")
