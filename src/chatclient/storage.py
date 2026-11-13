"""SQLite persistence: sessions, a branching message tree, and keyword search.

The database is a single file in WAL mode. Messages form a tree (``parent_id``);
each session's ``head_message_id`` marks the tip of the active branch, so
regenerating or undoing never destroys history.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import tzinfo
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chatclient.characters import Card, Character
from chatclient.text import normalize_for_search
from chatclient.timeutil import local_day, to_iso, utcnow

if TYPE_CHECKING:
    from chatclient.library import Library

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


def _m003_memory(conn: sqlite3.Connection) -> None:
    _run_script(
        conn,
        """
        CREATE TABLE summaries (
          id INTEGER PRIMARY KEY,
          session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
          kind TEXT NOT NULL CHECK (kind IN ('segment','synopsis','day')),
          start_message_id INTEGER,
          end_message_id INTEGER,
          day TEXT,
          content TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_summaries_session ON summaries(session_id, kind, end_message_id);
        CREATE INDEX idx_summaries_day ON summaries(session_id, kind, day);

        CREATE TABLE memories (
          id INTEGER PRIMARY KEY,
          session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
          character_id INTEGER REFERENCES characters(id) ON DELETE CASCADE,
          kind TEXT NOT NULL DEFAULT 'fact',
          content TEXT NOT NULL,
          importance REAL NOT NULL DEFAULT 0.5,
          pinned INTEGER NOT NULL DEFAULT 0,
          day TEXT,
          story_time TEXT,
          place TEXT,
          people_json TEXT NOT NULL DEFAULT '[]',
          source_message_id INTEGER,
          created_at TEXT NOT NULL,
          last_used_at TEXT,
          use_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX idx_memories_session ON memories(session_id, day);
        CREATE INDEX idx_memories_character ON memories(character_id, day);

        CREATE TABLE entities (
          id INTEGER PRIMARY KEY,
          session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
          character_id INTEGER REFERENCES characters(id) ON DELETE CASCADE,
          name TEXT NOT NULL,
          norm TEXT NOT NULL,
          kind TEXT NOT NULL DEFAULT 'person',
          updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_entities_scope ON entities(session_id, character_id, norm);
        CREATE TABLE memory_entities (
          memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
          entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
          PRIMARY KEY (memory_id, entity_id)
        );
        CREATE INDEX idx_memory_entities_entity ON memory_entities(entity_id);

        CREATE TABLE jobs (
          id INTEGER PRIMARY KEY,
          session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
          kind TEXT NOT NULL,
          payload_json TEXT NOT NULL DEFAULT '{}',
          status TEXT NOT NULL DEFAULT 'pending',
          attempts INTEGER NOT NULL DEFAULT 0,
          error TEXT,
          run_after TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );
        CREATE INDEX idx_jobs_status ON jobs(status, id);

        ALTER TABLE sessions ADD COLUMN scene_json TEXT
        """,
    )
    try:
        conn.execute("CREATE VIRTUAL TABLE memories_fts USING fts5(norm_text, tokenize='trigram')")
    except sqlite3.OperationalError:
        conn.execute(
            "CREATE VIRTUAL TABLE memories_fts USING fts5(norm_text, tokenize='unicode61')"
        )


def _m004_semantic(conn: sqlite3.Connection) -> None:
    _run_script(
        conn,
        """
        CREATE TABLE embeddings (
          owner_type TEXT NOT NULL CHECK (owner_type IN ('message','memory','summary')),
          owner_id INTEGER NOT NULL,
          session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
          model TEXT NOT NULL,
          vector BLOB NOT NULL,
          PRIMARY KEY (owner_type, owner_id, model)
        );
        CREATE INDEX idx_embeddings_model ON embeddings(model, owner_type);

        ALTER TABLE memories ADD COLUMN subject TEXT;
        ALTER TABLE memories ADD COLUMN attribute TEXT;
        ALTER TABLE memories ADD COLUMN value TEXT;
        ALTER TABLE memories ADD COLUMN slot TEXT;
        ALTER TABLE memories ADD COLUMN said_by TEXT;
        ALTER TABLE memories ADD COLUMN superseded_by INTEGER;
        CREATE INDEX idx_memories_slot ON memories(slot);

        CREATE TABLE conflicts (
          id INTEGER PRIMARY KEY,
          session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
          old_memory_id INTEGER NOT NULL,
          new_memory_id INTEGER NOT NULL,
          status TEXT NOT NULL DEFAULT 'new',
          created_at TEXT NOT NULL
        );
        CREATE INDEX idx_conflicts_session ON conflicts(session_id, status);

        ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 5
        """,
    )


def _m005_private(conn: sqlite3.Connection) -> None:
    conn.execute("ALTER TABLE sessions ADD COLUMN private INTEGER NOT NULL DEFAULT 0")


def new_uuid() -> str:
    return uuid.uuid4().hex


def _m006_files(conn: sqlite3.Connection) -> None:
    """Stable ids for the files that now hold characters and chats; ``log_size`` is how
    much of a chat's file the database has already taken in."""
    for table in ("sessions", "messages", "characters"):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN uuid TEXT")
        for (rid,) in conn.execute(f"SELECT id FROM {table}").fetchall():
            conn.execute(f"UPDATE {table} SET uuid = ? WHERE id = ?", (new_uuid(), rid))
        conn.execute(f"CREATE UNIQUE INDEX idx_{table}_uuid ON {table}(uuid)")
    conn.execute("ALTER TABLE sessions ADD COLUMN log_size INTEGER NOT NULL DEFAULT 0")


MIGRATIONS: list[Callable[[sqlite3.Connection], None]] = [
    _m001_init,
    _m002_characters,
    _m003_memory,
    _m004_semantic,
    _m005_private,
    _m006_files,
]


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
    scene: dict[str, Any] = field(default_factory=dict)  # place, time, mood… (memory pass)
    private: bool = False  # nothing from it is remembered, indexed or recalled elsewhere
    uuid: str = ""
    log_size: int = 0  # bytes of the chat file already in the database


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
    uuid: str = ""


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
        scene=json.loads(row["scene_json"] or "{}"),
        private=bool(row["private"]),
        uuid=row["uuid"] or "",
        log_size=row["log_size"] or 0,
    )


def _character(row: sqlite3.Row) -> Character:
    return Character(
        id=row["id"],
        kind=row["kind"],
        card=Card.from_json(row["card_json"]),
        builtin_key=row["builtin_key"],
        uuid=row["uuid"] or "",
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
        uuid=row["uuid"] or "",
    )


class Store:
    def __init__(
        self,
        path: Path | str,
        *,
        tz: tzinfo | None = None,
        day_boundary: str = "04:00",
        library: Library | None = None,
    ):
        self.path = str(path)
        self.tz = tz
        self.day_boundary = day_boundary
        # With a library, characters, personas and chats live in files (the source of
        # truth) and this database is their index. Without one (tests, the memory
        # evaluation), the database stands alone.
        self.library = library
        self._quiet = 0
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

    # ── files (source of truth when a library is attached) ─────────────────

    @contextmanager
    def quiet(self) -> Iterator[None]:
        """Index only: used while reading the files back in, so nothing is re-written."""
        self._quiet += 1
        try:
            yield
        finally:
            self._quiet -= 1

    @property
    def files(self) -> Library | None:
        return self.library if self.library is not None and not self._quiet else None

    def _append(self, session_id: int, event: dict[str, Any]) -> int | None:
        """Write an event to the session's file first; returns the new file size, which
        the caller stores in the same transaction as the change itself."""
        lib = self.files
        if lib is None:
            return None
        s = self.get_session(session_id)
        if s is None:
            return None
        try:
            return lib.append(self, s, event)
        except OSError:
            log.exception("could not write the chat file of session %s", session_id)
            return None

    def _set_log_size(self, session_id: int, size: int | None) -> None:
        if size is not None:
            self.conn.execute("UPDATE sessions SET log_size = ? WHERE id = ?", (size, session_id))

    def uuid_of_message(self, message_id: int | None) -> str | None:
        if message_id is None:
            return None
        row = self.conn.execute("SELECT uuid FROM messages WHERE id = ?", (message_id,)).fetchone()
        return row[0] if row else None

    def message_id(self, message_uuid: str | None) -> int | None:
        if not message_uuid:
            return None
        row = self.conn.execute(
            "SELECT id FROM messages WHERE uuid = ?", (message_uuid,)
        ).fetchone()
        return row[0] if row else None

    def uuid_of_character(self, character_id: int | None) -> str | None:
        char = self.get_character(character_id)
        return char.uuid if char else None

    def character_by_uuid(self, char_uuid: str | None) -> Character | None:
        if not char_uuid:
            return None
        row = self.conn.execute("SELECT * FROM characters WHERE uuid = ?", (char_uuid,)).fetchone()
        return _character(row) if row else None

    def session_by_uuid(self, session_uuid: str | None) -> Session | None:
        if not session_uuid:
            return None
        row = self.conn.execute("SELECT * FROM sessions WHERE uuid = ?", (session_uuid,)).fetchone()
        return _session(row) if row else None

    def all_sessions(self) -> list[Session]:
        return [_session(r) for r in self.conn.execute("SELECT * FROM sessions ORDER BY id")]

    def all_characters(self) -> list[Character]:
        return [_character(r) for r in self.conn.execute("SELECT * FROM characters ORDER BY id")]

    def session_messages(self, session_id: int) -> list[Message]:
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? ORDER BY id", (session_id,)
        ).fetchall()
        return [_message(r) for r in rows]

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

    def add_character(
        self,
        kind: str,
        card: Card,
        builtin_key: str | None = None,
        *,
        char_uuid: str | None = None,
    ) -> Character:
        now = to_iso(utcnow())
        char_uuid = char_uuid or new_uuid()
        if self.files is not None:  # the folder first: it is what survives
            self.files.write_card(Character(0, kind, card, builtin_key, char_uuid))
        cur = self.conn.execute(
            "INSERT INTO characters(kind, name, card_json, builtin_key, created_at, updated_at, "
            "uuid) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (kind, card.name, card.to_json(), builtin_key, now, now, char_uuid),
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
        char = self.get_character(character_id)
        if char is not None and self.files is not None:
            self.files.write_card(Character(char.id, char.kind, card, char.builtin_key, char.uuid))
        self.conn.execute(
            "UPDATE characters SET name = ?, card_json = ?, updated_at = ? WHERE id = ?",
            (card.name, card.to_json(), to_iso(utcnow()), character_id),
        )

    def delete_character(self, character_id: int) -> None:
        char = self.get_character(character_id)
        if char is not None and self.files is not None:
            self.files.remove_card(char)
        with self.tx():
            for table, column in (("memories_fts", "rowid"), ("embeddings", "owner_id")):
                extra = " AND owner_type = 'memory'" if table == "embeddings" else ""
                self.conn.execute(
                    f"DELETE FROM {table} WHERE {column} IN "
                    f"(SELECT id FROM memories WHERE character_id = ?){extra}",
                    (character_id,),
                )
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

    def create_session(
        self,
        model: str | None,
        title: str = "",
        *,
        character_id: int | None = None,
        persona_id: int | None = None,
        session_uuid: str | None = None,
        created_at: str | None = None,
    ) -> Session:
        now = created_at or to_iso(utcnow())
        with self.tx():
            cur = self.conn.execute(
                "INSERT INTO sessions(title, model, created_at, updated_at, character_id, "
                "persona_id, uuid) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (title, model, now, now, character_id, persona_id, session_uuid or new_uuid()),
            )
            sid = cur.lastrowid or 0
            s = self.get_session(sid)
            assert s is not None
            if self.files is not None:
                self._set_log_size(sid, self._append(sid, self.files.header(self, s)))
        session = self.get_session(sid)
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
            "private",
            "_at",  # replaying a file: the time the change was made
        }
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"unknown session fields: {unknown}")
        at = fields.pop("_at", None) if "_at" in fields else None
        size = None
        if self.files is not None:
            before = self.get_session(session_id)
            if before is not None and "character_id" in fields:
                # the chat file lives in its character's folder: move it along
                after = Session(**{**before.__dict__, "character_id": fields["character_id"]})
                self.files.move_chat(self, before, after)
            size = self._append(
                session_id, self.files.set_event(self, {**fields, "_at": at} if at else fields)
            )
        if "params" in fields:
            fields["params_json"] = json.dumps(fields.pop("params"), ensure_ascii=False)
        fields["updated_at"] = at or to_iso(utcnow())
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self.tx():
            self.conn.execute(
                f"UPDATE sessions SET {cols} WHERE id = ?", (*fields.values(), session_id)
            )
            self._set_log_size(session_id, size)

    def set_scene(self, session_id: int, scene: dict[str, Any]) -> None:
        """Not via update_session: background updates must not reorder the session list."""
        size = self._append(session_id, {"t": "set", "scene": scene or None})
        with self.tx():
            self.conn.execute(
                "UPDATE sessions SET scene_json = ? WHERE id = ?",
                (json.dumps(scene, ensure_ascii=False) if scene else None, session_id),
            )
            self._set_log_size(session_id, size)

    def forget_session_memory(self, session_id: int) -> None:
        """Drop everything memory derived from a session (notes, summaries, vectors,
        queued work). The messages themselves stay."""
        with self.tx():
            self.conn.execute(
                "DELETE FROM memories_fts WHERE rowid IN "
                "(SELECT id FROM memories WHERE session_id = ?)",
                (session_id,),
            )
            for table in ("embeddings", "conflicts", "jobs", "memories", "summaries", "entities"):
                self.conn.execute(f"DELETE FROM {table} WHERE session_id = ?", (session_id,))
            self.conn.execute(
                "DELETE FROM kv WHERE key IN (?, ?)",
                (f"memory_cursor:{session_id}", f"reflect_cursor:{session_id}:None"),
            )
        self.set_scene(session_id, {})

    def delete_session(self, session_id: int) -> None:
        s = self.get_session(session_id)
        if s is not None and self.files is not None:
            self.files.remove_chat(self, s)
        with self.tx():
            self.conn.execute(
                "DELETE FROM messages_fts WHERE rowid IN "
                "(SELECT id FROM messages WHERE session_id = ?)",
                (session_id,),
            )
            self.conn.execute(
                "DELETE FROM memories_fts WHERE rowid IN "
                "(SELECT id FROM memories WHERE session_id = ?)",
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
        message_uuid: str | None = None,
        created_at: str | None = None,
        day: str | None = None,
    ) -> Message:
        created = created_at or to_iso(utcnow())
        if day is None:
            day = local_day(utcnow(), self.tz, self.day_boundary).isoformat()
        message_uuid = message_uuid or new_uuid()
        size = self._append(
            session_id,
            {
                "t": "msg", "id": message_uuid, "parent": self.uuid_of_message(parent_id),
                "role": role, "content": content, "at": created, "day": day, "model": model,
                "tokens": [prompt_tokens, completion_tokens], "meta": meta or {},
                "head": set_head,
            },
        )  # fmt: skip
        with self.tx():
            cur = self.conn.execute(
                """
                INSERT INTO messages(session_id, parent_id, role, content, created_at, local_day,
                                     model, prompt_tokens, completion_tokens, meta_json, uuid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    parent_id,
                    role,
                    content,
                    created,
                    day,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    json.dumps(meta or {}, ensure_ascii=False),
                    message_uuid,
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
                    (mid, created, session_id),
                )
            self._set_log_size(session_id, size)
        msg = self.get_message(mid or 0)
        assert msg is not None
        return msg

    def set_message_meta(self, message_id: int, meta: dict[str, Any]) -> None:
        msg = self.get_message(message_id)
        size = None
        if msg is not None:
            size = self._append(msg.session_id, {"t": "meta", "id": msg.uuid, "meta": meta})
        with self.tx():
            self.conn.execute(
                "UPDATE messages SET meta_json = ? WHERE id = ?",
                (json.dumps(meta, ensure_ascii=False), message_id),
            )
            if msg is not None:
                self._set_log_size(msg.session_id, size)

    def update_message_content(self, message_id: int, content: str) -> None:
        msg = self.get_message(message_id)
        size = None
        if msg is not None:
            size = self._append(msg.session_id, {"t": "edit", "id": msg.uuid, "content": content})
        with self.tx():
            if msg is not None:
                self._set_log_size(msg.session_id, size)
            self.conn.execute("UPDATE messages SET content = ? WHERE id = ?", (content, message_id))
            self.conn.execute("DELETE FROM messages_fts WHERE rowid = ?", (message_id,))
            self.conn.execute(
                "INSERT INTO messages_fts(rowid, norm_text) VALUES (?, ?)",
                (message_id, normalize_for_search(content)),
            )
            # the text changed: its search vector is re-made in the background
            self.conn.execute(
                "DELETE FROM embeddings WHERE owner_type = 'message' AND owner_id = ?",
                (message_id,),
            )

    def siblings(self, msg: Message) -> list[Message]:
        """Alternative versions of a message: same parent and role, oldest first."""
        rows = self.conn.execute(
            "SELECT * FROM messages WHERE session_id = ? AND role = ? AND parent_id IS ? "
            "ORDER BY id",
            (msg.session_id, msg.role, msg.parent_id),
        ).fetchall()
        return [_message(r) for r in rows]

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
