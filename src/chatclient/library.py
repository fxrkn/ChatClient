"""Characters, personas and chats as plain files: the source of truth.

    characters/<name>/card.json          the character (CCv3; editable by hand)
    characters/<name>/chats/<date>_<id>.jsonl
    characters/<name>/memory.jsonl       snapshot of what memory learned
    personas/<name>/persona.json
    chats/<date>_<id>.jsonl              chats without a character

A chat file is an append-only log of events, one JSON object per line: a header
(``session``), then ``msg``, ``edit``, ``meta`` and ``set`` events. Every change is
written to the file first and then to the database, which is only an index: search,
vectors and memory work stay fast, and a lost or deleted database is rebuilt from the
files (``Library.sync`` at start-up also takes in files copied in or edited by hand).

Memories are worked out by a model over time, so they are worth keeping too: each
character's ``memory.jsonl`` is refreshed after memory work and restores them.

Ids in the files are UUIDs, so a folder or a chat can move between machines.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chatclient.characters import Card, Character
from chatclient.timeutil import to_iso, utcnow

if TYPE_CHECKING:
    from chatclient.storage import Message, Session, Store

log = logging.getLogger(__name__)

LAYOUT_VERSION = 1
MARKER = "library.json"
CARD_FILE = {"ai": "card.json", "user": "persona.json"}
MEMORY_FILE = "memory.jsonl"
_WIN_RESERVED = {"con", "prn", "aux", "nul"} | {
    f"{p}{i}" for p in ("com", "lpt") for i in range(10)
}


def slug(name: str) -> str:
    """A folder name that is safe on Windows and Linux; keeps letters of any language."""
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "", name).strip().strip(".")
    s = re.sub(r"\s+", "_", s)[:60] or "unnamed"
    return f"{s}_" if s.lower() in _WIN_RESERVED else s


def write_atomic(path: Path, text: str) -> None:
    """Write to a temporary file, then swap it in: a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def card_id(card: Card) -> str | None:
    ext = card.extra.get("extensions")
    ours = ext.get("chatclient") if isinstance(ext, dict) else None
    value = ours.get("id") if isinstance(ours, dict) else None
    return value if isinstance(value, str) and value else None


def set_card_id(card: Card, char_uuid: str | None) -> None:
    ext = card.extra.setdefault("extensions", {})
    if not isinstance(ext, dict):
        ext = card.extra["extensions"] = {}
    ours = ext.setdefault("chatclient", {})
    if char_uuid:
        ours["id"] = char_uuid
    else:
        ours.pop("id", None)


def read_events(path: Path, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Complete lines from ``offset`` on, and where they end. A half-written last line
    (a crash mid-write) is left for later; unreadable lines are skipped."""
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    *lines, rest = data.split(b"\n")
    events = []
    for raw in lines:
        if not raw.strip():
            continue
        try:
            ev = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            log.warning("skipping an unreadable line in %s", path)
            continue
        if isinstance(ev, dict):
            events.append(ev)
    return events, offset + len(data) - len(rest)


def parse_lines(text: str) -> list[dict[str, Any]]:
    out = []
    for line in text.splitlines():
        if line.strip():
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(ev, dict):
                out.append(ev)
    return out


class IdMap:
    """Ids while reading events in. Normally kept as they are; for a copy (the thing
    already exists here) every session and message gets a new id."""

    def __init__(self, fresh: bool = False):
        self.fresh = fresh
        self.map: dict[str, str] = {}

    def new(self, old: str | None) -> str:
        from chatclient.storage import new_uuid

        if not old:
            return new_uuid()
        if not self.fresh:
            return old
        return self.map.setdefault(old, new_uuid())

    def ref(self, old: str | None) -> str | None:
        return self.map.get(old, old) if old else None


@dataclass
class Imported:
    kind: str  # "character" | "persona" | "chat"
    name: str
    chats: int = 0
    memories: int = 0
    copy: bool = False  # it already existed, so a copy was made


class Library:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.characters_dir = self.root / "characters"
        self.personas_dir = self.root / "personas"
        self.chats_dir = self.root / "chats"
        self.exports_dir = self.root / "exports"
        self._folders: dict[str, Path] = {}

    # ── folders ─────────────────────────────────────────────────────────────

    def base(self, kind: str) -> Path:
        return self.personas_dir if kind == "user" else self.characters_dir

    def folder(self, char: Character) -> Path:
        """The character's folder (created, with a unique name, when it has none)."""
        known = self._folders.get(char.uuid)
        if known is not None and known.is_dir():
            return known
        base = self.base(char.kind)
        if base.is_dir():  # not cached yet: look for the folder holding this id
            for d in base.iterdir():
                if d.is_dir() and self._read_card_id(d / CARD_FILE[char.kind]) == char.uuid:
                    self._folders[char.uuid] = d
                    return d
        folder = self._free_name(base, slug(char.card.name))
        folder.mkdir(parents=True)
        self._folders[char.uuid] = folder
        return folder

    @staticmethod
    def _free_name(base: Path, name: str) -> Path:
        candidate, n = base / name, 2
        while candidate.exists():
            candidate, n = base / f"{name}-{n}", n + 1
        return candidate

    @staticmethod
    def _read_card_id(path: Path) -> str | None:
        try:
            return card_id(Card.from_json(path.read_text(encoding="utf-8-sig")))
        except (OSError, ValueError):
            return None

    def write_card(self, char: Character) -> Path:
        set_card_id(char.card, char.uuid)  # the id travels with the card
        folder = self.folder(char)
        wanted = slug(char.card.name)
        if folder.name != wanted and not folder.name.startswith(wanted + "-"):
            target = self._free_name(folder.parent, wanted)
            try:  # renamed character: rename its folder too
                os.replace(folder, target)
                folder = self._folders[char.uuid] = target
            except OSError:
                log.warning("could not rename %s", folder)
        path = folder / CARD_FILE[char.kind]
        write_atomic(path, json.dumps(char.card.to_dict(), ensure_ascii=False, indent=2) + "\n")
        return path

    def remove_card(self, char: Character) -> None:
        folder = self._folders.pop(char.uuid, None) or self.folder(char)
        base = self.base(char.kind).resolve()
        if folder.resolve().parent == base:  # never delete anything outside our folders
            shutil.rmtree(folder, ignore_errors=True)

    # ── chat files ──────────────────────────────────────────────────────────

    def chat_path(self, store: Store, s: Session) -> Path:
        name = f"{s.created_at[:10]}_{s.uuid[:8]}.jsonl"
        char = store.get_character(s.character_id)
        if char is not None and char.kind == "ai":
            return self.folder(char) / "chats" / name
        return self.chats_dir / name

    def append(self, store: Store, s: Session, event: dict[str, Any]) -> int:
        """Add one event to the session's file; returns the file's new size."""
        path = self.chat_path(store, s)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
        with open(path, "a+b") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":  # a crash cut the last line: start a fresh one
                    line = b"\n" + line
                f.seek(0, os.SEEK_END)
            f.write(line)
            f.flush()
            return f.tell()

    def header(self, store: Store, s: Session) -> dict[str, Any]:
        return {
            "t": "session", "v": LAYOUT_VERSION, "id": s.uuid, "at": s.created_at,
            "title": s.title, "model": s.model, "system_prompt": s.system_prompt,
            "params": s.params, "private": s.private,
            "character": store.uuid_of_character(s.character_id),
            "persona": store.uuid_of_character(s.persona_id),
        }  # fmt: skip

    def set_event(self, store: Store, fields: dict[str, Any]) -> dict[str, Any]:
        ev: dict[str, Any] = {"t": "set", "at": fields.get("_at") or to_iso(utcnow())}
        for key, value in fields.items():
            if key == "_at":
                continue
            if key == "head_message_id":
                ev["head"] = store.uuid_of_message(value)
            elif key == "character_id":
                ev["character"] = store.uuid_of_character(value)
            elif key == "persona_id":
                ev["persona"] = store.uuid_of_character(value)
            else:
                ev[key] = bool(value) if key in ("archived", "private") else value
        return ev

    @staticmethod
    def message_event(store: Store, m: Message, head: bool) -> dict[str, Any]:
        return {
            "t": "msg", "id": m.uuid, "parent": store.uuid_of_message(m.parent_id),
            "role": m.role, "content": m.content, "at": m.created_at, "day": m.local_day,
            "model": m.model, "tokens": [m.prompt_tokens, m.completion_tokens],
            "meta": m.meta or {}, "head": head,
        }  # fmt: skip

    def move_chat(self, store: Store, before: Session, after: Session) -> None:
        src, dst = self.chat_path(store, before), self.chat_path(store, after)
        if src != dst and src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)

    def remove_chat(self, store: Store, s: Session) -> None:
        self.chat_path(store, s).unlink(missing_ok=True)

    def write_chat(self, store: Store, s: Session) -> Path:
        """Write a whole chat file from the database (first run, repairs)."""
        events = [self.header(store, s)]
        events += [self.message_event(store, m, False) for m in store.session_messages(s.id)]
        events.append(
            {
                "t": "set", "at": s.updated_at, "head": store.uuid_of_message(s.head_message_id),
                "scene": s.scene or None, "archived": s.archived,
            }
        )  # fmt: skip
        path = self.chat_path(store, s)
        write_atomic(path, "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in events))
        store.conn.execute(
            "UPDATE sessions SET log_size = ? WHERE id = ?", (path.stat().st_size, s.id)
        )
        return path

    # ── reading events back in ──────────────────────────────────────────────

    def apply(
        self,
        store: Store,
        events: list[dict[str, Any]],
        ids: IdMap | None = None,
        session: Session | None = None,
    ) -> Session | None:
        """Replay events into the store. In quiet mode (start-up) only the index is
        written; otherwise (import) the store writes the files as it goes."""
        ids = ids or IdMap()
        for ev in events:
            t = ev.get("t")
            if t == "session":
                suid = ids.new(ev.get("id"))
                session = store.session_by_uuid(suid)
                if session is None:
                    char = store.character_by_uuid(ids.ref(ev.get("character")))
                    persona = store.character_by_uuid(ids.ref(ev.get("persona")))
                    session = store.create_session(
                        ev.get("model"), ev.get("title") or "",
                        character_id=char.id if char else None,
                        persona_id=persona.id if persona else None,
                        session_uuid=suid, created_at=ev.get("at"),
                    )  # fmt: skip
                    extra = {k: ev[k] for k in ("system_prompt", "params", "private") if ev.get(k)}
                    if extra:
                        store.update_session(session.id, **extra, _at=ev.get("at"))
            elif session is None:
                continue  # nothing to attach it to
            elif t == "msg":
                muid = ids.new(ev.get("id"))
                if store.message_id(muid) is not None:
                    continue
                pt, ct = ([*(ev.get("tokens") or []), None, None])[:2]
                store.add_message(
                    session.id, store.message_id(ids.ref(ev.get("parent"))),
                    ev.get("role") or "user", ev.get("content") or "",
                    model=ev.get("model"), prompt_tokens=pt, completion_tokens=ct,
                    meta=ev.get("meta") or {}, set_head=bool(ev.get("head", True)),
                    message_uuid=muid, created_at=ev.get("at"), day=ev.get("day"),
                )  # fmt: skip
            elif t in ("edit", "meta"):
                mid = store.message_id(ids.ref(ev.get("id")))
                if mid is None:
                    continue
                if t == "edit":
                    store.update_message_content(mid, ev.get("content") or "")
                else:
                    store.set_message_meta(mid, ev.get("meta") or {})
            elif t == "set":
                fields: dict[str, Any] = {}
                for key, value in ev.items():
                    if key == "head":
                        fields["head_message_id"] = store.message_id(ids.ref(value))
                    elif key in ("character", "persona"):
                        c = store.character_by_uuid(ids.ref(value))
                        fields[f"{key}_id"] = c.id if c else None
                    elif key == "scene":
                        store.set_scene(session.id, value or {})
                    elif key in ("title", "model", "system_prompt", "params"):
                        fields[key] = value
                    elif key in ("archived", "private"):
                        fields[key] = int(bool(value))
                if fields:
                    store.update_session(session.id, **fields, _at=ev.get("at"))
            if session is not None:
                session = store.get_session(session.id)
        return session

    # ── start-up ────────────────────────────────────────────────────────────

    def sync(self, store: Store) -> None:
        """Bring the index up to date with the files. Cheap when nothing changed (it
        only looks at file sizes); the first run writes the files from the database."""
        if not (self.root / MARKER).exists():
            self.migrate(store)
            return
        with store.quiet():
            cards_ok = self._sync_cards(store)
            self._sync_chats(store, delete_missing=cards_ok)
            self._sync_memory(store)

    def _card_folders(self) -> list[tuple[str, Path]]:
        out = []
        for kind in ("ai", "user"):
            base = self.base(kind)
            if base.is_dir():
                out += [(kind, d) for d in sorted(base.iterdir()) if d.is_dir()]
        return out

    def _sync_cards(self, store: Store) -> bool:
        ok, seen = True, set()
        for kind, folder in self._card_folders():
            path = folder / CARD_FILE[kind]
            if not path.is_file():
                continue
            try:
                card = Card.from_json(path.read_text(encoding="utf-8-sig"))
            except (OSError, ValueError):
                log.warning("unreadable card: %s", path)
                ok = False
                continue
            cid = card_id(card)
            if cid is None or cid in seen:  # new by hand, or a copied folder: a new id
                from chatclient.storage import new_uuid

                cid = new_uuid()
                set_card_id(card, cid)
                write_atomic(path, json.dumps(card.to_dict(), ensure_ascii=False, indent=2) + "\n")
            seen.add(cid)
            self._folders[cid] = folder
            existing = store.character_by_uuid(cid)
            if existing is None:
                store.add_character(kind, card, char_uuid=cid)
            elif existing.card.to_json() != card.to_json():
                store.update_character(existing.id, card)
        if ok:  # a folder that is gone means a character that is gone
            for char in store.all_characters():
                if char.uuid not in seen:
                    store.delete_character(char.id)
        return ok

    def _chat_files(self) -> list[Path]:
        files = sorted(self.chats_dir.glob("*.jsonl")) if self.chats_dir.is_dir() else []
        if self.characters_dir.is_dir():
            files += sorted(self.characters_dir.glob("*/chats/*.jsonl"))
        return files

    def _sync_chats(self, store: Store, delete_missing: bool) -> None:
        ok, seen = delete_missing, set()
        for path in self._chat_files():
            try:
                with open(path, encoding="utf-8") as f:
                    head = json.loads(f.readline())
                suid = head["id"] if head.get("t") == "session" else None
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                suid = None
            if not suid:
                log.warning("unreadable chat file: %s", path)
                ok = False
                continue
            seen.add(suid)
            size = path.stat().st_size
            s = store.session_by_uuid(suid)
            if s is not None and size == s.log_size:
                continue  # unchanged: the usual case
            offset = 0
            if s is not None and size > s.log_size:
                offset = s.log_size  # appended to: take in only the new part
            elif s is not None:  # shorter or rewritten: read it again from the top
                store.delete_session(s.id)
                s = None
            events, end = read_events(path, offset)
            s = self.apply(store, events, session=s)
            if s is not None:
                store.conn.execute("UPDATE sessions SET log_size = ? WHERE id = ?", (end, s.id))
        if ok:
            for s in store.all_sessions():
                if s.uuid not in seen:
                    store.delete_session(s.id)

    def _sync_memory(self, store: Store) -> None:
        for kind, folder in self._card_folders():
            path = folder / MEMORY_FILE
            if kind != "ai" or not path.is_file():
                continue
            char = store.character_by_uuid(self._read_card_id(folder / CARD_FILE[kind]))
            if char is not None and not self._has_memory(store, char):
                self.load_memory(store, parse_lines(path.read_text(encoding="utf-8")), char)

    @staticmethod
    def _has_memory(store: Store, char: Character) -> bool:
        row = store.conn.execute(
            "SELECT (SELECT COUNT(*) FROM memories WHERE character_id = ?) + "
            "(SELECT COUNT(*) FROM summaries x JOIN sessions s ON s.id = x.session_id "
            "WHERE s.character_id = ?)",
            (char.id, char.id),
        ).fetchone()
        return bool(row[0])

    def migrate(self, store: Store) -> None:
        """First run with files: write everything the database holds."""
        for d in (self.characters_dir, self.personas_dir, self.chats_dir):
            d.mkdir(parents=True, exist_ok=True)
        for char in store.all_characters():
            self.write_card(char)
            store.conn.execute(
                "UPDATE characters SET card_json = ? WHERE id = ?", (char.card.to_json(), char.id)
            )
        for s in store.all_sessions():
            self.write_chat(store, s)
        for char in store.all_characters():
            if char.kind == "ai" and self._has_memory(store, char):
                self.write_memory(store, char)
        write_atomic(self.root / MARKER, json.dumps({"layout": LAYOUT_VERSION}) + "\n")
        log.info("library written to %s", self.root)

    # ── memory snapshots ────────────────────────────────────────────────────

    def write_memory(self, store: Store, char: Character) -> Path:
        rows = store.conn.execute(
            """SELECT m.*, s.uuid AS s_uuid, src.uuid AS src_uuid FROM memories m
               LEFT JOIN sessions s ON s.id = m.session_id
               LEFT JOIN messages src ON src.id = m.source_message_id
               WHERE (m.character_id = ? OR s.character_id = ?)
                 AND COALESCE(s.private, 0) = 0 ORDER BY m.id""",
            (char.id, char.id),
        ).fetchall()
        lines = [
            {
                "t": "memory", "id": r["id"], "session": r["s_uuid"], "source": r["src_uuid"],
                "content": r["content"], "kind": r["kind"], "importance": r["importance"],
                "pinned": bool(r["pinned"]), "day": r["day"], "story_time": r["story_time"],
                "place": r["place"], "people": json.loads(r["people_json"] or "[]"),
                "subject": r["subject"], "attribute": r["attribute"], "value": r["value"],
                "slot": r["slot"], "said_by": r["said_by"], "superseded_by": r["superseded_by"],
                "at": r["created_at"], "use_count": r["use_count"],
            }
            for r in rows
        ]  # fmt: skip
        rows = store.conn.execute(
            """SELECT x.*, s.uuid AS s_uuid, a.uuid AS a_uuid, b.uuid AS b_uuid
               FROM summaries x JOIN sessions s ON s.id = x.session_id
               LEFT JOIN messages a ON a.id = x.start_message_id
               LEFT JOIN messages b ON b.id = x.end_message_id
               WHERE s.character_id = ? AND s.private = 0 ORDER BY x.id""",
            (char.id,),
        ).fetchall()
        lines += [
            {
                "t": "summary", "session": r["s_uuid"], "kind": r["kind"], "day": r["day"],
                "start": r["a_uuid"], "end": r["b_uuid"], "content": r["content"],
            }
            for r in rows
        ]  # fmt: skip
        path = self.folder(char) / MEMORY_FILE
        write_atomic(path, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines))
        return path

    def load_memory(
        self, store: Store, lines: list[dict[str, Any]], char: Character, ids: IdMap | None = None
    ) -> int:
        from chatclient.memory.repo import MemoryRepo, Scope

        ids = ids or IdMap()
        repo = MemoryRepo(store)
        new_ids: dict[int, int] = {}
        for ev in lines:
            if ev.get("t") == "memory":
                s = store.session_by_uuid(ids.ref(ev.get("session")))
                mem = repo.add_memory(
                    Scope(s.id if s else None, char.id), ev.get("content") or "",  # type: ignore[arg-type]
                    kind=ev.get("kind") or "fact", importance=float(ev.get("importance") or 0.5),
                    pinned=bool(ev.get("pinned")), day=ev.get("day"),
                    story_time=ev.get("story_time"), place=ev.get("place"),
                    people=ev.get("people") or [],
                    source_message_id=store.message_id(ids.ref(ev.get("source"))),
                    subject=ev.get("subject"), attribute=ev.get("attribute"),
                    value=ev.get("value"), said_by=ev.get("said_by"), key=ev.get("slot"),
                )  # fmt: skip
                store.conn.execute(
                    "UPDATE memories SET created_at = COALESCE(?, created_at), use_count = ? "
                    "WHERE id = ?",
                    (ev.get("at"), int(ev.get("use_count") or 0), mem.id),
                )
                if isinstance(ev.get("id"), int):
                    new_ids[ev["id"]] = mem.id
            elif ev.get("t") == "summary":
                s = store.session_by_uuid(ids.ref(ev.get("session")))
                if s is not None:
                    repo.add_summary(
                        s.id, ev.get("kind") or "segment", ev.get("content") or "",
                        start_message_id=store.message_id(ids.ref(ev.get("start"))),
                        end_message_id=store.message_id(ids.ref(ev.get("end"))),
                        day=ev.get("day"),
                    )  # fmt: skip
        for ev in lines:  # second pass: which memory replaced which
            old, newer = ev.get("id"), ev.get("superseded_by")
            if ev.get("t") == "memory" and old in new_ids and newer in new_ids:
                store.conn.execute(
                    "UPDATE memories SET superseded_by = ? WHERE id = ?",
                    (new_ids[newer], new_ids[old]),
                )
        return len(new_ids)

    # ── export / import ─────────────────────────────────────────────────────

    def export_character(
        self, store: Store, char: Character, dest: Path, *, chats: bool = True, memory: bool = True
    ) -> Path:
        """A zip: card.json (+ chats/*.jsonl, memory.jsonl). Private chats stay home."""
        card_path = self.write_card(char)
        dest.mkdir(parents=True, exist_ok=True)
        out = dest / f"{slug(char.card.name)}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(card_path, CARD_FILE[char.kind])
            if chats and char.kind == "ai":
                for s in store.all_sessions():
                    path = self.chat_path(store, s)
                    if s.character_id == char.id and not s.private and path.is_file():
                        z.write(path, f"chats/{path.name}")
            if memory and char.kind == "ai" and self._has_memory(store, char):
                z.write(self.write_memory(store, char), MEMORY_FILE)
        return out

    def import_path(self, store: Store, path: Path, kind: str = "ai") -> Imported:
        """A character package (.zip or folder), a card (.json/.png) or a chat (.jsonl)."""
        from chatclient.char_builder import read_card_file

        if path.is_dir() or zipfile.is_zipfile(path):
            files = _read_package(path)
            name = CARD_FILE["user"] if CARD_FILE["user"] in files else CARD_FILE["ai"]
            if name not in files:
                raise ValueError("no card.json or persona.json in it")
            kind = "user" if name == CARD_FILE["user"] else "ai"
            return self._import_package(store, Card.from_json(files[name]), kind, files)
        if path.suffix.lower() == ".jsonl":
            return self.import_chat(store, path.read_text(encoding="utf-8-sig"))
        return self._import_package(store, read_card_file(path), kind, {})

    def _import_package(
        self, store: Store, card: Card, kind: str, files: dict[str, str]
    ) -> Imported:
        cid = card_id(card)
        copy = cid is not None and store.character_by_uuid(cid) is not None
        ids = IdMap(fresh=copy)
        from chatclient.storage import new_uuid

        new_cid = new_uuid() if copy or not cid else cid
        if cid:
            ids.map[cid] = new_cid
        char = store.add_character(kind, card, char_uuid=new_cid)
        result = Imported("persona" if kind == "user" else "character", char.name, copy=copy)
        for name in sorted(files):
            is_chat = name.startswith("chats/") and name.endswith(".jsonl")
            if is_chat and self.apply(store, parse_lines(files[name]), ids) is not None:
                result.chats += 1
        if MEMORY_FILE in files and kind == "ai":
            result.memories = self.load_memory(store, parse_lines(files[MEMORY_FILE]), char, ids)
            if store.files is not None:
                self.write_memory(store, char)
        return result

    def import_chat(self, store: Store, text: str) -> Imported:
        events = parse_lines(text)
        head = next((e for e in events if e.get("t") == "session"), None)
        if head is None:
            raise ValueError("not a chat file (no session header)")
        copy = store.session_by_uuid(head.get("id")) is not None
        s = self.apply(store, events, IdMap(fresh=copy))
        return Imported("chat", (s.title if s else "") or "chat", chats=1, copy=copy)


def _read_package(path: Path) -> dict[str, str]:
    """Text files of a package, by their path inside it. A zip may wrap everything in
    one top folder (as when a folder is zipped by hand); that folder is ignored."""
    out: dict[str, str] = {}
    if path.is_dir():
        for f in path.rglob("*"):
            if f.is_file() and f.suffix in (".json", ".jsonl"):
                out[f.relative_to(path).as_posix()] = f.read_text(encoding="utf-8-sig")
    else:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                if not info.is_dir() and info.filename.endswith((".json", ".jsonl")):
                    if info.file_size > 200 * 1024 * 1024:
                        raise ValueError(f"{info.filename} is too large")
                    out[info.filename] = z.read(info).decode("utf-8-sig")
    if not any(k in out for k in CARD_FILE.values()):
        tops = {k.split("/", 1)[0] for k in out if "/" in k}
        if len(tops) == 1:
            top = tops.pop() + "/"
            out = {k[len(top) :]: v for k, v in out.items() if k.startswith(top)}
    return out
