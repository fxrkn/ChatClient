"""Everyday conversation tools: /edit, /continue, /ooc, /copy, /export, /lang, Ctrl+P."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

from chatclient.char_builder import safe_filename
from chatclient.core.chat import ChatError
from chatclient.export import FORMATS, ExportMessage, render
from chatclient.i18n import COMMAND_DOCS, LANG_KEY
from chatclient.timeutil import from_iso, to_local
from chatclient.ui.clipboard import copy_text
from chatclient.ui.picker import PickItem


def open_in_file_manager(path: Path) -> None:
    """Explorer on Windows, Finder on macOS, the desktop's file manager on Linux."""
    try:
        if sys.platform == "win32":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", str(path)])
    except OSError:
        pass  # no file manager (a server, a container): the path was printed anyway


if TYPE_CHECKING:
    from chatclient.ui.repl import Repl


class ToolCommands:
    # ── /edit ───────────────────────────────────────────────────────────────

    async def edit_command(self: Repl, arg: str) -> None:
        try:
            if arg.strip().lower() in ("reply", "ai", "r"):
                msg = self.chat.last_reply()
                if msg is None:
                    raise ChatError(self.t("edit_no_reply"))
                text = await self.ask(self.t("edit_reply_q"), default=msg.content)
                if not text or text == msg.content.strip():
                    self.info(self.t("aborted"))
                    return
                new = self.chat.replace_reply(text)
                sys.stdout.write("\n")
                self.write_message(
                    self.assistant_label(new.model), self.assistant_color(), new.content
                )
                self.info(self.t("edit_reply_saved"))
                return
            msg = self.chat.last_user_message()
            if msg is None:
                raise ChatError(self.t("edit_no_message"))
            text = await self.ask(self.t("edit_q"), default=msg.content)
        except (KeyboardInterrupt, EOFError):
            self.info(self.t("aborted"))
            return
        if not text or text == msg.content.strip():
            self.info(self.t("aborted"))
            return
        await self.reply(lambda r: self.chat.edit_and_resend(text, r))

    # ── /ooc ────────────────────────────────────────────────────────────────

    async def ooc_command(self: Repl, text: str) -> None:
        if not text.strip():
            raise ChatError(self.t("ooc_usage"))
        message = f"(OOC: {text.strip()})"
        await self.reply(lambda r: self.chat.send(message, r))

    # ── /copy ───────────────────────────────────────────────────────────────

    def copy_command(self: Repl, arg: str) -> None:
        if arg.strip().lower() == "all":
            text = render("txt", *self._export_parts())
        else:
            msg = self.chat.last_reply()
            if msg is None:
                raise ChatError(self.t("edit_no_reply"))
            text = msg.content
        how = copy_text(text)
        self.info(self.t("copied" if how == "system" else "copied_terminal"))

    # ── /export ─────────────────────────────────────────────────────────────

    def _export_parts(self: Repl) -> tuple[str, str, list[ExportMessage]]:
        s = self.chat.require_session()
        msgs = self.chat.history()
        if not msgs:
            raise ChatError(self.t("export_empty"))
        you = self.user_label()
        items = []
        for m in msgs:
            when = to_local(from_iso(m.created_at), self.chat.store.tz).strftime("%Y-%m-%d %H:%M")
            name = you if m.role == "user" else self.assistant_label(m.model)
            items.append(ExportMessage(name, m.content, m.role == "user", when))
        title = s.title or self.t("untitled")
        subtitle = " · ".join(x for x in (s.model or "", f"{len(msgs)} messages") if x)
        return title, subtitle, items

    def export_command(self: Repl, arg: str) -> None:
        fmt = (arg.strip().lower() or "md").lstrip(".")
        if fmt == "markdown":
            fmt = "md"
        lib = self.chat.store.library
        if fmt == "jsonl" and lib is not None:
            # the chat's own file: re-importable anywhere with /import
            s = self.chat.refresh()
            src = lib.chat_path(self.chat.store, s)
            if not src.is_file():
                raise ChatError(self.t("export_empty"))
            name = f"{safe_filename(s.title or 'chat')[:60]}-{date.today().isoformat()}.jsonl"
            path = self.paths.data_dir / "exports" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, path)
            self.info(self.t("exported", path=str(path)))
            return
        if fmt not in FORMATS:
            raise ChatError(self.t("export_usage", formats=", ".join([*FORMATS, "jsonl"])))
        title, subtitle, items = self._export_parts()
        content = render(fmt, title, subtitle, items)
        name = f"{safe_filename(title)[:60]}-{date.today().isoformat()}.{fmt}"
        path = self.paths.data_dir / "exports" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.info(self.t("exported", path=str(path)))

    def import_command(self: Repl, arg: str) -> None:
        """Anything exported before: a character package, a card, or a chat file."""
        if self.chat.store.library is None:
            raise ChatError(self.t("import_failed", error="no library"))
        self.import_char(arg)

    def open_command(self: Repl, arg: str) -> None:
        """Show a folder in the file manager: data, characters, personas, chats,
        exports, or a character by name (default: the current character)."""
        lib = self.chat.store.library
        root = self.paths.data_dir
        what = arg.strip()
        folders = {
            "data": root,
            "characters": root / "characters",
            "personas": root / "personas",
            "plain-chats": root / "plain-chats",
            "chats": root / "plain-chats",
            "exports": root / "exports",
        }
        if what.lower() in folders:
            target = folders[what.lower()]
        else:
            char = self.require_char(what) if what else self.chat.character()
            target = root if char is None or lib is None else lib.folder(char)
        target.mkdir(parents=True, exist_ok=True)
        self.info(str(target))
        open_in_file_manager(target)

    # ── /lang ───────────────────────────────────────────────────────────────

    def lang_command(self: Repl, arg: str) -> None:
        value = arg.strip().lower()
        if value not in ("en", "tr", "auto"):
            raise ChatError(self.t("lang_usage"))
        self.chat.store.kv_set(LANG_KEY, value)
        self.t.set_language(value)
        self.info(self.t("lang_set", lang=self.t.lang))

    # ── content, privacy, declined replies ──────────────────────────────────

    def private_command(self: Repl, arg: str) -> None:
        value = arg.strip().lower()
        if value in ("", "on", "off"):
            if value == "":
                value = "off" if self.chat.require_session().private else "on"
            self.chat.set_private(value == "on")
            self.info(self.t("private_on" if value == "on" else "private_off"))
            return
        raise ChatError("/private [on|off]")

    def content_command(self: Repl, arg: str) -> None:
        value = arg.strip().lower()
        if value:
            self.chat.set_content_mode(value)
        char = self.chat.character()
        if char is not None:
            self.info(self.t("content_mode_char", mode=self.chat.content_mode(), name=char.name))
        else:
            self.info(self.t("content_mode", mode=self.chat.content_mode()))

    def utility_command(self: Repl, arg: str) -> None:
        """/utility [provider:model | chat | reset]: the model for memory work."""
        value = arg.strip()
        if value.lower() in ("reset", "default", "off", "none"):
            self.chat.set_utility_model(None)
        elif value.lower() == "chat":
            self.chat.set_utility_model("chat")
        elif value:
            self.chat.set_utility_model(value)
        current = self.chat.utility_model()
        chat_model = self.chat.require_session().model
        if current and current != chat_model:
            self.info(self.t("utility_set", model=current))
        else:
            self.info(self.t("utility_chat", model=chat_model or "-"))
        engine = self.chat.memory
        if value and engine is not None:
            engine.failures = 0  # a new model gets a fresh start (and a fresh warning)
            engine.brief_failures = 0
            engine.last_brief_error = ""
            self.memory_warned = False
            self.brief_warned = False

    def fallback_command(self: Repl, arg: str) -> None:
        value = arg.strip()
        if value.lower() in ("off", "none"):
            self.chat.set_refusal_fallback(None)
        elif value:
            self.chat.set_refusal_fallback(value)
        current = self.chat.refusal_fallback()
        self.info(self.t("fallback_set", model=current) if current else self.t("fallback_off"))

    # ── Ctrl+P command palette ──────────────────────────────────────────────

    async def palette(self: Repl) -> str | None:
        """Pick a command; returns text to prefill (commands needing arguments) or None.
        Commands without arguments run straight away."""
        lang = self.t.lang
        items = [PickItem(doc, doc.usage(lang), doc.describe(lang)) for doc in COMMAND_DOCS]
        doc = await self.pick(self.t("palette_title"), items)
        if doc is None:
            return None
        if doc.args:
            return doc.command + " "
        await self.handle_command(doc.command)
        return None
