"""/char and /persona commands (mixed into the REPL)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.table import Table
from rich.text import Text

from chatclient.char_builder import (
    FIELD_TASKS,
    WizardAnswers,
    age_ok,
    build_card,
    generate_field,
    options,
    parse_choice,
    read_card_file,
    safe_filename,
    write_card_file,
)
from chatclient.characters import Card, Character, card_color, card_content, set_card_color
from chatclient.core.chat import ChatError
from chatclient.providers.base import ProviderError
from chatclient.providers.registry import parse_model_ref
from chatclient.theme import character_color, rich_style, valid_color
from chatclient.ui.picker import PickItem

if TYPE_CHECKING:
    from chatclient.ui.repl import Repl

EDITABLE_FIELDS = (
    "name",
    "description",
    "personality",
    "scenario",
    "first_mes",
    "mes_example",
    "post_history_instructions",
)

CHAR_SUBCOMMANDS = (
    "list", "new", "use", "show", "edit", "color", "ai", "import", "export", "delete",
)  # fmt: skip


def _ask_bindings() -> KeyBindings:
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    return kb


class CharacterCommands:
    """Mixin for :class:`Repl`; relies on its ``chat``, ``t``, ``console`` and helpers."""

    async def ask(self: Repl, question: str, default: str = "") -> str:
        session: PromptSession[str] = PromptSession(key_bindings=_ask_bindings(), multiline=True)
        self.heading(question)
        return (await session.prompt_async("  › ", default=default)).strip()

    def char_names(self: Repl) -> list[str]:
        return [c.name for c in self.chat.store.list_characters("ai")]

    def require_char(self: Repl, ref: str) -> Character:
        if not ref:
            raise ChatError(self.t("char_need_ref"))
        char = self.chat.store.find_character(ref, "ai")
        if char is None:
            raise ChatError(self.t("char_not_found", ref=ref))
        return char

    async def char_command(self: Repl, arg: str) -> None:
        sub, _, rest = arg.partition(" ")
        rest = rest.strip()
        match sub.lower():
            case "":
                char = self.chat.character()
                if char is None:
                    self.info(self.t("char_none"))
                else:
                    self.info(self.t("char_current", name=char.name, id=char.id))
            case "list" | "ls":
                self.list_chars()
            case "new" | "create":
                await self.create_char()
            case "use" | "start":
                if rest:
                    self.start_char(self.require_char(rest))
                elif (char := await self.pick_char()) is not None:
                    self.start_char(char)
            case "color" | "colour":
                self.char_color(rest)
            case "show":
                char = self.require_char(rest) if rest else self.chat.character()
                if char is None:
                    raise ChatError(self.t("char_none"))
                self.show_card(char.card)
            case "edit":
                await self.edit_char(self.require_char(rest) if rest else self.chat.character())
            case "delete" | "rm":
                await self.delete_char(self.require_char(rest))
            case "ai":
                await self.char_ai(rest)
            case "import":
                self.import_char(rest)
            case "export":
                self.export_char(rest)
            case _:
                # "/char <name>" is a shortcut for "/char use <name>"
                self.start_char(self.require_char(arg))

    async def pick_char(self: Repl) -> Character | None:
        chars = self.chat.store.list_characters("ai")
        if not chars:
            raise ChatError(self.t("char_list_hint"))
        items = [PickItem(c.id, c.name, ", ".join(c.card.tags[:3])) for c in chars]
        chosen = await self.pick(
            self.t("pick_char"), items, self.chat.require_session().character_id
        )
        return self.chat.store.get_character(chosen) if chosen is not None else None

    def char_color(self: Repl, value: str) -> None:
        char = self.chat.character()
        if char is None:
            raise ChatError(self.t("char_none"))
        value = value.strip().lower()
        if value in ("", "auto", "reset"):
            set_card_color(char.card, None)
        elif valid_color(value) and value != "default":
            set_card_color(char.card, value)
        else:
            raise ChatError(self.t("char_color_usage"))
        self.chat.store.update_character(char.id, char.card)
        color = self.assistant_color()
        self.console.print(
            self.t("char_color_set", name=char.name, color=color),
            style=rich_style(color, "bold"),
            markup=False,
        )

    def list_chars(self: Repl) -> None:
        table = Table(box=None, header_style=self.style("accent", "bold"), pad_edge=False)
        for col in ("#", "name", "tags", ""):
            table.add_column(col)
        current = self.chat.require_session().character_id
        for c in self.chat.store.list_characters("ai"):
            marker = "▸" if c.id == current else " "
            color = (
                character_color(c.name, self.theme, card_color(c.card))
                if self.ui.character_colors
                else self.theme.accent
            )
            table.add_row(
                f"{marker}{c.id}",
                Text(c.name, style=rich_style(color, "bold")),
                ", ".join(c.card.tags[:4]),
                self.t("builtin") if c.builtin_key else "",
            )
        self.console.print(table)
        self.info(self.t("char_list_hint"))

    def start_char(self: Repl, char: Character) -> None:
        self.chat.new_session(character_id=char.id)
        self.info(self.t("char_started", name=char.name, id=self.chat.require_session().id))
        if self.chat.require_session().model is None:
            self.warn(self.t("no_model"))
        self.show_history(3)

    def show_card(self: Repl, card: Card) -> None:
        for f in EDITABLE_FIELDS:
            value = getattr(card, f)
            if value.strip():
                self.heading(f)
                self.console.print(value.strip(), style=self.style("text"), markup=False)
        if card.tags:
            self.info("tags: " + ", ".join(card.tags))
        if content := card_content(card):
            self.info(f"content: {content}")

    async def ask_choice(self: Repl, question: str, key: str, max_n: int = 1) -> list[str]:
        opts = options(key, self.t.lang)
        menu = "   ".join(f"{i}) {o.label}" for i, o in enumerate(opts, 1))
        hint = self.t("choice_hint_multi", n=max_n) if max_n > 1 else self.t("choice_hint")
        self.heading(question)
        self.console.print(f"  {menu}", style=self.style("text"), markup=False)
        session: PromptSession[str] = PromptSession(key_bindings=_ask_bindings(), multiline=True)
        answer = await session.prompt_async(f"  {hint} › ")
        return parse_choice(answer, opts, max_n)

    async def create_char(self: Repl) -> None:
        """Wizard: multiple choice (or your own words) → template card; no LLM needed."""
        t = self.t
        try:
            name = await self.ask(t("q_name"))
            if not name:
                self.info(t("aborted"))
                return
            a = WizardAnswers(name=name)
            content = await self.ask_choice(t("q_content", name=name), "content")
            a.content = "mature" if content == ["mature"] else "standard"
            # identity
            while True:
                a.age = await self.ask(
                    t("q_age_mature" if a.content == "mature" else "q_age", name=name)
                )
                if age_ok(a.age, a.content):
                    break
                self.warn(t("age_adult_needed"))
            a.gender = await self.ask_choice(t("q_gender", name=name), "gender")
            a.concept = await self.ask(t("q_concept", name=name))
            a.appearance = await self.ask(t("q_appearance", name=name))
            # personality
            a.temperament = await self.ask_choice(t("q_temperament", name=name), "temperament", 2)
            a.speech = await self.ask_choice(t("q_speech", name=name), "speech")
            a.values = await self.ask_choice(t("q_values", name=name), "values", 2)
            a.fear = await self.ask_choice(t("q_fear", name=name), "fear")
            a.likes = await self.ask(t("q_likes", name=name))
            a.backstory = await self.ask(t("q_backstory", name=name))
            # the two of you, and the story
            a.relationship = await self.ask_choice(t("q_relationship"), "relationship")
            a.history = await self.ask(t("q_history", name=name))
            a.genre = await self.ask_choice(t("q_genre"), "genre")
            a.place = await self.ask(t("q_place"))
            if a.content == "mature":
                a.themes = await self.ask_choice(t("q_themes"), "themes", 6)
            a.example = await self.ask(t("q_example", name=name))
            card = build_card(a, t.lang)
            if await self.confirm(t("q_ai_help")):
                for field in ("first_mes", "mes_example"):
                    await self.ai_fill(card, field)
            if not card.first_mes:
                card.first_mes = await self.ask(t("q_first_mes", name=name))
            self.show_card(card)
            if not await self.confirm(t("q_save_char", name=name)):
                self.info(t("char_not_saved"))
                return
        except (KeyboardInterrupt, EOFError):
            self.info(t("aborted"))
            return
        char = self.chat.store.add_character("ai", card)
        self.info(t("char_created", name=name, id=char.id))
        try:
            start = await self.confirm(t("q_start_now", name=name))
        except (KeyboardInterrupt, EOFError):
            return
        if start:
            self.start_char(char)

    async def confirm(self: Repl, question: str, default: bool = True) -> bool:
        answer = (await self.ask(question)).lower()
        if not answer:
            return default
        return answer in ("y", "yes", "e", "evet")

    async def ai_fill(self: Repl, card: Card, field: str) -> bool:
        """Generate one field with the current model; the user accepts, retries or skips."""
        spec = self.chat.require_session().model or self.chat.default_model()
        if not spec:
            self.warn(self.t("no_model"))
            return False
        ref = parse_model_ref(spec)
        provider = self.chat.registry.get(ref.provider)
        for _ in range(3):
            self.info(self.t("ai_writing", field=self.t(f"field_{field}")))
            try:
                text = await generate_field(provider, ref.model, card, field, self.t.lang)
            except ProviderError as e:
                self.error(str(e).split("\n")[0])
                text = ""
            if not text:
                if not await self.confirm(self.t("ai_retry_q"), default=False):
                    return False
                continue
            self.console.print(text, markup=False)
            answer = (await self.ask(self.t("ai_accept_q"))).lower()
            if answer in ("", "y", "yes", "e", "evet"):
                setattr(card, field, text)
                return True
            if answer not in ("r", "retry", "t", "tekrar"):
                return False
        return False

    async def char_ai(self: Repl, rest: str) -> None:
        field = rest.strip() or "first_mes"
        if field not in FIELD_TASKS:
            raise ChatError(f"/char ai [{'|'.join(FIELD_TASKS)}]")
        char = self.chat.character()
        if char is None:
            raise ChatError(self.t("char_none"))
        try:
            if await self.ai_fill(char.card, field):
                self.chat.store.update_character(char.id, char.card)
                self.info(self.t("char_saved", name=char.name))
        except (KeyboardInterrupt, EOFError):
            self.info(self.t("aborted"))

    def import_char(self: Repl, rest: str, kind: str = "ai") -> None:
        path = Path(rest.strip().strip('"').strip("'")).expanduser()
        if not rest.strip():
            raise ChatError(f"/{'persona' if kind == 'user' else 'char'} import <file|folder>")
        if not path.exists():
            raise ChatError(self.t("file_not_found", path=str(path)))
        lib = self.chat.store.library
        try:
            if lib is not None:  # a package (.zip/folder), a card or a chat
                self.report_import(lib.import_path(self.chat.store, path, kind))
                return
            card = read_card_file(path)
        except (ValueError, UnicodeDecodeError, OSError) as e:
            raise ChatError(self.t("import_failed", error=str(e))) from e
        char = self.chat.store.add_character(kind, card)
        self.info(self.t("char_imported", name=card.name, id=char.id))

    def report_import(self: Repl, got: Any) -> None:
        self.info(
            self.t(f"imported_{got.kind}", name=got.name, chats=got.chats, memories=got.memories)
            + (self.t("imported_copy") if got.copy else "")
        )

    def export_char(self: Repl, rest: str, kind: str = "ai") -> None:
        """``/char export [name] [card]``: everything as a .zip, or just the card."""
        words = rest.split()
        card_only = bool(words) and words[-1].lower() == "card"
        ref = " ".join(words[:-1] if card_only else words)
        if kind == "user":
            char = self.chat.store.find_character(ref, "user") if ref else self.chat.persona()
        else:
            char = self.require_char(ref) if ref else self.chat.character()
        if char is None:
            raise ChatError(self.t("persona_none" if kind == "user" else "char_none"))
        lib = self.chat.store.library
        exports = self.paths.data_dir / "exports"
        if lib is None or card_only:
            path = write_card_file(char.card, exports / f"{safe_filename(char.name)}.json")
        else:
            path = lib.export_character(self.chat.store, char, exports)
        self.info(self.t("char_exported", name=char.name, path=str(path)))

    async def edit_char(self: Repl, char: Character | None) -> None:
        if char is None:
            raise ChatError(self.t("char_none"))
        self.info(self.t("edit_hint"))
        card = char.card
        try:
            for f in EDITABLE_FIELDS:
                value = await self.ask(f, default=getattr(card, f))
                if f == "name" and not value:
                    continue  # a card must keep a name
                setattr(card, f, value)
        except (KeyboardInterrupt, EOFError):
            self.info(self.t("aborted"))
            return
        self.chat.store.update_character(char.id, card)
        self.info(self.t("char_saved", name=card.name))

    async def delete_char(self: Repl, char: Character) -> None:
        used = self.chat.store.sessions_using(char.id)
        try:
            answer = await self.ask(self.t("confirm_delete_char", name=char.name, n=used))
        except (KeyboardInterrupt, EOFError):
            answer = ""
        if answer.lower() not in ("y", "yes", "e", "evet"):
            self.info(self.t("aborted"))
            return
        self.chat.delete_character(char)
        self.info(self.t("char_deleted", name=char.name))

    async def persona_command(self: Repl, arg: str) -> None:
        sub = arg.strip().lower()
        if sub in ("", "show"):
            p = self.chat.persona()
            if p is None:
                self.info(self.t("persona_none"))
            else:
                self.heading(p.name)
                if p.card.description.strip():
                    self.console.print(p.card.description, markup=False)
            return
        if sub == "clear":
            self.chat.clear_persona()
            self.info(self.t("persona_cleared"))
            return
        word, _, rest = arg.strip().partition(" ")
        if word.lower() == "export":
            self.export_char(rest, kind="user")
            return
        if word.lower() == "import":
            self.import_char(rest, kind="user")
            return
        if sub not in ("set", "edit"):
            raise ChatError("/persona [set|clear|export|import <file>]")
        current = self.chat.default_persona()
        try:
            name = await self.ask(self.t("q_persona_name"), current.name if current else "")
            if not name:
                self.info(self.t("aborted"))
                return
            about = await self.ask(
                self.t("q_persona_about"), current.card.description if current else ""
            )
        except (KeyboardInterrupt, EOFError):
            self.info(self.t("aborted"))
            return
        self.chat.set_persona(Card(name=name, description=about))
        self.info(self.t("persona_saved", name=name))
