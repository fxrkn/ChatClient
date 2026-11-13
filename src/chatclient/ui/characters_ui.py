"""/char and /persona commands (mixed into the REPL)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from rich.table import Table

from chatclient.char_builder import (
    FIELD_TASKS,
    WizardAnswers,
    build_card,
    generate_field,
    options,
    parse_choice,
    read_card_file,
    safe_filename,
    write_card_file,
)
from chatclient.characters import Card, Character
from chatclient.core.chat import ChatError
from chatclient.providers.base import ProviderError
from chatclient.providers.registry import parse_model_ref

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

CHAR_SUBCOMMANDS = ("list", "new", "use", "show", "edit", "ai", "import", "export", "delete")


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
                self.start_char(self.require_char(rest))
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
                # "/char Elif" is a shortcut for "/char use Elif"
                self.start_char(self.require_char(arg))

    def list_chars(self: Repl) -> None:
        table = Table(box=None, header_style=self.style("accent", "bold"), pad_edge=False)
        for col in ("#", "name", "tags", ""):
            table.add_column(col)
        current = self.chat.require_session().character_id
        for c in self.chat.store.list_characters("ai"):
            marker = "▸" if c.id == current else " "
            table.add_row(
                f"{marker}{c.id}",
                c.name,
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
            a.concept = await self.ask(t("q_concept", name=name))
            a.temperament = await self.ask_choice(t("q_temperament", name=name), "temperament", 2)
            a.speech = await self.ask_choice(t("q_speech", name=name), "speech")
            a.values = await self.ask_choice(t("q_values", name=name), "values", 2)
            a.fear = await self.ask_choice(t("q_fear", name=name), "fear")
            a.relationship = await self.ask_choice(t("q_relationship"), "relationship")
            a.genre = await self.ask_choice(t("q_genre"), "genre")
            a.place = await self.ask(t("q_place"))
            a.example = await self.ask(t("q_example", name=name))
            card = build_card(a, t.lang)
            if await self.confirm(t("q_ai_help")):
                for field in ("first_mes", "mes_example"):
                    await self.ai_fill(card, field)
            if not card.first_mes:
                card.first_mes = await self.ask(t("q_first_mes", name=name))
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

    def import_char(self: Repl, rest: str) -> None:
        path = Path(rest.strip().strip('"').strip("'")).expanduser()
        if not rest.strip():
            raise ChatError("/char import <file.json|file.png>")
        if not path.is_file():
            raise ChatError(self.t("file_not_found", path=str(path)))
        try:
            card = read_card_file(path)
        except (ValueError, UnicodeDecodeError) as e:
            raise ChatError(self.t("import_failed", error=str(e))) from e
        char = self.chat.store.add_character("ai", card)
        self.info(self.t("char_imported", name=card.name, id=char.id))

    def export_char(self: Repl, rest: str) -> None:
        char = self.require_char(rest) if rest.strip() else self.chat.character()
        if char is None:
            raise ChatError(self.t("char_none"))
        path = self.paths.data_dir / "exports" / f"{safe_filename(char.name)}.json"
        write_card_file(char.card, path)
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
        if sub not in ("set", "edit"):
            raise ChatError("/persona [set|clear]")
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
