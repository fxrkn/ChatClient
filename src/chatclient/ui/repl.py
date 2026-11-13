"""Single-line REPL front end (prompt_toolkit for input, Rich for tables)."""

from __future__ import annotations

import asyncio
import contextlib
import shlex
import signal
import sys
import time
from collections.abc import AsyncIterator, Callable, Iterable
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import HTML, FormattedText
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.table import Table

from chatclient import __version__
from chatclient.characters import card_color
from chatclient.core.chat import PARAM_TYPES, ChatError, ChatService, TurnResult
from chatclient.core.free_models import (
    ProbeResult,
    drop_working,
    find_openrouter,
    find_working,
    free_candidates,
    is_free_model,
    load_working,
    save_working,
)
from chatclient.i18n import COMMAND_DOCS, Translator, help_text
from chatclient.paths import AppPaths
from chatclient.providers.base import ProviderError, StreamEvent
from chatclient.theme import (
    PRESETS,
    Theme,
    ansi,
    character_color,
    ptk_color,
    resolve_theme,
    rich_style,
)
from chatclient.ui.characters_ui import CHAR_SUBCOMMANDS, CharacterCommands
from chatclient.ui.keys_ui import KEY_SUBCOMMANDS, KeyCommands
from chatclient.ui.memory_ui import MEMORY_SUBCOMMANDS, MemoryCommands
from chatclient.ui.picker import PickItem, pick
from chatclient.ui.render import BOLD, RESET, StreamWriter, reading_width
from chatclient.ui.setup_ui import SetupCommands
from chatclient.ui.tools_ui import ToolCommands

THEME_KEY = "ui_theme"
INDENT = "  "
CLEAR_LINE = "\x1b[2K"
PALETTE = "\x00palette"  # returned by the prompt when Ctrl+P is pressed

# Top-level commands for Tab completion, in the same order as /help.
COMMANDS = list(dict.fromkeys(doc.command.split()[0] for doc in COMMAND_DOCS))


class SlashCompleter(Completer):
    def __init__(
        self,
        models: Callable[[], list[str]],
        chars: Callable[[], list[str]],
        providers: Callable[[], list[str]] = list,
    ):
        self.models = models
        self.chars = chars
        self.providers = providers

    def get_completions(self, document: Document, event: CompleteEvent) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/"):
            return
        if " " not in text:
            for cmd in COMMANDS:
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text))
            return
        cmd, _, arg = text.partition(" ")
        if cmd == "/model":
            for m in self.models():
                if m.startswith(arg):
                    yield Completion(m, start_position=-len(arg))
        elif cmd == "/set" and " " not in arg:
            for p in PARAM_TYPES:
                if p.startswith(arg):
                    yield Completion(p, start_position=-len(arg))
        elif cmd == "/char":
            sub, space, name = arg.partition(" ")
            if not space:
                for s in CHAR_SUBCOMMANDS:
                    if s.startswith(sub):
                        yield Completion(s, start_position=-len(sub))
            elif sub == "ai":
                for f in ("first_mes", "mes_example", "description"):
                    if f.startswith(name):
                        yield Completion(f, start_position=-len(name))
            elif sub in ("use", "show", "edit", "delete", "export"):
                low = name.lower()
                for n in self.chars():
                    if n.lower().startswith(low):
                        yield Completion(n, start_position=-len(name))
        elif cmd == "/key":
            sub, space, name = arg.partition(" ")
            if not space:
                for s in KEY_SUBCOMMANDS:
                    if s.startswith(sub):
                        yield Completion(s, start_position=-len(sub))
            elif sub in ("set", "remove"):
                for n in self.providers():
                    if n.startswith(name):
                        yield Completion(n, start_position=-len(name))
        elif cmd in ("/memory", "/mem") and " " not in arg:
            for s in MEMORY_SUBCOMMANDS:
                if s.startswith(arg):
                    yield Completion(s, start_position=-len(arg))
        elif cmd == "/persona" and " " not in arg:
            for s in ("set", "clear"):
                if s.startswith(arg):
                    yield Completion(s, start_position=-len(arg))


class Repl(CharacterCommands, KeyCommands, ToolCommands, SetupCommands, MemoryCommands):
    def __init__(self, chat: ChatService, t: Translator, paths: AppPaths, show_stats: bool = True):
        self.chat = chat
        self.t = t
        self.paths = paths
        self.show_stats = show_stats
        self.console = Console(highlight=False)
        self.known_models: list[str] = []
        self.running = True
        self.last_result: TurnResult | None = None
        self.streaming = False  # a reply is being generated (memory jobs pause)
        self.job_task: asyncio.Future[bool] | None = None
        self.prefetch_task: asyncio.Future[None] | None = None  # brief for the draft
        self.prefetch_text = ""
        self.ui = chat.config.ui
        self.theme: Theme = self.load_theme()
        self.prompt_session: PromptSession[str] | None = None

    # ── theme ───────────────────────────────────────────────────────────────

    def load_theme(self) -> Theme:
        name = self.chat.store.kv_get(THEME_KEY) or self.ui.theme
        return resolve_theme(name, self.ui.colors)

    def style(self, key: str, extra: str = "") -> str:
        return rich_style(getattr(self.theme, key), extra)

    def ptk_style(self) -> Style:
        th = self.theme
        bar = (
            f"noreverse {ptk_color(th.bar_bg, bg=True)} {ptk_color(th.bar_fg)}"
            if th.bar_bg != "default"
            else ptk_color(th.bar_fg)
        )
        return Style.from_dict(
            {
                "bottom-toolbar": bar,
                "accent": f"{ptk_color(th.accent)} bold",
                "prompt-arrow": f"{ptk_color(th.accent)} bold",
                "completion-menu.completion": f"{ptk_color(th.bar_bg, bg=True)} {ptk_color(th.text)}",
                "completion-menu.completion.current": "reverse",
                "picker.title": f"{ptk_color(th.accent)} bold",
                "picker.item": ptk_color(th.text),
                "picker.selected": f"{ptk_color(th.accent)} bold",
                "picker.hint": ptk_color(th.dim),
            }
        )

    def apply_theme(self) -> None:
        self.theme = self.load_theme()
        if self.prompt_session is not None:
            self.prompt_session.style = self.ptk_style()

    # ── output helpers ──────────────────────────────────────────────────────

    def info(self, text: str) -> None:
        self.console.print(text, style=self.style("dim"), markup=False)

    def warn(self, text: str) -> None:
        self.console.print(text, style=self.style("warn"), markup=False)

    def error(self, text: str) -> None:
        self.console.print(text, style=self.style("error"), markup=False)

    def heading(self, text: str) -> None:
        self.console.print(text, style=self.style("accent", "bold"), markup=False)

    def write_message(self, label: str, label_color: str, content: str) -> None:
        """A finished message: name on its own line, indented body, blank line after."""
        out = sys.stdout
        out.write(f"{BOLD}{ansi(label_color)}{label}{RESET}\n")
        w = self.writer(out)
        w.write(content)
        w.finish()
        out.write("\n\n")
        out.flush()

    def writer(self, out: Any) -> StreamWriter:
        th = self.theme
        return StreamWriter(
            out,
            width=reading_width(self.ui.max_width),
            margin=0,
            indent=INDENT,
            base=ansi(th.text),
            action=ansi(th.action),
            style_actions=self.ui.style_actions,
            speech=ansi(th.speech) if self.ui.style_speech else "",
        )

    def assistant_label(self, model: str | None) -> str:
        char = self.chat.character()
        if char is not None:
            return char.name
        return (model or "AI").split(":", 1)[-1].split("/")[-1]

    def assistant_color(self) -> str:
        """The current character's own colour (or a stable automatic one), else the accent."""
        char = self.chat.character()
        if char is None or not self.ui.character_colors:
            return self.theme.accent
        return character_color(char.name, self.theme, card_color(char.card))

    async def pick(self, title: str, items: list[PickItem], current: Any = None) -> Any:
        return await pick(
            title, items, current=current, footer=self.t("picker_footer"), style=self.ptk_style()
        )

    def user_label(self) -> str:
        persona = self.chat.persona()
        return persona.name if persona else self.t("you")

    def toolbar(self) -> HTML:
        s = self.chat.session
        if s is None:
            return HTML("")
        title = s.title or self.t("untitled")
        model = s.model or "—"
        color = self.assistant_color() if s.character_id else self.theme.accent
        name = (
            f'<style fg="{color}"><b>{_esc(title)}</b></style>'
            if color != "default"
            else f"<b>{_esc(title)}</b>"
        )
        return HTML(f" {name}  ·  #{s.id}  ·  {_esc(model)}  ·  /help ")

    # ── main loop ───────────────────────────────────────────────────────────

    def banner(self) -> None:
        if self.ui.clear_on_start:
            self.console.clear()
        width = reading_width(self.ui.max_width)
        self.heading(f"ChatClient {__version__}")
        self.info(self.t("welcome"))
        self.console.print("─" * width, style=self.style("dim"), markup=False)

    async def run(self, startup_notes: list[str]) -> None:
        self.banner()
        for note in startup_notes:
            self.info(note)
        self.print_session_header()
        if self.chat.require_session().model is None:
            self.warn(self.t("no_model"))

        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event: Any) -> None:
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        empty = Condition(lambda: not get_app().current_buffer.text)

        # Browse the versions of the last reply (from /regen) with Alt/Ctrl + ←/→,
        # when nothing has been typed yet.
        @kb.add("escape", "left", filter=empty)
        @kb.add("c-left", filter=empty)
        def _prev(event: Any) -> None:
            event.app.exit(result="/prev")

        @kb.add("escape", "right", filter=empty)
        @kb.add("c-right", filter=empty)
        def _next(event: Any) -> None:
            event.app.exit(result="/next")

        @kb.add("c-p")
        def _palette(event: Any) -> None:
            event.app.exit(result=PALETTE)

        session: PromptSession[str] = PromptSession(
            history=FileHistory(str(self.paths.history_file)),
            completer=SlashCompleter(
                lambda: self.known_models, self.char_names, self.key_providers
            ),
            complete_while_typing=True,
            key_bindings=kb,
            multiline=True,
            bottom_toolbar=self.toolbar,
            prompt_continuation=lambda width, line_no, wrap: "  ",
            style=self.ptk_style(),
        )
        self.prompt_session = session
        # Unsent text is kept per session, so closing the app doesn't lose it.
        session.default_buffer.on_text_changed += lambda buf: self.save_draft(buf.text)
        # …and memory for it is prepared while it's being typed.
        session.default_buffer.on_text_changed += lambda buf: self.schedule_prefetch(buf.text)
        if self.needs_setup():
            await self.setup_wizard()
        elif self.load_draft():
            self.info(self.t("draft_restored"))
        worker = asyncio.ensure_future(self.memory_worker()) if self.chat.memory else None
        try:
            await self.main_loop(session)
        finally:
            if worker is not None:
                worker.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await worker

    async def main_loop(self, session: PromptSession[str]) -> None:
        prompt = FormattedText([("class:prompt-arrow", "› ")])
        prefill = ""
        while self.running:
            try:
                line = await session.prompt_async(prompt, default=prefill or self.load_draft())
            except KeyboardInterrupt:
                self.save_draft("")  # Ctrl+C discards what was typed
                self.info(self.t("ctrl_c_hint"))
                continue
            except EOFError:
                break
            prefill = ""
            if line == PALETTE:
                # The typed text stays saved as the draft, so cancelling the palette
                # brings it back; a command needing arguments replaces it.
                try:
                    prefill = await self.palette() or ""
                except ChatError as e:
                    self.warn(str(e))
                continue
            self.save_draft("")
            line = line.strip()
            if not line:
                continue
            try:
                if line.startswith("/"):
                    await self.handle_command(line)
                else:
                    await self.settle_prefetch(line)
                    await self.reply(lambda r, text=line: self.chat.send(text, r))
            except ChatError as e:
                self.warn(str(e))
            except ProviderError as e:
                self.error(str(e))
            except ValueError as e:
                self.warn(str(e))

    def _draft_key(self) -> str | None:
        return f"draft:{self.chat.session.id}" if self.chat.session else None

    def save_draft(self, text: str) -> None:
        if key := self._draft_key():
            self.chat.store.kv_set(key, text)

    def load_draft(self) -> str:
        key = self._draft_key()
        return (self.chat.store.kv_get(key) or "") if key else ""

    def print_session_header(self) -> None:
        s = self.chat.require_session()
        count = len(self.chat.history())
        self.info(
            self.t("session_opened", id=s.id, title=s.title or self.t("untitled"), count=count)
        )
        if count:
            self.show_history(6)

    # ── free-model fallback ─────────────────────────────────────────────────

    MAX_FALLBACKS = 3

    async def reply(self, start: Callable[[TurnResult], AsyncIterator[StreamEvent]]) -> None:
        """Stream a reply; if a free model is busy, retry on another working free model."""
        tried: set[str] = set()
        rescued = rewritten = False
        while True:
            try:
                await self.stream_reply(start)
                result = self.last_result
                if result is None or result.cancelled:
                    return
                fallback = self.chat.refusal_fallback()
                if result.refused and fallback and fallback != result.model and not rescued:
                    # The model declined: the configured fallback writes this turn
                    # instead (the declined version stays as an alternative, see /prev).
                    rescued = True
                    self.warn(self.t("refusal_fallback", old=result.model, new=fallback))
                    start = lambda r, m=fallback: self.chat.regenerate(r, model=m)  # noqa: E731
                    continue
                if result.repeated and not result.refused and not rewritten and self.auto_rewrite():
                    # Mostly recycled from earlier replies: ask once for a new one (the
                    # repeated version stays as an alternative, see /prev).
                    rewritten = True
                    self.warn(self.t("rewrite_repeat", pct=f"{result.repetition:.0%}"))
                    model = (
                        result.model if result.model != self.chat.require_session().model else None
                    )
                    start = lambda r, m=model: self.chat.regenerate(r, model=m, fresh=True)  # noqa: E731
                    continue
                return
            except ProviderError as e:
                ref = self.chat.model_ref()
                if (
                    ref is None
                    or not self.can_fallback(e, ref.model)
                    or len(tried) >= self.MAX_FALLBACKS
                ):
                    raise
                self.error(str(e).split("\n")[0])
                tried.add(ref.model)
                drop_working(self.chat.store, ref.provider, ref.model)
                nxt = await self.next_free_model(ref.provider, tried)
                if nxt is None:
                    raise ChatError(self.t("free_none")) from e
                self.chat.set_model(f"{ref.provider}:{nxt}")
                self.warn(self.t("free_switch", old=ref.model, new=nxt))
                start = self.chat.regenerate  # the user message is already saved

    def auto_rewrite(self) -> bool:
        # Only in stories: in a plain chat, asking the same thing twice may rightly
        # get the same answer.
        return self.chat.config.roleplay.rewrite_repeats and self.chat.character() is not None

    def can_fallback(self, e: ProviderError, model: str) -> bool:
        if not self.chat.config.models.free_fallback or not is_free_model(model):
            return False
        if self.last_result is not None and self.last_result.text:
            return False  # part of a reply was already shown
        return e.retryable or e.status in (402, 404)

    async def next_free_model(self, provider_name: str, exclude: set[str]) -> str | None:
        cached = [m for m in load_working(self.chat.store, provider_name) if m not in exclude]
        if cached:
            return cached[0]
        self.info(self.t("free_probing"))
        provider = self.chat.registry.get(provider_name)
        candidates = [m for m in await free_candidates(provider) if m not in exclude]
        results = await find_working(provider, candidates, want=2, max_probes=6)
        working = save_working(self.chat.store, provider_name, results)
        return working[0] if working else None

    async def free_command(self, arg: str) -> None:
        name = find_openrouter(self.chat.registry)
        if name is None:
            raise ChatError(self.t("free_no_openrouter"))
        if not self.chat.registry.get(name).api_key:
            raise ChatError(self.t("key_missing", name=name))
        if arg == "list":
            cached = load_working(self.chat.store, name)
            if not cached:
                self.info(self.t("free_none_cached"))
            for m in cached:
                self.console.print(f"  {name}:{m}", markup=False)
            return
        max_probes = 8
        if arg == "all":
            max_probes = 1000
        elif arg.isdigit():
            max_probes = max(1, int(arg))
        provider = self.chat.registry.get(name)
        candidates = await free_candidates(provider)
        if not candidates:
            raise ChatError(self.t("free_none"))
        n = min(max_probes, len(candidates))
        self.info(self.t("free_testing", n=n, total=len(candidates)))

        def show(r: ProbeResult) -> None:
            if r.ok:
                line = f"  ✓ {r.model}  {r.latency:.1f}s"
                self.console.print(line, style=self.style("user"), markup=False)
            else:
                self.info(f"  ✗ {r.model}  {r.error}")

        want = n if arg == "all" or arg.isdigit() else 3
        results = await find_working(provider, candidates, want=want, max_probes=n, on_result=show)
        working = save_working(self.chat.store, name, results)
        if not working:
            raise ChatError(self.t("free_none"))
        self.chat.set_model(f"{name}:{working[0]}")
        self.known_models = sorted(set(self.known_models) | {f"{name}:{m}" for m in working})
        self.info(self.t("free_selected", model=f"{name}:{working[0]}", n=len(working)))

    # ── streaming ───────────────────────────────────────────────────────────

    async def stream_reply(self, start: Callable[[TurnResult], AsyncIterator[StreamEvent]]) -> None:
        # The reply gets the model to itself: background memory work pauses (and an
        # interrupted job goes back to the queue).
        self.streaming = True
        self.interrupt_memory_job()
        try:
            await self._stream_reply(start)
        finally:
            self.streaming = False

    async def _stream_reply(
        self, start: Callable[[TurnResult], AsyncIterator[StreamEvent]]
    ) -> None:
        if self.chat.require_session().model is None:
            raise ChatError(self.t("no_model"))
        result = TurnResult()
        self.last_result = result
        label = self.assistant_label(self.chat.require_session().model)
        th = self.theme
        out = sys.stdout
        dim = ansi(th.dim)
        out.write(f"\n{BOLD}{ansi(self.assistant_color())}{label}{RESET}\n")
        typing = self.t("typing", name=label)
        out.write(f"{INDENT}{dim}{typing}{RESET}")
        out.flush()
        writer = self.writer(out)
        placeholder = True
        animate = self.ui.typing_indicator and out.isatty()

        async def tick() -> None:
            # "⠹ <name> is typing… 2.4s" until the first words arrive
            frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
            i = 0
            while True:
                await asyncio.sleep(0.1)
                i += 1
                elapsed = time.perf_counter() - result.started
                out.write(
                    f"\r{CLEAR_LINE}{INDENT}{dim}{frames[i % len(frames)]} {typing} {elapsed:.1f}s{RESET}"
                )
                out.flush()

        ticker = asyncio.ensure_future(tick()) if animate else None

        def clear_placeholder() -> None:
            nonlocal placeholder
            if not placeholder:
                return
            placeholder = False
            if ticker is not None:
                ticker.cancel()
            out.write(f"\r{CLEAR_LINE}{INDENT}{writer.base}")
            writer.start_at(len(INDENT))

        async def consume() -> None:
            async for ev in start(result):
                if ev.text:
                    clear_placeholder()
                    writer.write(ev.text)

        task = asyncio.ensure_future(consume())
        loop = asyncio.get_running_loop()

        def on_sigint(signum: int, frame: object) -> None:
            loop.call_soon_threadsafe(task.cancel)

        previous = signal.getsignal(signal.SIGINT)
        try:
            signal.signal(signal.SIGINT, on_sigint)
        except ValueError:  # not in main thread
            previous = None
        try:
            await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise  # we are being shut down, not a user stop
        except BaseException:
            had_text = not placeholder
            clear_placeholder()
            writer.finish()
            out.write("\n" if had_text else f"\r{CLEAR_LINE}")
            out.flush()
            raise
        finally:
            if ticker is not None:
                ticker.cancel()
            if previous is not None:
                signal.signal(signal.SIGINT, previous)

        clear_placeholder()
        writer.finish()
        out.write("\n")
        warn = ansi(th.warn)
        if result.cancelled:
            out.write(f"{INDENT}{warn}{self.t('cancelled')}{RESET}\n")
        elif result.refused and not self.chat.refusal_fallback():
            out.write(f"{INDENT}{warn}{self.t('refusal')}{RESET}\n")
        elif result.finish_reason in ("length", "max_tokens"):
            out.write(f"{INDENT}{warn}{self.t('truncated_len')}{RESET}\n")
        if self.show_stats and not result.cancelled:
            out.write(f"{ansi(th.dim)}{self.stats_line(result)}{RESET}\n")
        out.write("\n")
        out.flush()

    def stats_line(self, r: TurnResult) -> str:
        parts = []
        out_tokens = r.usage.completion_tokens if r.usage else None
        gen_time = r.elapsed - (r.time_to_first_token or 0)
        if out_tokens:
            parts.append(f"{out_tokens} tok")
            if gen_time > 0.05:
                parts.append(f"{out_tokens / gen_time:.1f} tok/s")
        if r.usage and r.usage.prompt_tokens:
            parts.append(f"ctx {r.usage.prompt_tokens}")
        if r.time_to_first_token is not None:
            parts.append(f"ttft {r.time_to_first_token:.2f}s")
        if r.usage and r.usage.load_seconds and r.usage.load_seconds >= 0.5:
            parts.append(self.t("model_load", s=f"{r.usage.load_seconds:.1f}"))
        parts.append(f"{r.elapsed:.1f}s")
        if r.model and r.model != self.chat.require_session().model:
            parts.append(r.model)  # a one-off /regen <model>
        if r.recalled:
            parts.append(self.t("recalled_n", n=r.recalled))
        if r.conflicts:
            parts.append(self.t("conflict_noted", n=r.conflicts))
        if r.context and r.context.dropped:
            parts.append(self.t("dropped_context", n=r.context.dropped))
        sibs, index = self.chat.alternatives()
        if len(sibs) > 1:
            parts.append(f"‹ {index + 1}/{len(sibs)} ›")
        return "  " + " · ".join(parts)

    # ── commands ────────────────────────────────────────────────────────────

    async def handle_command(self, line: str) -> None:
        cmd, _, arg = line.partition(" ")
        arg = arg.strip()
        cmd = cmd.lower()
        match cmd:
            case "/help" | "/?":
                self.console.print(help_text(self.t), markup=False)
            case "/quit" | "/exit" | "/q":
                self.running = False
            case "/clear" | "/cls":
                self.console.clear()
            case "/new":
                s = self.chat.new_session(title=arg)
                self.info(self.t("new_session", id=s.id))
                if s.model is None:
                    self.warn(self.t("no_model"))
            case "/sessions":
                self.show_sessions()
            case "/switch":
                if not arg:
                    sid = await self.pick_session()
                    if sid is None:
                        return
                elif arg.lstrip("#").isdigit():
                    sid = int(arg.lstrip("#"))
                else:
                    raise ChatError("/switch <id>")
                self.chat.open_session(sid)
                self.print_session_header()
            case "/next" | "/prev":
                await self.switch_alternative(1 if cmd == "/next" else -1)
            case "/rename":
                if not arg:
                    raise ChatError("/rename <title>")
                self.chat.store.update_session(self.chat.require_session().id, title=arg)
                self.chat.refresh()
                self.info(self.t("renamed", title=arg))
            case "/delete":
                await self.delete_session(arg)
            case "/history":
                n = int(arg) if arg.isdigit() else 20
                self.show_history(n)
            case "/model":
                if not arg:
                    arg = await self.pick_model() or ""
                    if not arg:
                        return
                ref = self.chat.set_model(arg)
                self.info(self.t("model_set", model=str(ref)))
            case "/models":
                await self.show_models(arg or None)
            case "/providers":
                self.show_providers()
            case "/system":
                self.system_command(arg)
            case "/set":
                self.set_command(arg)
            case "/regen" | "/retry":
                model = arg or None
                await self.reply(lambda r: self.chat.regenerate(r, model=model))
            case "/edit":
                await self.edit_command(arg)
            case "/continue" | "/cont":
                await self.reply(self.chat.continue_reply)
            case "/ooc":
                await self.ooc_command(arg)
            case "/copy":
                self.copy_command(arg)
            case "/export":
                self.export_command(arg)
            case "/lang":
                self.lang_command(arg)
            case "/setup":
                await self.setup_wizard()
            case "/private":
                self.private_command(arg)
            case "/content":
                self.content_command(arg)
            case "/fallback":
                self.fallback_command(arg)
            case "/memory" | "/mem":
                await self.memory_command(arg)
            case "/recall":
                await self.recall_command(arg)
            case "/scene":
                self.scene_command(arg)
            case "/summary":
                self.summary_command()
            case "/day":
                await self.day_command(arg)
            case "/free":
                await self.free_command(arg)
            case "/char" | "/character":
                await self.char_command(arg)
            case "/persona":
                await self.persona_command(arg)
            case "/theme":
                self.theme_command(arg)
            case "/key" | "/keys":
                await self.key_command(arg)
            case "/undo":
                n = self.chat.undo()
                self.info(self.t("undone", n=n))
            case "/search":
                self.search(arg)
            case "/import":
                self.import_command(arg)
            case "/open":
                self.open_command(arg)
            case "/config":
                root = self.paths.data_dir
                self.info(f"config:     {self.paths.config_file}")
                self.info(f"characters: {root / 'characters'}")
                self.info(f"personas:   {root / 'personas'}")
                self.info(f"chats:      {root / 'chats'}")
                self.info(f"index:      {self.paths.db_file}")
                self.info(f"logs:       {self.paths.log_file}")
            case _:
                self.warn(self.t("unknown_command", cmd=cmd))

    # ── pickers & alternatives ──────────────────────────────────────────────

    async def pick_model(self) -> str | None:
        if not self.known_models:
            self.info(self.t("listing_models"))
            results = await self.chat.registry.list_models()
            self.known_models = sorted(
                f"{name}:{m}"
                for name, models in results.items()
                if isinstance(models, list)
                for m in models
            )
        if not self.known_models:
            raise ChatError(self.t("no_models_found"))
        items = [PickItem(m, m.split(":", 1)[1], m.split(":", 1)[0]) for m in self.known_models]
        return await self.pick(self.t("pick_model"), items, self.chat.require_session().model)

    async def pick_session(self) -> int | None:
        items = []
        for it in self.chat.store.list_sessions():
            s = it.session
            hint = f"{s.model or '—'} · {it.message_count} · {s.updated_at[:10]}"
            items.append(PickItem(s.id, f"#{s.id} {s.title or self.t('untitled')}", hint))
        return await self.pick(self.t("pick_session"), items, self.chat.require_session().id)

    async def switch_alternative(self, step: int) -> None:
        msg = self.chat.switch_alternative(step)
        if msg is None:  # past the newest version: write a new one
            await self.reply(self.chat.regenerate)
            return
        sibs, index = self.chat.alternatives()
        sys.stdout.write("\n")
        self.write_message(self.assistant_label(msg.model), self.assistant_color(), msg.content)
        self.info(f"{INDENT}‹ {index + 1}/{len(sibs)} ›  {self.t('alt_hint')}")

    def theme_command(self, arg: str) -> None:
        name = arg.strip().lower()
        if not name:
            current = self.theme.name
            for key, th in PRESETS.items():
                marker = "▸" if key == current else " "
                self.console.print(f" {marker} {key:<8} ", end="", markup=False)
                samples = (
                    (th.accent, f"{self.t('theme_sample_name')} "),
                    (th.text, f"{self.t('theme_sample_text')} "),
                    (th.action, f"{self.t('theme_sample_action')} "),
                    (th.speech, f"{self.t('theme_sample_speech')} "),
                    (th.user, self.t("you")),
                )
                for color, sample in samples:
                    italic = "italic" if color == th.action else ""
                    self.console.print(
                        sample, style=rich_style(color, italic), end="", markup=False
                    )
                self.console.print()
            self.info(self.t("theme_hint"))
            return
        if name == "reset":
            self.chat.store.kv_set(THEME_KEY, "")
        elif name in PRESETS:
            self.chat.store.kv_set(THEME_KEY, name)
        else:
            raise ChatError(self.t("theme_unknown", name=name, names=", ".join(PRESETS)))
        self.apply_theme()
        self.info(self.t("theme_set", name=self.theme.name))

    def show_sessions(self) -> None:
        table = Table(box=None, header_style=self.style("accent", "bold"), pad_edge=False)
        for col in ("#", "title", "model", "msgs", "updated"):
            table.add_column(col, no_wrap=col != "title")
        current = self.chat.require_session().id
        for item in self.chat.store.list_sessions():
            s = item.session
            marker = "▸" if s.id == current else " "
            table.add_row(
                f"{marker}{s.id}",
                s.title or self.t("untitled"),
                s.model or "—",
                str(item.message_count),
                s.updated_at[:16].replace("T", " "),
            )
        self.console.print(table)

    async def delete_session(self, arg: str) -> None:
        if not arg.lstrip("#").isdigit():
            raise ChatError("/delete <id>")
        sid = int(arg.lstrip("#"))
        target = self.chat.store.get_session(sid)
        if target is None:
            raise ChatError(f"session {sid} not found")
        confirm: PromptSession[str] = PromptSession()
        answer = await confirm.prompt_async(
            self.t("confirm_delete", id=sid, title=target.title or self.t("untitled"))
        )
        if answer.strip().lower() not in ("y", "yes", "e", "evet"):
            self.info(self.t("aborted"))
            return
        was_current = self.chat.require_session().id == sid
        self.chat.store.delete_session(sid)
        self.info(self.t("deleted", id=sid))
        if was_current:
            self.chat.resume_or_create()
            self.print_session_header()

    def show_history(self, n: int) -> None:
        msgs = self.chat.history(limit=n)
        if msgs:
            sys.stdout.write("\n")
        you = self.user_label()
        for m in msgs:
            if m.role == "user":
                self.write_message(you, self.theme.user, m.content)
            else:
                self.write_message(self.assistant_label(m.model), self.assistant_color(), m.content)

    async def show_models(self, provider: str | None) -> None:
        names = [provider] if provider else None
        if provider:
            self.chat.registry.get(provider)  # validate
        self.info(self.t("listing_models"))
        results = await self.chat.registry.list_models(names)
        found: list[str] = []
        for name, models in results.items():
            if isinstance(models, ProviderError):
                if provider:  # only show errors when asked about a specific provider
                    self.error(str(models))
                else:
                    self.info(f"  {name}: {models}")
                continue
            self.heading(name)
            for m in models:
                ref = f"{name}:{m}"
                found.append(ref)
                self.console.print(f"  {ref}", markup=False)
        self.known_models = sorted(set(self.known_models) | set(found))

    def show_providers(self) -> None:
        table = Table(box=None, header_style=self.style("accent", "bold"), pad_edge=False)
        for col in ("name", "type", "endpoint", "key"):
            table.add_column(col)
        for name in self.chat.registry.names():
            pc = self.chat.registry.configs[name]
            try:
                p = self.chat.registry.get(name)
                endpoint = p.describe()
                has_key = "✓" if p.api_key else ("—" if pc.type in ("ollama", "echo") else "✗")
            except ProviderError as e:
                endpoint, has_key = str(e), "?"
            table.add_row(name, pc.type, endpoint, has_key)
        self.console.print(table)

    def system_command(self, arg: str) -> None:
        if not arg:
            prompt = self.chat.effective_system_prompt()
            custom = self.chat.require_session().system_prompt is not None
            self.info("(custom)" if custom else "(default)")
            self.console.print(prompt, markup=False)
        elif arg.lower() == "reset":
            self.chat.set_system_prompt(None)
            self.info(self.t("system_reset"))
        else:
            self.chat.set_system_prompt(arg)
            self.info(self.t("system_set"))

    def set_command(self, arg: str) -> None:
        if not arg:
            params = self.chat.gen_params()
            overrides = self.chat.require_session().params
            for name in PARAM_TYPES:
                if name == "context_tokens":
                    ref = self.chat.model_ref()
                    value: Any = self.chat.context_budget(ref, params) if ref else None
                else:
                    value = getattr(params, name)
                mark = "*" if name in overrides else " "
                shown = "default" if value is None else value
                self.console.print(f" {mark} {name:<20} {shown}", markup=False)
            self.info("  (* = set for this session)")
            return
        try:
            parts = shlex.split(arg)
        except ValueError:
            parts = arg.split()
        if len(parts) < 2:
            raise ChatError("/set <name> <value|reset>")
        name, raw = parts[0], " ".join(parts[1:])
        value = self.chat.set_param(name, raw)
        if value is None:
            self.info(self.t("param_reset", name=name))
        else:
            self.info(self.t("param_set", name=name, value=value))

    def search(self, arg: str) -> None:
        all_sessions = False
        if arg.startswith("--all"):
            all_sessions, arg = True, arg[5:].strip()
        if len(arg.strip()) < 3:
            self.warn(self.t("search_short"))
            return
        sid = None if all_sessions else self.chat.require_session().id
        hits = self.chat.store.search(arg, session_id=sid)
        if not hits:
            self.info(self.t("no_results"))
            return
        for h in hits:
            when = h.message.created_at[:16].replace("T", " ")
            where = f"#{h.message.session_id} " if all_sessions else ""
            who = self.t("you") if h.message.role == "user" else "AI"
            self.console.print(
                f"{where}{when} {who}: ", style=self.style("dim"), end="", markup=False
            )
            self.console.print(h.snippet, markup=False)


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


async def run_repl(
    chat: ChatService, t: Translator, paths: AppPaths, notes: list[str], show_stats: bool
) -> None:
    repl = Repl(chat, t, paths, show_stats)
    with contextlib.suppress(asyncio.CancelledError):
        await repl.run(notes)
    sys.stdout.write(f"{ansi(repl.theme.dim)}bye{RESET}\n")
