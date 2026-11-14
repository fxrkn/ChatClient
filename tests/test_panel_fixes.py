"""Regression tests for the panel review fixes: answers are not drafts, Ctrl+C in any
question, a failing panel falls back to inline, and the older-messages bookkeeping."""

import asyncio
import contextlib
import sys
from types import SimpleNamespace

from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput

from chatclient.config import Config, ProviderConfig
from chatclient.core.chat import ChatService
from chatclient.keys import KeyStore
from chatclient.providers.registry import ProviderRegistry
from chatclient.ui import clipboard
from chatclient.ui import panel as panel_mod
from chatclient.ui import repl as repl_mod
from chatclient.ui.repl import LAYOUT_KEY
from test_panel import Recorder, make_repl, until
from test_panel_flows import flat, running_panel

OR = "https://openrouter.ai/api/v1"


def draft_of(chat):
    return chat.store.kv_get(f"draft:{chat.require_session().id}") or ""


async def test_a_typed_key_never_reaches_the_draft_or_the_prefetch(tmp_path, store, monkeypatch):
    config = Config(providers={"openrouter": ProviderConfig(
        type="openai_compat", base_url=OR, api_key_env="OPENROUTER_API_KEY")})  # fmt: skip
    chat = ChatService(config, store, ProviderRegistry(config, KeyStore(tmp_path / "keys.json")))
    chat.new_session(model="echo:echo")
    async with running_panel(chat, tmp_path, monkeypatch) as (view, repl, send):
        seen = []
        repl.schedule_prefetch = seen.append
        await send("/key set openrouter")
        await until(lambda: view._password)
        send.raw("sk-secret-123")
        await until(lambda: view.input.text == "sk-secret-123")
        await asyncio.sleep(0.1)
        assert draft_of(chat) == ""
        assert not any("sk-secret" in s for s in seen)
        send.raw("\x03")
        await until(lambda: repl.t("aborted") in flat(view) and not view.is_busy())
        assert draft_of(chat) == "" and view.input.text == ""
        send.raw("hello")  # ordinary typing is still a draft
        await until(lambda: draft_of(chat) == "hello")
        send.raw("\x03")  # Ctrl+C on an idle prompt discards it, stored draft too
        await until(lambda: view.input.text == "")
        assert draft_of(chat) == ""
    await chat.registry.aclose()


async def test_text_typed_before_a_question_comes_back_after_it(chat, tmp_path, monkeypatch):
    async with running_panel(chat, tmp_path, monkeypatch) as (view, _repl, send):
        view.input.text = "mine"
        asking = asyncio.ensure_future(view.ask("Q?"))
        await until(lambda: view.question == "Q?")
        assert view.input.text == ""
        await send("the answer")
        assert await asyncio.wait_for(asking, 5) == "the answer"
        assert view.input.text == "mine"
        view.input.reset()


async def test_ctrl_c_in_the_delete_confirmation_aborts_and_the_app_lives(
    chat, tmp_path, monkeypatch
):
    sid = chat.require_session().id
    async with running_panel(chat, tmp_path, monkeypatch) as (view, repl, send):
        await send(f"/delete {sid}")
        await until(lambda: bool(view.question))
        send.raw("\x03")
        await until(lambda: repl.t("aborted") in flat(view) and not view.is_busy())
        assert view.app.is_running
        assert chat.store.get_session(sid) is not None
        await send("/help")
        await until(lambda: "/layout" in flat(view) and not view.is_busy())


async def test_a_failing_panel_falls_back_to_inline_and_is_remembered(
    chat, tmp_path, monkeypatch, capsys
):
    def broken(*a, **k):
        raise RuntimeError("no console here")

    monkeypatch.setattr(panel_mod, "PanelView", broken)
    chat.store.kv_set(LAYOUT_KEY, "panel")
    repl = make_repl(chat, tmp_path)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        task = asyncio.ensure_future(repl.run([]))
        try:
            await until(lambda: repl.prompt_session is not None)
            inp.send_text("/quit\r")
            await asyncio.wait_for(task, 5)
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task
    assert chat.store.kv_get(LAYOUT_KEY) == "inline"
    out = " ".join(capsys.readouterr().out.split())
    assert "RuntimeError: no console here" in out
    assert repl.panel is None


def test_panel_needs_a_terminal(chat, tmp_path, monkeypatch, real_panel_possible):
    repl = make_repl(chat, tmp_path)
    monkeypatch.setattr(repl_mod.sys, "stdout", SimpleNamespace(isatty=lambda: False))
    assert real_panel_possible(repl) is False


def fill(chat, n):
    session = chat.require_session()
    parent = None
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msg = chat.store.add_message(session.id, parent, role, f"tok{i:02d}tok")
        parent = msg.id


async def test_older_messages_have_no_duplicates_or_stray_sessions(chat, tmp_path, monkeypatch):
    fill(chat, 30)
    first = chat.require_session().id
    async with running_panel(chat, tmp_path, monkeypatch) as (view, _repl, send):
        assert view.hidden_before == 24 and view.shown_session == first
        await send("/history 3")  # repeats what is shown; the top is unchanged
        await until(lambda: flat(view).count("tok29tok") == 2 and not view.is_busy())
        assert view.hidden_before == 24
        view.load_older()
        assert flat(view).count("tok23tok") == 1 and flat(view).count("tok24tok") == 1
        assert view.hidden_before == 14
        view.load_older()
        view.load_older()
        assert view.hidden_before == 0
        assert all(flat(view).count(f"tok{i:02d}tok") == 1 for i in range(0, 24))

        await send("/clear")  # nothing above an emptied screen
        await until(lambda: view.log.text == "" and not view.is_busy())
        assert view.hidden_before == 0
        view.load_older()
        assert view.log.text == ""

        await send("/new")  # another conversation: its own, empty area
        await until(lambda: view.shown_session != first and not view.is_busy())
        assert view.hidden_before == 0 and "tok" not in flat(view)
        view.hidden_before = 5  # even a stale count cannot bring the old chat in
        view.load_older()
        assert "tok" not in flat(view) and view.hidden_before == 0

        await send(f"/switch {first}")  # back: shown from the top again, once
        await until(lambda: view.shown_session == first and not view.is_busy())
        assert view.hidden_before == 24
        assert flat(view).count("tok29tok") == 1 and flat(view).count("tok24tok") == 1


# ── selecting and copying chat text with the mouse ───────────────────────────


def mouse(kind, x, y, button=MouseButton.LEFT):
    return MouseEvent(Point(x, y), kind, button, frozenset())


@contextlib.contextmanager
def chat_area(chat, tmp_path, text):
    repl = make_repl(chat, tmp_path)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        view = panel_mod.PanelView(repl, repl.completer())
        view.log.text = text
        view.log.version += 1
        view._visible()  # (what the first render does)
        yield view, panel_mod.ScrollableText(view)


def drag(ctrl, start, end):
    ctrl.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, *start))
    ctrl.mouse_handler(mouse(MouseEventType.MOUSE_MOVE, *end))
    ctrl.mouse_handler(mouse(MouseEventType.MOUSE_UP, *end, button=MouseButton.NONE))


async def test_dragging_selects_highlights_and_copies_plain_log_lines(chat, tmp_path, monkeypatch):
    copied = []
    monkeypatch.setattr(clipboard, "copy_text", lambda t: copied.append(t) or "system")
    text = "hello \x1b[1mworld\x1b[0m\nsecond line here\nthird"
    with chat_area(chat, tmp_path, text) as (view, ctrl):
        ctrl.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, 6, 0))
        ctrl.mouse_handler(mouse(MouseEventType.MOUSE_MOVE, 5, 1))  # dragging
        marked = "".join(t for s, t in view._visible()[0] if "reverse" in s)
        assert marked == "world" + "second"  # shown highlighted
        ctrl.mouse_handler(mouse(MouseEventType.MOUSE_UP, 5, 1, MouseButton.NONE))
        assert copied == ["world\nsecond"]  # no ANSI codes, cut at the characters
        assert "12" in view.toolbar()[0][1] and view.selection.note
        view._note_gone()
        assert view.selection.note == ""


async def test_a_click_only_clears_and_the_wheel_still_scrolls(chat, tmp_path, monkeypatch):
    copied = []
    monkeypatch.setattr(clipboard, "copy_text", lambda t: copied.append(t) or "system")
    with chat_area(chat, tmp_path, "one\ntwo\nthree") as (view, ctrl):
        drag(ctrl, (0, 0), (1, 2))  # (column, row)
        assert copied == ["one\ntwo\nth"]
        ctrl.mouse_handler(mouse(MouseEventType.MOUSE_DOWN, 1, 1))
        ctrl.mouse_handler(mouse(MouseEventType.MOUSE_UP, 1, 1, MouseButton.NONE))
        assert view.selection.span() is None and len(copied) == 1
        assert not any("reverse" in s for s, _ in view._visible()[0])
        ctrl.mouse_handler(mouse(MouseEventType.SCROLL_UP, 0, 0, MouseButton.NONE))
        assert view.scroll == panel_mod.WHEEL_STEP


async def test_selection_follows_log_lines_when_scrolled(chat, tmp_path, monkeypatch):
    copied = []
    monkeypatch.setattr(clipboard, "copy_text", lambda t: copied.append(t) or "system")
    monkeypatch.setattr(panel_mod, "RENDER_LINES", 5)
    text = "\n".join(f"line{i:02d}" for i in range(50))
    with chat_area(chat, tmp_path, text) as (view, ctrl):
        view.scroll = 10  # the screen ends at line 39, so it starts at 35
        view.log.version += 1
        view._visible()
        drag(ctrl, (2, 2), (4, 4))  # (column, row): drawn rows 2..4 are log lines 37..39
        assert copied == ["ne37\nline38\nline3"]


async def test_clipboard_fallback_goes_to_the_terminal_not_the_chat(chat, tmp_path, monkeypatch):
    def osc52(text):
        sys.stdout.write("\x1b]52;c;QQ==\x07")
        return "terminal"

    monkeypatch.setattr(clipboard, "copy_text", osc52)
    with chat_area(chat, tmp_path, "abc def") as (view, ctrl):
        out = Recorder()
        view.app.output = out
        drag(ctrl, (0, 0), (3, 0))
        assert out.raw == ["\x1b]52;c;QQ==\x07"] and view.log.text == "abc def"
        assert "terminal" in view.toolbar()[0][1]
