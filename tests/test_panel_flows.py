"""The panel layout end to end: wizards, Ctrl+C, /key, narrow terminals, /clear."""

import asyncio
import contextlib

import httpx
import keyring
import respx
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from chatclient.config import Config, ProviderConfig
from chatclient.core.chat import ChatService
from chatclient.keys import KeyStore
from chatclient.providers.echo import EchoProvider
from chatclient.providers.registry import ProviderRegistry
from chatclient.ui import panel as panel_mod
from chatclient.ui.repl import LAYOUT_KEY
from test_panel import make_repl, until

OR = "https://openrouter.ai/api/v1"
GOOD = "sk-or-v1-" + "b" * 64


class NarrowOutput(DummyOutput):
    def get_size(self):
        return Size(rows=24, columns=60)


@contextlib.asynccontextmanager
async def running_panel(chat, tmp_path, monkeypatch, output=None):
    """The app in panel layout, started and idle: yields (view, repl, send)."""
    chat.store.kv_set(LAYOUT_KEY, "panel")
    repl = make_repl(chat, tmp_path)
    views = []
    real = panel_mod.PanelView

    def keep(*a, **k):
        views.append(real(*a, **k))
        return views[-1]

    monkeypatch.setattr(panel_mod, "PanelView", keep)
    with create_pipe_input() as inp, create_app_session(input=inp, output=output or DummyOutput()):
        task = asyncio.ensure_future(repl.run([]))
        try:
            await until(lambda: views and not views[0].is_busy() and views[0].app.is_running)
            view = views[0]

            async def send(text):
                """One line, then wait until the command (or the question) is handled."""
                inp.send_text(text + "\r")

            send.raw = inp.send_text  # keys without Enter, e.g. "\x03"
            yield view, repl, send
            await until(lambda: not view.is_busy())
            inp.send_text("/layout inline\r")
            await until(lambda: repl.panel is None and repl.prompt_session is not None)
            inp.send_text("/quit\r")
            await asyncio.wait_for(task, 5)
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(BaseException):
                    await task


def flat(view):
    """The chat text with line wrapping undone."""
    return " ".join(view.log.text.split())


async def answer(view, send, text):
    """Wait for a question in the box, answer it, wait until it is gone or replaced."""
    await until(lambda: bool(view.question))
    pending = view._answer
    question = view.question
    await send(text)
    await until(lambda: view._answer is not pending or not view.is_busy())
    return question


async def test_char_wizard_runs_in_the_box_and_saves(chat, tmp_path, monkeypatch):
    async with running_panel(chat, tmp_path, monkeypatch) as (view, repl, send):
        await send("/char new")
        asked = [await answer(view, send, "Kai")]  # name
        for text in ["", "", ""]:  # content, age, gender skipped
            asked.append(await answer(view, send, text))
        asked.append(await answer(view, send, "a ferry pilot"))  # concept
        for _ in range(13):  # appearance ... example line: skipped
            asked.append(await answer(view, send, ""))
        asked.append(await answer(view, send, "n"))  # no AI help
        asked.append(await answer(view, send, "*Waves.* Ahoy."))  # opening scene
        asked.append(await answer(view, send, "y"))  # save
        asked.append(await answer(view, send, "n"))  # don't start now
        await until(lambda: not view.is_busy())
        assert not view.question
        assert all(asked) and asked[0] == repl.t("q_name")
        assert asked[1].startswith(repl.t("q_content", name="Kai")[:20])
    kai = chat.store.find_character("kai")
    assert kai is not None
    assert kai.card.first_mes == "*Waves.* Ahoy."
    assert "a ferry pilot" in kai.card.description or "a ferry pilot" in kai.card.personality
    assert chat.require_session().character_id != kai.id  # "n": not started


async def test_ctrl_c_on_a_question_aborts_the_wizard_and_the_panel_lives(
    chat, tmp_path, monkeypatch
):
    async with running_panel(chat, tmp_path, monkeypatch) as (view, repl, send):
        await send("/char new")
        await answer(view, send, "Sam")
        await until(lambda: bool(view.question))  # content question
        send.raw("half an ans")
        await until(lambda: view.input.text == "half an ans")
        send.raw("\x03")
        await until(lambda: repl.t("aborted") in flat(view) and not view.is_busy())
        assert not view.question and view.input.text == ""
        assert chat.store.find_character("sam") is None  # nothing saved
        await send("hello again")
        await until(lambda: "hello again" in flat(view) and not view.is_busy())
        assert chat.history()[-1].content.endswith("hello again")


@respx.mock
async def test_key_set_is_asked_in_the_password_box(tmp_path, store, monkeypatch, fake_keychain):
    config = Config(providers={"openrouter": ProviderConfig(
        type="openai_compat", base_url=OR, api_key_env="OPENROUTER_API_KEY")})  # fmt: skip
    registry = ProviderRegistry(config, KeyStore(tmp_path / "keys.json"))
    chat = ChatService(config, store, registry)
    chat.new_session(model="echo:echo")
    route = respx.get(f"{OR}/key").mock(return_value=httpx.Response(200, json={"data": {}}))
    async with running_panel(chat, tmp_path, monkeypatch) as (view, _repl, send):
        await send("/key set openrouter")
        await until(lambda: bool(view.question))
        assert view._password is True and "openrouter" in view.question
        await send(GOOD)
        await until(lambda: "saved in the OS keychain" in flat(view) and not view.is_busy())
        assert view._password is False and not view.question
        assert GOOD not in flat(view)  # never echoed into the chat
    assert route.calls[0].request.headers["authorization"] == f"Bearer {GOOD}"
    assert keyring.get_password("chatclient", "openrouter") == GOOD
    await chat.registry.aclose()


@respx.mock
async def test_key_set_cancelled_with_ctrl_c_saves_nothing(tmp_path, store, monkeypatch):
    config = Config(providers={"openrouter": ProviderConfig(
        type="openai_compat", base_url=OR, api_key_env="OPENROUTER_API_KEY")})  # fmt: skip
    chat = ChatService(config, store, ProviderRegistry(config, KeyStore(tmp_path / "keys.json")))
    chat.new_session(model="echo:echo")
    route = respx.get(f"{OR}/key").mock(return_value=httpx.Response(200, json={"data": {}}))
    async with running_panel(chat, tmp_path, monkeypatch) as (view, repl, send):
        await send("/key set openrouter")
        await until(lambda: view._password)
        send.raw("sk-typed")
        await until(lambda: view.input.text == "sk-typed")
        send.raw("\x03")
        await until(lambda: repl.t("aborted") in flat(view) and not view.is_busy())
        assert not route.called
        assert keyring.get_password("chatclient", "openrouter") is None
    await chat.registry.aclose()


async def test_ctrl_c_stops_a_streaming_reply_and_keeps_the_partial_text(
    chat, tmp_path, monkeypatch
):
    monkeypatch.setattr(EchoProvider, "delay", 0.05)
    words = " ".join(f"word{i}" for i in range(60))  # about 3 seconds
    async with running_panel(chat, tmp_path, monkeypatch) as (view, repl, send):
        await send(words)
        await until(lambda: "word3" in flat(view), 5)
        assert view.is_busy()
        send.raw("\x03")
        await until(lambda: repl.t("cancelled") in flat(view) and not view.is_busy())
        assert "word3" in flat(view) and "word59" not in flat(view)
        await send("/help")  # still alive
        await until(lambda: "/layout" in flat(view) and not view.is_busy())


async def test_narrow_terminal_has_no_side_panel_but_chat_works(chat, tmp_path, monkeypatch):
    async with running_panel(chat, tmp_path, monkeypatch, output=NarrowOutput()) as (
        view,
        _repl,
        send,
    ):
        assert view.size()[0] == 60
        assert view.side_width() == 0
        assert view.chat_width() == 59
        await send("hello narrow")
        await until(lambda: "hello narrow" in flat(view) and not view.is_busy())
        assert view.side == [] and "echo" in flat(view)


async def test_clear_empties_the_chat_area_and_layout_reports(chat, tmp_path, monkeypatch):
    async with running_panel(chat, tmp_path, monkeypatch) as (view, repl, send):
        await send("hello there")
        await until(lambda: "hello there" in flat(view) and not view.is_busy())
        assert view.log.text.strip()
        await send("/clear")
        await until(lambda: view.log.text == "" and not view.is_busy())
        await send("/layout")
        want = " ".join(repl.t("layout_current", layout="panel").split())
        await until(lambda: want in flat(view) and not view.is_busy())
        assert chat.history()[-1].content  # clearing the screen keeps the chat itself


class ResizableOutput(DummyOutput):
    columns = 60
    rows = 24

    def get_size(self):
        return Size(rows=self.rows, columns=self.columns)


async def test_resize_rebuilds_the_side_panel(chat, tmp_path, monkeypatch):
    out = ResizableOutput()
    async with running_panel(chat, tmp_path, monkeypatch, output=out) as (view, _repl, _send):
        assert view.side_width() == 0 and view.side == []
        out.columns = 120  # the window is widened
        view.invalidate()
        await until(lambda: view.side_width() > 0 and view.side != [])
        assert view.repl.console.width == view.chat_width()
        out.columns = 60
        view.invalidate()
        await until(lambda: view.side == [] and view.chat_width() == 59)


async def test_ctrl_l_forgets_the_drawn_picture(chat, tmp_path, monkeypatch):
    async with running_panel(chat, tmp_path, monkeypatch) as (view, _repl, send):
        view._drawn = ("pic", 1, 2, 3)
        send.raw("\x0c")
        await until(lambda: view._drawn is None)


def test_chat_log_crlf_is_a_line_ending():
    from chatclient.ui.panel import ChatLog

    log = ChatLog()
    log.write("first\r\nsecond\r\n")
    assert log.text == "first\nsecond\n"
    log.write("third\r")  # a split pair: the \r and \n arrive in two writes
    log.write("\nfourth")
    assert log.text == "first\nsecond\nthird\nfourth"


def test_lone_cell_size_reply_does_not_turn_sixel_off(monkeypatch):
    from chatclient.ui import sixel

    class Tty:
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", Tty())
    monkeypatch.setattr("sys.stdout", Tty())
    monkeypatch.setattr(sixel, "_asked", None)
    monkeypatch.setattr(sixel, "_cell", None)
    monkeypatch.setattr(sixel, "_read_da1_windows", lambda t: "\x1b[6;20;10t")
    monkeypatch.setattr(sixel, "_read_da1_posix", lambda t: "\x1b[6;20;10t")
    sixel.ask_terminal()
    assert sixel._asked is None and sixel.cell_size() == (10, 20)
    monkeypatch.setattr(sixel, "_read_da1_windows", lambda t: "\x1b[?61;4c")
    monkeypatch.setattr(sixel, "_read_da1_posix", lambda t: "\x1b[?61;4c")
    sixel.ask_terminal()
    assert sixel._asked is True


async def test_only_the_current_picture_stays_encoded(chat, tmp_path, monkeypatch):
    from PIL import Image

    from chatclient.ui import sixel

    monkeypatch.setattr(sixel, "encode", lambda p, w, h: f"<{p.name}>")
    repl = make_repl(chat, tmp_path)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        view = panel_mod.PanelView(repl, repl.completer())
        for name in ("a.png", "b.png"):
            Image.new("RGB", (8, 8)).save(tmp_path / name)
            view._pic = (str(tmp_path / name), 8, 8)
            view._prepare_picture()
            assert view._encoded == (view._pic, f"<{name}>")
        view._pic = None
        view._prepare_picture()
        assert view._encoded is None


async def test_side_panel_scrolls_with_a_hint_and_keys(chat, tmp_path, monkeypatch):
    from chatclient.characters import Card

    card = Card(
        name="Lena",
        description="Lena, 27, a night-shift nurse.",
        personality="Temperament: calm\nSpeech: dry humour\nValues: loyal\nFear: crowds",
    )
    char = chat.store.add_character("ai", card)
    chat.new_session(model="echo:echo", character_id=char.id)
    out = ResizableOutput()
    out.columns, out.rows = 100, 9  # the panel (about 11 lines) does not fit
    async with running_panel(chat, tmp_path, monkeypatch, output=out) as (view, _repl, send):
        await until(lambda: len(view._side_lines()) > view.side_height() > 0)

        def shown():
            return "".join(text for _, text in view._side_view())

        assert "↓" in shown() and "↑" not in shown() and view.side_scroll == 0
        send.raw("\x1b\x1b[6~")  # Alt+PageDown
        await until(lambda: view.side_scroll > 0)
        assert "↑" in shown()
        send.raw("\x1b\x1b[5~")  # Alt+PageUp
        await until(lambda: view.side_scroll == 0)
        view.scroll_side(1000)  # clamped: the last line shows, no hint below
        assert "↓" not in shown() and "gentle" not in shown()
        assert view.scroll == 0  # the chat did not move


def test_side_scroll_moves_or_hides_the_picture(monkeypatch, chat, tmp_path):
    from chatclient.ui import sixel

    monkeypatch.setattr(sixel, "encode", lambda p, w, h: "<sixel>")
    repl = make_repl(chat, tmp_path)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        view = panel_mod.PanelView(repl, repl.completer())
        view.side = [("", "\n" * 4 + "\n"), ("", "text\n" * 20)]  # 4 picture lines, a gap
        view._pic, view._pic_rows = (str(tmp_path / "p.png"), 10, 10), 4
        view._prepare_picture()
        app = type("A", (), {"output": type("O", (), {
            "write_raw": lambda s, d: raws.append(d), "flush": lambda s: None})(),
            "renderer": type("R", (), {"clear": lambda s: cleared.append(1)})(),
            "invalidate": lambda s: None})()  # fmt: skip
        raws, cleared = [], []
        view._draw_picture(app)  # at the top: drawn on row 1
        assert len(raws) == 1 and "\x1b[1;" in raws[0]
        view.side_scroll = 3  # its first lines are out of view: hidden like a change
        view._draw_picture(app)
        assert cleared == [1] and view._drawn is None
        view.side_scroll = 0
        view._draw_picture(app)
        assert len(raws) == 2 and "\x1b[1;" in raws[1]
        view.side = [("", "\n" * 2 + "text\n" * 3)]  # fits and none is hidden
        assert view._picture_row() == 1
