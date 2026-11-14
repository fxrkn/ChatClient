"""The panel layout: chat on the left, the character beside it, questions in the input."""

import asyncio

from PIL import Image
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from chatclient.characters import Card
from chatclient.i18n import Translator
from chatclient.panel_data import traits
from chatclient.paths import resolve_paths
from chatclient.ui import panel as panel_mod
from chatclient.ui import sixel
from chatclient.ui.panel import ChatLog, fit, thumbnail
from chatclient.ui.repl import LAYOUT_KEY, Repl


def test_chat_log_keeps_text_and_rewrites_the_typing_line():
    log = ChatLog()
    log.write("Lena\n  ⠋ Lena is typing… 0.1s")
    log.write("\r\x1b[2K  ⠙ Lena is typing… 0.2s")
    log.write("\r\x1b[2K  Hello there")
    log.write("\n\x1bPq#0;2;0;0;0~-\x1b\\\n")  # a Sixel picture: dropped
    log.write("\x1b[2J\x1b[H\x1b[1mbold\x1b[0m")  # screen control dropped, colours kept
    assert log.text == "Lena\n  Hello there\n\n\x1b[1mbold\x1b[0m"
    assert log.isatty()


def test_thumbnail_fits_the_panel(tmp_path):
    path = tmp_path / "p.png"
    Image.new("RGB", (512, 768), (200, 10, 10)).save(path)
    rows = thumbnail(path, 30, max_rows=100)
    assert len(rows) == 22 and all(len(r) == 30 for r in rows)  # 2 pixels per cell
    tall = thumbnail(path, 30, max_rows=10)  # not enough room: narrower, not cut
    assert len(tall) == 10 and len(tall[0]) == 13
    assert tall[0][0] == ("fg:#c80a0a bg:#c80a0a", "▀")


def test_traits_drop_labels():
    assert traits("Temperament: warm, teasing.\nSpeaks: softly\n\nloves rain") == [
        "warm, teasing", "softly", "loves rain"]  # fmt: skip


def test_cell_size_answer_and_fit():
    assert sixel.parse_cell("\x1b[6;20;10t\x1b[?61;4;6c") == (10, 20)
    assert sixel.parse_cell("\x1b[?61;4c") is None  # the terminal didn't say
    assert fit((512, 768), 256, 1000) == (256, 384)
    assert fit((512, 768), 1000, 384) == (256, 384)
    assert fit((100, 100), 500, 500) == (100, 100)  # never enlarged


class Recorder:
    def __init__(self):
        self.raw, self.cleared = [], 0
        self.output, self.renderer = self, self

    def write_raw(self, data):
        self.raw.append(data)

    def flush(self):
        pass

    def clear(self):
        self.cleared += 1

    def invalidate(self):
        pass


async def test_picture_is_drawn_once_and_replaced_cleanly(chat, tmp_path, monkeypatch):
    path = tmp_path / "p.png"
    Image.new("RGB", (64, 96), (10, 200, 10)).save(path)
    monkeypatch.setattr(sixel, "encode", lambda p, w, h: f"<sixel {w}x{h}>")
    repl = make_repl(chat, tmp_path)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        view = panel_mod.PanelView(repl, repl.completer())
        app = Recorder()
        view._pic = (str(path), 64, 96)
        view._draw_picture(app)
        assert app.raw == []  # not encoded yet: drawing never encodes
        view._prepare_picture()
        view._draw_picture(app)
        view._draw_picture(app)  # nothing changed: not drawn again
        assert app.raw == ["\x1b7\x1b[1;48H<sixel 64x96>\x1b8"]  # 80 columns, panel 34
        view._pic = None  # e.g. another chat: the old picture is wiped
        view._draw_picture(app)
        assert app.cleared == 1 and view._drawn is None


async def test_scrolling_up_brings_in_older_messages(chat, tmp_path, monkeypatch):
    repl = make_repl(chat, tmp_path)
    for i in range(15):
        await repl.reply(lambda r, i=i: chat.send(f"message number {i}", r))
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        view = panel_mod.PanelView(repl, repl.completer())
        repl.panel = view
        monkeypatch.setattr("sys.stdout", view.log)
        view.shown_session = chat.require_session().id
        repl.show_history(6, top=True)
        assert view.hidden_before == 24 and "message number 11" not in view.log.text
        view.scroll_by(panel_mod.SCROLL_STEP)
        assert view.hidden_before == 14
        assert "message number 7" in view.log.text and "older messages" in view.log.text
        assert view.log.text.index("message number 7") < view.log.text.index("number 14")


def make_repl(chat, tmp_path):
    paths = resolve_paths(tmp_path)
    paths.ensure()
    chat.config.ui.clear_on_start = False
    return Repl(chat, Translator("en"), paths, show_stats=False)


async def until(check, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        if check():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out")


async def test_panel_chats_shows_the_character_and_asks_in_the_box(chat, tmp_path, monkeypatch):
    char = chat.store.add_character("ai", Card(
        name="Lena", description="Lena, 27, a night-shift nurse.",
        personality="Temperament: calm\nSpeech: dry humour"))  # fmt: skip
    lena = chat.new_session(model="echo:echo", character_id=char.id)
    chat.new_session(model="echo:echo", title="other")
    chat.open_session(lena.id)
    chat.store.kv_set(LAYOUT_KEY, "panel")
    repl = make_repl(chat, tmp_path)
    views = []
    real = panel_mod.PanelView

    def keep(*a, **k):
        views.append(real(*a, **k))
        return views[-1]

    monkeypatch.setattr(panel_mod, "PanelView", keep)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        task = asyncio.ensure_future(repl.run([]))
        await until(lambda: views and not views[0].is_busy() and views[0].app.is_running)
        view = views[0]
        side = "".join(text for _, text in view.side)
        assert "Lena · 27" in side and "calm" in side and "dry humour" in side
        inp.send_text("hello\r")
        await until(lambda: "hello" in "".join(m.content for m in chat.history()[-1:]))
        await until(lambda: not view.is_busy())
        assert "Lena" in view.log.text
        # a list to pick from: numbered in the chat, answered in the input box
        inp.send_text("/switch\r")
        await until(lambda: bool(view.question))
        assert "#" in view.log.text and "1)" in view.log.text
        inp.send_text("1\r")
        await until(lambda: not view.is_busy())
        inp.send_text("/layout inline\r")  # back to one column, then quit there
        await until(lambda: repl.panel is None and repl.prompt_session is not None)
        inp.send_text("/quit\r")
        await task
    assert chat.store.kv_get(LAYOUT_KEY) == "inline"
    import sys

    assert not isinstance(sys.stdout, ChatLog)  # the terminal is back
