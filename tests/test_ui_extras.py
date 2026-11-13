import io
import re

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from chatclient.characters import Card, card_color, set_card_color
from chatclient.core.chat import ChatError, TurnResult
from chatclient.i18n import Translator
from chatclient.paths import resolve_paths
from chatclient.theme import PRESETS, character_color, is_light
from chatclient.ui.picker import Picker, PickItem, pick
from chatclient.ui.render import StreamWriter
from chatclient.ui.repl import Repl

ANSI = re.compile(r"\x1b\[[0-9;]*m")
DOWN, ALT_LEFT, ALT_RIGHT = "\x1b[B", "\x1b[1;3D", "\x1b[1;3C"


async def drain(agen):
    return [ev async for ev in agen]


async def run_script(chat, tmp_path, keys):
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths, show_stats=True)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(keys)
        await repl.run([])
    return repl


# ── speech highlighting ──────────────────────────────────────────────────────


def styled(text: str) -> str:
    buf = io.StringIO()
    w = StreamWriter(
        buf, width=200, margin=0, base="<T>", action="<A>", style_actions=True, speech="<S>"
    )
    w.write(text)
    w.finish()
    return buf.getvalue()


def test_speech_is_coloured_and_quotes_stay():
    out = styled('She smiles. "Come in," she says.')
    assert '<S>"Come in,"<T> she says.' in out
    assert (
        ANSI.sub("", out).replace("<T>", "").replace("<S>", "")
        == 'She smiles. "Come in," she says.'
    )


def test_curly_quotes_and_action_inside_speech():
    out = styled("“Merhaba *gülümser* dostum” dedi.")
    assert "<S>“Merhaba" in out
    assert "\x1b[3m<A>gülümser\x1b[23m<S>" in out  # back to speech colour after the action
    assert "dostum”<T>" in out


def test_speech_off_by_default():
    buf = io.StringIO()
    w = StreamWriter(buf, width=200, margin=0, base="<T>")
    w.write('"hi"')
    w.finish()
    assert "<S>" not in buf.getvalue()


# ── character colours ────────────────────────────────────────────────────────


def test_character_colors_are_stable_and_theme_aware():
    soft, light, classic = PRESETS["soft"], PRESETS["light"], PRESETS["classic"]
    assert not is_light(soft) and is_light(light)
    assert character_color("Beta", soft) == character_color("Beta", soft)  # stable
    assert character_color("Beta", soft) != character_color("Beta", light)  # palette per theme
    assert (
        len({character_color(n, soft) for n in ("Beta", "Alpha", "Gamma", "Delta", "Epsilon")}) >= 3
    )
    assert character_color("Beta", soft, "#123456") == "#123456"
    assert character_color("Beta", classic) == "default"


def test_card_color_survives_export_roundtrip():
    card = Card(name="X", extra={"extensions": {"other_app": {"k": 1}}})
    set_card_color(card, "#ff8800")
    back = Card.from_json(card.to_json())
    assert card_color(back) == "#ff8800"
    assert back.extra["extensions"]["other_app"] == {"k": 1}  # other apps' data kept
    set_card_color(back, None)
    assert card_color(back) is None and "chatclient" not in back.extra["extensions"]


# ── picker ───────────────────────────────────────────────────────────────────


def test_picker_filter_move_and_select():
    items = [PickItem(1, "Kütüphane"), PickItem(2, "Liman"), PickItem(3, "İstanbul", "şehir")]
    p = Picker("t", items, current=2)
    assert p.selected().value == 2
    p.move(1)
    p.move(1)
    assert p.selected().value == 1  # wraps around
    p.type("kutup")  # diacritic-insensitive
    assert [i.value for i in p.visible()] == [1]
    p.backspace()
    p.backspace()
    p.backspace()
    p.backspace()
    p.backspace()
    p.type("sehir")  # hints are searchable too
    assert p.selected().value == 3
    p.type("zzz")
    assert p.selected() is None


async def test_pick_keyboard_flow():
    items = [PickItem("a", "Alpha"), PickItem("b", "Beta"), PickItem("c", "Gamma")]
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(DOWN + "\r")
        assert await pick("t", items) == "b"
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text("gam\r")
        assert await pick("t", items) == "c"
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text("\x03")  # Ctrl+C cancels
        assert await pick("t", items) is None


# ── alternatives (swipes) ────────────────────────────────────────────────────


async def test_alternatives_navigation(chat):
    await drain(chat.send("bir", TurnResult()))
    await drain(chat.regenerate(TurnResult()))
    await drain(chat.regenerate(TurnResult()))
    sibs, i = chat.alternatives()
    assert len(sibs) == 3 and i == 2
    first = chat.switch_alternative(-2)
    assert first.id == sibs[0].id and chat.history()[-1].id == sibs[0].id
    with pytest.raises(ChatError):
        chat.switch_alternative(-1)
    assert chat.switch_alternative(2).id == sibs[2].id
    assert chat.switch_alternative(1) is None  # past the newest → caller generates


async def test_no_alternatives_for_greeting(chat, shipped):
    chat.seed_builtin_characters()
    alpha = chat.store.find_character("alpha")
    chat.new_session(model="echo:echo", character_id=alpha.id)
    with pytest.raises(ChatError):
        chat.switch_alternative(1)


async def test_repl_swipe_keys_and_commands(chat, tmp_path, capsys):
    keys = "merhaba\r/regen\r" + ALT_LEFT + "/next\r" + ALT_RIGHT + "/quit\r"
    await run_script(chat, tmp_path, keys)
    out = ANSI.sub("", capsys.readouterr().out)
    assert "‹ 2/2 ›" in out  # after /regen
    assert "‹ 1/2 ›" in out  # Alt+← went back
    assert "‹ 3/3 ›" in out  # Alt+→ on the newest wrote a new version
    sibs, _ = chat.alternatives()
    assert len(sibs) == 3


# ── pickers wired into commands ──────────────────────────────────────────────


async def test_model_switch_and_char_pickers(chat, tmp_path, shipped):
    chat.seed_builtin_characters()
    first = chat.require_session().id
    chat.new_session(model="echo:echo", title="ikinci")
    keys = (
        "/model\r\r"  # only echo:echo answers (the local server is offline) → choose it
        "/switch\r" + DOWN + "\r"  # sessions newest first → second row is the first session
        "/char use\r" + "alpha\r"  # type to filter, Enter
        "/quit\r"
    )
    await run_script(chat, tmp_path, keys)
    s = chat.require_session()
    assert s.character_id == chat.store.find_character("alpha").id
    assert any(x.session.id == first for x in chat.store.list_sessions())


async def test_char_color_command(chat, tmp_path, capsys, shipped):
    chat.seed_builtin_characters()
    alpha = chat.store.find_character("alpha")
    chat.new_session(model="echo:echo", character_id=alpha.id)
    await run_script(chat, tmp_path, "/char color #ff8800\r/char color blue\r/quit\r")
    out = capsys.readouterr().out
    assert "Alpha now uses #ff8800" in out
    assert "Usage: /char color" in out
    assert card_color(chat.store.find_character("alpha").card) == "#ff8800"


async def test_typing_placeholder_without_terminal(chat, tmp_path, capsys):
    await run_script(chat, tmp_path, "selam\r/quit\r")
    out = ANSI.sub("", capsys.readouterr().out)
    assert "echo is typing…" in out  # static (no animation when not a terminal)
    assert "selam" in out
