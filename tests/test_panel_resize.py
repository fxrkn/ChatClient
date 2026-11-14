"""The side panel's width: drag the separator, keys, /layout width, saved, clamped."""

from __future__ import annotations

import pytest
from PIL import Image
from prompt_toolkit.application import create_app_session, set_app
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput
from tests.conftest import ALPHA
from tests.test_panel import make_repl

from chatclient.ui import panel as panel_mod
from chatclient.ui import sixel
from chatclient.ui.repl import PANEL_WIDTH_KEY


@pytest.fixture
def view(chat, tmp_path):
    """A panel on an 80x40 screen (DummyOutput), the panel layout on."""
    repl = make_repl(chat, tmp_path)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        v = panel_mod.PanelView(repl, repl.completer())
        repl.panel = v
        yield v


def mouse(kind, x, button=MouseButton.LEFT):
    return MouseEvent(Point(x, 5), kind, button, frozenset())


def test_width_default_and_saved_override(view):
    repl = view.repl
    assert view.side_width() == 34  # [ui] panel_width
    view.resize(30)
    assert repl.chat.store.kv_get(PANEL_WIDTH_KEY) == "30" and repl.panel_width() == 30
    assert view.side_width() == 30


def test_width_is_clamped_to_the_panel_and_the_chat(view):
    assert view.size()[0] == 80
    view.resize(5)
    assert view.side_width() == panel_mod.MIN_SIDE_WIDTH == 20
    view.resize(500)  # the chat keeps 40 columns (+2 for the separator and margin)
    assert view.side_width() == 80 - panel_mod.MIN_CHAT_WIDTH - 2
    assert view.chat_width() >= panel_mod.MIN_CHAT_WIDTH


def test_keys_widen_and_narrow(view):
    bindings = {tuple(b.keys): b.handler for b in view.app.key_bindings.bindings}
    narrower, wider = bindings[("escape", ",")], bindings[("escape", ".")]
    wider(None)
    assert view.side_width() == 34 + panel_mod.RESIZE_STEP
    narrower(None)
    narrower(None)
    assert view.side_width() == 34 - panel_mod.RESIZE_STEP
    for _ in range(30):
        narrower(None)
    assert view.side_width() == 20  # stops at the minimum


def test_layout_width_command(view):
    repl = view.repl
    repl.layout_command("width 28")
    assert view.side_width() == 28 and repl.panel_width() == 28
    repl.layout_command("width 99")  # too wide for this screen: clamped, and it says so
    assert view.side_width() == 38
    repl.layout_command("width")  # shows the current one
    for bad in ("width x", "width 20 30", "width -5"):
        with pytest.raises(Exception, match="width"):
            repl.layout_command(bad)


def test_layout_width_is_saved_without_a_panel(chat, tmp_path):
    repl = make_repl(chat, tmp_path)
    assert repl.panel is None
    repl.layout_command("width 10")
    assert repl.panel_width() == 20  # never below the minimum
    repl.layout_command("width 44")
    assert repl.panel_width() == 44


def test_dragging_the_separator_resizes_live_and_saves_on_release(view):
    repl = view.repl
    sep = 80 - 34 - 1  # the │ column
    view._separator_event(mouse(MouseEventType.MOUSE_DOWN, sep))
    assert view._dragging
    view._separator_event(mouse(MouseEventType.MOUSE_MOVE, sep - 6))  # 6 columns to the left
    assert view.side_width() == 38  # 40 wanted; the chat keeps its 40
    assert repl.chat.store.kv_get(PANEL_WIDTH_KEY) is None  # not saved while dragging
    view._separator_event(mouse(MouseEventType.MOUSE_MOVE, 52))
    assert view.side_width() == 80 - 52 - 1
    view._separator_event(mouse(MouseEventType.MOUSE_UP, 52, MouseButton.NONE))
    assert not view._dragging and view._live_width is None
    assert repl.chat.store.kv_get(PANEL_WIDTH_KEY) == "27" and repl.panel_width() == 27


def test_a_click_without_moving_changes_nothing(view):
    sep = 80 - 34 - 1
    view._separator_event(mouse(MouseEventType.MOUSE_DOWN, sep))
    view._separator_event(mouse(MouseEventType.MOUSE_UP, sep, MouseButton.NONE))
    assert view.side_width() == 34 and not view._dragging
    assert view.repl.chat.store.kv_get(PANEL_WIDTH_KEY) is None


def test_a_missed_release_ends_the_drag(view):
    sep = 80 - 34 - 1
    view._separator_event(mouse(MouseEventType.MOUSE_DOWN, sep))
    view._separator_event(mouse(MouseEventType.MOUSE_MOVE, sep - 4))
    view._separator_event(mouse(MouseEventType.MOUSE_MOVE, sep - 4, MouseButton.NONE))  # hovering
    assert not view._dragging and view.side_width() == 38


async def test_the_screen_hands_the_pointer_to_the_separator_and_then_catches_it(view):
    """Rendered for real: the │ column and, while dragging, the whole screen, send mouse
    events to the drag handler (the pointer may leave the 1-column separator)."""
    app = view.app
    with set_app(app):
        app.layout.update_parents_relations()  # (the running app does this)
        app.renderer.render(app, app.layout)
        handlers = app.renderer.mouse_handlers.mouse_handlers
        sep = 80 - 34 - 1
        for x in (0, sep - 1, sep + 1, 79):  # chat and panel: not the separator
            handlers[5][x](mouse(MouseEventType.MOUSE_DOWN, x))
            assert not view._dragging
        handlers[5][sep](mouse(MouseEventType.MOUSE_DOWN, sep))
        assert view._dragging
        app.renderer.render(app, app.layout)  # the catcher is on now
        handlers = app.renderer.mouse_handlers.mouse_handlers
        handlers[3][10](mouse(MouseEventType.MOUSE_MOVE, 50))  # far from the separator
        assert view.side_width() == 80 - 50 - 1
        handlers[3][10](mouse(MouseEventType.MOUSE_UP, 50, MouseButton.NONE))
        assert not view._dragging and view.repl.panel_width() == 29


def test_the_picture_is_encoded_again_at_the_new_size(chat, view, tmp_path, monkeypatch):
    pic = tmp_path / "wide.png"
    Image.new("RGB", (1000, 200), (10, 200, 10)).save(pic)
    calls = []
    monkeypatch.setattr(sixel, "supported", lambda setting: True)
    monkeypatch.setattr(sixel, "cell_size", lambda: (10, 20))
    monkeypatch.setattr(sixel, "encode", lambda p, w, h: calls.append((w, h)) or f"<{w}x{h}>")
    char = chat.store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=char.id)
    monkeypatch.setattr(view.repl, "last_picture", lambda: pic)
    view.refresh()
    assert view._pic[1:] == (320, 64) and calls == [(320, 64)]  # panel 34: (34 - 2) * 10 px
    view._drawn = (*view._pic, 47, 80, 40)  # as if it were on screen
    view.resize(24)
    assert view._pic[1:] == (220, 44) and calls[-1] == (220, 44)
    assert view._encoded[0] == view._pic  # the new encoding replaced the old one


def test_chat_text_width_follows_the_panel(view):
    view.resize(20)
    wide = view.chat_width()
    view.resize(38)
    assert view.chat_width() == wide - 18
    assert view.repl.console.width == view.chat_width()
