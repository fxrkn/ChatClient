"""The side panel's data (shared by the terminal panel and ``--serve``)."""

from __future__ import annotations

import ast
from pathlib import Path

from tests.conftest import ALPHA

from chatclient import panel_data as panel_data_module
from chatclient.config import MemoryConfig
from chatclient.memory.engine import MemoryEngine
from chatclient.panel_data import panel_data, traits


def test_traits_drop_labels_and_trailing_dots():
    text = "Temperament: warm.\nA very long label that is no label: keep\n\nplain line"
    assert traits(text) == ["warm", "A very long label that is no label: keep", "plain line"]
    assert traits("") == []


def test_no_character_uses_the_title(chat):
    d = panel_data(chat)
    assert d.name is None and d.age is None and d.picture is None
    assert d.personality == [] and d.scene == {} and d.memories == []
    assert d.model == "echo:echo" and not d.private


def test_character_scene_memories_and_picture(chat, tmp_path):
    chat.memory = MemoryEngine(
        chat.config.model_copy(update={"memory": MemoryConfig(embeddings="off")}),
        chat.store,
        chat.registry,
    )
    char = chat.store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=char.id)
    s = chat.require_session()
    chat.store.set_scene(s.id, {"location": "a harbour", "mood": " ", "time": "dusk", "x": "no"})
    scope = chat.memory.scope(s)
    for i in range(7):
        chat.memory.repo.add_memory(scope, f"fact {i}", kind="preference")
    pic = tmp_path / "shot.png"

    d = panel_data(chat, pic)
    assert d.name == "Alpha" and d.personality == ["Calm test personality"]
    assert d.scene == {"location": "a harbour", "time": "dusk"}
    assert len(d.memories) == 5
    assert set(d.memories[0]) == {"content", "pinned", "kind"}
    assert d.picture == str(pic.resolve())

    chat.set_private(True)  # private chats show no memories
    assert panel_data(chat).memories == []
    assert panel_data(chat).as_dict()["private"] is True


def test_the_module_imports_no_terminal_or_ui_code():
    """(rich still comes in through images/characters; that is outside this module.)"""
    tree = ast.parse(Path(panel_data_module.__file__).read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    bad = [n for n in names if n.split(".")[0] in ("rich", "prompt_toolkit") or ".ui" in n]
    assert bad == []
