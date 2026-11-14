"""Characters, personas and chats as files; the database is an index rebuilt from them."""

from __future__ import annotations

import json
import zipfile

import pytest
from tests.conftest import ALPHA

from chatclient.characters import Card
from chatclient.config import Config, ProviderConfig
from chatclient.core.chat import ChatService, TurnResult
from chatclient.library import Library, card_id
from chatclient.memory.repo import MemoryRepo, Scope
from chatclient.providers.echo import EchoProvider
from chatclient.providers.registry import ProviderRegistry
from chatclient.storage import Store


def open_home(root):
    root.mkdir(parents=True, exist_ok=True)
    lib = Library(root)
    store = Store(root / "index.db", library=lib)
    lib.sync(store)
    config = Config(
        providers={"local": ProviderConfig(type="openai_compat", base_url="http://x/v1")}
    )
    return lib, store, ChatService(config, store, ProviderRegistry(config))


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr(EchoProvider, "delay", 0.0)
    lib, store, chat = open_home(tmp_path / "data")
    yield tmp_path / "data", lib, store, chat
    store.close()


async def say(chat, text):
    r = TurnResult()
    async for _ in chat.send(text, r):
        pass
    return r


async def regen(chat):
    r = TurnResult()
    async for _ in chat.regenerate(r):
        pass


def events(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


async def test_a_character_and_its_chats_are_plain_files(home):
    root, _lib, store, chat = home
    alpha = store.add_character("ai", Card(**{**ALPHA.__dict__, "extra": {}}))
    card_file = root / "characters" / "Alpha" / "card.json"
    assert card_id(Card.from_json(card_file.read_text(encoding="utf-8"))) == alpha.uuid
    chat.new_session(model="echo:echo", character_id=alpha.id)
    await say(chat, "hello")
    (chat_file,) = (root / "characters" / "Alpha" / "chats").glob("*.jsonl")
    evs = events(chat_file)
    assert evs[0]["t"] == "session" and evs[0]["character"] == alpha.uuid
    assert [e["role"] for e in evs if e["t"] == "msg"] == ["assistant", "user", "assistant"]
    assert chat.require_session().log_size == chat_file.stat().st_size


async def test_the_index_is_rebuilt_from_the_files(home):
    root, _lib, store, chat = home
    alpha = store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=alpha.id)
    await say(chat, "first")
    await say(chat, "second")
    await regen(chat)  # a branch: two versions of the last reply
    store.set_message_meta(chat.last_reply().id, {"refused": True})
    before = [(m.role, m.content) for m in chat.history()]
    versions = len(chat.alternatives()[0])
    store.close()
    (root / "index.db").unlink()
    for extra in root.glob("index.db-*"):
        extra.unlink()

    _lib, store2, chat2 = open_home(root)
    s = store2.latest_session()
    chat2.open_session(s.id)
    assert [(m.role, m.content) for m in chat2.history()] == before
    assert len(chat2.alternatives()[0]) == versions
    assert chat2.last_reply().meta == {"refused": True}
    assert chat2.character().name == "Alpha"
    store2.close()


async def test_changes_made_to_the_files_are_picked_up(home):
    root, _lib, store, chat = home
    chat.new_session(model="echo:echo")
    await say(chat, "hello")
    s = chat.require_session()
    (path,) = (root / "plain-chats").glob("*.jsonl")
    reply = chat.last_reply()
    with open(path, "a", encoding="utf-8") as f:  # edited elsewhere (another device, by hand)
        f.write(json.dumps({"t": "edit", "id": reply.uuid, "content": "changed"}) + "\n")
    card_dir = root / "characters" / "Hand_made"
    card_dir.mkdir(parents=True)
    (card_dir / "card.json").write_text(json.dumps({"name": "Hand made"}), encoding="utf-8")
    store.close()
    _lib, store2, _chat2 = open_home(root)
    assert store2.get_message(reply.id).content == "changed"
    assert store2.get_session(s.id).log_size == path.stat().st_size
    made = store2.find_character("Hand made")
    assert made is not None and made.uuid  # given an id, written back to its file
    assert card_id(Card.from_json((card_dir / "card.json").read_text(encoding="utf-8")))
    store2.close()


async def test_a_deleted_folder_means_a_deleted_character(home):
    import shutil

    root, _lib, store, chat = home
    alpha = store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=alpha.id)
    await say(chat, "hi")
    store.close()
    shutil.rmtree(root / "characters" / "Alpha")
    _lib, store2, _chat2 = open_home(root)
    assert store2.find_character("Alpha") is None and store2.all_sessions() == []
    store2.close()


async def test_a_half_written_line_is_skipped_and_repaired(home):
    root, _lib, store, chat = home
    chat.new_session(model="echo:echo")
    await say(chat, "hello")
    (path,) = (root / "plain-chats").glob("*.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"t": "msg", "id": "cut-off-by-a-cra')
    await say(chat, "after the crash")  # starts on a fresh line
    store.close()
    _lib, store2, chat2 = open_home(root)
    chat2.open_session(store2.latest_session().id)
    assert [m.content for m in chat2.history() if m.role == "user"] == ["hello", "after the crash"]
    assert len(chat2.history()) == 4
    store2.close()


async def test_renaming_moves_the_folder_and_deleting_removes_it(home):
    root, _lib, store, chat = home
    alpha = store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=alpha.id)
    await say(chat, "hi")
    card = alpha.card
    card.name = "Alpha Prime"
    store.update_character(alpha.id, card)
    assert (root / "characters" / "Alpha_Prime" / "card.json").is_file()
    assert not (root / "characters" / "Alpha").exists()
    assert len(list((root / "characters" / "Alpha_Prime" / "chats").glob("*.jsonl"))) == 1
    chat.delete_character(store.get_character(alpha.id))
    assert not (root / "characters" / "Alpha_Prime").exists()


async def test_existing_database_is_written_out_on_first_run(tmp_path, monkeypatch):
    monkeypatch.setattr(EchoProvider, "delay", 0.0)
    root = tmp_path / "data"
    root.mkdir()
    old = Store(root / "index.db")  # before files: a database only
    config = Config(
        providers={"local": ProviderConfig(type="openai_compat", base_url="http://x/v1")}
    )
    chat = ChatService(config, old, ProviderRegistry(config))
    beta = old.add_character("ai", Card(name="Beta", description="d"))
    chat.new_session(model="echo:echo", character_id=beta.id)
    await say(chat, "kept")
    MemoryRepo(old).add_memory(Scope(chat.require_session().id, beta.id), "Beta likes tea.")
    old.close()

    _lib, store, _chat = open_home(root)  # first run with files
    assert (root / "characters" / "Beta" / "card.json").is_file()
    assert len(list((root / "characters" / "Beta" / "chats").glob("*.jsonl"))) == 1
    assert "Beta likes tea." in (root / "characters" / "Beta" / "memory.jsonl").read_text("utf-8")
    store.close()
    (root / "index.db").unlink()  # and everything comes back from the files alone
    _lib, store, chat = open_home(root)
    beta = store.find_character("Beta")
    chat.open_session(store.latest_session().id)
    assert [m.role for m in chat.history()] == ["user", "assistant"]
    assert chat.history()[0].content == "kept"
    assert [m.content for m in MemoryRepo(store).list_memories(Scope(0, beta.id))] == [
        "Beta likes tea."
    ]
    store.close()


async def test_export_and_import_on_another_machine(home, tmp_path):
    _root, lib, store, chat = home
    alpha = store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=alpha.id)
    await say(chat, "remember the lighthouse")
    MemoryRepo(store).add_memory(
        Scope(chat.require_session().id, alpha.id), "They saw a lighthouse."
    )
    chat.new_session(model="echo:echo", character_id=alpha.id)
    chat.set_private(True)
    await say(chat, "a secret")
    out = lib.export_character(store, store.get_character(alpha.id), tmp_path / "out")
    names = zipfile.ZipFile(out).namelist()
    assert "card.json" in names and "memory.jsonl" in names
    assert sum(n.startswith("chats/") for n in names) == 1  # the private chat stays home

    other_lib, other, other_chat = open_home(tmp_path / "other")
    got = other_lib.import_path(other, out)
    assert (got.kind, got.chats, got.memories, got.copy) == ("character", 1, 1, False)
    a2 = other.find_character("Alpha")
    assert a2.uuid == alpha.uuid  # same character on both machines
    other_chat.open_session(other.latest_session().id)
    assert "remember the lighthouse" in [m.content for m in other_chat.history()]
    assert (tmp_path / "other" / "characters" / "Alpha" / "memory.jsonl").is_file()
    other.close()


async def test_importing_something_already_here_makes_a_copy(home, tmp_path):
    _root, lib, store, chat = home
    alpha = store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=alpha.id)
    await say(chat, "hi")
    out = lib.export_character(store, store.get_character(alpha.id), tmp_path / "out")
    got = lib.import_path(store, out)
    assert got.copy and got.chats == 1
    alphas = [c for c in store.list_characters("ai") if c.name == "Alpha"]
    assert len(alphas) == 2 and alphas[0].uuid != alphas[1].uuid
    assert len(store.all_sessions()) == 2  # the chat was copied, not merged


async def test_a_chat_file_can_be_imported_alone(home, tmp_path):
    root, _lib, _store, chat = home
    chat.new_session(model="echo:echo")
    await say(chat, "portable chat")
    (path,) = (root / "plain-chats").glob("*.jsonl")
    _other_lib, other, other_chat = open_home(tmp_path / "other")
    got = _other_lib.import_path(other, path)
    assert got.kind == "chat" and not got.copy
    other_chat.open_session(other.latest_session().id)
    assert [m.role for m in other_chat.history()] == ["user", "assistant"]
    assert other_chat.history()[0].content == "portable chat"
    other.close()


def test_personas_have_their_own_folder(home):
    root, lib, store, chat = home
    chat.new_session(model="echo:echo")
    chat.set_persona(Card(name="Sam", description="28"))
    assert (root / "personas" / "Sam" / "persona.json").is_file()
    path = lib.export_character(store, chat.persona(), root / "out")
    assert zipfile.ZipFile(path).namelist() == ["persona.json"]


def test_folder_names_are_safe():
    from chatclient.library import slug

    assert slug('A/B:C*?"') == "ABC"
    assert slug("Çağla Şen") == "Çağla_Şen"
    assert slug("con") == "con_" and slug("...") == "unnamed"


def test_portable_mode_keeps_data_next_to_the_install(tmp_path, monkeypatch):
    import sys

    from chatclient.paths import resolve_paths

    monkeypatch.delenv("CHATCLIENT_HOME", raising=False)
    venv = tmp_path / "ChatClient" / ".venv"
    venv.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(venv))
    assert resolve_paths().data_dir != tmp_path / "ChatClient" / "ChatClientData"
    (tmp_path / "ChatClient" / "portable").write_text("")
    assert resolve_paths().data_dir == (tmp_path / "ChatClient" / "ChatClientData").resolve()


async def test_the_old_chats_folder_moves_to_plain_chats(home):
    root, _lib, store, chat = home
    chat.new_session(model="echo:echo")
    await say(chat, "from before the rename")
    store.close()
    old = root / "chats"
    (root / "plain-chats").rename(old)  # as an earlier version left it
    _lib, store2, chat2 = open_home(root)
    assert not old.exists() and len(list((root / "plain-chats").glob("*.jsonl"))) == 1
    chat2.open_session(store2.latest_session().id)
    assert chat2.history()[0].content == "from before the rename"
    store2.close()


async def test_pictures_travel_with_the_character(home, tmp_path):
    root, lib, store, chat = home
    alpha = store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=alpha.id)
    await say(chat, "hi")
    images = root / "characters" / "Alpha" / "images"
    images.mkdir()
    (images / "shown.png").write_bytes(b"PNG1")
    (images / "shown.json").write_text('{"prompt": "x"}', encoding="utf-8")
    reply = chat.last_reply()
    store.set_message_meta(reply.id, {"images": ["shown.png"]})
    chat.new_session(model="echo:echo", character_id=alpha.id)
    chat.set_private(True)
    await say(chat, "secret")
    (images / "secret.png").write_bytes(b"PNG2")
    store.set_message_meta(chat.last_reply().id, {"images": ["secret.png"]})

    out = lib.export_character(store, store.get_character(alpha.id), tmp_path / "out")
    names = zipfile.ZipFile(out).namelist()
    assert "images/shown.png" in names and "images/shown.json" in names
    assert "images/secret.png" not in names  # drawn in a private chat: stays home

    other_lib, other, _ = open_home(tmp_path / "other")
    other_lib.import_path(other, out)
    got = tmp_path / "other" / "characters" / "Alpha" / "images" / "shown.png"
    assert got.read_bytes() == b"PNG1"
    other.close()
