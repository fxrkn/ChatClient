"""/edit, /continue, /ooc, /copy, /export, /lang, /regen <model>, Ctrl+P, drafts, /setup."""

import re

import httpx
import keyring
import pytest
import respx
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from chatclient.config import Config, ProviderConfig
from chatclient.core.chat import CONTINUE_NOTE, ChatError, ChatService, TurnResult
from chatclient.export import ExportMessage, render
from chatclient.i18n import Translator
from chatclient.keys import KeyStore
from chatclient.paths import resolve_paths
from chatclient.providers.base import StreamEvent
from chatclient.providers.echo import EchoProvider
from chatclient.providers.registry import ProviderRegistry
from chatclient.ui import tools_ui
from chatclient.ui.repl import Repl

ANSI = re.compile(r"\x1b\[[0-9;]*m")
DOWN = "\x1b[B"


async def drain(agen):
    return [ev async for ev in agen]


async def run_keys(chat, tmp_path, keys, lang="en"):
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator(lang), paths, show_stats=True)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(keys)
        await repl.run([])
    return repl


# ── chat service ─────────────────────────────────────────────────────────────


async def test_edit_and_resend_keeps_original_as_branch(chat):
    await drain(chat.send("ilk hali", TurnResult()))
    original = chat.last_user_message()
    await drain(chat.edit_and_resend("düzeltilmiş hali", TurnResult()))
    msgs = chat.history()
    assert [m.content for m in msgs if m.role == "user"] == ["düzeltilmiş hali"]
    assert "düzeltilmiş hali" in msgs[-1].content  # echo replied to the edited text
    assert chat.store.get_message(original.id) is not None  # original still stored
    assert msgs[0].meta["edited_from"] == original.id


async def test_replace_reply_is_a_new_version(chat):
    await drain(chat.send("selam", TurnResult()))
    old = chat.last_reply()
    new = chat.replace_reply("Elle düzeltilmiş yanıt.")
    assert chat.last_reply().id == new.id and new.content == "Elle düzeltilmiş yanıt."
    sibs, index = chat.alternatives()
    assert [m.id for m in sibs] == [old.id, new.id] and index == 1
    with pytest.raises(ChatError):
        chat.replace_reply("   ")


async def test_continue_appends_to_the_same_message(chat, monkeypatch):
    await drain(chat.send("anlat", TurnResult()))
    head = chat.last_reply()
    seen = {}

    async def more(self, model, messages, params):
        seen["last"] = messages[-1].content
        yield StreamEvent(text="ve sonra devam etti.")

    monkeypatch.setattr(EchoProvider, "stream_chat", more)
    r = TurnResult()
    await drain(chat.continue_reply(r))
    assert seen["last"] == CONTINUE_NOTE  # asked to continue, without prefill
    after = chat.last_reply()
    assert after.id == head.id  # same message, extended
    assert after.content == head.content + " ve sonra devam etti."
    assert chat.store.search("devam etti", session_id=chat.session.id)  # search index updated


async def test_continue_needs_a_reply(chat):
    with pytest.raises(ChatError):
        await drain(chat.continue_reply(TurnResult()))


async def test_regen_with_another_model_keeps_session_model(chat, monkeypatch):
    await drain(chat.send("selam", TurnResult()))
    used = {}

    async def other(self, model, messages, params):
        used["provider"] = self.name
        yield StreamEvent(text="from the other model")

    monkeypatch.setattr(EchoProvider, "stream_chat", other)
    chat.registry.configs["echo2"] = chat.registry.configs["echo"]
    r = TurnResult()
    await drain(chat.regenerate(r, model="echo2:echo"))
    assert used["provider"] == "echo2"
    assert chat.last_reply().model == "echo2:echo"
    assert chat.require_session().model == "echo:echo"  # unchanged
    with pytest.raises(Exception, match="unknown provider"):
        await drain(chat.regenerate(TurnResult(), model="nope:x"))
    assert chat.last_reply().content == "from the other model"  # history untouched by the error


# ── export ───────────────────────────────────────────────────────────────────


def test_export_formats():
    msgs = [
        ExportMessage("Sen", "Merhaba <b>", True, "2026-11-13 10:00"),
        ExportMessage("Ada", '*gülümser* "Hoş geldin."\n\nİkinci paragraf.', False),
    ]
    md = render("md", "Başlık", "echo:echo · 2 messages", msgs)
    assert md.startswith("# Başlık") and "**Sen** · 2026-11-13 10:00" in md
    html = render("html", "Başlık", "", msgs)
    assert "Merhaba &lt;b&gt;" in html  # escaped
    assert '<em class="action">gülümser</em>' in html
    assert '<span class="speech">&quot;Hoş geldin.&quot;</span>' in html
    assert html.count("<p>") == 3
    txt = render("txt", "Başlık", "", msgs)
    assert "Ada:\n*gülümser*" in txt
    with pytest.raises(ValueError):
        render("pdf", "x", "", msgs)


# ── REPL commands ────────────────────────────────────────────────────────────


async def test_repl_edit_ooc_continue_copy_export(chat, tmp_path, capsys, monkeypatch):
    copied = []
    monkeypatch.setattr(tools_ui, "copy_text", lambda text: copied.append(text) or "system")
    keys = (
        "ilk mesaj\r"
        "/edit\r" + "\x08" * 5 + "yeni\r"  # backspace "mesaj", type "yeni"
        "/ooc hikâyeyi yavaşlat\r"
        "/continue\r"
        "/copy\r"
        "/export html\r"
        "/export pdf\r"
        "/quit\r"
    )
    await run_keys(chat, tmp_path, keys)
    out = ANSI.sub("", capsys.readouterr().out)
    users = [m.content for m in chat.history() if m.role == "user"]
    assert users == ["ilk yeni", "(OOC: hikâyeyi yavaşlat)"]
    assert copied and copied[0] == chat.last_reply().content
    assert "Copied to the clipboard." in out
    exported = list((tmp_path / "exports").glob("*.html"))
    assert len(exported) == 1 and "ilk yeni" in exported[0].read_text(encoding="utf-8")
    assert "Usage: /export" in out


async def test_repl_edit_reply_and_lang(chat, tmp_path, capsys):
    await drain(chat.send("selam", TurnResult()))
    await run_keys(
        chat, tmp_path, "/edit reply\r" + "\x15" + "Düzeltildi.\r/lang tr\r/help\r/quit\r"
    )
    out = ANSI.sub("", capsys.readouterr().out)
    assert chat.last_reply().content == "Düzeltildi."
    assert "Arayüz dili: Türkçe" in out and "Komutlar" in out
    assert chat.store.kv_get("ui_lang") == "tr"


async def test_palette_runs_or_prefills(chat, tmp_path, capsys):
    keys = (
        "\x10" + "sessions\r"  # Ctrl+P, filter, Enter → runs /sessions
        "\x10" + "ooc\r" + "test\r"  # Ctrl+P → /ooc needs an argument → prefilled
        "/quit\r"
    )
    await run_keys(chat, tmp_path, keys)
    out = ANSI.sub("", capsys.readouterr().out)
    assert "updated" in out  # /sessions table header
    assert chat.last_user_message().content == "(OOC: test)"


async def test_draft_survives_restart_and_ctrl_c_discards(chat, tmp_path, capsys):
    key = f"draft:{chat.session.id}"
    # Type something, then leave through the palette (/quit) without sending it.
    await run_keys(chat, tmp_path, "yarım kalan" + "\x10" + "/quit\r")
    assert chat.store.kv_get(key) == "yarım kalan"
    # Next start: it's back in the input box; Enter sends it.
    await run_keys(chat, tmp_path, "\r/quit\r")
    assert "unsent message was restored" in capsys.readouterr().out
    assert chat.last_user_message().content == "yarım kalan"
    assert chat.store.kv_get(key) == ""
    # Ctrl+C at the prompt discards the text instead.
    await run_keys(chat, tmp_path, "vazgeçtim" + "\x03" + "/quit\r")
    assert chat.store.kv_get(key) == ""


async def test_cancelled_palette_restores_typed_text(chat, tmp_path):
    await run_keys(chat, tmp_path, "yazıyordum" + "\x10" + "\x03" + " devam\r" + "/quit\r")
    assert chat.last_user_message().content == "yazıyordum devam"


# ── first-run setup ──────────────────────────────────────────────────────────

OR = "https://openrouter.ai/api/v1"


@pytest.fixture
def fresh_chat(store, tmp_path):
    config = Config(
        providers={
            "openrouter": ProviderConfig(type="openai_compat", base_url=OR),
            "ollama": ProviderConfig(type="ollama", base_url="http://127.0.0.1:9"),
        }
    )
    chat = ChatService(config, store, ProviderRegistry(config, KeyStore(tmp_path / "k.json")))
    chat.new_session()
    assert chat.require_session().model is None
    return chat


@respx.mock
async def test_setup_cloud_path(fresh_chat, tmp_path, capsys):
    import json

    respx.get(f"{OR}/key").mock(return_value=httpx.Response(200, json={"data": {}}))
    respx.get(f"{OR}/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "x/free-model:free"}]})
    )
    body = f"data: {json.dumps({'choices': [{'delta': {'content': 'OK'}}]})}\n\ndata: [DONE]\n\n"
    respx.post(f"{OR}/chat/completions").mock(return_value=httpx.Response(200, content=body))
    keys = (
        DOWN + "\r"  # "Online, free models"
        "sk-or-v1-" + "b" * 64 + "\r"  # paste key
        "Mira\r"  # your name
        "n\r"  # don't create a character now
        "/quit\r"
    )
    await run_keys(fresh_chat, tmp_path, keys)
    out = capsys.readouterr().out
    assert "Welcome to ChatClient" in out
    assert fresh_chat.require_session().model == "openrouter:x/free-model:free"
    assert keyring.get_password("chatclient", "openrouter")
    assert fresh_chat.persona().name == "Mira"
    assert fresh_chat.store.kv_get("onboarded") == "1"


async def test_setup_local_path_without_server_and_not_shown_again(fresh_chat, tmp_path, capsys):
    await run_keys(fresh_chat, tmp_path, "\r" + "n\r" + "/quit\r")  # local → none found → no retry
    out = capsys.readouterr().out
    assert "No local server with a model was found" in out
    assert "ollama.com/download" in out
    capsys.readouterr()
    await run_keys(fresh_chat, tmp_path, "/quit\r")
    assert "Welcome to ChatClient" not in capsys.readouterr().out


async def test_existing_users_skip_setup(chat, tmp_path, capsys):
    await run_keys(chat, tmp_path, "/quit\r")
    assert "Welcome to ChatClient" not in capsys.readouterr().out
    assert chat.store.kv_get("onboarded") == "1"
