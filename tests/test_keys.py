import os
import stat

import httpx
import keyring
import pytest
import respx
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from chatclient.config import Config, ProviderConfig
from chatclient.core.chat import ChatService
from chatclient.i18n import Translator
from chatclient.keys import SOURCE_CONFIG, SOURCE_ENV, SOURCE_FILE, SOURCE_KEYCHAIN, KeyStore, mask
from chatclient.paths import resolve_paths
from chatclient.providers.registry import ProviderRegistry
from chatclient.ui.repl import Repl

OR = "https://openrouter.ai/api/v1"
GOOD = "sk-or-v1-" + "a" * 64


def pc(**kw):
    return ProviderConfig(type="openai_compat", base_url=OR, api_key_env="OPENROUTER_API_KEY", **kw)


def no_keychain(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("No recommended backend was available")

    monkeypatch.setattr(keyring, "get_password", broken)
    monkeypatch.setattr(keyring, "set_password", broken)
    monkeypatch.setattr(keyring, "delete_password", broken)


# ── KeyStore ─────────────────────────────────────────────────────────────────


def test_lookup_order_env_keychain_file_config(tmp_path, monkeypatch):
    ks = KeyStore(tmp_path / "keys.json")
    cfg = pc(api_key="from-config")
    assert ks.lookup("openrouter", cfg) == ("from-config", SOURCE_CONFIG)
    ks._file_write({"openrouter": "from-file"})
    assert ks.lookup("openrouter", cfg) == ("from-file", SOURCE_FILE)
    keyring.set_password("chatclient", "openrouter", "from-keychain")
    assert ks.lookup("openrouter", cfg) == ("from-keychain", SOURCE_KEYCHAIN)
    monkeypatch.setenv("OPENROUTER_API_KEY", " from-env ")
    assert ks.lookup("openrouter", cfg) == ("from-env", SOURCE_ENV)


def test_save_prefers_keychain_and_cleans_file(tmp_path):
    ks = KeyStore(tmp_path / "keys.json")
    ks._file_write({"openrouter": "old"})
    assert ks.save("openrouter", GOOD) == SOURCE_KEYCHAIN
    assert keyring.get_password("chatclient", "openrouter") == GOOD
    assert not (tmp_path / "keys.json").exists()  # stale copy removed


def test_save_falls_back_to_private_file_without_keychain(tmp_path, monkeypatch):
    no_keychain(monkeypatch)
    path = tmp_path / "cfg" / "keys.json"
    ks = KeyStore(path)
    assert ks.save("openrouter", GOOD) == SOURCE_FILE
    assert ks.get("openrouter", pc()) == GOOD
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert ks.remove("openrouter") and ks.get("openrouter", pc()) is None
    assert not path.exists()


def test_save_without_keychain_or_file_raises(monkeypatch):
    no_keychain(monkeypatch)
    with pytest.raises(RuntimeError):
        KeyStore(None).save("x", "y")


def test_mask_never_reveals_key():
    assert mask(GOOD) == "sk-or-v1-…aaaa"
    assert mask("short") == "•••••"


# ── /key in the REPL ─────────────────────────────────────────────────────────


@pytest.fixture
def key_chat(store, tmp_path):
    config = Config(providers={"openrouter": pc(), "local": ProviderConfig(
        type="openai_compat", base_url="http://localhost:1234/v1")})  # fmt: skip
    registry = ProviderRegistry(config, KeyStore(tmp_path / "keys.json"))
    chat = ChatService(config, store, registry)
    chat.new_session(model="openrouter:some/model:free")
    return chat


async def run_keys(chat, tmp_path, keys):
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths, show_stats=False)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(keys)
        await repl.run([])
    await chat.registry.aclose()


@respx.mock
async def test_key_set_verifies_saves_and_applies(key_chat, tmp_path, capsys):
    route = respx.get(f"{OR}/key").mock(return_value=httpx.Response(200, json={"data": {}}))
    assert key_chat.registry.get("openrouter").api_key is None
    await run_keys(key_chat, tmp_path, f"/key set openrouter\r{GOOD}\r/key\r/quit\r")
    out = capsys.readouterr().out
    assert route.calls[0].request.headers["authorization"] == f"Bearer {GOOD}"
    assert "saved in the OS keychain" in out
    assert GOOD not in out and "sk-or-v1-…aaaa" in out  # list shows only a masked key
    assert "local" not in out.split("provider")[-1]  # local servers need no key
    assert key_chat.registry.get("openrouter").api_key == GOOD  # picked up without restart


@respx.mock
async def test_key_set_rejected_key_is_not_saved(key_chat, tmp_path, capsys):
    respx.get(f"{OR}/key").mock(
        return_value=httpx.Response(401, json={"error": {"message": "User not found."}})
    )
    await run_keys(key_chat, tmp_path, "/key set openrouter\rsk-or-v1-wrong\r/quit\r")
    assert "rejected this key; nothing was saved" in capsys.readouterr().out
    assert keyring.get_password("chatclient", "openrouter") is None


@respx.mock
async def test_key_set_falls_back_to_file_on_linux_without_keychain(
    key_chat, tmp_path, capsys, monkeypatch
):
    no_keychain(monkeypatch)
    respx.get(f"{OR}/key").mock(return_value=httpx.Response(200, json={"data": {}}))
    await run_keys(key_chat, tmp_path, f"/key openrouter\r{GOOD}\r/quit\r")
    assert "readable only by you" in capsys.readouterr().out
    assert key_chat.registry.get("openrouter").api_key == GOOD


async def test_env_override_is_reported(key_chat, tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-env")
    with respx.mock:
        respx.get(f"{OR}/key").mock(return_value=httpx.Response(200, json={"data": {}}))
        await run_keys(key_chat, tmp_path, f"/key set openrouter\r{GOOD}\r/quit\r")
    assert "takes priority" in capsys.readouterr().out


async def test_key_remove_and_free_without_key(key_chat, tmp_path, capsys):
    keyring.set_password("chatclient", "openrouter", GOOD)
    await run_keys(key_chat, tmp_path, "/key remove openrouter\r/free\r/quit\r")
    out = capsys.readouterr().out
    assert "Removed the saved key for openrouter" in out
    assert "No API key for openrouter. Add one with: /key set openrouter" in out


async def test_key_cancel_and_unknown_provider(key_chat, tmp_path, capsys):
    await run_keys(key_chat, tmp_path, "/key set openrouter\r\r/key set nope\r/key set\r/quit\r")
    out = capsys.readouterr().out
    assert "Aborted." in out
    assert "Unknown provider 'nope'" in out
    assert "Usage: /key set <provider>" in out


@respx.mock
async def test_401_during_chat_points_to_key_command(key_chat, tmp_path, capsys):
    key_chat.set_model("openrouter:paid/model")
    respx.post(f"{OR}/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "No auth credentials found"}})
    )
    await run_keys(key_chat, tmp_path, "merhaba\r/quit\r")
    assert "/key set openrouter" in capsys.readouterr().out
