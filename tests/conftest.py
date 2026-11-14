from __future__ import annotations

import pytest

from chatclient.characters import Card
from chatclient.config import Config, ProviderConfig
from chatclient.core.chat import ChatService
from chatclient.providers.echo import EchoProvider
from chatclient.providers.registry import ProviderRegistry
from chatclient.storage import Store

# Neutral, test-only cards (the app itself ships no characters).
ALPHA = Card(
    name="Alpha",
    description="{{char}} is a test character.",
    personality="Calm test personality.",
    scenario="A test scene with {{user}}.",
    first_mes="*Test action.* Test greeting line.",
    mes_example="<START>\n{{user}}: test question\n{{char}}: test answer",
    tags=["test"],
)
BETA = Card(
    name="Beta",
    description="{{char}} ikinci bir test karakteri.",
    personality="Test kişiliği.",
    first_mes="*Test eylemi.* Test selamı.",
    tags=["test"],
)


@pytest.fixture
def real_panel_possible():
    """The real Repl.panel_possible (the autouse fixture below replaces it)."""
    from chatclient.ui.repl import Repl

    return Repl.panel_possible


@pytest.fixture(autouse=True)
def terminal_for_panel(monkeypatch, real_panel_possible):
    """Tests run without a terminal; the panel layout is allowed anyway."""
    from chatclient.ui.repl import Repl

    monkeypatch.setattr(Repl, "panel_possible", lambda self: True)


@pytest.fixture(autouse=True)
def fake_keychain(monkeypatch):
    """Tests never touch the real OS keychain: an in-memory one replaces it."""
    import keyring

    vault: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(keyring, "get_password", lambda s, u: vault.get((s, u)))
    monkeypatch.setattr(keyring, "set_password", lambda s, u, p: vault.__setitem__((s, u), p))
    monkeypatch.setattr(keyring, "delete_password", lambda s, u: vault.pop((s, u)))
    for var in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return vault


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def config():
    return Config(
        providers={
            "local": ProviderConfig(type="openai_compat", base_url="http://localhost:1234/v1"),
        }
    )


@pytest.fixture
def chat(config, store, monkeypatch):
    monkeypatch.setattr(EchoProvider, "delay", 0.0)
    registry = ProviderRegistry(config)
    service = ChatService(config, store, registry)
    service.new_session(model="echo:echo")
    return service


@pytest.fixture
def shipped(monkeypatch):
    """Pretend the app ships these characters (none are bundled for real)."""
    cards = {"alpha": ALPHA, "beta": BETA}
    monkeypatch.setattr("chatclient.core.chat.builtin_cards", lambda: dict(cards))
    return cards
