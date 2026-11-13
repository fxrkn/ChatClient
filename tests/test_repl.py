"""Drive the REPL with scripted keystrokes (no real terminal needed)."""

from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from chatclient.i18n import Translator
from chatclient.paths import resolve_paths
from chatclient.ui.repl import Repl


async def run_script(chat, tmp_path, keys: str, lang: str = "en") -> Repl:
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator(lang), paths, show_stats=True)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(keys)
        await repl.run([])
    return repl


async def test_repl_chat_and_commands(chat, tmp_path, capsys):
    keys = (
        "merhaba dünya\r"
        "/set temperature 0.4\r"
        "/regen\r"
        "/sessions\r"
        "/search dünya\r"
        "/new ikinci\r"
        "/undo\r"
        "/nonsense\r"
        "/quit\r"
    )
    await run_script(chat, tmp_path, keys)
    out = capsys.readouterr().out
    assert "merhaba dünya" in out  # echo reply streamed
    assert "temperature = 0.4" in out
    assert "Unknown command" in out
    assert "nothing to undo" in out  # new session is empty
    sessions = chat.store.list_sessions()
    assert len(sessions) == 2
    first = next(s for s in sessions if s.session.title == "merhaba dünya")
    assert first.message_count == 3  # user + reply + regenerated reply (branch kept)
    assert first.session.params == {"temperature": 0.4}


async def test_repl_character_flow(chat, tmp_path, capsys, shipped):
    chat.seed_builtin_characters()
    keys = (
        "/char list\r"
        "/char use rowan\r"
        "merhaba\r"
        "/persona set\r"
        "Mira\r"
        "Gezgin bir şifacı.\r"
        "/char new\r"
        "Deniz\r"  # name
        "A fisherman from Bodrum, 40\r"  # concept
        "1,9\r"  # temperament: warm, calm
        "1\r"  # speech: short and dry
        "my own words\r"  # values: free text
        "\r"  # fear: skipped
        "1\r"  # relationship: strangers
        "1\r"  # genre: modern day
        "On the pier at dawn\r"  # place
        "Deniz bugün huysuz.\r"  # example line
        "n\r"  # no AI help
        "*Ağları topluyor.* Erkencisin.\r"  # first message typed by hand
        "\r"  # start now? (default yes)
        "/char\r"
        "/quit\r"
    )
    await run_script(chat, tmp_path, keys)
    out = capsys.readouterr().out
    assert "Rowan" in out and "Elif" in out  # list
    assert "Shut that door" in out  # Rowan's greeting shown
    deniz = chat.store.find_character("deniz")
    assert deniz is not None
    p = deniz.card.personality
    assert "Temperament: warm and sincere; calm and unhurried." in p
    assert "Speech: speaks in short, dry sentences" in p
    assert "Values: my own words." in p and "fear" not in p
    assert deniz.card.scenario == (
        "{{char}} and {{user}}: they are meeting for the first time. Scene: On the pier at dawn."
    )
    assert deniz.card.mes_example == "<START>\n{{char}}: Deniz bugün huysuz."
    s = chat.require_session()
    assert s.character_id == deniz.id  # started chatting with the new character
    assert chat.persona().name == "Mira"
    assert chat.history()[0].content.startswith("*Ağları topluyor.*")
    assert "Character: Deniz" in out


async def test_repl_wizard_ai_fill_and_export_import(chat, tmp_path, capsys, monkeypatch):
    from chatclient.providers.base import StreamEvent
    from chatclient.providers.echo import EchoProvider

    async def writer(self, model, messages, params):
        task = messages[-1].content
        text = (
            "<START>\n{{user}}: hi\n{{char}}: *nods* hey" if "<START>" in task else "*Waves.* Hi!"
        )
        yield StreamEvent(text=text)

    monkeypatch.setattr(EchoProvider, "stream_chat", writer)
    keys = (
        "/char new\r"
        "Mo\r"
        "a baker\r"
        "\r"
        "\r"
        "\r"
        "\r"
        "\r"
        "\r"
        "\r"
        "\r"  # mostly skipped
        "y\r"  # AI help
        "y\r"  # accept opening scene
        "r\r"  # retry example lines
        "y\r"  # accept the retry
        "n\r"  # don't start now
        "/char export mo\r"
        "/quit\r"
    )
    await run_script(chat, tmp_path, keys)
    mo = chat.store.find_character("mo")
    assert mo.card.first_mes == "*Waves.* Hi!"
    assert mo.card.mes_example.startswith("<START>\n{{user}}: hi")
    exported = tmp_path / "exports" / "Mo.json"
    assert exported.is_file()
    capsys.readouterr()
    await run_script(chat, tmp_path, f'/char import "{exported}"\r/quit\r')
    assert "Imported Mo" in capsys.readouterr().out
    assert len([c for c in chat.store.list_characters() if c.name == "Mo"]) == 2


async def test_repl_theme_command_persists(chat, tmp_path, capsys):
    await run_script(chat, tmp_path, "/theme\r/theme warm\r/theme neon\r/quit\r")
    out = capsys.readouterr().out
    assert "soft" in out and "rose" in out  # list with samples
    assert "Theme: warm" in out and "Unknown theme 'neon'" in out
    assert chat.store.kv_get("ui_theme") == "warm"
    # a new REPL picks the saved theme up
    paths = resolve_paths(tmp_path)
    assert Repl(chat, Translator("en"), paths).theme.name == "warm"
    await run_script(chat, tmp_path, "/theme reset\r/quit\r")
    assert Repl(chat, Translator("en"), paths).theme.name == chat.config.ui.theme


async def test_repl_turkish_ui(chat, tmp_path, capsys):
    await run_script(chat, tmp_path, "/help\r/quit\r", lang="tr")
    assert "Komutlar" in capsys.readouterr().out


async def test_repl_multiline_with_alt_enter(chat, tmp_path):
    await run_script(chat, tmp_path, "satır bir\x1b\rsatır iki\r/quit\r")
    user = chat.history()[0]
    assert user.content == "satır bir\nsatır iki"
