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
        "/char use alpha\r"
        "merhaba\r"
        "/persona set\r"
        "Mira\r"
        "Gezgin bir şifacı.\r"
        "/char new\r"
        "Deniz\r"  # name
        "\r"  # content: skipped = standard
        "\r"  # age: skipped (fine for a standard character)
        "\r"  # gender: skipped
        "A fisherman from Bodrum, 40\r"  # concept
        "\r"  # appearance: skipped
        "1,9\r"  # temperament: warm, calm
        "1\r"  # speech: short and dry
        "my own words\r"  # values: free text
        "\r"  # fear: skipped
        "\r"  # likes: skipped
        "\r"  # background: skipped
        "1\r"  # relationship: strangers
        "\r"  # history: skipped
        "\r"  # who leads: skipped (balanced)
        "1\r"  # genre: modern day
        "On the pier at dawn\r"  # place
        "Deniz bugün huysuz.\r"  # example line
        "n\r"  # no AI help
        "*Ağları topluyor.* Erkencisin.\r"  # first message typed by hand
        "\r"  # save? (default yes)
        "\r"  # start now? (default yes)
        "/char\r"
        "/quit\r"
    )
    await run_script(chat, tmp_path, keys)
    out = capsys.readouterr().out
    assert "Alpha" in out and "Beta" in out  # list
    assert "Test greeting line" in out  # Alpha's greeting shown
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


SKIP_13 = "\r" * 13


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
        "\r\r\r"  # content, age, gender skipped
        "a baker\r"
        f"{SKIP_13}"  # appearance … example line: all skipped
        "y\r"  # AI help
        "y\r"  # accept opening scene
        "r\r"  # retry example lines
        "y\r"  # accept the retry
        "y\r"  # save
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


async def test_character_wizard_mature_needs_an_adult_age(chat, tmp_path, capsys):
    from chatclient.characters import card_content

    answers = [
        "/char new", "Kai", "2",   # name, content: mature
        "16", "31",                 # under 18 is asked again
        "1", "runs a bar", "tall",  # gender, concept, appearance
        "6", "1", "1", "", "jazz", "",  # temperament, speech, values, fear, likes, background
        "2", "we kissed once", "3", "6", "his flat",  # relationship, history, lead, genre, place
        "1,2", "",                  # themes, example line
        "n", "Hi.", "y", "n",       # no AI help, opening line, save, don't start
        "/quit",
    ]  # fmt: skip
    await run_script(chat, tmp_path, "\r".join(answers) + "\r")
    out = capsys.readouterr().out
    assert "18 or older" in out
    kai = chat.store.find_character("Kai")
    assert kai is not None and card_content(kai.card) == "mature"
    assert kai.card.description.startswith("{{char}} is a 31-year-old man.")
    assert "old friends who go back years. we kissed once." in kai.card.scenario
    from chatclient.characters import card_initiative

    assert card_initiative(chat.store.find_character("Kai").card) == "lead"
