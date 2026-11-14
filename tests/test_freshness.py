"""Repetition is measured, and a recycled reply is rewritten once."""

from __future__ import annotations

from tests.conftest import ALPHA

from chatclient.core import freshness
from chatclient.core.chat import TurnResult
from chatclient.providers.base import StreamEvent
from chatclient.providers.echo import EchoProvider

LOOP = (
    "My breath hitches, a sharp intake that feels too loud in the quiet room. I stare at "
    "your lips, the red mark on my paper blurring in my vision."
)


def test_repetition_is_measured_in_any_language():
    assert freshness.repetition(LOOP, [LOOP]) == 1.0
    assert freshness.repetition(LOOP, ["We walk down the empty hall together."]) < 0.1
    tr = "Nefesim kesiliyor, sessiz odada fazla yüksek gelen keskin bir soluk. Dudaklarına bakıyorum."
    assert freshness.is_repetitive(tr, ["Önce bir şey dedim. " + tr])
    assert freshness.repetition("Yes.", ["Yes."]) == 0.0  # too short to judge


def test_fresh_note_is_built_from_the_last_reply():
    note = freshness.fresh_note(LOOP, "Sam")
    assert '"My breath hitches, a sharp intake…"' in note and "Sam" in note
    assert freshness.fresh_note("", "Sam") == ""


async def say(chat, text):
    r = TurnResult()
    async for _ in chat.send(text, r):
        pass
    return r


def scripted(monkeypatch, replies):
    prompts: list[str] = []
    it = iter(replies)

    async def model(self, model, messages, params):
        prompts.append(messages[-1].content)
        yield StreamEvent(text=next(it))

    monkeypatch.setattr(EchoProvider, "stream_chat", model)
    return prompts


async def test_the_story_note_changes_every_turn(chat, monkeypatch):
    char = chat.store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=char.id)
    prompts = scripted(monkeypatch, [LOOP, "We walk out into the cold night air together."])
    await say(chat, "hello")
    await say(chat, "let's go")
    assert 'Your previous reply began "My breath hitches' in prompts[-1]


async def test_a_copied_reply_is_flagged(chat, monkeypatch):
    char = chat.store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=char.id)
    scripted(monkeypatch, [LOOP, LOOP])
    first = await say(chat, "i won't fail you")
    second = await say(chat, "don't worry")
    assert not first.repeated and second.repeated and second.repetition == 1.0


async def test_repl_rewrites_a_recycled_reply_once(chat, monkeypatch, tmp_path):
    from chatclient.i18n import Translator
    from chatclient.paths import resolve_paths
    from chatclient.ui.repl import Repl

    char = chat.store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=char.id)
    fresh = "Joy grabs her bag and we walk out into the cold night air together, laughing."
    prompts = scripted(monkeypatch, [LOOP, LOOP, fresh])
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths)
    await repl.reply(lambda r: chat.send("i won't fail you", r))
    await repl.reply(lambda r: chat.send("don't worry", r))
    assert chat.last_reply().content == fresh
    assert "Your draft repeated earlier replies" in prompts[-1]
    sibs, _ = chat.alternatives()
    assert [m.content for m in sibs] == [LOOP, fresh]  # the repeat is kept (/prev)


async def test_rewriting_can_be_turned_off(chat, monkeypatch, tmp_path):
    from chatclient.i18n import Translator
    from chatclient.paths import resolve_paths
    from chatclient.ui.repl import Repl

    chat.config.roleplay.rewrite_repeats = False
    char = chat.store.add_character("ai", ALPHA)
    chat.new_session(model="echo:echo", character_id=char.id)
    scripted(monkeypatch, [LOOP, LOOP])
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths)
    await repl.reply(lambda r: chat.send("a", r))
    await repl.reply(lambda r: chat.send("b", r))
    assert chat.last_reply().content == LOOP


def test_example_lines_are_capped():
    from chatclient.char_builder import clean_field

    many = "\n".join(f"<START>\n{{{{user}}}}: q{i}\n{{{{char}}}}: a{i}" for i in range(9))
    out = clean_field("mes_example", many)
    assert out.count("<START>") == 2 and "q1" in out and "q2" not in out
