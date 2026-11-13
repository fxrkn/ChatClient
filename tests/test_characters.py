import json

import pytest

from chatclient.characters import (
    Card,
    apply_macros,
    builtin_cards,
    compile_reminder,
    compile_system_prompt,
    format_examples,
)
from chatclient.core.chat import ChatError, TurnResult
from chatclient.core.context import build_context
from chatclient.providers.base import ChatMessage, GenParams
from chatclient.providers.echo import EchoProvider


async def drain(agen):
    return [ev async for ev in agen]


# ── cards ────────────────────────────────────────────────────────────────────


def test_card_reads_v3_v2_and_bare():
    v3 = {"spec": "chara_card_v3", "data": {"name": "A", "first_mes": "hi", "extensions": {"x": 1}}}
    v2 = {"spec": "chara_card_v2", "data": {"name": "B", "personality": "kind"}}
    bare = {"char_name": "C", "char_persona": "grumpy"}
    assert Card.from_json(v3).first_mes == "hi"
    assert Card.from_json(json.dumps(v2)).personality == "kind"
    c = Card.from_json(bare)
    assert (c.name, c.personality) == ("C", "grumpy")
    with pytest.raises(ValueError):
        Card.from_json({"data": {"description": "no name"}})


def test_card_roundtrip_keeps_unknown_fields():
    src = {"spec": "chara_card_v3", "data": {"name": "A", "creator": "me", "extensions": {"k": 2}}}
    out = Card.from_json(src).to_dict()
    assert out["spec"] == "chara_card_v3"
    assert out["data"]["creator"] == "me" and out["data"]["extensions"] == {"k": 2}


def test_macros():
    assert (
        apply_macros("{{char}} sees {{ User }} and <BOT>", "Beta", "Ali")
        == "Beta sees Ali and Beta"
    )


def test_no_characters_are_bundled_and_loader_copes():
    assert builtin_cards() == {}


def test_retired_builtins_are_removed_or_kept_if_used(chat, shipped):
    chat.seed_builtin_characters()
    alpha = chat.store.find_character("alpha")
    chat.new_session(model="echo:echo", character_id=alpha.id)  # Alpha is in use
    shipped.clear()  # a new version ships no characters
    chat.seed_builtin_characters()
    assert chat.store.find_character("beta") is None  # unused → removed
    kept = chat.store.find_character("alpha")
    assert kept is not None and kept.builtin_key is None  # used → now the user's own
    assert chat.require_session().character_id == kept.id


# ── prompt compilation ───────────────────────────────────────────────────────


def test_system_prompt_has_rules_card_persona_examples():
    card = Card(
        name="Beta",
        description="{{char}} runs a bookshop.",
        personality="warm",
        scenario="Rainy evening; {{user}} walks in.",
        mes_example="<START>\n{{user}}: hi\n{{char}}: hello",
    )
    persona = Card(name="Ali", description="A tired engineer.")
    text = compile_system_prompt(card, persona)
    assert "You are Beta" in text and "Ali" in text
    assert "Beta runs a bookshop." in text
    assert "Rainy evening; Ali walks in." in text
    assert "## About Ali\nA tired engineer." in text
    assert "Ali: hi\nBeta: hello" in text
    assert "{{" not in text  # all macros resolved


def test_card_system_prompt_replaces_rules_with_original_placeholder():
    card = Card(name="X", system_prompt="Custom intro.\n{{original}}")
    text = compile_system_prompt(card, None, rules="RULES for {{char}}")
    assert text.startswith("Custom intro.\nRULES for X")


def test_reminder_names_without_repeating_traits_or_post_history():
    card = Card(name="Alpha", personality="Gentle giant, slow to speak.")
    r = compile_reminder(card, Card(name="Mira"))
    assert "Alpha" in r and "Mira" in r
    assert "Gentle giant" not in r  # traits repeated every turn get parroted back
    card.post_history_instructions = "Keep {{char}} gruff with {{user}}."
    assert compile_reminder(card, None) == "Keep Alpha gruff with User."


def test_format_examples_splits_blocks():
    assert format_examples("<START>\na\n<START>\nb") == "a\n---\nb"


def test_reminder_goes_on_newest_user_turn_only(store):
    s = store.create_session(None)
    a = store.add_message(s.id, None, "user", "first")
    b = store.add_message(s.id, a.id, "assistant", "reply")
    c = store.add_message(s.id, b.id, "user", "second")
    ctx = build_context("sys", store.branch(c.id), 4000, reminder="stay in character")
    assert ctx.messages[1].content == "first"
    assert ctx.messages[-1].content.startswith("[Story note")
    assert ctx.messages[-1].content.endswith("second")
    # the stored message is untouched
    assert store.get_message(c.id).content == "second"


# ── chat service ─────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _ship_test_characters(request):
    if "test_no_characters_are_bundled" not in request.node.name:
        request.getfixturevalue("shipped")


def _alpha(chat):
    chat.seed_builtin_characters()
    return chat.store.find_character("alpha")


def test_seeding_is_idempotent_and_respects_deletion(chat):
    assert chat.seed_builtin_characters() >= 2
    assert chat.seed_builtin_characters() == 0
    beta = chat.store.find_character("beta")
    chat.delete_character(beta)
    assert chat.seed_builtin_characters() == 0  # deleted built-ins don't come back
    assert chat.store.find_character("beta") is None


def test_find_character_by_id_name_and_prefix(chat):
    alpha = _alpha(chat)
    assert chat.store.find_character(f"#{alpha.id}").name == "Alpha"
    assert chat.store.find_character("ALPHA").id == alpha.id
    assert chat.store.find_character("alp").id == alpha.id
    assert chat.store.find_character("nobody") is None


async def test_character_session_flow(chat, monkeypatch):
    alpha = _alpha(chat)
    chat.new_session(model="echo:echo", character_id=alpha.id)
    s = chat.require_session()
    assert s.title == "Alpha" and s.character_id == alpha.id
    greeting = chat.history()
    assert len(greeting) == 1 and greeting[0].role == "assistant"
    assert greeting[0].meta.get("greeting")
    with pytest.raises(ChatError):
        chat.undo()  # the opening line can't be undone

    seen: dict = {}

    async def spy(self, model, messages, params):
        seen["messages"] = list(messages)
        seen["params"] = params
        async for ev in original(self, model, messages, params):
            yield ev

    original = EchoProvider.stream_chat
    monkeypatch.setattr(EchoProvider, "stream_chat", spy)
    await drain(chat.send("Bir oda var mı?", TurnResult()))

    msgs: list[ChatMessage] = seen["messages"]
    assert msgs[0].role == "system" and "# Alpha" in msgs[0].content
    assert "Current date:" in msgs[0].content
    assert msgs[1].role == "assistant"  # greeting is part of the context
    assert msgs[-1].content.endswith("Bir oda var mı?") and "Story note" in msgs[-1].content
    assert seen["params"].stop == ["\nUser:"]
    # stored history stays clean
    assert chat.history()[-2].content == "Bir oda var mı?"


async def test_persona_changes_names_and_stop(chat):
    alpha = _alpha(chat)
    chat.set_persona(Card(name="Mira", description="A wandering herbalist."))
    chat.new_session(model="echo:echo", character_id=alpha.id)
    assert chat.persona().name == "Mira"
    assert chat.gen_params().stop == ["\nMira:"]
    assert "## About Mira" in chat.effective_system_prompt()
    chat.clear_persona()
    assert chat.persona() is None


def test_plain_session_unchanged(chat):
    assert chat.character() is None
    assert chat.reminder() is None
    assert chat.gen_params().stop is None
    assert chat.effective_system_prompt() == chat.config.system_prompt


def test_user_stop_setting_wins(chat):
    alpha = _alpha(chat)
    chat.new_session(model="echo:echo", character_id=alpha.id)
    chat.set_param("stop", "###")
    assert chat.gen_params().stop == ["###"]


def test_reminder_can_be_disabled(chat):
    alpha = _alpha(chat)
    chat.new_session(model="echo:echo", character_id=alpha.id)
    chat.config.roleplay.reminder = False
    assert chat.reminder() is None
    assert isinstance(chat.gen_params(), GenParams)
