"""The director: gaps around the brief's private direction (leaks, stale notes, regen,
continue, retries, no character, failing brief). The basics live in test_memory.py."""

import json

import pytest

from chatclient.characters import MATURE_RULES, Card, apply_macros
from chatclient.config import Config, MemoryConfig, ProviderConfig
from chatclient.core import freshness
from chatclient.core.chat import ChatService, TurnResult
from chatclient.i18n import Translator
from chatclient.memory.engine import MemoryEngine
from chatclient.memory.tasks import BRIEF_SCHEMA, parse_brief
from chatclient.providers.echo import EchoProvider
from chatclient.providers.registry import ProviderRegistry
from test_memory import drain, fake_model

SECRET = "Ada is secretly worried about the map."


@pytest.fixture
def mem(store, monkeypatch):
    calls: list[str] = []
    briefs: list[dict] = []
    monkeypatch.setattr(EchoProvider, "stream_chat", fake_model(calls, briefs))
    config = Config(
        memory=MemoryConfig(every_exchanges=2, keep_recent_messages=2, summarize_after_tokens=200),
        providers={"local": ProviderConfig(type="openai_compat", base_url="http://localhost:1/v1")},
    )
    registry = ProviderRegistry(config)
    engine = MemoryEngine(config, store, registry)
    chat = ChatService(config, store, registry, engine)
    char = store.add_character("ai", Card(name="Ada", personality="Curious."))
    chat.new_session(model="echo:echo", character_id=char.id)
    chat.set_persona(Card(name="Ali"))
    return chat, engine, calls, briefs


def test_direction_is_required_by_the_schema_but_optional_when_parsing():
    assert "direction" in BRIEF_SCHEMA["required"]
    # a model (or fallback path) that leaves it out must not break the brief
    b = parse_brief(json.dumps({"about_past": False, "relevant": [], "note": ""}))
    assert b is not None and b.direction == ""


async def test_no_character_means_no_direction_even_if_the_model_writes_one(mem):
    chat, _engine, calls, briefs = mem
    chat.new_session(model="echo:echo")  # a plain chat
    briefs.append({"about_past": False, "direction": SECRET})
    r = TurnResult()
    await drain(chat.send("Hello", r))
    assert r.text and r.direction == "" and chat.last_direction == ""
    assert "Private direction" not in r.context.messages[-1].content
    brief = [c for c in calls if "memory brief" in c]
    assert not brief or "- direction: leave empty." in brief[0]


async def test_a_failed_brief_clears_the_direction_instead_of_keeping_the_old_one(mem):
    chat, engine, _calls, briefs = mem
    briefs.append({"about_past": False, "direction": SECRET})
    await drain(chat.send("One", TurnResult()))
    assert chat.last_direction == SECRET

    async def boom(*_a, **_k):
        raise TimeoutError

    engine._complete = boom  # the brief times out
    r = TurnResult()
    await drain(chat.send("Two", r))
    assert r.text and r.direction == "" and chat.last_direction == ""
    assert SECRET not in r.context.messages[-1].content  # not carried into the next turn


async def test_regen_and_edit_get_their_own_direction(mem):
    chat, _engine, _calls, briefs = mem
    briefs += [
        {"about_past": False, "direction": "first take"},
        {"about_past": False, "direction": "second take"},
        {"about_past": False, "direction": "third take"},
    ]
    await drain(chat.send("Hi", TurnResult()))
    r = TurnResult()
    await drain(chat.regenerate(r))
    assert r.direction == chat.last_direction == "second take"
    last = r.context.messages[-1].content
    assert "second take" in last and "first take" not in last
    r = TurnResult()
    await drain(chat.edit_and_resend("Hi there", r))
    assert r.direction == chat.last_direction == "third take"
    assert "third take" in r.context.messages[-1].content


async def test_continue_keeps_last_direction_for_the_command(mem):
    chat, _engine, calls, briefs = mem
    briefs.append({"about_past": False, "direction": "keep it short"})
    await drain(chat.send("Hi", TurnResult()))
    before = len([c for c in calls if "memory brief" in c])
    r = TurnResult()
    await drain(chat.continue_reply(r))
    assert len([c for c in calls if "memory brief" in c]) == before  # no new brief
    assert chat.last_direction == "keep it short"
    assert "keep it short" in calls[-1] and r.direction == "keep it short"  # same reply: kept
    # a reply from another turn does not inherit it
    briefs.append({"about_past": False, "direction": ""})
    await drain(chat.send("Again", TurnResult()))
    await drain(chat.continue_reply(r2 := TurnResult()))
    assert "keep it short" not in calls[-1] and r2.direction == ""


async def test_a_repeat_rewrite_still_carries_the_rewrite_note_and_a_direction(mem):
    chat, _engine, _calls, briefs = mem
    briefs += [{"about_past": False, "direction": "d1"}, {"about_past": False, "direction": "d2"}]
    first = TurnResult()
    await drain(chat.send("Hi", first))
    r = TurnResult()
    await drain(chat.regenerate(r, fresh=True))
    last = r.context.messages[-1].content
    assert freshness.rewrite_note(first.text, chat.user_name()) in last
    assert "d2" in last and "d1" not in last


async def test_the_direction_never_reaches_storage_or_later_prompts(mem):
    chat, engine, calls, briefs = mem
    for i in range(3):
        briefs.append({"about_past": False, "direction": f"{SECRET} {i}"})
        await drain(chat.send(f"Message {i}", TurnResult()))
    await engine.run_all()
    store = chat.store
    texts = [m.content for m in store.branch(chat.refresh().head_message_id, limit=50)]
    assert not any("worried about the map" in t or "Private direction" in t for t in texts)
    for table in ("memories", "kv", "jobs"):
        rows = store.conn.execute(f"SELECT * FROM {table}").fetchall()
        assert "worried about the map" not in json.dumps([tuple(map(str, r)) for r in rows])
    # memory passes (extraction, summaries) are not shown the direction
    chat_calls = [c for c in calls if "memory brief" not in c and "Return ONLY" in c]
    assert not any("worried about the map" in c for c in chat_calls)


async def test_a_private_chat_stores_nothing_even_with_a_failing_brief(mem):
    chat, engine, _calls, _briefs = mem
    chat.set_private(True)
    engine.cfg.brief = False
    r = TurnResult()
    await drain(chat.send("Hello", r))
    assert r.text and r.direction == ""
    assert engine.repo.job_stats() == {}


async def test_mature_rules_stay_in_the_system_prompt_and_reach_the_brief(mem):
    chat, _engine, calls, briefs = mem
    chat.set_content_mode("mature")
    briefs.append({"about_past": False, "direction": "x"})
    r = TurnResult()
    await drain(chat.send("Hi", r))
    system = r.context.messages[0].content
    assert apply_macros(MATURE_RULES, "Ada", "Ali") in system
    assert "Content setting: mature" in next(c for c in calls if "memory brief" in c)


def test_direction_command_text_in_turkish():
    assert "yönlendirme" in Translator("tr")("direction_none")
    assert "No direction yet" in Translator("en")("direction_none")


async def test_continue_after_regen_and_prev_next_uses_only_its_own_reply_direction(mem):
    chat, _engine, calls, briefs = mem
    briefs += [
        {"about_past": False, "direction": "dir-one"},
        {"about_past": False, "direction": "dir-two"},
    ]
    await drain(chat.send("Hi", TurnResult()))
    await drain(chat.regenerate(TurnResult()))
    assert chat.current_direction() == "dir-two"
    chat.switch_alternative(-1)  # back to the first version
    await drain(chat.continue_reply(r := TurnResult()))
    assert "dir-one" not in calls[-1] and "dir-two" not in calls[-1] and r.direction == ""
    chat.switch_alternative(+1)  # the second version again
    await drain(chat.continue_reply(r := TurnResult()))
    assert "dir-two" in calls[-1] and r.direction == "dir-two"


async def test_a_direction_does_not_survive_a_switch_or_a_new_chat(mem):
    chat, _engine, calls, briefs = mem
    chat.set_private(True)
    briefs.append({"about_past": False, "direction": "private-note"})
    await drain(chat.send("Hi", TurnResult()))
    private_id = chat.session.id
    assert chat.current_direction() == "private-note"
    char = chat.character()
    chat.new_session(model="echo:echo", character_id=char.id)
    assert chat.current_direction() == ""
    chat.open_session(private_id)
    assert chat.current_direction() == "private-note"  # its own chat
    other = chat.new_session(model="echo:echo", character_id=char.id)
    await drain(chat.send("Hello", TurnResult()))
    await drain(chat.continue_reply(r := TurnResult()))
    assert "private-note" not in calls[-1] and r.direction == "" and other.id != private_id


def _failing(engine):
    from chatclient.providers.base import ProviderError

    async def boom(*_a, **_k):
        raise ProviderError("quota-detail-xyz")

    engine._complete = boom


async def test_keystroke_prefetch_does_not_trip_the_failure_threshold(mem):
    chat, engine, _calls, _briefs = mem
    _failing(engine)
    for draft in ("hel", "hello"):
        await engine.prefetch(chat.refresh(), draft, [], set(), "standard")
    assert not engine.brief_struggling


async def test_a_failed_prefetch_is_not_retried_by_the_send(mem):
    chat, engine, _calls, _briefs = mem
    _failing(engine)
    await engine.prefetch(chat.refresh(), "hello", [], set(), "standard")
    # a draft only records the reason; the send counts the failure (once)
    assert engine.brief_failures == 0 and "quota-detail-xyz" in engine.last_brief_error
    await drain(chat.send("hello", TurnResult()))  # reuses the prefetch: no second attempt
    assert engine.brief_failures == 1


async def test_a_private_chat_failure_reason_is_not_stored_or_made_of_the_text(mem):
    chat, engine, _calls, _briefs = mem
    chat.set_private(True)
    _failing(engine)
    await drain(chat.send("my secret words", TurnResult()))
    assert "secret" not in engine.last_brief_error
    store = chat.store
    for table in ("memories", "kv", "jobs"):
        rows = store.conn.execute(f"SELECT * FROM {table}").fetchall()
        assert "quota-detail-xyz" not in json.dumps([tuple(map(str, r)) for r in rows])


async def test_a_good_brief_leaves_no_error(mem):
    chat, engine, _calls, briefs = mem
    briefs.append({"about_past": False, "direction": ""})
    await drain(chat.send("Hi", TurnResult()))
    assert engine.brief_failures == 0 and engine.last_brief_error == ""


def test_every_new_director_string_has_turkish():
    from chatclient.i18n import STRINGS

    for key in STRINGS:
        if key.startswith(("director_", "direction_")):
            assert STRINGS[key].get("tr"), key


async def test_a_brief_error_stays_in_the_chat_it_happened_in(mem, tmp_path, capsys):
    import io

    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from chatclient.paths import resolve_paths
    from chatclient.serve import Server
    from chatclient.ui.repl import Repl

    chat, engine, _calls, _briefs = mem
    chat.set_private(True)
    _failing(engine)
    r = TurnResult()
    await drain(chat.send("hello", r))
    a_id = chat.session.id
    assert "quota-detail-xyz" in engine.brief_error_for(a_id)
    assert Server(chat, io.BytesIO()).turn_info(r)["direction_error"]  # shown in A itself
    chat.new_session(model="echo:echo", character_id=chat.character().id)  # chat B
    assert engine.brief_error_for(chat.session.id) == ""
    assert Server(chat, io.BytesIO()).turn_info(TurnResult())["direction_error"] in ("", None)
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths, show_stats=False)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text("/direction\r/quit\r")
        await repl.run([])
    assert "quota-detail-xyz" not in capsys.readouterr().out
