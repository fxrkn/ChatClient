"""Long-term memory: parsing model output, storage, jobs, summaries, the prompt."""

import json
import re
from datetime import timedelta

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from chatclient.characters import Card
from chatclient.config import Config, MemoryConfig, ProviderConfig
from chatclient.core.chat import ChatService, TurnResult
from chatclient.i18n import Translator
from chatclient.memory.engine import MemoryEngine
from chatclient.memory.repo import MemoryRepo, Scope, normalize_key, similarity, trigram_query
from chatclient.memory.tasks import (
    BRIEF_SCHEMA,
    brief_prompt,
    parse_brief,
    parse_json,
    parse_pass,
)
from chatclient.paths import resolve_paths
from chatclient.providers.base import StreamEvent
from chatclient.providers.echo import EchoProvider
from chatclient.providers.registry import ProviderRegistry
from chatclient.ui.repl import Repl

ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ── forgiving parsing of model output ────────────────────────────────────────


def test_parse_pass_tolerates_fences_trailing_commas_and_loose_values():
    raw = """Sure! Here you go:
```json
{"memories": [
   {"content": "Ali Bodrum'da balıkçı.", "kind": "fact", "importance": "high", "people": "Ali, Ayşe",},
   {"content": "Ali'nin kedisi Pamuk.", "importance": 4, "fact": {"key": "Ali.cat.name", "value": "Pamuk"}},
   "Ali kahveyi şekersiz içer.",
   {"content": "   "},
 ],
 "scene": {"location": "Liman", "mood": "huzurlu", "threads": "haritanın sırrı", "time": "unknown"},
}
```"""
    res = parse_pass(raw)
    assert [m.content for m in res.memories] == [
        "Ali Bodrum'da balıkçı.", "Ali'nin kedisi Pamuk.", "Ali kahveyi şekersiz içer.",
    ]  # fmt: skip
    assert res.memories[0].importance == 0.8 and res.memories[0].people == ["Ali", "Ayşe"]
    assert (res.memories[1].key, res.memories[1].value) == ("Ali.cat.name", "Pamuk")
    assert res.scene == {"location": "Liman", "mood": "huzurlu", "threads": ["haritanın sırrı"]}
    assert parse_json("no json here") is None
    assert parse_pass("[1, 2]") is None


def test_parse_brief_is_forgiving():
    b = parse_brief("""{"about_past": true, "dates": ["2026-11-12", {"from": "2026-11-09", "to": "bad"}],
        "relevant": "M3, Q7", "claims": [{"claim": "we ate tacos", "status": "Maybe"}, "x"],
        "note": null}""")  # fmt: skip
    assert b.about_past and b.dates == [("2026-11-12", "2026-11-12")]
    assert b.relevant == ["M3", "Q7"]
    assert b.claims[0].status == "unsupported" and b.claims[1].claim == "x"
    assert b.note == "" and b.direction == ""  # direction is optional
    assert parse_brief("nothing") is None
    d = parse_brief('{"about_past": false, "direction": "  Short banter; Ada teases.  "}')
    assert d.direction == "Short banter; Ada teases."
    assert parse_brief('{"direction": null}').direction == ""
    assert "direction" in BRIEF_SCHEMA["properties"]


def test_brief_prompt_carries_the_character_view_and_the_initiative_hint():
    p = brief_prompt(
        "Ada", "Ali", "today", [], "hi", [], profile="A curious sailor.",
        initiative="Ada tends to lead.", scene="location: harbour", content="mature",
    )  # fmt: skip
    text = p[-1].content
    assert "A curious sailor." in text and "Ada tends to lead." in text
    assert "location: harbour" in text and "Content setting: mature" in text
    assert "- direction: a private note" in text and "not a rule" in text
    bare = brief_prompt("Ada", "Ali", "today", [], "hi", [])[-1].content
    assert "- direction: leave empty." in bare and "Initiative hint" not in bare


def test_text_helpers():
    assert similarity("Ali Bodrum'da balıkçıdır", "ali bodrumda balikcidir") == 1.0
    assert similarity("Ali balıkçı", "Ayşe öğretmen") == 0.0
    assert normalize_key("Ali.Cat_Name ") == "ali cat name"
    assert normalize_key("  ") is None
    q = trigram_query("Haritayı")
    assert '"har"' in q and '"ita"' in q and " OR " in q


# ── storage ──────────────────────────────────────────────────────────────────


def test_fuzzy_search_finds_suffixed_words_without_rules(store):
    s = store.create_session(None)
    repo = MemoryRepo(store)
    scope = Scope(s.id)
    harita = repo.add_memory(scope, "Deniz fenerinde mühürlü bir harita bulundu.",
                             place="Deniz Feneri", people=["Ayşe"], day="2026-11-12")  # fmt: skip
    repo.add_memory(scope, "Ali çayı şekersiz içer.", kind="preference")
    assert repo.keyword_search(scope, "haritayı nerede bulmuştuk?")[0] == harita.id
    assert repo.keyword_search(scope, "fenerindeki")[0] == harita.id
    ents = {name: eid for eid, name, _n, _k in repo.entities(scope)}
    assert set(ents) == {"Deniz Feneri", "Ayşe"}
    assert repo.memories_for_entities([ents["Ayşe"]]) == [harita.id]
    assert repo.memories_on_days(scope, ["2026-11-12"]) == [harita.id]
    assert repo.find_duplicate(scope, "Deniz fenerinde mühürlü harita bulundu") is not None
    assert repo.set_pinned(harita.id, True) and repo.pinned_ids(scope) == [harita.id]
    assert repo.delete_memory(harita.id) and harita.id not in repo.keyword_search(scope, "harita")


def test_character_scope_spans_sessions(store):
    char = store.add_character("ai", Card(name="Ada"))
    s1, s2 = store.create_session(None), store.create_session(None)
    repo = MemoryRepo(store)
    repo.add_memory(Scope(s1.id, char.id), "Kullanıcının kedisinin adı Pamuk.")
    assert repo.keyword_search(Scope(s2.id, char.id), "kedimin adı neydi") != []
    assert repo.keyword_search(Scope(s2.id), "kedimin adı neydi") == []  # session scope: no


def test_deleting_session_or_character_cleans_memory_index(store):
    char = store.add_character("ai", Card(name="Ada"))
    s = store.create_session(None)
    repo = MemoryRepo(store)
    repo.add_memory(Scope(s.id), "silinecek oturum anısı")
    repo.add_memory(Scope(s.id, char.id), "silinecek karakter anısı")
    store.delete_character(char.id)
    store.delete_session(s.id)
    assert store.conn.execute("SELECT COUNT(*) FROM memories_fts").fetchone()[0] == 0
    assert store.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0] == 0


def test_job_queue(store):
    s = store.create_session(None)
    repo = MemoryRepo(store)
    assert repo.enqueue(s.id, "memory_pass") and not repo.enqueue(s.id, "memory_pass")  # deduped
    assert repo.enqueue(s.id, "day", {"day": "2026-11-12"})
    assert repo.enqueue(None, "index")
    assert repo.next_job().kind == "index"  # cheapest first
    job = repo.next_job()
    assert job.kind == "memory_pass"
    repo.release_job(job.id)
    assert repo.next_job().id == job.id  # back in the queue
    repo.fail_job(job.id, "boom", retry_after="9999-01-01T00:00:00+00:00", give_up=False)
    assert repo.next_job().kind == "day"  # the failed one waits for its retry time
    repo.reset_stale_jobs()
    assert repo.job_stats()["pending"] == 3


def test_scene_update_does_not_reorder_sessions(store):
    a = store.create_session(None)
    store.create_session(None)
    before = [x.session.id for x in store.list_sessions()]
    MemoryRepo(store).set_scene(a.id, {"location": "Liman"})
    assert [x.session.id for x in store.list_sessions()] == before
    assert store.get_session(a.id).scene == {"location": "Liman"}


# ── the engine end to end (a fake "utility model") ───────────────────────────

PASS_JSON = {
    "memories": [
        {"content": "Ali deniz fenerinde mühürlü bir harita buldu.", "kind": "event",
         "importance": 5, "people": ["Ali"], "place": "Deniz Feneri"},
        {"content": "Ali çayını şekersiz içer.", "kind": "preference", "importance": 3},
    ],
    "scene": {"location": "Eski liman", "mood": "meraklı", "threads": ["haritanın sırrı"]},
}  # fmt: skip


def fake_model(calls, briefs):
    async def stream(self, model, messages, params):
        prompt = messages[-1].content
        calls.append("\n".join(m.content for m in messages))
        if "Return ONLY a JSON object" in prompt and "memory brief" in prompt:
            text = json.dumps(briefs.pop(0) if briefs else {"about_past": False})
        elif "Return ONLY a JSON object" in prompt:
            text = json.dumps(PASS_JSON, ensure_ascii=False)
        elif "story so far" in prompt:
            text = "SYNOPSIS: Ali ve Ada limanda tanıştı."
        elif "Summarise this part" in prompt:
            text = "SEGMENT: Ali haritadan bahsetti."
        elif "talked about and did on" in prompt:
            text = "DAY: Ali ve Ada deniz fenerine gitti."
        elif "Which calendar days" in prompt:
            text = json.dumps({"dates": [{"from": "2026-11-12", "to": "2026-11-12"}]})
        else:
            text = "Tamam."  # an ordinary chat reply
        yield StreamEvent(text=text)

    return stream


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
    char = store.add_character("ai", Card(name="Ada", personality="Meraklı."))
    chat.new_session(model="echo:echo", character_id=char.id)
    chat.set_persona(Card(name="Ali"))
    return chat, engine, calls, briefs


async def drain(agen):
    return [ev async for ev in agen]


async def test_memory_pass_extracts_memories_entities_and_scene(mem):
    chat, engine, _calls, _briefs = mem
    await drain(chat.send("Deniz fenerinde bir harita bulduk!", TurnResult()))
    assert engine.repo.job_stats() == {}  # one reply: not due yet (every_exchanges=2)
    await drain(chat.send("Çayımı şekersiz içerim bu arada.", TurnResult()))
    assert engine.repo.job_stats().get("pending") == 1
    assert await engine.run_all() >= 1
    scope = engine.scope(chat.require_session())
    contents = [m.content for m in engine.repo.list_memories(scope)]
    assert "Ali deniz fenerinde mühürlü bir harita buldu." in contents
    assert chat.refresh().scene["location"] == "Eski liman"
    await drain(chat.send("bir", TurnResult()))
    await drain(chat.send("iki", TurnResult()))
    await engine.run_all()
    assert len(engine.repo.list_memories(scope)) == 2  # no duplicates


async def test_brief_selects_memories_and_resolves_dates(mem):
    chat, engine, calls, briefs = mem
    scope = engine.scope(chat.require_session())
    yesterday = (engine.today() - timedelta(days=1)).isoformat()
    harita = engine.repo.add_memory(scope, "Ali deniz fenerinde mühürlü bir harita buldu.",
                                    place="Deniz Feneri", day=yesterday, importance=0.9)  # fmt: skip
    engine.repo.add_memory(scope, "Ali'nin kedisinin adı Pamuk.", day="2026-01-01")
    engine.repo.set_scene(chat.session.id, {"location": "Eski liman", "mood": "meraklı"})
    briefs.append({"about_past": True, "dates": [{"from": yesterday, "to": yesterday}],
                   "relevant": [f"M{harita.id}"], "claims": []})  # fmt: skip
    r = TurnResult()
    await drain(chat.send("Dün deniz fenerinde ne bulmuştuk?", r))
    brief_prompt = next(c for c in calls if "memory brief" in c)
    assert f"[M{harita.id}]" in brief_prompt  # candidates found by search, shown to the model
    prompt = calls[-1]
    assert "Current scene — location: Eski liman" in prompt
    assert "meraklı" not in prompt  # mood is not repeated every turn (it steers small models)
    assert f"({yesterday} · Deniz Feneri) Ali deniz fenerinde mühürlü bir harita buldu." in prompt
    assert "Pamuk" not in prompt  # the brief didn't pick it
    assert r.recalled == 1
    assert chat.last_user_message().content == "Dün deniz fenerinde ne bulmuştuk?"  # clean


async def test_the_direction_rides_next_to_the_newest_message(mem):
    from chatclient.characters import set_card_initiative

    chat, _engine, calls, briefs = mem
    char = chat.character()
    set_card_initiative(char.card, "lead")
    chat.store.update_character(char.id, char.card)
    briefs.append({"about_past": False, "direction": "Ali wants quick banter; Ada teases back."})
    r = TurnResult()
    await drain(chat.send("Selam Ada!", r))
    brief_prompt = next(c for c in calls if "memory brief" in c)
    assert "Meraklı." in brief_prompt and "Ada tends to take the lead" in brief_prompt
    assert "Content setting: standard" in brief_prompt
    system = r.context.messages[0].content
    assert "take the lead" not in system  # initiative is no longer a system rule
    last = r.context.messages[-1]
    assert last.role == "user" and "Private direction for this one reply" in last.content
    assert "Ali wants quick banter; Ada teases back." in last.content
    # inside the note, after the reminder, directly in front of the newest message
    assert last.content.index("Stay in character") < last.content.index("Ali wants quick banter")
    assert last.content.index("Ali wants quick banter") < last.content.index("Selam Ada!")
    assert r.direction == chat.last_direction == "Ali wants quick banter; Ada teases back."


async def test_no_direction_means_no_section_and_no_crash(mem):
    chat, engine, _calls, briefs = mem
    briefs.append({"about_past": False, "direction": ""})
    r = TurnResult()
    await drain(chat.send("Selam!", r))
    assert "Private direction" not in r.context.messages[-1].content
    assert r.text and r.direction == "" and chat.last_direction == ""
    engine.cfg.brief = False  # no brief at all: the base prompt alone
    r2 = TurnResult()
    await drain(chat.send("Nasılsın?", r2))
    assert "Private direction" not in r2.context.messages[-1].content and r2.text


async def test_direction_does_not_show_in_another_chat(mem, tmp_path, capsys):
    chat, _engine, _calls, briefs = mem
    briefs.append({"about_past": False, "direction": "Secret mood from chat A."})
    await drain(chat.send("Selam", TurnResult()))
    assert chat.current_direction() == "Secret mood from chat A."
    chat.new_session(model="echo:echo", character_id=chat.character().id)  # chat B
    assert chat.current_direction() == ""
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths, show_stats=False)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text("/direction\r/quit\r")
        await repl.run([])
    out = ANSI.sub("", capsys.readouterr().out)
    assert "No direction yet" in out and "Secret mood" not in out


async def test_a_failing_brief_is_reported_once_and_again_after_recovery(store, tmp_path, capsys):
    import httpx
    import respx

    config = Config(
        memory=MemoryConfig(every_exchanges=50),
        providers={"local": ProviderConfig(type="openai_compat", base_url="http://util/v1")},
    )
    registry = ProviderRegistry(config)
    engine = MemoryEngine(config, store, registry)
    chat = ChatService(config, store, registry, engine)
    chat.set_utility_model("local:m")
    char = store.add_character("ai", Card(name="Ada", personality="Curious."))
    chat.new_session(model="echo:echo", character_id=char.id)
    briefs = 0

    def reply(request):
        nonlocal briefs
        if b"memory brief" not in request.content:
            return httpx.Response(402, json={"error": {"message": "requires more credits"}})
        briefs += 1
        if briefs == 4:  # the one success
            body = json.dumps({"about_past": False, "direction": "Ada is calm."})
            data = f"data: {json.dumps({'choices': [{'delta': {'content': body}}]})}\n\ndata: [DONE]\n\n"
            return httpx.Response(200, content=data.encode())
        return httpx.Response(402, json={"error": {"message": "requires more credits"}})

    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths, show_stats=False)
    keys = "one\rtwo\rthree\r/direction\rfour\r/direction\rfive\rsix\r/quit\r"
    with respx.mock:
        respx.post("http://util/v1/chat/completions").mock(side_effect=reply)
        with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
            inp.send_text(keys)
            await repl.run([])
    out = ANSI.sub("", capsys.readouterr().out)
    assert out.count("The director isn't working") == 2  # after #2, not #3; again after #6
    assert out.index("The director isn't working") < out.index(
        "No direction: the last brief failed"
    )
    assert "local: " in out and "402" in out  # the provider's reason is named
    assert "Ada is calm." in out  # the success shows, and cleared the error
    assert engine.brief_struggling and "402" in engine.last_brief_error


async def test_a_private_chat_gets_a_direction_but_no_memory(mem):
    chat, engine, calls, briefs = mem
    scope = engine.scope(chat.require_session())
    engine.repo.add_memory(scope, "Ali'nin kedisinin adı Pamuk.")  # elsewhere: must not leak
    chat.set_private(True)
    store = chat.store

    def snapshot():
        return [
            store.conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("memories", "memories_fts", "jobs", "kv")
        ] + [engine.repo.job_stats().get("pending", 0)]

    before = snapshot()
    briefs.append({"about_past": True, "relevant": ["M1"], "direction": "Ada is guarded; short."})
    await engine.prefetch(chat.refresh(), "Kedim?", [], set(), "standard")  # allowed
    assert engine.prefetched_for(chat.session.id, "Kedim?") is not None
    r = TurnResult()
    await drain(chat.send("Kedim?", r))
    brief_prompt = next(c for c in calls if "memory brief" in c)
    assert "(none found)" in brief_prompt and "Pamuk" not in brief_prompt  # no search
    assert "Pamuk" not in calls[-1]  # nothing recalled into the reply
    assert "Ada is guarded; short." in r.context.messages[-1].content
    assert r.direction == chat.last_direction == "Ada is guarded; short."
    assert r.recalled == 0
    after = snapshot()
    assert after[:3] == before[:3] and after[4] == before[4]  # memories, index, jobs
    assert engine.prefetched_for(chat.session.id, "Kedim?") is None  # not kept around
    # kv only gets the session bookkeeping the chat itself does, never memory state
    assert not any(
        k.startswith(("memory_", "reflect_"))
        for (k,) in store.conn.execute("SELECT key FROM kv").fetchall()
    )


async def test_without_a_brief_the_best_search_results_are_used(mem):
    chat, engine, calls, _briefs = mem
    engine.cfg.brief = False
    scope = engine.scope(chat.require_session())
    engine.repo.add_memory(scope, "Ali'nin kedisinin adı Pamuk.")
    await drain(chat.send("Kedimin adı neydi?", TurnResult()))
    assert not any("memory brief" in c for c in calls)
    assert "Ali'nin kedisinin adı Pamuk." in calls[-1]


async def test_pinned_memories_are_always_recalled(mem):
    chat, engine, calls, _briefs = mem
    scope = engine.scope(chat.require_session())
    m = engine.repo.add_memory(scope, "Ali'nin doğum günü 3 Mart.")
    engine.repo.set_pinned(m.id, True)
    await drain(chat.send("Selam!", TurnResult()))
    assert "doğum günü 3 Mart" in calls[-1]


async def test_memories_from_an_undone_branch_are_ignored(mem):
    chat, engine, _calls, _briefs = mem
    await drain(chat.send("Benim adım aslında Veli.", TurnResult()))
    source = chat.last_reply()
    scope = engine.scope(chat.require_session())
    engine.repo.add_memory(scope, "Kullanıcının adı Veli.", source_message_id=source.id)
    chat.undo()
    await drain(chat.send("başka bir şey", TurnResult()))
    cands = await engine.candidates(chat.require_session(), "adım Veli", chat.history(), set())
    assert cands.memories == []


async def test_segment_summaries_replace_dropped_messages_in_the_prompt(mem):
    chat, engine, calls, _briefs = mem
    for i in range(8):
        await drain(chat.send(f"Uzun bir mesaj numara {i}. " + "kelime " * 60, TurnResult()))
    await engine.run_all()
    assert engine.repo.summaries(chat.session.id, "segment")
    chat.set_param("context_tokens", "600")
    r = TurnResult()
    await drain(chat.send("şimdi?", r))
    assert "## Story so far" in calls[-1] and "SEGMENT: Ali haritadan bahsetti." in calls[-1]
    assert r.context.dropped > 0 and r.context.story_parts > 0


async def test_day_digest_job_and_dated_recall(mem):
    chat, engine, calls, briefs = mem
    await drain(chat.send("Bugün deniz fenerine gittik.", TurnResult()))
    yesterday = (engine.today() - timedelta(days=1)).isoformat()
    chat.store.conn.execute(
        "UPDATE messages SET local_day = ? WHERE session_id = ?", (yesterday, chat.session.id)
    )
    await drain(chat.send("Merhaba yeniden", TurnResult()))  # schedules the digest
    await engine.run_all()
    assert [d.day for d in engine.repo.summaries(chat.session.id, "day")] == [yesterday]
    briefs.append({"about_past": True, "dates": [{"from": yesterday, "to": yesterday}]})
    await drain(chat.send("dün ne yapmıştık?", TurnResult()))
    assert f"What happened on {yesterday}: DAY: Ali ve Ada deniz fenerine gitti." in calls[-1]


async def test_failing_model_retries_then_gives_up(mem, monkeypatch):
    chat, engine, _calls, _briefs = mem

    async def garbage(self, model, messages, params):
        yield StreamEvent(text="I cannot do JSON today.")

    await drain(chat.send("bir", TurnResult()))
    await drain(chat.send("iki", TurnResult()))
    monkeypatch.setattr(EchoProvider, "stream_chat", garbage)
    for _ in range(3):
        await engine.run_all()
        chat.store.conn.execute("UPDATE jobs SET run_after = NULL")  # skip the wait
    assert engine.repo.job_stats().get("failed") == 1
    assert "unreadable" in engine.repo.last_job_error()


async def test_memory_disabled(mem):
    chat, engine, calls, _briefs = mem
    engine.cfg.enabled = False
    await drain(chat.send("bir", TurnResult()))
    await drain(chat.send("iki", TurnResult()))
    assert engine.repo.job_stats() == {}
    assert not any("memory brief" in c for c in calls)


# ── REPL commands ────────────────────────────────────────────────────────────


async def test_memory_commands(mem, tmp_path, capsys):
    chat, engine, _calls, _briefs = mem
    keys = (
        "Deniz fenerinde harita bulduk\r" "Çay şekersiz\r"
        "/memory run\r" "/memory\r" "/memory add Ali'nin kedisi Pamuk\r"
        "/memory pin 1\r" "/memory delete 2\r" "/memory pamuk\r"
        "/recall harita\r" "/scene\r" "/summary\r" "/day dün\r" "/memory status\r" "/quit\r"
    )  # fmt: skip
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths, show_stats=True)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text(keys)
        await repl.run([])
    out = ANSI.sub("", capsys.readouterr().out)
    assert "Done (" in out
    assert "Ali deniz fenerinde mühürlü bir harita buldu." in out
    assert "Pinned #1" in out and "Forgot #2" in out
    assert "Ali'nin kedisi Pamuk" in out
    assert "Eski liman" in out
    assert "No day summary for 2026-11-12 yet." in out  # the model resolved "dün"
    assert "memories (shared across this character's chats)" in out
    assert engine.repo.pinned_ids(engine.scope(chat.require_session())) == [1]


async def test_direction_command(mem, tmp_path, capsys):
    chat, _engine, _calls, briefs = mem
    briefs.append({"about_past": False, "direction": "Ada is wary and keeps it short."})
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths, show_stats=False)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text("/direction\rSelam\r/direction\r/quit\r")
        await repl.run([])
    out = ANSI.sub("", capsys.readouterr().out)
    assert "No direction yet" in out
    assert "Ada is wary and keeps it short." in out
