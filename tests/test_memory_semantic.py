"""Semantic candidates, raw-message quotes, contradictions by fact key, reflection."""

import json
import re

import pytest

from chatclient.characters import Card
from chatclient.config import Config, MemoryConfig, ProviderConfig
from chatclient.core.chat import ChatService, TurnResult
from chatclient.memory.embed import VectorIndex, to_blob
from chatclient.memory.engine import MemoryEngine
from chatclient.memory.repo import same_value
from chatclient.providers.base import StreamEvent
from chatclient.providers.echo import EchoProvider
from chatclient.providers.registry import ProviderRegistry
from chatclient.text import normalize_for_search

# A tiny stand-in for a real embedding model: words map to "concepts", so texts with
# the same meaning but no shared words end up close (like the real model does).
CONCEPTS = {
    "sad": ("uzgun", "kotu", "huzun", "agla", "dertli"),
    "pet": ("kedi", "evcil", "hayvan", "pamuk"),
    "night": ("aksam", "gece"),
    "map": ("harita", "hazine"),
    "work": ("yazilim", "isim", "meslek", "calis"),
}


class FakeEmbedder:
    name = "fake:concepts"
    loaded = True

    async def embed(self, texts):
        out = []
        for text in texts:
            norm = normalize_for_search(text)
            vec = [float(sum(norm.count(w) for w in words)) for words in CONCEPTS.values()]
            out.append([*vec, 0.01])  # never all-zero
        return out


def scripted_model(script, calls):
    """Memory passes pop JSON from ``script``; briefs pass; chat replies say ok."""

    async def stream(self, model, messages, params):
        prompt = messages[-1].content
        calls.append("\n".join(m.content for m in messages))
        if "memory brief" in prompt:
            text = '{"about_past": false}'
        elif "Do they contradict each other" in prompt:
            # a stand-in judge: different cities clash, two tellings of a moment don't
            same_moment = "unfamiliar" in prompt and "first time" in prompt
            text = json.dumps({"contradict": not same_moment})
        elif "Return ONLY a JSON object" in prompt:
            text = json.dumps(script.pop(0) if script else {"memories": []}, ensure_ascii=False)
        elif "what has" in prompt and "learned" in prompt:
            text = "Ali üzgün olduğunda kısa yazıyor.\nAli kedisine çok bağlı."
        else:
            text = "Tamam."
        yield StreamEvent(text=text)

    return stream


@pytest.fixture
def setup(store, monkeypatch):
    script: list[dict] = []
    calls: list[str] = []
    monkeypatch.setattr(EchoProvider, "stream_chat", scripted_model(script, calls))
    config = Config(
        memory=MemoryConfig(every_exchanges=1, keep_recent_messages=2),
        providers={"local": ProviderConfig(type="openai_compat", base_url="http://localhost:1/v1")},
    )
    registry = ProviderRegistry(config)
    engine = MemoryEngine(config, store, registry)
    engine.embedder = FakeEmbedder()
    engine.semantic = "waiting"
    chat = ChatService(config, store, registry, engine)
    char = store.add_character("ai", Card(name="Ada"))
    chat.new_session(model="echo:echo", character_id=char.id)
    chat.set_persona(Card(name="Ali"))
    return chat, engine, script, calls


async def say(chat, text):
    r = TurnResult()
    async for _ in chat.send(text, r):
        pass
    return r


def test_same_value():
    assert same_value("Pamuk", "pamuk.")
    assert same_value("İstanbul", "Istanbul'da")
    assert not same_value("Pamuk", "Boncuk")
    assert same_value(None, "x")  # unknown is never a contradiction


def test_vector_index_filters_by_type_session_and_owner():
    idx = VectorIndex()
    idx.load("m", [("memory", 1, 10, to_blob([1, 0])), ("message", 2, 10, to_blob([1, 0.1])),
                   ("message", 3, 20, to_blob([0, 1]))])  # fmt: skip
    assert idx.search([1, 0], owner_type="message")[0][0] == 2
    assert [i for i, _ in idx.search([1, 0], owner_type="message", sessions={20})] == [3]
    assert idx.search([1, 0], owner_type="memory", owners={99}) == []
    assert idx.search([1, 0, 0], owner_type="memory") == []  # wrong dimension: ignored


# ── stage 0: indexing ────────────────────────────────────────────────────────


async def test_index_runs_first_and_vectors_everything(setup):
    chat, engine, script, _ = setup
    script.append({"memories": [{"content": "Ali o akşam çok üzgündü."}]})
    await say(chat, "Bugün kendimi çok kötü hissettim.")
    job = engine.repo.next_job()
    assert job.kind == "index"  # cheapest first, though queued alongside the memory pass
    engine.repo.release_job(job.id)
    await engine.run_all()
    assert engine.semantic == "ready"
    assert engine.repo.count_unindexed(engine.embedder.name) == 0  # incl. the new memory


# ── candidates that don't depend on shared words or on the extractor ─────────


async def test_meaning_based_candidates_without_shared_words(setup):
    chat, engine, _script, calls = setup
    engine.cfg.brief = False  # look at the raw search results
    scope = engine.scope(chat.require_session())
    engine.repo.add_memory(scope, "Ali o akşam çok üzgündü çünkü babasını özlüyordu.")
    engine.repo.add_memory(scope, "Ali İstanbul'da yazılımcı olarak çalışıyor.")
    engine.repo.enqueue(None, "index")
    await engine.run_all()
    await say(chat, "O dertli gecemi hatırlıyor musun?")  # no word in common with the memory
    assert "Ali o akşam çok üzgündü" in calls[-1]
    assert "yazılımcı" not in calls[-1]  # below the similarity floor


async def test_raw_messages_are_quoted_when_nothing_was_extracted(setup):
    chat, engine, _script, calls = setup
    engine.cfg.brief = False
    await say(chat, "Bu arada kız kardeşimin adı Zeynep, Ankara'da yaşıyor.")
    for i in range(6):
        await say(chat, f"başka bir konu {i}")
    await engine.run_all()  # the scripted extractor returns nothing for all of these
    assert engine.repo.count_memories(engine.scope(chat.require_session())) == 0
    chat.set_param("context_tokens", "900")  # so the early message is off screen
    await say(chat, "Zeynep geçen hafta ne demişti?")
    assert 'Ali: "Bu arada kız kardeşimin adı Zeynep, Ankara\'da yaşıyor."' in calls[-1]


async def test_quotes_skip_messages_still_on_screen(setup):
    chat, engine, _script, calls = setup
    engine.cfg.brief = False
    await say(chat, "Zeynep bugün aradı.")
    await engine.run_all()
    await say(chat, "Zeynep ne demişti?")
    assert "Earlier words, quoted" not in calls[-1]  # it's right there in the history


# ── contradictions: the model names the fact, the code compares ──────────────

CAT_PAMUK = {"memories": [{"content": "Ali'nin kedisinin adı Pamuk.", "said_by": "Ali",
             "fact": {"key": "Ali.cat.name", "value": "Pamuk"}}]}  # fmt: skip
CAT_BONCUK = {"memories": [{"content": "Ali'nin kedisinin adı Boncuk.", "said_by": "Ali",
              "fact": {"key": "ali.cat.name", "value": "Boncuk"}}]}  # fmt: skip


async def test_user_contradiction_is_raised_once(setup):
    chat, engine, script, calls = setup
    script += [CAT_PAMUK, CAT_BONCUK]
    await say(chat, "Kedimin adı Pamuk.")
    await engine.run_all()
    await say(chat, "Boncuk bugün çok yaramazdı.")
    pass_prompt = None
    await engine.run_all()
    pass_prompt = next(c for c in reversed(calls) if "Known fact keys" in c)
    assert "- ali cat name = Pamuk" in pass_prompt  # existing keys are offered for reuse
    scope = engine.scope(chat.require_session())
    mems = {m.value: m for m in engine.repo.list_memories(scope)}
    assert mems["Pamuk"].outdated and not mems["Boncuk"].outdated  # newest wins, old kept

    r = await say(chat, "Neyse, ne yapalım bugün?")
    assert r.conflicts == 1
    assert "Contradiction: Ali said earlier that Ali'nin kedisinin adı Pamuk." in calls[-1]
    r = await say(chat, "Tamam.")
    assert r.conflicts == 0  # brought up once only


async def test_two_tellings_of_a_moment_are_not_a_contradiction(setup):
    """Seen in a real chat: the same moment described twice, under one fact key, made
    the character say "but you told me…" for nothing."""
    chat, engine, script, _calls = setup
    first = {
        "memories": [
            {
                "content": "Emma found the taste unfamiliar.",
                "kind": "fact",
                "fact": {"key": "emma.reaction.taste", "value": "unfamiliar"},
            }
        ]
    }
    again = {"memories": [{"content": "Emma, for the first time, found it unfamiliar but fine.",
                           "kind": "fact", "fact": {"key": "emma.reaction.taste",
                                                    "value": "not unpleasant"}}]}  # fmt: skip
    as_event = {"memories": [{"content": "Emma tasted it.", "kind": "event",
                              "fact": {"key": "emma.reaction.taste", "value": "tasted"}}]}  # fmt: skip
    script += [first, again, as_event]
    for text in ("a", "b", "c"):
        await say(chat, text)
        await engine.run_all()
    assert engine.repo.conflicts(chat.require_session().id, None) == []  # judged compatible
    r = await say(chat, "d")
    assert r.conflicts == 0


async def test_character_contradiction_asks_for_self_correction(setup):
    chat, engine, script, calls = setup
    char_says = {"memories": [{"content": "Ada, kedinin adının Tekir olduğunu söyledi.",
                 "said_by": "Ada", "fact": {"key": "Ali.cat.name", "value": "Tekir"}}]}  # fmt: skip
    script += [CAT_PAMUK, char_says]
    await say(chat, "Kedimin adı Pamuk.")
    await engine.run_all()
    await say(chat, "Kedim nasıl sence?")
    await engine.run_all()
    await say(chat, "Hmm.")
    assert "you (Ada) later said" in calls[-1] and "correct it naturally" in calls[-1]


async def test_same_value_is_a_duplicate_not_a_conflict(setup):
    chat, engine, script, _ = setup
    script += [CAT_PAMUK, CAT_PAMUK]
    await say(chat, "Kedimin adı Pamuk.")
    await engine.run_all()
    await say(chat, "Pamuk uyuyor.")
    await engine.run_all()
    assert engine.repo.conflicts(chat.session.id, status=None) == []
    assert engine.repo.count_memories(engine.scope(chat.require_session())) == 1


# ── stage 2: reflection ──────────────────────────────────────────────────────


async def test_reflection_adds_insights(setup):
    chat, engine, _script, _calls = setup
    scope = engine.scope(chat.require_session())
    for i in range(12):
        engine.repo.add_memory(scope, f"Ali ile ilgili ayrı bir bilgi numara {i} burada.")
    await say(chat, "selam")
    await engine.run_all()
    insights = sorted(
        m.content for m in engine.repo.list_memories(scope, 50) if m.kind == "insight"
    )
    assert insights == ["Ali kedisine çok bağlı.", "Ali üzgün olduğunda kısa yazıyor."]


# ── /memory status shows the semantic state ──────────────────────────────────


async def test_status_and_conflict_commands(setup, tmp_path, capsys):
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from chatclient.i18n import Translator
    from chatclient.paths import resolve_paths
    from chatclient.ui.repl import Repl

    chat, engine, script, _ = setup
    script += [CAT_PAMUK, CAT_BONCUK]
    await say(chat, "Kedimin adı Pamuk.")
    await engine.run_all()
    await say(chat, "Boncuk uyuyor.")
    await engine.run_all()
    paths = resolve_paths(tmp_path)
    paths.ensure()
    repl = Repl(chat, Translator("en"), paths)
    with create_pipe_input() as inp, create_app_session(input=inp, output=DummyOutput()):
        inp.send_text("/memory status\r/memory conflicts\r/memory\r/quit\r")
        await repl.run([])
    out = re.sub(r"\x1b\[[0-9;]*m", "", capsys.readouterr().out)
    assert "Meaning-based recall: ready fake:concepts" in out
    assert "Ali'nin kedisinin adı Pamuk." in out and "→" in out
    assert "outdated" in out


def test_only_the_users_own_words_are_quoted_back(tmp_path):
    """The character's old prose quoted back gets copied by small models (a loop)."""
    from chatclient.characters import Card
    from chatclient.config import Config, MemoryConfig
    from chatclient.memory.engine import Candidates, MemoryEngine
    from chatclient.providers.registry import ProviderRegistry
    from chatclient.storage import Store

    store = Store(tmp_path / "q.db")
    config = Config(memory=MemoryConfig(embeddings="off"))
    engine = MemoryEngine(config, store, ProviderRegistry(config))
    char = store.add_character("ai", Card(name="Vera"))
    s = store.create_session("echo:echo")
    store.update_session(s.id, character_id=char.id)
    u = store.add_message(s.id, None, "user", "My exam is on Friday.")
    a = store.add_message(s.id, u.id, "assistant", "My mind is a storm crashing on the shore.")
    s = store.get_session(s.id)
    rec = engine._select(s, Candidates([], [u, a], []), None, set())
    assert [q.id for q in rec.quotes] == [u.id]
    store.close()


def test_memory_pass_asks_for_facts_not_style():
    from chatclient.memory.tasks import memory_pass_prompt

    prompt = memory_pass_prompt("Vera", "Furkan", {}, [], [("2026-11-13", "Vera", "x")])[-1].content
    assert "Never record how something was described" in prompt
    assert "probably made up" in prompt
