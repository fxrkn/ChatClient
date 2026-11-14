"""chatclient --serve: JSON lines for other front ends (e.g. the Rust TUI)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest
from tests.conftest import ALPHA

from chatclient.providers.base import StreamEvent
from chatclient.providers.echo import EchoProvider
from chatclient.serve import Server


class Out:
    def __init__(self):
        self.lines: list[dict] = []

    def write(self, data: bytes) -> None:
        for line in data.decode("utf-8").splitlines():
            self.lines.append(json.loads(line))

    def flush(self) -> None:
        pass


class Client:
    """Talks to a Server over (fake) stdin/stdout, like a front end would."""

    def __init__(self, chat):
        self.out = Out()
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.server = Server(chat, self.out)
        self.task = asyncio.ensure_future(self.server.run(self._lines()))
        self.next_id = 0

    async def _lines(self):
        while (line := await self.queue.get()) is not None:
            yield line

    def send_raw(self, line: str) -> None:
        self.queue.put_nowait(line)

    def start(self, method: str, **params) -> int:
        self.next_id += 1
        self.send_raw(json.dumps({"id": self.next_id, "method": method, "params": params}))
        return self.next_id

    async def wait(self, rid, timeout: float = 5.0) -> dict:
        for _ in range(int(timeout / 0.01)):
            for msg in self.out.lines:
                if msg.get("id") == rid and ("result" in msg or "error" in msg):
                    return msg
            await asyncio.sleep(0.01)
        raise AssertionError(f"no answer to {rid}: {self.out.lines}")

    async def call(self, method: str, **params) -> dict:
        return await self.wait(self.start(method, **params))

    def events(self, rid) -> list[dict]:
        return [m for m in self.out.lines if m.get("id") == rid and "event" in m]

    async def close(self) -> None:
        await self.call("shutdown")
        self.queue.put_nowait(None)
        await asyncio.wait_for(self.task, 5)


@pytest.fixture
async def client(chat):
    c = Client(chat)
    yield c
    if not c.task.done():
        await c.close()


async def test_ready_then_hello(client):
    msg = await client.call("hello")
    assert client.out.lines[0]["event"] == "ready" and client.out.lines[0]["protocol"] == 1
    assert msg["result"]["session"]["model"] == "echo:echo"


async def test_send_streams_tokens_then_the_result(client):
    rid = client.start("send", text="hello there")
    msg = await client.wait(rid)
    kinds = [e["event"] for e in client.events(rid)]
    assert kinds[0] == "start" and "token" in kinds
    text = "".join(e["text"] for e in client.events(rid) if e["event"] == "token")
    assert msg["result"]["message"]["content"] == text and "hello there" in text
    history = (await client.call("history"))["result"]
    assert [m["role"] for m in history] == ["user", "assistant"]


async def test_one_reply_at_a_time_and_cancel(client, monkeypatch):
    async def slow(self, model, messages, params):
        for word in ("one ", "two ", "three "):
            await asyncio.sleep(0.2)
            yield StreamEvent(text=word)

    monkeypatch.setattr(EchoProvider, "stream_chat", slow)
    first = client.start("send", text="tell me a story")
    await asyncio.sleep(0.3)
    busy = await client.call("send", text="and another")
    assert busy["error"]["code"] == "busy"
    assert "result" in await client.call("history")  # reading is fine meanwhile
    cancel = await client.call("cancel")
    assert cancel["result"]["cancelled"] is True
    done = await client.wait(first)
    assert done["result"]["cancelled"] is True
    assert done["result"]["message"]["content"].startswith("one")  # kept what was written


async def test_errors_are_answers_not_crashes(client):
    client.send_raw("this is not json")
    assert (await client.wait(None))["error"]["code"] == "parse_error"
    assert (await client.call("nope"))["error"]["code"] == "method_not_found"
    assert (await client.call("send"))["error"]["code"] == "invalid_params"
    assert (await client.call("session.open", id=999))["error"]["code"] == "chat"
    assert (await client.call("hello"))["result"]  # still serving


async def test_characters_sessions_and_versions(client, chat):
    chat.store.add_character("ai", ALPHA)
    chars = (await client.call("characters.list"))["result"]
    assert [c["name"] for c in chars] == ["Alpha"]
    opened = (await client.call("session.new", character="Alpha"))["result"]
    assert opened["session"]["character"]["name"] == "Alpha"
    assert opened["history"][0]["role"] == "assistant"  # the greeting
    await client.wait(client.start("send", text="hi"))
    regen = await client.wait(client.start("regen"))
    assert regen["result"]["alternatives"] == {"count": 2, "index": 1}
    back = await client.wait(client.start("alt", step=-1))
    assert back["result"]["alternatives"]["index"] == 0
    undo = (await client.call("undo"))["result"]
    assert undo["hidden"] == 2
    sessions = (await client.call("sessions.list"))["result"]
    assert sessions[0]["character"]["name"] == "Alpha" and "messages" in sessions[0]


async def test_a_repeated_reply_is_rewritten_with_a_retry_event(client, chat, monkeypatch):
    loop_text = (
        "My breath hitches, a sharp intake that feels too loud in the quiet room. I stare at "
        "your lips, the red mark on my paper blurring in my vision."
    )
    replies = iter([loop_text, loop_text, "We walk out into the cold night air together."])

    async def model(self, model, messages, params):
        yield StreamEvent(text=next(replies))

    monkeypatch.setattr(EchoProvider, "stream_chat", model)
    chat.store.add_character("ai", ALPHA)
    await client.call("session.new", character="Alpha")
    await client.wait(client.start("send", text="a"))
    rid = client.start("send", text="b")
    msg = await client.wait(rid)
    (retry,) = [e for e in client.events(rid) if e["event"] == "retry"]
    assert retry["reason"] == "repeat" and retry["repetition"] == 1.0
    assert msg["result"]["message"]["content"].startswith("We walk out")


async def test_panel_without_a_character(client):
    got = (await client.call("panel"))["result"]
    assert got["name"] is None and got["age"] is None and got["picture"] is None
    assert got["personality"] == [] and got["scene"] == {} and got["memories"] == []
    assert got["model"] == "echo:echo" and got["private"] is False
    assert got["content"] in ("standard", "mature")


async def test_panel_for_a_character_with_scene_and_memories(client, chat, tmp_path, monkeypatch):
    from chatclient.config import MemoryConfig
    from chatclient.memory.engine import MemoryEngine

    chat.memory = MemoryEngine(
        chat.config.model_copy(update={"memory": MemoryConfig(embeddings="off")}),
        chat.store,
        chat.registry,
    )
    char = chat.store.add_character("ai", ALPHA)
    await client.call("session.new", character="Alpha")
    s = chat.require_session()
    chat.store.set_scene(s.id, {"location": "a harbour", "mood": " ", "time": "dusk", "x": "no"})
    chat.memory.repo.add_memory(chat.memory.scope(s), "Alpha likes rain.", kind="preference")
    pic = tmp_path / "shot.png"
    pic.write_bytes(b"png")
    monkeypatch.setattr("chatclient.serve.pictures_of", lambda *a: [(None, pic)])

    got = (await client.call("panel"))["result"]
    assert got["name"] == "Alpha" and got["personality"] == ["Calm test personality"]
    assert got["scene"] == {"location": "a harbour", "time": "dusk"}  # blanks and unknown keys out
    assert got["memories"] == [
        {"content": "Alpha likes rain.", "pinned": False, "kind": "preference"}
    ]
    assert got["picture"] == str(pic.resolve())

    chat.set_private(True)  # private chats show no memories
    assert (await client.call("panel"))["result"]["memories"] == []
    assert char.id == s.character_id


async def test_panel_event_follows_a_reply_and_works_while_writing(client, monkeypatch):
    rid = client.start("send", text="hello")
    await client.wait(rid)
    await asyncio.sleep(0.05)
    assert {"event": "panel", "why": "reply"} in client.out.lines

    async def slow(self, model, messages, params):
        await asyncio.sleep(0.3)
        yield StreamEvent(text="late")

    monkeypatch.setattr(EchoProvider, "stream_chat", slow)
    client.start("send", text="again")
    await asyncio.sleep(0.1)
    assert "result" in await client.call("panel")  # reading is fine meanwhile


def test_the_real_command_line(tmp_path):
    """What the Rust TUI will do: start the process, write lines, read lines."""
    # no embedding model download in a test (or on CI)
    (tmp_path / "config.toml").write_text('[memory]\nembeddings = "off"\n', encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "chatclient", "--serve", "--home", str(tmp_path), "-m", "echo:echo"],
        input='{"id":1,"method":"hello"}\n{"id":2,"method":"send","params":{"text":"héllo"}}\n'
        '{"id":3,"method":"shutdown"}\n',
        capture_output=True, text=True, encoding="utf-8", timeout=60,
    )  # fmt: skip
    lines = [json.loads(x) for x in proc.stdout.splitlines()]  # stdout is only protocol
    assert proc.returncode == 0, proc.stderr
    assert lines[0]["event"] == "ready"
    sent = next(x for x in lines if x.get("id") == 2 and "result" in x)
    assert "héllo" in sent["result"]["message"]["content"]
    assert (tmp_path / "plain-chats").is_dir()  # the same files as the REPL
