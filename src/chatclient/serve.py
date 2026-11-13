"""Headless mode for other front ends: ``chatclient --serve``.

One JSON object per line on stdin (requests) and stdout (responses and events); logs
never go to stdout. The protocol is described in docs/PROTOCOL.md. Everything behind it
(characters, memory, files, models) is the same ChatService the terminal REPL uses, and
so are the rules for retrying a declined or repeated reply.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, BinaryIO

from chatclient import __version__
from chatclient.characters import card_content
from chatclient.core.chat import ChatError, ChatService, TurnResult
from chatclient.providers.base import ProviderError, StreamEvent
from chatclient.storage import Message, Session

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 1
WORKER_IDLE_SECONDS = 1.0
# allowed while a reply is being written (they change nothing)
READ_ONLY = {"hello", "session.get", "history", "sessions.list", "characters.list",
             "models.list", "draft"}  # fmt: skip

Starter = Callable[[TurnResult], AsyncIterator[StreamEvent]]


class RequestError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ── what goes over the wire ─────────────────────────────────────────────────


def message_info(m: Message) -> dict[str, Any]:
    return {
        "id": m.id, "uuid": m.uuid, "role": m.role, "content": m.content, "model": m.model,
        "created_at": m.created_at, "meta": m.meta or {},
    }  # fmt: skip


class Server:
    def __init__(self, chat: ChatService, out: BinaryIO):
        self.chat = chat
        self.out = out
        self.generation: asyncio.Task[None] | None = None
        self.job_task: asyncio.Task[bool] | None = None
        self.prefetch_task: asyncio.Task[None] | None = None
        self.running = True
        self._write_lock = threading.Lock()
        self.methods: dict[str, Callable[[dict[str, Any]], Awaitable[Any]]] = {
            "hello": self.hello,
            "session.get": self.session_get,
            "sessions.list": self.sessions_list,
            "session.new": self.session_new,
            "session.open": self.session_open,
            "history": self.history,
            "undo": self.undo,
            "characters.list": self.characters_list,
            "models.list": self.models_list,
            "model.set": self.model_set,
            "content.set": self.content_set,
            "private.set": self.private_set,
            "draft": self.draft,
        }
        # these stream a reply: "start", "token"… events, then the result
        self.streaming: dict[str, Callable[[dict[str, Any]], Starter]] = {
            "send": self._send,
            "regen": self._regen,
            "continue": lambda p: self.chat.continue_reply,
            "edit": self._edit,
        }

    def _send(self, p: dict[str, Any]) -> Starter:
        text = _text(p)  # checked now, not once streaming has begun
        return lambda r: self.chat.send(text, r)

    def _edit(self, p: dict[str, Any]) -> Starter:
        text = _text(p)
        return lambda r: self.chat.edit_and_resend(text, r)

    def _regen(self, p: dict[str, Any]) -> Starter:
        model = p.get("model") or None
        return lambda r: self.chat.regenerate(r, model=model)

    # ── wire ────────────────────────────────────────────────────────────────

    def emit(self, obj: dict[str, Any]) -> None:
        data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        with self._write_lock:
            self.out.write(data)
            self.out.flush()

    def reply(self, rid: Any, result: Any) -> None:
        self.emit({"id": rid, "result": result})

    def fail(self, rid: Any, code: str, message: str) -> None:
        self.emit({"id": rid, "error": {"code": code, "message": message}})

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self, lines: AsyncIterator[str], notes: list[str] = ()) -> None:  # type: ignore[assignment]
        worker = asyncio.ensure_future(self.memory_worker())
        self.emit(
            {
                "event": "ready", "protocol": PROTOCOL_VERSION, "version": __version__,
                "notes": list(notes),
            }
        )  # fmt: skip
        try:
            async for line in lines:
                if not line.strip():
                    continue
                await self.handle(line)
                if not self.running:
                    break
        finally:
            self.running = False
            for task in (self.generation, self.prefetch_task, worker):
                if task is not None and not task.done():
                    task.cancel()
            for task in (self.generation, worker):
                if task is not None:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

    async def handle(self, line: str) -> None:
        try:
            req = json.loads(line)
            if not isinstance(req, dict):
                raise ValueError
        except ValueError:
            self.fail(None, "parse_error", "each line must be one JSON object")
            return
        rid, method = req.get("id"), req.get("method")
        params = req.get("params") or {}
        if not isinstance(params, dict):
            self.fail(rid, "invalid_params", "params must be an object")
            return
        try:
            if method == "shutdown":
                # a reply being written is finished first (send "cancel" to stop it)
                if self.generation is not None and not self.generation.done():
                    with contextlib.suppress(Exception):
                        await self.generation
                self.running = False
                self.reply(rid, {})
            elif method == "cancel":
                stopped = self.generation is not None and not self.generation.done()
                if stopped:
                    self.generation.cancel()  # type: ignore[union-attr]
                self.reply(rid, {"cancelled": stopped})
            elif method == "alt":
                self.start_generation(rid, self.alternative(int(params.get("step", 1))))
            elif method in self.streaming:
                self.start_generation(rid, self.streaming[method](params))
            elif method in self.methods:
                writing = self.generation is not None and not self.generation.done()
                if writing and method not in READ_ONLY:
                    raise RequestError("busy", "a reply is being written; cancel it first")
                self.reply(rid, await self.methods[method](params))
            else:
                raise RequestError("method_not_found", f"unknown method {method!r}")
        except RequestError as e:
            self.fail(rid, e.code, str(e))
        except ChatError as e:
            self.fail(rid, "chat", str(e))
        except ProviderError as e:
            self.fail(rid, "provider", str(e))
        except (KeyError, TypeError, ValueError) as e:
            self.fail(rid, "invalid_params", str(e) or type(e).__name__)
        except Exception as e:  # never let one request take the server down
            log.exception("request %r failed", method)
            self.fail(rid, "internal", f"{type(e).__name__}: {e}")

    # ── replies (streamed) ──────────────────────────────────────────────────

    def start_generation(self, rid: Any, start: Starter | None) -> None:
        if self.generation is not None and not self.generation.done():
            raise RequestError("busy", "a reply is being written; cancel it first")
        if start is None:  # /next past the newest version: write a new one
            start = self.chat.regenerate
        self.interrupt_memory_job()
        self.generation = asyncio.ensure_future(self.generate(rid, start))

    async def generate(self, rid: Any, start: Starter) -> None:
        done: set[str] = set()
        result = TurnResult()
        try:
            while True:
                result = TurnResult()
                self.emit({"id": rid, "event": "start", "model": self.chat.require_session().model})
                async for ev in start(result):
                    if ev.text:
                        self.emit({"id": rid, "event": "token", "text": ev.text})
                retry = self.chat.next_attempt(result, done)
                if retry is None:
                    break
                self.emit(
                    {
                        "id": rid, "event": "retry", "reason": retry.reason,
                        "from": result.model, "to": retry.model,
                        "repetition": round(result.repetition, 3),
                    }
                )  # fmt: skip
                start = retry.start
            self.reply(rid, self.turn_info(result))
        except asyncio.CancelledError:
            if self.running:  # stopped by "cancel": what was written is kept
                self.reply(rid, self.turn_info(result))
            raise
        except ChatError as e:
            self.fail(rid, "chat", str(e))
        except ProviderError as e:
            self.fail(rid, "provider", str(e))
        except Exception as e:
            log.exception("generation failed")
            self.fail(rid, "internal", f"{type(e).__name__}: {e}")

    def turn_info(self, r: TurnResult) -> dict[str, Any]:
        out_tokens = r.usage.completion_tokens if r.usage else None
        sibs, index = self.chat.alternatives()
        return {
            "message": message_info(r.message) if r.message else None,
            "model": r.model, "cancelled": r.cancelled, "finish_reason": r.finish_reason,
            "refused": r.refused, "repetition": round(r.repetition, 3),
            "stats": {
                "completion_tokens": out_tokens,
                "prompt_tokens": r.usage.prompt_tokens if r.usage else None,
                "seconds": round(r.elapsed, 3),
                "first_token_seconds": (
                    round(r.time_to_first_token, 3) if r.time_to_first_token is not None else None
                ),
                "recalled": r.recalled, "conflicts": r.conflicts,
                "dropped_context": r.context.dropped if r.context else 0,
            },
            "alternatives": {"count": len(sibs), "index": index},
        }  # fmt: skip

    def alternative(self, step: int) -> Starter | None:
        """Show another version of the last reply; past the newest, write a new one."""
        msg = self.chat.switch_alternative(step)
        if msg is None:
            return None

        async def shown(result: TurnResult) -> AsyncIterator[StreamEvent]:
            result.message, result.text, result.model = msg, msg.content, msg.model or ""
            result.ended = result.started
            yield StreamEvent(msg.content)

        return shown

    # ── simple requests ─────────────────────────────────────────────────────

    def session_info(self, s: Session | None = None) -> dict[str, Any]:
        """A session; without ``s``, the current one (with its content setting)."""
        current = s is None
        s = s or self.chat.refresh()
        char = self.chat.store.get_character(s.character_id)
        persona = self.chat.store.get_character(s.persona_id)
        info = {
            "id": s.id, "uuid": s.uuid, "title": s.title, "model": s.model,
            "private": s.private, "created_at": s.created_at, "updated_at": s.updated_at,
            "character": {"id": char.id, "name": char.name} if char else None,
            "persona": {"id": persona.id, "name": persona.name} if persona else None,
        }  # fmt: skip
        if current:
            info["content"] = self.chat.content_mode()
        return info

    async def hello(self, p: dict[str, Any]) -> Any:
        return {
            "protocol": PROTOCOL_VERSION, "version": __version__,
            "session": self.session_info(), "user_name": self.chat.user_name(),
        }  # fmt: skip

    async def session_get(self, p: dict[str, Any]) -> Any:
        return self.session_info()

    async def sessions_list(self, p: dict[str, Any]) -> Any:
        return [
            {**self.session_info(item.session), "messages": item.message_count}
            for item in self.chat.store.list_sessions(int(p.get("limit", 50)))
        ]

    async def session_new(self, p: dict[str, Any]) -> Any:
        char_id = None
        if ref := p.get("character"):
            char = self.chat.store.find_character(str(ref), "ai")
            if char is None:
                raise ChatError(f"no character matches {ref!r}")
            char_id = char.id
        model = p.get("model") or self.chat.require_session().model  # keep the current one
        self.chat.new_session(model=model, title=p.get("title") or "", character_id=char_id)
        return {"session": self.session_info(), "history": await self.history({})}

    async def session_open(self, p: dict[str, Any]) -> Any:
        self.chat.open_session(int(p["id"]))
        return {"session": self.session_info(), "history": await self.history({})}

    async def history(self, p: dict[str, Any]) -> Any:
        limit = p.get("limit")
        return [message_info(m) for m in self.chat.history(int(limit) if limit else None)]

    async def undo(self, p: dict[str, Any]) -> Any:
        return {"hidden": self.chat.undo(), "history": await self.history({})}

    async def characters_list(self, p: dict[str, Any]) -> Any:
        kind = "user" if p.get("kind") == "persona" else "ai"
        return [
            {
                "id": c.id, "uuid": c.uuid, "name": c.name, "tags": c.card.tags,
                "description": c.card.description, "content": card_content(c.card),
            }
            for c in self.chat.store.list_characters(kind)
        ]  # fmt: skip

    async def models_list(self, p: dict[str, Any]) -> Any:
        results = await self.chat.registry.list_models()
        return {
            name: models if isinstance(models, list) else {"error": str(models)}
            for name, models in results.items()
        }

    async def model_set(self, p: dict[str, Any]) -> Any:
        return {"model": str(self.chat.set_model(str(p["model"])))}

    async def content_set(self, p: dict[str, Any]) -> Any:
        scope = self.chat.set_content_mode(str(p["mode"]))
        return {"mode": self.chat.content_mode(), "scope": scope}

    async def private_set(self, p: dict[str, Any]) -> Any:
        self.chat.set_private(bool(p["on"]))
        return self.session_info()

    async def draft(self, p: dict[str, Any]) -> Any:
        """The user is typing: prepare memory now so sending doesn't wait for it."""
        text = str(p.get("text") or "")
        if self.prefetch_task is not None and not self.prefetch_task.done():
            self.prefetch_task.cancel()
        self.prefetch_task = asyncio.ensure_future(self._prefetch(text))
        return {}

    async def _prefetch(self, text: str) -> None:
        try:
            await self.chat.prefetch_memory(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("prefetch failed")

    # ── memory work in the background ───────────────────────────────────────

    async def memory_worker(self) -> None:
        engine = self.chat.memory
        if engine is None:
            return
        while self.running:
            await asyncio.sleep(WORKER_IDLE_SECONDS)
            busy = self.generation is not None and not self.generation.done()
            if busy or (self.prefetch_task is not None and not self.prefetch_task.done()):
                continue
            self.job_task = asyncio.ensure_future(engine.run_next())
            try:
                did = await self.job_task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
                continue  # interrupted by a reply: the job goes back to the queue
            except Exception:
                log.exception("memory worker error")
                did = False
            finally:
                self.job_task = None
            if not did:
                await asyncio.sleep(3)

    def interrupt_memory_job(self) -> None:
        if self.job_task is not None and not self.job_task.done():
            self.job_task.cancel()


def _text(p: dict[str, Any]) -> str:
    text = p.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text is required")
    return text


async def stdin_lines() -> AsyncIterator[str]:
    """Lines from stdin without blocking the event loop (works on Windows too)."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def read() -> None:
        for raw in sys.stdin.buffer:
            loop.call_soon_threadsafe(queue.put_nowait, raw.decode("utf-8", "replace"))
        loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=read, daemon=True, name="stdin").start()
    while (line := await queue.get()) is not None:
        yield line


async def serve_stdio(chat: ChatService, notes: list[str] = ()) -> None:  # type: ignore[assignment]
    await Server(chat, sys.stdout.buffer).run(stdin_lines(), notes)
