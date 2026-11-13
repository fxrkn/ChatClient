"""Memory commands and the background worker (mixed into the REPL)."""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta
from typing import TYPE_CHECKING

from rich.table import Table
from rich.text import Text

from chatclient.core.chat import ChatError

if TYPE_CHECKING:
    from chatclient.memory.engine import MemoryEngine
    from chatclient.ui.repl import Repl

log = logging.getLogger(__name__)

MEMORY_SUBCOMMANDS = ("list", "add", "pin", "unpin", "delete", "conflicts", "status", "run")
WORKER_IDLE_SECONDS = 1.5
PREFETCH_PAUSE = 0.8  # seconds without typing before the brief is prepared


class MemoryCommands:
    # ── background worker ───────────────────────────────────────────────────

    def engine(self: Repl) -> MemoryEngine:
        if self.chat.memory is None:
            raise ChatError(self.t("memory_off"))
        return self.chat.memory

    async def memory_worker(self: Repl) -> None:
        """Runs queued memory jobs one at a time while no reply is streaming.
        A reply starting cancels the current job; it goes back to the queue."""
        engine = self.chat.memory
        if engine is None:
            return
        while self.running:
            await asyncio.sleep(WORKER_IDLE_SECONDS)
            if self.streaming or self.prefetch_running():
                continue
            self.job_task = asyncio.ensure_future(engine.run_next())
            try:
                did = await self.job_task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise  # the app is closing
                continue  # interrupted by a new reply
            except Exception:
                log.exception("memory worker error")
                did = False
            finally:
                self.job_task = None
            if not did:
                await asyncio.sleep(3)

    def interrupt_memory_job(self: Repl) -> None:
        if self.job_task is not None and not self.job_task.done():
            self.job_task.cancel()

    # ── preparing memory while the user types ───────────────────────────────

    def prefetch_running(self: Repl) -> bool:
        return self.prefetch_task is not None and not self.prefetch_task.done()

    def schedule_prefetch(self: Repl, text: str) -> None:
        """Called on every keystroke: after a short pause, prepare the brief for the draft.
        Typing on cancels and restarts it; an emptied box (e.g. after Enter) leaves a
        running preparation alone, since that is exactly the message being sent."""
        if self.chat.memory is None or not self.chat.memory.cfg.enabled:
            return
        text = text.strip()
        if not text:
            return
        if self.prefetch_running():
            if self.prefetch_text == text:
                return
            self.prefetch_task.cancel()  # type: ignore[union-attr]
        if text.startswith("/"):
            return
        self.prefetch_text = text
        self.prefetch_task = asyncio.ensure_future(self._prefetch_after_pause(text))

    async def _prefetch_after_pause(self: Repl, text: str) -> None:
        await asyncio.sleep(PREFETCH_PAUSE)
        if self.streaming:
            return
        self.interrupt_memory_job()  # the brief gets the model first
        try:
            await self.chat.prefetch_memory(text)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("preparing memory for the draft failed")

    async def settle_prefetch(self: Repl, sent: str) -> None:
        """On Enter: let a preparation for exactly this message finish; drop any other."""
        if not self.prefetch_running():
            return
        if self.prefetch_text == sent.strip():
            try:
                await asyncio.shield(self.prefetch_task)  # type: ignore[arg-type]
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling():
                    raise
            except Exception:
                pass
        else:
            self.prefetch_task.cancel()  # type: ignore[union-attr]

    # ── /memory ─────────────────────────────────────────────────────────────

    async def memory_command(self: Repl, arg: str) -> None:
        engine = self.engine()
        sub, _, rest = arg.strip().partition(" ")
        rest = rest.strip()
        scope = engine.scope(self.chat.require_session())
        match sub.lower():
            case "" | "list":
                self.show_memories(engine.repo.list_memories(scope, 30))
            case "add":
                if not rest:
                    raise ChatError(self.t("memory_add_usage"))
                m = engine.repo.add_memory(
                    scope, rest, kind="fact", importance=0.8, pinned=False,
                    day=engine.today().isoformat(),
                )  # fmt: skip
                if engine.embedder is not None:
                    engine.repo.enqueue(None, "index")
                engine.save_snapshot(scope.session_id)
                self.info(self.t("memory_added", id=m.id))
            case "pin" | "unpin":
                if not rest.lstrip("#").isdigit():
                    raise ChatError(self.t("memory_id_usage", cmd=sub))
                if not engine.repo.set_pinned(int(rest.lstrip("#")), sub == "pin"):
                    raise ChatError(self.t("memory_not_found", id=rest))
                engine.save_snapshot(scope.session_id)
                self.info(self.t("memory_pinned" if sub == "pin" else "memory_unpinned", id=rest))
            case "delete" | "rm" | "forget":
                if not rest.lstrip("#").isdigit():
                    raise ChatError(self.t("memory_id_usage", cmd=sub))
                if not engine.repo.delete_memory(int(rest.lstrip("#"))):
                    raise ChatError(self.t("memory_not_found", id=rest))
                engine.save_snapshot(scope.session_id)
                self.info(self.t("memory_deleted", id=rest))
            case "status":
                self.memory_status()
            case "conflicts" | "conflict":
                self.show_conflicts()
            case "run":
                self.info(self.t("memory_running"))
                n = await engine.run_all()
                self.info(self.t("memory_ran", n=n))
                if err := engine.repo.last_job_error():
                    self.warn(self.t("memory_last_error", error=err))
            case _:
                # "/memory <text>" searches
                self.show_memories(engine.repo.list_memories(scope, 30, query=arg))

    def show_memories(self: Repl, memories: list) -> None:
        if not memories:
            self.info(self.t("memory_none"))
            return
        table = Table(box=None, header_style=self.style("accent", "bold"), pad_edge=False)
        for col in ("#", "day", "", "memory"):
            table.add_column(col, no_wrap=col != "memory")
        for m in memories:
            mark = "◆" if m.pinned else ("★" if m.importance >= 0.8 else "")
            where = f" ({m.place})" if m.place else ""
            if m.outdated:
                body = Text(f"{m.content} — {self.t('memory_outdated')}", style=self.style("dim"))
            elif m.kind == "insight":
                body = Text(m.content, style=self.style("action", "italic"))
            else:
                body = Text(m.content + where, style=self.style("text"))
            table.add_row(str(m.id), Text(m.day or "", style=self.style("dim")), mark, body)
        self.console.print(table)
        self.info(self.t("memory_hint"))

    def show_conflicts(self: Repl) -> None:
        conflicts = self.engine().repo.conflicts(self.chat.require_session().id, status=None)
        if not conflicts:
            self.info(self.t("conflicts_none"))
            return
        for c in conflicts:
            self.console.print(
                f"  {c.old.day or ''}  {c.old.content}", style=self.style("dim"), markup=False
            )
            self.console.print(
                f"  → {c.new.day or ''}  {c.new.content}", style=self.style("warn"), markup=False
            )

    def memory_status(self: Repl) -> None:
        engine = self.engine()
        s = self.chat.require_session()
        scope = engine.scope(s)
        stats = engine.repo.job_stats()
        segments = len(engine.repo.summaries(s.id, "segment"))
        days = len(engine.repo.summaries(s.id, "day"))
        self.info(
            self.t(
                "memory_status",
                memories=engine.repo.count_memories(scope),
                segments=segments,
                days=days,
                pending=stats.get("pending", 0) + stats.get("running", 0),
                failed=stats.get("failed", 0),
                scope=self.t("scope_character") if scope.character_id else self.t("scope_session"),
            )
        )
        if err := engine.repo.last_job_error():
            self.warn(self.t("memory_last_error", error=err))
        sem = engine.semantic
        detail = engine.embedder.name if engine.embedder else ""
        if sem == "ready" and engine.embedder:
            waiting = engine.repo.count_unindexed(engine.embedder.name)
            if waiting:
                detail += " · " + self.t("semantic_waiting_n", n=waiting)
        self.info(self.t("semantic_status", state=self.t(f"semantic_{sem}"), detail=detail))

    # ── /recall, /scene, /summary, /day ─────────────────────────────────────

    async def recall_command(self: Repl, text: str) -> None:
        if not text.strip():
            raise ChatError(self.t("recall_usage"))
        engine = self.engine()
        s = self.chat.refresh()
        branch = self.chat.store.branch(s.head_message_id) if s.head_message_id else []
        cands = await engine.candidates(s, text, branch, visible=set())
        brief = await engine.make_brief(s, text, branch[-8:], cands)
        rec = engine._select(s, cands, brief, visible=set())
        if brief is not None:
            for a, b in brief.dates:
                self.info(self.t("recall_date", range=a if a == b else f"{a} … {b}"))
            for c in brief.claims:
                self.info(f"  {c.status}: {c.claim}")
            if brief.note:
                self.info(f"  → {brief.note}")
        else:
            self.info(self.t("recall_no_brief"))
        for d in rec.days:
            self.heading(d.day or "")
            self.console.print(d.content, style=self.style("text"), markup=False)
        if rec.memories:
            self.show_memories(rec.memories)
        for msg in rec.quotes:
            who = self.user_label() if msg.role == "user" else self.assistant_label(msg.model)
            text_ = " ".join(msg.content.split())
            self.console.print(
                f'  [{msg.local_day}] {who}: "{text_[:200]}{"…" if len(text_) > 200 else ""}"',
                style=self.style("dim"),
                markup=False,
            )
        if not rec.count:
            self.info(self.t("memory_none"))

    def scene_command(self: Repl, arg: str) -> None:
        engine = self.engine()
        s = self.chat.refresh()  # updated in the background
        if arg.strip().lower() == "clear":
            engine.repo.set_scene(s.id, {})
            self.chat.refresh()
            self.info(self.t("scene_cleared"))
            return
        if not s.scene:
            self.info(self.t("scene_none"))
            return
        for key, value in s.scene.items():
            shown = "; ".join(value) if isinstance(value, list) else str(value)
            self.console.print(f"  {key:<13} {shown}", style=self.style("text"), markup=False)

    def summary_command(self: Repl) -> None:
        engine = self.engine()
        s = self.chat.require_session()
        synopsis = engine.repo.summaries(s.id, "synopsis")
        segments = engine.repo.summaries(s.id, "segment")
        if not synopsis and not segments:
            self.info(self.t("summary_none"))
            return
        if synopsis:
            self.heading(self.t("summary_story"))
            self.console.print(synopsis[-1].content, style=self.style("text"), markup=False)
        since = (synopsis[-1].end_message_id or 0) if synopsis else 0
        for seg in segments:
            if (seg.end_message_id or 0) > since:
                self.console.print(f"• {seg.content}", style=self.style("text"), markup=False)

    async def day_command(self: Repl, arg: str) -> None:
        engine = self.engine()
        s = self.chat.require_session()
        if arg.strip():
            days = await engine.resolve_dates(s, arg)  # the model reads "dün", "12 Ekim"…
        else:
            days = [(engine.today() - timedelta(days=1)).isoformat()]
        if not days:
            raise ChatError(self.t("day_usage"))
        digests = engine.repo.day_summaries(engine.scope(s), days)
        if not digests:
            self.info(self.t("day_none", days=", ".join(days[:3]) + ("…" if len(days) > 3 else "")))
            return
        for d in digests:
            self.heading(d.day or "")
            self.console.print(d.content, style=self.style("text"), markup=False)
