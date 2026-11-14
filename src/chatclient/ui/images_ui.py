"""/pic: pictures of the scene through a local ComfyUI (mixed into the REPL)."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import shutil
import sys
from collections.abc import Awaitable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from prompt_toolkit.application import get_app_or_none, run_in_terminal

from chatclient.core.chat import ChatError
from chatclient.images import (
    ComfyClient,
    ImageError,
    ImageMaker,
    Picture,
    analyze,
    card_look,
    card_pic_extra,
    describe,
    image_settings,
    load_workflow,
    pictures_of,
    plan_of,
    problem,
    rebuild,
    saved_prompt,
    set_card_look,
    set_card_pic_extra,
    tune,
    workflow_file,
)
from chatclient.providers.base import ProviderError
from chatclient.ui import sixel

if TYPE_CHECKING:
    from chatclient.ui.repl import Repl

log = logging.getLogger(__name__)
PIC_SUBCOMMANDS = ("setup", "info", "model", "lora", "vpred", "set", "look", "extra", "show",
                   "again", "auto", "inline")  # fmt: skip
INLINE_KEY = "images_inline"
AUTO_KEY = "images_auto"
HISTORY_PICTURES = 2  # shown again when a conversation is reopened


def _settings(text: str) -> dict[str, Any]:
    """``cfg=5 steps=28 sampler=euler scheduler=normal size=832x1216``"""
    out: dict[str, Any] = {}
    for part in text.replace(",", " ").split():
        key, _, value = part.partition("=")
        key = key.lower()
        if key in ("cfg",):
            out["cfg"] = float(value)
        elif key in ("steps",):
            out["steps"] = int(value)
        elif key in ("sampler", "scheduler"):
            out[key] = value
        elif key == "size" and "x" in value:
            w, h = value.lower().split("x", 1)
            out["width"], out["height"] = int(w), int(h)
        else:
            raise ChatError("/pic set cfg=5 steps=28 sampler=euler scheduler=normal size=832x1216")
    if not out:
        raise ChatError("/pic set cfg=5 steps=28 sampler=euler scheduler=normal size=832x1216")
    return out


class ImageCommands:
    """Mixin for :class:`Repl`."""

    def workflow_path(self: Repl) -> Path:
        return workflow_file(self.paths.data_dir)

    async def pic_command(self: Repl, arg: str) -> None:
        sub, _, rest = arg.strip().partition(" ")
        if sub.lower() == "setup":
            if rest.strip():
                self.pic_setup(rest)
            else:
                await self.pic_wizard()
        elif sub.lower() == "look":
            self.pic_look(rest)
        elif sub.lower() == "extra":
            char = self.chat.character()
            if char is None:
                raise ChatError(self.t("char_none"))
            tags = rest.strip()
            if tags:
                set_card_pic_extra(char.card, None if tags.lower() == "reset" else tags)
                self.chat.store.update_character(char.id, char.card)
            current = card_pic_extra(char.card) or self.t("pic_extra_unset")
            self.info(self.t("pic_extra", name=char.name, tags=current))
        elif sub.lower() == "inline":
            value = rest.strip().lower()
            if value:
                if value not in ("auto", "sixel", "off"):
                    raise ChatError("/pic inline auto|sixel|off")
                self.chat.store.kv_set(INLINE_KEY, value)
            mode = self.inline_setting()
            self.info(self.t("pic_inline", mode=mode, works=sixel.supported(mode)))
        elif sub.lower() == "show":
            path = self.last_picture()
            if path is None:
                raise ChatError(self.t("pic_none"))
            await self.show_picture(path, asked=True)
        elif sub.lower() == "again":
            source = self.last_picture()
            if source is None:
                raise ChatError(self.t("pic_none"))
            self.pic_start("", again=source)
        elif sub.lower() in ("info", "model", "lora", "vpred", "set"):
            await self.pic_workflow(sub.lower(), rest.strip())
        elif sub.lower() == "auto":
            value = rest.strip().lower()
            if value:
                if value not in ("on", "off"):
                    raise ChatError("/pic auto on|off")
                self.chat.store.kv_set(AUTO_KEY, value)
            self.info(self.t("pic_auto_on" if self.auto_pictures() else "pic_auto_off"))
        elif (near := difflib.get_close_matches(sub.lower(), PIC_SUBCOMMANDS, 1, 0.75)) and (
            rest.strip() or len(arg.split()) == 1
        ):
            # "/pic setuo …": a typo, not something to draw
            raise ChatError(self.t("pic_did_you_mean", cmd=f"/pic {near[0]} {rest}".strip()))
        else:
            self.pic_start(arg.strip())

    # ── /pic setup: the wizard ───────────────────────────────────────────────

    async def _choose(self: Repl, question: str, options: list[str], default: int = 1) -> int:
        """A numbered choice; returns 1-based. Enter takes the default."""
        self.heading(question)
        for i, label in enumerate(options, 1):
            self.console.print(f"  {i}) {label}", markup=False)
        while True:
            answer = (await self.ask(self.t("pic_wiz_pick", n=default))).strip()
            if not answer:
                return default
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return int(answer)

    async def pic_wizard(self: Repl) -> None:
        """Local ComfyUI or a cloud model, step by step; nothing needs exporting."""
        t = self.t
        try:
            where = await self._choose(t("pic_wiz_where"), [t("pic_wiz_local"), t("pic_wiz_cloud")])
            if where == 2:
                await self._wizard_cloud()
            else:
                await self._wizard_local()
            auto = (await self.ask(t("pic_wiz_auto"))).strip().lower()
            self.chat.store.kv_set(AUTO_KEY, "on" if auto in ("y", "yes", "e", "evet") else "off")
        except (KeyboardInterrupt, EOFError):
            self.info(t("aborted"))
            return
        self.info(t("pic_wiz_done"))
        if self.chat.character() is not None and await self.confirm(t("pic_wiz_test")):
            self.pic_start("")

    async def _wizard_cloud(self: Repl) -> None:
        from chatclient.core.free_models import find_openrouter
        from chatclient.images import CloudImages, save_settings

        name = find_openrouter(self.chat.registry)
        if name is None or not self.chat.registry.get(name).api_key:
            raise ChatError(self.t("pic_wiz_no_key"))
        self.info(self.t("pic_wiz_cloud_note"))
        try:
            models = await CloudImages(self.chat.registry.get(name), "").models()
        except ImageError as e:
            raise ChatError(str(e)) from e
        if not models:
            raise ChatError(self.t("pic_wiz_no_models"))
        n = await self._choose(self.t("pic_wiz_cloud_model"), models)
        save_settings(self.chat.store, backend="cloud", cloud_model=f"{name}:{models[n - 1]}",
                      style="natural")  # fmt: skip

    async def _wizard_local(self: Repl) -> None:
        from chatclient.images import basic_workflow, save_settings

        cfg = image_settings(self.chat)
        url = (await self.ask(self.t("pic_wiz_url"), cfg.url)).strip() or cfg.url
        comfy = ComfyClient(url)
        try:
            checkpoints = await comfy.available("CheckpointLoaderSimple", "ckpt_name")
        except ImageError as e:
            raise ChatError(self.t("pic_wiz_no_comfy", url=url)) from e
        own = await self._choose(self.t("pic_wiz_workflow"),
                                 [self.t("pic_wiz_simple"), self.t("pic_wiz_own")])  # fmt: skip
        save_settings(self.chat.store, backend="comfyui", url=url, style="tags")
        if own == 2:
            path = await self.ask(self.t("pic_wiz_own_path"))
            self.pic_setup(path)
            return
        if not checkpoints:
            raise ChatError(self.t("pic_wiz_no_ckpt"))
        ckpt = checkpoints[await self._choose(self.t("pic_models"), checkpoints) - 1]
        family = await self._choose(self.t("pic_wiz_family"),
                                    [self.t("pic_wiz_sd15"), self.t("pic_wiz_sdxl")])  # fmt: skip
        graph = basic_workflow(ckpt, "sd15" if family == 1 else "sdxl")
        vpred = (await self.ask(self.t("pic_wiz_vpred"))).strip().lower()
        if vpred in ("y", "yes", "e", "evet"):
            graph = rebuild(graph, [], vpred=True)
        dest = self.workflow_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
        d = describe(graph)
        self.info(self.t("pic_info", ckpt=d["checkpoint"], loras="-",
                         vpred="on" if d["vpred"] else "off", cfg=d["cfg"], steps=d["steps"],
                         sampler=d["sampler"], scheduler=d["scheduler"], size=d["size"]))  # fmt: skip

    def pic_setup(self: Repl, raw: str) -> None:
        """/pic setup <file>: the user's own workflow (ComfyUI's Save (API Format))."""
        src = Path(raw.strip().strip('"').strip("'")).expanduser()
        if not raw.strip():
            raise ChatError(self.t("pic_setup_usage"))
        try:
            wf = load_workflow(src)
        except ImageError as e:
            raise ChatError(str(e)) from e
        dest = self.workflow_path()
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)  # a copy: the original may move or change
        from chatclient.images import save_settings

        save_settings(self.chat.store, backend="comfyui")
        ckpt = next((n["inputs"].get("ckpt_name") for n in wf.graph.values()
                     if isinstance(n, dict) and "ckpt_name" in n.get("inputs", {})), "?")  # fmt: skip
        self.info(self.t("pic_setup_done", model=ckpt))
        sampler = wf.graph[wf.sampler]["inputs"]
        cfg, steps = sampler.get("cfg"), sampler.get("steps")
        if isinstance(cfg, (int, float)) and (cfg > 12 or cfg < 2):
            # outside what most SD checkpoints are made for: burnt, or ignoring the prompt
            self.warn(self.t("pic_cfg_odd", cfg=cfg, steps=steps))

    # ── editing the workflow without ComfyUI's node editor ─────────────────

    async def pic_workflow(self: Repl, what: str, arg: str) -> None:
        path = self.workflow_path()
        if image_settings(self.chat).backend == "cloud":
            raise ChatError(self.t("pic_cloud_no_workflow"))
        if not path.is_file():
            raise ChatError(self.t("pic_setup_usage"))
        graph = json.loads(path.read_text(encoding="utf-8-sig"))
        comfy = ComfyClient(self.chat.config.images.url)
        try:
            if what == "model":
                graph = tune(graph, checkpoint=await self._pick(
                    comfy, "CheckpointLoaderSimple", "ckpt_name", arg, "pic_models"))  # fmt: skip
            elif what == "lora":
                loras, vpred = plan_of(graph)
                if arg.lower() in ("off", "none", "reset"):
                    loras = []
                elif arg:
                    *words, last = arg.split()
                    strength = 0.8
                    try:
                        strength = float(last)
                    except ValueError:
                        words.append(last)
                    name = await self._pick(comfy, "LoraLoader", "lora_name", " ".join(words),
                                            "pic_loras")  # fmt: skip
                    loras = [x for x in loras if x[0] != name] + [(name, strength)]
                elif not loras:
                    await self._pick(comfy, "LoraLoader", "lora_name", "", "pic_loras")
                    return
                graph = rebuild(graph, loras, vpred)
            elif what == "vpred":
                loras, _ = plan_of(graph)
                if arg.lower() not in ("on", "off"):
                    raise ChatError("/pic vpred on|off")
                graph = rebuild(graph, loras, arg.lower() == "on")
            elif what == "set":
                graph = tune(graph, **_settings(arg))
        except ImageError as e:
            raise ChatError(str(e)) from e
        if what != "info":
            analyze(graph)  # still a usable workflow
            path.write_text(json.dumps(graph, ensure_ascii=False, indent=1), encoding="utf-8")
        d = describe(graph)
        loras = ", ".join(f"{n} ({s:g})" for n, s in d["loras"]) or "-"
        self.info(self.t("pic_info", ckpt=d["checkpoint"], loras=loras,
                         vpred="on" if d["vpred"] else "off", cfg=d["cfg"], steps=d["steps"],
                         sampler=d["sampler"], scheduler=d["scheduler"], size=d["size"]))  # fmt: skip

    async def _pick(self: Repl, comfy: ComfyClient, node: str, field: str, ref: str,
                    title: str) -> str:  # fmt: skip
        """A name ComfyUI has: by number or part of the name; without one, list them."""
        names = await comfy.available(node, field)
        if not ref:
            self.heading(self.t(title))
            for i, n in enumerate(names, 1):
                self.console.print(f"  {i}) {n}", markup=False)
            raise ChatError(self.t("pic_pick_hint"))
        if ref.isdigit() and 1 <= int(ref) <= len(names):
            return names[int(ref) - 1]
        hits = [n for n in names if ref.lower() in n.lower()]
        if len(hits) != 1:
            raise ChatError(self.t("pic_pick_none", ref=ref, n=len(hits)))
        return hits[0]

    def pic_look(self: Repl, text: str) -> None:
        """Show or set the character's fixed look (used in every picture)."""
        char = self.chat.character()
        if char is None:
            raise ChatError(self.t("char_none"))
        text = text.strip()
        if text:
            set_card_look(char.card, None if text.lower() == "reset" else text)
            self.chat.store.update_character(char.id, char.card)
        look = card_look(char.card)
        self.info(self.t("pic_look", name=char.name, look=look or self.t("pic_look_unset")))

    def pic_start(self: Repl, wish: str, again: Path | None = None) -> None:
        if problem(self.chat, self.workflow_path()) is not None:  # now, not after "drawing…"
            raise ChatError(self.t("pic_setup_usage"))
        if self.chat.character() is None:
            raise ChatError(self.t("char_none"))
        if self.pic_running():
            raise ChatError(self.t("pic_busy"))
        maker = ImageMaker(self.chat, self.workflow_path())
        self.info(self.t("pic_drawing"))
        job = maker.again(again) if again is not None else maker.draw(wish)
        self.pic_task = asyncio.ensure_future(self._draw(job))

    def pic_running(self: Repl) -> bool:
        return self.pic_task is not None and not self.pic_task.done()

    # ── auto mode: a picture when the model sees a new moment ───────────────

    def auto_pictures(self: Repl) -> bool:
        return self.chat.store.kv_get(AUTO_KEY) == "on"

    def after_reply_picture(self: Repl) -> None:
        """Called after each reply. Cheap when off; when on, the decision and the drawing
        happen in the background, so the chat never waits."""
        if not self.auto_pictures() or self.pic_running() or self.chat.character() is None:
            return
        if problem(self.chat, self.workflow_path()) is not None:
            return
        self.pic_task = asyncio.ensure_future(self._auto_picture())

    async def _auto_picture(self: Repl) -> None:
        maker = ImageMaker(self.chat, self.workflow_path())
        last = self.last_picture()
        try:
            if not await maker.worth_drawing(saved_prompt(last) if last else ""):
                return
        except (ImageError, ProviderError, ChatError) as e:
            log.info("auto picture skipped: %s", e)
            return
        await self._draw(maker.draw(""))

    async def _draw(self: Repl, job: Awaitable[Picture]) -> None:
        """In the background: chatting goes on, the result is announced when it's ready."""
        try:
            pic = await job
        except (ImageError, ProviderError, ChatError) as e:
            await self.announce(self.t("pic_failed", error=str(e).split("\n")[0]), warn=True)
            return
        except Exception as e:  # never break the chat
            log.exception("picture failed")
            await self.announce(self.t("pic_failed", error=f"{type(e).__name__}: {e}"), warn=True)
            return
        log.info("picture prompt: %s", json.dumps(pic.prompt, ensure_ascii=False))
        await self.announce(self.t("pic_ready", path=str(pic.path), s=f"{pic.seconds:.0f}"))
        await self.show_picture(pic.path)

    async def show_picture(self: Repl, path: Path, asked: bool = False) -> None:
        """In the terminal when it can draw (Sixel); otherwise in the image viewer.
        In the panel layout the panel shows a preview; /pic show opens the picture."""
        cfg = self.chat.config.images
        if self.panel is not None:
            self.panel.refresh()
            if asked or cfg.open:
                from chatclient.ui.tools_ui import open_in_file_manager

                open_in_file_manager(path)
            return
        inline = sixel.supported(self.inline_setting())
        if inline:
            try:
                data = await asyncio.to_thread(sixel.encode, path, cfg.inline_width)
            except Exception:
                log.exception("could not encode %s", path)
                inline = False
            else:

                def draw() -> None:
                    sys.stdout.write(data + "\n")
                    sys.stdout.flush()

                if get_app_or_none() is not None and not self.streaming:
                    await run_in_terminal(draw)
                else:
                    draw()
        if cfg.open or not inline:
            from chatclient.ui.tools_ui import open_in_file_manager

            open_in_file_manager(path)

    def inline_setting(self: Repl) -> str:
        """/pic inline (saved in the app) wins over [images] inline."""
        saved = self.chat.store.kv_get(INLINE_KEY)
        return saved if saved in ("auto", "sixel", "off") else self.chat.config.images.inline

    def last_picture(self: Repl) -> Path | None:
        """The newest picture tied to the visible conversation."""
        char = self.chat.character()
        if char is None:
            return None
        found = pictures_of(self.chat.store, char, self.chat.history(limit=40))
        return found[-1][1] if found else None

    def history_pictures(self: Repl, msgs: list[Any]) -> dict[int, Path]:
        """Pictures to show again when a conversation is reopened: the latest ones only
        (drawing many would make reopening slow)."""
        char = self.chat.character()
        if char is None or self.panel is not None or not sixel.supported(self.inline_setting()):
            return {}  # (the panel shows the latest picture by itself)
        found = pictures_of(self.chat.store, char, msgs)[-HISTORY_PICTURES:]
        return {m.id: path for m, path in found}

    def draw_now(self: Repl, path: Path) -> None:
        """Inline, right where it is called (history; nothing is being typed then)."""
        try:
            sys.stdout.write(sixel.encode(path, self.chat.config.images.inline_width) + "\n")
            sys.stdout.flush()
        except Exception:
            log.exception("could not show %s", path)

    async def announce(self: Repl, text: str, warn: bool = False) -> None:
        """Print without breaking the line being typed."""
        show = (lambda: self.warn(text)) if warn else (lambda: self.info(text))
        if self.panel is not None:  # the chat area is not where the typing happens
            show()
        elif get_app_or_none() is not None and not self.streaming:
            await run_in_terminal(show)
        else:
            show()
