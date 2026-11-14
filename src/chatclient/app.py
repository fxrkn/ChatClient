"""Entry point: argument parsing, wiring, and the interactive or one-shot run."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import getpass
import logging
import os
import sys
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

from chatclient import __version__
from chatclient.config import Config, ConfigError, load_config
from chatclient.core.chat import ChatError, ChatService, TurnResult
from chatclient.i18n import LANG_KEY, Translator
from chatclient.keys import SOURCE_KEYCHAIN, KeyStore
from chatclient.library import Library
from chatclient.memory.engine import MemoryEngine
from chatclient.paths import AppPaths, resolve_paths
from chatclient.providers.base import ProviderError
from chatclient.providers.registry import ProviderRegistry, verify_key
from chatclient.storage import Store
from chatclient.timeutil import resolve_tz


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="chatclient",
        description="Terminal AI chat for conversation, storytelling and roleplay.",
    )
    p.add_argument("--version", action="version", version=f"chatclient {__version__}")
    p.add_argument("-m", "--model", help="model to use, as provider:model")
    p.add_argument("-s", "--session", type=int, help="open session by id")
    p.add_argument("-n", "--new", action="store_true", help="start a new session")
    p.add_argument("-c", "--char", help="start a new session with this character (name or id)")
    p.add_argument("-p", "--prompt", help="send one message, print the reply, and exit")
    p.add_argument("--home", help="keep config and data in this directory")
    p.add_argument("--lang", choices=["auto", "tr", "en"], help="interface language")
    p.add_argument("--set-key", metavar="PROVIDER", help="store an API key in the OS keychain")
    p.add_argument("--debug", action="store_true", help="verbose logging to the log file")
    p.add_argument(
        "--eval-roleplay",
        action="store_true",
        help="play scripted chats with a real model and score the replies (uses -m and --judge)",
    )
    p.add_argument("--judge", help="model that grades --eval-roleplay replies (provider:model)")
    p.add_argument("--compare", help="an earlier --eval-roleplay result file to compare with")
    p.add_argument("--no-memory", action="store_true", help="--eval-roleplay without memory")
    p.add_argument("--regrade", help="grade an earlier --eval-roleplay result again (--judge)")
    p.add_argument(
        "--serve",
        action="store_true",
        help="run headless for another front end: JSON lines on stdin/stdout (docs/PROTOCOL.md)",
    )
    p.add_argument(
        "--eval-memory",
        action="store_true",
        help="measure the memory brief with a real model (uses -m, else the utility/chat model)",
    )
    return p.parse_args(argv)


def setup_logging(paths: AppPaths, debug: bool) -> None:
    handler = RotatingFileHandler(
        paths.log_file, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    for noisy in ("httpx", "httpcore", "httpx2", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    args = parse_args(argv)
    paths = resolve_paths(args.home)
    paths.ensure()
    setup_logging(paths, args.debug)

    created = not paths.config_file.exists()
    try:
        config = load_config(paths.config_file)
    except ConfigError as e:
        print(e, file=sys.stderr)
        return 2
    t = Translator(args.lang or config.ui.language)

    if args.set_key:
        return set_key(args.set_key, config, KeyStore(paths.key_file))
    if args.eval_memory:
        return asyncio.run(eval_memory(args, config, paths))
    if args.eval_roleplay or args.regrade:
        return asyncio.run(eval_roleplay(args, config, paths))

    notes = [t("config_created", path=paths.config_file)] if created else []
    try:
        return asyncio.run(run(args, config, paths, t, notes))
    except KeyboardInterrupt:
        return 130


def set_key(provider: str, config: Config, keys: KeyStore) -> int:
    if provider not in config.providers:
        print(f"unknown provider '{provider}' (configured: {', '.join(config.providers)})")
        return 2
    pc = config.providers[provider]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", getpass.GetPassWarning)
            key = getpass.getpass(f"API key for {provider} (input hidden): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        key = ""
    if not key:
        print("no key entered")
        return 1
    try:
        asyncio.run(verify_key(provider, pc, key))
    except ProviderError as e:
        if e.status in (401, 403):
            print(f"the key was rejected by {provider}; nothing was saved")
            return 1
        print(f"could not verify the key ({e}); saving it anyway")
    try:
        where = keys.save(provider, key)
    except Exception as e:
        env = pc.api_key_env or f"{provider.upper()}_API_KEY"
        print(f"could not store the key: {e}")
        print(f"use an environment variable instead, e.g. {env}=<your key>")
        return 1
    place = "the OS keychain" if where == SOURCE_KEYCHAIN else str(keys.file)
    print(f"saved the key for {provider} in {place}")
    if pc.api_key_env and os.environ.get(pc.api_key_env):
        print(f"note: the {pc.api_key_env} environment variable is set and takes priority")
    return 0


async def eval_memory(args: argparse.Namespace, config: Config, paths: AppPaths) -> int:
    from chatclient.memory.evaluate import run_eval

    model = args.model or config.models.utility or config.models.chat
    if not model:
        store = Store(paths.db_file)
        model = store.kv_get("last_model") or ""
        store.close()
    if not model:
        print("error: no model; use -m provider:model", file=sys.stderr)
        return 1
    print(f"Evaluating the memory brief with {model}…")
    rate = await run_eval(config, model, paths.data_dir / "models")
    return 0 if rate >= 0.5 else 1


async def open_app(
    args: argparse.Namespace, config: Config, paths: AppPaths, t: Translator, notes: list[str]
) -> ChatService:
    """Everything a front end needs: files synced, memory started, a session open and a
    model chosen. Raises ChatError/ValueError with a message for the user."""
    tz = resolve_tz(config.memory.timezone)
    library = Library(paths.data_dir)
    store = Store(paths.db_file, tz=tz, day_boundary=config.memory.day_boundary, library=library)
    library.sync(store)  # files are the source of truth; quick when nothing changed
    if not args.lang and (saved_lang := store.kv_get(LANG_KEY)):
        t.set_language(saved_lang)  # chosen in the app with /lang
    registry = ProviderRegistry(config, KeyStore(paths.key_file))
    memory = MemoryEngine(config, store, registry) if config.memory.enabled else None
    if memory is not None:
        await memory.start_semantic(paths.data_dir / "models")
    chat = ChatService(config, store, registry, memory)
    chat.seed_builtin_characters()
    if args.char:
        char = store.find_character(args.char, "ai")
        if char is None:
            raise ChatError(f"no character matches {args.char!r}")
        chat.new_session(character_id=char.id)
    elif args.session:
        chat.open_session(args.session)
    elif args.new:
        chat.new_session()
    else:
        chat.resume_or_create()

    if args.model:
        chat.set_model(args.model)
    elif chat.require_session().model is None:
        default = chat.default_model()
        if default:
            chat.set_model(default)
        else:
            ref = await registry.autodetect()
            if ref:
                chat.set_model(str(ref))
                notes.append(t("autodetected", model=str(ref)))
            else:
                notes.append(t("no_local"))
    return chat


async def close_app(chat: ChatService) -> None:
    await chat.registry.aclose()
    chat.store.close()


async def eval_roleplay(args: argparse.Namespace, config: Config, paths: AppPaths) -> int:
    from datetime import datetime

    from chatclient.evaluate_roleplay import regrade, run_roleplay_eval

    store = Store(paths.db_file)
    last = store.kv_get("last_model") or ""
    store.close()
    chat = args.model or config.models.chat or last
    judge = args.judge or config.models.utility or chat
    if not chat:
        print("error: no model; use -m provider:model", file=sys.stderr)
        return 1
    evals = paths.data_dir / "evals"
    out = evals / f"roleplay-{datetime.now():%Y%m%d-%H%M%S}.json"

    def find(name: str) -> Path:
        p = Path(name)
        return p if p.is_file() else evals / name

    compare = find(args.compare) if args.compare else None
    if args.regrade:
        await regrade(config, judge, find(args.regrade), out, compare)
        return 0
    await run_roleplay_eval(
        config, chat, judge, paths.data_dir / "models", out, compare, memory=not args.no_memory
    )
    return 0


async def run(
    args: argparse.Namespace, config: Config, paths: AppPaths, t: Translator, notes: list[str]
) -> int:
    try:
        chat = await open_app(args, config, paths, t, notes)
    except (ChatError, ProviderError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2 if isinstance(e, ValueError) else 1
    try:
        if args.serve:
            from chatclient.serve import serve_stdio

            await serve_stdio(chat, notes)
            return 0

        if args.prompt is not None:
            return await one_shot(chat, args.prompt, t)

        from chatclient.ui.render import enable_windows_vt
        from chatclient.ui.repl import run_repl

        enable_windows_vt()
        await run_repl(chat, t, paths, notes, config.ui.show_stats)
        return 0
    except (ChatError, ProviderError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        await close_app(chat)


async def one_shot(chat: ChatService, prompt: str, t: Translator) -> int:
    if chat.require_session().model is None:
        print(t("no_model"), file=sys.stderr)
        return 1
    result = TurnResult()
    try:
        async for ev in chat.send(prompt, result):
            if ev.text:
                sys.stdout.write(ev.text)
                sys.stdout.flush()
    finally:
        sys.stdout.write("\n")
    if os.environ.get("CHATCLIENT_STATS"):
        print(f"[{result.model} · {result.elapsed:.2f}s]", file=sys.stderr)
    return 0
