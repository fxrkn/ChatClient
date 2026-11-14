"""/key command: manage API keys from inside the app (mixed into the REPL)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from rich.table import Table

from chatclient.core.chat import ChatError
from chatclient.keys import SOURCE_KEYCHAIN, mask
from chatclient.providers.base import ProviderError
from chatclient.providers.registry import verify_key

if TYPE_CHECKING:
    from chatclient.ui.repl import Repl

KEY_SUBCOMMANDS = ("list", "set", "remove")


class KeyCommands:
    def key_providers(self: Repl) -> list[str]:
        reg = self.chat.registry
        return [n for n in reg.names() if n != "echo" and reg.needs_key(n)]

    async def key_command(self: Repl, arg: str) -> None:
        sub, _, name = arg.strip().partition(" ")
        name = name.strip()
        match sub.lower():
            case "" | "list":
                self.list_keys()
            case "set" | "add":
                await self.set_key(self.require_key_provider(name))
            case "remove" | "rm" | "delete":
                await self.remove_key(self.require_key_provider(name))
            case _:
                # "/key openrouter" is a shortcut for "/key set openrouter"
                await self.set_key(self.require_key_provider(sub))

    def require_key_provider(self: Repl, name: str) -> str:
        if not name:
            raise ChatError(self.t("key_usage", names=", ".join(self.key_providers())))
        if name not in self.chat.registry.configs or name == "echo":
            raise ChatError(self.t("key_unknown", name=name, names=", ".join(self.key_providers())))
        return name

    def list_keys(self: Repl) -> None:
        reg = self.chat.registry
        table = Table(box=None, header_style=self.style("accent", "bold"), pad_edge=False)
        for col in ("provider", "key", "from"):
            table.add_column(col)
        for name in self.key_providers():
            key, source = reg.keys.lookup(name, reg.configs[name])
            table.add_row(
                name,
                mask(key) if key else "—",
                self.t(f"key_src_{source}") if source else "",
            )
        self.console.print(table)
        self.info(self.t("key_list_hint"))

    async def set_key(self: Repl, name: str) -> None:
        pc = self.chat.registry.configs[name]
        question = self.t("key_prompt", name=name)
        try:
            if self.panel is not None:
                key = (await self.panel.ask(question, password=True)).strip()
            else:
                session: PromptSession[str] = PromptSession()
                key = (await session.prompt_async(question, is_password=True)).strip()
        except (KeyboardInterrupt, EOFError):
            key = ""
        if not key:
            self.info(self.t("aborted"))
            return
        self.info(self.t("key_checking"))
        try:
            await verify_key(name, pc, key)
        except ProviderError as e:
            if e.status in (401, 403):
                raise ChatError(self.t("key_rejected", name=name)) from e
            self.warn(self.t("key_unverified", error=str(e).split("\n")[0]))
        try:
            where = self.chat.registry.keys.save(name, key)
        except Exception as e:
            raise ChatError(self.t("key_save_failed", error=str(e))) from e
        await self.chat.registry.reset(name)
        if where == SOURCE_KEYCHAIN:
            self.info(self.t("key_saved_keychain", name=name))
        else:
            self.info(self.t("key_saved_file", name=name, path=str(self.chat.registry.keys.file)))
        if pc.api_key_env and os.environ.get(pc.api_key_env):
            self.warn(self.t("key_env_override", env=pc.api_key_env))

    async def remove_key(self: Repl, name: str) -> None:
        reg = self.chat.registry
        removed = reg.keys.remove(name)
        await reg.reset(name)
        self.info(self.t("key_removed" if removed else "key_nothing_removed", name=name))
        _key, source = reg.keys.lookup(name, reg.configs[name])
        if source:
            self.warn(self.t("key_still_from", source=self.t(f"key_src_{source}")))
