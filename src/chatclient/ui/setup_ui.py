"""First-run setup: get a new user from "just installed" to chatting, without config files."""

from __future__ import annotations

from typing import TYPE_CHECKING

from chatclient.characters import Card
from chatclient.core.chat import ChatError
from chatclient.core.free_models import find_openrouter
from chatclient.providers.base import ProviderError
from chatclient.providers.registry import is_local
from chatclient.ui.picker import PickItem

if TYPE_CHECKING:
    from chatclient.ui.repl import Repl

ONBOARDED_KEY = "onboarded"
SUGGESTED_LOCAL_MODEL = "qwen3.5:4b"


class SetupCommands:
    def needs_setup(self: Repl) -> bool:
        """Only brand-new users without any model see the wizard automatically."""
        if self.chat.store.kv_get(ONBOARDED_KEY):
            return False
        if self.chat.require_session().model or self.chat.default_model():
            self.chat.store.kv_set(ONBOARDED_KEY, "1")  # existing user: never nag
            return False
        return True

    async def setup_wizard(self: Repl) -> None:
        t = self.t
        self.heading(t("setup_title"))
        self.info(t("setup_intro"))
        try:
            choice = await self.pick(
                t("setup_how"),
                [
                    PickItem("local", t("setup_local"), t("setup_local_hint")),
                    PickItem("cloud", t("setup_cloud"), t("setup_cloud_hint")),
                    PickItem("skip", t("setup_skip")),
                ],
            )
            if choice == "local":
                await self.setup_local()
            elif choice == "cloud":
                await self.setup_cloud()
            if choice in ("local", "cloud") and self.chat.require_session().model:
                await self.setup_persona_and_character()
        except (KeyboardInterrupt, EOFError):
            self.info(t("aborted"))
        except (ChatError, ProviderError) as e:
            self.warn(str(e).split("\n")[0])
        self.chat.store.kv_set(ONBOARDED_KEY, "1")
        self.info(t("setup_done"))

    async def setup_local(self: Repl) -> None:
        reg = self.chat.registry
        local = [n for n, pc in reg.configs.items() if pc.type != "echo" and is_local(pc)]
        for attempt in range(2):
            self.info(self.t("setup_looking"))
            found = await reg.list_models(local, timeout=3.0)
            options = [
                PickItem(f"{name}:{m}", m, name)
                for name in local
                if isinstance(models := found[name], list)
                for m in models
                if "embed" not in m.lower()
            ]
            if options:
                spec = await self.pick(self.t("pick_model"), options)
                if spec:
                    self.chat.set_model(spec)
                    self.info(self.t("model_set", model=spec))
                return
            self.warn(self.t("setup_no_local"))
            self.info(self.t("setup_install_ollama", model=SUGGESTED_LOCAL_MODEL))
            if attempt == 0 and not await self.confirm(self.t("setup_retry_q")):
                return

    async def setup_cloud(self: Repl) -> None:
        name = find_openrouter(self.chat.registry)
        if name is None:
            raise ChatError(self.t("free_no_openrouter"))
        if not self.chat.registry.get(name).api_key:
            self.info(self.t("setup_key_info"))
            await self.set_key(name)
        if not self.chat.registry.get(name).api_key:
            return  # cancelled or rejected
        await self.free_command("")

    async def setup_persona_and_character(self: Repl) -> None:
        t = self.t
        if self.chat.persona() is None:
            name = await self.ask(t("setup_your_name"))
            if name:
                self.chat.set_persona(Card(name=name))
        chars = self.chat.store.list_characters("ai")
        if chars:
            if await self.confirm(t("setup_pick_char_q")):
                char = await self.pick_char()
                if char is not None:
                    self.start_char(char)
        elif await self.confirm(t("setup_create_char_q")):
            await self.create_char()
