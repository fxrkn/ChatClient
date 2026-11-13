"""Configuration: TOML file validated with Pydantic.

Layering (lowest to highest priority): built-in defaults → config.toml →
provider settings → per-session overrides (stored in the database, set via /set).
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from chatclient.theme import COLOR_KEYS, DEFAULT_THEME, PRESETS, valid_color

log = logging.getLogger(__name__)

KEYRING_SERVICE = "chatclient"

DEFAULT_SYSTEM_PROMPT = (
    "You are a warm, thoughtful conversation partner and a skilled storyteller. "
    "Always reply in the same language the user writes in, unless they ask otherwise. "
    "Stay consistent with everything established earlier in the conversation: names, "
    "places, events, relationships and the user's preferences. "
    "Write naturally; avoid needless disclaimers."
)

ProviderType = Literal["openai_compat", "ollama", "anthropic", "gemini", "echo"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProviderConfig(_Strict):
    type: ProviderType
    base_url: str | None = None
    # API key sources, checked in this order: env var → OS keychain → literal (discouraged)
    api_key_env: str | None = None
    api_key: str | None = None
    headers: dict[str, str] = Field(default_factory=dict)
    timeout: float = 120.0
    context_tokens: int | None = None  # overrides [context].max_tokens for this provider
    # Reply-length cap when the session doesn't set one. Cloud services reserve credit
    # for the cap, so sending none (= the model's maximum) can fail on small balances.
    max_tokens: int | None = None
    # openai_compat
    stream_usage: bool = True
    # ollama
    num_ctx: int | None = None
    keep_alive: str | None = "30m"
    # Reasoning ("thinking") for models that support it. Off by default: in chat and
    # roleplay it mostly adds a long silent pause before every reply.
    think: bool | None = False
    # anthropic
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    refusal_fallback: bool = True


class GenerationConfig(_Strict):
    temperature: float | None = 0.8
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None  # None = provider default
    repetition_penalty: float | None = None
    seed: int | None = None
    stop: list[str] | None = None


class ContextConfig(_Strict):
    max_tokens: int = 8192  # prompt budget (system prompt + history)


class ModelsConfig(_Strict):
    chat: str = ""  # "provider:model"; empty = last used, else auto-detect
    utility: str = ""  # used later for summaries / memory extraction
    free_fallback: bool = True  # if a free OpenRouter model is busy, switch to another one
    refusal_fallback: str = ""  # "provider:model" that rewrites a reply the chat model declined


class UIConfig(_Strict):
    language: Literal["auto", "tr", "en"] = "en"
    show_stats: bool = True
    theme: str = DEFAULT_THEME  # soft | warm | nord | rose | light | classic
    colors: dict[str, str] = Field(default_factory=dict)  # per-colour overrides of the theme
    clear_on_start: bool = True
    max_width: int = 90  # reading width for replies; long lines are tiring to read
    style_actions: bool = True  # show *actions* in italics without the asterisks
    style_speech: bool = True  # "quoted speech" in a slightly brighter colour
    character_colors: bool = True  # each character's name in its own colour
    typing_indicator: bool = True  # "<name> is typing… 2.1s" while waiting for a reply

    @field_validator("theme")
    @classmethod
    def _check_theme(cls, v: str) -> str:
        if v not in PRESETS:
            raise ValueError(f"unknown theme {v!r} (available: {', '.join(PRESETS)})")
        return v

    @field_validator("colors")
    @classmethod
    def _check_colors(cls, v: dict[str, str]) -> dict[str, str]:
        for key, value in v.items():
            if key not in COLOR_KEYS:
                raise ValueError(f"unknown colour {key!r} (known: {', '.join(COLOR_KEYS)})")
            if not valid_color(value):
                raise ValueError(f"{key} = {value!r}: use '#rrggbb' or 'default'")
        return v

    @field_validator("max_width")
    @classmethod
    def _check_width(cls, v: int) -> int:
        if v < 30:
            raise ValueError("max_width must be at least 30")
        return v


class MemoryConfig(_Strict):
    timezone: str = "auto"  # "auto" = system timezone, or an IANA name like "Europe/Istanbul"
    day_boundary: str = "04:00"  # a chat at 01:30 still belongs to the previous day
    enabled: bool = True  # long-term memory: summaries, remembered facts, scene
    scope: Literal["character", "session"] = (
        "character"  # share memories across a character's chats
    )
    every_exchanges: int = Field(default=1, ge=1)  # memory pass after this many replies
    embeddings: str = "auto"  # semantic search: auto | off | fastembed[:model] | ollama:<model>
    brief: bool = True  # the utility model reads each message before the reply (smart recall)
    brief_timeout: float = Field(default=20.0, gt=0)  # seconds; then recall goes on without it
    keep_recent_messages: int = Field(default=12, ge=2)  # never summarised away / recalled twice
    summarize_after_tokens: int = Field(default=1500, ge=200)  # older text before a summary
    recall_tokens: int = Field(default=600, ge=0)  # budget for recalled memories per turn
    summary_tokens: int = Field(default=900, ge=0)  # budget for the "story so far"

    @field_validator("day_boundary")
    @classmethod
    def _check_boundary(cls, v: str) -> str:
        hh, _, mm = v.partition(":")
        if not (hh.isdigit() and mm.isdigit() and 0 <= int(hh) < 24 and 0 <= int(mm) < 60):
            raise ValueError("day_boundary must look like HH:MM")
        return v


class RoleplayConfig(_Strict):
    rules: str | None = None  # replaces the built-in narration rules for character chats
    reminder: bool = True  # short in-character note next to the newest message
    stop_at_user_turn: bool = True  # stop generating when the model starts writing as you
    content: Literal["standard", "mature"] = "standard"  # mature: adult fiction, no lectures
    rewrite_repeats: bool = True  # a reply mostly recycled from earlier ones is rewritten once


class Config(_Strict):
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    roleplay: RoleplayConfig = Field(default_factory=RoleplayConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    @field_validator("providers")
    @classmethod
    def _check_names(cls, v: dict[str, ProviderConfig]) -> dict[str, ProviderConfig]:
        for name in v:
            if ":" in name or not name.strip():
                raise ValueError(f"invalid provider name {name!r} (no ':' allowed)")
            if name == "echo":
                raise ValueError("'echo' is a reserved built-in provider name")
        return v


class ConfigError(Exception):
    pass


DEFAULT_CONFIG_TOML = '''\
# ChatClient configuration
# Models are addressed as "provider:model", e.g. "ollama:llama3.1:8b" or "anthropic:claude-opus-5-5".
# A built-in "echo" provider (model "echo:echo") is always available for offline testing.

# system_prompt = """..."""   # uncomment to replace the default system prompt

[ui]
language = "en"          # en | tr | auto (= system language); the AI answers in your language
show_stats = true        # show token count and speed after each reply
theme = "soft"           # soft | warm | nord | rose | light | classic  (try them live with /theme)
clear_on_start = true    # start with a clean screen
max_width = 90           # reading width of replies (characters)
style_actions = true     # *actions* shown in italics, without the asterisks
style_speech = true      # "quoted speech" a touch brighter than narration
character_colors = true  # each character's name in its own colour (/char color to pick one)
typing_indicator = true  # "<name> is typing… 2.1s" while waiting
# The font itself is set in your terminal app (Windows Terminal → Settings → Profile →
# Appearance). Colours can be fine-tuned here, as "#rrggbb" or "default":
# [ui.colors]
# text = "#d6d3d1"       # reply text
# accent = "#93c5fd"     # character name, prompt arrow
# user = "#a7c4a0"       # your name
# action = "#a8a29e"     # *actions*
# speech = "#f0e6d6"     # "quoted speech"
# dim = "#78716c"        # info lines and stats
# bar_bg = "#262626"     # bottom bar background
# bar_fg = "#a8a29e"     # bottom bar text

[models]
chat = ""                # empty = last used model, otherwise auto-detect a local server
utility = ""             # model for background memory work; empty = the chat model
free_fallback = true     # a busy free OpenRouter model is swapped for a working one (see /free)
refusal_fallback = ""    # model that rewrites a declined reply, e.g. a local one (see /fallback)

[generation]             # defaults; override per session with /set
temperature = 0.8
# top_p = 0.95
# top_k = 40
# min_p = 0.05
# max_tokens = 1024      # unset = provider default
# repetition_penalty = 1.1
# seed = 42

[context]
max_tokens = 8192        # prompt budget for system prompt + history

[roleplay]               # character chats (/char)
reminder = true          # in-character note next to your newest message (keeps small models on track)
stop_at_user_turn = true # cut the reply if the model starts writing your lines
content = "standard"     # "mature": adult fiction, no lectures or disclaimers (see /content)
rewrite_repeats = true   # a reply that mostly repeats earlier ones is rewritten once, automatically
# rules = """..."""      # replace the built-in narration rules

[memory]
enabled = true           # long-term memory: summaries, remembered facts, scene state
scope = "character"      # character = a character remembers you across all your chats with it
every_exchanges = 1      # update memory after every N replies (runs in the background)
embeddings = "auto"      # meaning-based recall: auto (local model, ~0.2 GB once) | off
brief = true             # the utility model reads each message (dates, what's relevant,
                         # fact checks); prepared while you type. false = faster, simpler
recall_tokens = 600      # space for recalled memories in each prompt
summary_tokens = 900     # space for the "story so far"
timezone = "auto"        # or e.g. "Europe/Istanbul"
day_boundary = "04:00"

# ── Local providers ──────────────────────────────────────────────────────────
[providers.ollama]
type = "ollama"
base_url = "http://localhost:11434"
num_ctx = 8192           # Ollama's own default is small; keep >= [context].max_tokens
keep_alive = "30m"
think = false            # reasoning models (Qwen3, DeepSeek-R1…): false = much faster replies

[providers.lmstudio]
type = "openai_compat"
base_url = "http://localhost:1234/v1"

[providers.llamacpp]
type = "openai_compat"
base_url = "http://localhost:8080/v1"

# ── Cloud providers (keys: env var, or `chatclient --set-key NAME` for the OS keychain) ──
[providers.openai]
type = "openai_compat"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"
context_tokens = 32000

[providers.openrouter]
type = "openai_compat"
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
context_tokens = 32000

[providers.anthropic]
type = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
context_tokens = 32000
effort = "low"           # low = fastest replies; raise to medium/high for richer writing
refusal_fallback = true  # retry declined requests on a fallback model (newest Claude models)

[providers.gemini]
type = "gemini"
api_key_env = "GEMINI_API_KEY"
context_tokens = 32000

# [providers.groq]
# type = "openai_compat"
# base_url = "https://api.groq.com/openai/v1"
# api_key_env = "GROQ_API_KEY"
'''


def load_config(path: Path, *, create: bool = True) -> Config:
    """Load config from ``path``; write the commented default template if missing."""
    if not path.exists():
        if not create:
            return Config()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(DEFAULT_CONFIG_TOML, encoding="utf-8")
        log.info("wrote default config to %s", path)
    try:
        # utf-8-sig: Windows editors (Notepad, PowerShell) often prepend a BOM.
        data = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from e
    try:
        return Config.model_validate(data)
    except ValidationError as e:
        lines = [f"{path}: invalid configuration"]
        for err in e.errors():
            loc = ".".join(str(p) for p in err["loc"])
            lines.append(f"  {loc}: {err['msg']}")
        raise ConfigError("\n".join(lines)) from e
