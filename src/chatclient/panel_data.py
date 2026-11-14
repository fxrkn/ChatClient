"""What the side panel shows, as plain data (no terminal code).

Both the terminal panel and ``chatclient --serve`` (the ``panel`` request) build it
here, so every front end shows the same thing. Finding the latest picture is left to
the caller: the REPL and the server each already know how.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from chatclient.images import stated_age

if TYPE_CHECKING:
    from chatclient.core.chat import ChatService

SCENE_KEYS = ("location", "time", "activity", "mood", "relationship")
MEMORIES_SHOWN = 5
TRAITS_SHOWN = 4


def traits(personality: str) -> list[str]:
    """The card's personality lines without their labels ("Temperament: warm" → "warm")."""
    out = []
    for raw in personality.splitlines():
        head, sep, rest = raw.partition(":")
        text = rest if sep and len(head) <= 20 else raw
        if text.strip():
            out.append(text.strip().rstrip("."))
    return out


@dataclass
class PanelData:
    picture: str | None = None  # absolute path
    name: str | None = None  # None without a character: show ``title``
    title: str = ""
    age: int | None = None
    content: str = "standard"
    private: bool = False
    personality: list[str] = field(default_factory=list)
    scene: dict[str, str] = field(default_factory=dict)
    memories: list[dict[str, Any]] = field(default_factory=list)  # content, pinned, kind
    model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def panel_data(chat: ChatService, picture: Path | None = None) -> PanelData:
    s = chat.require_session()
    char = chat.character()
    fresh = chat.store.get_session(s.id)  # the memory pass updates it in the background
    scene = (fresh.scene if fresh is not None else s.scene) or {}
    engine = chat.memory
    mems = (
        engine.repo.list_memories(engine.scope(s), MEMORIES_SHOWN)
        if engine is not None and not s.private
        else []
    )
    return PanelData(
        picture=str(picture.resolve()) if picture else None,
        name=char.name if char else None,
        title=s.title,
        age=stated_age(char.card) if char else None,
        content=chat.content_mode(),
        private=s.private,
        personality=traits(char.card.personality)[:TRAITS_SHOWN] if char else [],
        scene={k: v for k in SCENE_KEYS if isinstance(v := scene.get(k), str) and v.strip()},
        memories=[{"content": m.content, "pinned": m.pinned, "kind": m.kind} for m in mems],
        model=s.model or "",
    )
