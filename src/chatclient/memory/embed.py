"""Semantic search: turn text into vectors locally, and find the closest ones.

Backends (chosen by ``[memory] embeddings``):
  "auto"            Ollama's embedding model if one is installed, else fastembed
  "fastembed[:id]"  in-process ONNX model, downloaded once (~0.2 GB), CPU only
  "ollama:<model>"  an embedding model served by the local Ollama
  "off"             keyword/name/date recall only

Model loading is slow (seconds) and happens in the background worker, never on the
turn's critical path: until a model is ready, recall simply skips the semantic part.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
import warnings
from array import array
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import httpx

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

DEFAULT_FASTEMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Ollama embedding models that understand many languages (preferred in "auto").
OLLAMA_MULTILINGUAL = ("bge-m3", "paraphrase-multilingual", "granite-embedding", "multilingual-e5")


class Embedder(Protocol):
    name: str  # stored with every vector; a different model means re-indexing

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@contextlib.contextmanager
def _quiet():
    """Model downloads print progress bars and warnings; keep the chat screen clean.

    Only stderr is captured (that's where progress bars and warnings go). stdout is
    never touched: this runs in a worker thread, and a reply may start streaming to
    stdout at the same time."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        yield


class FastEmbedder:
    def __init__(self, model: str, cache_dir: Path):
        self.model_id = model
        self.name = f"fastembed:{model}"
        self.cache_dir = cache_dir
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            with _quiet():
                from fastembed import TextEmbedding

                for noisy in ("fastembed", "huggingface_hub", "onnxruntime"):
                    logging.getLogger(noisy).setLevel(logging.ERROR)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(self.model_id, cache_dir=str(self.cache_dir))
        return self._model

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        with _quiet():
            return [list(map(float, v)) for v in model.embed(texts)]

    @property
    def loaded(self) -> bool:
        return self._model is not None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts)


class OllamaEmbedder:
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.name = f"ollama:{model}"
        self.base_url = base_url.rstrip("/")
        self.loaded = True

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed", json={"model": self.model, "input": texts}
            )
            resp.raise_for_status()
            return resp.json()["embeddings"]


def fastembed_available() -> bool:
    try:
        import fastembed  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


async def pick_embedder(setting: str, models_dir: Path, ollama_url: str | None) -> Embedder | None:
    """Resolve the ``[memory] embeddings`` setting to a backend (or None = off)."""
    setting = (setting or "auto").strip()
    if setting == "off":
        return None
    if setting.startswith("ollama:") and ollama_url:
        return OllamaEmbedder(setting.split(":", 1)[1], ollama_url)
    if setting.startswith("fastembed"):
        model = setting.split(":", 1)[1] if ":" in setting else DEFAULT_FASTEMBED_MODEL
        return FastEmbedder(model, models_dir) if fastembed_available() else None
    # auto: an already installed multilingual Ollama embedding model, else fastembed
    if ollama_url:
        with contextlib.suppress(Exception):
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"{ollama_url.rstrip('/')}/api/tags")
                names = [m["name"] for m in resp.json().get("models", [])]
            for name in names:
                if any(k in name for k in OLLAMA_MULTILINGUAL):
                    return OllamaEmbedder(name, ollama_url)
    if fastembed_available():
        return FastEmbedder(DEFAULT_FASTEMBED_MODEL, models_dir)
    return None


# ── vector storage & search ─────────────────────────────────────────────────


def to_blob(vec: list[float]) -> bytes:
    norm = sum(x * x for x in vec) ** 0.5 or 1.0
    return array("f", (x / norm for x in vec)).tobytes()  # normalised: dot = cosine


class VectorIndex:
    """All vectors of one model in memory: (type, id, session) + a matrix for fast dot products."""

    def __init__(self) -> None:
        self.keys: list[tuple[str, int, int | None]] = []
        self.rows: list[bytes] = []
        self._matrix: np.ndarray | None = None
        self.loaded_model: str | None = None

    def load(self, model: str, rows: list[tuple[str, int, int | None, bytes]]) -> None:
        self.keys = [(t, i, s) for t, i, s, _ in rows]
        self.rows = [b for *_, b in rows]
        self._matrix = None
        self.loaded_model = model

    def add(self, owner_type: str, owner_id: int, session_id: int | None, blob: bytes) -> None:
        self.keys.append((owner_type, owner_id, session_id))
        self.rows.append(blob)
        self._matrix = None

    def search(
        self,
        query: list[float],
        *,
        owner_type: str,
        sessions: set[int] | None = None,
        owners: set[int] | None = None,
        exclude: set[int] | None = None,
        limit: int = 20,
    ) -> list[tuple[int, float]]:
        """(owner_id, cosine) best first, restricted to a type and to sessions/owners."""
        if not self.rows:
            return []
        import numpy as np

        if self._matrix is None:
            self._matrix = np.frombuffer(b"".join(self.rows), dtype=np.float32).reshape(
                len(self.rows), -1
            )
        q = np.frombuffer(to_blob(query), dtype=np.float32)
        if q.shape[0] != self._matrix.shape[1]:
            return []
        scores = self._matrix @ q
        out: list[tuple[int, float]] = []
        for idx in np.argsort(-scores):
            t, oid, sid = self.keys[idx]
            if t != owner_type:
                continue
            if sessions is not None and sid not in sessions:
                continue
            if owners is not None and oid not in owners:
                continue
            if exclude and oid in exclude:
                continue
            out.append((oid, float(scores[idx])))
            if len(out) >= limit:
                break
        return out
