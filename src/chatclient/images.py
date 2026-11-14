"""Pictures of the scene, drawn by a local ComfyUI (``/pic``).

- The user's own ComfyUI workflow (exported with "Save (API Format)") is used as it is:
  its checkpoint, LoRAs, sampler and size. We only find, from the graph itself, the
  positive and negative prompt nodes, the seed and the batch size, and fill them in.
- The prompt is written by the utility model from what is known: the character's
  fixed look (worked out once from the card and kept, so every picture shows the same
  person), the scene, and the latest messages. No word lists in the code.
- One safeguard is code, not model: with mature content on, a picture is only made of
  a character whose card states an adult age, "adult" is always in the prompt and
  childlike traits are always in the negative prompt.
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from chatclient.characters import Card, Character
from chatclient.memory import tasks
from chatclient.providers.base import ChatMessage, Provider

log = logging.getLogger(__name__)

ADULT_AGE = 18
# Always in the negative prompt (and enforced, not left to the prompt model).
SAFETY_NEGATIVE = "child, loli, shota, young, underage, teen, petite child body"


class ImageError(Exception):
    """A user-facing problem (ComfyUI not running, a workflow we can't use…)."""


# ── the workflow ────────────────────────────────────────────────────────────


@dataclass
class Workflow:
    graph: dict[str, Any]
    positive: str  # node ids
    negative: str | None
    sampler: str
    latent: str | None

    @property
    def negative_text(self) -> str:
        if self.negative is None:
            return ""
        return str(self.graph[self.negative]["inputs"].get("text") or "")


def _text_node(graph: dict[str, Any], link: Any, depth: int = 0) -> str | None:
    """Follow a conditioning link back to the node holding the prompt text."""
    if not (isinstance(link, list) and link and str(link[0]) in graph) or depth > 6:
        return None
    node_id = str(link[0])
    inputs = graph[node_id].get("inputs", {})
    if isinstance(inputs.get("text"), str):
        return node_id
    for value in inputs.values():  # e.g. a conditioning combine/concat node
        if isinstance(value, list) and (found := _text_node(graph, value, depth + 1)):
            return found
    return None


def analyze(graph: dict[str, Any]) -> Workflow:
    if not isinstance(graph, dict) or not graph:
        raise ImageError("not a ComfyUI workflow (use Save (API Format))")
    if "nodes" in graph and "links" in graph:
        raise ImageError("this is the UI format; in ComfyUI use Save (API Format)")
    for node_id, node in graph.items():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        if "KSampler" in str(node.get("class_type", "")) and "positive" in inputs:
            positive = _text_node(graph, inputs["positive"])
            if positive is None:
                continue
            latent = inputs.get("latent_image")
            latent_id = str(latent[0]) if isinstance(latent, list) else None
            if latent_id and "batch_size" not in graph.get(latent_id, {}).get("inputs", {}):
                latent_id = None
            return Workflow(graph, positive, _text_node(graph, inputs.get("negative")),
                            str(node_id), latent_id)  # fmt: skip
    raise ImageError("no KSampler with a text prompt found in this workflow")


# ── settings (config, overridden by what /pic setup saved in the app) ───────

SETTING_KEYS = {  # ImagesConfig field → key in the store
    "backend": "images_backend", "cloud_model": "images_cloud_model", "url": "images_url",
    "style": "images_style",
}  # fmt: skip


def image_settings(chat: Any) -> Any:
    """The config's [images], with the choices made in /pic setup on top."""
    cfg = chat.config.images.model_copy()
    for field, key in SETTING_KEYS.items():
        if value := chat.store.kv_get(key):
            setattr(cfg, field, value)
    return cfg


def save_settings(store: Any, **values: str) -> None:
    for field, value in values.items():
        store.kv_set(SETTING_KEYS[field], value)


def problem(chat: Any, workflow_path: Path) -> str | None:
    """Why no picture can be made yet (None: ready)."""
    cfg = image_settings(chat)
    if cfg.backend == "cloud":
        return None if cfg.cloud_model else "no cloud image model yet: /pic setup"
    return None if workflow_path.is_file() else "pictures aren't set up yet: /pic setup"


# The workflow /pic setup makes when the user has none: ComfyUI's plain text-to-image.
FAMILY_DEFAULTS = {
    "sd15": {"width": 512, "height": 768, "steps": 24, "cfg": 7.0,
             "sampler": "dpmpp_2m", "scheduler": "karras"},
    "sdxl": {"width": 832, "height": 1216, "steps": 26, "cfg": 6.0,
             "sampler": "euler_ancestral", "scheduler": "normal"},
}  # fmt: skip


def basic_workflow(checkpoint: str, family: str = "sd15") -> dict[str, Any]:
    d = FAMILY_DEFAULTS[family]
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": d["width"], "height": d["height"], "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {
            "seed": 1, "steps": d["steps"], "cfg": d["cfg"], "sampler_name": d["sampler"],
            "scheduler": d["scheduler"], "denoise": 1.0, "model": ["1", 0],
            "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0]}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"filename_prefix": "chatclient",
                                                     "images": ["6", 0]}},
    }  # fmt: skip


def workflow_file(data_dir: Path) -> Path:
    """Where /pic setup keeps its copy of the user's workflow."""
    return data_dir / "images" / "workflow.json"


def load_workflow(path: Path) -> Workflow:
    try:
        return analyze(json.loads(path.read_text(encoding="utf-8-sig")))
    except (OSError, ValueError) as e:
        raise ImageError(f"can't read the workflow {path}: {e}") from e


def fill(wf: Workflow, prompt: str, negative: str, seed: int) -> dict[str, Any]:
    g = copy.deepcopy(wf.graph)
    g[wf.positive]["inputs"]["text"] = prompt
    if wf.negative is not None:
        g[wf.negative]["inputs"]["text"] = negative
    sampler = g[wf.sampler]["inputs"]
    for key in ("seed", "noise_seed"):
        if key in sampler:
            sampler[key] = seed
    if wf.latent is not None:
        g[wf.latent]["inputs"]["batch_size"] = 1  # one picture per moment
    return g


# ── editing the workflow from the app (no node wiring in ComfyUI) ───────────

OURS = "chatclient:"  # title prefix of the nodes we add, so they can be replaced cleanly


def _is_ours(node: Any) -> bool:
    return isinstance(node, dict) and str(node.get("_meta", {}).get("title", "")).startswith(OURS)


def _relink(graph: dict[str, Any], old: tuple[str, int], new: list[Any], skip: set[str]) -> None:
    """Every input fed by output ``old`` is fed by ``new`` instead."""
    for node_id, node in graph.items():
        if node_id in skip or not isinstance(node, dict):
            continue
        for key, value in node.get("inputs", {}).items():
            if isinstance(value, list) and len(value) == 2 and (str(value[0]), value[1]) == old:
                node["inputs"][key] = list(new)


def _checkpoint(graph: dict[str, Any]) -> str:
    for node_id, node in graph.items():
        if isinstance(node, dict) and "ckpt_name" in node.get("inputs", {}):
            return str(node_id)
    raise ImageError("no checkpoint loader in this workflow")


def plan_of(graph: dict[str, Any]) -> tuple[list[tuple[str, float]], bool]:
    """The LoRAs and v-prediction switch we added earlier."""
    loras, vpred = [], False
    for node in graph.values():
        if _is_ours(node) and node.get("class_type") == "LoraLoader":
            loras.append(
                (str(node["inputs"]["lora_name"]), float(node["inputs"]["strength_model"]))
            )
        elif _is_ours(node) and node.get("class_type") == "ModelSamplingDiscrete":
            vpred = True
    return loras, vpred


def rebuild(graph: dict[str, Any], loras: list[tuple[str, float]], vpred: bool) -> dict[str, Any]:
    """Put our LoRA chain (and v-prediction sampling, for v-pred checkpoints) between the checkpoint and everything that used it."""
    g = copy.deepcopy(graph)
    for node_id in [n for n, node in g.items() if _is_ours(node)]:  # take ours out first
        inputs = g[node_id]["inputs"]
        _relink(g, (node_id, 0), inputs["model"], {node_id})
        if "clip" in inputs:
            _relink(g, (node_id, 1), inputs["clip"], {node_id})
        del g[node_id]
    ckpt = _checkpoint(g)
    model, clip = [ckpt, 0], [ckpt, 1]
    next_id = max((int(n) for n in g if str(n).isdigit()), default=0) + 1
    added: set[str] = set()
    for name, strength in loras:
        nid = str(next_id)
        next_id += 1
        g[nid] = {
            "class_type": "LoraLoader", "_meta": {"title": f"{OURS} lora"},
            "inputs": {"lora_name": name, "strength_model": strength, "strength_clip": strength,
                       "model": model, "clip": clip},
        }  # fmt: skip
        model, clip = [nid, 0], [nid, 1]
        added.add(nid)
    if vpred:
        nid = str(next_id)
        g[nid] = {
            "class_type": "ModelSamplingDiscrete", "_meta": {"title": f"{OURS} v-prediction"},
            "inputs": {"sampling": "v_prediction", "zsnr": True, "model": model},
        }  # fmt: skip
        model = [nid, 0]
        added.add(nid)
    _relink(g, (ckpt, 0), model, added)
    _relink(g, (ckpt, 1), clip, added)
    return g


def tune(graph: dict[str, Any], **values: Any) -> dict[str, Any]:
    """cfg, steps, sampler, scheduler, width, height, checkpoint."""
    g = copy.deepcopy(graph)
    wf = analyze(g)
    sampler = g[wf.sampler]["inputs"]
    for ours, theirs in (("cfg", "cfg"), ("steps", "steps"), ("sampler", "sampler_name"),
                         ("scheduler", "scheduler")):  # fmt: skip
        if values.get(ours) is not None:
            sampler[theirs] = values[ours]
    if wf.latent is not None:
        for key in ("width", "height"):
            if values.get(key):
                g[wf.latent]["inputs"][key] = int(values[key])
    if values.get("checkpoint"):
        g[_checkpoint(g)]["inputs"]["ckpt_name"] = values["checkpoint"]
    return g


def describe(graph: dict[str, Any]) -> dict[str, Any]:
    wf = analyze(graph)
    s = graph[wf.sampler]["inputs"]
    size = graph[wf.latent]["inputs"] if wf.latent else {}
    loras, vpred = plan_of(graph)
    return {
        "checkpoint": graph[_checkpoint(graph)]["inputs"]["ckpt_name"], "loras": loras,
        "vpred": vpred, "cfg": s.get("cfg"), "steps": s.get("steps"),
        "sampler": s.get("sampler_name"), "scheduler": s.get("scheduler"),
        "size": f"{size.get('width', '?')}x{size.get('height', '?')}",
    }  # fmt: skip


# ── ComfyUI ─────────────────────────────────────────────────────────────────


class ComfyClient:
    def __init__(self, url: str, timeout: float = 180.0):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.client_id = uuid.uuid4().hex

    async def available(self, node: str, field: str) -> list[str]:
        """What ComfyUI has installed, e.g. ("LoraLoader", "lora_name")."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                info = (await http.get(f"{self.url}/object_info/{node}")).json()
            return [str(x) for x in info[node]["input"]["required"][field][0]]
        except (httpx.HTTPError, KeyError, ValueError, TypeError) as e:
            raise ImageError(f"ComfyUI is not reachable at {self.url} ({e})") from e

    async def draw(self, graph: dict[str, Any]) -> bytes:
        async with httpx.AsyncClient(timeout=30.0) as http:
            try:
                resp = await http.post(f"{self.url}/prompt",
                                       json={"prompt": graph, "client_id": self.client_id})  # fmt: skip
            except httpx.HTTPError as e:
                raise ImageError(f"ComfyUI is not reachable at {self.url} ({e})") from e
            if resp.status_code != 200:
                raise ImageError(f"ComfyUI refused the workflow: {resp.text[:400]}")
            prompt_id = resp.json().get("prompt_id")
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.timeout
            while loop.time() < deadline:
                await asyncio.sleep(0.5)
                hist = (await http.get(f"{self.url}/history/{prompt_id}")).json()
                entry = hist.get(prompt_id)
                if not entry:
                    continue
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    raise ImageError(f"ComfyUI failed: {json.dumps(status)[:400]}")
                for out in entry.get("outputs", {}).values():
                    for img in out.get("images", []):
                        view = await http.get(f"{self.url}/view", params={
                            "filename": img["filename"], "subfolder": img.get("subfolder", ""),
                            "type": img.get("type", "output")})  # fmt: skip
                        view.raise_for_status()
                        return view.content
            raise ImageError(f"ComfyUI took longer than {self.timeout:.0f}s")


class CloudImages:
    """An image model behind an OpenAI-style API (OpenRouter: image output in a chat
    reply). Online models apply their own content rules; most refuse adult pictures."""

    def __init__(self, provider: Any, model: str):
        self.provider, self.model = provider, model

    async def models(self) -> list[str]:
        """The provider's models that can output images."""
        try:
            infos = await self.provider.list_model_info()
        except Exception as e:
            raise ImageError(f"can't list the models: {e}") from e
        return sorted(
            str(m["id"]) for m in infos
            if "image" in ((m.get("architecture") or {}).get("output_modalities") or [])
        )  # fmt: skip

    async def draw(self, prompt: str) -> bytes:
        import base64

        body = {"model": self.model, "modalities": ["image", "text"],
                "messages": [{"role": "user", "content": prompt}]}  # fmt: skip
        try:
            resp = await self.provider.client.post(
                f"{self.provider.base_url}/chat/completions", json=body, timeout=180.0
            )
        except httpx.HTTPError as e:
            raise ImageError(f"{self.provider.name} is not reachable ({e})") from e
        if resp.status_code != 200:
            raise ImageError(f"{self.provider.name}: HTTP {resp.status_code}: {resp.text[:300]}")
        message = ((resp.json().get("choices") or [{}])[0]).get("message") or {}
        for image in message.get("images") or []:
            url = str((image.get("image_url") or {}).get("url") or "")
            if url.startswith("data:") and "," in url:
                return _as_png(base64.b64decode(url.split(",", 1)[1]))
        reason = str(message.get("content") or "no picture in the answer")[:200]
        raise ImageError(f"the model made no picture ({reason})")


def _as_png(data: bytes) -> bytes:
    """Cloud models may answer with JPEG/WebP; pictures are kept as PNG."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        out = io.BytesIO()
        img.convert("RGB").save(out, "PNG")
        return out.getvalue()


# ── the prompt (written by the model) ───────────────────────────────────────

PROMPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"prompt": {"type": "string"}},
    "required": ["prompt"],
}

STYLE = {
    "tags": "comma-separated Danbooru-style tags in English (e.g. 1girl, brown hair, round "
    "glasses, sitting, classroom, evening light), 20-40 tags",
    "natural": "one or two plain English sentences describing the picture",
}


def look_prompt(card: Card, style: str) -> list[ChatMessage]:
    """Once per character: the fixed look, so every picture shows the same person."""
    sheet = " ".join(x for x in (card.description, card.personality) if x)
    sheet = sheet.replace("{{char}}", card.name).replace("{{user}}", "the user")
    return [ChatMessage("user", (
        f"Character sheet of {card.name}: {sheet}\n\n"
        f"Describe only how {card.name} looks, as "
        f"{'at most 8 comma-separated Danbooru-style tags' if style == 'tags' else 'one short phrase'}"
        ": age as stated, sex, hair, eyes, build and the one "
        "or two features that make them recognisable (e.g. glasses). No clothing, pose, "
        "place, expression, mood or small facial details. If the sheet doesn't say, choose "
        'something fitting. Return ONLY JSON: {"prompt": "..."}'
    ))]  # fmt: skip


def scene_prompt(card: Card, look: str, user_look: str, scene: dict[str, Any],
                 recent: list[tuple[str, str]], wish: str, style: str) -> list[ChatMessage]:  # fmt: skip
    convo = "\n".join(f"{who}: {text[-700:]}" for who, text in recent)
    facts = ", ".join(f"{k}: {v}" for k, v in scene.items() if isinstance(v, str))
    extra = (
        f"\n\nThe user asks for: {wish}\nThis decides what the picture shows: build the "
        "pose, action and framing around it, even if the story was elsewhere; keep from "
        "the story only what fits it (place, light, clothing). Nothing that contradicts it."
        if wish
        else ""
    )
    return [ChatMessage("user", (
        f"Write the prompt for an image of the moment the conversation below has reached, "
        f"as {STYLE[style]}.\n\n{card.name} looks like this (use it as it is): {look}\n"
        + (f"The user looks like this: {user_look}\n" if user_look else "")
        + (f"Scene: {facts}\n" if facts else "")
        + f"\nLatest messages:\n{convo}{extra}\n\n"
        "Most important first: who is in the picture (e.g. 1girl, 1boy), what is happening "
        "and the pose, clothing or nudity as it is now, the framing and camera angle, the "
        "expression; then the place and the light. Choose ONE framing that can actually "
        "show the action (an action seen from behind is not a close-up of the face; in a "
        "pov picture the viewer's own face is never seen). Only things that can be drawn: "
        "no moods or ideas (intimate, trusting, submission…), nothing that contradicts. "
        f"At most {12 if wish else 20} tags. Don't describe anyone's face or hair (their "
        "look is added for you), and no names: an image model doesn't know these people. "
        'Return ONLY JSON: {"prompt": "..."}'
    ))]  # fmt: skip


WORTH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"draw": {"type": "boolean"}},
    "required": ["draw"],
}


def worth_prompt(recent: list[tuple[str, str]], last_prompt: str) -> str:
    convo = "\n".join(f"{who}: {text}" for who, text in recent)
    before = f"The last picture showed: {last_prompt}\n\n" if last_prompt else ""
    return (
        f"{before}The latest exchange of a story:\n{convo}\n\n"
        "Has something visibly new happened that is worth a new picture: a new place, a "
        "new action or pose, clothes coming off or on, a striking moment? Talk, small "
        "changes or the same scene as the last picture: no. "
        'Return ONLY JSON: {"draw": true or false}'
    )


async def _ask(provider: Provider, model: str, messages: list[ChatMessage]) -> str:
    raw = await tasks.complete(provider, model, messages, max_tokens=400, temperature=0.4,
                               schema=PROMPT_SCHEMA)  # fmt: skip
    data = tasks.parse_json(raw) or {}
    text = str(data.get("prompt") or "").strip()
    if not text:
        raise ImageError("the model wrote no image prompt (it may have refused)")
    return " ".join(text.split())


# ── the safeguard ───────────────────────────────────────────────────────────


def stated_age(card: Card) -> int | None:
    from chatclient.char_builder import parse_age

    return parse_age(card.description)


def check_adult(card: Card, mature: bool) -> None:
    """Mature content: only characters whose card states an adult age are drawn."""
    if not mature:
        return
    age = stated_age(card)
    if age is None or age < ADULT_AGE:
        raise ImageError(
            f"{card.name}'s card doesn't state an adult age; with mature content on, only "
            "characters stated to be 18 or older are drawn. Add the age to the "
            "description (/char edit)."
        )


LOOK_TAGS = 8  # a look longer than this (older ones) is cut: it would bury the action
# Words that never go into a mature picture from hand-written tags (the user's own
# "+tags" and extras; the model-written part is covered by the negative prompt).
CHILDLIKE = ("child", "loli", "shota", "underage", "kid", "toddler", "infant")


def split_wish(text: str) -> tuple[str, str]:
    """``/pic she blushes +pov, from below``: before "+" a wish for the model, after it
    tags used exactly as written."""
    wish, _, tags = text.partition("+")
    return wish.strip(), ", ".join(t.strip() for t in tags.replace("+", ",").split(",")
                                   if t.strip())  # fmt: skip


def check_tags(tags: str, mature: bool) -> None:
    if not mature:
        return
    words = {w.strip(" ,()").lower() for w in tags.replace(",", " ").split()}
    if bad := sorted(words.intersection(CHILDLIKE)):
        raise ImageError(f"not in a mature picture: {', '.join(bad)}")


def assemble(
    quality: str, look: str, scene: str, mature: bool, style: str = "tags",
    exact: str = "", extra: str = "",
) -> str:  # fmt: skip
    """Order matters (SD reads the prompt in 75-token chunks and weighs the start most):
    quality, the user's exact tags, what is happening, the fixed extras, the look."""
    if style == "tags":
        look = ", ".join([t.strip() for t in look.split(",") if t.strip()][:LOOK_TAGS])
    parts = [quality, "adult" if mature else "", exact, scene, extra, look]
    text = ", ".join(p.strip(" ,") for p in parts if p and p.strip(" ,"))
    if style != "tags":
        return text
    seen: set[str] = set()
    tags = []
    for tag in (t.strip() for t in text.split(",")):  # a tag said twice counts once
        if tag and tag.lower() not in seen:
            seen.add(tag.lower())
            tags.append(tag)
    return ", ".join(tags)


# Used when neither the workflow nor [images] negative has one: without any, SD models
# lose much of their quality (a workflow exported with empty prompt boxes, for one).
DEFAULT_NEGATIVE = (
    "worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, "
    "missing fingers, extra digits, fewer digits, blurry, jpeg artifacts, watermark, text, "
    "signature"
)


def negative_for(base: str, extra: str = "") -> str:
    base, extra = base.strip().rstrip(","), extra.strip().rstrip(",")
    if not base and not extra:
        base = DEFAULT_NEGATIVE
    parts = [base, extra, SAFETY_NEGATIVE]
    return ", ".join(p for p in parts if p)


def _new_seed() -> int:
    return int.from_bytes(uuid.uuid4().bytes[:4], "big")


# ── putting it together ─────────────────────────────────────────────────────

LOOK_KEY = "look"  # the fixed look, kept in the card (extensions.chatclient.look)
EXTRA_KEY = "pic_extra"  # the character's own fixed tags (e.g. a LoRA trigger)


def _card_value(card: Card, key: str) -> str:
    ext = card.extra.get("extensions")
    ours = ext.get("chatclient") if isinstance(ext, dict) else None
    value = ours.get(key) if isinstance(ours, dict) else None
    return value if isinstance(value, str) else ""


def _set_card_value(card: Card, key: str, value: str | None) -> None:
    ext = card.extra.setdefault("extensions", {})
    if not isinstance(ext, dict):
        ext = card.extra["extensions"] = {}
    ours = ext.setdefault("chatclient", {})
    if value:
        ours[key] = value
    else:
        ours.pop(key, None)


def card_look(card: Card) -> str:
    return _card_value(card, LOOK_KEY)


def set_card_look(card: Card, look: str | None) -> None:
    _set_card_value(card, LOOK_KEY, look)


def card_pic_extra(card: Card) -> str:
    return _card_value(card, EXTRA_KEY)


def set_card_pic_extra(card: Card, tags: str | None) -> None:
    _set_card_value(card, EXTRA_KEY, tags)


def pictures_of(store: Any, char: Character, messages: list[Any]) -> list[tuple[Any, Path]]:
    """(message, picture file) for the messages that have pictures, oldest first."""
    lib = store.library
    base = (lib.folder(char) if lib is not None else Path(store.path).parent) / "images"
    out = []
    for m in messages:
        for name in (m.meta or {}).get("images") or []:
            if (path := base / name).is_file():
                out.append((m, path))
    return out


def saved_prompt(picture: Path) -> str:
    try:
        return str(json.loads(picture.with_suffix(".json").read_text(encoding="utf-8"))["prompt"])
    except (OSError, ValueError, KeyError):
        return ""


@dataclass
class Picture:
    path: Path
    prompt: str
    seconds: float


class ImageMaker:
    """Draws the current moment of a chat (used by /pic and by --serve)."""

    def __init__(self, chat: Any, workflow_path: Path):
        self.chat = chat  # ChatService
        self.cfg = image_settings(chat)
        self.workflow_path = workflow_path

    @property
    def cloud(self) -> bool:
        return self.cfg.backend == "cloud"

    def _workflow(self) -> Workflow | None:
        if self.cloud:
            return None
        if not self.workflow_path.is_file():
            raise ImageError("pictures aren't set up yet: /pic setup")
        return load_workflow(self.workflow_path)

    def _utility(self) -> Any:
        from chatclient.providers.registry import parse_model_ref

        spec = self.chat.utility_model()
        if not spec:
            raise ImageError("no model to write the picture's prompt")
        return parse_model_ref(spec)

    async def look_of(self, char: Character) -> str:
        """The fixed look: written once by the model, then kept in the card (editable
        with /pic look), so every picture shows the same person."""
        if look := card_look(char.card):
            return look
        ref = self._utility()
        look = await _ask(self.chat.registry.get(ref.provider), ref.model,
                          look_prompt(char.card, self.cfg.style))  # fmt: skip
        set_card_look(char.card, look)
        self.chat.store.update_character(char.id, char.card)
        return look

    async def draw(self, wish: str = "") -> Picture:
        loop = asyncio.get_running_loop()
        started = loop.time()
        if (why := problem(self.chat, self.workflow_path)) is not None:
            raise ImageError(why)
        workflow = self._workflow()
        char = self.chat.character()
        if char is None:
            raise ImageError("pictures are made in character chats (/char)")
        mature = self.chat.content_mode() == "mature"
        check_adult(char.card, mature)
        wish, exact = split_wish(wish)
        extra = ", ".join(x for x in (card_pic_extra(char.card), self.cfg.extra) if x.strip())
        check_tags(f"{exact}, {extra}", mature)
        look = await self.look_of(char)
        if exact and not wish:
            # only "+tags": exactly what the user wrote, as typing it into ComfyUI would
            # (the model would add the story's pose, the place, moods… and blur it)
            scene = ""
        else:
            if exact:  # the picture is built around them, not around the story's last pose
                wish = f"{wish} {exact}".strip()
            persona = self.chat.persona()
            user_look = await self.look_of(persona) if persona and persona.card.description else ""
            recent = [
                (self.chat.user_name() if m.role == "user" else char.name, m.content)
                for m in self.chat.history(limit=4)
            ]
            ref = self._utility()
            scene = await _ask(
                self.chat.registry.get(ref.provider), ref.model,
                scene_prompt(char.card, look, user_look, self.chat.refresh().scene, recent,
                             wish, self.cfg.style),
            )  # fmt: skip
        prompt = assemble(self.cfg.quality, look, scene, mature, self.cfg.style, exact, extra)
        negative = negative_for(workflow.negative_text if workflow else "", self.cfg.negative)
        # a new seed each time: in SD the seed decides much of the composition, so a fixed
        # one pulls every picture toward the same pose; the look tags keep the person
        return await self._render(workflow, char, prompt, negative, _new_seed(), started)

    async def again(self, source: Path) -> Picture:
        """The same picture again with another seed: no new prompt, so it's quicker."""
        started = asyncio.get_running_loop().time()
        workflow = self._workflow()
        char = self.chat.character()
        if char is None:
            raise ImageError("pictures are made in character chats (/char)")
        check_adult(char.card, self.chat.content_mode() == "mature")
        try:
            saved = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ImageError("that picture has no saved prompt to draw again") from e
        negative = negative_for(str(saved.get("negative") or ""), "")
        return await self._render(
            workflow, char, str(saved["prompt"]), negative, _new_seed(), started
        )

    async def _render(self, workflow: Workflow | None, char: Character, prompt: str,
                      negative: str, seed: int, started: float) -> Picture:  # fmt: skip
        if workflow is None:  # cloud: no negative prompt or seed to give
            from chatclient.providers.registry import parse_model_ref

            ref = parse_model_ref(self.cfg.cloud_model)
            data = await CloudImages(self.chat.registry.get(ref.provider), ref.model).draw(prompt)
        else:
            graph = fill(workflow, prompt, negative, seed)
            data = await ComfyClient(self.cfg.url, self.cfg.timeout).draw(graph)
        path = self._save(char, data)
        # next to the picture: how it was made (for /pic again and the gallery)
        path.with_suffix(".json").write_text(
            json.dumps({"prompt": prompt, "negative": negative, "seed": seed},
                       ensure_ascii=False, indent=1), encoding="utf-8")  # fmt: skip
        self._attach(path)
        return Picture(path, prompt, asyncio.get_running_loop().time() - started)

    async def worth_drawing(self, last_prompt: str = "") -> bool:
        """Auto mode: has something visibly new happened since the last picture? The
        model decides (small talk or the same pose: no), so pictures come at moments."""
        char = self.chat.character()
        if char is None:
            return False
        recent = [
            (self.chat.user_name() if m.role == "user" else char.name, m.content[-600:])
            for m in self.chat.history(limit=2)
        ]
        ref = self._utility()
        raw = await tasks.complete(
            self.chat.registry.get(ref.provider), ref.model,
            [ChatMessage("user", worth_prompt(recent, last_prompt))],
            max_tokens=20, temperature=0.0, schema=WORTH_SCHEMA,
        )  # fmt: skip
        data = tasks.parse_json(raw) or {}
        return data.get("draw") is True

    def _save(self, char: Character, data: bytes) -> Path:
        from datetime import datetime

        lib = self.chat.store.library
        base = lib.folder(char) if lib is not None else Path(self.chat.store.path).parent
        folder = base / "images"
        folder.mkdir(parents=True, exist_ok=True)
        reply = self.chat.last_reply()
        tag = reply.uuid[:8] if reply is not None and reply.uuid else uuid.uuid4().hex[:8]
        path = folder / f"{datetime.now():%Y%m%d-%H%M%S}-{tag}.png"
        path.write_bytes(data)
        return path

    def _attach(self, path: Path) -> None:
        """The picture belongs to the reply it shows (kept in the chat file too)."""
        reply = self.chat.last_reply()
        if reply is None:
            return
        meta = dict(reply.meta or {})
        meta["images"] = [*meta.get("images", []), path.name]
        self.chat.store.set_message_meta(reply.id, meta)
