"""/pic: the user's ComfyUI workflow, a model-written prompt, the adult safeguard."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from chatclient.characters import Card, set_card_content
from chatclient.core.chat import TurnResult
from chatclient.images import (
    SAFETY_NEGATIVE,
    ImageError,
    ImageMaker,
    analyze,
    card_look,
    check_adult,
    fill,
)
from chatclient.providers.base import StreamEvent
from chatclient.providers.echo import EchoProvider

# A plain txt2img workflow in API format, trimmed to what matters.
WORKFLOW = {
    "9": {"inputs": {"filename_prefix": "ComfyUI", "images": ["65", 0]}, "class_type": "SaveImage"},
    "65": {"inputs": {"samples": ["70", 0], "vae": ["75", 2]}, "class_type": "VAEDecode"},
    "67": {"inputs": {"text": "masterpiece, 1girl", "clip": ["75", 1]},
           "class_type": "CLIPTextEncode"},
    "70": {"inputs": {"seed": 42, "steps": 13, "cfg": 1, "model": ["75", 0],
                      "positive": ["67", 0], "negative": ["78", 0], "latent_image": ["77", 0]},
           "class_type": "KSampler"},
    "75": {"inputs": {"ckpt_name": "anime_model.safetensors"},
           "class_type": "CheckpointLoaderSimple"},
    "77": {"inputs": {"width": 512, "height": 512, "batch_size": 47},
           "class_type": "EmptyLatentImage"},
    "78": {"inputs": {"text": "lowres, worst quality", "clip": ["75", 1]},
           "class_type": "CLIPTextEncode"},
}  # fmt: skip
PNG = b"\x89PNG\r\n\x1a\nfake"
NEW_MOMENT = {"draw": True}  # what the stand-in model answers to "worth a new picture?"


def fake_comfy(sent: list[dict]) -> None:
    """Routes for a respx mock: a ComfyUI that accepts anything and returns PNG."""
    respx.post("http://127.0.0.1:8188/prompt").mock(
        side_effect=lambda r: sent.append(json.loads(r.content)["prompt"])
        or httpx.Response(200, json={"prompt_id": "p"}))  # fmt: skip
    respx.get("http://127.0.0.1:8188/history/p").mock(return_value=httpx.Response(200, json={
        "p": {"outputs": {"9": {"images": [{"filename": "a.png"}]}}}}))  # fmt: skip
    respx.get("http://127.0.0.1:8188/view").mock(return_value=httpx.Response(200, content=PNG))


def test_the_workflow_is_read_from_its_own_graph():
    wf = analyze(WORKFLOW)
    assert (wf.positive, wf.negative, wf.sampler, wf.latent) == ("67", "78", "70", "77")
    g = fill(wf, "a cat", "bad", 7)
    assert g["67"]["inputs"]["text"] == "a cat" and g["78"]["inputs"]["text"] == "bad"
    assert g["70"]["inputs"]["seed"] == 7
    assert g["77"]["inputs"]["batch_size"] == 1  # never 47 pictures per moment
    assert WORKFLOW["77"]["inputs"]["batch_size"] == 47  # the original is untouched
    with pytest.raises(ImageError, match="API Format"):
        analyze({"nodes": [], "links": []})  # the UI format


def test_mature_pictures_need_a_stated_adult_age():
    check_adult(Card(name="X", description="no age here"), mature=False)  # fine
    with pytest.raises(ImageError, match="adult age"):
        check_adult(Card(name="X", description="a student"), mature=True)
    with pytest.raises(ImageError):
        check_adult(Card(name="X", description="{{char}} is a 16-year-old girl."), mature=True)
    check_adult(Card(name="X", description="{{char}} is a 22-year-old woman."), mature=True)


@pytest.fixture
def pic(chat, tmp_path, monkeypatch):
    asked: list[str] = []

    async def writer(self, model, messages, params):
        prompt = messages[-1].content
        asked.append(prompt)
        if "Describe only how" in prompt:
            text = json.dumps({"prompt": "1girl, 22 years old, brown hair, round glasses"})
        elif "worth a new picture" in prompt:
            text = json.dumps({"draw": NEW_MOMENT["draw"]})
        elif "Write the prompt for an image" in prompt:
            text = json.dumps({"prompt": "sitting in a lecture hall, smiling, upper body"})
        else:
            text = "*smiles* Hello."
        yield StreamEvent(text=text)

    monkeypatch.setattr(EchoProvider, "stream_chat", writer)
    card = Card(name="Lena", description="{{char}} is a 22-year-old woman. Brown hair.")
    set_card_content(card, "mature")
    lena = chat.store.add_character("ai", card)
    chat.new_session(model="echo:echo", character_id=lena.id)
    wf = tmp_path / "workflow.json"
    wf.write_text(json.dumps(WORKFLOW), encoding="utf-8")
    return ImageMaker(chat, wf), asked, lena


async def test_a_picture_of_the_moment(pic, chat):
    maker, asked, lena = pic
    async for _ in chat.send("hi", TurnResult()):
        pass
    sent: list[dict] = []

    def queue(request):
        sent.append(json.loads(request.content)["prompt"])
        return httpx.Response(200, json={"prompt_id": "p1"})

    with respx.mock:
        respx.post("http://127.0.0.1:8188/prompt").mock(side_effect=queue)
        respx.get("http://127.0.0.1:8188/history/p1").mock(return_value=httpx.Response(200, json={
            "p1": {"status": {"status_str": "success"},
                   "outputs": {"9": {"images": [{"filename": "a.png", "type": "output"}]}}}}))  # fmt: skip
        respx.get("http://127.0.0.1:8188/view").mock(return_value=httpx.Response(200, content=PNG))
        picture = await maker.draw("close-up")

    graph = sent[0]
    positive, negative = graph["67"]["inputs"]["text"], graph["78"]["inputs"]["text"]
    assert positive.startswith(
        "masterpiece, best quality, adult, sitting in a lecture hall"
    )  # action first`n    assert positive.endswith("1girl, 22 years old, brown hair, round glasses")  # then the look
    assert "lecture hall" in positive
    assert negative.startswith("lowres") and SAFETY_NEGATIVE in negative
    assert graph["77"]["inputs"]["batch_size"] == 1
    assert "The user asks for: close-up" in asked[-1]
    assert picture.path.read_bytes() == PNG and picture.path.parent.name == "images"
    assert card_look(chat.store.get_character(lena.id).card).startswith("1girl")  # kept
    assert chat.last_reply().meta["images"] == [picture.path.name]  # tied to the reply


async def test_comfyui_not_running_is_a_clear_message(pic, chat):
    maker, _asked, _lena = pic
    with respx.mock:
        respx.post("http://127.0.0.1:8188/prompt").mock(side_effect=httpx.ConnectError("no"))
        with pytest.raises(ImageError, match="not reachable"):
            await maker.draw()


async def test_pic_command_runs_in_the_background(chat, tmp_path):
    from tests.test_repl import run_script

    await run_script(chat, tmp_path, "/pic look\r/quit\r")  # no character: a clear error
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps(WORKFLOW), encoding="utf-8")
    await run_script(chat, tmp_path, f'/pic setup "{wf}"\r/quit\r')
    assert (tmp_path / "images" / "workflow.json").is_file()


def test_tags_said_twice_count_once():
    from chatclient.images import assemble

    out = assemble("masterpiece", "1girl, red hair", "1girl, Red Hair, holding a book", False)
    assert out == "masterpiece, 1girl, Red Hair, holding a book"  # scene first; repeats dropped
    assert (
        assemble("", "a woman", "she reads", True, style="natural") == "adult, she reads, a woman"
    )


async def test_pic_says_what_is_missing_right_away(chat, tmp_path, capsys):
    from tests.test_repl import run_script

    await run_script(chat, tmp_path, '/pic\r/pic setuo "C:\\x\\wf.json"\r/quit\r')
    out = capsys.readouterr().out
    assert "set up yet: /pic setup" in out and "Drawing" not in out
    assert 'Did you mean: /pic setup "C:\\x\\wf.json"' in out


def _decode_sixel(data: str) -> dict[tuple[int, int], tuple[int, int, int]]:
    """A tiny Sixel reader for the test: (x, y) → RGB percent of every painted pixel."""
    import re

    body = data[data.index("q") + 1 : data.rindex("\x1b\\")]
    palette, pixels = {}, {}
    color, x, top = 0, 0, 0
    tokens = re.findall(r'"[\d;]+|#\d+(?:;\d+;\d+;\d+;\d+)?|!\d+.|[?-~]|\$|-', body)
    for tok in tokens:
        if tok.startswith('"'):
            continue
        if tok.startswith("#"):
            parts = [int(p) for p in tok[1:].split(";")]
            color = parts[0]
            if len(parts) == 5:
                palette[color] = tuple(parts[2:])
            continue
        if tok == "$":
            x = 0
            continue
        if tok == "-":
            x, top = 0, top + 6
            continue
        n, ch = (int(tok[1:-1]), tok[-1]) if tok.startswith("!") else (1, tok)
        bits = ord(ch) - 63
        for _ in range(n):
            for row in range(6):
                if bits >> row & 1:
                    pixels[(x, top + row)] = palette[color]
            x += 1
    return pixels


def test_sixel_draws_the_picture(tmp_path):
    from PIL import Image

    from chatclient.ui import sixel

    img = Image.new("RGB", (20, 13), (255, 0, 0))
    for x in range(20):
        for y in range(13):
            if (x // 5 + y // 4) % 2:
                img.putpixel((x, y), (0, 0, 255))
    path = tmp_path / "p.png"
    img.save(path)
    data = sixel.encode(path, max_width=20)
    assert data.startswith('\x1bPq"1;1;20;13') and data.endswith("\x1b\\")
    pixels = _decode_sixel(data)
    assert len(pixels) == 20 * 13  # every pixel painted exactly where it belongs
    for (x, y), rgb in pixels.items():
        assert rgb == ((0, 0, 100) if (x // 5 + y // 4) % 2 else (100, 0, 0))


def test_inline_only_where_the_terminal_can(monkeypatch):
    from chatclient.ui import sixel

    for var in ("WT_SESSION", "TERM_PROGRAM", "TERM"):
        monkeypatch.delenv(var, raising=False)
    assert not sixel.supported("auto") and sixel.supported("sixel")
    monkeypatch.setenv("WT_SESSION", "x")  # Windows Terminal
    assert sixel.supported("auto") and not sixel.supported("off")


async def test_serve_draws_in_the_background(chat, monkeypatch, tmp_path):
    from tests.test_serve import Client

    from chatclient.images import ImageMaker, Picture

    async def fake_draw(self, wish=""):
        return Picture(tmp_path / "x.png", f"1girl, {wish}", 1.0)

    monkeypatch.setattr(ImageMaker, "draw", fake_draw)
    c = Client(chat)
    assert (await c.call("picture"))["error"]["code"] == "image"  # no workflow yet
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps(WORKFLOW), encoding="utf-8")
    assert "workflow" in (await c.call("picture.setup", path=str(wf)))["result"]
    done = await c.call("picture", wish="close-up")
    assert done["result"]["prompt"] == "1girl, close-up"
    await c.close()


def test_the_terminal_is_asked_what_it_can_draw(monkeypatch):
    from chatclient.ui import sixel

    assert sixel.has_sixel("\x1b[?61;4;6;7;14;21;22;23;24;28;32;42c")  # Windows Terminal 1.22+
    assert not sixel.has_sixel("\x1b[?1;0c")  # old conhost
    assert not sixel.has_sixel("\x1b[?64;1;2;6;9;15;18;21;22c")  # a VT terminal without it
    assert not sixel.has_sixel("")
    for var in ("WT_SESSION", "TERM_PROGRAM", "TERM"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(sixel, "_asked", True)  # the answer beats the guess
    assert sixel.supported("auto") and not sixel.supported("off")
    monkeypatch.setattr(sixel, "_asked", False)
    assert not sixel.supported("auto") and sixel.supported("sixel")


async def test_pic_inline_can_be_switched_in_the_app(chat, tmp_path, capsys):
    from tests.test_repl import run_script

    await run_script(chat, tmp_path, "/pic inline sixel\r/pic inline\r/quit\r")
    assert chat.store.kv_get("images_inline") == "sixel"
    assert "drawing here: True" in capsys.readouterr().out


def test_a_long_look_does_not_bury_the_action():
    from chatclient.images import LOOK_TAGS, assemble

    look = ", ".join(f"trait{i}" for i in range(30))
    out = assemble("q", look, "from behind, ass, looking back", True).split(", ")
    assert out[:5] == ["q", "adult", "from behind", "ass", "looking back"]
    assert sum(t.startswith("trait") for t in out) == LOOK_TAGS


async def test_setup_warns_about_an_odd_cfg(chat, tmp_path, capsys):
    from tests.test_repl import run_script

    burnt = json.loads(json.dumps(WORKFLOW))
    burnt["70"]["inputs"].update({"cfg": 25, "steps": 13})  # swapped by mistake
    wf = tmp_path / "wf.json"
    wf.write_text(json.dumps(burnt), encoding="utf-8")
    await run_script(chat, tmp_path, f'/pic setup "{wf}"\r/quit\r')
    assert "cfg 25" in capsys.readouterr().out


def test_exact_tags_and_extras():
    from chatclient.images import assemble, check_tags, split_wish

    assert split_wish("she blushes +pov, from below") == ("she blushes", "pov, from below")
    assert split_wish("+pov +nsfw") == ("", "pov, nsfw")
    assert split_wish("close-up") == ("close-up", "")
    out = assemble("q", "red hair", "sitting", True, exact="pov", extra="lora_x")
    assert out == "q, adult, pov, sitting, lora_x, red hair"  # the user's own tags lead
    check_tags("pov, nsfw", mature=True)
    with pytest.raises(ImageError, match="loli"):
        check_tags("pov, (loli)", mature=True)
    check_tags("child", mature=False)  # a standard story may show a child (not mature)


async def test_exact_tags_reach_comfyui(pic, chat):
    from chatclient.images import set_card_pic_extra

    maker, asked, lena = pic
    chat.config.images.negative = "lowres"
    card = chat.store.get_character(lena.id).card
    set_card_pic_extra(card, "lena_lora")
    chat.store.update_character(lena.id, card)
    async for _ in chat.send("hi", TurnResult()):
        pass
    sent: list[dict] = []
    with respx.mock:
        respx.post("http://127.0.0.1:8188/prompt").mock(
            side_effect=lambda r: sent.append(json.loads(r.content)["prompt"])
            or httpx.Response(200, json={"prompt_id": "p"}))  # fmt: skip
        respx.get("http://127.0.0.1:8188/history/p").mock(return_value=httpx.Response(200, json={
            "p": {"outputs": {"9": {"images": [{"filename": "a.png"}]}}}}))  # fmt: skip
        respx.get("http://127.0.0.1:8188/view").mock(return_value=httpx.Response(200, content=PNG))
        await maker.draw("smiling +pov, from below")
    positive, negative = sent[0]["67"]["inputs"]["text"], sent[0]["78"]["inputs"]["text"]
    assert positive.startswith("masterpiece, best quality, adult, pov, from below")
    assert "lena_lora" in positive
    assert "lowres" in negative and SAFETY_NEGATIVE in negative
    assert "The user asks for: smiling pov, from below" in asked[-1]
    assert "even if the story was elsewhere" in asked[-1]  # the wish leads, not the old pose


async def test_again_redraws_without_asking_the_model(pic, chat):
    maker, asked, _lena = pic
    async for _ in chat.send("hi", TurnResult()):
        pass
    sent: list[dict] = []
    with respx.mock:
        fake_comfy(sent)
        first = await maker.draw()
        saved = json.loads(first.path.with_suffix(".json").read_text(encoding="utf-8"))
        before = len(asked)
        second = await maker.again(first.path)
    assert len(asked) == before  # no new prompt: quicker
    assert sent[1]["67"]["inputs"]["text"] == saved["prompt"]
    assert sent[1]["70"]["inputs"]["seed"] != saved["seed"]  # a different draw
    assert chat.last_reply().meta["images"] == [first.path.name, second.path.name]


def _repl_for(chat, tmp_path):
    import shutil

    from chatclient.i18n import Translator
    from chatclient.paths import resolve_paths
    from chatclient.ui.repl import Repl

    paths = resolve_paths(tmp_path)
    paths.ensure()
    (tmp_path / "images").mkdir(exist_ok=True)
    shutil.copyfile(tmp_path / "workflow.json", tmp_path / "images" / "workflow.json")
    return Repl(chat, Translator("en"), paths)


@pytest.mark.parametrize("moment", [True, False])
async def test_auto_draws_only_new_moments(pic, chat, tmp_path, moment, monkeypatch):
    _maker, _asked, _lena = pic
    monkeypatch.setitem(NEW_MOMENT, "draw", moment)
    repl = _repl_for(chat, tmp_path)
    chat.store.kv_set("images_auto", "on")
    sent: list[dict] = []
    with respx.mock:
        fake_comfy(sent)
        await repl.reply(lambda r: chat.send("I take off my jacket", r))
        await repl.pic_task  # the decision and drawing happen in the background
    assert bool(sent) is moment
    assert bool((chat.last_reply().meta or {}).get("images")) is moment


async def test_reopening_shows_the_pictures_under_their_replies(pic, chat, tmp_path, capsys):
    maker, _asked, _lena = pic
    async for _ in chat.send("hi", TurnResult()):
        pass
    with respx.mock:
        fake_comfy([])
        pic_ = await maker.draw()
    from PIL import Image

    Image.new("RGB", (8, 8), (200, 50, 50)).save(pic_.path)  # a real image to encode
    repl = _repl_for(chat, tmp_path)
    chat.store.kv_set("images_inline", "sixel")
    capsys.readouterr()
    repl.show_history(6)
    out = capsys.readouterr().out
    assert "\x1bPq" in out  # drawn in the terminal
    assert out.index("\x1bPq") > out.index("Hello")  # under the reply it belongs to


async def test_serve_auto_and_again(pic, chat, monkeypatch, tmp_path):
    from tests.test_serve import Client

    _maker, _asked, _lena = pic
    async for _ in chat.send("hi", TurnResult()):
        pass
    c = Client(chat)
    wf = tmp_path / "workflow.json"
    await c.call("picture.setup", path=str(wf))
    with respx.mock:
        fake_comfy([])
        monkeypatch.setitem(NEW_MOMENT, "draw", False)
        assert (await c.call("picture", auto=True))["result"] == {"skipped": True}
        monkeypatch.setitem(NEW_MOMENT, "draw", True)
        assert "path" in (await c.call("picture", auto=True))["result"]
        assert "path" in (await c.call("picture.again"))["result"]
    await c.close()


def test_loras_and_vpred_are_wired_in_and_out():
    from chatclient.images import describe, plan_of, rebuild, tune

    g = rebuild(WORKFLOW, [("a.safetensors", 0.7), ("b.safetensors", 1.0)], vpred=True)
    loras, vpred = plan_of(g)
    assert loras == [("a.safetensors", 0.7), ("b.safetensors", 1.0)] and vpred
    a, b, v = (
        n
        for n, node in g.items()
        if node.get("_meta", {}).get("title", "").startswith("chatclient:")
    )
    assert g[a]["inputs"]["model"] == ["75", 0] and g[a]["inputs"]["clip"] == ["75", 1]
    assert g[b]["inputs"]["model"] == [a, 0]
    assert g[v]["inputs"]["model"] == [b, 0] and g[v]["inputs"]["sampling"] == "v_prediction"
    assert g["70"]["inputs"]["model"] == [v, 0]  # the sampler uses the whole chain
    assert g["67"]["inputs"]["clip"] == [b, 1] and g["78"]["inputs"]["clip"] == [b, 1]
    assert g["65"]["inputs"]["vae"] == ["75", 2]  # the VAE stays with the checkpoint
    back = rebuild(g, [], vpred=False)  # taking them out restores the original wiring
    assert back == WORKFLOW
    t = tune(g, cfg=5, steps=28, sampler="euler", width=832, height=1216, checkpoint="x.ckpt")
    d = describe(t)
    assert (d["cfg"], d["steps"], d["sampler"], d["size"], d["checkpoint"]) == (
        5, 28, "euler", "832x1216", "x.ckpt")  # fmt: skip


async def test_pic_lora_and_model_commands(chat, tmp_path, capsys):
    from tests.test_repl import run_script

    (tmp_path / "images").mkdir()
    wf = tmp_path / "images" / "workflow.json"
    wf.write_text(json.dumps(WORKFLOW), encoding="utf-8")
    info = {
        "LoraLoader": {"input": {"required": {"lora_name": [["detail_lora.safetensors"]]}}},
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [
            ["anime_model.safetensors", "other_model_vpred.safetensors"]]}}},
    }  # fmt: skip
    with respx.mock:
        respx.get(url__regex=r".*/object_info/(\w+)").mock(
            side_effect=lambda r: httpx.Response(200, json={
                r.url.path.rsplit("/", 1)[1]: info[r.url.path.rsplit("/", 1)[1]]}))  # fmt: skip
        await run_script(chat, tmp_path, "/pic model\r/pic model other\r/pic lora detail 0.6\r"
                         "/pic vpred on\r/pic set cfg=5 steps=28 size=832x1216\r/quit\r")  # fmt: skip
    out = capsys.readouterr().out
    assert "2) other_model_vpred.safetensors" in out  # listed
    saved = json.loads(wf.read_text(encoding="utf-8"))
    from chatclient.images import describe

    d = describe(saved)
    assert d["checkpoint"] == "other_model_vpred.safetensors"
    assert d["loras"] == [("detail_lora.safetensors", 0.6)] and d["vpred"]
    assert (d["cfg"], d["steps"], d["size"]) == (5.0, 28, "832x1216")


async def test_tags_alone_skip_the_model(pic, chat):
    maker, asked, _lena = pic
    async for _ in chat.send("hi", TurnResult()):
        pass
    await maker.look_of(chat.character())  # the look exists already
    before = len(asked)
    sent: list[dict] = []
    with respx.mock:
        fake_comfy(sent)
        await maker.draw("+kneeling, pov")
        await maker.draw("+kneeling, pov")
    assert len(asked) == before  # no scene written by the model: exactly the user's tags
    positive = sent[0]["67"]["inputs"]["text"]
    assert positive == ("masterpiece, best quality, adult, kneeling, pov, 1girl, 22 years old, "
                        "brown hair, round glasses")  # fmt: skip
    assert sent[0]["70"]["inputs"]["seed"] != sent[1]["70"]["inputs"]["seed"]  # varied


def test_an_empty_negative_gets_a_sensible_one():
    from chatclient.images import DEFAULT_NEGATIVE, negative_for

    assert negative_for("", "") == f"{DEFAULT_NEGATIVE}, {SAFETY_NEGATIVE}"
    assert negative_for("my own", "") == f"my own, {SAFETY_NEGATIVE}"  # the user's wins
    assert negative_for("", "lowres") == f"lowres, {SAFETY_NEGATIVE}"


async def test_setup_wizard_makes_a_workflow_for_you(chat, tmp_path, capsys):
    from tests.test_repl import run_script

    from chatclient.images import describe

    info = {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [
        ["small_anime.safetensors", "big_xl_vpred.safetensors"]]}}}}  # fmt: skip
    with respx.mock:
        respx.get(url__regex=r".*/object_info/CheckpointLoaderSimple").mock(
            return_value=httpx.Response(200, json=info)
        )
        # local · default address · simple workflow · model 2 · SDXL · v-pred yes · auto no
        await run_script(chat, tmp_path, "/pic setup\r1\r\r1\r2\r2\ry\rn\r/quit\r")
    saved = json.loads((tmp_path / "images" / "workflow.json").read_text(encoding="utf-8"))
    d = describe(saved)
    assert d["checkpoint"] == "big_xl_vpred.safetensors" and d["vpred"]
    assert d["size"] == "832x1216"  # SDXL defaults
    assert chat.store.kv_get("images_backend") == "comfyui"
    assert chat.store.kv_get("images_auto") == "off"
    assert "Pictures are set up" in capsys.readouterr().out


async def test_cloud_images_come_back_as_png(tmp_path):
    import base64
    import io

    from PIL import Image

    from chatclient.config import ProviderConfig
    from chatclient.images import CloudImages
    from chatclient.providers.openai_compat import OpenAICompatProvider

    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 128, 255)).save(buf, "JPEG")
    jpeg = base64.b64encode(buf.getvalue()).decode()
    p = OpenAICompatProvider("openrouter", ProviderConfig(
        type="openai_compat", base_url="https://openrouter.ai/api/v1"), api_key="k")  # fmt: skip
    with respx.mock:
        respx.get("https://openrouter.ai/api/v1/models").mock(return_value=httpx.Response(200, json={
            "data": [{"id": "a/text-only", "architecture": {"output_modalities": ["text"]}},
                     {"id": "b/painter", "architecture": {"output_modalities": ["image", "text"]}}]}))  # fmt: skip
        sent = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": [{"message": {"images": [
                {"image_url": {"url": f"data:image/jpeg;base64,{jpeg}"}}]}}]}))  # fmt: skip
        cloud = CloudImages(p, "b/painter")
        assert await cloud.models() == ["b/painter"]  # only models that can draw
        data = await cloud.draw("a cat")
    assert data.startswith(b"\x89PNG")  # kept as PNG (the terminal and the gallery expect it)
    assert json.loads(sent.calls[0].request.content)["modalities"] == ["image", "text"]
    await p.aclose()
