import base64
import json
import struct
import zlib

import pytest

from chatclient.char_builder import (
    WizardAnswers,
    build_card,
    clean_field,
    field_prompt,
    generate_field,
    options,
    parse_choice,
    read_card_file,
    safe_filename,
    write_card_file,
)
from chatclient.characters import Card, compile_system_prompt
from chatclient.providers.base import StreamEvent
from chatclient.providers.echo import EchoProvider


def test_parse_choice_numbers_custom_and_limits():
    opts = options("temperament", "tr")
    assert parse_choice("1", opts, 1) == [opts[0].phrase]
    assert parse_choice("1, 3", opts, 2) == [opts[0].phrase, opts[2].phrase]
    assert parse_choice("1 2 3", opts, 2) == [opts[0].phrase, opts[1].phrase]  # capped
    assert parse_choice("99", opts, 1) == []  # out of range ignored
    assert parse_choice("biraz deli", opts, 2) == ["biraz deli"]  # own words
    assert parse_choice("", opts, 2) == []


def test_build_card_turkish_reads_naturally():
    a = WizardAnswers(
        name="Deniz",
        concept="Bodrumlu bir balıkçı, 40 yaşında",
        temperament=["sakin, telaşsız", "huysuz ama kalbi iyi"],
        speech=["kısa, kuru cümleler kurar; gereksiz söz sevmez"],
        values=["özgürlük"],
        fear=["yalnız kalmak"],
        relationship=["ilk kez karşılaşıyorlar"],
        genre=["günümüzde, gündelik hayatın içinde"],
        place="şafakta, iskelede",
        example="Deniz bugün huysuz.",
    )
    card = build_card(a, "tr")
    assert card.description.startswith("{{char}}: Bodrumlu bir balıkçı, 40 yaşında.")
    assert "Hikâye günümüzde" in card.description
    assert "Mizaç: sakin, telaşsız; huysuz ama kalbi iyi." in card.personality
    assert "En büyük korkusu: yalnız kalmak." in card.personality
    assert (
        card.scenario == "{{char}} ve {{user}}: ilk kez karşılaşıyorlar. Sahne: şafakta, iskelede."
    )
    assert card.mes_example == "<START>\n{{char}}: Deniz bugün huysuz."
    text = compile_system_prompt(card, None)
    assert "Deniz: Bodrumlu" in text and "{{" not in text


def test_build_card_minimal_answers():
    card = build_card(WizardAnswers(name="X"), "en")
    assert card.name == "X" and card.description == "" and card.personality == ""


def test_clean_mes_example_normalises_markers():
    assert clean_field("mes_example", "< start >\n{{user}}: hi").startswith("<START>\n")
    assert (
        clean_field("mes_example", "{{user}}: hi\n{{char}}: yo")
        == "<START>\n{{user}}: hi\n{{char}}: yo"
    )
    assert clean_field("first_mes", '  "Hello there."  ') == "Hello there."


def test_field_prompt_is_small_and_in_language():
    card = Card(name="Rowan", description="An innkeeper.", personality="Gentle.")
    msgs = field_prompt(card, "first_mes", "tr")
    assert "Write in Turkish" in msgs[0].content
    assert "Name: Rowan" in msgs[1].content and "Task:" in msgs[1].content
    assert sum(len(m.content) for m in msgs) < 2000  # fits easily in a small model


async def test_generate_field_strips_thinking(monkeypatch):
    async def fake(self, model, messages, params):
        for piece in ["<think>hmm</think>", '"*Kapı açılıyor.* ', 'Hoş geldin."']:
            yield StreamEvent(text=piece)

    monkeypatch.setattr(EchoProvider, "stream_chat", fake)
    from chatclient.config import ProviderConfig

    p = EchoProvider("echo", ProviderConfig(type="echo"))
    text = await generate_field(p, "echo", Card(name="E"), "first_mes", "tr")
    assert text == "*Kapı açılıyor.* Hoş geldin."


# ── import / export ──────────────────────────────────────────────────────────


def _chunk(ctype: bytes, body: bytes) -> bytes:
    return struct.pack(">I", len(body)) + ctype + body + struct.pack(">I", zlib.crc32(ctype + body))


def _png(*chunks: bytes) -> bytes:
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    return b"\x89PNG\r\n\x1a\n" + ihdr + b"".join(chunks) + _chunk(b"IEND", b"")


def test_import_png_v2_text_chunk(tmp_path):
    card = {"spec": "chara_card_v2", "data": {"name": "Şule", "first_mes": "Merhaba!"}}
    encoded = base64.b64encode(json.dumps(card, ensure_ascii=False).encode()).decode()
    path = tmp_path / "sule.png"
    path.write_bytes(_png(_chunk(b"tEXt", b"chara\x00" + encoded.encode("latin-1"))))
    c = read_card_file(path)
    assert c.name == "Şule" and c.first_mes == "Merhaba!"


def test_import_png_prefers_ccv3_and_reads_itxt(tmp_path):
    v2 = base64.b64encode(json.dumps({"data": {"name": "Old"}}).encode())
    v3 = base64.b64encode(json.dumps({"spec": "chara_card_v3", "data": {"name": "New"}}).encode())
    itxt = b"ccv3\x00" + b"\x01\x00" + b"\x00" + b"\x00" + zlib.compress(v3)
    path = tmp_path / "c.png"
    path.write_bytes(_png(_chunk(b"tEXt", b"chara\x00" + v2), _chunk(b"iTXt", itxt)))
    assert read_card_file(path).name == "New"


def test_import_rejects_png_without_card(tmp_path):
    path = tmp_path / "plain.png"
    path.write_bytes(_png())
    with pytest.raises(ValueError, match="no embedded"):
        read_card_file(path)


def test_json_export_import_roundtrip(tmp_path):
    card = Card(name="Juno", description="Mechanic.", tags=["sci-fi"], extra={"creator": "me"})
    path = write_card_file(card, tmp_path / "out" / f"{safe_filename('Juno / v2')}.json")
    assert path.name == "Juno__v2.json"
    back = read_card_file(path)
    assert (back.name, back.description, back.tags) == ("Juno", "Mechanic.", ["sci-fi"])
    assert back.extra["creator"] == "me"


def test_import_json_with_bom(tmp_path):
    path = tmp_path / "bom.json"
    path.write_bytes("﻿".encode() + json.dumps({"name": "Bom"}).encode())
    assert read_card_file(path).name == "Bom"
