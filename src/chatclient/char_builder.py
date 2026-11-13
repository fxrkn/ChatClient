"""Free character creation: multiple-choice wizard → template card, plus optional
single-field AI helpers that are small enough for small local models.

Everything except ``generate_field`` is deterministic code (no LLM calls).
"""

from __future__ import annotations

import base64
import json
import re
import struct
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from chatclient.characters import Card
from chatclient.providers.base import ChatMessage, GenParams, Provider, ThinkFilter

# ── wizard options ──────────────────────────────────────────────────────────
# Each option: (label shown in the menu, phrase used in the card text)


@dataclass(frozen=True)
class Option:
    label: str
    phrase: str


def _opts(pairs: Sequence[tuple[str, str]]) -> list[Option]:
    return [Option(label, phrase) for label, phrase in pairs]


OPTIONS: dict[str, dict[str, list[Option]]] = {
    "temperament": {
        "tr": _opts(
            [
                ("Sıcak", "sıcak ve içten"),
                ("Mesafeli", "mesafeli, kolay açılmayan"),
                ("Şakacı", "şakacı, oyunbaz"),
                ("İğneleyici", "iğneleyici, alaycı"),
                ("Utangaç", "utangaç, çekingen"),
                ("Özgüvenli", "özgüvenli, kararlı"),
                ("Hüzünlü", "içten içe hüzünlü"),
                ("Enerjik", "enerjik, yerinde duramayan"),
                ("Sakin", "sakin, telaşsız"),
                ("Gizemli", "gizemli, az şey ele veren"),
                ("Huysuz", "huysuz ama kalbi iyi"),
                ("Meraklı", "meraklı, soru soran"),
            ]
        ),
        "en": _opts(
            [
                ("Warm", "warm and sincere"),
                ("Reserved", "reserved, slow to open up"),
                ("Playful", "playful and teasing"),
                ("Sarcastic", "sarcastic, dry-witted"),
                ("Shy", "shy and hesitant"),
                ("Confident", "confident and decisive"),
                ("Melancholic", "quietly melancholic"),
                ("Energetic", "energetic and restless"),
                ("Calm", "calm and unhurried"),
                ("Mysterious", "mysterious, gives little away"),
                ("Grumpy", "grumpy but kind-hearted"),
                ("Curious", "curious, full of questions"),
            ]
        ),
    },
    "speech": {
        "tr": _opts(
            [
                ("Kısa ve kuru", "kısa, kuru cümleler kurar; gereksiz söz sevmez"),
                ("Sıcak ve geveze", "sıcak, akıcı ve bol sohbetli konuşur"),
                ("Kibar ve resmî", "kibar, özenli ve biraz resmî konuşur"),
                ("Şiirsel", "benzetmelerle, şiirsel bir dille konuşur"),
                ("Sokak ağzı", "rahat, argolu, sokak ağzıyla konuşur"),
                ("Tedirgin", "duraksayarak, cümlelerini yarım bırakarak konuşur"),
            ]
        ),
        "en": _opts(
            [
                ("Short and dry", "speaks in short, dry sentences; no wasted words"),
                ("Warm and chatty", "talks warmly and freely, loves a good chat"),
                ("Polite and formal", "speaks politely, carefully, a little formally"),
                ("Poetic", "speaks in images and metaphors"),
                ("Street", "casual, slangy, street-smart speech"),
                ("Nervous", "hesitant, trails off mid-sentence"),
            ]
        ),
    },
    "values": {
        "tr": _opts(
            [
                ("Dürüstlük", "dürüstlük"),
                ("Sadakat", "sadakat"),
                ("Özgürlük", "özgürlük"),
                ("Aile", "aile"),
                ("Hırs", "başarı ve hırs"),
                ("Bilgi", "bilgi ve merak"),
                ("Adalet", "adalet"),
                ("Sanat", "sanat ve güzellik"),
                ("Macera", "macera"),
                ("Huzur", "huzur ve sadelik"),
            ]
        ),
        "en": _opts(
            [
                ("Honesty", "honesty"),
                ("Loyalty", "loyalty"),
                ("Freedom", "freedom"),
                ("Family", "family"),
                ("Ambition", "ambition and success"),
                ("Knowledge", "knowledge and curiosity"),
                ("Justice", "justice"),
                ("Art", "art and beauty"),
                ("Adventure", "adventure"),
                ("Peace", "peace and simplicity"),
            ]
        ),
    },
    "fear": {
        "tr": _opts(
            [
                ("Yalnızlık", "yalnız kalmak"),
                ("Başarısızlık", "başarısız olmak"),
                ("Kayıp", "sevdiklerini kaybetmek"),
                ("Zayıf görünmek", "zayıf görünmek"),
                ("Geçmişi", "geçmişinin onu bulması"),
                ("Kontrolü kaybetmek", "kontrolü kaybetmek"),
                ("Terk edilmek", "terk edilmek"),
                ("Sıradanlık", "sıradan bir hayat"),
            ]
        ),
        "en": _opts(
            [
                ("Loneliness", "being alone"),
                ("Failure", "failing"),
                ("Loss", "losing the people they love"),
                ("Looking weak", "looking weak"),
                ("The past", "their past catching up with them"),
                ("Losing control", "losing control"),
                ("Abandonment", "being abandoned"),
                ("An ordinary life", "an ordinary, wasted life"),
            ]
        ),
    },
    "relationship": {
        "tr": _opts(
            [
                ("Yabancı", "ilk kez karşılaşıyorlar"),
                ("Eski dost", "uzun yıllardır tanışan eski dostlar"),
                ("İş arkadaşı", "aynı yerde çalışıyorlar"),
                ("Komşu", "komşular; birbirlerini uzaktan tanıyorlar"),
                ("Rakip", "rakipler; aralarında tatlı sert bir çekişme var"),
                ("Akıl hocası", "{{char}} bir akıl hocası, {{user}} ise ondan öğrenen kişi"),
                ("Yol arkadaşı", "aynı yolculuğa çıkmış yol arkadaşları"),
            ]
        ),
        "en": _opts(
            [
                ("Strangers", "they are meeting for the first time"),
                ("Old friends", "old friends who go back years"),
                ("Colleagues", "they work at the same place"),
                ("Neighbours", "neighbours who know each other by sight"),
                ("Rivals", "rivals with a friendly, prickly competition"),
                ("Mentor", "{{char}} is a mentor guiding {{user}}"),
                ("Travel companions", "companions on the same journey"),
            ]
        ),
    },
    "genre": {
        "tr": _opts(
            [
                ("Günümüz", "günümüzde, gündelik hayatın içinde"),
                ("Fantastik", "büyünün gerçek olduğu fantastik bir dünyada"),
                ("Bilimkurgu", "uzak gelecekte, bilimkurgu bir dünyada"),
                ("Tarihî", "geçmiş bir yüzyılda"),
                ("Gizem", "gizemli olaylarla dolu, gerilimli bir atmosferde"),
                ("Romantik", "duygusal, romantik bir hikâyede"),
            ]
        ),
        "en": _opts(
            [
                ("Modern day", "in the present day, in everyday life"),
                ("Fantasy", "in a fantasy world where magic is real"),
                ("Sci-fi", "in the far future, in a science-fiction world"),
                ("Historical", "in a past century"),
                ("Mystery", "in a tense atmosphere full of mysteries"),
                ("Romance", "in an emotional, romantic story"),
            ]
        ),
    },
}


def options(key: str, lang: str) -> list[Option]:
    return OPTIONS[key].get(lang) or OPTIONS[key]["en"]


def parse_choice(answer: str, opts: Sequence[Option], max_n: int) -> list[str]:
    """ "1,3" → the phrases of options 1 and 3; anything else is the user's own words."""
    answer = answer.strip()
    if not answer:
        return []
    nums = [p for p in re.split(r"[,\s]+", answer) if p]
    if all(n.isdigit() for n in nums):
        picked = []
        for n in nums[:max_n]:
            i = int(n) - 1
            if 0 <= i < len(opts) and opts[i].phrase not in picked:
                picked.append(opts[i].phrase)
        return picked
    return [answer]


@dataclass
class WizardAnswers:
    name: str
    concept: str = ""
    temperament: list[str] | None = None
    speech: list[str] | None = None
    values: list[str] | None = None
    fear: list[str] | None = None
    relationship: list[str] | None = None
    genre: list[str] | None = None
    place: str = ""
    example: str = ""


def _join(items: list[str] | None, lang: str) -> str:
    # Phrases often contain commas/"and" themselves, so separate them with semicolons.
    return "; ".join(i for i in items or [] if i)


def build_card(a: WizardAnswers, lang: str) -> Card:
    """Turn wizard answers into a card, written in the interface language."""
    tr = lang == "tr"
    concept = a.concept.strip().rstrip(".")
    desc = [f"{{{{char}}}}: {concept}." if concept else ""]
    if a.genre:
        desc.append(
            f"Hikâye {_join(a.genre, lang)} geçiyor."
            if tr
            else f"The story is set {_join(a.genre, lang)}."
        )
    pers = []
    if a.temperament:
        pers.append(
            f"Mizaç: {_join(a.temperament, lang)}."
            if tr
            else f"Temperament: {_join(a.temperament, lang)}."
        )
    if a.values:
        pers.append(
            f"Değer verdikleri: {_join(a.values, lang)}."
            if tr
            else f"Values: {_join(a.values, lang)}."
        )
    if a.fear:
        pers.append(
            f"En büyük korkusu: {_join(a.fear, lang)}."
            if tr
            else f"Deepest fear: {_join(a.fear, lang)}."
        )
    if a.speech:
        pers.append(
            f"Konuşma tarzı: {_join(a.speech, lang)}."
            if tr
            else f"Speech: {_join(a.speech, lang)}."
        )
    scen = []
    if a.relationship:
        rel = _join(a.relationship, lang)
        scen.append(
            f"{{{{char}}}} ve {{{{user}}}}: {rel}."
            if tr
            else f"{{{{char}}}} and {{{{user}}}}: {rel}."
        )
    if a.place.strip():
        place = a.place.strip().rstrip(".")
        scen.append(f"Sahne: {place}." if tr else f"Scene: {place}.")
    return Card(
        name=a.name.strip(),
        description=" ".join(d for d in desc if d),
        personality="\n".join(pers),
        scenario=" ".join(scen),
        mes_example=f"<START>\n{{{{char}}}}: {a.example.strip()}" if a.example.strip() else "",
        tags=["custom"],
    )


# ── single-field AI helpers ─────────────────────────────────────────────────

LANG_NAMES = {"tr": "Turkish", "en": "English"}

FIELD_TASKS = {
    "first_mes": (
        "Write {{char}}'s opening message for a one-on-one roleplay chat with {{user}}. "
        "80–160 words. Set the scene with concrete sensory details (place, light, sound, smell). "
        "Physical actions go in *asterisks*; speech is plain text. Show {{char}}'s personality "
        "through behaviour and voice. Address {{user}} in the second person. End with a line or "
        "question {{user}} can easily respond to. Never write {{user}}'s words or actions."
    ),
    "mes_example": (
        "Write two short example exchanges that show {{char}}'s voice and personality. "
        "Use exactly this format, keeping the literal {{user}} and {{char}} markers:\n"
        "<START>\n{{user}}: (a short line)\n{{char}}: (2–4 sentences, actions in *asterisks*)\n"
        "<START>\n{{user}}: (a short line)\n{{char}}: (2–4 sentences)"
    ),
    "description": (
        "Rewrite this character description as 3–5 vivid sentences. Keep every fact, and add "
        "concrete details: appearance, clothing, a habit, and the place they spend their days. "
        "Write in the third person, refer to the character as {{char}}."
    ),
}


def field_prompt(card: Card, field: str, lang: str) -> list[ChatMessage]:
    language = LANG_NAMES.get(lang, "the same language as the character sheet")
    sheet = "\n".join(
        f"{label}: {value.strip()}"
        for label, value in (
            ("Name", card.name),
            ("Description", card.description),
            ("Personality", card.personality),
            ("Scenario", card.scenario),
        )
        if value.strip()
    )
    task = FIELD_TASKS[field]
    return [
        ChatMessage(
            "system",
            "You are a skilled fiction writer helping to build a roleplay character. "
            f"Write in {language}. Output only the requested text: no titles, notes or quotes.",
        ),
        ChatMessage("user", f"Character sheet:\n{sheet}\n\nTask: {task}"),
    ]


def clean_field(field: str, text: str) -> str:
    text = text.strip().strip('"').strip()
    if field == "mes_example":
        # Normalise markers small models tend to vary.
        text = re.sub(r"(?im)^\s*<\s*start\s*>\s*$", "<START>", text)
        if "<START>" not in text:
            text = "<START>\n" + text
    return text


async def generate_field(provider: Provider, model: str, card: Card, field: str, lang: str) -> str:
    """One short, focused request — small models handle these well."""
    params = GenParams(temperature=0.9, max_tokens=500)
    think = ThinkFilter()
    parts: list[str] = []
    async for ev in provider.stream_chat(model, field_prompt(card, field, lang), params):
        if ev.text:
            parts.append(think.feed(ev.text))
    parts.append(think.flush())
    return clean_field(field, "".join(parts))


# ── import / export ─────────────────────────────────────────────────────────

MAX_CARD_BYTES = 20 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_text_chunks(data: bytes) -> dict[str, str]:
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG file")
    out: dict[str, str] = {}
    pos = len(PNG_SIGNATURE)
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        ctype = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"tEXt":
            key, _, value = body.partition(b"\x00")
            out[key.decode("latin-1")] = value.decode("latin-1")
        elif ctype == b"iTXt":
            key, _, rest = body.partition(b"\x00")
            compressed, method = rest[0], rest[1]
            rest = rest[2:]
            _lang, _, rest = rest.partition(b"\x00")
            _tkey, _, text = rest.partition(b"\x00")
            if compressed and method == 0:
                text = zlib.decompress(text)
            out[key.decode("latin-1")] = text.decode("utf-8")
        elif ctype == b"IEND":
            break
    return out


def read_card_file(path: Path) -> Card:
    """Import a community card: CCv3/V2 JSON, or a PNG with embedded card data."""
    size = path.stat().st_size
    if size > MAX_CARD_BYTES:
        raise ValueError("file is too large to be a character card")
    raw = path.read_bytes()
    if raw.startswith(PNG_SIGNATURE):
        chunks = _png_text_chunks(raw)
        encoded = chunks.get("ccv3") or chunks.get("chara")
        if not encoded:
            raise ValueError("this PNG has no embedded character card")
        return Card.from_json(base64.b64decode(encoded).decode("utf-8"))
    return Card.from_json(raw.decode("utf-8-sig"))


def write_card_file(card: Card, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(card.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def safe_filename(name: str) -> str:
    cleaned = re.sub(r"[^\w\-. ]+", "", name, flags=re.UNICODE).strip().replace(" ", "_")
    return cleaned or "character"
