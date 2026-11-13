from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from chatclient.text import estimate_tokens, normalize_for_search, short_title
from chatclient.timeutil import local_day


def test_turkish_casing_and_diacritics_fold():
    assert normalize_for_search("İSTANBUL") == "istanbul"
    assert normalize_for_search("Istanbul") == "istanbul"
    assert normalize_for_search("ıstanbul") == "istanbul"
    assert normalize_for_search("Kütüphane’deki  ŞİŞE") == "kutuphane'deki sise"
    assert normalize_for_search("Çğöşü") == "cgosu"


def test_estimate_tokens_is_conservative():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 300) >= 100


def test_short_title():
    assert short_title("merhaba") == "merhaba"
    long = "bu çok uzun bir başlık olacak ve kesilmesi gerekiyor çünkü sınırı aşıyor"
    t = short_title(long, 30)
    assert t.endswith("…") and len(t) <= 31


def test_local_day_respects_boundary():
    tz = ZoneInfo("Europe/Istanbul")  # UTC+3
    # 22:30 UTC on the 12th = 01:30 local on the 13th → still "the 12th" with a 04:00 boundary
    dt = datetime(2026, 11, 12, 22, 30, tzinfo=UTC)
    assert local_day(dt, tz, "04:00").isoformat() == "2026-11-12"
    assert local_day(dt, tz, "00:00").isoformat() == "2026-11-13"
