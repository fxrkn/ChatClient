"""Time helpers. Timestamps are stored in UTC; "days" are local, with a boundary hour."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def utcnow() -> datetime:
    return datetime.now(UTC)


def to_iso(dt: datetime) -> str:
    return dt.astimezone(UTC).isoformat(timespec="microseconds")


def from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


def resolve_tz(name: str) -> tzinfo | None:
    """``None`` means the system's local timezone."""
    if not name or name == "auto":
        return None
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ValueError(f"unknown timezone {name!r}") from e


def to_local(dt: datetime, tz: tzinfo | None) -> datetime:
    return dt.astimezone(tz) if tz is not None else dt.astimezone()


def local_day(dt: datetime, tz: tzinfo | None, boundary: str = "04:00") -> date:
    """The calendar day a moment belongs to, from the user's point of view.

    With a 04:00 boundary, 01:30 on the 13th counts as the night of the 12th.
    """
    hh, mm = (int(x) for x in boundary.split(":"))
    return (to_local(dt, tz) - timedelta(hours=hh, minutes=mm)).date()
