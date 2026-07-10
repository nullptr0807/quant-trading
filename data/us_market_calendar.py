"""NYSE/Nasdaq regular-session calendar gates for US execution."""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache


@lru_cache(maxsize=1)
def _calendar():
    """XNYS and XNAS share the regular equity-session calendar used here."""
    import exchange_calendars as xcals

    return xcals.get_calendar("XNYS")


def is_us_regular_session(now: datetime | None = None) -> bool:
    """Return True only inside the current NYSE regular session.

    The exchange calendar supplies holidays, DST-adjusted opens/closes, and
    special closes (for example the 13:00 ET Black Friday half-day). Fail
    closed when the calendar dependency or timestamp conversion is unavailable.
    """
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    try:
        import pandas as pd

        timestamp = pd.Timestamp(current.astimezone(timezone.utc))
        return bool(_calendar().is_open_on_minute(timestamp, ignore_breaks=False))
    except Exception:
        return False


def us_regular_session_bounds(now: datetime | None = None) -> tuple[datetime, datetime] | None:
    """Return today's regular-session UTC bounds, or None on a non-session day."""
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    try:
        import pandas as pd
        from zoneinfo import ZoneInfo

        local_day = current.astimezone(ZoneInfo("America/New_York")).date()
        session = pd.Timestamp(local_day)
        calendar = _calendar()
        if not calendar.is_session(session):
            return None
        opened = calendar.session_open(session).to_pydatetime()
        closed = calendar.session_close(session).to_pydatetime()
        return opened.astimezone(timezone.utc), closed.astimezone(timezone.utc)
    except Exception:
        return None
