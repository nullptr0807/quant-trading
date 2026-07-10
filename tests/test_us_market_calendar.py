from __future__ import annotations

from datetime import datetime, timezone


def test_us_regular_session_calendar_normal_day_boundaries():
    from data.us_market_calendar import is_us_regular_session

    assert not is_us_regular_session(datetime(2026, 7, 10, 13, 29, tzinfo=timezone.utc))
    assert is_us_regular_session(datetime(2026, 7, 10, 13, 30, tzinfo=timezone.utc))
    assert is_us_regular_session(datetime(2026, 7, 10, 19, 59, tzinfo=timezone.utc))
    assert not is_us_regular_session(datetime(2026, 7, 10, 20, 0, tzinfo=timezone.utc))


def test_us_regular_session_calendar_holiday_and_weekend():
    from data.us_market_calendar import is_us_regular_session

    assert not is_us_regular_session(datetime(2026, 7, 3, 15, 0, tzinfo=timezone.utc))
    assert not is_us_regular_session(datetime(2026, 7, 4, 15, 0, tzinfo=timezone.utc))
    assert not is_us_regular_session(datetime(2026, 12, 25, 15, 0, tzinfo=timezone.utc))


def test_us_regular_session_calendar_half_day_close():
    from data.us_market_calendar import is_us_regular_session, us_regular_session_bounds

    bounds = us_regular_session_bounds(datetime(2026, 11, 27, 16, 0, tzinfo=timezone.utc))
    assert bounds is not None
    opened, closed = bounds
    assert opened.isoformat() == "2026-11-27T14:30:00+00:00"
    assert closed.isoformat() == "2026-11-27T18:00:00+00:00"
    assert is_us_regular_session(datetime(2026, 11, 27, 17, 59, tzinfo=timezone.utc))
    assert not is_us_regular_session(datetime(2026, 11, 27, 18, 0, tzinfo=timezone.utc))


def test_us_regular_session_calendar_dst_adjusts_utc_open():
    from data.us_market_calendar import us_regular_session_bounds

    summer = us_regular_session_bounds(datetime(2026, 7, 10, 15, 0, tzinfo=timezone.utc))
    winter = us_regular_session_bounds(datetime(2026, 1, 9, 16, 0, tzinfo=timezone.utc))
    assert summer is not None and winter is not None
    assert summer[0].isoformat() == "2026-07-10T13:30:00+00:00"
    assert winter[0].isoformat() == "2026-01-09T14:30:00+00:00"
