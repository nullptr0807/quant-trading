"""Realtime quote metadata and fail-closed validation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Iterable, Mapping


@dataclass(frozen=True)
class RealtimeQuote:
    """One realtime quote with provider time preserved separately from receipt time."""

    ticker: str
    price: float
    source: str
    source_timestamp: datetime | None
    received_at: datetime
    tradable: bool
    prev_close: float | None = None
    volume: float | None = None

    def age_seconds(self, now: datetime | None = None) -> float | None:
        if self.source_timestamp is None:
            return None
        current = _as_utc(now or datetime.now(timezone.utc))
        return max(0.0, (current - _as_utc(self.source_timestamp)).total_seconds())


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def prices_from_quotes(quotes: Mapping[str, RealtimeQuote]) -> dict[str, float]:
    """Project metadata quotes onto the legacy ``dict[str, float]`` API."""
    return {
        ticker: float(quote.price)
        for ticker, quote in quotes.items()
        if quote.price is not None and isfinite(float(quote.price)) and float(quote.price) > 0
    }


def validate_required_quotes(
    quotes: Mapping[str, RealtimeQuote],
    required_tickers: Iterable[str],
    *,
    now: datetime | None = None,
    max_age_seconds: float,
    require_source_timestamp: bool = True,
) -> dict:
    """Validate 100% positive, fresh, tradable coverage for required tickers."""
    current = _as_utc(now or datetime.now(timezone.utc))
    required = sorted({str(ticker) for ticker in required_tickers if ticker})
    missing: list[str] = []
    invalid: list[str] = []
    stale: list[str] = []
    untradable: list[str] = []
    ages: list[float] = []

    for ticker in required:
        quote = quotes.get(ticker)
        if quote is None:
            missing.append(ticker)
            continue
        try:
            price = float(quote.price)
        except (TypeError, ValueError):
            invalid.append(ticker)
            continue
        if not isfinite(price) or price <= 0:
            invalid.append(ticker)
        if not quote.tradable:
            untradable.append(ticker)
        age = quote.age_seconds(current)
        if age is None:
            if require_source_timestamp:
                stale.append(ticker)
        else:
            ages.append(age)
            if age > max_age_seconds:
                stale.append(ticker)

    valid = sorted(
        set(required) - set(missing) - set(invalid) - set(stale) - set(untradable)
    )
    expected = len(required)
    return {
        "ok": len(valid) == expected,
        "expected": expected,
        "valid": len(valid),
        "coverage": (len(valid) / expected) if expected else 1.0,
        "missing": missing,
        "invalid": invalid,
        "stale": stale,
        "untradable": untradable,
        "max_age_seconds": max(ages) if ages else None,
        "oldest_source_timestamp": min(
            (
                _as_utc(source_timestamp)
                for ticker in valid
                if (source_timestamp := quotes[ticker].source_timestamp) is not None
            ),
            default=None,
        ),
    }
