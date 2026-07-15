"""Audited security-identifier changes used by live and replay paths."""
from __future__ import annotations

from datetime import datetime, timezone


SYMBOL_CHANGES = {
    ("US", "IAC"): {
        "new_ticker": "PPLI",
        "effective_at": "2026-06-03T00:00:00+00:00",
        "ratio": 1.0,
        "evidence": "SEC CIK 1800227: IAC Inc. renamed People Inc; current Nasdaq ticker PPLI",
    },
}


def _parse_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def canonical_ticker(ticker: str, market: str, at: str | datetime) -> str:
    change = SYMBOL_CHANGES.get((str(market).upper(), str(ticker).upper()))
    if change and _parse_utc(at) >= _parse_utc(change["effective_at"]):
        return str(change["new_ticker"])
    return str(ticker)


def ticker_lifecycle_block_reason(ticker: str, market: str, at: str | datetime) -> str | None:
    change = SYMBOL_CHANGES.get((str(market).upper(), str(ticker).upper()))
    if change and _parse_utc(at) >= _parse_utc(change["effective_at"]):
        return f"symbol_changed_to_{change['new_ticker']}"
    return None
