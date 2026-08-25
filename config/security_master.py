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

# Audit findings with insufficient evidence for a safe identifier mapping.  They
# remain readable in historical prices/trades, but candidate selection and the
# execution layer block new buys from the audit effective date.  A replacement
# must never be guessed from a similar company name/ticker.
SECURITY_LIFECYCLE = {
    **{
        ("US", ticker): {
            "status": "temporarily_unavailable",
            "effective_at": "2026-07-10T00:00:00+00:00",
            "replacement_ticker": None,
            "evidence": "2026-07 universe/provider audit; mapping evidence pending review",
        }
        for ticker in ("APLS", "BK", "BLD", "CTRA", "CWEN-A", "JHG", "MASI", "NSA")
    },
    **{
        ("US", ticker): {
            "status": "temporarily_unavailable",
            "effective_at": "2026-08-24T00:00:00+00:00",
            "replacement_ticker": None,
            "evidence": "2026-08-25 target-session provider audit: no 2026-08-24 bar; mapping evidence pending review",
        }
        for ticker in ("EA", "EQR", "LBRDA", "LBRDK", "WBS")
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
    key = (str(market).upper(), str(ticker).upper())
    change = SYMBOL_CHANGES.get(key)
    if change and _parse_utc(at) >= _parse_utc(change["effective_at"]):
        return f"symbol_changed_to_{change['new_ticker']}"
    lifecycle = SECURITY_LIFECYCLE.get(key)
    if lifecycle and _parse_utc(at) >= _parse_utc(lifecycle["effective_at"]):
        return str(lifecycle["status"])
    return None


def active_universe_tickers(
    tickers, market: str, at: str | datetime,
) -> list[str]:
    """Filter unavailable names from new-signal research universe checks.

    This must not be used to drop held symbols from ledger valuation: existing
    positions remain auditable and fail closed independently.
    """
    return [
        str(ticker) for ticker in tickers
        if ticker_lifecycle_block_reason(str(ticker), market, at) is None
    ]
