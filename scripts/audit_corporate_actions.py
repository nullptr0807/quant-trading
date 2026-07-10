#!/usr/bin/env python3
"""Read-only corporate-action audit for paper accounts.

Finds stock splits / bonus share events that occurred while an account held a
position, then estimates whether current positions or historical realized PnL
may be wrong because the ledger never adjusted shares/avg_cost.

READ ONLY: writes reports under /tmp/hermes/corporate_action_audit/ only.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "trading.db"
OUT_DIR = Path("/tmp/hermes/corporate_action_audit")
CN_SUFFIX_RE = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")


@dataclass
class Action:
    ticker: str
    market: str
    ex_date: str
    action_type: str
    ratio: float | None = None         # split/bonus multiplier: 4:1 split => 4.0, 10转4 => 1.4
    cash_per_share: float | None = None
    description: str = ""
    source: str = ""
    raw: str = ""


@dataclass
class AffectedInterval:
    account: str
    market: str
    ticker: str
    ex_date: str
    action_type: str
    ratio: float | None
    cash_per_share: float | None
    description: str
    held_shares_before: float
    open_after_action: bool
    current_db_shares: float | None
    expected_current_shares: float | None
    current_price: float | None
    estimated_current_equity_impact: float | None
    interval_start: str | None
    interval_end: str | None
    source: str
    severity: str


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def parse_dt(s: Any) -> pd.Timestamp | None:
    if s is None:
        return None
    try:
        return pd.to_datetime(str(s), utc=True)
    except Exception:
        return None


def date_str(x: Any) -> str | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s in {"NaT", "nan", "None", "--"}:
        return None
    try:
        return pd.to_datetime(s).strftime("%Y-%m-%d")
    except Exception:
        return s[:10] if len(s) >= 10 else None


def is_cn_ticker(ticker: str) -> bool:
    return bool(CN_SUFFIX_RE.match(str(ticker)))


def code6(ticker: str) -> str:
    return str(ticker).split(".")[0]


def load_relevant_tickers(con: sqlite3.Connection, markets: set[str], focus_accounts: set[str] | None) -> list[tuple[str, str]]:
    params: list[Any] = []
    where = ["market IN (%s)" % ",".join("?" for _ in markets)]
    params.extend(sorted(markets))
    if focus_accounts:
        where.append("account IN (%s)" % ",".join("?" for _ in focus_accounts))
        params.extend(sorted(focus_accounts))
    rows = con.execute(
        f"SELECT DISTINCT market, ticker FROM trades WHERE {' AND '.join(where)} ORDER BY market,ticker",
        params,
    ).fetchall()
    # Also include currently open positions (may have no trades in focused window due legacy restores)
    params2: list[Any] = []
    where2 = ["market IN (%s)" % ",".join("?" for _ in markets)]
    params2.extend(sorted(markets))
    if focus_accounts:
        where2.append("account IN (%s)" % ",".join("?" for _ in focus_accounts))
        params2.extend(sorted(focus_accounts))
    rows2 = con.execute(
        f"SELECT DISTINCT market, ticker FROM positions WHERE shares>0 AND {' AND '.join(where2)} ORDER BY market,ticker",
        params2,
    ).fetchall()
    out = {(r["market"], r["ticker"]) for r in rows} | {(r["market"], r["ticker"]) for r in rows2}
    return sorted(out)


def _iter_action_series(raw: Any, preferred_column: str):
    """Yield ``(index, value)`` from yfinance Series/DataFrame variants.

    Recent yfinance releases sometimes return a one-column DataFrame whose
    column label (``Stock Splits``/``Dividends``) is itself encountered by naive
    ``Series.items`` assumptions. Normalize explicitly so an API shape change
    cannot turn every split into a DateParseError.
    """
    if raw is None:
        return []
    if isinstance(raw, pd.DataFrame):
        if raw.empty:
            return []
        if preferred_column in raw.columns:
            series = raw[preferred_column]
        elif len(raw.columns) == 1:
            series = raw.iloc[:, 0]
        else:
            raise ValueError(
                f"ambiguous action frame columns: {list(map(str, raw.columns))}"
            )
    elif isinstance(raw, pd.Series):
        series = raw
    else:
        series = pd.Series(raw)
    return list(series.items())


def _fetch_yahoo_chart_events(ticker: str, start: str, end: str) -> list[Action] | None:
    """Independent Yahoo chart endpoint fallback for yfinance object failures."""
    import urllib.parse
    import urllib.request
    start_epoch = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_epoch = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    symbol = urllib.parse.quote(ticker, safe="")
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        f"?period1={start_epoch}&period2={end_epoch}&interval=1d&events=div%2Csplits"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        payload = json.load(urllib.request.urlopen(req, timeout=15))
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if not result:
            return None
    except Exception:
        return None
    actions: list[Action] = []
    events = result.get("events") or {}
    for event in (events.get("splits") or {}).values():
        d = datetime.fromtimestamp(int(event["date"]), tz=timezone.utc).strftime("%Y-%m-%d")
        ratio = float(event.get("numerator", 0)) / float(event.get("denominator", 1))
        actions.append(Action(ticker=ticker, market="US", ex_date=d, action_type="split",
                              ratio=ratio, description=event.get("splitRatio", ""),
                              source="yahoo.chart.events", raw=json.dumps(event)))
    for event in (events.get("dividends") or {}).values():
        d = datetime.fromtimestamp(int(event["date"]), tz=timezone.utc).strftime("%Y-%m-%d")
        actions.append(Action(ticker=ticker, market="US", ex_date=d, action_type="cash_dividend",
                              cash_per_share=float(event["amount"]), description="chart dividend",
                              source="yahoo.chart.events", raw=json.dumps(event)))
    return actions


def fetch_us_actions(ticker: str, start: str, end: str) -> list[Action]:
    import yfinance as yf

    actions: list[Action] = []
    tk = yf.Ticker(ticker)
    # Splits
    try:
        splits = tk.splits
        for idx, val in _iter_action_series(splits, "Stock Splits"):
            d = pd.to_datetime(idx).strftime("%Y-%m-%d")
            if start <= d <= end:
                ratio = float(val)
                if ratio and math.isfinite(ratio) and abs(ratio - 1.0) > 1e-9:
                    actions.append(Action(
                        ticker=ticker, market="US", ex_date=d, action_type="split",
                        ratio=ratio, description=f"split {ratio}:1", source="yfinance.splits",
                        raw=json.dumps({"date": d, "ratio": ratio}, ensure_ascii=False),
                    ))
    except Exception as e:
        actions.append(Action(ticker=ticker, market="US", ex_date="", action_type="fetch_error", description=repr(e), source="yfinance.splits"))

    # Dividends are cash actions, not share-count actions. Include for context.
    try:
        divs = tk.dividends
        for idx, val in _iter_action_series(divs, "Dividends"):
            d = pd.to_datetime(idx).strftime("%Y-%m-%d")
            if start <= d <= end:
                amount = float(val)
                if amount and math.isfinite(amount):
                    actions.append(Action(
                        ticker=ticker, market="US", ex_date=d, action_type="cash_dividend",
                        cash_per_share=amount, description=f"cash dividend {amount}", source="yfinance.dividends",
                        raw=json.dumps({"date": d, "amount": amount}, ensure_ascii=False),
                    ))
    except Exception as e:
        actions.append(Action(ticker=ticker, market="US", ex_date="", action_type="fetch_error", description=repr(e), source="yfinance.dividends"))
    return actions


def fetch_cn_actions(ticker: str, start: str, end: str) -> list[Action]:
    import akshare as ak

    code = code6(ticker)
    actions: list[Action] = []
    try:
        df = ak.stock_fhps_detail_em(symbol=code)
    except Exception as primary_error:
        # Eastmoney intermittently returns None/shape errors for STAR names.
        # Independent THS data can prove an equity has no implemented action;
        # index tickers have no issuer corporate actions by definition.
        if code in {"000300", "000001", "399001", "399006", "000016", "000905"}:
            return []
        try:
            ths = ak.stock_fhps_detail_ths(symbol=code)
        except Exception as fallback_error:
            return [Action(
                ticker=ticker, market="CN", ex_date="", action_type="fetch_error",
                description=f"primary={primary_error!r}; fallback={fallback_error!r}",
                source="akshare.stock_fhps_detail_em+ths",
            )]
        if ths is None or ths.empty:
            return []
        # THS fallback is sufficient to classify no-action rows. Parsing complex
        # implemented plans remains fail-closed until Eastmoney recovers.
        relevant = ths[
            ths["A股除权除息日"].notna()
            & ths["A股除权除息日"].astype(str).map(
                lambda value: bool((d := date_str(value)) and start <= d <= end)
            )
        ]
        if relevant.empty:
            return []
        return [Action(
            ticker=ticker, market="CN", ex_date="", action_type="fetch_error",
            description="THS reports implemented actions but primary structured ratio source failed",
            source="akshare.stock_fhps_detail_ths",
            raw=relevant.to_json(force_ascii=False, date_format="iso"),
        )]
    if df is None or df.empty:
        return []
    for _, row in df.iterrows():
        ex = date_str(row.get("除权除息日"))
        if not ex or not (start <= ex <= end):
            continue
        desc = str(row.get("现金分红-现金分红比例描述") or "").strip()
        raw = row.to_json(force_ascii=False, date_format="iso")
        # Eastmoney's 送转股份-送转总比例 appears as shares per 10 shares. 10转4 => 4.0 => multiplier 1.4
        transfer = row.get("送转股份-送转总比例")
        try:
            transfer_f = float(transfer)
        except Exception:
            transfer_f = float("nan")
        if math.isfinite(transfer_f) and abs(transfer_f) > 1e-12:
            ratio = 1.0 + transfer_f / 10.0
            actions.append(Action(
                ticker=ticker, market="CN", ex_date=ex, action_type="bonus_or_transfer",
                ratio=ratio, description=desc or f"10转/送 {transfer_f:g}",
                source="akshare.stock_fhps_detail_em", raw=raw,
            ))
        cash = row.get("现金分红-现金分红比例")
        try:
            cash_f = float(cash)
        except Exception:
            cash_f = float("nan")
        if math.isfinite(cash_f) and abs(cash_f) > 1e-12:
            # Eastmoney gives cash per 10 shares.
            actions.append(Action(
                ticker=ticker, market="CN", ex_date=ex, action_type="cash_dividend",
                cash_per_share=cash_f / 10.0, description=desc or f"10派{cash_f:g}",
                source="akshare.stock_fhps_detail_em", raw=raw,
            ))
    return actions


def trade_rows(con: sqlite3.Connection, account: str, market: str, ticker: str) -> list[sqlite3.Row]:
    return con.execute(
        "SELECT * FROM trades WHERE account=? AND market=? AND ticker=? ORDER BY timestamp,id",
        (account, market, ticker),
    ).fetchall()


def current_position(con: sqlite3.Connection, account: str, market: str, ticker: str) -> sqlite3.Row | None:
    return con.execute(
        "SELECT * FROM positions WHERE account=? AND market=? AND ticker=?",
        (account, market, ticker),
    ).fetchone()


def accounts_for_ticker(con: sqlite3.Connection, market: str, ticker: str, focus_accounts: set[str] | None) -> list[str]:
    params: list[Any] = [market, ticker]
    where = "market=? AND ticker=?"
    if focus_accounts:
        where += " AND account IN (%s)" % ",".join("?" for _ in focus_accounts)
        params.extend(sorted(focus_accounts))
    rows = con.execute(f"SELECT DISTINCT account FROM trades WHERE {where}", params).fetchall()
    params2: list[Any] = [market, ticker]
    where2 = "market=? AND ticker=? AND shares>0"
    if focus_accounts:
        where2 += " AND account IN (%s)" % ",".join("?" for _ in focus_accounts)
        params2.extend(sorted(focus_accounts))
    rows2 = con.execute(f"SELECT DISTINCT account FROM positions WHERE {where2}", params2).fetchall()
    return sorted({r["account"] for r in rows} | {r["account"] for r in rows2})


def intervals_held(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """Approximate holding intervals from trade rows, ignoring corporate actions."""
    intervals: list[dict[str, Any]] = []
    shares = 0.0
    start_ts: str | None = None
    shares_before = 0.0
    for r in rows:
        ts = r["timestamp"]
        side = str(r["side"]).lower()
        qty = float(r["shares"] or 0.0)
        if side == "buy":
            if shares <= 1e-9:
                start_ts = ts
                shares_before = 0.0
            shares += qty
        elif side == "sell":
            if shares > 1e-9 and start_ts is not None:
                shares -= qty
                if shares <= 1e-9:
                    intervals.append({"start": start_ts, "end": ts})
                    start_ts = None
                    shares = max(0.0, shares)
            else:
                # oversell/legacy dirty trade; ignore for interval construction
                shares = max(0.0, shares - qty)
    if shares > 1e-9 and start_ts is not None:
        intervals.append({"start": start_ts, "end": None})
    return intervals


def shares_before_action(rows: list[sqlite3.Row], ex_date: str) -> float:
    cutoff = pd.Timestamp(ex_date).tz_localize("UTC") + pd.Timedelta(days=1)
    shares = 0.0
    for r in rows:
        ts = parse_dt(r["timestamp"])
        if ts is None or ts >= cutoff:
            continue
        qty = float(r["shares"] or 0.0)
        if str(r["side"]).lower() == "buy":
            shares += qty
        elif str(r["side"]).lower() == "sell":
            shares -= qty
    return max(0.0, shares)


def action_in_interval(action_date: str, interval: dict[str, Any]) -> bool:
    ad = pd.Timestamp(action_date).tz_localize("UTC")
    st = parse_dt(interval["start"])
    en = parse_dt(interval["end"]) if interval.get("end") else None
    if st is None:
        return False
    # Action applies if held at start of ex-date; conservatively include if bought before end of ex-date.
    day_end = ad + pd.Timedelta(days=1)
    if st >= day_end:
        return False
    if en is not None and en < ad:
        return False
    return True


def estimate_current_impact(pos: sqlite3.Row | None, ratio: float | None) -> tuple[float | None, float | None, float | None]:
    if pos is None or ratio is None or not math.isfinite(ratio):
        return None, None, None
    old_sh = float(pos["shares"] or 0.0)
    px = pos["current_price"]
    px_f = float(px) if px is not None else None
    expected = old_sh * ratio
    impact = (expected - old_sh) * px_f if px_f is not None else None
    return expected, px_f, impact


def run_audit(markets: set[str], focus_accounts: set[str] | None, start: str, end: str, max_tickers: int | None) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = OUT_DIR / "corporate_actions_cache.json"
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}
    else:
        cache = {}

    con = connect()
    tickers = load_relevant_tickers(con, markets, focus_accounts)
    if max_tickers:
        tickers = tickers[:max_tickers]
    actions: list[Action] = []
    fetch_errors: list[dict[str, Any]] = []

    for i, (market, ticker) in enumerate(tickers, 1):
        # Version the cache by parser contract. Never let provider/API-shape
        # failures cached by older code survive after the parser is fixed.
        key = f"v2:{market}:{ticker}:{start}:{end}"
        if key in cache:
            acts = [Action(**a) for a in cache[key]]
            if any(a.action_type == "fetch_error" for a in acts):
                acts = None
        else:
            acts = None
        if acts is None:
            print(f"[{i}/{len(tickers)}] fetch {market} {ticker}", flush=True)
            if market == "US":
                acts = fetch_us_actions(ticker, start, end)
                time.sleep(0.02)
            elif market == "CN":
                acts = fetch_cn_actions(ticker, start, end)
                time.sleep(0.08)
            else:
                acts = []
            cache[key] = [asdict(a) for a in acts]
            if i % 25 == 0:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
        for a in acts:
            if a.action_type == "fetch_error":
                fetch_errors.append(asdict(a))
            else:
                actions.append(a)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))

    affected: list[AffectedInterval] = []
    by_ticker_action: dict[tuple[str, str, str], Action] = {}
    for a in actions:
        by_ticker_action[(a.market, a.ticker, a.ex_date + ":" + a.action_type)] = a
        accounts = accounts_for_ticker(con, a.market, a.ticker, focus_accounts)
        for acct in accounts:
            rows = trade_rows(con, acct, a.market, a.ticker)
            intervals = intervals_held(rows)
            hit = any(action_in_interval(a.ex_date, iv) for iv in intervals)
            if not hit:
                continue
            held_before = shares_before_action(rows, a.ex_date)
            pos = current_position(con, acct, a.market, a.ticker)
            open_after = pos is not None and float(pos["shares"] or 0.0) > 0
            expected, current_px, impact = estimate_current_impact(pos, a.ratio)
            if a.action_type in {"split", "bonus_or_transfer"} and a.ratio and held_before > 1e-9:
                severity = "critical" if open_after else "warning"
            elif a.action_type == "cash_dividend" and held_before > 1e-9:
                severity = "info"
            else:
                severity = "info"
            # Find interval containing action for display
            iv0 = next((iv for iv in intervals if action_in_interval(a.ex_date, iv)), {"start": None, "end": None})
            affected.append(AffectedInterval(
                account=acct, market=a.market, ticker=a.ticker, ex_date=a.ex_date,
                action_type=a.action_type, ratio=a.ratio, cash_per_share=a.cash_per_share,
                description=a.description, held_shares_before=held_before,
                open_after_action=open_after,
                current_db_shares=float(pos["shares"]) if pos else None,
                expected_current_shares=expected,
                current_price=current_px,
                estimated_current_equity_impact=impact,
                interval_start=iv0.get("start"), interval_end=iv0.get("end"),
                source=a.source, severity=severity,
            ))

    con.close()

    actions_csv = OUT_DIR / "corporate_actions_found.csv"
    affected_csv = OUT_DIR / "affected_account_intervals.csv"
    errors_csv = OUT_DIR / "corporate_action_fetch_errors.csv"
    with actions_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(actions[0]).keys()) if actions else ["ticker", "market", "ex_date", "action_type", "ratio", "cash_per_share", "description", "source", "raw"])
        writer.writeheader()
        for a in actions:
            writer.writerow(asdict(a))
    with affected_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(affected[0]).keys()) if affected else list(AffectedInterval.__dataclass_fields__.keys()))
        writer.writeheader()
        for a in affected:
            writer.writerow(asdict(a))
    with errors_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ticker", "market", "ex_date", "action_type", "ratio", "cash_per_share", "description", "source", "raw"])
        writer.writeheader()
        for e in fetch_errors:
            writer.writerow(e)

    # Summaries
    df_aff = pd.DataFrame([asdict(a) for a in affected])
    summary: dict[str, Any] = {
        "tickers_scanned": len(tickers),
        "actions_found": len(actions),
        "share_count_actions_found": sum(1 for a in actions if a.action_type in {"split", "bonus_or_transfer"}),
        "cash_dividends_found": sum(1 for a in actions if a.action_type == "cash_dividend"),
        "affected_intervals": len(affected),
        "fetch_errors": len(fetch_errors),
    }
    if not df_aff.empty:
        summary.update({
            "affected_accounts": int(df_aff["account"].nunique()),
            "affected_tickers": int(df_aff["ticker"].nunique()),
            "critical_open_share_actions": int(((df_aff["severity"] == "critical") & (df_aff["action_type"].isin(["split", "bonus_or_transfer"]))).sum()),
            "estimated_open_equity_impact_abs": float(pd.to_numeric(df_aff["estimated_current_equity_impact"], errors="coerce").abs().sum(skipna=True)),
            "estimated_open_equity_impact_net": float(pd.to_numeric(df_aff["estimated_current_equity_impact"], errors="coerce").sum(skipna=True)),
        })

    # Markdown report
    md_path = OUT_DIR / "corporate_action_audit.md"
    md: list[str] = []
    md.append("# Corporate action audit (read-only)\n\n")
    md.append(f"DB: `{DB_PATH}`  \nRange: `{start}` → `{end}`  \nMarkets: `{', '.join(sorted(markets))}`  \n")
    if focus_accounts:
        md.append(f"Focus accounts: `{', '.join(sorted(focus_accounts))}`  \n")
    md.append("\n## Summary\n\n")
    for k, v in summary.items():
        md.append(f"- {k}: **{v}**\n")
    if not df_aff.empty:
        md.append("\n## Highest estimated open-position impacts\n\n")
        d = df_aff.copy()
        d["impact_abs"] = pd.to_numeric(d["estimated_current_equity_impact"], errors="coerce").abs()
        d = d.sort_values(["severity", "impact_abs"], ascending=[True, False])
        md.append("|severity|acct|mkt|ticker|ex_date|action|ratio|held_before|current_shares|expected_shares|px|impact|desc|\n")
        md.append("|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for _, r in d.head(60).iterrows():
            md.append(
                f"|{r.get('severity')}|{r.get('account')}|{r.get('market')}|{r.get('ticker')}|{r.get('ex_date')}|{r.get('action_type')}|"
                f"{r.get('ratio') if pd.notna(r.get('ratio')) else ''}|{r.get('held_shares_before'):.4g}|"
                f"{r.get('current_db_shares') if pd.notna(r.get('current_db_shares')) else ''}|"
                f"{r.get('expected_current_shares') if pd.notna(r.get('expected_current_shares')) else ''}|"
                f"{r.get('current_price') if pd.notna(r.get('current_price')) else ''}|"
                f"{r.get('estimated_current_equity_impact') if pd.notna(r.get('estimated_current_equity_impact')) else ''}|"
                f"{str(r.get('description'))[:80]}|\n"
            )
        md.append("\n## Affected counts by action type\n\n")
        md.append(df_aff.groupby(["market", "action_type", "severity"]).size().reset_index(name="count").to_markdown(index=False))
        md.append("\n")
    md.append("\n## Output files\n\n")
    md.append(f"- actions: `{actions_csv}`\n")
    md.append(f"- affected intervals: `{affected_csv}`\n")
    md.append(f"- fetch errors: `{errors_csv}`\n")
    md.append("\n## Notes\n\n")
    md.append("- This script does **not** modify the DB.\n")
    md.append("- `critical` means an account appears to still hold shares after a share-count action; current positions likely need adjustment if the action was not already applied.\n")
    md.append("- `warning` means a share-count action occurred during a historical holding interval that later closed; realized PnL should be replayed with the action.\n")
    md.append("- Cash dividends are listed as `info`; cash-credit policy still needs to be defined before repair.\n")
    md_path.write_text("".join(md))
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    return {
        "summary": summary,
        "paths": {
            "summary": str(summary_path),
            "md": str(md_path),
            "actions_csv": str(actions_csv),
            "affected_csv": str(affected_csv),
            "errors_csv": str(errors_csv),
            "cache": str(cache_path),
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Read-only corporate action audit")
    p.add_argument("--markets", default="US,CN", help="comma-separated markets")
    p.add_argument("--accounts", default="", help="optional comma-separated focus accounts")
    p.add_argument("--start", default="2026-01-01")
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    p.add_argument("--max-tickers", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    markets = {x.strip().upper() for x in args.markets.split(",") if x.strip()}
    focus = {x.strip() for x in args.accounts.split(",") if x.strip()} or None
    result = run_audit(markets, focus, args.start, args.end, args.max_tickers)
    print("SUMMARY_JSON")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
