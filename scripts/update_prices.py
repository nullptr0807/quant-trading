"""Per-minute price-only updater + intraday stop-loss watchdog.

Lightweight companion to the hourly run_once(). Runs every minute during market
hours and:
  1. Reads current holdings from DB.
  2. Pulls realtime quotes via the market-aware fetcher (data.cn_fetcher.get_fetcher_for).
  3. For each position: if pnl_pct <= -stop_loss (per-strategy config),
     execute an immediate sell (respecting slippage + fees). Persists trade,
     updates cash + removes position.
  4. Recomputes each account's equity = cash + Σ(shares × realtime_price).
  5. Writes a fresh snapshot into `accounts` table + updates
     `positions.current_price` so the dashboard shows intraday movement.

Does NOT:
  - fetch historical data
  - compute factors / mine GP
  - make buy decisions (only protective sells)
  - send Telegram reports

Safe to run every minute via cron. Typical wall-time 2-5s.

Usage:
    python -m scripts.update_prices
"""
from __future__ import annotations

import logging
import os
import sys
import sqlite3
import time
from datetime import datetime, timezone

# Make sibling packages importable when run as module or script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from accounts.strategies import STRATEGIES  # noqa: E402
from accounts.gp_strategies import active_gp_strategies_for_market  # noqa: E402
from config.settings import ACCOUNT_PREFIX  # noqa: E402
from trading.costs import CNCosts, MoomooAUCosts  # noqa: E402
from data.cn_fetcher import get_fetcher_for  # noqa: E402

LOG = logging.getLogger("update_prices")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trading.db"
)

# Build stop-loss lookup per account
STOP_LOSS_BY_ACCT: dict[str, float] = {}
for _s in STRATEGIES:
    STOP_LOSS_BY_ACCT[_s.id] = _s.stop_loss
for _g in active_gp_strategies_for_market("US"):
    STOP_LOSS_BY_ACCT[_g.id] = _g.stop_loss
for _g in active_gp_strategies_for_market("CN"):
    STOP_LOSS_BY_ACCT[f"{ACCOUNT_PREFIX.get('CN', '')}{_g.id}"] = _g.stop_loss

_COST_MODELS = {"US": MoomooAUCosts(), "CN": CNCosts()}


def _costs_for_market(market: str):
    return _COST_MODELS.get(market, _COST_MODELS["US"])


def _currency_symbol(market: str) -> str:
    return "¥" if market == "CN" else "$"


def _notify_stop_losses(executed: list[dict]) -> None:
    """Send a compact Telegram alert listing every stop-loss sell from this tick."""
    if not executed:
        return
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token or os.environ.get("QUANT_DISABLE_TELEGRAM") == "1":
        LOG.debug("Telegram disabled — skipping alert")
        return
    try:
        from reports.telegram import TelegramReporter
        reporter = TelegramReporter(token=token)
    except Exception as e:
        LOG.warning("TelegramReporter init failed: %s", e)
        return

    lines = [f"\U0001F6D1 盘中止损触发 ({len(executed)} 笔)"]
    # Group by account for readability
    by_acct: dict[str, list[dict]] = {}
    for e in executed:
        by_acct.setdefault(e["account"], []).append(e)
    for acct in sorted(by_acct):
        for e in by_acct[acct]:
            lines.append(
                f"  [{acct}] {e['ticker']} {e['shares']:.0f}股 @ ${e['price']:.2f}  "
                f"成本${e['avg_cost']:.2f}  PnL {e['pnl_pct']:+.2f}% "
                f"(阈值 -{e['stop_loss']:.1f}%)"
            )
    try:
        reporter.send_message("\n".join(lines))
    except Exception as e:
        LOG.warning("Telegram send_message failed: %s", e)


_CN_SUFFIXES = (".SH", ".SZ", ".BJ")


def _is_cn(ticker: str) -> bool:
    return ticker.upper().endswith(_CN_SUFFIXES)


# Fetcher singletons — instantiated lazily, reused per cron tick.
_FETCHERS: dict[str, object] = {}


def _fetcher_for(market: str):
    f = _FETCHERS.get(market)
    if f is None:
        f = get_fetcher_for(market)
        _FETCHERS[market] = f
    return f


def fetch_quotes(tickers: list[str]) -> dict[str, float]:
    """Fetch latest realtime price per ticker via the market-aware fetcher.

    Single source of truth — same path as main.py. US tickers go through
    DataFetcher (yfinance fast_info → Finnhub fallback); CN tickers go
    through CNDataFetcher (akshare 1m bar). Without this unification the
    every-minute updater and the hourly cycle quoted off different sources
    and equity ping-ponged at every :01.

    Returns {ticker: last_price}. Tickers we could not quote are absent.
    """
    if not tickers:
        return {}

    us = [t for t in tickers if not _is_cn(t)]
    cn = [t for t in tickers if _is_cn(t)]

    prices: dict[str, float] = {}
    if us:
        try:
            prices.update(_fetcher_for("US").get_realtime_quotes(us))
        except Exception as e:
            LOG.warning("US realtime fetch failed: %s", e)
    if cn:
        try:
            prices.update(_fetcher_for("CN").get_realtime_quotes(cn))
        except Exception as e:
            LOG.warning("CN realtime fetch failed: %s", e)
    return prices


def _force_price_update() -> bool:
    """Manual repair mode: refresh prices/snapshots outside market hours.

    This must NOT make the stop-loss path think the market is tradeable; forced
    refresh is for stale snapshot repair only, not for weekend/off-hours sells.
    """
    return os.environ.get("QUANT_FORCE_PRICE_UPDATE") == "1"


def _is_us_market_hours_now() -> bool:
    """US extended hours: Mon-Fri 04:00-20:00 ET. Mirrors main.is_market_hours()."""
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    h = now.hour
    return 8 <= h or h < 1


def _is_cn_market_hours_now() -> bool:
    """CN A-share hours: Mon-Fri 09:30-11:30 + 13:00-15:00 CST (UTC+8).

    Mirrors main.is_market_hours_cn(). Gate for writing CN-account snapshots
    so we don't flood accounts table with identical equity values after the
    CN session has closed (equity curve visually compresses earlier days).
    """
    try:
        from zoneinfo import ZoneInfo
        cst = ZoneInfo("Asia/Shanghai")
    except Exception:
        return False
    now = datetime.now(cst)
    if now.weekday() >= 5:
        return False
    m = now.hour * 60 + now.minute
    return (9 * 60 + 30 <= m < 11 * 60 + 30) or (13 * 60 <= m < 15 * 60)


def _is_market_hours_now() -> bool:
    """Back-compat aggregate market-hours check."""
    return _is_us_market_hours_now() or _is_cn_market_hours_now()


def _is_market_open_for(market: str) -> bool:
    """Return whether a specific account market is currently tradeable."""
    if market == "CN":
        return _is_cn_market_hours_now()
    return _is_us_market_hours_now()


def _insert_trade_event(
    cur: sqlite3.Cursor,
    *,
    now_iso: str,
    market: str,
    account: str,
    ticker: str,
    shares: float,
    price: float,
    avg_cost: float,
    pnl_pct: float,
    pnl_dollar: float,
    stop_loss: float,
    reason: str,
) -> None:
    """Insert the dashboard trade event in the SAME SQLite transaction.

    Do not call core.events.emit() here: it opens a second SQLite connection
    while update_prices is already holding a write transaction, which causes
    `database is locked` and drops the audit event even though the trade row is
    committed. Keeping the event write on this cursor makes trades/events
    atomic for the stop-loss path.
    """
    sym = _currency_symbol(market)
    reason_label = {"stop_loss": "🛑 stop-loss", "trailing_stop": "📉 trailing-stop"}.get(reason, reason)
    title = (
        f"SELL {ticker} ×{int(shares)} @ {sym}{price:.2f} · "
        f"{reason_label} PnL {pnl_pct*100:+.2f}% ({sym}{pnl_dollar:+.2f})"
    )
    detail = {
        "shares": shares,
        "price": price,
        "reason": reason,
        "pnl_pct": pnl_pct,
        "pnl_dollar": pnl_dollar,
        "avg_cost": avg_cost,
        "stop_loss_threshold": stop_loss,
        "source": "update_prices.stop_loss",
    }
    import json
    cur.execute(
        "INSERT INTO events (ts, category, severity, account, ticker, title, detail, market) "
        "VALUES (?, 'trade', 'info', ?, ?, ?, ?, ?)",
        (now_iso, account, ticker, title, json.dumps(detail, ensure_ascii=False), market),
    )


def check_stop_losses(
    conn: sqlite3.Connection,
    prices: dict[str, float],
) -> list[dict]:
    """Scan all positions, execute protective sells where pnl <= -stop_loss.

    Mutates DB inline (removes position row, updates account_state.cash,
    inserts trade row). Returns list of executed sells for logging.

    Market-hours guard: simulation mirrors real trading. Outside US extended
    hours we never execute trades — even protective stop-losses wait for the
    next session open (real brokers behave the same; off-hours quotes are stale
    and untradeable).
    """
    if not _is_market_hours_now():
        LOG.info("Outside US extended-hours window — stop-loss check skipped")
        return []
    cur = conn.cursor()
    now_iso = datetime.now(timezone.utc).isoformat()
    executed: list[dict] = []

    # Skip retired accounts entirely — frozen positions, no protective sells.
    retired_accts = {
        r["account_id"]
        for r in cur.execute(
            "SELECT account_id FROM account_meta WHERE status='retired'"
        ).fetchall()
    }
    market_by_acct = {
        r["account_id"]: r["market"]
        for r in cur.execute(
            "SELECT account_id, market FROM account_meta"
        ).fetchall()
    }

    pos_rows = cur.execute(
        "SELECT account, ticker, shares, avg_cost FROM positions"
    ).fetchall()

    for r in pos_rows:
        acct, ticker, shares, avg_cost = r["account"], r["ticker"], r["shares"], r["avg_cost"]
        if acct in retired_accts:
            continue
        acct_market = market_by_acct.get(acct, "US")
        if not _is_market_open_for(acct_market):
            continue
        if shares <= 0 or avg_cost <= 0:
            continue
        stop_loss = STOP_LOSS_BY_ACCT.get(acct)
        if stop_loss is None:
            # Benchmark / unknown account — no stop loss
            continue
        px = prices.get(ticker)
        if not px:
            continue
        pnl_pct = (px - avg_cost) / avg_cost

        triggered_reason = None
        triggered_msg = None

        # 1) Fixed stop loss (per-account threshold)
        if pnl_pct <= -stop_loss:
            triggered_reason = "stop_loss"
            triggered_msg = f"pnl {pnl_pct*100:.2f}% <= -{stop_loss*100:.1f}%"

        # 2) Trailing stop (only if armed via risk_regime, otherwise None)
        if triggered_reason is None:
            try:
                from trading.risk_regime import get_effective_trailing_stop
                trail_pct = get_effective_trailing_stop()
            except Exception:
                trail_pct = None
            if trail_pct is not None:
                # Peak price since this position was opened — best available
                # proxy: max(market_price) from positions_history for
                # (account, ticker) since the most recent BUY trade.
                peak_row = cur.execute("""
                    SELECT MAX(market_price) FROM positions_history
                     WHERE account=? AND ticker=?
                       AND timestamp >= COALESCE(
                           (SELECT MAX(timestamp) FROM trades
                             WHERE account=? AND ticker=? AND side='buy'),
                           '1900-01-01'
                       )
                """, (acct, ticker, acct, ticker)).fetchone()
                peak = peak_row[0] if peak_row and peak_row[0] else max(px, avg_cost)
                # Make sure peak is at least entry — guards against early
                # snapshots before any tick reached current price.
                peak = max(peak, avg_cost, px)
                drawdown_from_peak = (peak - px) / peak if peak > 0 else 0
                if drawdown_from_peak >= trail_pct and px > avg_cost * (1 - stop_loss):
                    # Only fire if we wouldn't have fired the fixed stop already
                    triggered_reason = "trailing_stop"
                    triggered_msg = (f"drawdown {drawdown_from_peak*100:.2f}% from peak "
                                     f"${peak:.2f} >= trail {trail_pct*100:.1f}%")

        if triggered_reason is None:
            continue

        # Trigger stop loss
        costs = _costs_for_market(acct_market)
        exec_price = costs.slippage(px, "sell")
        fees = costs.calculate("sell", shares, px)
        proceeds = shares * exec_price - fees["total_fees"]

        # Update cash
        cash_row = cur.execute(
            "SELECT cash FROM account_state WHERE account=? AND market=?", (acct, acct_market)
        ).fetchone()
        if cash_row is None:
            LOG.warning("Skip stop-loss %s/%s: no account_state row", acct, ticker)
            continue
        new_cash = cash_row["cash"] + proceeds
        cur.execute(
            "UPDATE account_state SET cash=?, updated_at=? WHERE account=? AND market=?",
            (new_cash, now_iso, acct, acct_market),
        )

        # Remove position
        cur.execute(
            "DELETE FROM positions WHERE account=? AND ticker=? AND market=?",
            (acct, ticker, acct_market),
        )

        # Record trade
        cur.execute(
            "INSERT INTO trades (account, ticker, side, shares, price, cost, slippage, timestamp, market) "
            "VALUES (?, ?, 'sell', ?, ?, ?, ?, ?, ?)",
            (acct, ticker, shares, px, fees["total_fees"],
             (px - exec_price) * shares, now_iso, acct_market),
        )

        pnl_dollar = (proceeds - (avg_cost * shares))
        _insert_trade_event(
            cur,
            now_iso=now_iso,
            market=acct_market,
            account=acct,
            ticker=ticker,
            shares=shares,
            price=px,
            avg_cost=avg_cost,
            pnl_pct=pnl_pct,
            pnl_dollar=pnl_dollar,
            stop_loss=stop_loss,
            reason=triggered_reason,
        )

        executed.append({
            "account": acct, "ticker": ticker, "shares": shares,
            "price": px, "avg_cost": avg_cost, "pnl_pct": pnl_pct * 100,
            "stop_loss": stop_loss * 100, "reason": triggered_reason,
            "detail": triggered_msg,
        })
        LOG.warning(
            "%s [%s/%s] %s: %.0f sh @ %s%.2f (cost %s%.2f, %s)",
            triggered_reason.upper(), acct_market, acct, ticker, shares,
            _currency_symbol(acct_market), px, _currency_symbol(acct_market), avg_cost,
            triggered_msg,
        )

    return executed


def update_equity_snapshots(db_path: str = DB_PATH) -> dict:
    """Main entry point."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    us_open = _is_us_market_hours_now()
    cn_open = _is_cn_market_hours_now()
    force_update = _force_price_update()
    fetch_us = us_open or force_update
    fetch_cn = cn_open or force_update
    LOG.info(
        "market-hours gate: US_open=%s CN_open=%s force_update=%s",
        us_open, cn_open, force_update,
    )

    # Fetch quotes only for markets whose snapshots/trades can be written on
    # this tick. The old path fetched every held US+CN ticker before checking
    # market hours; during CN-only windows it still spent time on US, and during
    # US-only windows a hung CN quote provider could block US snapshots. Keep
    # the quote universe aligned with the markets that are actually open.
    # QUANT_FORCE_PRICE_UPDATE=1 is a repair-only override for stale snapshots;
    # it refreshes prices without allowing off-hours stop-loss sells.
    held = [
        r[0]
        for r in cur.execute(
            """
            SELECT DISTINCT p.ticker
            FROM positions p
            LEFT JOIN account_meta m ON m.account_id = p.account
            WHERE COALESCE(m.status, 'active') != 'retired'
              AND (
                    (COALESCE(m.market, p.market, 'US') = 'US' AND ?)
                 OR (COALESCE(m.market, p.market, 'US') = 'CN' AND ?)
              )
            """,
            (1 if fetch_us else 0, 1 if fetch_cn else 0),
        ).fetchall()
    ]
    if not held:
        LOG.info("No open-market active positions found — nothing to update.")
        return {"accounts_updated": 0, "tickers": 0}

    t0 = time.time()
    prices = fetch_quotes(held)
    dt_fetch = time.time() - t0
    LOG.info(
        "Fetched realtime quotes for %d/%d tickers in %.1fs",
        len(prices), len(held), dt_fetch,
    )
    if not prices:
        LOG.warning("No realtime prices — skipping.")
        return {"accounts_updated": 0, "tickers": 0}

    # 1. Stop-loss check FIRST (may remove positions / change cash)
    stop_executed = check_stop_losses(conn, prices)
    if stop_executed:
        conn.commit()  # persist before equity recomputation
        # Fire-and-forget Telegram alert (don't block equity update on failure)
        try:
            _notify_stop_losses(stop_executed)
        except Exception as e:
            LOG.warning("Stop-loss Telegram alert failed: %s", e)

    # 2. Equity recomputation using fresh state
    now_iso = datetime.now(timezone.utc).isoformat()
    cash_by_acct = {
        r["account"]: r["cash"]
        for r in cur.execute("SELECT account, cash FROM account_state").fetchall()
    }
    pos_rows = cur.execute(
        "SELECT account, ticker, shares, avg_cost FROM positions"
    ).fetchall()
    pos_by_acct: dict[str, list[dict]] = {}
    for r in pos_rows:
        pos_by_acct.setdefault(r["account"], []).append(dict(r))

    updated = 0
    # Account → market lookup (default 'US' for legacy accounts).
    market_by_acct: dict[str, str] = {}
    retired_accts: set[str] = set()
    for r in cur.execute(
        "SELECT account_id, market, status FROM account_meta"
    ).fetchall():
        market_by_acct[r["account_id"]] = r["market"]
        if r["status"] == "retired":
            retired_accts.add(r["account_id"])
    skipped_by_market: dict[str, int] = {}
    skipped_retired = 0
    # include accounts with no positions (all-cash) so snapshot keeps refreshing
    for acct, cash in cash_by_acct.items():
        if acct in retired_accts:
            skipped_retired += 1
            continue  # retired: no equity drift, last snapshot is final
        market = market_by_acct.get(acct, "US")
        # Gate snapshots by each account's own market session. Writing a
        # static equity when the market is closed (price unchanged) fills
        # the equity curve with hundreds of identical points per hour and
        # visually squashes earlier, real trading days.
        if market == "US" and not fetch_us:
            skipped_by_market["US"] = skipped_by_market.get("US", 0) + 1
            continue
        if market == "CN" and not fetch_cn:
            skipped_by_market["CN"] = skipped_by_market.get("CN", 0) + 1
            continue

        positions = pos_by_acct.get(acct, [])
        missing_prices = [p["ticker"] for p in positions if p["ticker"] not in prices]
        if missing_prices:
            # Do not silently mark held positions at avg_cost. That writes
            # equity ≈ initial_cash for winning/losing accounts and creates
            # fake V-dips / sawtooth curves (CB13 was the smoking gun). Skip
            # this account until a real quote arrives; the last good snapshot
            # remains the visible value.
            LOG.warning(
                "Skip snapshot for %s: missing realtime prices for %d/%d held tickers: %s",
                acct, len(missing_prices), len(positions), ",".join(missing_prices[:8]),
            )
            continue
        equity = cash
        for p in positions:
            px = prices[p["ticker"]]
            equity += p["shares"] * px
            cur.execute(
                "UPDATE positions SET current_price=?, updated_at=? "
                "WHERE account=? AND ticker=? AND market=?",
                (px, now_iso, acct, p["ticker"], market),
            )
        cur.execute(
            "INSERT INTO accounts (name, cash, equity, timestamp, market) "
            "VALUES (?, ?, ?, ?, ?)",
            (acct, round(cash, 4), round(equity, 4), now_iso, market),
        )
        updated += 1

    if skipped_by_market:
        LOG.info("Skipped off-hours snapshots: %s", skipped_by_market)
    if skipped_retired:
        LOG.info("Skipped %d retired accounts (frozen)", skipped_retired)

    conn.commit()

    # Risk-regime evaluation: compute portfolio drawdown, auto-arm/disarm
    # the trailing stop. Best-effort — never block snapshot updates on this.
    try:
        from trading import risk_regime
        rr = risk_regime.evaluate_and_update()
        if rr.get("transitioned"):
            LOG.warning("Risk regime transition: %s", rr)
    except Exception as e:
        LOG.warning("risk-regime evaluation failed: %s", e)

    conn.close()
    LOG.info(
        "Updated %d snapshots; %d stop-loss sells executed",
        updated, len(stop_executed),
    )
    return {
        "accounts_updated": updated,
        "tickers": len(held),
        "prices_fetched": len(prices),
        "stop_losses": len(stop_executed),
        "fetch_seconds": round(dt_fetch, 2),
    }


if __name__ == "__main__":
    t0 = time.time()
    try:
        stats = update_equity_snapshots()
        LOG.info("Done in %.1fs: %s", time.time() - t0, stats)
    except Exception as e:
        LOG.exception("update_prices failed: %s", e)
        sys.exit(1)
