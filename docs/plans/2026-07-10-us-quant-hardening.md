# US Quant Runtime Hardening Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Eliminate stop-loss blind windows and false-success writes, make health/cron fail closed, and close the raw-price/corporate-action/account-metadata operational loop without altering strategy definitions.

**Architecture:** Keep expensive adjusted-price/factor production in dedicated jobs and make live cycles consume persisted factor rows. Introduce one atomic trade writer for normal A/B/F/Q/IDX trades, while retaining the already-atomic per-sell updater path. Make monitoring and process exit semantics fail closed; add explicit raw-price freshness and corporate-action monitoring. Preserve audit data: no destructive ledger/history edits without a timestamped DB backup and verification.

**Tech Stack:** Python 3.12, SQLite/WAL, pandas, pytest, bash/flock/timeout, cron.

---

### Task 1: Regression tests for fail-closed health and process exits

**Objective:** Prove ledger criticals and market failures return non-zero.

**Files:**
- Modify: `tests/test_operational_hardening.py`
- Modify: `scripts/health_check.py`
- Modify: `main.py`

**Steps:**
1. Add a test where a fake ledger subprocess reports critical only when `--fail-on-critical` is present; verify `run_ledger()` passes the flag.
2. Add a test that monkeypatches `_run_for_market` to fail one market; verify the market runner raises/non-zero after processing all requested markets.
3. Run focused tests and observe RED.
4. Add `--fail-on-critical` and aggregate market failures into a final `RuntimeError`/non-zero exit.
5. Run focused and full tests.

### Task 2: Atomic normal-trade persistence

**Objective:** A trade cannot mutate durable account state without a matching trade and event row.

**Files:**
- Modify: `trading/account.py`
- Modify: `trading/engine.py`
- Modify: `data/store.py`
- Modify: `main.py`
- Create/modify: `tests/test_atomic_trade_persistence.py`

**Steps:**
1. Add RED tests for callback failure: buy/sell must leave cash, positions, and trade_log unchanged.
2. Add RED integration test using a temporary DataStore: successful trade atomically writes `trades`, `events`, `account_state`, `positions`; injected DB failure writes none and leaves memory unchanged.
3. Introduce account snapshot/restore or preview/apply primitives so engine can roll back mutation when persistence fails.
4. Replace the best-effort `_on_trade` callback with a fail-closed persistence callback returning/raising before success is exposed.
5. Persist trade + event + current state in one SQLite transaction. Keep update_prices stop-loss path unchanged.
6. Run focused and full tests.

### Task 3: Persisted-signal live cycle and short critical section

**Objective:** Live trading cycles stop recomputing/writing Alpha/GP/F factors and no longer hold the stop-loss lock for 13–19 minutes.

**Files:**
- Modify: `main.py`
- Modify: `scripts/run_cycle.sh`
- Modify: `scripts/run_cycle_quiet.sh`
- Modify: `scripts/update_prices.sh`
- Create/modify: `tests/test_persisted_live_signals.py`

**Steps:**
1. Add RED tests that `run_once(live_fast=True)` does not call `compute_factors`/`mine_gp_factors` and reconstructs A/GP/F latest cross-sections from `factor_values` scoped by market/group.
2. Add coverage/staleness gates: incomplete/stale persisted signals no-op the affected account and emit an error event; never liquidate on empty signals.
3. Add a fast-cycle CLI mode and point hourly/intra-hour cron wrappers at it.
4. Split locks: a preparation phase may fetch quotes outside the writer lock; only trade + state mutation uses the short writer lock. Ensure stop-loss updater is never blocked by factor computation.
5. Retain `scripts.refresh_factors` as the expensive producer after adjusted backfill.
6. Verify a forced/off-hours fast cycle performs no network-wide factor recomputation and completes locally.

### Task 4: Correct completed-session freshness and remove redundant writes

**Objective:** Do not refetch the entire universe intraday because today's incomplete daily bar is absent.

**Files:**
- Modify: `data/fetcher.py`
- Modify: `data/cn_fetcher.py`
- Modify: `main.py`
- Modify: `tests/test_operational_hardening.py`

**Steps:**
1. Add RED date-boundary tests for pre-close, post-close, weekend, and exclusive Yahoo end-date behavior.
2. Implement `latest_completed_session_date(market, now)` (ET/CST-aware, conservative holiday handling via latest cache/backfill target).
3. Fetch through target+1 day because Yahoo end is exclusive.
4. Remove `fetch_data()`'s per-ticker `save_prices()` loop; `get_historical()` already persists downloaded bars.
5. Run focused/full tests and compare a dry cache-classification probe before/after.

### Task 5: Harden CN historical fetch and scheduled wrappers

**Objective:** No historical provider can leave an orphaned process for days.

**Files:**
- Modify: `data/cn_fetcher.py`
- Modify/create: `scripts/backfill_prices_daily.sh`
- Modify: crontab
- Modify: `tests/test_operational_hardening.py`

**Steps:**
1. Add RED test with one never-completing future; batch returns bounded partial results and shuts down without waiting.
2. Replace `ThreadPoolExecutor.map` with `as_completed(..., timeout=...)`, cancellation and non-waiting shutdown.
3. Add per-market backfill wrapper with flock, OS timeout, structured start/OK/FAIL logs and alert behavior.
4. Replace direct cron commands with wrapper calls.
5. Terminate the verified orphaned Jul-8 process only after the new bounded path is deployed; verify no stale process/lock remains.

### Task 6: Raw execution-price and corporate-action operational closure

**Objective:** Raw prices stay current and execution/history audits never read adjusted prices.

**Files:**
- Modify: `scripts/ledger_watchdog.py`
- Modify: `scripts/health_check.py`
- Modify/create: raw backfill/corporate-action wrappers
- Modify: crontab
- Modify: `tests/test_operational_hardening.py`

**Steps:**
1. Add RED tests that US historical account audit reads `prices_raw`, while factor freshness reads `prices`.
2. Add raw 1d freshness checks, adjusted/raw coverage counts, process/backfill freshness checks and active-only ledger checking.
3. Schedule US and CN raw 1d incremental backfills after adjusted backfills.
4. Schedule a bounded corporate-action audit; alert on open-position split/bonus actions or fetch failure coverage, never auto-mutate holdings.
5. Verify latest raw dates/coverage and raw-vs-adjusted split-sensitive smoke cases.

### Task 7: Metadata and retired archival integrity cleanup

**Objective:** Active counts represent operational accounts and retired archival issues do not mask live ledger health.

**Files:**
- Modify: `scripts/ledger_watchdog.py`
- Modify: `scripts/health_check.py`
- Add one migration/cleanup script under `scripts/`
- Modify: tests

**Steps:**
1. Add RED tests: default operational watchdog checks active accounts; `--history-include-retired` adds a separate archival section.
2. Mark inert `C01` metadata inactive/retired through a backup-first migration and emit an audit event.
3. Repair B03's frozen snapshot only after backing up affected rows and verifying the repair event's recorded rebuilt equity; classify B07/B09 as archived frozen-state discrepancies unless explicitly repaired.
4. Verify active count, watchdog, dashboard summary, and audit events.

### Task 8: End-to-end verification and independent review

**Objective:** Prove the system is safe and the requested fixes are actually operating.

**Steps:**
1. `source venv/bin/activate && python -m pytest -q`
2. `python -m compileall -q main.py data trading accounts factors scripts tests`
3. `python scripts/health_check.py --json --skip-quotes` and confirm ledger failures would propagate.
4. Run active US ledger watchdog with `--fail-on-critical`; run retired archival audit separately.
5. Run SQLite `PRAGMA quick_check` and active snapshot reconciliation.
6. Exercise a fast off-hours cycle and verify no full Alpha/GP computation, no trade, correct exit code, and bounded runtime.
7. Verify cron wrappers, locks, no orphaned backfill/retrain processes, adjusted/raw latest coverage, and factor/Qlib freshness.
8. Dispatch independent spec and quality reviewers; fix all critical/important findings and re-run verification.
