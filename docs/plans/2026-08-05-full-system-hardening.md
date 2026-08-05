# Full Quant-System Hardening Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Fix every confirmed P0/P1/P2 issue from the 2026-08-05 US quant-system audit without rewriting historical fills or interrupting data/research monitoring.

**Architecture:** First pause execution-bearing schedules and create a verified online DB backup. Then implement independent code streams in isolated worktrees: market-scoped risk/execution safety, operational/recovery/data-quality hardening, and time-safe research replay. Merge only after focused and full-suite tests, migrate production DB with audit events, verify ledger/health/runtime, then resume execution schedules and observe the first safe live cycle.

**Tech Stack:** Python 3.11, SQLite WAL, Bash/flock/cron, pytest, FastAPI dashboard, Qlib checkpoints.

---

### Task 1: Safety freeze and backup
- Pause `update_prices.sh`, `run_cycle.sh`, and `run_cycle_quiet.sh` cron lines only.
- Keep backfill, factors, Qlib, health checks, corporate-action audits, and ledger watchdogs active.
- Pause duplicate Hermes price backfill job.
- Stop the completed-but-stuck CN factor process.
- Create SQLite online backup, run `quick_check`, zstd test, and SHA256.

### Task 2: Market-scoped risk regime and incident audit
- Migrate `risk_regime` from singleton to per-market state.
- Require explicit market on every risk API and execution caller.
- Compute drawdown only from active live accounts in the requested market.
- Add regression tests proving CN drawdown cannot arm US.
- Add a read-only-first/audited counterfactual classifier for historical trailing-stop trades; archive from capital-allocation analytics without deleting fills.

### Task 3: Remove execution bypasses
- Make every production execution path consume positive, timestamped, fresh, tradable quote metadata.
- Disable or fail closed legacy full-cycle execution that only has scalar/historical prices.
- Add tests for stale, untradable, missing and historical-fallback quotes.

### Task 4: Scheduler, lock and process lifecycle hardening
- Add market-scoped factor-refresh wrappers with flock, timeout, explicit OK/FAIL and alerts.
- Make CN historical workers terminate cleanly after timeout.
- Use one CN backfill write lock across adjusted/raw jobs.
- Emit lock-timeout/scheduled-start/last-success metrics for updater and cycles.
- Fix health-check parsing for scope-aware backfill logs and persistent-alert escalation.
- Remove duplicate scheduler ownership.

### Task 5: Recovery, permissions and retention
- Add daily SQLite-online-backup wrapper with SHA256, zstd test and restore quick-check.
- Add cron schedule and retention/archive policy.
- Tighten `.env`, DB, logs and backups permissions with private umask.
- Add logrotate and disk thresholds.

### Task 6: Data and account-state quality
- Explicitly quarantine inactive/renamed/delisted universe names from new buys while preserving history.
- Mark F14/F16 `mining_failed/non_tradeable` rather than active-running.
- Preserve raw OHLC rows but quarantine invariant violations in research reads/health output.
- Distinguish RTH from extended-hours quote health and surface last complete valuation time.

### Task 7: Research validity and replay
- Require point-in-time universe or explicit biased-research opt-in in every legacy backtest.
- Prevent current GP expressions from being presented as historical walk-forward results.
- Use adjusted prices for signal research but raw prices/corporate actions for broker-like cash/share replay.
- Complete Qlib checkpoint-per-day replay and consistency tests; continue fail-closed behavior outside checkpoint coverage.

### Task 8: Dashboard and API performance
- Surface degraded quote coverage, valuation timestamp, non-tradeable accounts, risk state by market, and research caveats.
- Cache/downsample heavy equity-curve and signal-quality responses while preserving exact downloadable data.

### Task 9: Integration and deployment
- Run focused tests, full pytest, compileall, shell syntax, diff check, secret scan, DB quick-check, US/CN ledger watchdogs, health check and local API smoke.
- Independent spec and code-quality review.
- Commit changes, apply audited migrations, verify backups and permissions.
- Resume execution schedules only after all gates pass and inspect the first live tick/cycle.
