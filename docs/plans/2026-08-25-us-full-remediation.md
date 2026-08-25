# US Quant + Dashboard Full Remediation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Repair all confirmed US ledger, corporate-action, operational-health, API correctness/performance, pagination, and mobile UI defects while keeping execution frozen.

**Architecture:** Preserve immutable trades; add an auditable idempotent corporate-action state migration for open positions, backed by a verified online backup. Make held-symbol corporate-action checks a fast blocking gate with durable health, retain full-universe scans as a slower secondary audit, and keep runtime quote/execution paths fail-closed. Harden dashboard facts by market, use composite event cursors, bound payloads and move synchronous DB work off the event loop.

**Tech Stack:** Python 3.12, SQLite WAL, pytest, FastAPI/aiosqlite, vanilla JS/CSS, systemd/nginx.

---

### Task 1: Corporate-action gate and auditable split repair
- Add regression tests for AVB-like 2.793:1 open split, total-cost preservation, idempotence, raw post-action coordinate and no historical trade rewrite.
- Add a held/recent-symbol-first corporate-action gate that completes before the full-universe audit, writes scheduler_runs/operational_health, and fails the affected market/account closed.
- Add an explicit `--apply` repair command requiring backup handle, with audit table/event; default read-only.
- Fix the daily wrapper timeout/health lifecycle and eliminate permanent alert-noise semantics.
- Verify focused tests and shell syntax.

### Task 2: Runtime/ledger and Qlib operational closure
- Promote unresolved open share actions and stale held marks to blocking health/watchdog findings.
- Make raw-coordinate discontinuity detection split-aware.
- Align history audit window with scheduled raw ledger coverage or report scope explicitly.
- Define and expose dividend accounting policy without inventing historical cash credits.
- Fix qlib lock-timeout health writer arguments and record every qlib success/failure/timeout.
- Verify focused and full quant tests.

### Task 3: Dashboard correctness and isolation
- Correct group lifecycle/readiness counts and add F/IDX group summaries.
- Bind account detail, recent trades and anchor queries by market.
- Add tests for active-nontradeable versus retired and same-ID cross-market fixtures.
- Fix event pagination with a stable composite cursor and >page-size same-timestamp regression test.

### Task 4: Dashboard performance and mobile
- Downsample/account-window equity and benchmark curves; lazy/bounded snapshot history; preserve chart semantics.
- Move Factor Lab catalog DB aggregation into a worker thread and add short TTL cache.
- Make mobile navigation wrap or horizontally scroll without widening the document.
- Escape DB-originated dynamic text in templates.
- Benchmark cold/hot endpoints and inspect all routes at desktop and 390px.

### Task 5: Migration, deployment and release gates
- Wait for verified backup completion and validate artifact/SHA/restore drill.
- Run split repair preview, apply A02/AVB migration, then independently verify shares, avg cost, total cost, state/equity, events and idempotent rerun.
- Run full quant/dashboard tests, compile, JS syntax, shell syntax, git diff check and independent review.
- Restart dashboard only for backend changes; verify APIs/browser/public cache headers.
- Keep cycles paused and updater `no-trades`; verify first actual-session no-trades tick before any later live decision.
