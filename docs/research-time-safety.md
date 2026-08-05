# Research replay time-safety contract

Backtests and historical replays fail closed unless every coordinate below is
explicitly time-safe. An override makes a run **research-only**; it does not make
the result valid for capital allocation.

## Required result metadata

Every result records `universe_mode`, `signal_price_mode`,
`execution_price_mode`, `model_provenance`, `look_ahead_validity`,
`capital_allocation_valid`, and `warnings`.

## Universe and GP

* `universe_membership(market,date,ticker)` must cover every market session in
  the requested window. Membership is consumed on its exact date and is never
  back/forward-filled. `--allow-survivorship-biased` is explicitly invalid for
  capital allocation.
* Today's persisted GP expression is hindsight-selected. Legacy GP replay fails
  unless `--allow-gp-hindsight` is passed; that override remains invalid. A GP
  result can become valid only when expression checkpoints selected using data
  available at each historical as-of date are supplied.

## US replay price coordinates

`scripts/replay_us.py` defaults to `--execution-mode broker`:

* signals/factors use adjusted `prices` through T;
* execution and valuation use raw `prices_raw` at T+1 open;
* fees/slippage update cash and average cost;
* split/bonus share ratios and cash dividends are applied before the effective
  day's opening execution.

Broker mode never falls back to adjusted prices. It also requires:

```sql
corporate_actions(market,ticker,ex_date,action_type,ratio,cash_per_share)
corporate_action_coverage(market,ticker,start_date,end_date,status)
```

Coverage rows must be `status='complete'` for the entire replay interval. Until
raw prices and action coverage exist for every required ticker/date, broker mode
fails. `--execution-mode signal-only` remains available, uses adjusted prices,
and is always labelled invalid for capital allocation.

## Qlib checkpoint replay

`factors.qlib_checkpoint.predict_checkpoint_scores()` loads the exact daily
checkpoint at T, verifies its recorded score, frozen model/dataset/processors and
training-window bound, selects scores dated T, and permits only an execution date
after T. Missing dates are never filled from a newer model.

```bash
python -m factors.qlib_checkpoint --market US --model Q01 \
  --as-of 2026-07-09 --execution-date 2026-07-10
```

Replay can start only at the first date common to every requested model, and the
entire requested date grid must be present. Historical coverage before the first
fully archived checkpoint cannot be reconstructed without retraining and is not
claimed. CN mirror checkpoints additionally fail unless their sidecar explicitly
records `extra.point_in_time_complete=true`; current incomplete CN semantics are
not treated as completed coverage.