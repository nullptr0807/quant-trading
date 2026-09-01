"""Market-scoped Qlib training selection from account lifecycle state."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

from config.settings import ACCOUNT_PREFIX


def select_qlib_model_ids(
    db_path: str | Path,
    market: str,
    available_ids: Iterable[str],
    requested_ids: Iterable[str] | None = None,
) -> list[str]:
    """Select models for a scheduled run; explicit requests override lifecycle.

    Scheduled/default runs fail closed if model metadata is incomplete. This
    prevents a missing/corrupt lifecycle row from silently suppressing training.
    """
    market = str(market).upper()
    if market not in {"US", "CN"}:
        raise ValueError(f"unsupported market: {market}")
    available = list(dict.fromkeys(str(x) for x in available_ids))
    available_set = set(available)

    if requested_ids is not None:
        requested = list(dict.fromkeys(str(x) for x in requested_ids))
        unknown = [x for x in requested if x not in available_set]
        if unknown:
            raise ValueError(f"unknown Qlib model ids: {unknown}")
        return requested

    prefix = ACCOUNT_PREFIX.get(market, "")
    account_by_model = {model: f"{prefix}{model}" for model in available}
    if not account_by_model:
        return []
    marks = ",".join("?" for _ in account_by_model)
    path = Path(db_path).expanduser().resolve()
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        rows = con.execute(
            f"SELECT account_id,status FROM account_meta "
            f"WHERE market=? AND account_id IN ({marks})",
            (market, *account_by_model.values()),
        ).fetchall()
    finally:
        con.close()
    status_by_account = {str(account): status for account, status in rows}
    missing = [account for account in account_by_model.values() if account not in status_by_account]
    if missing:
        raise RuntimeError(
            f"{market} Qlib lifecycle metadata incomplete; missing accounts: {missing}"
        )
    invalid = {
        account: status for account, status in status_by_account.items()
        if status not in {"active", "retired"}
    }
    if invalid:
        raise RuntimeError(f"{market} Qlib lifecycle status invalid: {invalid}")
    return [
        model for model in available
        if status_by_account[account_by_model[model]] == "active"
    ]