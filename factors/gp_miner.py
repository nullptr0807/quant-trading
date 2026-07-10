"""
Genetic Programming based Alpha Factor Mining module.
Uses gplearn to evolve alpha expressions from OHLCV price data.
"""

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from gplearn.genetic import SymbolicTransformer
from gplearn.functions import make_function

warnings.filterwarnings("ignore")

MINED_ALPHAS_PATH = os.path.expanduser("~/quant-trading/factors/mined_alphas.json")

# --- Protected custom functions for gplearn ---

def _protected_div(x1, x2):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x2) > 1e-10, x1 / x2, 0.0)

def _protected_sqrt(x1):
    return np.sqrt(np.abs(x1))

def _protected_log(x1):
    return np.log(np.abs(x1) + 1.0)

def _neg(x1):
    return -x1

def _protected_inv(x1):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x1) > 1e-10, 1.0 / x1, 0.0)

def _max2(x1, x2):
    return np.maximum(x1, x2)

def _min2(x1, x2):
    return np.minimum(x1, x2)

gp_div = make_function(function=_protected_div, name="div", arity=2)
gp_sqrt = make_function(function=_protected_sqrt, name="sqrt_abs", arity=1)
gp_log = make_function(function=_protected_log, name="log_abs1", arity=1)
gp_neg = make_function(function=_neg, name="neg", arity=1)
gp_inv = make_function(function=_protected_inv, name="inv", arity=1)
gp_max = make_function(function=_max2, name="max2", arity=2)
gp_min = make_function(function=_min2, name="min2", arity=2)

GP_FUNCTION_SET = ["add", "sub", "mul", gp_div, gp_sqrt, gp_log, gp_neg, gp_inv, gp_max, gp_min]

# Feature column names (legacy B-family terminal set). Keep this stable so old
# mined expressions continue to map X0..X12 exactly as before.
FEATURE_COLS = [
    "o_c", "h_c", "l_c", "v_vma20",
    "ma_5", "ma_10", "ma_20",
    "std_5", "std_10", "std_20",
    "ret_1", "ret_5", "ret_10",
]

# Expanded terminal set for F-family / FactorMiner-style mining. These are
# computed by _compute_features but are opt-in via gp_feature_subset so legacy
# B-family accounts are not silently changed.
FACTORMINER_FEATURE_COLS = FEATURE_COLS + [
    "range_pos", "upper_pos", "lower_shadow", "upper_shadow", "gap_1",
    "dvol_vma20", "ret_1_dvol", "absret_1_dvol", "vol_of_vol_20",
    "skew_20", "kurt_20", "pv_corr_20",
    "slope_20", "trend_r2_20", "trend_resi_20",
]

# Expression evaluator for gplearn symbolic expressions
_EVAL_FUNCS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": _protected_div,
    "sqrt_abs": _protected_sqrt,
    "log_abs1": _protected_log,
    "neg": _neg,
    "inv": _protected_inv,
    "max2": _max2,
    "min2": _min2,
}


def _eval_expression(expr_str: str, X: np.ndarray) -> np.ndarray:
    """Evaluate a gplearn expression string on feature matrix X."""
    tokens = _tokenize(expr_str)
    result, _ = _parse_expr(tokens, 0, X)
    return result


def _tokenize(expr_str: str) -> list[str]:
    """Tokenize gplearn expression into function names, variables, and constants."""
    tokens = []
    i = 0
    s = expr_str.strip()
    while i < len(s):
        if s[i] in " \t\n":
            i += 1
        elif s[i] in "(),":
            i += 1  # skip delimiters but track parens for parsing
        elif s[i] == 'X':
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(s[i:j])
            i = j
        elif s[i] == '-' and i + 1 < len(s) and (s[i+1].isdigit() or s[i+1] == '.'):
            j = i + 1
            while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif s[i].isdigit() or s[i] == '.':
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == '.'):
                j += 1
            tokens.append(s[i:j])
            i = j
        elif s[i].isalpha() or s[i] == '_':
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            i += 1
    return tokens


def _parse_expr(tokens: list[str], pos: int, X: np.ndarray):
    """Recursive descent parser for gplearn expressions."""
    if pos >= len(tokens):
        return np.zeros(len(X)), pos

    tok = tokens[pos]

    # Variable reference (X0, X1, ...)
    if tok.startswith("X") and tok[1:].isdigit():
        idx = int(tok[1:])
        if idx < X.shape[1]:
            return X[:, idx].copy(), pos + 1
        return np.zeros(len(X)), pos + 1

    # Numeric constant
    try:
        val = float(tok)
        return np.full(len(X), val), pos + 1
    except ValueError:
        pass

    # Function call
    if tok in _EVAL_FUNCS:
        func = _EVAL_FUNCS[tok]
        # Determine arity
        arity = 1 if tok in ("sqrt_abs", "log_abs1", "neg", "inv") else 2
        arg1, pos = _parse_expr(tokens, pos + 1, X)
        if arity == 2:
            arg2, pos = _parse_expr(tokens, pos, X)
            return func(arg1, arg2), pos
        return func(arg1), pos

    # Unknown — skip
    return np.zeros(len(X)), pos + 1


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute normalized features from OHLCV DataFrame.

    The first 13 columns are the legacy B-family terminal set. Additional
    columns are FactorMiner-style terminals and are only used when explicitly
    referenced by gp_feature_subset.
    """
    c = df["close"].astype(float)
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    v = df["volume"].astype(float)
    eps = 1e-10

    feat = pd.DataFrame(index=df.index)
    feat["o_c"] = o / c.replace(0, np.nan)
    feat["h_c"] = h / c.replace(0, np.nan)
    feat["l_c"] = l / c.replace(0, np.nan)
    v_ma20 = v.rolling(20).mean()
    feat["v_vma20"] = v / v_ma20.replace(0, np.nan)
    feat["ma_5"] = c.rolling(5).mean() / c.replace(0, np.nan)
    feat["ma_10"] = c.rolling(10).mean() / c.replace(0, np.nan)
    feat["ma_20"] = c.rolling(20).mean() / c.replace(0, np.nan)
    feat["std_5"] = c.rolling(5).std() / c.replace(0, np.nan)
    feat["std_10"] = c.rolling(10).std() / c.replace(0, np.nan)
    feat["std_20"] = c.rolling(20).std() / c.replace(0, np.nan)
    feat["ret_1"] = c.pct_change(1)
    feat["ret_5"] = c.pct_change(5)
    feat["ret_10"] = c.pct_change(10)

    # FactorMiner-style expanded features.
    rng = (h - l).replace(0, np.nan)
    prev_close = c.shift(1)
    dollar_vol = (c * v).replace(0, np.nan)
    dollar_vol_ma20 = dollar_vol.rolling(20).mean()
    ret_1 = feat["ret_1"]

    feat["range_pos"] = (c - l) / rng
    feat["upper_pos"] = (h - c) / rng
    feat["lower_shadow"] = (np.minimum(o, c) - l) / rng
    feat["upper_shadow"] = (h - np.maximum(o, c)) / rng
    feat["gap_1"] = o / prev_close.replace(0, np.nan) - 1.0
    feat["dvol_vma20"] = dollar_vol / dollar_vol_ma20.replace(0, np.nan)
    feat["ret_1_dvol"] = ret_1 / (dollar_vol / 1e9 + eps)
    feat["absret_1_dvol"] = ret_1.abs() / (dollar_vol / 1e9 + eps)
    feat["vol_of_vol_20"] = feat["v_vma20"].rolling(20).std()
    feat["skew_20"] = ret_1.rolling(20).skew()
    feat["kurt_20"] = ret_1.rolling(20).kurt()
    feat["pv_corr_20"] = c.pct_change().rolling(20).corr(v.pct_change())

    x = np.arange(20, dtype=float)
    x_centered = x - x.mean()
    denom = float((x_centered ** 2).sum())

    def _slope(arr):
        arr = np.asarray(arr, dtype=float)
        if np.isnan(arr).any() or denom == 0:
            return np.nan
        y = arr - arr.mean()
        return float((x_centered * y).sum() / denom / (arr[-1] if abs(arr[-1]) > eps else 1.0))

    def _r2(arr):
        arr = np.asarray(arr, dtype=float)
        if np.isnan(arr).any():
            return np.nan
        y = arr - arr.mean()
        ss_tot = float((y ** 2).sum())
        if ss_tot <= eps:
            return 0.0
        slope = float((x_centered * y).sum() / denom)
        y_hat = slope * x_centered
        ss_res = float(((y - y_hat) ** 2).sum())
        return max(0.0, 1.0 - ss_res / ss_tot)

    def _resi(arr):
        arr = np.asarray(arr, dtype=float)
        if np.isnan(arr).any():
            return np.nan
        y = arr - arr.mean()
        slope = float((x_centered * y).sum() / denom)
        y_hat_last = arr.mean() + slope * x_centered[-1]
        return float((arr[-1] - y_hat_last) / (arr[-1] if abs(arr[-1]) > eps else 1.0))

    feat["slope_20"] = c.rolling(20).apply(_slope, raw=True)
    feat["trend_r2_20"] = c.rolling(20).apply(_r2, raw=True)
    feat["trend_resi_20"] = c.rolling(20).apply(_resi, raw=True)
    return feat


def _build_y_target(close: pd.Series, target: str) -> pd.Series:
    """Build a forward-looking y target from a price series.

    Supported:
      next_1d_ret      — close.pct_change().shift(-1)
      next_3d_ret      — forward 3-day return
      next_5d_ret      — forward 5-day return
      next_5d_sharpe   — forward 5-day mean_ret / std_ret (risk-adjusted)
      next_5d_minret_neg — negative of forward-5d worst single-day return (defensive: high = resilient)
      reversal_2d      — -1 * forward 2-day return (mean reversion)
    """
    r1 = close.pct_change()
    if target == "next_1d_ret":
        return r1.shift(-1)
    if target == "next_3d_ret":
        return (close.shift(-3) / close - 1.0)
    if target == "next_5d_ret":
        return (close.shift(-5) / close - 1.0)
    if target == "next_5d_sharpe":
        # Forward 5d daily-return Sharpe aligned to today: uses returns
        # from t+1..t+5 only. Do NOT use rolling+double-shift; that drifts
        # to t+5..t+9 and changes the target horizon.
        fwd = pd.concat([r1.shift(-i) for i in range(1, 6)], axis=1)
        full_window = fwd.count(axis=1) == 5
        fwd_mean = fwd.mean(axis=1)
        fwd_std = fwd.std(axis=1)
        out = fwd_mean / fwd_std.replace(0, np.nan)
        return out.where(full_window)
    if target == "next_5d_minret_neg":
        # Worst single-day return over immediate next 5 days, sign-flipped so
        # "high" = resilient.
        fwd = pd.concat([r1.shift(-i) for i in range(1, 6)], axis=1)
        full_window = fwd.count(axis=1) == 5
        return (-fwd.min(axis=1)).where(full_window)
    if target == "reversal_2d":
        return -((close.shift(-2) / close) - 1.0)
    # fallback
    return r1.shift(-1)


def _prepare_dataset(
    historical_data: dict[str, pd.DataFrame],
    feature_cols: list[str] | None = None,
    y_target: str = "next_1d_ret",
    *,
    return_dates: bool = False,
):
    """Stack ticker samples in global chronological order.

    Returning globally sorted dates is essential for a real time holdout: the
    prior ticker-by-ticker ``vstack`` made the last 20% mostly a ticker split,
    not a future-period split.
    """
    cols = feature_cols if feature_cols else FEATURE_COLS
    rows: list[pd.DataFrame] = []
    for ticker, df in historical_data.items():
        feat = _compute_features(df)
        target = _build_y_target(df["close"], y_target)
        combined = feat.assign(target=target).dropna(subset=list(cols) + ["target"])
        if len(combined) < 5:
            continue
        frame = combined[list(cols) + ["target"]].copy()
        frame["_date"] = pd.to_datetime(frame.index, utc=True).normalize()
        frame["_ticker"] = str(ticker)
        rows.append(frame)
    if not rows:
        empty = (np.empty((0, len(cols))), np.empty(0))
        if return_dates:
            return (*empty, np.empty(0, dtype="datetime64[ns]"))
        return empty
    stacked = pd.concat(rows, axis=0).sort_values(
        ["_date", "_ticker"], kind="stable"
    )
    X = stacked[list(cols)].to_numpy()
    y = stacked["target"].to_numpy()
    if return_dates:
        dates = stacked["_date"].dt.tz_localize(None).to_numpy()
        return X, y, dates
    return X, y


class GPAlphaMiner:
    """Genetic programming based alpha factor miner."""

    def __init__(self):
        self._transformer = None

    def mine_factors(
        self,
        historical_data: dict[str, pd.DataFrame],
        n_factors: int = 20,
        generations: int = 20,
        base_seed: int = 42,
        population_size: int = 300,
        n_runs: int = 5,
        parsimony_coefficient: float = 0.01,
        y_target: str = "next_1d_ret",
        feature_subset: list[str] | tuple[str, ...] | None = None,
        dedup_threshold: float = 0.85,
    ) -> list[dict]:
        """Mine alpha factors using GP evolution with multiple seeds for diversity.

        y_target: forward target name (see _build_y_target).
        feature_subset: subset of FEATURE_COLS to use as terminals; None = all 13.
        dedup_threshold: |corr| above which a candidate is dropped vs already-kept factors.
        """
        cols = list(feature_subset) if feature_subset else list(FEATURE_COLS)
        X, y, dates = _prepare_dataset(
            historical_data,
            feature_cols=cols,
            y_target=y_target,
            return_dates=True,
        )
        if len(X) < 20:
            return []
        split_target = max(10, int(len(X) * 0.8))
        if len(X) - split_target < 10:
            return []
        # Never split a cross-section across train/OOS. Move the boundary to
        # the first row of the next global date so one market date belongs to
        # exactly one side of the holdout.
        split_date = dates[split_target]
        split = int(np.searchsorted(dates, split_date, side="left"))
        if split < 10 or len(X) - split < 10:
            return []
        X_train, y_train = X[:split], y[:split]
        X_oos, y_oos = X[split:], y[split:]
        train_end = pd.Timestamp(dates[split - 1]).isoformat()
        oos_start = pd.Timestamp(dates[split]).isoformat()

        # Run multiple smaller GP with different seeds for diversity
        all_programs = []
        components_per_run = max(4, n_factors // n_runs + 2)

        for seed_i in range(n_runs):
            st = SymbolicTransformer(
                population_size=population_size,
                tournament_size=10,
                generations=generations,
                n_components=components_per_run,
                hall_of_fame=components_per_run * 2,
                function_set=GP_FUNCTION_SET,
                metric="spearman",
                parsimony_coefficient=parsimony_coefficient,
                max_samples=min(0.8, 4000 / max(len(X_train), 1)),
                random_state=seed_i * 17 + base_seed,
                n_jobs=1,
                verbose=0,
            )
            st.fit(X_train, y_train)
            Xt_train = st.transform(X_train)
            Xt_oos = st.transform(X_oos)

            for i, prog in enumerate(st._best_programs):
                train_col = Xt_train[:, i]
                col = Xt_oos[:, i]
                train_valid = ~(np.isnan(train_col) | np.isinf(train_col))
                valid = ~(np.isnan(col) | np.isinf(col))
                if train_valid.sum() < 10 or valid.sum() < 10:
                    continue
                train_ic, _ = spearmanr(train_col[train_valid], y_train[train_valid])
                ic, _ = spearmanr(col[valid], y_oos[valid])
                if np.isnan(train_ic) or np.isnan(ic):
                    continue
                all_programs.append({
                    "name": f"gp_alpha_{seed_i}_{i:02d}",
                    "expression": str(prog),
                    "fitness": float(prog.fitness_),
                    "train_ic": float(train_ic),
                    "oos_ic": float(ic),
                    "ic": float(ic),
                    "selection_basis": "global_date_20pct_holdout_ic",
                    "train_end": train_end,
                    "oos_start": oos_start,
                    "feature_cols": cols,   # which features X0..Xk map to
                    "y_target": y_target,   # what y this was fit against
                    "_col": col,
                })

        # Rank by abs IC, deduplicate
        all_programs.sort(key=lambda f: abs(f["ic"]), reverse=True)

        kept = []
        kept_cols = []
        for f in all_programs:
            col = f["_col"]
            drop = False
            for kc in kept_cols:
                valid = ~(np.isnan(col) | np.isinf(col) | np.isnan(kc) | np.isinf(kc))
                if valid.sum() < 10:
                    continue
                corr = abs(np.corrcoef(col[valid], kc[valid])[0, 1])
                if corr > dedup_threshold:
                    drop = True
                    break
            if not drop:
                kept.append({k: v for k, v in f.items() if k != "_col"})
                kept_cols.append(col)
            if len(kept) >= n_factors:
                break

        # Store the last transformer for compute_gp_factors fallback
        self._transformers = []
        self._kept_expressions = kept
        return kept

    def compute_gp_factors(
        self,
        historical_data: dict[str, pd.DataFrame],
        mined_factors: list[dict] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Apply mined GP factor expressions to data using eval. Returns {ticker: factors_df}."""
        if not mined_factors:
            return {}

        result = {}
        for ticker, df in historical_data.items():
            feat = _compute_features(df)
            if feat.empty:
                result[ticker] = pd.DataFrame(index=df.index)
                continue

            factor_frames = []
            for f_info in mined_factors:
                # Per-factor feature subset (B11+); fall back to all 13 (B01-B10 legacy)
                cols = f_info.get("feature_cols") or FEATURE_COLS
                try:
                    valid_idx = feat.dropna(subset=cols).index
                    if len(valid_idx) < 1:
                        continue
                    X_tick = feat.loc[valid_idx, cols].values
                    col = _eval_expression(f_info["expression"], X_tick)
                    factor_frames.append(pd.Series(col, index=valid_idx, name=f_info["name"]))
                except Exception:
                    continue

            if factor_frames:
                fdf = pd.concat(factor_frames, axis=1)
                result[ticker] = fdf.reindex(df.index)
            else:
                result[ticker] = pd.DataFrame(index=df.index)
        return result

    @staticmethod
    def save_factors(factors: list[dict], path: str = MINED_ALPHAS_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(factors, f, indent=2)

    @staticmethod
    def load_factors(path: str = MINED_ALPHAS_PATH) -> list[dict]:
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            return json.load(f)

    @staticmethod
    def save_per_account_factors(all_factors: dict[str, list[dict]], path: str = None):
        """Save per-account mined factors: {account_id: [factor_list]}."""
        if path is None:
            path = os.path.join(os.path.dirname(MINED_ALPHAS_PATH), "mined_alphas_per_account.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(all_factors, f, indent=2)

    @staticmethod
    def load_per_account_factors(path: str = None) -> dict[str, list[dict]]:
        """Load per-account mined factors."""
        if path is None:
            path = os.path.join(os.path.dirname(MINED_ALPHAS_PATH), "mined_alphas_per_account.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r") as f:
            return json.load(f)
