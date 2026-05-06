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

# Feature column names (terminal set)
FEATURE_COLS = [
    "o_c", "h_c", "l_c", "v_vma20",
    "ma_5", "ma_10", "ma_20",
    "std_5", "std_10", "std_20",
    "ret_1", "ret_5", "ret_10",
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
    """Compute normalized features from OHLCV DataFrame."""
    c = df["close"]
    feat = pd.DataFrame(index=df.index)
    feat["o_c"] = df["open"] / c
    feat["h_c"] = df["high"] / c
    feat["l_c"] = df["low"] / c
    v_ma20 = df["volume"].rolling(20).mean()
    feat["v_vma20"] = df["volume"] / v_ma20.replace(0, np.nan)
    feat["ma_5"] = c.rolling(5).mean() / c
    feat["ma_10"] = c.rolling(10).mean() / c
    feat["ma_20"] = c.rolling(20).mean() / c
    feat["std_5"] = c.rolling(5).std() / c
    feat["std_10"] = c.rolling(10).std() / c
    feat["std_20"] = c.rolling(20).std() / c
    feat["ret_1"] = c.pct_change(1)
    feat["ret_5"] = c.pct_change(5)
    feat["ret_10"] = c.pct_change(10)
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
        # forward 5d daily-return Sharpe (no annualization needed; rank IC is scale-invariant)
        fwd_mean = r1.shift(-5).rolling(5).mean().shift(-4)
        fwd_std = r1.shift(-5).rolling(5).std().shift(-4)
        return fwd_mean / fwd_std.replace(0, np.nan)
    if target == "next_5d_minret_neg":
        # worst single-day return over next 5 days, sign-flipped so "high" = resilient
        fwd_min = r1.shift(-5).rolling(5).min().shift(-4)
        return -fwd_min
    if target == "reversal_2d":
        return -((close.shift(-2) / close) - 1.0)
    # fallback
    return r1.shift(-1)


def _prepare_dataset(
    historical_data: dict[str, pd.DataFrame],
    feature_cols: list[str] | None = None,
    y_target: str = "next_1d_ret",
):
    """Stack all tickers into X (features) and y (configurable forward target).

    feature_cols: subset of FEATURE_COLS to use as terminals; None = all 13.
    y_target:    name passed to _build_y_target.
    """
    cols = feature_cols if feature_cols else FEATURE_COLS
    all_X, all_y = [], []
    for ticker, df in historical_data.items():
        feat = _compute_features(df)
        target = _build_y_target(df["close"], y_target)
        combined = feat.assign(target=target).dropna()
        if len(combined) < 5:
            continue
        all_X.append(combined[cols].values)
        all_y.append(combined["target"].values)
    if not all_X:
        return np.empty((0, len(cols))), np.empty(0)
    return np.vstack(all_X), np.concatenate(all_y)


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
        X, y = _prepare_dataset(historical_data, feature_cols=cols, y_target=y_target)
        if len(X) < 10:
            return []

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
                max_samples=min(0.8, 4000 / max(len(X), 1)),
                random_state=seed_i * 17 + base_seed,
                n_jobs=1,
                verbose=0,
            )
            st.fit(X, y)
            Xt = st.transform(X)

            for i, prog in enumerate(st._best_programs):
                col = Xt[:, i]
                valid = ~(np.isnan(col) | np.isinf(col))
                if valid.sum() < 10:
                    continue
                ic, _ = spearmanr(col[valid], y[valid])
                if np.isnan(ic):
                    continue
                all_programs.append({
                    "name": f"gp_alpha_{seed_i}_{i:02d}",
                    "expression": str(prog),
                    "fitness": float(prog.fitness_),
                    "ic": float(ic),
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
            valid_idx = feat.dropna().index
            if len(valid_idx) < 1:
                result[ticker] = pd.DataFrame(index=df.index)
                continue

            factor_cols = {}
            for f_info in mined_factors:
                # Per-factor feature subset (B11+); fall back to all 13 (B01-B10 legacy)
                cols = f_info.get("feature_cols") or FEATURE_COLS
                try:
                    X_tick = feat.loc[valid_idx, cols].values
                    col = _eval_expression(f_info["expression"], X_tick)
                    factor_cols[f_info["name"]] = col
                except Exception:
                    factor_cols[f_info["name"]] = np.full(len(valid_idx), np.nan)

            fdf = pd.DataFrame(factor_cols, index=valid_idx)
            result[ticker] = fdf.reindex(df.index)
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
