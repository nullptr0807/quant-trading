"""FactorMiner-style GP backend.

This keeps gplearn as the candidate generator, but wraps it with the pieces
from the FactorMiner paper that matter for our experiment:

- expanded terminal set (provided by factors.gp_miner._compute_features)
- F-family global correlation screening
- replacement of weaker correlated F factors
- human-readable mining memory and per-run logs

It is intentionally isolated from legacy B-family mining. B accounts still use
GPAlphaMiner directly and persist to factors/mined_alphas_per_account.json.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from factors.gp_miner import GPAlphaMiner, _compute_features, _eval_expression

DEFAULT_BASE_DIR = Path(os.path.expanduser("~/quant-trading/factors/factor_miner_gp"))
DEFAULT_FACTORS_PATH = DEFAULT_BASE_DIR / "mined_alphas_f.json"


class FactorMinerGPBackend:
    """FactorMiner-style wrapper around GPAlphaMiner candidate generation."""

    def __init__(
        self,
        base_dir: str | Path = DEFAULT_BASE_DIR,
        global_corr_threshold: float = 0.6,
        replacement_ic_mult: float = 1.3,
    ):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.base_dir / "mining_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path = self.base_dir / "mining_memory.json"
        self.global_corr_threshold = global_corr_threshold
        self.replacement_ic_mult = replacement_ic_mult

    def mine_factors(
        self,
        account_id: str,
        historical_data: dict[str, pd.DataFrame],
        all_mined: dict[str, list[dict]],
        family_account_ids: Iterable[str],
        n_factors: int = 12,
        generations: int = 20,
        base_seed: int = 42,
        population_size: int = 300,
        n_runs: int = 4,
        parsimony_coefficient: float = 0.006,
        y_target: str = "next_1d_ret",
        feature_subset: list[str] | tuple[str, ...] | None = None,
        dedup_threshold: float = 0.6,
    ) -> list[dict]:
        """Generate GP candidates, then admit through F-family memory screen."""
        generator = GPAlphaMiner()
        # Ask for extra candidates because global correlation filtering is stricter
        # than a normal account-local gplearn run.
        candidate_count = max(n_factors * 3, n_factors + 8)
        candidates = generator.mine_factors(
            historical_data,
            n_factors=candidate_count,
            generations=generations,
            base_seed=base_seed,
            population_size=population_size,
            n_runs=n_runs,
            parsimony_coefficient=parsimony_coefficient,
            y_target=y_target,
            feature_subset=feature_subset,
            dedup_threshold=dedup_threshold,
        )
        admitted, _report = self.screen_candidates(
            account_id=account_id,
            candidates=candidates,
            historical_data=historical_data,
            all_mined=all_mined,
            family_account_ids=set(family_account_ids),
            n_factors=n_factors,
        )
        return admitted

    def screen_candidates(
        self,
        account_id: str,
        candidates: list[dict],
        historical_data: dict[str, pd.DataFrame],
        all_mined: dict[str, list[dict]],
        family_account_ids: set[str],
        n_factors: int,
    ) -> tuple[list[dict], dict]:
        """Apply FactorMiner-style global F-family corr/replacement screening.

        Important invariant: we do NOT admit a correlated fallback merely to keep
        an account invested. That would violate the FactorMiner "correlation red
        sea" premise. The only correlation escape hatch is the paper-compatible
        replacement rule: a stronger candidate may replace exactly one weaker
        active factor, non-destructively, while audit history is preserved.
        """
        family_library: list[tuple[str, dict]] = []
        for acct, factors in (all_mined or {}).items():
            if acct in family_account_ids and acct != account_id:
                family_library.extend((acct, f) for f in factors if f.get("active", True))

        # Within a re-mine of the same account, treat newly admitted candidates as
        # part of the family library so the batch also deduplicates globally.
        admitted: list[dict] = []
        admitted_library: list[tuple[str, dict]] = []
        report = {
            "account_id": account_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "global_corr_threshold": self.global_corr_threshold,
            "replacement_ic_mult": self.replacement_ic_mult,
            "screen_policy": "strict_red_sea_with_single_conflict_replacement",
            "candidates": [],
        }

        for raw in sorted(candidates or [], key=lambda f: abs(float(f.get("ic", 0.0))), reverse=True):
            cand = dict(raw)
            cand["backend"] = "factor_miner_gp"
            cand["family"] = "F"
            cand["active"] = True
            cand["source_account"] = account_id
            cand["name"] = self._name_for(account_id, cand, len(admitted))

            compare_library = family_library + admitted_library
            conflicts = self._conflicts(cand, compare_library, historical_data)
            nearest = conflicts[0] if conflicts else self._nearest_factor(cand, compare_library, historical_data)
            status = "admitted"
            replace_target = None
            if conflicts:
                conflict = conflicts[0]
                old = conflict["factor"]
                old_ic = abs(float(old.get("ic", 0.0)))
                new_ic = abs(float(cand.get("ic", 0.0)))
                only_one_conflict = len(conflicts) == 1
                if only_one_conflict and old_ic > 0 and new_ic >= old_ic * self.replacement_ic_mult:
                    status = "replacement"
                    replace_target = conflict
                else:
                    status = "rejected_corr"

            item = {
                "name": cand["name"],
                "expression": cand.get("expression"),
                "ic": float(cand.get("ic", 0.0)),
                "feature_cols": cand.get("feature_cols"),
                "y_target": cand.get("y_target"),
                "status": status,
                "nearest": self._nearest_summary(nearest),
                "conflict_count": len(conflicts),
                "conflicts": [self._nearest_summary(c) for c in conflicts[:5]],
                "family_tag": self._family_tag(cand),
            }
            report["candidates"].append(item)

            if status == "rejected_corr":
                continue
            if status == "replacement" and replace_target:
                old = replace_target["factor"]
                # Non-destructive replacement: mark the old factor inactive so
                # audit history stays intact, but future screening/trading uses
                # the stronger replacement. This mirrors FactorMiner's library
                # evolution without deleting raw upstream artifacts.
                old["active"] = False
                old["replaced_by"] = cand["name"]
                old["replaced_at"] = report["timestamp"]
                item["replace_target"] = self._nearest_summary(replace_target)
                family_library = [(a, f) for a, f in family_library if f is not old]
                admitted = [f for f in admitted if f is not old]
                admitted_library = [(a, f) for a, f in admitted_library if f is not old]
            admitted.append(cand)
            admitted_library.append((account_id, cand))
            if len(admitted) >= n_factors:
                break

        if not admitted:
            report["failure_reason"] = (
                "no_candidates" if not candidates else "all_candidates_rejected_by_strict_red_sea_screen"
            )
        self._update_memory(report)
        self._write_report(account_id, report)
        return admitted, report

    @staticmethod
    def load_factors(path: str | Path = DEFAULT_FACTORS_PATH) -> dict[str, list[dict]]:
        p = Path(path)
        if not p.exists():
            return {}
        with p.open() as f:
            return json.load(f)

    @staticmethod
    def save_factors(all_factors: dict[str, list[dict]], path: str | Path = DEFAULT_FACTORS_PATH):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            json.dump(all_factors, f, indent=2)

    def _nearest_factor(
        self,
        cand: dict,
        library: list[tuple[str, dict]],
        historical_data: dict[str, pd.DataFrame],
    ) -> dict | None:
        cand_vec = self._factor_vector(cand, historical_data)
        if cand_vec is None:
            return None
        best = None
        for acct, factor in library:
            vec = self._factor_vector(factor, historical_data)
            corr = self._abs_corr(cand_vec, vec)
            if corr is None:
                continue
            if best is None or corr > best["abs_corr"]:
                best = {"account": acct, "factor": factor, "abs_corr": corr}
        return best

    def _conflicts(
        self,
        cand: dict,
        library: list[tuple[str, dict]],
        historical_data: dict[str, pd.DataFrame],
    ) -> list[dict]:
        cand_vec = self._factor_vector(cand, historical_data)
        if cand_vec is None:
            return []
        conflicts = []
        for acct, factor in library:
            corr = self._abs_corr(cand_vec, self._factor_vector(factor, historical_data))
            if corr is not None and corr >= self.global_corr_threshold:
                conflicts.append({"account": acct, "factor": factor, "abs_corr": corr})
        conflicts.sort(key=lambda x: x["abs_corr"], reverse=True)
        return conflicts

    def _conflict_count(
        self,
        cand: dict,
        library: list[tuple[str, dict]],
        historical_data: dict[str, pd.DataFrame],
    ) -> int:
        return len(self._conflicts(cand, library, historical_data))

    @staticmethod
    def _factor_vector(factor: dict, historical_data: dict[str, pd.DataFrame]) -> pd.Series | None:
        pieces = []
        cols = factor.get("feature_cols")
        if not cols:
            return None
        for ticker, df in historical_data.items():
            try:
                feat = _compute_features(df)
                valid_idx = feat.dropna(subset=cols).index
                if len(valid_idx) == 0:
                    continue
                values = _eval_expression(factor["expression"], feat.loc[valid_idx, cols].values)
                s = pd.Series(values, index=pd.MultiIndex.from_product([[ticker], valid_idx]))
                pieces.append(s.replace([np.inf, -np.inf], np.nan).dropna())
            except Exception:
                continue
        if not pieces:
            return None
        out = pd.concat(pieces)
        return out if len(out) >= 10 else None

    @staticmethod
    def _abs_corr(a: pd.Series | None, b: pd.Series | None) -> float | None:
        if a is None or b is None:
            return None
        df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
        if len(df) < 10:
            return None
        corr = df["a"].rank(pct=True).corr(df["b"].rank(pct=True))
        if pd.isna(corr):
            return None
        return abs(float(corr))

    @staticmethod
    def _name_for(account_id: str, cand: dict, idx: int) -> str:
        expr_name = str(cand.get("name", f"cand_{idx}"))
        if expr_name.startswith(f"fm_{account_id}_"):
            return expr_name
        # Real gplearn candidates are gp_alpha_* and would collide across
        # accounts if persisted as-is. Unit-test/custom candidates can keep
        # explicit names for readability.
        if expr_name.startswith("gp_alpha_"):
            return f"fm_{account_id}_{idx:02d}_{expr_name}"
        return expr_name

    @staticmethod
    def _nearest_summary(nearest: dict | None) -> dict | None:
        if not nearest:
            return None
        f = nearest["factor"]
        return {
            "account": nearest["account"],
            "name": f.get("name"),
            "expression": f.get("expression"),
            "ic": f.get("ic"),
            "abs_corr": nearest["abs_corr"],
        }

    @staticmethod
    def _family_tag(factor: dict) -> str:
        cols = set(factor.get("feature_cols") or [])
        expr = str(factor.get("expression", "")).lower()
        if "gap_1" in cols:
            return "gap"
        if cols & {"ret_1", "ret_5", "ret_10"}:
            return "return_momentum"
        if cols & {"v_vma20", "dvol_vma20", "ret_1_dvol", "absret_1_dvol", "pv_corr_20"}:
            return "volume_liquidity"
        if cols & {"std_5", "std_10", "std_20", "vol_of_vol_20", "skew_20", "kurt_20"}:
            return "risk_regime"
        if cols & {"slope_20", "trend_r2_20", "trend_resi_20"} or "resi" in expr:
            return "trend_regression"
        if cols & {"range_pos", "upper_pos", "lower_shadow", "upper_shadow"}:
            return "range_candle"
        return "other"

    def _update_memory(self, report: dict):
        if self.memory_path.exists():
            try:
                memory = json.loads(self.memory_path.read_text())
            except Exception:
                memory = {}
        else:
            memory = {}
        memory.setdefault("recommended", {})
        memory.setdefault("forbidden", {})
        memory.setdefault("recent_reports", [])

        rec = Counter(memory.get("recommended", {}))
        forb = Counter(memory.get("forbidden", {}))
        for item in report.get("candidates", []):
            tag = item.get("family_tag", "other")
            if item.get("status") in {"admitted", "replacement"}:
                rec[tag] += 1
            elif item.get("status") == "rejected_corr":
                forb["duplicate"] += 1
                forb[tag] += 1
        memory["recommended"] = dict(rec)
        memory["forbidden"] = dict(forb)
        memory["recent_reports"] = (memory.get("recent_reports", []) + [
            {
                "account_id": report.get("account_id"),
                "timestamp": report.get("timestamp"),
                "admitted": sum(1 for c in report.get("candidates", []) if c.get("status") in {"admitted", "replacement"}),
                "rejected_corr": sum(1 for c in report.get("candidates", []) if c.get("status") == "rejected_corr"),
            }
        ])[-20:]
        self.memory_path.write_text(json.dumps(memory, indent=2))

    def _write_report(self, account_id: str, report: dict):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.logs_dir / f"{ts}_{account_id}.json"
        path.write_text(json.dumps(report, indent=2))
