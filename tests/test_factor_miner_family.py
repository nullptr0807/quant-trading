import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class FactorMinerFamilyConfigTest(unittest.TestCase):
    def test_f_family_configs_are_enabled_for_us_and_cn_and_isolated_from_b_family(self):
        from accounts.gp_strategies import GP_STRATEGIES

        by_id = {g.id: g for g in GP_STRATEGIES}
        f_ids = [f"F{i:02d}" for i in range(11, 17)]

        self.assertTrue(all(fid in by_id for fid in f_ids))
        for fid in f_ids:
            cfg = by_id[fid]
            self.assertEqual(cfg.family, "F")
            self.assertEqual(cfg.mining_backend, "factor_miner_gp")
            self.assertEqual(tuple(cfg.enabled_markets), ("US", "CN"))
            self.assertGreaterEqual(cfg.gp_global_corr_threshold, 0.5)
            self.assertLessEqual(cfg.gp_global_corr_threshold, 0.7)

        self.assertEqual(by_id["B11"].family, "B")
        self.assertEqual(by_id["B11"].mining_backend, "gplearn")

    def test_market_filter_enables_f_family_for_us_and_cn(self):
        from accounts.gp_strategies import active_gp_strategies_for_market

        us_ids = {g.id for g in active_gp_strategies_for_market("US")}
        cn_ids = {g.id for g in active_gp_strategies_for_market("CN")}

        self.assertIn("F11", us_ids)
        self.assertIn("B11", us_ids)
        self.assertIn("F11", cn_ids)
        self.assertIn("B11", cn_ids)

    def test_minute_updater_stop_loss_map_includes_cn_factor_miner_accounts(self):
        from scripts.update_prices import STOP_LOSS_BY_ACCT

        self.assertIn("F11", STOP_LOSS_BY_ACCT)
        self.assertIn("CF11", STOP_LOSS_BY_ACCT)
        self.assertEqual(STOP_LOSS_BY_ACCT["F11"], STOP_LOSS_BY_ACCT["CF11"])


class FactorMinerFeatureTest(unittest.TestCase):
    def test_extended_features_exist_without_changing_base_default_terminal_set(self):
        from factors.gp_miner import FACTORMINER_FEATURE_COLS, FEATURE_COLS, _compute_features

        dates = pd.date_range("2025-01-01", periods=80, freq="D")
        close = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
        df = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]) * 1.001,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "volume": np.linspace(1_000_000, 2_000_000, len(dates)),
            },
            index=dates,
        )

        feat = _compute_features(df)
        for col in [
            "range_pos",
            "upper_pos",
            "gap_1",
            "dvol_vma20",
            "ret_1_dvol",
            "skew_20",
            "kurt_20",
            "pv_corr_20",
            "slope_20",
            "trend_resi_20",
        ]:
            self.assertIn(col, feat.columns)
            self.assertIn(col, FACTORMINER_FEATURE_COLS)

        self.assertNotIn("range_pos", FEATURE_COLS)
        self.assertNotIn("skew_20", FEATURE_COLS)


class FactorMinerMemoryScreenTest(unittest.TestCase):
    def test_global_correlation_screen_rejects_duplicate_and_writes_memory(self):
        from factors.factor_miner_gp import FactorMinerGPBackend

        dates = pd.date_range("2025-01-01", periods=80, freq="D")
        close = pd.Series(np.linspace(100, 130, len(dates)), index=dates)
        alternating_gap = pd.Series(np.where(np.arange(len(dates)) % 2 == 0, 0.01, -0.01), index=dates)
        df = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]) * (1.0 + alternating_gap),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.linspace(1_000_000, 1_500_000, len(dates)),
            },
            index=dates,
        )
        historical = {"AAA": df, "BBB": df * 1.01}

        with tempfile.TemporaryDirectory() as tmp:
            backend = FactorMinerGPBackend(base_dir=Path(tmp), global_corr_threshold=0.5)
            existing = {
                "F11": [
                    {
                        "name": "fm_F11_00",
                        "expression": "X0",
                        "ic": 0.04,
                        "feature_cols": ["ret_1"],
                        "y_target": "next_1d_ret",
                        "backend": "factor_miner_gp",
                        "family": "F",
                    }
                ],
                "F10": [
                    {
                        "name": "outside_current_family",
                        "expression": "X0",
                        "ic": 0.99,
                        "feature_cols": ["gap_1"],
                        "y_target": "next_1d_ret",
                        "backend": "factor_miner_gp",
                        "family": "F",
                    }
                ],
            }
            candidates = [
                {
                    "name": "dup",
                    "expression": "X0",
                    "ic": 0.05,
                    "feature_cols": ["ret_1"],
                    "y_target": "next_1d_ret",
                },
                {
                    "name": "novel",
                    "expression": "X0",
                    "ic": 0.06,
                    "feature_cols": ["gap_1"],
                    "y_target": "next_1d_ret",
                },
            ]

            admitted, report = backend.screen_candidates(
                account_id="F12",
                candidates=candidates,
                historical_data=historical,
                all_mined=existing,
                family_account_ids={"F11", "F12"},
                n_factors=5,
            )

            self.assertEqual([f["name"] for f in admitted], ["novel"])
            statuses = {item["name"]: item["status"] for item in report["candidates"]}
            self.assertEqual(statuses["dup"], "rejected_corr")
            self.assertEqual(statuses["novel"], "admitted")
            dup_item = next(item for item in report["candidates"] if item["name"] == "dup")
            self.assertEqual(dup_item["conflict_count"], 1)

            memory = json.loads((Path(tmp) / "mining_memory.json").read_text())
            self.assertGreaterEqual(memory["forbidden"]["duplicate"], 1)
            self.assertGreaterEqual(memory["recommended"]["gap"], 1)

    def test_single_conflict_stronger_candidate_replaces_instead_of_rejecting(self):
        from factors.factor_miner_gp import FactorMinerGPBackend

        dates = pd.date_range("2025-01-01", periods=80, freq="D")
        close = pd.Series(np.linspace(100, 130, len(dates)), index=dates)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": np.linspace(1_000_000, 1_500_000, len(dates)),
            },
            index=dates,
        )
        historical = {"AAA": df, "BBB": df * 1.01}

        with tempfile.TemporaryDirectory() as tmp:
            backend = FactorMinerGPBackend(
                base_dir=Path(tmp),
                global_corr_threshold=0.5,
                replacement_ic_mult=1.3,
            )
            old = {
                "name": "weak_old",
                "expression": "X0",
                "ic": 0.04,
                "feature_cols": ["ret_1"],
                "y_target": "next_1d_ret",
                "active": True,
            }
            admitted, report = backend.screen_candidates(
                account_id="F12",
                candidates=[{"name": "strong_new", "expression": "X0", "ic": 0.08, "feature_cols": ["ret_1"]}],
                historical_data=historical,
                all_mined={"F11": [old]},
                family_account_ids={"F11", "F12"},
                n_factors=5,
            )

            self.assertEqual([f["name"] for f in admitted], ["strong_new"])
            self.assertFalse(old["active"])
            self.assertEqual(old["replaced_by"], "strong_new")
            self.assertEqual(report["candidates"][0]["status"], "replacement")


if __name__ == "__main__":
    unittest.main()
