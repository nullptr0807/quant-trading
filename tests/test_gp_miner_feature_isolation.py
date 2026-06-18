import unittest

import numpy as np
import pandas as pd


class GPMinerFeatureIsolationTest(unittest.TestCase):
    def _constant_volume_frame(self):
        idx = pd.date_range("2025-01-01", periods=80, freq="D")
        close = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
        return pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                # constant volume makes some expanded features all-NaN, e.g. pv_corr_20
                "volume": 1_000_000.0,
            },
            index=idx,
        )

    def test_prepare_dataset_drops_only_requested_feature_subset(self):
        from factors.gp_miner import _prepare_dataset

        X, y = _prepare_dataset({"AAA": self._constant_volume_frame()}, feature_cols=["ret_1"], y_target="next_1d_ret")

        self.assertGreater(len(X), 20)
        self.assertEqual(X.shape[1], 1)
        self.assertEqual(len(X), len(y))

    def test_compute_gp_factors_uses_per_factor_valid_rows_not_global_dropna(self):
        from factors.gp_miner import GPAlphaMiner

        miner = GPAlphaMiner()
        factors = [
            {
                "name": "ret_factor",
                "expression": "X0",
                "feature_cols": ["ret_1"],
                "ic": 0.1,
            }
        ]
        result = miner.compute_gp_factors({"AAA": self._constant_volume_frame()}, factors)

        self.assertIn("AAA", result)
        self.assertIn("ret_factor", result["AAA"].columns)
        self.assertGreater(result["AAA"]["ret_factor"].notna().sum(), 20)


if __name__ == "__main__":
    unittest.main()
