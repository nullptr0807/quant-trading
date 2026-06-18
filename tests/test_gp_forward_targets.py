import unittest

import pandas as pd


class GPForwardTargetAlignmentTest(unittest.TestCase):
    def test_next_5d_sharpe_uses_immediate_next_five_returns(self):
        from factors.gp_miner import _build_y_target

        close = pd.Series([100, 101, 103, 106, 110, 115, 121, 128, 136, 145], dtype=float)
        r1 = close.pct_change()

        got = _build_y_target(close, "next_5d_sharpe")
        expected_t0 = r1.iloc[1:6].mean() / r1.iloc[1:6].std()
        expected_t1 = r1.iloc[2:7].mean() / r1.iloc[2:7].std()

        self.assertAlmostEqual(float(got.iloc[0]), float(expected_t0), places=12)
        self.assertAlmostEqual(float(got.iloc[1]), float(expected_t1), places=12)
        self.assertTrue(got.iloc[-4:].isna().all())

    def test_next_5d_minret_neg_uses_immediate_next_five_returns(self):
        from factors.gp_miner import _build_y_target

        close = pd.Series([100, 99, 101, 98, 103, 102, 108, 107, 109, 111], dtype=float)
        r1 = close.pct_change()

        got = _build_y_target(close, "next_5d_minret_neg")
        expected_t0 = -r1.iloc[1:6].min()
        expected_t1 = -r1.iloc[2:7].min()

        self.assertAlmostEqual(float(got.iloc[0]), float(expected_t0), places=12)
        self.assertAlmostEqual(float(got.iloc[1]), float(expected_t1), places=12)
        self.assertTrue(got.iloc[-4:].isna().all())


if __name__ == "__main__":
    unittest.main()
