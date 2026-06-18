import unittest


class GPRuntimeWindowTest(unittest.TestCase):
    def test_factor_miner_accounts_use_extended_compute_window(self):
        from main import QuantSystem

        system = QuantSystem(market="US")

        self.assertGreaterEqual(system._gp_compute_history_days(), 90)

    def test_failure_marker_is_not_tradeable_and_prevents_missing_retries(self):
        from main import QuantSystem
        from accounts.gp_strategies import active_gp_strategies_for_market

        system = QuantSystem(market="US")
        f14 = next(g for g in active_gp_strategies_for_market("US") if g.id == "F14")
        marker = system._factor_miner_failure_marker(f14, "all_candidates_rejected_by_strict_red_sea_screen")

        self.assertFalse(system._is_active_mined_factor(marker))
        self.assertEqual(system._active_mined_count([marker]), 0)

        system._per_account_mined = {}
        for g in system.gp_strategies:
            if getattr(g, "mining_backend", "gplearn") == "factor_miner_gp":
                system._per_account_mined[g.id] = [marker]
            else:
                system._per_account_mined[g.id] = [{"name": f"{g.id}_dummy", "expression": "X0", "active": True}]
        self.assertEqual(system._missing_gp_strategies(), [])


if __name__ == "__main__":
    unittest.main()
