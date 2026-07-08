import unittest


class AlphaRuntimeWindowTest(unittest.TestCase):
    def test_alpha158_live_window_has_20d_warmup_margin(self):
        from main import QuantSystem

        system = QuantSystem(market="US")
        self.assertGreaterEqual(system._alpha_history_days(), 60)


if __name__ == "__main__":
    unittest.main()
