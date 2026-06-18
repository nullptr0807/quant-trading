import unittest


class SignalGeneratorCompatTest(unittest.TestCase):
    def test_accepts_decorrelate_keyword_used_by_main(self):
        from factors.signal import SignalGenerator

        gen = SignalGenerator(buy_top=30, sell_top=10, decorrelate=False)

        self.assertEqual(gen.buy_top, 30)
        self.assertEqual(gen.sell_top, 10)
        self.assertFalse(gen.decorrelate)


if __name__ == "__main__":
    unittest.main()
