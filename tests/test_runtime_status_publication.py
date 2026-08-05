from types import SimpleNamespace

from main import QuantSystem


class Store:
    def __init__(self):
        self.rows = []

    def set_account_runtime_status(self, *args):
        self.rows.append(args)


def test_gp_runtime_status_distinguishes_ready_and_non_tradeable():
    system = object.__new__(QuantSystem)
    system.market = "US"
    system.store = Store()
    system.gp_strategies = [SimpleNamespace(id="F14"), SimpleNamespace(id="F15"), SimpleNamespace(id="F16")]
    system._per_account_mined = {
        "F14": [{"status": "mining_failed", "active": False, "reason": "NO_ADMISSIBLE_FACTOR"}],
        "F15": [{"expression": "add(X0,X1)", "active": True}],
        "F16": [],
    }

    system._publish_gp_runtime_statuses()

    rows = {row[0]: row for row in system.store.rows}
    assert rows["F14"][2:4] == ("non_tradeable", "NO_ADMISSIBLE_FACTOR")
    assert rows["F15"][2] == "ready"
    assert rows["F15"][4]["active_factors"] == 1
    assert rows["F16"][2:4] == ("non_tradeable", "empty_config")
