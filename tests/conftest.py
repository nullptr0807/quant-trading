"""Test guard: keep this checkout ahead of the production repo on sys.path."""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = str(Path(__file__).resolve().parents[1])
PRODUCTION_ROOT = str(Path.home() / "quant-trading")


def pytest_sessionstart(session):
    for name in tuple(sys.modules):
        if name == "scripts" or name.startswith("scripts."):
            sys.modules.pop(name, None)
    sys.path[:] = [p for p in sys.path if p != PRODUCTION_ROOT]
    if ROOT in sys.path:
        sys.path.remove(ROOT)
    sys.path.insert(0, ROOT)


def pytest_runtest_setup(item):
    # Some legacy modules still prepend ~/quant-trading during import. Remove it
    # before each test so later namespace-package imports cannot resolve there.
    for name, module in tuple(sys.modules.items()):
        path = str(getattr(module, "__file__", "") or "")
        if (name == "scripts" or name.startswith("scripts.")) and path.startswith(PRODUCTION_ROOT):
            sys.modules.pop(name, None)
    sys.path[:] = [p for p in sys.path if p != PRODUCTION_ROOT]
    if ROOT in sys.path:
        sys.path.remove(ROOT)
    sys.path.insert(0, ROOT)
