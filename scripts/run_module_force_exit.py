#!/usr/bin/env python3
"""Run a one-shot module and terminate leftover provider worker threads.

Some third-party market-data calls ignore cancellation and leave non-daemon
ThreadPoolExecutor workers alive after the job has committed and returned. The
scheduler already bounds the whole process; this runner additionally exits only
after the module's main path has returned, flushing logs first. It must only be
used for one-shot CLIs whose durable writes are committed before return.
"""
from __future__ import annotations

import os
import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_module_force_exit.py module [args...]")
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    code = 0
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=False)
    except SystemExit as exc:
        code = int(exc.code or 0) if isinstance(exc.code, (int, type(None))) else 1
    except BaseException:
        code = 1
        raise
    finally:
        try:
            sys.stdout.flush(); sys.stderr.flush()
        finally:
            if code == 0:
                os._exit(0)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
