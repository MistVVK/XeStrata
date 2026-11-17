"""The product never starts a debugger on a user's machine.

When ggml aborts (a failed GGML_ASSERT, an uncaught C++ exception in a program that links ggml) it prints a backtrace first: on Linux it forks
`gdb --batch -ex attach <pid>` (then lldb, then backtrace_symbols), on macOS it prints frames and three WARNING lines, on Windows nothing.  A
debugger that attaches to a process the server is about to end writes its own errors into the log ("Couldn't get registers: No such process")
and, because it inherits the reader's stdout, into the pipe the server reads the reader's READY line from.  `GGML_NO_BACKTRACE` (any value)
switches it all off.  Every child Strata starts gets it, unless a developer asks for the backtrace with STRATA_BACKTRACE=1 (then nothing is
set here, and the environment is left as the developer made it).
"""
from __future__ import annotations

import os

NO_BACKTRACE = "GGML_NO_BACKTRACE"
DEVELOPER_SWITCH = "STRATA_BACKTRACE"


def developer_wants_backtrace(env=None) -> bool:
    """STRATA_BACKTRACE is set to anything but empty / 0 / no / off / false, in `env` or in this process's environment."""
    for source in (env, os.environ):
        value = str((source or {}).get(DEVELOPER_SWITCH, "")).strip().lower()
        if value and value not in ("0", "no", "off", "false"):
            return True
    return False


def quiet_crash_env(env: dict | None = None) -> dict:
    """`env` (this process's own environment when None) with GGML_NO_BACKTRACE=1 set unless it is already there or the developer switch is on.
    Changes the dict it is given and returns it, so it fits where an environment is built: `env = quiet_crash_env(dict(os.environ))`."""
    target = os.environ if env is None else env
    if NO_BACKTRACE not in target and not developer_wants_backtrace(target):
        target[NO_BACKTRACE] = "1"
    return target
