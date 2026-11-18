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
import sys

NO_BACKTRACE = "GGML_NO_BACKTRACE"
DEVELOPER_SWITCH = "STRATA_BACKTRACE"

#: glibc gives every thread that allocates its own memory arena, up to eight per core; a server that starts a thread per request keeps a
#: little of each. Measured on the serve soak (l620, 120 min, same head, engine and schedule): server RSS 312 MiB and +4.5 %/h by default,
#: 252 MiB and +0.69 %/h with MALLOC_ARENA_MAX=2. A person never sets environment variables, so Control sets it where it starts the server
#: and the engine. An explicit value (the person's, a service file's) always wins. Linux only: no other C library reads it.
ARENA_VAR = "MALLOC_ARENA_MAX"
ARENA_VALUE = "2"
M_ARENA_MAX = -8                                    # glibc's mallopt(3) parameter number


def arena_env(env: dict | None = None) -> dict:
    """`env` (this process's own environment when None) with MALLOC_ARENA_MAX=2 on Linux unless it is already there. Changes the dict it is
    given and returns it, so it fits where a child's environment is built. It acts on the CHILDREN: a process reads the variable once, at start."""
    target = os.environ if env is None else env
    if sys.platform.startswith("linux") and ARENA_VAR not in target:
        target[ARENA_VAR] = ARENA_VALUE
    return target


def limit_own_arenas() -> bool:
    """The same limit for THIS process, for a server that was started by hand (nothing set the variable before it began): glibc's
    mallopt(M_ARENA_MAX, 2) works while the process runs. False (and nothing done) when the variable was set at start, off Linux, or when the
    C library has no such call (musl)."""
    if not sys.platform.startswith("linux") or ARENA_VAR in os.environ:
        return False
    try:
        import ctypes
        return bool(ctypes.CDLL(None).mallopt(M_ARENA_MAX, int(ARENA_VALUE)))
    except (OSError, AttributeError, ValueError):
        return False


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
