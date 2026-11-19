"""End this process when the process that started it ends (standard library only; strata.py calls it first thing).

The app host starts Control programs that download and pack for minutes (`strata update`, `strata drafting fetch`, the picture reader's
fetch) with STRATA_EXIT_WITH_PARENT=1. When the app quits, the host stops them; this is the second line for the case where the host itself
is gone without saying so (killed, crashed): the program notices its parent has ended and stops instead of writing on for minutes.
tools/exit_with_parent.py is the same code for the preparation tools under tools/ (they cannot import strata_control).
"""
from __future__ import annotations

import os
import threading
import time


def exit_with_parent(env=None, poll: float = 1.5) -> bool:
    """Start the watcher when STRATA_EXIT_WITH_PARENT=1. True when one runs. The process ends with exit code 1 (os._exit: nothing is flushed;
    what the program was writing is the part its next run resumes or redoes)."""
    env = os.environ if env is None else env
    if (env.get("STRATA_EXIT_WITH_PARENT") or "").strip() != "1":
        return False
    parent = os.getppid()
    if parent <= 1:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = ctypes.c_void_p
        k32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        handle = k32.OpenProcess(0x00100000, False, parent)      # SYNCHRONIZE: a handle keeps the answer true if the id is reused
        if not handle:
            return False                                         # no access (or already gone): nothing to watch
        gone = lambda: k32.WaitForSingleObject(handle, 0) == 0
    else:
        gone = lambda: os.getppid() != parent

    def watch():
        while not gone():
            time.sleep(poll)
        try:
            os.write(2, b"the program that started this one has ended: stopping\n")
        except Exception:                                        # noqa: BLE001 - stderr is usually a pipe to the host that is gone: the write fails, the exit must not
            pass
        os._exit(1)
    threading.Thread(target=watch, daemon=True).start()
    return True
