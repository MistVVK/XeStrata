"""exit_with_parent(): end this process when the process that started it ends.

setup.py starts its preparation tools with STRATA_EXIT_WITH_PARENT=1 (setup.run). A tool that calls this at its start does not go on writing
when setup is cancelled or killed: a kill that reaches setup.py but not its children (the app's Prepare and load, quit while the model packer
runs) would otherwise leave the packer running on its own. tools/mtp_fetch.py has the same code for its downloads (#3257).
"""
import os
import threading
import time


def exit_with_parent(env=None, poll=1.5):
    """Start the watcher when STRATA_EXIT_WITH_PARENT=1. Returns True when one runs. The process ends with exit code 1 (os._exit: a packer
    in the middle of a write is stopped at once, the files it was writing are the resumable temporary ones)."""
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
        h = k32.OpenProcess(0x00100000, False, parent)       # SYNCHRONIZE: a handle keeps the answer true if the id is reused
        if not h:
            return False                                     # no access (or already gone): nothing to watch
        gone = lambda: k32.WaitForSingleObject(h, 0) == 0
    else:
        gone = lambda: os.getppid() != parent

    def watch():
        while not gone():
            time.sleep(poll)
        os.write(2, b"the process that started this step has ended: stopping\n")
        os._exit(1)
    threading.Thread(target=watch, daemon=True).start()
    return True
