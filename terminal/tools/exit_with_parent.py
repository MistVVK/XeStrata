"""exit_with_parent(): end this process when the process that started it ends.

setup.py starts its preparation tools with STRATA_EXIT_WITH_PARENT set to its own process id (setup.run; the desktop app sets 1). A tool that calls this at its start does not go on writing
when setup is cancelled or killed: a kill that reaches setup.py but not its children (the app's Prepare and load, quit while the model packer
runs) would otherwise leave the packer running on its own. tools/mtp_fetch.py has the same code for its downloads (#3257).
"""
import os
import threading
import time


def exit_with_parent(env=None, poll=1.5):
    """Start the watcher when STRATA_EXIT_WITH_PARENT is 1 (the parent) or a process id. Returns True when one runs. The process ends with exit code 1 (os._exit: a packer
    in the middle of a write is stopped at once, the files it was writing are the resumable temporary ones)."""
    env = os.environ if env is None else env
    raw = (env.get("STRATA_EXIT_WITH_PARENT") or "").strip()
    if raw == "1":                                           # "whoever started me": read the sentinel before the digits
        parent = os.getppid()
    elif raw.isdigit():                                      # setup.py names its own id (#6190): a venv's python.exe on Windows is a launcher
        parent = int(raw)                                    # whose child is the interpreter, so getppid() there is a stub, not setup
    else:
        return False
    if parent <= 1 or parent == os.getpid():
        return False
    direct = os.getppid()
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
        def _alive(pid):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return False
            except PermissionError:
                return True                                  # it exists; it is not ours to signal
            return True
        gone = lambda: os.getppid() != direct or not _alive(parent)

    def watch():
        while not gone():
            time.sleep(poll)
        try:
            os.write(2, b"the process that started this step has ended: stopping\n")
        except Exception:                                        # noqa: BLE001 - the stream may be a pipe to a process that is gone: the write fails, the exit must not
            pass
        os._exit(1)
    threading.Thread(target=watch, daemon=True).start()
    return True
