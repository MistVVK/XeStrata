"""serve/winjob.py: a contained child - and whatever it starts - ends when the process that contained it ends, even
when that process is killed outright (the console window closed, Task Manager), with no cleanup of its own.

    python -m unittest serve.test_winjob      (Windows only; skipped elsewhere)

The test holds a handle to the child and to the grandchild (kernel32, from the standard library) instead of asking `tasklist` whether a PID is
listed: a handle names one process object, so a PID that Windows hands out again to some other program cannot pass for the child, and when a
process is still there after the wait the failure says which process it is (image name, creation time, seconds since the kill).
"""
import os
import subprocess
import sys
import time
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# stands in for the server: contains a child that starts a grandchild, prints both pids, then waits to be killed
PARENT = f"""
import subprocess, sys, time
sys.path.insert(0, {str(ROOT)!r})
from serve.winjob import contain
code = ("import subprocess, sys, time; g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        "print(g.pid, flush=True); time.sleep(120)")
child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
ok = contain(child)
print(ok, child.pid, child.stdout.readline().strip(), flush=True)
time.sleep(120)
"""

_PROCESS_TERMINATE, _PROCESS_QUERY_LIMITED_INFORMATION, _SYNCHRONIZE = 0x0001, 0x1000, 0x00100000
_WAIT_OBJECT_0 = 0
_FILETIME_UNIX_EPOCH_S = 11644473600          # seconds from 1601-01-01 (FILETIME) to 1970-01-01

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _k32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _k32.WaitForSingleObject.restype = wintypes.DWORD
    _k32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    _k32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    _k32.GetProcessTimes.argtypes = (wintypes.HANDLE,) + (ctypes.POINTER(wintypes.FILETIME),) * 4
    _k32.QueryFullProcessImageNameW.argtypes = (wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD))


def _identity(handle) -> tuple[str, float]:
    """(image file name, creation time as seconds since 1970) of the process behind an open handle."""
    created, a, b, c = (wintypes.FILETIME() for _ in range(4))
    if not _k32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(a), ctypes.byref(b), ctypes.byref(c)):
        raise ctypes.WinError(ctypes.get_last_error())
    text, size = ctypes.create_unicode_buffer(1024), wintypes.DWORD(1024)
    name = Path(text.value).name if _k32.QueryFullProcessImageNameW(handle, 0, text, ctypes.byref(size)) else "?"
    return name, ((created.dwHighDateTime << 32) | created.dwLowDateTime) / 1e7 - _FILETIME_UNIX_EPOCH_S


def _when(seconds: float) -> str:
    return datetime.fromtimestamp(seconds).strftime("%H:%M:%S.%f")[:-3]


class Held:
    """A process held by a handle. The handle keeps the process object (and so its identity) alive; `exited()` asks that object, never the PID."""

    def __init__(self, pid: int, label: str):
        self.pid, self.label = pid, label
        self.handle = _k32.OpenProcess(_SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION | _PROCESS_TERMINATE, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.image, self.created = _identity(self.handle)

    def exited(self) -> bool:
        return _k32.WaitForSingleObject(self.handle, 0) == _WAIT_OBJECT_0

    def holder_of_the_pid_now(self) -> str:
        """What a fresh OpenProcess by PID finds: image name and creation time, and whether it is the process held here."""
        handle = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, self.pid)
        if not handle:
            return "no process has the PID"
        try:
            image, created = _identity(handle)
        finally:
            _k32.CloseHandle(handle)
        same = abs(created - self.created) < 0.001           # the creation time is what tells two processes with one PID apart (an exited process has no image name to ask for)
        return "%s created %s (%s)" % (image if image != "?" else "an exited process", _when(created), "the same process" if same else "NOT the process held")

    def explain(self, since_kill: float) -> str:
        listed = subprocess.run(["tasklist", "/FI", f"PID eq {self.pid}", "/NH"], capture_output=True, text=True).stdout.strip().replace("\n", " | ")
        return ("%s outlived the server: pid %d is %s, created %s; %.1f s after the kill it is still running; the PID is held now by: %s; tasklist: %s"
                % (self.label, self.pid, self.image, _when(self.created), since_kill, self.holder_of_the_pid_now(), listed or "-"))

    def end(self) -> None:
        if self.handle:
            if not self.exited():                     # on a failure, don't leave them sleeping (this very process, by handle)
                _k32.TerminateProcess(self.handle, 1)
            _k32.CloseHandle(self.handle)
            self.handle = None


@unittest.skipUnless(os.name == "nt", "job objects are Windows-only")
class TestWinJob(unittest.TestCase):
    def test_children_end_with_a_killed_server(self):
        # a venv's python.exe is a launcher that runs the real interpreter as its child: kill() must reach the latter
        python = getattr(sys, "_base_executable", None) or sys.executable
        parent = subprocess.Popen([python, "-c", PARENT], stdout=subprocess.PIPE, text=True)
        child = grandchild = None
        try:
            ok, child_pid, grandchild_pid = parent.stdout.readline().split()
            child, grandchild = Held(int(child_pid), "the child"), Held(int(grandchild_pid), "the grandchild")
            self.assertEqual(ok, "True")
            self.assertFalse(child.exited() or grandchild.exited())
            parent.kill()                       # TerminateProcess: no atexit, no finally - like closing the window
            t_kill = time.monotonic()
            parent.wait(10)
            deadline = time.monotonic() + 10
            while not (child.exited() and grandchild.exited()) and time.monotonic() < deadline:
                time.sleep(0.05)
            for held in (child, grandchild):
                if not held.exited():
                    self.fail(held.explain(time.monotonic() - t_kill))
        finally:
            if parent.poll() is None:
                parent.kill()
            for held in (child, grandchild):
                if held is not None:
                    held.end()

    def test_elsewhere_or_without_a_process_it_is_a_no_op(self):
        from serve.winjob import contain
        self.assertFalse(contain(None))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(os.name == "nt", "power throttling is a Windows setting")
class TestNoThrottle(unittest.TestCase):
    def test_a_contained_child_is_opted_out_of_power_throttling(self):
        # #691: contain() also sets ProcessPowerThrottling (EcoQoS off), so a minimized server window does not move the
        # engine to the E-cores
        from serve import winjob
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(winjob.no_throttle(int(child._handle)))
            self.assertTrue(winjob.contain(child))
        finally:
            child.kill()
            child.wait(10)
