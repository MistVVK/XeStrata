"""Windows file-lock and process-identity primitives for the managed runtime (ctypes; no pywin32).

POSIX gets these from ``fcntl.flock`` and ``os.kill(pid, 0)``. Neither exists on Windows, and ``os.kill`` there is
``TerminateProcess``: calling it with signal 0 would END the process, so a liveness check must never be written that way.

* ``try_lock`` / ``unlock``: a byte-range lock (``LockFileEx``) on one byte far beyond the data, so the file stays
  readable by other processes while a lock is held (a lock that covered the data would make reads fail). It supports
  shared locks, which ``msvcrt.locking`` does not, and is released by the OS when the process ends, however it ends.
* ``process_started`` / ``pid_alive``: whether a process exists and when it was created. A PID recorded in a state
  file is only a hint; the creation time is what tells the process that was recorded from a later one that reused
  the number. When the process cannot be opened, a creation time from before the last boot still decides: no process
  survives a restart (``created_before_boot``).
"""

from __future__ import annotations

import os

WINDOWS = os.name == "nt"

#: the locked byte. Beyond any data file the runtime writes (they are all far smaller).
LOCK_OFFSET = 1 << 16

#: clock slack when a recorded creation time is compared with the boot time: one minute, in 100 ns ticks.
_BOOT_SLACK = 60 * 10_000_000


def created_before_boot(started: int | None, boot: int | None) -> bool:
    """Whether a recorded creation time (100 ns ticks since 1601) is from before the last boot (same unit). No process
    survives a restart, so a record like that names a process that is gone, whatever holds its number now. False when
    either time is unknown, and within a minute of the boot time, so a clock adjustment never clears a live record."""
    if type(started) is not int or type(boot) is not int or started <= 0 or boot <= 0:
        return False
    return started < boot - _BOOT_SLACK

if WINDOWS:
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _Overlapped(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_void_p), ("InternalHigh", ctypes.c_void_p),
                    ("Offset", wintypes.DWORD), ("OffsetHigh", wintypes.DWORD), ("hEvent", wintypes.HANDLE)]

    _k32.LockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                ctypes.POINTER(_Overlapped)]
    _k32.LockFileEx.restype = wintypes.BOOL
    _k32.UnlockFileEx.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD,
                                  ctypes.POINTER(_Overlapped)]
    _k32.UnlockFileEx.restype = wintypes.BOOL
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]
    _k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    _k32.GetExitCodeProcess.restype = wintypes.BOOL
    _k32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    _k32.GetProcessTimes.restype = wintypes.BOOL

    _k32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                ctypes.POINTER(wintypes.DWORD)]
    _k32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    _k32.GetSystemTimeAsFileTime.argtypes = [ctypes.POINTER(wintypes.FILETIME)]
    _k32.GetSystemTimeAsFileTime.restype = None
    _k32.GetTickCount64.argtypes = []
    _k32.GetTickCount64.restype = ctypes.c_ulonglong

    _LOCKFILE_FAIL_IMMEDIATELY = 0x1
    _LOCKFILE_EXCLUSIVE_LOCK = 0x2
    _ERROR_LOCK_VIOLATION = 33
    _ERROR_IO_PENDING = 997
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259
    _ERROR_ACCESS_DENIED = 5

    def _overlapped() -> _Overlapped:
        o = _Overlapped()
        o.Offset = LOCK_OFFSET & 0xFFFFFFFF
        o.OffsetHigh = LOCK_OFFSET >> 32
        return o

    def try_lock(fd: int, *, exclusive: bool) -> bool:
        """Take the lock without waiting. False when another holder has it."""
        handle = msvcrt.get_osfhandle(fd)
        flags = _LOCKFILE_FAIL_IMMEDIATELY | (_LOCKFILE_EXCLUSIVE_LOCK if exclusive else 0)
        if _k32.LockFileEx(handle, flags, 0, 1, 0, ctypes.byref(_overlapped())):
            return True
        code = ctypes.get_last_error()
        if code in (_ERROR_LOCK_VIOLATION, _ERROR_IO_PENDING):
            return False
        raise ctypes.WinError(code, "LockFileEx")

    def unlock(fd: int) -> None:
        _k32.UnlockFileEx(msvcrt.get_osfhandle(fd), 0, 1, 0, ctypes.byref(_overlapped()))

    def process_started(pid: int) -> int | None:
        """The creation time (100 ns ticks since 1601) of a running process, or None when there is no such
        process, it has exited, or its times cannot be read."""
        if type(pid) is not int or pid <= 0:
            return None
        handle = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            code = wintypes.DWORD()
            if not _k32.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != _STILL_ACTIVE:
                return None
            created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
            if not _k32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                        ctypes.byref(kernel), ctypes.byref(user)):
                return None
            return (created.dwHighDateTime << 32) | created.dwLowDateTime
        finally:
            _k32.CloseHandle(handle)

    def process_image(pid: int) -> str | None:
        """The executable path of a running process, or None when it cannot be read."""
        if type(pid) is not int or pid <= 0:
            return None
        handle = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if not _k32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return None
            return buffer.value
        finally:
            _k32.CloseHandle(handle)

    def boot_time() -> int | None:
        """When the system last started (100 ns ticks since 1601): the current time less the milliseconds since boot,
        which include time asleep. None when it cannot be read."""
        now = wintypes.FILETIME()
        _k32.GetSystemTimeAsFileTime(ctypes.byref(now))
        current = (now.dwHighDateTime << 32) | now.dwLowDateTime
        uptime = _k32.GetTickCount64()
        if current <= 0 or uptime <= 0:
            return None
        return current - uptime * 10_000

    def pid_alive(pid: int, started: int | None = None) -> bool:
        """Whether `pid` is a running process, and, when `started` is given, the one created at that time."""
        if type(pid) is not int or pid <= 0:
            return False
        now = process_started(pid)
        if now is None:
            # OpenProcess can be refused for a protected process that does exist; treat that as present.
            handle = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                _k32.CloseHandle(handle)
                return False                          # opened, but exited or unreadable: not running
            if ctypes.get_last_error() != _ERROR_ACCESS_DENIED:
                return False
            # Refused, so its creation time cannot be compared. After a restart the number often belongs to a system
            # process this user may not open; a record created before the boot cannot be that process (none survives
            # a restart), so it is gone. Without a recorded time, or with one after the boot, keep treating it as present.
            return not created_before_boot(started, boot_time())
        return started is None or now == started
else:
    def try_lock(fd: int, *, exclusive: bool) -> bool:
        raise OSError("Windows only")

    def unlock(fd: int) -> None:
        raise OSError("Windows only")

    def process_started(pid: int) -> int | None:
        return None

    def process_image(pid: int) -> str | None:
        return None

    def boot_time() -> int | None:
        return None

    def pid_alive(pid: int, started: int | None = None) -> bool:
        raise OSError("Windows only")
