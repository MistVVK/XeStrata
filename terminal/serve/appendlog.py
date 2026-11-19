"""#5175: a log file that several child processes write to through handles of their own.

On Windows every handle keeps its own file position, and a child process writes at the position of the handle it was given.
Python's "a" mode only moves to the end of the file for writes made by Python itself.  So the image encoder's handle, opened
when the server started, put its lines at the position the log had then, over the engine's start-up lines, and the engine's
next lines could land over the encoder's in turn.

`open_append` opens the file with FILE_APPEND_DATA and without FILE_WRITE_DATA: Windows then ignores the position of every
write made through the handle (a child's too) and puts it at the end of the file.  Everywhere else it is the ordinary
`open(path, "a")`, which already has O_APPEND semantics for children.
"""
from __future__ import annotations

import os


def open_append(path, encoding: str = "utf-8"):
    """A text file open for appending whose every write, also a child's through the inherited handle, lands at the end."""
    if os.name != "nt":
        return open(path, "a", encoding=encoding)
    try:
        return _open_append_windows(os.fspath(path), encoding)
    except OSError:
        return open(path, "a", encoding=encoding)          # the old behaviour: never fail a start over the log's open mode


def _open_append_windows(path: str, encoding: str):
    import ctypes
    import msvcrt
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                     wintypes.DWORD, wintypes.HANDLE]
    FILE_APPEND_DATA, FILE_READ_ATTRIBUTES, SYNCHRONIZE = 0x0004, 0x0080, 0x00100000
    SHARE_ALL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL = 0x7, 4, 0x80
    handle = kernel32.CreateFileW(path, FILE_APPEND_DATA | FILE_READ_ATTRIBUTES | SYNCHRONIZE, SHARE_ALL, None,
                                  OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, None)
    if handle in (None, wintypes.HANDLE(-1).value):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_APPEND | os.O_WRONLY)
    except OSError:
        kernel32.CloseHandle(handle)
        raise
    return os.fdopen(fd, "a", encoding=encoding)
