"""The OS and machine facts Control needs, answered without asking WMI on Windows (#4975, #5185).

`platform.system()`, `.machine()`, `.release()`, `.version()`, `.processor()` and `.node()` all go through
`platform.uname()`. On Windows with Python 3.12 that calls `win32_ver()` and `_get_machine_win32()`, which query WMI
(`platform._wmi_query`) with no timeout: a stuck `winmgmt` service froze every Control command that resolves the engine
component ("Strata Control did not answer within 60 s (models list)"), although the interpreter, the environment and the
registry know the same facts. On Windows these functions read only those. Everywhere else they are the `platform` calls
(an `os.uname()` there, not a service), looked up at call time so a test can replace them.

Pure and tiny on purpose, like targets.py: the start path imports it.
"""

from __future__ import annotations

import os
import socket
import sys
import platform

_WINDOWS_MACHINES = {"amd64": "AMD64", "x86_64": "AMD64", "arm64": "ARM64", "aarch64": "ARM64", "x86": "x86", "i386": "x86",
                     "i686": "x86", "ia64": "ia64"}


def is_windows() -> bool:
    return sys.platform.startswith("win")


def system() -> str:
    """`platform.system()`: "Windows", "Linux", "Darwin" ..."""
    return "Windows" if is_windows() else platform.system()


def _windows_machine() -> str:
    """Windows names its architecture in the environment the loader gives every process: PROCESSOR_ARCHITEW6432 is the real
    one inside a 32-bit process on 64-bit Windows, PROCESSOR_ARCHITECTURE the process's own (which is also what an x64
    program emulated on ARM64 sees: the engine that fits is the x64 one). `platform.uname()` itself read these first on
    Python 3.11 and before. Without them (a stripped environment) the interpreter's own build tag answers."""
    for key in ("PROCESSOR_ARCHITEW6432", "PROCESSOR_ARCHITECTURE"):
        value = os.environ.get(key, "").strip()
        if value:
            return _WINDOWS_MACHINES.get(value.lower(), value)
    try:
        import sysconfig
        tag = sysconfig.get_platform().lower()
    except Exception:                                    # noqa: BLE001 - a fact lookup never fails a start
        tag = ""
    if "arm64" in tag:
        return "ARM64"
    if "amd64" in tag or "x86_64" in tag:
        return "AMD64"
    return "x86" if tag in ("win32", "") else tag


def machine() -> str:
    """`platform.machine()`: "AMD64", "ARM64" on Windows; "x86_64", "arm64", "aarch64" elsewhere."""
    return _windows_machine() if is_windows() else platform.machine()


def release() -> str:
    """`platform.release()`: "10" or "11" on Windows (11 from build 22000), the kernel release elsewhere."""
    if is_windows():
        info = _windows_version()
        if info is None:
            return ""
        return "11" if info[0] == 10 and info[2] >= 22000 else str(info[0])
    return platform.release()


def version() -> str:
    """`platform.version()`: "10.0.19045" on Windows, the kernel version elsewhere."""
    if is_windows():
        info = _windows_version()
        return "" if info is None else f"{info[0]}.{info[1]}.{info[2]}"
    return platform.version()


def _windows_version() -> tuple[int, int, int] | None:
    try:
        info = sys.getwindowsversion()                   # type: ignore[attr-defined]
        return int(info.major), int(info.minor), int(info.build)
    except Exception:                                    # noqa: BLE001
        return None


def processor() -> str:
    """`platform.processor()`: the CPU's description (PROCESSOR_IDENTIFIER on Windows; the empty string when unknown)."""
    if is_windows():
        return os.environ.get("PROCESSOR_IDENTIFIER", "").strip()
    return platform.processor()


def node() -> str:
    """`platform.node()`: the computer's name."""
    if is_windows():
        try:
            return socket.gethostname()
        except OSError:
            return os.environ.get("COMPUTERNAME", "")
    return platform.node()


def platform_string() -> str:
    """A one-line description of the OS for reports (`platform.platform()` asks the same questions as `uname`)."""
    if is_windows():
        return f"Windows-{release()}-{version()}"
    return platform.platform()
