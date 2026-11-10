"""pytest, tools/: the tests here describe Linux and Windows PCs on x86 (NVIDIA, AMD and Intel cards; the Spark and the Mac tests say
so themselves by patching the platform or the CPU). On a Mac every one of them ran the Mac or ARM branch instead: setup.main() goes to
the Metal installer (`sys.platform == "darwin"`), strata_mcp takes the unified-memory path, and an Apple CPU reads as ARM64.

On macOS only, for each test:
* setup.py and tools/strata_mcp.py see sys.platform "linux" (their module-level `sys` is a view of the real one with another
  platform; the rest of the process, and the tests' own skip decisions, still see darwin);
* the CPU reads as x86_64 (platform.machine and strata_control.sysinfo.machine), as on GitHub's Linux runner.
Linux and Windows are not touched. A test that wants darwin or arm64 patches it on top, as before (setup.sys.platform, is_arm64)."""
import platform
import sys
from contextlib import ExitStack
from unittest import mock

import pytest

LINUX_VIEW = ("setup", "strata_mcp")


class _SysView:
    """The real sys module with its own `platform`; every other read and write goes to sys itself."""

    def __init__(self, real, platform_name):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "platform", platform_name)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def __setattr__(self, name, value):
        if name == "platform":
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_real"), name, value)

    def __delattr__(self, name):
        if name == "platform":
            object.__setattr__(self, name, "linux")
        else:
            delattr(object.__getattribute__(self, "_real"), name)


@pytest.fixture(autouse=True)
def _a_linux_x86_pc_on_macos():
    if sys.platform != "darwin":
        yield
        return
    from strata_control import sysinfo
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(platform, "machine", lambda: "x86_64"))
        stack.enter_context(mock.patch.object(sysinfo, "machine", lambda: "x86_64"))
        for name in LINUX_VIEW:
            mod = sys.modules.get(name)
            if mod is not None and getattr(mod, "sys", None) is sys:
                stack.enter_context(mock.patch.object(mod, "sys", _SysView(sys, "linux")))
        yield
