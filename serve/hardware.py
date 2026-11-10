# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Standard-library hardware information shared by setup and the server."""
import ctypes
import os
import struct
import sys
from pathlib import Path


def linux_desktop(env=None) -> bool:
    env = os.environ if env is None else env
    return sys.platform.startswith("linux") and bool(env.get("WAYLAND_DISPLAY") or env.get("DISPLAY"))


def proc_text(name: str, *, encoding=None, errors=None) -> str:
    return Path("/proc", name).read_text(encoding=encoding, errors=errors)


def missing_devices(vendor: str, render: str | None) -> list:
    need = [render or ("/dev/nvidiactl" if vendor == "nvidia" else "a render node")]
    if vendor == "amd":
        need.append("/dev/kfd")
    return [path for path in need if not Path(path).exists()]


def xe_regions(fd: int, truncate: bool = False) -> list[tuple[int, int, int]]:
    """VRAM regions as (total, used, CPU-visible) bytes; OSError if the query fails."""
    import fcntl
    query = bytearray(struct.pack("=QIIQQQ", 0, 1, 0, 0, 0, 0))
    request = 0xC0286440
    fcntl.ioctl(fd, request, query)
    size = struct.unpack_from("=I", query, 12)[0]
    if size < 8:
        raise OSError("short xe memory regions query")
    buf = ctypes.create_string_buffer(size)
    struct.pack_into("=Q", query, 16, ctypes.addressof(buf))
    fcntl.ioctl(fd, request, query)
    count = struct.unpack_from("=I", buf, 0)[0]
    capacity = (size - 8) // 88
    if count > capacity and not truncate:
        raise OSError("short xe memory regions")
    count = min(count, capacity)
    out = []
    for i in range(count):
        cls, _, _, total, used, visible, _ = struct.unpack_from("=HHIQQQQ", buf, 8 + 88 * i)
        if cls == 1:
            out.append((total, used, visible))
    return out
