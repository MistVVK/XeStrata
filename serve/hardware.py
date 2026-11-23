# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Standard-library hardware information shared by setup and the server."""
import ctypes
import os
import re
import struct
import sys
from pathlib import Path

UDEV_PCI_HWDB = ("/usr/lib/udev/hwdb.d/20-pci-vendor-model.hwdb",
                 "/usr/etc/udev/hwdb.d/20-pci-vendor-model.hwdb",
                 "/etc/udev/hwdb.d/20-pci-vendor-model.hwdb")


def linux_desktop(env=None) -> bool:
    env = os.environ if env is None else env
    return sys.platform.startswith("linux") and bool(env.get("WAYLAND_DISPLAY") or env.get("DISPLAY"))


def proc_text(name: str, *, encoding=None, errors=None) -> str:
    return Path("/proc", name).read_text(encoding=encoding, errors=errors)


def ram_gb() -> float:
    """The RAM this machine has, in GB; 0.0 when it cannot be read."""
    if sys.platform == "win32":
        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = _Status()
        m.dwLength = ctypes.sizeof(_Status)
        try:
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))   # type: ignore[attr-defined]
        except (AttributeError, OSError):
            return 0.0
        return m.ullTotalPhys / 2 ** 30
    try:
        for line in proc_text("meminfo").splitlines():
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024 / 2 ** 30
    except (OSError, IndexError, ValueError):
        pass
    return 0.0


def pci_product_name(vendor, device) -> str | None:
    """The card's product name in udev's PCI hardware database, or None.  A line like
    `ID_MODEL_FROM_DATABASE=Battlemage G31 [Arc Pro B70]` names the die before the brackets and the board in them:
    this returns "Arc Pro B70", not the die.  A bracket that holds no product ("Intel Graphics") is not a name.
    The IDs come from the card, so no maker or model is picked here."""
    v = int(vendor, 16) if isinstance(vendor, str) else int(vendor)
    d = int(device, 16) if isinstance(device, str) else int(device)
    key = "pci:v%08xd%08x" % (v, d)     # udev writes both IDs in eight hexadecimal digits
    for path in UDEV_PCI_HWDB:
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(lines):
            if not line.lower().startswith(key):
                continue
            for value in lines[i + 1:i + 5]:
                if value.strip() and value[:1] not in (" ", "\t"):
                    break                                # the next device's entry
                if not value.strip().startswith("ID_MODEL_FROM_DATABASE="):
                    continue
                name = value.strip().split("=", 1)[1].strip()
                found = re.search(r"\[([^]]+)]\s*$", name)
                if not found:
                    return None
                product = found.group(1).strip()
                return None if product.lower().startswith(("intel graphics", "graphics", "gpu")) else product
    return None


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
