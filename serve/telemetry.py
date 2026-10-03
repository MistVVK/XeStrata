# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: the Intel GPU on the xe driver, from Linux (temperature and power from hwmon, the PCIe link, the load of this
  user's processes from their DRM fdinfo) and its VRAM from Level Zero's sysman through ctypes, so no pip package is
  needed.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  Linux /proc and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import glob
import os
import platform
import sys
import threading
import time

HISTORY = 60


# ------------------------------------------------------------------------------------------------ the Intel GPU
class _ZesPciAddress(ctypes.Structure):
    _fields_ = [("domain", ctypes.c_uint32), ("bus", ctypes.c_uint32), ("device", ctypes.c_uint32),
                ("function", ctypes.c_uint32)]


class _ZesPciSpeed(ctypes.Structure):
    _fields_ = [("gen", ctypes.c_int32), ("width", ctypes.c_int32), ("max_bandwidth", ctypes.c_int64)]


class _ZesPciProps(ctypes.Structure):    # zes_pci_properties_t
    _fields_ = [("stype", ctypes.c_uint32), ("pnext", ctypes.c_void_p), ("address", _ZesPciAddress),
                ("max_speed", _ZesPciSpeed), ("bw", ctypes.c_uint8), ("pkt", ctypes.c_uint8),
                ("replay", ctypes.c_uint8)]


class _ZesMemState(ctypes.Structure):    # zes_mem_state_t
    _fields_ = [("stype", ctypes.c_uint32), ("pnext", ctypes.c_void_p), ("health", ctypes.c_int32),
                ("free", ctypes.c_uint64), ("size", ctypes.c_uint64)]


class _Zes:
    """The device's VRAM through Level Zero's management API (sysman, libze_loader.so.1, which the GPU runtime
    installs) through ctypes: the reading Linux's sysfs does not give for the xe driver."""

    STYPE_PCI_PROPERTIES, STYPE_MEM_STATE = 0x2, 0x1E

    def __init__(self, pci):
        self.mems = []
        try:
            self.lib = ctypes.CDLL("libze_loader.so.1")
            if self.lib.zesInit(ctypes.c_uint32(0)) != 0:
                return
            n = ctypes.c_uint32(0)
            if self.lib.zesDriverGet(ctypes.byref(n), None) != 0 or n.value == 0:
                return
            drivers = (ctypes.c_void_p * n.value)()
            self.lib.zesDriverGet(ctypes.byref(n), drivers)
            for drv in drivers:
                m = ctypes.c_uint32(0)
                if self.lib.zesDeviceGet(ctypes.c_void_p(drv), ctypes.byref(m), None) != 0:
                    continue
                devs = (ctypes.c_void_p * m.value)()
                self.lib.zesDeviceGet(ctypes.c_void_p(drv), ctypes.byref(m), devs)
                for dev in devs:
                    props = _ZesPciProps(stype=self.STYPE_PCI_PROPERTIES)
                    if self.lib.zesDevicePciGetProperties(ctypes.c_void_p(dev), ctypes.byref(props)) != 0:
                        continue
                    a = props.address
                    if f"{a.domain:04x}:{a.bus:02x}:{a.device:02x}.{a.function:x}" != pci:
                        continue
                    k = ctypes.c_uint32(0)
                    self.lib.zesDeviceEnumMemoryModules(ctypes.c_void_p(dev), ctypes.byref(k), None)
                    mods = (ctypes.c_void_p * k.value)()
                    self.lib.zesDeviceEnumMemoryModules(ctypes.c_void_p(dev), ctypes.byref(k), mods)
                    self.mems = list(mods)
        except (OSError, AttributeError):
            self.mems = []

    def read(self):
        """(used, total) bytes over the device's memory modules, or (None, None)."""
        free = total = 0
        for mod in self.mems:
            st = _ZesMemState(stype=self.STYPE_MEM_STATE)
            try:
                if self.lib.zesMemoryGetState(ctypes.c_void_p(mod), ctypes.byref(st)) != 0:
                    return None, None
            except (OSError, AttributeError):
                return None, None
            free += st.free
            total += st.size
        return (total - free, total) if total else (None, None)


def _read(path, conv=int):
    try:
        with open(path, encoding="ascii") as f:
            return conv(f.read().strip())
    except (OSError, ValueError):
        return None


class _XeGpu:
    """An Intel GPU on the xe driver: temperature and power from Linux's sysfs (hwmon), the PCIe link from the PCI
    path, the load from the DRM clients' fdinfo, the VRAM through sysman.  `pci` is the card's address as the config
    gives it (setup's gpu_pci); without one, the first card on the xe driver."""
    LINK_GEN = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}

    def __init__(self, pci=None):
        self.dev = None
        for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
            dev = os.path.realpath(os.path.join(card, "device"))
            if os.path.basename(os.path.realpath(os.path.join(dev, "driver"))) != "xe":
                continue
            if pci is None or os.path.basename(dev) == pci:
                self.dev = dev
                break
        if self.dev is None:
            return
        self.pci = os.path.basename(self.dev)
        hw = glob.glob(os.path.join(self.dev, "hwmon", "hwmon*"))
        self.hwmon = hw[0] if hw else None
        self.zes = _Zes(self.pci)
        self._prev = None   # (time, energy uJ)
        self._cycles = {}   # DRM client id -> (engine cycles, total cycles)

    def ok(self):
        return self.dev is not None

    def name(self):
        if not self.ok():
            return None
        return f"Intel GPU {_read(os.path.join(self.dev, 'device'), str) or ''} at {self.pci}".replace("0x", "")

    def _link(self):
        """The card's widest link up its PCI path: a card with a switch of its own reports x1 on its own port."""
        best, p = None, self.dev
        while p.startswith("/sys/devices/pci"):
            speed = _read(os.path.join(p, "current_link_speed"), str)
            width = _read(os.path.join(p, "current_link_width"))
            top = _read(os.path.join(p, "max_link_speed"), str)
            if speed and width:
                g = self.LINK_GEN.get(speed.split()[0])
                gmax = self.LINK_GEN.get(top.split()[0]) if top else None
                if g and (best is None or (g * width) > best[0] * best[1]):
                    best = (g, width, gmax)
            p = os.path.dirname(p)
        return best or (None, None, None)

    def _busy(self):
        """The render and compute engines' load from this user's processes on the card: the xe driver counts each
        DRM client's engine cycles in /proc/<pid>/fdinfo (the card-wide load needs root on this driver)."""
        now = {}
        for f in glob.glob("/proc/[0-9]*/fdinfo/*"):
            try:
                with open(f, encoding="ascii", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if "drm-driver:\txe" not in text or f"drm-pdev:\t{self.pci}" not in text:
                continue
            kv = {}
            for line in text.splitlines():
                k, _, v = line.partition(":")
                kv.setdefault(k, v.strip())
            cid = kv.get("drm-client-id")
            try:
                busy = sum(int(kv.get(f"drm-cycles-{e}", "0")) for e in ("rcs", "ccs"))
                total = int(kv.get("drm-total-cycles-ccs") or kv.get("drm-total-cycles-rcs") or 0)
            except ValueError:
                continue
            if cid and total:
                now[cid] = (busy, total)
        prev, self._cycles = self._cycles, now
        spans = [now[c][1] - prev[c][1] for c in now if c in prev and now[c][1] > prev[c][1]]
        if not spans:
            return 0.0 if not now else None
        used = sum(max(0, now[c][0] - prev[c][0]) for c in now if c in prev)
        return max(0.0, min(100.0, 100.0 * used / max(spans)))

    def read(self):
        out = {}
        t = time.time()
        energy = _read(os.path.join(self.hwmon, "energy1_input")) if self.hwmon else None
        prev, self._prev = self._prev, (t, energy)
        out["util"] = self._busy()
        if prev and t > prev[0] and energy is not None and prev[1] is not None and energy >= prev[1]:
            out["power"] = (energy - prev[1]) / 1e6 / (t - prev[0])
        if self.hwmon:
            temps = [_read(f) for f in glob.glob(os.path.join(self.hwmon, "temp*_input"))]
            temps = [v / 1000.0 for v in temps if v]
            out["temp"] = max(temps) if temps else None
            cap = _read(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap else None
        out["mem_used"], out["mem_total"] = self.zes.read()
        out["pcie_gen"], out["pcie_width"], out["pcie_gen_max"] = self._link()
        out["pcie_rx_mb"] = out["pcie_tx_mb"] = None   # no counters for the link's traffic
        return out


def free_vram_mib(gpu_pci=None):
    """Free VRAM of the card (by PCI address, None: the first on the xe driver) in MiB, or None when it cannot be
    read."""
    g = _XeGpu(gpu_pci)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_name():
    if os.path.exists("/proc/cpuinfo"):
        for line in open("/proc/cpuinfo", encoding="utf-8", errors="replace"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        try:
            f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
            return f[3] + f[4], sum(f)
        except (OSError, ValueError):
            return None

    def cpu(self):
        cur = self._times()
        prev, self.prev = self.prev, cur
        if not cur or not prev or cur[1] == prev[1]:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - (cur[0] - prev[0]) / (cur[1] - prev[1]))))

    @staticmethod
    def ram():
        try:
            info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    def __init__(self, extra=None, gpu_pci=None):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_pci`: the PCI address
        of the card the engine runs on (setup's gpu_pci), or None for the first card on the xe driver."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        self.gpus = [(0, _XeGpu(gpu_pci))]
        self.gpus = [(i, g) for i, g in self.gpus if g.ok()] or self.gpus[:1]
        self.gpu = self.gpus[0][1]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        self.static = {
            "gpu_name": " + ".join(g.name() or "?" for _, g in self.gpus) if self.gpu.ok() else None,
            "gpu_count": len(self.gpus),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
        self._disk_prev = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _disk(self):
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def sample(self):
        s = {}
        if self.gpu.ok():
            reads = [(i, g.read()) for i, g in self.gpus]
            g = dict(reads[0][1])
            if len(reads) > 1:
                def vals(k):
                    return [r[k] for _, r in reads if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                u = vals("util")
                g["util"] = sum(u) / len(u) if u else None
                t = vals("temp")
                g["temp"] = max(t) if t else None
                s["gpus"] = [{"index": i, "util": r.get("util"), "mem_used": r.get("mem_used"),
                              "mem_total": r.get("mem_total"), "temp": r.get("temp"), "power": r.get("power")}
                             for i, r in reads]
            s.update({f"gpu_{k}": v for k, v in g.items()})
        if self.ps:
            try:
                s["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                s["ram_used"], s["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        else:
            s["cpu"] = self.fallback.cpu()
            s["ram_used"], s["ram_total"] = self.fallback.ram()
        s["disk_read_mb"], s["disk_write_mb"] = self._disk()
        if self.extra:
            try:
                s.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return s

    def _loop(self):
        while True:
            s = self.sample()
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s", "prefill_tok_s_mean"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            time.sleep(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
