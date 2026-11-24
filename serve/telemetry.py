# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: the Intel GPU on the xe or i915 driver, from Linux (temperature and power from hwmon, the PCIe link, the load
  of this user's processes from their DRM fdinfo) and its VRAM from Level Zero's sysman through ctypes, so no pip
  package is needed.  An NVIDIA GPU (the contrib build): NVIDIA's NVML library (libnvidia-ml.so.1, nvml.dll on
  Windows, installed with the driver) through ctypes: load, VRAM, temperature, power, PCIe link and throughput.
  Windows has no sysfs: any other GPU there is read as Task Manager reads it, load and VRAM in use from the
  performance counters, name, address, VRAM and temperature from the kernel's graphics interface (_WinGpu).
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  Linux /proc and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import glob
import functools
import os
import platform
import re
import sys
import threading
import time

from serve import hardware

HISTORY = 60
WIN = sys.platform == "win32"


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
    installs) through ctypes: the reading Linux's sysfs does not give for the xe and i915 drivers."""

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


def _xe_vram(pci):
    """(used, total) VRAM bytes, or (None, None) when unavailable or its usage is hidden."""
    for node in sorted(glob.glob(f"/sys/bus/pci/devices/{pci}/drm/renderD*")):
        try:
            fd = os.open("/dev/dri/" + os.path.basename(node), os.O_RDWR | os.O_CLOEXEC)
        except OSError:
            continue
        try:
            regions = hardware.xe_regions(fd, truncate=True)
            total, used = sum(r[0] for r in regions), sum(r[1] for r in regions)
            return (used, total) if total and used else (None, None)
        except (OSError, ValueError):
            return None, None
        finally:
            os.close(fd)
    return None, None


def _read(path, conv=int):
    try:
        with open(path, encoding="ascii") as f:
            return conv(f.read().strip())
    except (OSError, ValueError):
        return None


PCI_IDS = ("/usr/share/hwdata/pci.ids", "/usr/share/misc/pci.ids")


def _pci_ids_name(dev):
    """The card's name in the system's PCI ID list ("Battlemage G31 [Intel Graphics]", "Navi 44 [Radeon RX 9050 /
    9060 XT]") from its sysfs folder's vendor and device IDs, or None: neither the xe driver nor sysman names a card
    the driver does not know by its product name, and amdgpu gives product_name for few cards."""
    vendor = _read(os.path.join(dev or "", "vendor"), str)
    device = _read(os.path.join(dev or "", "device"), str)
    if not vendor or not device:
        return None
    vendor, device = vendor.lower().replace("0x", ""), device.lower().replace("0x", "")
    for path in PCI_IDS:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                in_vendor = False
                for line in f:
                    if line.startswith("#") or not line.strip():   # comments sit inside a vendor's list too
                        continue
                    if not line.startswith("\t"):
                        in_vendor = line.startswith(vendor + " ")
                    elif in_vendor and line.startswith(f"\t{device} "):
                        return line.strip()[len(device):].strip() or None
        except OSError:
            continue
    return None


AMDGPU_IDS = ("/usr/share/libdrm/amdgpu.ids", "/opt/amdgpu/share/libdrm/amdgpu.ids",
              "/opt/rocm/lib/rocm_sysdeps/share/libdrm/amdgpu.ids", "/opt/rocm/share/libdrm/amdgpu.ids")


def _amdgpu_ids_name(dev):
    """The card's marketing name in libdrm's amdgpu.ids ("AMD Radeon AI PRO R9700") from its sysfs folder's device
    ID and revision, or None: the list ROCm's runtime names a card by, where amdgpu gives no product_name (the board
    has no FRU EEPROM) and the system has no PCI ID list (a container)."""
    device = _read(os.path.join(dev or "", "device"), str)
    revision = _read(os.path.join(dev or "", "revision"), str)
    if not device or not revision:
        return None
    key = (device.upper().replace("0X", ""), revision.upper().replace("0X", "").zfill(2))
    for path in AMDGPU_IDS:
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    fields = [x.strip() for x in line.split(",", 2)]
                    if len(fields) == 3 and (fields[0].upper(), fields[1].upper()) == key and fields[2]:
                        return fields[2]
        except OSError:
            continue
    return None


def _gen(text):
    """The PCIe generation of a link-speed file's contents ("16.0 GT/s PCIe" -> 4), or None."""
    try:
        return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}[float(str(text).split()[0])]
    except (ValueError, TypeError, IndexError, KeyError):
        return None


def _bdf(s):
    """True for a sysfs pci device name ("0000:03:00.0")."""
    return (len(s) == 12 and s[4] == ":" and s[7] == ":" and s[10] == "." and s[11] in "01234567"
            and all(c in "0123456789abcdef" for c in s[0:4] + s[5:7] + s[8:10]))


def _pci_hops(dev):
    """The sysfs folders of the PCI devices from the root port down to the card `dev`."""
    hops, prefix = [], []
    for part in os.path.realpath(dev or "").split("/"):
        prefix.append(part)
        if _bdf(part):
            hops.append("/".join(prefix))
    return hops


def _path_link(dev, hops=None):
    """The PCIe link a card actually gets (upstream 7f3c0da2): the narrowest and slowest hop between the root port and
    the card, from each hop's max_link_speed / max_link_width (a card that is Gen4 on its own hop behind a Gen3 root
    port runs at Gen3).  pcie_own_gen keeps the card's own hop aside.  The bottleneck is claimed only when every hop
    could be read (pcie_path: how many were, "4/4"); else the card's own negotiated link.  `hops`: _pci_hops(dev)."""
    hops = _pci_hops(dev) if hops is None else hops
    gens, widths, seen = [], [], 0
    for d in hops:
        g = _gen(_read(os.path.join(d, "max_link_speed"), str))
        w = _read(os.path.join(d, "max_link_width"))
        seen += 1 if (g is not None or w is not None) else 0
        if g:
            gens.append(g)
        if w:
            widths.append(w)
    own_gen = _gen(_read(os.path.join(dev or "", "current_link_speed"), str))
    whole = bool(hops) and seen == len(hops)
    gen = min(gens) if (whole and gens) else own_gen
    width = min(widths) if (whole and widths) else _read(os.path.join(dev or "", "current_link_width"))
    return {"pcie_gen": gen, "pcie_gen_max": gen, "pcie_own_gen": own_gen, "pcie_width": width,
            "pcie_path": "%d/%d" % (seen, len(hops))}


class _XeGpu:
    """An Intel GPU on the xe or i915 driver: temperature and power from Linux's sysfs (hwmon), the PCIe link from the
    PCI path, the load from the DRM clients' fdinfo, the VRAM through sysman.  `pci` is the card's address as the
    config gives it (setup's gpu_pci); without one, the first card on one of `drivers`."""
    LINK_GEN = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}

    def __init__(self, pci=None, drivers=("xe", "i915")):
        self.dev = None
        for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
            dev = os.path.realpath(os.path.join(card, "device"))
            driver = os.path.basename(os.path.realpath(os.path.join(dev, "driver")))
            if driver not in drivers:
                continue
            if pci is None or os.path.basename(dev) == pci:
                self.dev, self.driver = dev, driver
                break
        if self.dev is None:
            return
        self.pci = os.path.basename(self.dev)
        hw = glob.glob(os.path.join(self.dev, "hwmon", "hwmon*"))
        self.hwmon = hw[0] if hw else None
        self.zes = _Zes(self.pci)
        self._prev = None   # (time, energy uJ)
        self._cycles = {}   # DRM client id -> (engine cycles, total cycles); i915: (engine ns, sample time ns)

    def ok(self):
        return self.dev is not None

    def name(self):
        if not self.ok():
            return None
        device = (_read(os.path.join(self.dev, 'device'), str) or '').replace('0x', '')
        vendor = _read(os.path.join(self.dev, 'vendor'), str)
        name = (hardware.pci_product_name(vendor, device) if vendor and device else None) or \
            _pci_ids_name(self.dev) or f"GPU {device}"
        return f"Intel {name} at {self.pci}"

    def _link(self):
        """The card's widest link up its PCI path: a card with a switch of its own reports x1 on its own port.  Its
        generation now, its width, and the highest generation the path above it allows: the lowest max_link_speed from
        that link up to the root port (the B70's own switch can do Gen5 behind a Gen4 root port)."""
        best, tops, p = None, [], self.dev
        while p.startswith("/sys/devices/pci"):
            speed = _read(os.path.join(p, "current_link_speed"), str)
            width = _read(os.path.join(p, "current_link_width"))
            top = _read(os.path.join(p, "max_link_speed"), str)
            tops.append(self.LINK_GEN.get(top.split()[0]) if top else None)
            if speed and width:
                g = self.LINK_GEN.get(speed.split()[0])
                if g and (best is None or (g * width) > best[0] * best[1]):
                    best = (g, width, len(tops) - 1)
            p = os.path.dirname(p)
        if best is None:
            return None, None, None
        above = [t for t in tops[best[2]:] if t]
        return best[0], best[1], min(above) if above else None

    def _busy(self):
        """The render and compute engines' load from this user's processes on the card: the xe driver counts each
        DRM client's engine cycles in /proc/<pid>/fdinfo, i915 its engine time in ns (the card-wide load needs root
        on these drivers)."""
        now = {}
        t_ns = time.monotonic_ns()
        for f in glob.glob("/proc/[0-9]*/fdinfo/*"):
            try:
                with open(f, encoding="ascii", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if f"drm-driver:\t{self.driver}\n" not in text or f"drm-pdev:\t{self.pci}" not in text:
                continue
            kv = {}
            for line in text.splitlines():
                k, _, v = line.partition(":")
                kv.setdefault(k, v.strip())
            cid = kv.get("drm-client-id")
            try:
                if self.driver == "i915":
                    busy = sum(int(kv.get(f"drm-engine-{e}", "0 ns").split()[0]) for e in ("render", "compute"))
                    total = t_ns
                else:   # a class's cycles count all its engines (drm-engine-capacity-*, 1 when not given)
                    busy = sum(int(kv.get(f"drm-cycles-{e}", "0")) / int(kv.get(f"drm-engine-capacity-{e}", "1") or 1)
                               for e in ("rcs", "ccs"))
                    total = int(kv.get("drm-total-cycles-ccs") or kv.get("drm-total-cycles-rcs") or 0)
            except (ValueError, IndexError):
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
            # the GPU's own sensor (label "pkg") as the GPU temperature, as NVIDIA's core and AMD's edge readings
            # are; the VRAM's sensors (labels vram, vram_ch_N) apart.  A hwmon without labels: the hottest, as before.
            temps = {}
            for f in glob.glob(os.path.join(self.hwmon, "temp*_input")):
                v = _read(f)
                if v:
                    temps[_read(f.replace("_input", "_label"), str) or ""] = v / 1000.0
            vram = [v for k, v in temps.items() if k.startswith("vram")]
            out["temp"] = temps.get("pkg") if "pkg" in temps else (max(temps.values()) if temps else None)
            out["mem_temp"] = max(vram) if vram else None
            cap = _read(os.path.join(self.hwmon, "power1_cap")) or _read(os.path.join(self.hwmon, "power1_max"))
            out["power_limit"] = cap / 1e6 if cap else None
        # the xe driver's hwmon can keep a sleeping GT's last temperatures (a Battlemage card that has cooled shows the
        # load's; github.com/hugs7/intel-xe-battlemage-hwmon-fix): the web app marks the reading while nothing runs
        out["temp_may_be_stale"] = self.driver == "xe" and out["util"] is not None and out["util"] < 1.0
        out["mem_used"], out["mem_total"] = self.zes.read()
        if out["mem_total"] is None and self.driver == "xe":
            out["mem_used"], out["mem_total"] = _xe_vram(self.pci)
        out["pcie_gen"], out["pcie_width"], out["pcie_gen_max"] = self._link()
        out["pcie_rx_mb"] = out["pcie_tx_mb"] = None   # no counters for the link's traffic
        return out


# ------------------------------------------------------------------------------------------------ an AMD GPU
SYSFS = "/sys"


class _AmdGpu:
    """An AMD GPU on the amdgpu driver, from its sysfs files (no ROCm library needed; upstream #301): load
    (gpu_busy_percent), VRAM (mem_info_vram_used / _total), and from its hwmon folder the temperature (temp1_input, the
    edge sensor, m°C), power (power1_average or power1_input, µW) and its cap (power1_cap).  `pci` is the card's
    address as the config gives it; without one, the first card on the amdgpu driver."""

    def __init__(self, pci=None, sysfs=None):
        self.dev = None
        for card in sorted(glob.glob(os.path.join(sysfs or SYSFS, "class", "drm", "card[0-9]*"))):
            dev = os.path.realpath(os.path.join(card, "device"))
            if os.path.basename(os.path.realpath(os.path.join(dev, "driver"))) != "amdgpu":
                continue
            if pci is None or os.path.basename(dev) == pci:
                self.dev = dev
                break
        self.hwmon = None
        if self.dev:
            hw = sorted(glob.glob(os.path.join(self.dev, "hwmon", "hwmon*")))
            self.hwmon = hw[0] if hw else None

    def ok(self):
        return self.dev is not None

    def name(self):
        if not self.ok():
            return None
        listed = _pci_ids_name(self.dev)
        return (_read(os.path.join(self.dev, "product_name"), str) or _amdgpu_ids_name(self.dev)
                or (f"AMD {listed}" if listed else "AMD Radeon"))

    _gen = staticmethod(_gen)
    _bdf = staticmethod(_bdf)

    def _hops(self):
        return _pci_hops(self.dev)

    def link(self):
        return _path_link(self.dev, self._hops())

    def read(self):
        out = {"util": _read(os.path.join(self.dev, "gpu_busy_percent")),
               "mem_used": _read(os.path.join(self.dev, "mem_info_vram_used")),
               "mem_total": _read(os.path.join(self.dev, "mem_info_vram_total"))}
        if self.hwmon:
            t = _read(os.path.join(self.hwmon, "temp1_input"))
            out["temp"] = t / 1000.0 if t is not None else None
            for f in glob.glob(os.path.join(self.hwmon, "temp*_label")):   # the VRAM's sensor ("mem"), apart
                if _read(f, str) == "mem":
                    m = _read(f.replace("_label", "_input"))
                    out["mem_temp"] = m / 1000.0 if m is not None else None
            p = _read(os.path.join(self.hwmon, "power1_average"))
            if p is None:
                p = _read(os.path.join(self.hwmon, "power1_input"))
            out["power"] = p / 1e6 if p is not None else None
            cap = _read(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap is not None else None
        out.update(self.link())
        out["pcie_rx_mb"] = out["pcie_tx_mb"] = None   # no counters for the link's traffic
        return out


@functools.lru_cache(maxsize=16)
def _nv_process_device(pci: str):
    return _NvGpu(pci)


def nvml_process_memory_mib(pid: int, pci: str) -> int | None:
    """The process's device memory via NVML's compute list (gpu.cpp's process and MIG rules)."""
    class Process(ctypes.Structure):
        _fields_ = [("pid", ctypes.c_uint), ("used", ctypes.c_ulonglong),
                    ("gpu_instance", ctypes.c_uint), ("compute_instance", ctypes.c_uint)]

    gpu = _nv_process_device(pci)
    if not gpu.ok():
        return None
    try:
        fn = gpu.lib.nvmlDeviceGetComputeRunningProcesses_v3
        n = ctypes.c_uint(0)
        rc = fn(gpu.dev, ctypes.byref(n), None)
        if rc not in (0, 7) or not n.value:
            return None
        rows = (Process * n.value)()
        if fn(gpu.dev, ctypes.byref(n), rows) != 0:
            return None
        matches = [row.used for row in rows[:n.value] if row.pid == pid]
        if not matches or any(v == 0xffffffffffffffff for v in matches):
            return None
        return sum(matches) >> 20
    except (AttributeError, OSError):
        return None


# ------------------------------------------------------------------------------------------------ an NVIDIA GPU
class _NvGpu:
    """An NVIDIA GPU through NVML, with _XeGpu's interface.  `pci` is the card's address as the config gives it
    (setup's gpu_pci); without one, NVML's first card."""

    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, pci=None):
        self.lib = self.dev = None
        self.pci = pci
        for n in ("nvml.dll",) if WIN else ("libnvidia-ml.so.1", "libnvidia-ml.so"):
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            if self.lib.nvmlInit_v2() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            r = self.lib.nvmlDeviceGetHandleByPciBusId_v2(pci.encode(), ctypes.byref(h)) if pci else \
                self.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(0), ctypes.byref(h))
            if r != 0:
                self.lib = None
                return
            self.dev = h
        except (AttributeError, OSError):
            self.lib = None

    def ok(self):
        return self.lib is not None and self.dev is not None

    def _uint(self, fn, *args):
        v = ctypes.c_uint()
        try:
            return v.value if getattr(self.lib, fn)(self.dev, *args, ctypes.byref(v)) == 0 else None
        except (AttributeError, OSError):
            return None

    def name(self):
        if not self.ok():
            return None
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace") + (f" at {self.pci}" if self.pci else "")
        except (AttributeError, OSError):
            pass
        return None

    def read(self):
        out = {}
        u = self.Util()
        try:
            if self.lib.nvmlDeviceGetUtilizationRates(self.dev, ctypes.byref(u)) == 0:
                out["util"] = u.gpu
        except (AttributeError, OSError):
            pass
        m = self.Mem()
        try:
            if self.lib.nvmlDeviceGetMemoryInfo(self.dev, ctypes.byref(m)) == 0:
                out["mem_used"], out["mem_total"] = m.used, m.total
        except (AttributeError, OSError):
            pass
        out["temp"] = self._uint("nvmlDeviceGetTemperature", ctypes.c_uint(0))          # NVML_TEMPERATURE_GPU
        mw = self._uint("nvmlDeviceGetPowerUsage")
        out["power"] = mw / 1000.0 if mw is not None else None
        lim = self._uint("nvmlDeviceGetEnforcedPowerLimit")
        out["power_limit"] = lim / 1000.0 if lim is not None else None
        own = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")       # drops at idle (power saving)
        link = _path_link(os.path.join(SYSFS, "bus", "pci", "devices", self.pci)) if self.pci and not WIN else {}
        if link.get("pcie_gen"):   # the link the card gets, as for AMD: the same meaning in every maker's panel
            out.update(link)
            out["pcie_own_gen"] = own
        else:
            out["pcie_gen"] = own
            out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
            out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        rx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(1))                 # NVML_PCIE_UTIL_RX_BYTES, KB/s
        tx = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(0))
        out["pcie_rx_mb"] = rx / 1024.0 if rx is not None else None
        out["pcie_tx_mb"] = tx / 1024.0 if tx is not None else None
        if WIN and out.get("util") is None and self.pci:
            # a laptop's NVIDIA GPU at rest: NVML answers the load once, then NVML_ERROR_UNKNOWN (RTX 3050 Laptop,
            # driver 555.99); Windows' counters still have it
            if getattr(self, "win", None) is None:
                self.win = _WinGpu(self.pci)
            if self.win.ok():
                out["util"] = self.win.read()["util"]
        return out


# ------------------------------------------------------------------------------------------------ a GPU on Windows
class _Luid(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _KmtAdapterInfo(ctypes.Structure):          # D3DKMT_ADAPTERINFO
    _fields_ = [("hAdapter", ctypes.c_uint32), ("AdapterLuid", _Luid), ("NumOfSources", ctypes.c_uint32),
                ("bPrecisePresentRegionsPreferred", ctypes.c_int32)]


class _KmtEnumAdapters2(ctypes.Structure):        # D3DKMT_ENUMADAPTERS2
    _fields_ = [("NumAdapters", ctypes.c_uint32), ("pAdapters", ctypes.POINTER(_KmtAdapterInfo))]


class _KmtQuery(ctypes.Structure):                # D3DKMT_QUERYADAPTERINFO
    _fields_ = [("hAdapter", ctypes.c_uint32), ("Type", ctypes.c_int32), ("pPrivateDriverData", ctypes.c_void_p),
                ("PrivateDriverDataSize", ctypes.c_uint32)]


class _KmtAddress(ctypes.Structure):              # D3DKMT_ADAPTERADDRESS
    _fields_ = [("BusNumber", ctypes.c_uint32), ("DeviceNumber", ctypes.c_uint32), ("FunctionNumber", ctypes.c_uint32)]


class _KmtSegments(ctypes.Structure):             # D3DKMT_SEGMENTSIZEINFO
    _fields_ = [("DedicatedVideoMemorySize", ctypes.c_uint64), ("DedicatedSystemMemorySize", ctypes.c_uint64),
                ("SharedSystemMemorySize", ctypes.c_uint64)]


class _KmtRegistry(ctypes.Structure):             # D3DKMT_ADAPTERREGISTRYINFO
    _fields_ = [("AdapterString", ctypes.c_wchar * 260), ("BiosString", ctypes.c_wchar * 260),
                ("DacType", ctypes.c_wchar * 260), ("ChipType", ctypes.c_wchar * 260)]


class _KmtPerf(ctypes.Structure):                 # D3DKMT_ADAPTER_PERFDATA
    _fields_ = [("PhysicalAdapterIndex", ctypes.c_uint32), ("MemoryFrequency", ctypes.c_uint64),
                ("MaxMemoryFrequency", ctypes.c_uint64), ("MaxMemoryFrequencyOC", ctypes.c_uint64),
                ("MemoryBandwidth", ctypes.c_uint64), ("PCIEBandwidth", ctypes.c_uint64), ("FanRPM", ctypes.c_uint32),
                ("Power", ctypes.c_uint32), ("Temperature", ctypes.c_uint32), ("PowerStateOverride", ctypes.c_ubyte)]


# KMTQUERYADAPTERINFOTYPE (d3dkmthk.h)
KMT_SEGMENT_SIZE, KMT_ADDRESS, KMT_REGISTRY, KMT_PERF = 3, 6, 8, 62


def _kmt_query(gdi, handle, kind, out):
    q = _KmtQuery(handle, kind, ctypes.cast(ctypes.byref(out), ctypes.c_void_p), ctypes.sizeof(out))
    try:
        return gdi.D3DKMTQueryAdapterInfo(ctypes.byref(q)) == 0
    except (AttributeError, OSError):
        return False


def _kmt_adapters():
    """The GPUs Windows' graphics kernel lists that have a PCI address (not its software renderer), each
    {"handle", "luid" (high, low), "pci" ("0000:01:00.0", as the engine writes it), "name", "vram" (dedicated bytes)};
    [] where gdi32 does not have the call."""
    try:
        gdi = ctypes.WinDLL("gdi32")
        e = _KmtEnumAdapters2()
        if gdi.D3DKMTEnumAdapters2(ctypes.byref(e)) != 0 or not e.NumAdapters:
            return []
        arr = (_KmtAdapterInfo * e.NumAdapters)()
        e.pAdapters = arr
        if gdi.D3DKMTEnumAdapters2(ctypes.byref(e)) != 0:
            return []
    except (AttributeError, OSError):
        return []
    found: list[dict] = []
    for a in arr[:e.NumAdapters]:
        addr, seg, reg = _KmtAddress(), _KmtSegments(), _KmtRegistry()
        luid = (a.AdapterLuid.HighPart & 0xFFFFFFFF, a.AdapterLuid.LowPart)
        if not _kmt_query(gdi, a.hAdapter, KMT_ADDRESS, addr) or addr.BusNumber == 0xFFFFFFFF or \
                any(f["luid"] == luid for f in found):     # the software renderer answers with no bus
            continue
        _kmt_query(gdi, a.hAdapter, KMT_SEGMENT_SIZE, seg)
        _kmt_query(gdi, a.hAdapter, KMT_REGISTRY, reg)
        found.append({"handle": a.hAdapter, "luid": luid, "name": reg.AdapterString.strip() or None,
                      "pci": f"0000:{addr.BusNumber:02x}:{addr.DeviceNumber:02x}.{addr.FunctionNumber:x}",
                      "vram": seg.DedicatedVideoMemorySize, "gdi": gdi})
    return found


class _PdhValue(ctypes.Structure):                # PDH_FMT_COUNTERVALUE, read as a double
    _fields_ = [("CStatus", ctypes.c_uint32), ("doubleValue", ctypes.c_double)]


class _PdhItem(ctypes.Structure):                 # PDH_FMT_COUNTERVALUE_ITEM_W
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _PdhValue)]


class _Pdh:
    """Every GPU's load and dedicated memory in use from Windows' performance counters, the ones Task Manager shows:
    the load of an adapter is that of its busiest engine (the sum over the processes of each engine's share), as
    Task Manager's GPU column.  Collected at most twice a second for all the cards together."""
    PATHS = ("\\GPU Engine(*)\\Utilization Percentage", "\\GPU Adapter Memory(*)\\Dedicated Usage")
    ENGINE = re.compile(r"luid_0x([0-9a-f]+)_0x([0-9a-f]+)_phys_(\d+)_eng_(\d+)", re.I)
    ADAPTER = re.compile(r"luid_0x([0-9a-f]+)_0x([0-9a-f]+)", re.I)
    MORE_DATA = 0x800007D2
    FORMAT = 0x00000200 | 0x00008000             # PDH_FMT_DOUBLE | PDH_FMT_NOCAP100

    def __init__(self):
        self.lib = None
        self.lock = threading.Lock()
        self.at = 0.0
        self.util: dict = {}
        self.used: dict = {}
        try:
            lib = ctypes.WinDLL("pdh")
            lib.PdhGetFormattedCounterArrayW.restype = ctypes.c_uint32
            q = ctypes.c_void_p()
            if lib.PdhOpenQueryW(None, None, ctypes.byref(q)) != 0:
                return
            self.counters = []
            for path in self.PATHS:
                c = ctypes.c_void_p()
                if lib.PdhAddEnglishCounterW(q, path, None, ctypes.byref(c)) != 0:
                    return
                self.counters.append(c)
            lib.PdhCollectQueryData(q)                # a rate needs two collections: the first one is the start
            self.lib, self.query = lib, q
        except (AttributeError, OSError):
            self.lib = None

    def _array(self, counter):
        size, count = ctypes.c_uint32(0), ctypes.c_uint32(0)
        get = self.lib.PdhGetFormattedCounterArrayW
        if get(counter, self.FORMAT, ctypes.byref(size), ctypes.byref(count), None) != self.MORE_DATA:
            return []
        buf = ctypes.create_string_buffer(size.value)
        if get(counter, self.FORMAT, ctypes.byref(size), ctypes.byref(count), buf) != 0:
            return []
        items = ctypes.cast(buf, ctypes.POINTER(_PdhItem * count.value)).contents
        return [(i.szName or "", i.FmtValue.doubleValue) for i in items if i.FmtValue.CStatus in (0, 1)]

    def read(self, luid):
        """(load %, dedicated memory in use in bytes) of the adapter `luid` (high, low), None for one not read."""
        if self.lib is None:
            return None, None
        with self.lock:
            if time.time() - self.at >= 0.5:
                self.at = time.time()
                try:
                    if self.lib.PdhCollectQueryData(self.query) == 0:
                        engines: dict = {}
                        for name, v in self._array(self.counters[0]):
                            m = self.ENGINE.search(name)
                            if m:
                                key = (int(m[1], 16), int(m[2], 16))
                                eng = engines.setdefault(key, {})
                                eng[(m[3], m[4])] = eng.get((m[3], m[4]), 0.0) + v
                        self.util = {k: min(100.0, max(e.values())) for k, e in engines.items()}
                        used: dict = {}
                        for name, v in self._array(self.counters[1]):
                            m = self.ADAPTER.search(name)
                            if m:
                                key = (int(m[1], 16), int(m[2], 16))
                                used[key] = used.get(key, 0.0) + v
                        self.used = used
                except (AttributeError, OSError):
                    pass
            u = self.used.get(luid)
            return self.util.get(luid, 0.0 if self.used else None), int(u) if u is not None else None


class _WinGpu:
    """A GPU on Windows of any maker, with _XeGpu's interface: its name, address, VRAM and temperature from the
    graphics kernel (D3DKMT, gdi32), its load and VRAM in use from the performance counters (_Pdh).  The temperature
    is there where the driver reports it (WDDM 2.4's adapter performance data); power is not (Windows gives it as a
    share of the card's limit only).  `pci` is the card's address as the engine writes it ("0000:01:00.0"); without
    one, the first card with memory of its own."""
    adapters: list | None = None
    pdh: _Pdh | None = None

    def __init__(self, pci=None):
        self.a = None
        if _WinGpu.adapters is None:
            _WinGpu.adapters = _kmt_adapters() if WIN else []
        for a in _WinGpu.adapters:
            if (a["pci"] == str(pci).lower()) if pci else a["vram"] > 0:
                self.a = a
                break
        if self.a is not None:
            self.pci = self.a["pci"]
            if _WinGpu.pdh is None:
                _WinGpu.pdh = _Pdh()

    def ok(self):
        return self.a is not None

    def name(self):
        return f"{self.a['name'] or 'GPU'} at {self.pci}" if self.ok() else None

    def read(self):
        out = {"util": None, "mem_used": None, "mem_total": self.a["vram"] or None, "temp": None}
        if _WinGpu.pdh is not None:
            out["util"], out["mem_used"] = _WinGpu.pdh.read(self.a["luid"])
        perf = _KmtPerf()
        if _kmt_query(self.a["gdi"], self.a["handle"], KMT_PERF, perf) and perf.Temperature:
            out["temp"] = perf.Temperature / 10.0
        out["pcie_rx_mb"] = out["pcie_tx_mb"] = None   # no counters for the link's traffic
        return out


class _NvPciInfo(ctypes.Structure):   # nvmlPciInfo_t
    _fields_ = [("busIdLegacy", ctypes.c_char * 16), ("domain", ctypes.c_uint), ("bus", ctypes.c_uint),
                ("device", ctypes.c_uint), ("pciDeviceId", ctypes.c_uint), ("pciSubSystemId", ctypes.c_uint),
                ("busId", ctypes.c_char * 32)]


def nvidia_pci(index):
    """The PCI address ("0000:09:00.0", as sysfs writes it) of the NVIDIA GPU nvidia-smi numbers `index` (NVML numbers
    them the same way), or None without NVML or that card: the config's "gpu" names the cards so."""
    n = _NvGpu(None)
    if n.lib is None:
        return None
    h = ctypes.c_void_p()
    info = _NvPciInfo()
    try:
        if n.lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(h)) != 0 or \
                n.lib.nvmlDeviceGetPciInfo_v3(h, ctypes.byref(info)) != 0:
            return None
    except (AttributeError, OSError):
        return None
    return f"{info.domain:04x}:{info.bus:02x}:{info.device:02x}.0"


def _gpu(pci=None):
    """The card the engine runs on: the Intel GPU at `pci` (or the first on the xe driver), else the AMD one (amdgpu),
    else the NVIDIA one, else (without `pci`) the first on the i915 driver.  Without `pci` an i915 card comes last:
    it is most often the processor's graphics beside the card the engine runs on.  On Windows: the NVIDIA one through
    NVML, else any maker's through _WinGpu."""
    if WIN:
        n = _NvGpu(pci)
        return n if n.ok() else _WinGpu(pci)
    g = _XeGpu(pci, ("xe", "i915") if pci else ("xe",))
    if g.ok():
        return g
    a = _AmdGpu(pci)
    if a.ok():
        return a
    n = _NvGpu(pci)
    if n.ok():
        return n
    i = _XeGpu(None, ("i915",)) if pci is None else g
    return i if i.ok() else g


def free_vram_mib(gpu_pci=None):
    """Free VRAM of the card (by PCI address, None: as _gpu chooses) in MiB, or None
    when it cannot be read."""
    g = _gpu(gpu_pci)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


def total_vram_mib(gpu_pci=None):
    """Total VRAM on the selected card in MiB, or None when it cannot be read."""
    g = _gpu(gpu_pci)
    if not g.ok():
        return None
    total = g.read().get("mem_total")
    return int(total >> 20) if total is not None else None


def ram_gb() -> float:
    """The RAM the machine has, in GB (0.0 when it cannot be read): serve/autotune.py's choices and the panels."""
    return hardware.ram_gb()


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_name():
    if os.path.exists("/proc/cpuinfo"):
        text = hardware.proc_text("cpuinfo", encoding="utf-8", errors="replace")
        for line in text.splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        try:
            with open("/proc/stat") as fh:      # closed after reading: no ResourceWarning on every sample
                f = [int(x) for x in fh.readline().split()[1:]]
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
            info = dict(line.split(":", 1) for line in hardware.proc_text("meminfo").splitlines())
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ the sampler
class Telemetry:
    # a card's readings kept as a history of their own (history["gpus"]), for the Monitor's tab of that card
    CARD_SERIES = ("util", "mem_used", "temp", "power", "pcie_rx_mb")

    def __init__(self, extra=None, gpus=None):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpus`: the cards to read,
        [{"pci": address or None, "roles": [...], ...}] (serve.server's gpu_roles: the text engine's cards and the
        image engine's), numbered in that order; None: the one card _gpu chooses, the text engine's.  The flat gpu_*
        readings and their history are the text engine's cards' (the frozen-engine check reads them, and an image
        job on a card of its own must not hide a frozen text engine); "gpus" has every card's, the keys of its entry
        in `gpus` beside them."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        cards = gpus or [{"pci": None, "roles": ["text"]}]
        found = [(i, c, _gpu(c.get("pci"))) for i, c in enumerate(cards)]
        self.gpus = [x for x in found if x[2].ok()] or found[:1]
        self.gpu = self.gpus[0][2]
        self.text = [x for x in self.gpus if "text" in (x[1].get("roles") or ())] or self.gpus[:1]
        self.card_hist = [collections.defaultdict(lambda: collections.deque(maxlen=HISTORY)) for _ in self.gpus]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        self.names = [g.name() for _, _, g in self.gpus]
        self.static = {
            "gpu_name": " + ".join(n or "?" for n in self.names) if self.gpu.ok() else None,
            "gpu_count": len(self.gpus),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
        self._disk_prev = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _disk(self):
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        if c is None:   # psutil found no disk (a gVisor container, Windows with its disk counters off)
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
            reads = [g.read() for _, _, g in self.gpus]
            text = [r for r, x in zip(reads, self.gpus) if x in self.text]
            g = dict(text[0])
            if len(text) > 1:
                def vals(k):
                    return [r[k] for r in text if r.get(k) is not None]
                for k in ("mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"):
                    v = vals(k)
                    g[k] = sum(v) if v else None
                u = vals("util")
                g["util"] = sum(u) / len(u) if u else None
                t = vals("temp")
                g["temp"] = max(t) if t else None
                t = vals("mem_temp")
                g["mem_temp"] = max(t) if t else None
            s.update({f"gpu_{k}": v for k, v in g.items()})
            s["gpus"] = [{"index": i, **{k: v for k, v in c.items() if k != "pci"}, "pci": c.get("pci") or
                          getattr(dev, "pci", None), "name": n, **r}
                         for (i, c, dev), n, r in zip(self.gpus, self.names, reads)]
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

    def close(self):
        """Ends the sampler thread (a server that stops, a test's service): it used to run for the life of the
        process."""
        self._stop.set()

    def _loop(self):
        """One sample a second, for as long as the server lives.  A failing sample (a machine whose GPU or psutil
        source is not readable) must not end the thread, or `GET /metrics` silently stops updating: it says so once
        and keeps trying (upstream b3db5a1c8, #3590)."""
        complained = False
        while not self._stop.is_set():
            try:
                s = self.sample()
            except Exception as e:                    # noqa: BLE001 - a monitor must outlive its source
                if not complained:                    # once, not once a second
                    print(f"[xestrata] metrics: a sample failed ({e}); the sampler keeps trying", flush=True)
                    complained = True
                self._stop.wait(1.0)
                continue
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s", "prefill_tok_s_mean", "image_step_s"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
                for h, card in zip(self.card_hist, s.get("gpus") or ()):
                    for k in self.CARD_SERIES:
                        v = card.get(k)
                        h[k].append(round(v, 2) if isinstance(v, float) else v)
            self._stop.wait(1.0)

    def snapshot(self):
        with self.lock:
            history: dict = {k: list(v) for k, v in self.hist.items()}
            if self.now.get("gpus"):
                history["gpus"] = [{k: list(v) for k, v in h.items()} for h in self.card_hist]
            return {"now": dict(self.now), "history": history, "static": dict(self.static)}
