"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.  With the AMD backend (#301): the
  amdgpu driver's Linux sysfs files - load, VRAM, temperature and power; on Windows, for the Monitor, the driver's AMD Display
  Library (ADL2 PMLog, #4884).  On macOS (Metal backend): IOKit's
  IOAccelerator registry - load and VRAM; the temperature and power sensors need root (`powermetrics`), and an Apple
  Silicon GPU sits on the die, so there is no PCIe link.
- CPU, RAM, disk: `psutil` when it is installed (setup installs it); without it the CPU and RAM readings fall back to
  the OS (Windows GlobalMemoryStatusEx / GetSystemTimes, Linux /proc) and the disk rate is absent.
Anything that cannot be read is None; nothing here can stop the server.
"""
from __future__ import annotations

import collections
import ctypes
import os
from strata_control import sysinfo
import re
import subprocess
import sys
import threading
import time

HISTORY = 60
# A GP100 (Pascal) reports PCIe traffic of just under 200 GiB/s now and then, even idle: 2^22 KB in NVML's 20 ms window,
# which reads like the counter stepping back and wrapping.  No link carries it (PCIe 6.0 x16 is under 120 GiB/s).
PCIE_KB_S_MAX = 150 << 20


# ------------------------------------------------------------------------------------------------ NVML
class _Nvml:
    class Util(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class Mem(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self, index=0):
        self.lib = self.dev = None
        names = ["nvml.dll", os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                                          "NVIDIA Corporation", "NVSMI", "nvml.dll")] if os.name == "nt" \
            else ["libnvidia-ml.so.1", "libnvidia-ml.so"]
        for n in names:
            try:
                self.lib = ctypes.CDLL(n)
                break
            except OSError:
                continue
        if self.lib is None:
            return
        try:
            init = getattr(self.lib, "nvmlInit_v2", None) or self.lib.nvmlInit
            if init() != 0:
                self.lib = None
                return
            h = ctypes.c_void_p()
            get = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", None) or self.lib.nvmlDeviceGetHandleByIndex
            if get(ctypes.c_uint(index), ctypes.byref(h)) != 0:
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
        buf = ctypes.create_string_buffer(96)
        try:
            if self.lib.nvmlDeviceGetName(self.dev, buf, ctypes.c_uint(96)) == 0:
                return buf.value.decode(errors="replace")
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
        out["pcie_gen"] = self._uint("nvmlDeviceGetCurrPcieLinkGeneration")        # drops at idle (power saving)
        out["pcie_gen_max"] = self._uint("nvmlDeviceGetMaxPcieLinkGeneration")
        out["pcie_width"] = self._uint("nvmlDeviceGetCurrPcieLinkWidth")
        # the widest the card itself can go: read beside the negotiated width, so a card capped to x8 in a x16
        # slot, or trained down to x4 by the board, is visible as the gap instead of looking like an x8 card by choice
        out["pcie_width_max"] = self._uint("nvmlDeviceGetMaxPcieLinkWidth")
        out["pcie_rx_mb"] = self._pcie_mb(1)                                             # NVML_PCIE_UTIL_RX_BYTES
        out["pcie_tx_mb"] = self._pcie_mb(0)
        return out

    def _pcie_mb(self, counter):
        kb = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(counter))           # KB/s
        if kb is not None and kb > PCIE_KB_S_MAX:                                        # wrapped: the next read is good
            kb = self._uint("nvmlDeviceGetPcieThroughput", ctypes.c_uint(counter))
        return kb / 1024.0 if kb is not None and kb <= PCIE_KB_S_MAX else None


# ------------------------------------------------------------------------------------------------ AMD (Linux sysfs)
SYSFS = "/sys"


def amd_device_dir(index, sysfs=None):
    """The amdgpu sysfs folder (/sys/class/drm/renderD<N>/device) of the AMD GPU that HIP numbers `index`: the KFD
    topology's GPU nodes in order, the CPU nodes skipped, linked to their render node by drm_render_minor - the
    numbering setup's amd_gpus() and HIP_VISIBLE_DEVICES use.  None when there is no such card (or no amdgpu)."""
    base = os.path.join(sysfs or SYSFS, "class", "kfd", "kfd", "topology", "nodes")
    try:
        nodes = sorted((n for n in os.listdir(base) if n.isdigit()), key=int)
    except OSError:
        return None
    gpus = []
    for n in nodes:
        try:
            with open(os.path.join(base, n, "properties"), encoding="utf-8") as f:
                props = dict(line.strip().partition(" ")[::2] for line in f if line.strip())
            if int(props.get("gfx_target_version") or 0) == 0 or int(props.get("simd_count") or 0) == 0:
                continue
            gpus.append(props)
        except (OSError, ValueError):
            continue
    if not 0 <= index < len(gpus) or not gpus[index].get("drm_render_minor"):
        return None
    dev = os.path.join(sysfs or SYSFS, "class", "drm", "renderD" + gpus[index]["drm_render_minor"].strip(), "device")
    return dev if os.path.isdir(dev) else None


class _Amd:
    """#301: an AMD card's readings from the amdgpu driver's sysfs files (Linux; no ROCm library needed), with _Nvml's
    interface: load (gpu_busy_percent), VRAM (mem_info_vram_used / _total), and from its hwmon folder the temperature
    (temp1_input, the edge sensor, m°C), power (power1_average or power1_input, µW) and its cap (power1_cap)."""

    def __init__(self, index=0, sysfs=None):
        self.dev = amd_device_dir(index, sysfs)
        self.hwmon = None
        if self.dev:
            try:
                hw = sorted(os.listdir(os.path.join(self.dev, "hwmon")))
                self.hwmon = os.path.join(self.dev, "hwmon", hw[0]) if hw else None
            except OSError:
                pass

    def ok(self):
        return self.dev is not None

    @staticmethod
    def _int(path):
        try:
            with open(path, encoding="utf-8") as f:
                return int(f.read().strip())
        except (OSError, ValueError, TypeError):
            return None

    def name(self):
        try:
            with open(os.path.join(self.dev, "product_name"), encoding="utf-8") as f:
                return f.read().strip() or "AMD Radeon"
        except (OSError, TypeError):
            return "AMD Radeon"

    @staticmethod
    def _gen(text):
        """The PCIe generation of a link-speed file's contents ("16.0 GT/s PCIe" -> 4), or None."""
        try:
            return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}[float(str(text).split()[0])]
        except (ValueError, TypeError, IndexError, KeyError):
            return None

    @staticmethod
    def _bdf(s):
        """True for a sysfs pci device name ("0000:03:00.0"); nothing is imported for it."""
        return (len(s) == 12 and s[4] == ":" and s[7] == ":" and s[10] == "." and s[11] in "01234567"
                and all(c in "0123456789abcdef" for c in s[0:4] + s[5:7] + s[8:10]))

    @staticmethod
    def _read(d, name):
        try:
            with open(os.path.join(d, name), encoding="utf-8") as f:
                return f.read().strip()
        except (OSError, TypeError):
            return None

    def _hops(self):
        """The PCIe devices between the root port and this card, the card last: the sysfs path names every
        hop (a card behind a bridge chain has more than one; a directly attached card has one)."""
        out, prefix = [], []
        for part in os.path.realpath(self.dev or "").split("/"):
            prefix.append(part)
            if self._bdf(part):
                out.append("/".join(prefix))
        return out

    def link(self):
        """The PCIe link the card actually gets: the **narrowest/slowest hop** between the root port and the card,
        from each hop's `max_link_speed` / `max_link_width` (the capability, so a power-saving downgrade or a Gen3
        slot cannot make it read low).  A card that is Gen4 on its own hop but sits behind a Gen3 root port really
        runs at Gen3, and that is the number a PCIe bandwidth budget needs.
        `pcie_own_gen` and `pcie_width_max` keep the card's own capability aside: the gap between the two numbers is
        what a user has to see (a x16 card that only reaches x8).

        The bottleneck is only claimed when **every** hop of the path could be read; `pcie_path` reports how many
        of them were ("4/4").  A kernel or a container that hides part of /sys/devices would otherwise drop the
        unreadable hops in silence and report the card's own Gen4 hop as the whole path - i.e. be optimistic
        exactly where it matters.  When the walk is incomplete the reading falls back to the card's own negotiated
        link and pcie_path says so."""
        hops = self._hops()
        gens, widths, read = [], [], 0
        for d in hops:
            g = self._gen(_Amd._read(d, "max_link_speed"))
            w = self._int(os.path.join(d, "max_link_width"))
            read += 1 if (g is not None or w is not None) else 0
            if g:
                gens.append(g)
            if w:
                widths.append(w)
        own = self.dev or ""
        own_gen = self._gen(_Amd._read(own, "current_link_speed"))
        whole = bool(hops) and read == len(hops)
        gen = min(gens) if (whole and gens) else own_gen
        width = min(widths) if (whole and widths) else self._int(os.path.join(own, "current_link_width"))
        # `pcie_width_max` is the card's own cap (its own hop, kept last by `_hops()`), where `pcie_width` above is
        # the narrowest hop of the whole path: a card that runs at x8 behind a x8 bridge, or in a slot the board
        # trained down, reads as the gap instead of looking like an x8 card by choice.  Same split as `pcie_own_gen`.
        cap = self._int(os.path.join(own, "max_link_width")) if own else None
        if cap is None and hops:
            cap = self._int(os.path.join(hops[-1], "max_link_width"))
        return {"pcie_gen": gen, "pcie_gen_max": gen, "pcie_own_gen": own_gen, "pcie_width": width,
                "pcie_width_max": cap, "pcie_path": "%d/%d" % (read, len(hops))}


    def read(self):
        own = self.dev or ""      # (#4884) a box with no amdgpu sysfs device (Windows): None readings, never a TypeError
        out = {"util": self._int(os.path.join(own, "gpu_busy_percent")),
               "mem_used": self._int(os.path.join(own, "mem_info_vram_used")),
               "mem_total": self._int(os.path.join(own, "mem_info_vram_total"))}
        if self.hwmon:
            t = self._int(os.path.join(self.hwmon, "temp1_input"))
            out["temp"] = t / 1000.0 if t is not None else None
            p = self._int(os.path.join(self.hwmon, "power1_average"))
            if p is None:
                p = self._int(os.path.join(self.hwmon, "power1_input"))
            out["power"] = p / 1e6 if p is not None else None
            cap = self._int(os.path.join(self.hwmon, "power1_cap"))
            out["power_limit"] = cap / 1e6 if cap is not None else None
        out.update(self.link())
        return out


# ------------------------------------------------------------------------------------------------ AMD (Windows ADL2)
# #4884 (Lip Wee Yeo, ported from #2704's telemetry part): an AMD card's readings on Windows come from the driver's own AMD Display
# Library (atiadlxx.dll), the interface HWInfo and LibreHardwareMonitor use, so no ROCm or pip package is needed.  ADL2's PMLog
# (Overdrive 8) gives load, temperature and power; DedicatedVRAMUsage + MemoryInfoX4 give VRAM; Device Manager's Plug-and-Play
# record the PCIe link.  ctypes only; every call is guarded (the ABI varies by Adrenalin version); anything that cannot be read is
# None and nothing here can stop the server.  The Monitor only: monitor_reader() hands it out, gpu_reader() does not, so the VRAM checks the
# server and Control decide with (free_vram_mib, gpu_vram_mib, the drafter's sizing) keep their Windows AMD behaviour.

class _AdlSensor(ctypes.Structure):
    """One PMLog sensor reading: whether the card reports it, and its raw value."""
    _fields_ = [("supported", ctypes.c_int), ("value", ctypes.c_int)]


class _Adl:
    """The ADL2 ctypes bindings; `lib` is None when atiadlxx.dll is absent (and off Windows)."""

    class AdapterInfo(ctypes.Structure):
        _fields_ = [("Size", ctypes.c_int), ("AdapterIndex", ctypes.c_int),
                    ("UDID", ctypes.c_char * 256), ("BusNumber", ctypes.c_int),
                    ("DeviceNumber", ctypes.c_int), ("FunctionNumber", ctypes.c_int),
                    ("VendorID", ctypes.c_int), ("AdapterName", ctypes.c_char * 256),
                    ("DisplayName", ctypes.c_char * 256), ("Present", ctypes.c_int),
                    ("Exist", ctypes.c_int), ("DriverPath", ctypes.c_char * 256),
                    ("DriverPathExt", ctypes.c_char * 256), ("PNPString", ctypes.c_char * 256),
                    ("OSDisplayIndex", ctypes.c_int)]

    class MemInfoX4(ctypes.Structure):
        _fields_ = [("iMemorySize", ctypes.c_longlong), ("strMemoryType", ctypes.c_char * 256),
                    ("iMemoryBandwidth", ctypes.c_longlong), ("iHyperMemorySize", ctypes.c_longlong),
                    ("iInvisibleMemorySize", ctypes.c_longlong), ("iVisibleMemorySize", ctypes.c_longlong),
                    ("iVramVendorRevId", ctypes.c_longlong), ("iMemoryBandwidthX2", ctypes.c_longlong),
                    ("iMemoryBitRateX2", ctypes.c_longlong)]

    class PmLogSupport(ctypes.Structure):
        _fields_ = [("usSensors", ctypes.c_ushort * 256), ("iReserved", ctypes.c_int * 16)]

    class PmLogStartInput(ctypes.Structure):
        _fields_ = [("usSensors", ctypes.c_ushort * 256), ("ulSampleRate", ctypes.c_uint),
                    ("iReserved", ctypes.c_int * 15)]

    class PmLogStartOutput(ctypes.Structure):
        _fields_ = [("pLoggingAddress", ctypes.c_void_p)]

    class PmLogDataOutput(ctypes.Structure):
        _fields_ = [("size", ctypes.c_int), ("sensors", _AdlSensor * 256)]

    # PMLog sensor ids (LibreHardwareMonitor's ADLPMLogSensors enum)
    S_EDGE, S_MEM, S_ASIC_POWER, S_ACTIVITY_GFX, S_ACTIVITY_MEM = 8, 9, 23, 19, 20
    S_GFX_POWER, S_BUS_SPEED, S_BUS_LANES, S_TOTAL_POWERLIMIT = 30, 40, 41, 47
    S_BOARD_POWER = 73
    # the temperature: the first one the card reports.  Edge first (what the discrete cards of #4884 were checked with), then the
    # graphics, hotspot and SoC sensors (#3682: a Radeon 8060S reports the graphics one, 28, and no edge), then the memory
    S_TEMPS = (8, 28, 27, 29, 9)

    def _alloc(self, size):
        # a raw heap pointer ADL can free(): ADL2_Main_Control_Create allocates through this callback and frees on Destroy
        p = self.crt.malloc(size) if size else None
        return ctypes.cast(p, ctypes.c_void_p).value if p else None

    def __init__(self):
        self.lib = None
        if os.name != "nt":
            return
        try:
            self.lib = ctypes.CDLL("atiadlxx.dll")
            # atiadlxx links ucrtbase.dll, so malloc from ucrtbase shares the heap ADL frees with; else the process's C runtime
            try:
                self.crt = ctypes.CDLL("ucrtbase.dll")
            except OSError:
                self.crt = ctypes.CDLL(None)
            self.crt.malloc.restype = ctypes.c_void_p
            self.crt.malloc.argtypes = [ctypes.c_size_t]
            self.ALLOC = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_int)(self._alloc)
        except (OSError, TypeError, ValueError, AttributeError):
            self.lib = None


def _adl_cards(buf, count):
    """The ADL adapter rows that are cards, one per card (ADL lists a card once per display; deduped by its DEV_xxxx code):
    every present discrete card, or, when there is none, the integrated Radeon (an APU alone, Strix Halo) - the same rule
    as windows_amd_cards().  [(adapter row, DEV code or None, name, integrated)]."""
    disc, integ, seen = [], [], set()
    for i in range(count):
        nm = buf[i].AdapterName
        if not nm or not (buf[i].Present or buf[i].Exist):
            continue
        udid = buf[i].UDID or b""
        dev = udid.split(b"DEV_", 1)[1].split(b"&", 1)[0] if b"DEV_" in udid else None
        key = dev or nm
        if key in seen:
            continue
        seen.add(key)
        row = (i, dev.decode(errors="replace").upper() if dev else None, nm.decode(errors="replace").strip())
        (integ if b"Graphics" in nm else disc).append(row)
    return [r + (False,) for r in disc] or [r + (True,) for r in integ]


class _AmdWin:
    """An AMD card's readings on Windows through ADL2's PMLog, with _Nvml's interface: util, mem_used, mem_total, temp,
    power, power_limit, and the PCIe link: pcie_gen / pcie_gen_max / pcie_width_max from Device Manager's record of the card,
    pcie_width from PMLog's BUS_LANES (the lanes the card runs at).  `index` is HIP's number of the card: matched to ADL's
    rows by name in windows_amd_cards() order (the order the rest of the server takes as HIP's), else by position."""

    STALL_LIMIT = 60   # consecutive 1 s samples with identical util+power before the stream is restarted

    def __init__(self, index=0):
        self.adl = _Adl()
        self.lib = self.adl.lib
        self.ctx = None
        self.adapter = None
        self.card_count = 0
        self.dev_code = None
        self.display_name = None
        self.integrated = False
        self.device = ctypes.c_uint(0)
        self.data = None
        self._stall = 0
        self._sig = None
        if self.lib is not None:
            try:
                self._setup(index)
            except Exception:  # noqa: BLE001 - a driver that answers oddly leaves the reader off, never the server down
                self.adapter = None

    def _call(self, fn_name, *args):
        try:
            return getattr(self.lib, fn_name)(*args)
        except (AttributeError, OSError, ctypes.ArgumentError):
            return None

    def _setup(self, index):
        ctx = ctypes.c_void_p()
        if self._call("ADL2_Main_Control_Create", self.adl.ALLOC, 1, ctypes.byref(ctx)) != 0:
            return
        self.ctx = ctx
        n = ctypes.c_int(0)
        if self._call("ADL2_Adapter_NumberOfAdapters_Get", ctx, ctypes.byref(n)) != 0:
            return
        count = max(1, min(n.value, 40))
        buf = (self.adl.AdapterInfo * count)()
        if self._call("ADL2_Adapter_AdapterInfo_Get", ctx, ctypes.cast(buf, ctypes.c_void_p), ctypes.sizeof(buf)) != 0:
            return
        cards = _adl_cards(buf, count)
        self.card_count = len(cards)
        pick = _pick_card([c[2] for c in cards], index, [c["name"] for c in windows_amd_cards()])
        if pick is None:
            return
        self.adapter, self.dev_code, self.display_name, self.integrated = cards[pick]
        # start the PMLog stream here: without it the driver answers QueryPMLogData_Get with its last-known buffer (sensors
        # frozen at idle; measured on an R9700 by the PR's author)
        self._start_pmlog()

    def ok(self):
        return self.lib is not None and self.adapter is not None

    def name(self):
        return (self.display_name or "AMD Radeon") if self.ok() else "AMD Radeon"

    def _read_pmlog(self):
        if self.data is None:
            self.data = self.adl.PmLogDataOutput()
        if self._call("ADL2_New_QueryPMLogData_Get", self.ctx, ctypes.c_int(self.adapter), ctypes.byref(self.data)) != 0:
            return None
        return self.data

    def _start_pmlog(self):
        if not self.ok() or self.device.value:
            return
        if self._call("ADL2_Device_PMLog_Device_Create", self.ctx, ctypes.c_int(self.adapter), ctypes.byref(self.device)) != 0:
            return
        sup = self.adl.PmLogSupport()
        if self._call("ADL2_Adapter_PMLog_Support_Get", self.ctx, ctypes.c_int(self.adapter), ctypes.byref(sup)) != 0:
            self._destroy_device()
            return
        inp = self.adl.PmLogStartInput()
        for k in range(256):
            inp.usSensors[k] = sup.usSensors[k]
        inp.ulSampleRate = 1000
        out = self.adl.PmLogStartOutput()
        if self._call("ADL2_Adapter_PMLog_Start", self.ctx, ctypes.c_int(self.adapter), ctypes.byref(inp), ctypes.byref(out),
                      self.device) != 0:
            self._destroy_device()

    def _destroy_device(self):
        if self.device.value and self.ctx is not None and self.adapter is not None:
            self._call("ADL2_Device_PMLog_Device_Destroy", self.ctx, ctypes.c_int(self.adapter), self.device)
        self.device = ctypes.c_uint(0)

    def _restart_pmlog(self):
        """Stop and re-create the stream: in deep idle the driver tears sampling down while QueryPMLogData_Get keeps
        answering with the frozen buffer, which only a stall check notices."""
        if self.device.value and self.adapter is not None:
            self._call("ADL2_Adapter_PMLog_Stop", self.ctx, ctypes.c_int(self.adapter), self.device)
        self._destroy_device()
        self._start_pmlog()

    def _watch_stall(self, out):
        """util+power identical STALL_LIMIT samples in a row (1 s each): the stream died; restart it once."""
        sig = (out.get("util"), out.get("power"))
        if sig == self._sig and sig != (None, None):
            self._stall += 1
        else:
            self._stall = 1
        self._sig = sig
        if self._stall >= self.STALL_LIMIT:
            self._restart_pmlog()
            self._stall = 0

    def close(self):
        """Give back the PMLog device handle and the ADL context (#1434: +2 handles a reader that is never closed).  Safe twice."""
        if self.ctx is not None:
            self._destroy_device()
            self._call("ADL2_Main_Control_Destroy", self.ctx)
            self.ctx = None
        self.adapter = None
        self.card_count = 0

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown can leave the driver DLL unloaded already
            pass

    def read(self):
        out = {}
        if not self.ok():
            return out
        d = self._read_pmlog()
        if d is None:
            self._start_pmlog()
            d = self._read_pmlog()
        lanes = None
        if d is not None:
            def val(sid):
                s = d.sensors[sid]
                return s.value if s.supported else None
            u = [x for x in (val(self.adl.S_ACTIVITY_GFX), val(self.adl.S_ACTIVITY_MEM)) if x is not None]
            out["util"] = max(u) if u else None
            t = next((v for v in (val(sid) for sid in self.adl.S_TEMPS) if v is not None), None)
            out["temp"] = float(t) if t is not None else None
            # RDNA3+ report power as BOARD_POWER (73); older cards ASIC_POWER (23) or GFX_POWER (30); watts (factor 1.0)
            p = val(self.adl.S_BOARD_POWER)
            if p is None:
                p = val(self.adl.S_ASIC_POWER)
            if p is None:
                p = val(self.adl.S_GFX_POWER)
            out["power"] = float(p) if p else None
            lim = val(self.adl.S_TOTAL_POWERLIMIT)
            out["power_limit"] = float(lim) if lim else None
            lanes = val(self.adl.S_BUS_LANES)
        used = ctypes.c_int(0)
        if self._call("ADL2_Adapter_DedicatedVRAMUsage_Get", self.ctx, ctypes.c_int(self.adapter), ctypes.byref(used)) == 0 \
                and used.value > 0:
            out["mem_used"] = used.value * 1024 * 1024
        mi = self.adl.MemInfoX4()
        if self._call("ADL2_Adapter_MemoryInfoX4_Get", self.ctx, ctypes.c_int(self.adapter), ctypes.byref(mi)) == 0 \
                and mi.iMemorySize > 0:
            out["mem_total"] = int(mi.iMemorySize)
        # an integrated Radeon has no PCIe slot: Device Manager still records an internal x16 link for it (Strix Halo: Gen4 x16), which
        # is not a link the Monitor should show
        if not self.integrated:
            out.update(_adl_link(_dm_link_map().get(self.dev_code or ""), lanes))
        self._watch_stall(out)
        return out


def _pick_card(adl_names, index, hip_names):
    """Which of ADL's card rows is HIP's card `index`: the k-th row named like hip_names[index] (k = how many cards of
    that name come before it in HIP's order), else row `index`; None when there is no such row."""
    if 0 <= index < len(hip_names):
        want = hip_names[index].strip().lower()
        k = sum(1 for n in hip_names[:index] if n.strip().lower() == want)
        rows = [r for r, n in enumerate(adl_names) if n.strip().lower() == want]
        if k < len(rows):
            return rows[k]
    return index if 0 <= index < len(adl_names) else None


def _adl_link(dm, lanes):
    """The link fields from Device Manager's (current speed, current width, max speed, max width) of the card and PMLog's
    BUS_LANES.  PMLog's BUS_SPEED is the GPU's internal bus (power-saved to Gen4 at idle), so the generation is Device
    Manager's; Device Manager's CurrentLinkWidth reports the slot's capability, so the width is BUS_LANES.  None for a
    field that cannot be read (an APU's on-die GPU has no PCIe link)."""
    out = {}
    if dm:
        cur_speed, _cur_width, max_speed, max_width = dm
        out["pcie_gen"] = cur_speed or None
        out["pcie_gen_max"] = max_speed or None
        out["pcie_width_max"] = max_width or None
    out["pcie_width"] = lanes or None
    return out


_DM_LINK = None        # {DEV code: (current speed, current width, max speed, max width)} from Device Manager, once read
_DM_LINK_RETRY = 0.0   # a failed query is retried after this monotonic time, not on every 1-second sample
_DM_LINK_STARTED = False
_DM_LINK_LOCK = threading.Lock()


def _dm_link_map():
    """Every present display device's PCIe link from Device Manager's Plug-and-Play records, keyed by its DEV_xxxx code.  The
    first call starts the (seconds-long) query in a background thread and returns {}; a failed query is retried at most every
    60 s; {} off Windows or when unavailable."""
    global _DM_LINK, _DM_LINK_STARTED
    if _DM_LINK is not None:
        return _DM_LINK
    if os.name != "nt":
        _DM_LINK = {}
        return _DM_LINK
    if time.monotonic() >= _DM_LINK_RETRY:
        with _DM_LINK_LOCK:
            if _DM_LINK is None and not _DM_LINK_STARTED:
                _DM_LINK_STARTED = True
                threading.Thread(target=_dm_link_query, daemon=True, name="dm-link").start()
    return _DM_LINK if _DM_LINK is not None else {}


def _dm_link_query():
    """The Device Manager query behind _dm_link_map, in its own thread, with no console window (CREATE_NO_WINDOW)."""
    global _DM_LINK, _DM_LINK_RETRY, _DM_LINK_STARTED
    ps = ("Get-PnpDevice -Class Display -ErrorAction SilentlyContinue | Where-Object { $_.Status -ne 'Not Connected' } | "
          "ForEach-Object { $id = $_.InstanceId; if ($id -match 'DEV_([0-9A-Fa-f]{4})') { $dev = $matches[1] } else { return }; "
          "$v = foreach ($k in 'CurrentLinkSpeed','CurrentLinkWidth','MaxLinkSpeed','MaxLinkWidth') { "
          "(Get-PnpDeviceProperty -InstanceId $id -KeyName ('DEVPKEY_PciDevice_' + $k) -ErrorAction SilentlyContinue).Data }; "
          "Write-Output ('DEV_' + $dev + ' ' + ($v -join ' ')) }")
    try:
        done = subprocess.run(["powershell", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", ps], capture_output=True,
                              text=True, timeout=20, stdin=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        _DM_LINK_RETRY = time.monotonic() + 60
        _DM_LINK_STARTED = False
        return
    m = {}
    for line in (done.stdout or "").splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0].startswith("DEV_"):
            try:
                m[parts[0][4:].upper()] = tuple(int(x) for x in parts[1:])
            except ValueError:
                continue
    if not m and done.returncode != 0:
        _DM_LINK_RETRY = time.monotonic() + 60
        _DM_LINK_STARTED = False
        return
    _DM_LINK = m


def metal_working_set():
    """The Apple GPU's memory budget in bytes: MTLCreateSystemDefaultDevice's recommendedMaxWorkingSetSize, through
    the Metal framework and objc_msgSend (no PyObjC) - the same property metal/src/mtl_rt.mm reads for the engine
    (its "GPU working set" log line).  None off macOS, or when it cannot be read: a selector the device does not
    answer would raise an Objective-C exception no ctypes call can catch, so the class is asked first."""
    if sys.platform != "darwin":
        return None
    try:
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        objc = ctypes.CDLL("/usr/lib/libobjc.dylib")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.object_getClass.argtypes = [ctypes.c_void_p]
        objc.object_getClass.restype = ctypes.c_void_p
        objc.class_respondsToSelector.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        objc.class_respondsToSelector.restype = ctypes.c_ubyte
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        objc.objc_msgSend.restype = ctypes.c_ulonglong
        dev = metal.MTLCreateSystemDefaultDevice()
        if not dev:
            return None
        sel = objc.sel_registerName(b"recommendedMaxWorkingSetSize")
        if not objc.class_respondsToSelector(objc.object_getClass(dev), sel):
            return None
        return int(objc.objc_msgSend(ctypes.c_void_p(dev), ctypes.c_void_p(sel))) or None
    except (OSError, AttributeError, ValueError):
        return None


class _Metal:
    """macOS: an Apple GPU's readings from the IOKit registry (`ioreg -l -w0 -r -c IOAccelerator`, 22 ms), with
    _Nvml's interface: the load and the memory are the driver's PerformanceStatistics of the index'th accelerator
    node - Device Utilization % and the in-use system memory in bytes.  The total is the device's Metal working set,
    the budget one app gets of the unified memory.  No temperature or power reading exists without root
    (`powermetrics`), and an Apple Silicon GPU sits on the die, so the PCIe cards have nothing to show."""
    note = ("no GPU temperature or power reading without root (powermetrics), and an Apple Silicon GPU sits on the "
            "die, so there is no PCIe link to show; the engine's own figures are in its log")
    cmd = ("ioreg", "-l", "-w0", "-r", "-c", "IOAccelerator")

    def __init__(self, index=0, run=None):
        self.index = index
        self._run = run or self._ioreg
        self.nodes = len(self._blocks())
        self._name = None
        b = self._block()
        if b:
            m = re.search(r'"model" = "([^"]+)"', b)
            self._name = m.group(1) if m else None

    @staticmethod
    def _ioreg():
        try:
            return subprocess.run(list(_Metal.cmd), capture_output=True, text=True, timeout=10).stdout or ""
        except (OSError, subprocess.SubprocessError):
            return ""

    def _blocks(self):
        """The accelerator nodes, in the registry's order: every node that carries a PerformanceStatistics."""
        try:
            out = self._run() or ""
        except OSError:                     # ioreg missing, or a caller's runner that raises
            return []
        return [b for b in re.split(r"^\s*\+-o ", out, flags=re.M) if '"PerformanceStatistics"' in b]

    def _block(self):
        b = self._blocks()
        return b[self.index] if 0 <= self.index < len(b) else None

    def ok(self):
        return self.index < self.nodes

    def name(self):
        return self._name

    def read(self):
        b = self._block()
        if b is None:
            return {}

        def num(key):
            m = re.search(r'"%s"=([0-9.]+)' % re.escape(key), b)
            return float(m.group(1)) if m else None

        out = {}
        u = num("Device Utilization %")
        if u is None:                      # older chips report only the renderer and the tiler
            r, t = num("Renderer Utilization %"), num("Tiler Utilization %")
            u = max(v for v in (r, t) if v is not None) if (r is not None or t is not None) else None
        out["util"] = int(u) if u is not None else None
        used = num("In use system memory")
        out["mem_used"] = int(used) if used is not None else None
        out["mem_total"] = metal_working_set()
        return out


def gpu_reader(index=0, amd=False):
    """The card's readings: NVML (NVIDIA), the amdgpu sysfs files with the AMD backend (#301), or IOKit's
    IOAccelerator registry on macOS (Metal backend).  `amd` comes first: no Mac runs the HIP backend, and the
    tests' AMD trees are read on a Mac too."""
    if amd:
        return _Amd(index)
    return _Metal(index) if sys.platform == "darwin" else _Nvml(index)


def monitor_reader(index=0, amd=False):
    """The Monitor's reader (Telemetry): gpu_reader's, except that with the AMD backend on Windows, where the amdgpu sysfs reader
    has nothing to read, the driver's ADL2 (#4884).  Every other caller (the server's VRAM checks, the drafter's sizing) keeps
    gpu_reader, so their decisions on Windows AMD are unchanged."""
    g = gpu_reader(index, amd)
    if amd and os.name == "nt" and not g.ok():
        return _AmdWin(index)
    return g


_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"


def windows_display_adapters(vendor=None) -> list:
    """#1587: the display adapters Windows has drivers for, from the registry's display class (no subprocess, no vendor
    library): [{"name", "provider", "driver", "vram_bytes"}] in the class's order.  `vendor`: "amd", "nvidia" or
    "intel" keeps that vendor's cards.  [] off Windows or when nothing can be read."""
    if os.name != "nt":
        return []
    try:
        import winreg
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except OSError:
        return []
    out = []
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(root, i)
        except OSError:
            break
        i += 1
        if not sub.isdigit():
            continue
        try:
            k = winreg.OpenKey(root, sub)
        except OSError:
            continue

        def val(name, k=k):
            try:
                return winreg.QueryValueEx(k, name)[0]
            except OSError:
                return None
        name = val("DriverDesc")
        if not isinstance(name, str) or not name.strip():
            continue
        mem = val("HardwareInformation.qwMemorySize")
        if mem is None:
            mem = val("HardwareInformation.MemorySize")
        if isinstance(mem, (bytes, bytearray)):
            mem = int.from_bytes(bytes(mem[:8]), "little") if mem else None
        out.append({"name": name.strip(), "provider": str(val("ProviderName") or ""), "driver": val("DriverVersion"),
                    "vram_bytes": mem if isinstance(mem, int) and mem > 0 else None})
    if vendor:
        keys = {"amd": ("advanced micro devices", "amd", "radeon"), "nvidia": ("nvidia",), "intel": ("intel",)}[vendor]
        out = [a for a in out if any(s in (a["provider"] + " " + a["name"]).lower() for s in keys)]
    return out


def windows_amd_cards(adapters=None) -> list:
    """The AMD cards HIP runs on, as windows_display_adapters lists them.  A Ryzen's integrated Radeon is listed too
    (512 MiB on a Ryzen 5 7600): beside a card with 2 GiB or more it is left out (an APU alone keeps it)."""
    cards = windows_display_adapters("amd") if adapters is None else adapters
    return [c for c in cards if (c["vram_bytes"] or 0) >= 2 << 30] or cards


def free_vram_mib(index=0, amd=False):
    """Free VRAM of a card in MiB, or None when it cannot be read."""
    g = gpu_reader(index, amd)
    if not g.ok():
        return None
    r = g.read()
    if r.get("mem_total") is None or r.get("mem_used") is None:
        return None
    return int((r["mem_total"] - r["mem_used"]) >> 20)


# ------------------------------------------------------------------------------------------------ CPU / RAM
def _cpu_topology(root: str = "/sys/devices/system"):
    """(sockets, NUMA nodes) of this computer: how many CPU packages it has and how many memory nodes (None where it cannot be read).
    Linux reads /sys, Windows asks GetLogicalProcessorInformationEx / GetNumaHighestNodeNumber, macOS asks sysctl (Apple silicon and every Mac
    sold so far: one package, no NUMA)."""
    try:
        if os.name == "nt":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            needed = ctypes.c_ulong(0)
            kernel32.GetLogicalProcessorInformationEx(3, None, ctypes.byref(needed))          # RelationProcessorPackage
            buf = ctypes.create_string_buffer(needed.value)
            sockets = None
            if needed.value and kernel32.GetLogicalProcessorInformationEx(3, buf, ctypes.byref(needed)):
                sockets, offset = 0, 0
                while offset + 8 <= needed.value:
                    relationship = int.from_bytes(buf.raw[offset:offset + 4], "little")
                    size = int.from_bytes(buf.raw[offset + 4:offset + 8], "little")
                    if size <= 0:
                        break
                    sockets += 1 if relationship == 3 else 0
                    offset += size
            high = ctypes.c_ulong(0)
            numa = high.value + 1 if kernel32.GetNumaHighestNodeNumber(ctypes.byref(high)) else None
            return (sockets or None), numa
        if sys.platform == "darwin":
            import subprocess
            out = subprocess.run(["sysctl", "-n", "hw.packages"], capture_output=True, text=True, timeout=5).stdout.strip()
            return (int(out) if out.isdigit() else 1), None
        import glob
        packages = set()
        for f in glob.glob(os.path.join(root, "cpu", "cpu[0-9]*", "topology", "physical_package_id")):
            try:
                with open(f, encoding="utf-8") as fh:
                    packages.add(fh.read().strip())
            except OSError:
                continue
        nodes = len(glob.glob(os.path.join(root, "node", "node[0-9]*")))
        return (len(packages) or None), (nodes or None)
    except Exception:                                        # noqa: BLE001 - a missing figure is shown as missing, never guessed
        return None, None


def _cpu_name():
    if os.name == "nt":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    elif os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return sysinfo.processor() or None


class _CpuRamFallback:
    """CPU load and RAM without psutil."""

    def __init__(self):
        self.prev = self._times()

    def _times(self):
        if os.name == "nt":
            idle, kern, user = (ctypes.c_ulonglong() for _ in range(3))
            if ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kern), ctypes.byref(user)):
                return idle.value, kern.value + user.value           # kernel time includes idle
            return None
        try:
            with open("/proc/stat") as fh:
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
        if os.name == "nt":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            m = MS()
            m.dwLength = ctypes.sizeof(MS)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                return m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys
            return None, None
        try:
            with open("/proc/meminfo") as fh:
                info = dict(line.split(":", 1) for line in fh)
            total = int(info["MemTotal"].split()[0]) * 1024
            avail = int(info["MemAvailable"].split()[0]) * 1024
            return total - avail, total
        except (OSError, KeyError, ValueError):
            return None, None


# ------------------------------------------------------------------------------------------------ the sampler
# usable bandwidth of one lane per PCIe generation, GB/s (after the line code): only the ORDER matters here
PCIE_LANE_GBS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}


def pcie_bottleneck(reads) -> dict | None:
    """The card whose PCIe link carries the least: lane rate of its generation x its width, all five fields from that ONE card (#5801 #5981),
    never a link put together from two cards.  The generation is the one the link can reach (pcie_gen_max; the current one when that is
    not reported), since NVML reports a lower one while a card is idle; the width is the negotiated one.  A card whose generation or
    width is unknown never wins.  None when no card reports both.  One card: that card."""
    best = None
    for index, r in reads:
        gen = r.get("pcie_gen_max") if r.get("pcie_gen_max") is not None else r.get("pcie_gen")
        width = r.get("pcie_width")
        if gen not in PCIE_LANE_GBS or not isinstance(width, (int, float)) or width <= 0:
            continue
        rate = PCIE_LANE_GBS[gen] * width
        if best is None or rate < best[0]:
            best = (rate, index, r)
    if best is None:
        return None
    _, index, r = best
    return {"card": index, "gen": r.get("pcie_gen"), "gen_max": r.get("pcie_gen_max"), "width": r.get("pcie_width"),
            "width_max": r.get("pcie_width_max")}


class Telemetry:
    def __init__(self, extra=None, gpu_index=0, gpu_indices=None, amd=False):
        """`extra()` -> dict of more series to record each second (the server's tok/s).  `gpu_index`: the card the
        engine runs on, numbered as nvidia-smi and NVML number them (by PCI bus); `gpu_indices`: all of them when
        the model is split across several (issue #112) - the gpu_* readings are then their total (memory, power,
        PCIe traffic), mean (load) or hottest (temperature), the PCIe link fields stay card 0's, gpu_pcie_bottleneck names the
        card with the slowest link (pcie_bottleneck), and "gpus" has each card's own, its link and its power limit included.  `amd`: the
        AMD backend's cards, numbered as HIP numbers them, read from sysfs (#301)."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        idx = list(gpu_indices) if gpu_indices and len(gpu_indices) > 1 else [gpu_index]
        self.gpus = [(i, monitor_reader(i, amd)) for i in idx]
        self.gpus = [(i, g) for i, g in self.gpus if g.ok()] or self.gpus[:1]
        self.gpu = self.gpus[0][1]
        try:
            import psutil  # noqa: F401
            self.ps = sys.modules["psutil"]
        except ImportError:
            self.ps = None
        self.fallback = _CpuRamFallback()
        gpu_name = " + ".join(g.name() or "?" for _, g in self.gpus) if self.gpu.ok() else None
        win_amd = None
        if gpu_name is None and amd and os.name == "nt":
            # #1587: no readings for an AMD card on Windows yet, but its name (and VRAM size) are in the registry, so the
            # dashboard and strata_bench do not call the run "CPU only".  The class order is taken as HIP's order.
            cards = windows_amd_cards()
            win_amd = [cards[i] for i in idx if 0 <= i < len(cards)] or cards[:1]
            gpu_name = " + ".join(c["name"] for c in win_amd) if win_amd else None
        # the macOS reader knows the device's working set (the VRAM budget of the unified memory); None elsewhere
        vram_total = None
        if win_amd is None and isinstance(self.gpu, _Metal) and self.gpu.ok():
            ws = metal_working_set()
            vram_total = ws >> 20 if ws else None
        self.static = {
            "gpu_name": gpu_name,
            "gpu_vram_total_mib": (sum(c["vram_bytes"] for c in win_amd) >> 20
                                   if win_amd and all(c["vram_bytes"] for c in win_amd) else vram_total),
            "gpu_count": len(self.gpus),
            # #1380: the AMD readings are the amdgpu driver's Linux sysfs files; a Windows AMD card has none yet, and the
            # dashboard said "not readable (NVML)" or showed empty tiles with no word why
            "gpu_note": ("no GPU load or VRAM readings for AMD cards on Windows yet (Linux reads them from the amdgpu "
                         "driver); the engine's own VRAM figures are in its log" if amd and not self.gpu.ok()
                         else getattr(self.gpu, "note", None)),
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "sockets": _cpu_topology()[0],
            "numa_nodes": _cpu_topology()[1],
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
                # A link is not additive.  The top-level gpu_pcie_* fields stay card 0's reading (their meaning since #112); the card whose
                # link is the slowest is named apart in gpu_pcie_bottleneck below, and each card's own link is in "gpus".
                s["gpus"] = [{"index": i, "name": reader.name(), "util": r.get("util"), "mem_used": r.get("mem_used"),
                              "mem_total": r.get("mem_total"), "temp": r.get("temp"), "power": r.get("power"),
                              "power_limit": r.get("power_limit"),
                              "pcie_gen": r.get("pcie_gen"), "pcie_gen_max": r.get("pcie_gen_max"),
                              "pcie_width": r.get("pcie_width"), "pcie_width_max": r.get("pcie_width_max"),
                              "pcie_rx_mb": r.get("pcie_rx_mb"), "pcie_tx_mb": r.get("pcie_tx_mb"),
                              "pcie_own_gen": r.get("pcie_own_gen"), "pcie_path": r.get("pcie_path")}
                             for (i, r), (_, reader) in zip(reads, self.gpus)]
            s.update({f"gpu_{k}": v for k, v in g.items()})
            s["gpu_pcie_bottleneck"] = pcie_bottleneck(reads)
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
        """Ends the sampler thread (a server that stops, a test's service): it used to run for the life of the process."""
        self._stop.set()

    def _loop(self):
        """One sample a second, for as long as the server lives.

        A failing sample must not end the thread: on a machine where the GPU/psutil source is not readable
        (a CI runner, a container without NVML) the first exception used to kill the sampler outright, so
        `GET /metrics` silently stopped updating and `test_responses.NoLeakedSampler` saw "the server did
        not start its hardware sampler" - the thread had started and died between two readings.
        """
        complained = False
        while not self._stop.is_set():
            try:
                s = self.sample()
            except Exception as e:                    # noqa: BLE001 - a monitor must outlive its source
                if not complained:                    # once, not once a second
                    print(f"[strata] metrics: a sample failed ({e}); the sampler keeps trying", flush=True)
                    complained = True
                self._stop.wait(1.0)
                continue
            with self.lock:
                self.now = s
                for k in ("gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                          "disk_read_mb", "tok_s", "prefill_tok_s_mean"):
                    v = s.get(k)
                    self.hist[k].append(round(v, 2) if isinstance(v, float) else v)
            self._stop.wait(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {k: list(v) for k, v in self.hist.items()},
                    "static": dict(self.static)}
