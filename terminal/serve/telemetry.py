"""serve/telemetry.py - hardware readings for the web app's Monitor tab (idea from PR #22 by code-martin).

A background thread samples once a second and keeps the last 60 readings of each series for the sparklines:
- GPU: NVIDIA's own NVML library (nvml.dll / libnvidia-ml.so.1, installed with every driver) through ctypes, so no
  pip package is needed: load, VRAM, temperature, power, PCIe link and throughput.  With the AMD backend (#301): the
  amdgpu driver's Linux sysfs files - load, VRAM, temperature and power; on Windows, for the Monitor, the driver's AMD Display
  Library (ADL2 PMLog, #4884).  On macOS (Metal backend): IOKit's
  IOAccelerator registry - load and VRAM; temperature and power from the HID sensors and IOReport's energy counter
  without root (AppleSensors, as Stats reads them; opt-in, only for the Monitor), and an Apple Silicon GPU sits on the die, so there is no PCIe link.
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
import struct
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
        if os.name != "nt" or sys.platform != "win32":      # (sys.platform too: a test that sets os.name to "nt" on Linux must not reach a Windows DLL load)
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
        except (OSError, TypeError, ValueError, AttributeError, ImportError):
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
        if buf[i].VendorID not in (0x1002, 1002) and b"VEN_1002" not in udid.upper():   # ADL can list another vendor's adapter (an NVIDIA card beside the iGPU)
            continue
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
    that name come before it in HIP's order); None when no row has that name (the Monitor then shows what it showed before
    rather than another card's readings).  Only when the registry lists no AMD card at all: row `index`."""
    if hip_names:
        if not 0 <= index < len(hip_names):
            # A run config can name a HIP device number that the registry list does not reach (HIP counts a Ryzen's integrated
            # Radeon, which the list leaves out: "gpu": 1 on a PC with an iGPU and one card, #5946).  With exactly one card in
            # the list and exactly one ADL row of that name the reading is unambiguous; with anything else nothing is guessed.
            if index > 0 and len(hip_names) == 1:
                only = [r for r, n in enumerate(adl_names) if n.strip().lower() == hip_names[0].strip().lower()]
                if len(only) == 1:
                    return only[0]
            return None
        want = hip_names[index].strip().lower()
        k = sum(1 for n in hip_names[:index] if n.strip().lower() == want)
        rows = [r for r, n in enumerate(adl_names) if n.strip().lower() == want]
        return rows[k] if k < len(rows) else None
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


class _SMCKeyData(ctypes.Structure):
    """AppleSMC's 80-byte call record (the SMCKeyData_t of Stats' SMC/smc.swift): key, then the command in data8."""
    class _Info(ctypes.Structure):
        _fields_ = [("size", ctypes.c_uint32), ("type", ctypes.c_uint32), ("attr", ctypes.c_uint8)]

    _fields_ = [("key", ctypes.c_uint32), ("vers", ctypes.c_uint8 * 6), ("plimit", ctypes.c_uint8 * 16),
                ("info", _Info), ("result", ctypes.c_uint8), ("status", ctypes.c_uint8), ("data8", ctypes.c_uint8),
                ("data32", ctypes.c_uint32), ("bytes", ctypes.c_uint8 * 32)]


# AppleSMC's GPU die temperature keys, by chip.  The rows are Stats' (exelban/stats, MIT: Modules/Sensors/values.swift at
# commit 032f4a5e4, the GPU temperature rows of each chip generation, in Stats' order); nothing here was worked out
# independently.  Only the M5's have been run, on an M5 Pro (idle readings also on an M5 Max); the rest are copied and have not been run on those chips.
# A key does not keep its meaning from chip to chip (Tg0j is GPU 2 on an M2, GPU 7 on an M4 and GPU 5 on an M5, and an
# M3's are Tf..), and on the M4 the base chip and the Pro/Max differ in their first two (SMC_GPU_TIER_KEYS), so the
# table is keyed by the chip and never guessed.  A key a chip variant does not have is skipped, and a chip that is not
# here reads none and falls back to the HID sensors.
SMC_GPU_KEYS = {
    "M1": ("Tg05", "Tg0D", "Tg0L", "Tg0T"),
    "M2": ("Tg0f", "Tg0j"),
    "M3": ("Tf14", "Tf18", "Tf19", "Tf1A", "Tf24", "Tf28", "Tf29", "Tf2A"),
    "M4": ("Tg0K", "Tg0L", "Tg0d", "Tg0e", "Tg0j", "Tg0k"),
    "M5": ("Tg0U", "Tg0X", "Tg0d", "Tg0g", "Tg0j", "Tg1Y", "Tg1c", "Tg1g"),
    "M6": ("Tg1e", "Tg0d", "Tg0c", "Tg0b", "Tg1c", "Tg1d", "Tg07", "Tg1f", "Tg09", "Tg0a", "Tg1g", "Tg1h"),
    "A18 Pro": ("Tg05", "Tg0D", "Tg0L", "Tg0e"),
}
SMC_GPU_TIER_KEYS = {("M4", ""): ("Tg0G", "Tg0H"), ("M4", "Pro"): ("Tg1U", "Tg1k"), ("M4", "Max"): ("Tg1U", "Tg1k")}


def smc_gpu_keys(brand):
    """The SMC keys of the GPU's die sensors for this CPU brand string ("Apple M4 Pro"), or None for a chip the table
    does not have (Intel, or an Apple chip newer than Stats' table)."""
    m = re.search(r"\bApple (M\d+|A\d+ Pro)(?: (Pro|Max|Ultra))?\b", brand or "")
    keys = SMC_GPU_KEYS.get(m.group(1)) if m else None
    if not keys:
        return None
    if m.group(1) == "M4":                          # the first two keys depend on the tier
        first = SMC_GPU_TIER_KEYS.get(("M4", m.group(2) or ""))
        return first + keys if first else None
    return keys


class _AppleNative:
    """The macOS calls behind the Apple Silicon GPU's temperature and power, made without root the way Stats
    (exelban/stats, MIT: Modules/Sensors/reader.m, readers.swift) makes them: IOHIDEventSystemClient's temperature
    sensors (Apple's usage page 0xff00, usage 5), AppleSMC's per-chip GPU temperature keys, and IOReport's "Energy
    Model" channels.  All are private API: every function is looked up by name, and a macOS without one leaves that
    reading empty instead of raising."""
    EV_TEMP = 15                                    # kIOHIDEventTypeTemperature
    UTF8 = 0x08000100                               # kCFStringEncodingUTF8
    SINT32 = 3                                      # kCFNumberSInt32Type
    SMC_INDEX, SMC_KEY_INFO, SMC_READ = 2, 9, 5     # the AppleSMC user-client selector, and its two commands
    SMC_FLT = int.from_bytes(b"flt ", "big")

    def __init__(self):
        self.hid = None                             # the HID client, None without the calls
        self.rep = None                             # (subscription, its channels), None without IOReport
        self.smc = None                             # (AppleSMC connection, this chip's GPU keys), None without them
        self._prev = None                           # the last IOReport sample, for the next delta
        try:
            self.cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
            self.io = ctypes.CDLL("/System/Library/Frameworks/IOKit.framework/IOKit")
        except OSError:
            return
        try:
            self._cf_init()
        except (AttributeError, ValueError):
            return
        for init in (self._hid_init, self._report_init, self._smc_init):
            try:
                init()
            except (OSError, AttributeError, ValueError):
                pass

    @staticmethod
    def _sig(lib, sigs):
        for name, res, args in sigs:
            f = getattr(lib, name)                  # AttributeError when this macOS has no such symbol
            f.restype, f.argtypes = res, args

    def _cf_init(self):
        vp, long_ = ctypes.c_void_p, ctypes.c_long
        self._sig(self.cf, [
            ("CFRelease", None, [vp]),
            ("CFStringCreateWithCString", vp, [vp, ctypes.c_char_p, ctypes.c_uint32]),
            ("CFStringGetCString", ctypes.c_bool, [vp, ctypes.c_char_p, long_, ctypes.c_uint32]),
            ("CFNumberCreate", vp, [vp, long_, vp]),
            ("CFDictionaryCreate", vp, [vp, vp, vp, long_, vp, vp]),
            ("CFDictionaryCreateMutableCopy", vp, [vp, long_, vp]),
            ("CFDictionaryGetCount", long_, [vp]),
            ("CFDictionaryGetValue", vp, [vp, vp]),
            ("CFArrayGetCount", long_, [vp]),
            ("CFArrayGetValueAtIndex", vp, [vp, long_])])

    def _str(self, s):
        return self.cf.CFStringCreateWithCString(None, s.encode(), self.UTF8)

    def _pystr(self, ref):
        buf = ctypes.create_string_buffer(256)
        return buf.value.decode("utf-8", "replace") if ref and self.cf.CFStringGetCString(ref, buf, 256, self.UTF8) else ""

    def _hid_init(self):
        vp = ctypes.c_void_p
        self._sig(self.io, [
            ("IOHIDEventSystemClientCreate", vp, [vp]),
            ("IOHIDEventSystemClientSetMatching", ctypes.c_int, [vp, vp]),
            ("IOHIDEventSystemClientCopyServices", vp, [vp]),
            ("IOHIDServiceClientCopyProperty", vp, [vp, vp]),
            ("IOHIDServiceClientCopyEvent", vp, [vp, ctypes.c_int64, ctypes.c_int32, ctypes.c_int64]),
            ("IOHIDEventGetFloatValue", ctypes.c_double, [vp, ctypes.c_int32])])
        cf = self.cf
        page, usage = ctypes.c_int32(0xFF00), ctypes.c_int32(5)
        keys = (vp * 2)(self._str("PrimaryUsagePage"), self._str("PrimaryUsage"))
        vals = (vp * 2)(cf.CFNumberCreate(None, self.SINT32, ctypes.byref(page)),
                        cf.CFNumberCreate(None, self.SINT32, ctypes.byref(usage)))
        kcb = ctypes.addressof(ctypes.c_char.in_dll(cf, "kCFTypeDictionaryKeyCallBacks"))
        vcb = ctypes.addressof(ctypes.c_char.in_dll(cf, "kCFTypeDictionaryValueCallBacks"))
        match = cf.CFDictionaryCreate(None, ctypes.cast(keys, vp), ctypes.cast(vals, vp), 2, kcb, vcb)
        for o in (*keys, *vals):                    # the dictionary keeps its own references
            if o:
                cf.CFRelease(o)
        if not match:
            return
        client = self.io.IOHIDEventSystemClientCreate(None)
        if client:
            self.io.IOHIDEventSystemClientSetMatching(client, match)
            self._product = self._str("Product")
            self.hid = client
        cf.CFRelease(match)

    def _report_init(self):
        vp = ctypes.c_void_p
        rep = ctypes.CDLL("/usr/lib/libIOReport.dylib")
        self._sig(rep, [
            ("IOReportCopyChannelsInGroup", vp, [vp, vp, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint64]),
            ("IOReportCreateSubscription", vp, [vp, vp, ctypes.POINTER(vp), ctypes.c_uint64, vp]),
            ("IOReportCreateSamples", vp, [vp, vp, vp]),
            ("IOReportCreateSamplesDelta", vp, [vp, vp, vp]),
            ("IOReportChannelGetChannelName", vp, [vp]),
            ("IOReportChannelGetUnitLabel", vp, [vp]),
            ("IOReportSimpleGetIntegerValue", ctypes.c_int64, [vp, ctypes.c_int32])])
        self.lib_rep = rep
        group = self._str("Energy Model")
        chan = rep.IOReportCopyChannelsInGroup(group, None, 0, 0, 0)
        self.cf.CFRelease(group)
        if not chan:
            return
        mut = self.cf.CFDictionaryCreateMutableCopy(None, self.cf.CFDictionaryGetCount(chan), chan)
        self.cf.CFRelease(chan)
        subbed = vp()
        sub = rep.IOReportCreateSubscription(None, mut, ctypes.byref(subbed), 0, None)
        if sub and subbed:
            self._channels_key = self._str("IOReportChannels")
            self.rep = (sub, subbed)

    def _smc_init(self):
        brand = ctypes.create_string_buffer(128)
        size = ctypes.c_size_t(len(brand))
        libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
        if libc.sysctlbyname(b"machdep.cpu.brand_string", brand, ctypes.byref(size), None, 0):
            return
        keys = smc_gpu_keys(brand.value.decode("utf-8", "replace"))
        if not keys:
            return
        vp, u32 = ctypes.c_void_p, ctypes.c_uint32
        self._sig(self.io, [
            ("IOServiceMatching", vp, [ctypes.c_char_p]),
            ("IOServiceGetMatchingService", u32, [u32, vp]),
            ("IOServiceOpen", ctypes.c_int, [u32, u32, u32, ctypes.POINTER(u32)]),
            ("IOObjectRelease", ctypes.c_int, [u32]),
            ("IOConnectCallStructMethod", ctypes.c_int,
             [u32, u32, vp, ctypes.c_size_t, vp, ctypes.POINTER(ctypes.c_size_t)])])
        dev = self.io.IOServiceGetMatchingService(0, self.io.IOServiceMatching(b"AppleSMC"))
        if not dev:
            return
        conn = u32()
        opened = self.io.IOServiceOpen(dev, u32.in_dll(libc, "mach_task_self_").value, 0, ctypes.byref(conn)) == 0
        self.io.IOObjectRelease(dev)
        if opened:
            self.smc = (conn.value, keys)

    def _smc_call(self, key, cmd, size=0):
        """The SMC's answer to one command on a key, or None when the key is not there (result 0x84) or the call fails."""
        i, o = _SMCKeyData(), _SMCKeyData()
        i.key, i.data8, i.info.size = int.from_bytes(key.encode(), "big"), cmd, size
        n = ctypes.c_size_t(ctypes.sizeof(o))
        r = self.io.IOConnectCallStructMethod(self.smc[0], self.SMC_INDEX, ctypes.byref(i), ctypes.sizeof(i),
                                              ctypes.byref(o), ctypes.byref(n))
        return o if r == 0 and o.result == 0 else None

    def gpu_smc(self):
        """[°C] of this chip's GPU SMC keys that exist (all four-byte floats, as on an M5); [] without them."""
        if not self.smc:
            return []
        out = []
        for key in self.smc[1]:
            info = self._smc_call(key, self.SMC_KEY_INFO)
            if not info or (info.info.type, info.info.size) != (self.SMC_FLT, 4):
                continue
            val = self._smc_call(key, self.SMC_READ, 4)
            if val:
                out.append(struct.unpack("<f", bytes(val.bytes[:4]))[0])
        return out

    def temps(self):
        """[(sensor name, °C)] of every HID temperature sensor; [] without the calls."""
        if not self.hid:
            return []
        cf, io = self.cf, self.io
        svcs = io.IOHIDEventSystemClientCopyServices(self.hid)
        if not svcs:
            return []
        out = []
        try:
            for i in range(cf.CFArrayGetCount(svcs)):
                s = cf.CFArrayGetValueAtIndex(svcs, i)
                name = io.IOHIDServiceClientCopyProperty(s, self._product)
                ev = io.IOHIDServiceClientCopyEvent(s, self.EV_TEMP, 0, 0)
                try:
                    if name and ev:
                        out.append((self._pystr(name), io.IOHIDEventGetFloatValue(ev, self.EV_TEMP << 16)))
                finally:
                    for o in (name, ev):
                        if o:
                            cf.CFRelease(o)
        finally:
            cf.CFRelease(svcs)
        return out

    def energy(self):
        """[(channel name, unit label, value)] of the Energy Model channels since the previous call; [] on the
        first call and without IOReport."""
        if not self.rep:
            return []
        cf, rep = self.cf, self.lib_rep
        cur = rep.IOReportCreateSamples(self.rep[0], self.rep[1], None)
        prev, self._prev = self._prev, cur
        if not (prev and cur):
            if prev:
                cf.CFRelease(prev)
            return []
        delta = rep.IOReportCreateSamplesDelta(prev, cur, None)
        cf.CFRelease(prev)
        if not delta:
            return []
        out = []
        try:
            arr = cf.CFDictionaryGetValue(delta, self._channels_key)   # not retained: owned by the delta
            for i in range(cf.CFArrayGetCount(arr) if arr else 0):
                ch = cf.CFArrayGetValueAtIndex(arr, i)
                out.append((self._pystr(rep.IOReportChannelGetChannelName(ch)),
                            self._pystr(rep.IOReportChannelGetUnitLabel(ch)),
                            rep.IOReportSimpleGetIntegerValue(ch, 0)))
        finally:
            cf.CFRelease(delta)
        return out


class AppleSensors:
    """The Apple Silicon GPU's temperature and power from _AppleNative's raw readings (a fake one in the tests).

    Temperature: the mean of the chip's SMC GPU keys (Stats' per-chip TgXX table, which is where an M5 has them: its
    HID service names no GPU sensor), else of the HID sensors named "GPU MTR Temp..."; readings outside 0-110 °C are
    dropped as Stats does.  A chip with neither shows no temperature.
    Power: the "GPU Energy" channel's energy since the last reading over the time since it.  Opt-in:
    STRATA_MACOS_SENSORS=1."""
    GPU_PREFIX = "GPU MTR Temp"
    JOULES = {"mJ": 1e-3, "uJ": 1e-6, "µJ": 1e-6, "nJ": 1e-9}

    def __init__(self, native=None, clock=time.monotonic):
        self.native = native if native is not None else _AppleNative()
        self.clock = clock
        self._t = None
        self.energy()                               # the first delta's baseline

    def gpu_temp(self):
        v = [t for t in getattr(self.native, "gpu_smc", list)() if 0 < t < 110]   # 0: the SMC's "no data"
        if not v:
            v = [t for name, t in self.native.temps() if name.startswith(self.GPU_PREFIX) and 0 <= t < 110]
        return round(sum(v) / len(v), 1) if v else None

    def energy(self):
        """Joules the GPU used since the previous call, and the seconds since it; (None, None) on the first call."""
        rows, now = self.native.energy(), self.clock()
        prev, self._t = self._t, now
        j = [v * self.JOULES[u] for name, u, v in rows if name.endswith("GPU Energy") and u in self.JOULES]
        return (sum(j), now - prev) if j and prev is not None and now > prev else (None, None)

    def gpu_power(self):
        j, dt = self.energy()
        return round(j / dt, 2) if j is not None else None

    def available(self):
        """Whether either reading can be there: a GPU temperature now, or IOReport's subscription (its first power
        reading needs a second sample)."""
        return self.gpu_temp() is not None or bool(getattr(self.native, "rep", None))


def apple_sensors():
    """AppleSensors on macOS when STRATA_MACOS_SENSORS=1, else None: the sensors are private API, so they are opt-in and
    a Mac's Monitor is what it was without the variable."""
    if sys.platform != "darwin" or os.environ.get("STRATA_MACOS_SENSORS") != "1":
        return None
    return AppleSensors()


class _Metal:
    """macOS: an Apple GPU's readings from the IOKit registry (`ioreg -l -w0 -r -c IOAccelerator`, 22 ms), with
    _Nvml's interface: the load and the memory are the driver's PerformanceStatistics of the index'th accelerator
    node - Device Utilization % and the in-use system memory in bytes.  The total is the device's Metal working set,
    the budget one app gets of the unified memory.  The temperature and the power come from AppleSensors when
    gpu_reader passes them (macOS's private sensor calls, no root); without them the note keeps the old words.  An
    Apple Silicon GPU sits on the die, so the PCIe cards have nothing to show."""
    note = ("no GPU temperature or power reading without root (powermetrics), and an Apple Silicon GPU sits on the "
            "die, so there is no PCIe link to show; the engine's own figures are in its log")
    cmd = ("ioreg", "-l", "-w0", "-r", "-c", "IOAccelerator")

    _first: list = []                       # (M2 11-16) the registry as the first reader of this process saw it: the nodes and the name do not change

    def __init__(self, index=0, run=None, sensors=None):
        self.index = index
        self._run = run or self._ioreg
        self.sensors = None
        if sensors is not None:
            self.attach_sensors(sensors)
        if run is None:                     # the real registry: read once per process for what does not change (a plan asked it six times)
            if not _Metal._first:
                _Metal._first.append(self._ioreg())
            text = _Metal._first[0]
            blocks = self._blocks(text)
        else:
            blocks = self._blocks()
        self.nodes = len(blocks)
        self._name = None
        b = blocks[index] if 0 <= index < len(blocks) else None
        if b:
            m = re.search(r'"model" = "([^"]+)"', b)
            self._name = m.group(1) if m else None

    _recent: list = []                      # (M2) [monotonic time, text]: one registry read serves the questions of the same moment
    FRESH_S = 1.0                           # the server's telemetry polls every few seconds, so its readings stay live

    @staticmethod
    def _ioreg():
        now = time.monotonic()
        if _Metal._recent and now - _Metal._recent[0][0] < _Metal.FRESH_S:
            return _Metal._recent[0][1]
        try:
            text = subprocess.run(list(_Metal.cmd), capture_output=True, stdin=subprocess.DEVNULL, text=True, timeout=10).stdout or ""
        except (OSError, subprocess.SubprocessError):
            text = ""
        _Metal._recent[:] = [(time.monotonic(), text)]
        return text

    def attach_sensors(self, sensors):
        """AppleSensors for this reader (monitor_reader hands them over; the server's own readers never get them), or None."""
        self.sensors = sensors
        if sensors is not None and sensors.available():
            self.note = ("GPU temperature and power from macOS's sensors without root (STRATA_MACOS_SENSORS=1 "
                         "turned them on); an Apple Silicon GPU sits on the die, so there is no PCIe link to show")

    def _blocks(self, text=None):
        """The accelerator nodes, in the registry's order: every node that carries a PerformanceStatistics. `text`: a registry already read."""
        try:
            out = (self._run() if text is None else text) or ""
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
        if self.sensors is not None:
            out["temp"] = self.sensors.gpu_temp()
            out["power"] = self.sensors.gpu_power()
        return out


# ------------------------------------------------------------------------------------------------ Intel Arc
#: the cards setup names by PCI id (sycl/setup_intel.py INTEL_ARC; the same table as sycl/serve/xe_telemetry.py)
ARC_NAMES = {"e223": "Arc Pro B70", "e221": "Arc Pro B60", "e211": "Arc Pro B60", "e20b": "Arc B580", "e20c": "Arc B570",
             "e212": "Arc B50", "56a0": "Arc A770", "56a1": "Arc A750", "56a2": "Arc A580", "56a5": "Arc A380",
             "56a6": "Arc A310", "5690": "Arc A770M"}


class _ZesPciAddress(ctypes.Structure):
    _fields_ = [("domain", ctypes.c_uint32), ("bus", ctypes.c_uint32), ("device", ctypes.c_uint32), ("function", ctypes.c_uint32)]


class _ZesPciSpeed(ctypes.Structure):
    _fields_ = [("gen", ctypes.c_int32), ("width", ctypes.c_int32), ("max_bandwidth", ctypes.c_int64)]


class _ZesPciProps(ctypes.Structure):    # zes_pci_properties_t
    _fields_ = [("stype", ctypes.c_uint32), ("pnext", ctypes.c_void_p), ("address", _ZesPciAddress), ("max_speed", _ZesPciSpeed),
                ("bw", ctypes.c_uint8), ("pkt", ctypes.c_uint8), ("replay", ctypes.c_uint8)]


class _ZesMemState(ctypes.Structure):    # zes_mem_state_t
    _fields_ = [("stype", ctypes.c_uint32), ("pnext", ctypes.c_void_p), ("health", ctypes.c_int32), ("free", ctypes.c_uint64),
                ("size", ctypes.c_uint64)]


class _ZesPciStats(ctypes.Structure):    # zes_pci_stats_t: monotonic counters, timestamp in microseconds
    _fields_ = [("timestamp", ctypes.c_uint64), ("replayCounter", ctypes.c_uint64), ("packetCounter", ctypes.c_uint64),
                ("rxCounter", ctypes.c_uint64), ("txCounter", ctypes.c_uint64), ("speed", _ZesPciSpeed)]


class _Zes:
    """#5988 (PR #5994, Rob Riggs): an Intel card's VRAM and PCIe counters through Level Zero's management API (sysman,
    libze_loader.so.1, installed with the GPU runtime), matched to the card by its PCI address so a PC with two Intel cards reports
    the one asked for. Runs as the calling user (xpu-smi stats reads the same API without root). The PCIe counters come from Intel
    PMT, whose sysfs node is root-only unless the host grants it: then sysman refuses and the reading is None, never a zero."""

    STYPE_PCI_PROPERTIES, STYPE_MEM_STATE = 0x2, 0x1E

    def __init__(self, pci, lib=None):
        self.lib, self.mems, self.dev = None, [], None
        try:
            self.lib = lib or ctypes.CDLL("libze_loader.so.1")
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
                    self.dev = ctypes.c_void_p(dev)
                    k = ctypes.c_uint32(0)
                    if self.lib.zesDeviceEnumMemoryModules(ctypes.c_void_p(dev), ctypes.byref(k), None) != 0:
                        continue
                    mods = (ctypes.c_void_p * k.value)()
                    self.lib.zesDeviceEnumMemoryModules(ctypes.c_void_p(dev), ctypes.byref(k), mods)
                    self.mems = list(mods)
        except (OSError, AttributeError, TypeError):
            self.mems = []

    def read(self):
        """(used, total) bytes over the card's memory modules, or (None, None) when sysman cannot say."""
        free = total = 0
        if self.lib is None or not self.mems:
            return None, None
        for mod in self.mems:
            st = _ZesMemState(stype=self.STYPE_MEM_STATE)
            try:
                if self.lib.zesMemoryGetState(ctypes.c_void_p(mod), ctypes.byref(st)) != 0:
                    return None, None
            except (OSError, AttributeError, TypeError):
                return None, None
            free += st.free
            total += st.size
        return (total - free, total) if total else (None, None)

    def integrated(self):
        """True for a GPU built into the processor, False for a card, None when sysman cannot say: zesDeviceGetProperties'
        core properties (ze_device_properties_t) carry ZE_DEVICE_PROPERTY_FLAG_INTEGRATED. Read from a raw buffer: stype
        0x1 at 0, the core's stype 0x3 at 16, its vendorId at 36 (checked to be Intel's, else the layout is not this one)
        and its flags at 44."""
        if self.lib is None or self.dev is None:
            return None
        buf = ctypes.create_string_buffer(4096)
        ctypes.c_uint32.from_buffer(buf, 0).value = 0x1           # ZES_STRUCTURE_TYPE_DEVICE_PROPERTIES
        ctypes.c_uint32.from_buffer(buf, 16).value = 0x3          # ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES
        try:
            if self.lib.zesDeviceGetProperties(self.dev, buf) != 0:
                return None
        except (OSError, AttributeError, TypeError):
            return None
        if ctypes.c_uint32.from_buffer(buf, 36).value != 0x8086:
            return None
        return bool(ctypes.c_uint32.from_buffer(buf, 44).value & 0x1)

    def pci_stats(self):
        """(rx bytes, tx bytes, timestamp in us) of the card's PCIe link, or None (sysman refused or has none)."""
        if self.lib is None or self.dev is None:
            return None
        st = _ZesPciStats()
        try:
            if self.lib.zesDevicePciGetStats(self.dev, ctypes.byref(st)) != 0:
                return None
        except (OSError, AttributeError, TypeError):
            return None
        return st.rxCounter, st.txCounter, st.timestamp


class _Intel:
    """#5353: an Intel Arc card's readings from sysfs, for every route that starts serve/server.py (the packaged Terminal and app,
    the container), not only sycl/serve/server_intel.py. The discrete Intel cards in sysfs order (as setup numbers them): the xe
    driver's (a tile0 folder) and the i915 driver's (an Arc A-series has a hwmon; an integrated Intel GPU has neither, so a UHD
    iGPU beside the card is not taken for it). Temperature (xe's "pkg" sensor, else temp1), power (from the energy counter between
    two readings), power limit and the PCIe link are world-readable sysfs. #5988 (PR #5994, Rob Riggs): VRAM and PCIe traffic from
    Level Zero sysman (_Zes), and the load of the GPU clients this user can see in /proc/<pid>/fdinfo (the engine the server
    started: xe counts cycles, i915 nanoseconds), so no root and no helper service are needed. The card-wide load across other
    users' processes would need root: this is the engine's own load. A fresh /run/gpustat.json from the optional root sampler
    (sycl/tools/gpustat.py) still wins for load and VRAM. A reading no source gives is None, never a zero."""
    STAT = "/run/gpustat.json"

    def __init__(self, index=0, sysfs=None, stat=None, proc=None, zes=None):
        import glob
        root = sysfs or "/sys"
        self._proc = proc or "/proc"            # (attribute names as sycl/serve/xe_telemetry.py had them: tools/test_xe_telemetry.py sets them)
        self._stat_path = stat                  # None: STAT, read when used, so an instance can point STAT elsewhere
        self.dev = None
        cards = []
        for d in sorted(glob.glob(f"{root}/class/drm/card[0-9]*/device"), key=lambda p: int(re.sub(r"\D", "", p.split("/card")[-1]) or 0)):
            try:
                with open(f"{d}/vendor") as f:
                    if f.read().strip() != "0x8086":
                        continue
            except OSError:
                continue
            if os.path.isdir(f"{d}/tile0") or glob.glob(f"{d}/hwmon/hwmon*/name"):
                cards.append(os.path.realpath(d))
        if 0 <= index < len(cards):
            self.dev = cards[index]
        self.hwmon = None
        for n in sorted(glob.glob(f"{self.dev}/hwmon/hwmon*/name")) if self.dev else []:
            self.hwmon = os.path.dirname(n)
        self._e = None                          # (time, energy in uJ) of the last reading
        self.pci = os.path.basename(self.dev) if self.dev else ""   # "0000:03:00.0", as fdinfo's drm-pdev names it
        self.zes = zes if zes is not None else (_Zes(self.pci) if self.dev and sys.platform.startswith("linux") else None)
        self._busy_prev = {}                    # client id -> (busy, total) of the last read
        self._fd_mem = None                     # bytes the DRM clients this user can see hold on the card (fdinfo), last read
        self.driver = ("xe" if os.path.isdir(f"{self.dev}/tile0") else "i915") if self.dev else None
        self._pci = None                        # (rx bytes, tx bytes, ts us) of the last read

    def ok(self):
        return self.dev is not None and sys.platform.startswith("linux")

    @staticmethod
    def _rd(p):
        try:
            with open(p) as f:
                return f.read().strip()
        except OSError:
            return None

    def _stat(self):
        try:
            import json
            with open(self._stat_path or self.STAT) as f:
                st = json.load(f)
            return st if time.time() - float(st.get("ts", 0)) < 15 else None
        except (OSError, ValueError, TypeError):
            return None

    def name(self):
        st = self._stat()
        if (st or {}).get("name"):
            return st["name"]
        pid = (self._rd(f"{self.dev}/device") or "").lower().replace("0x", "")
        return "Intel " + ARC_NAMES.get(pid, "Arc GPU")

    def _link(self):
        """The link the card trained at: its own functions sit behind an internal x1 switch, so the first port up the path wider
        than x1; the max is what card AND slot allow (the root port caps it)."""
        gen = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
        d, chain = self.dev, []
        while d and os.path.exists(f"{d}/current_link_speed"):
            chain.append(d)
            d = os.path.dirname(d)
        for p in chain:
            w = int(self._rd(f"{p}/current_link_width") or 0)
            if w > 1:
                g = lambda q, k: gen.get((self._rd(f"{q}/{k}_link_speed") or "").split(" ")[0])
                card, slot = g(p, "max"), g(chain[-1], "max")
                return g(p, "current"), min(x for x in (card, slot) if x) if (card or slot) else None, w
        return None, None, None

    def _busy(self):
        """#5988 (PR #5994): the load of the DRM clients on this card that this user can read in <proc>/<pid>/fdinfo, as a
        percentage between two reads. xe: render + compute cycles (per engine capacity) over the total cycles; i915: render +
        compute busy nanoseconds over the wall time. Only the card's own driver's clients count (one unit per reading). A client
        counts when it has an earlier reading and both of its counters moved forward; when no client does (all new, a reset, none),
        the load is unknown (None), never 0 % (GUI-1116's rules for #5994, 0f6994f52). A client open in several descriptors of
        one process counts once; an engine class with capacity 0 is left out."""
        import glob
        now, wall, mem, drivers = {}, time.monotonic_ns(), {}, {}
        for f in glob.glob(f"{self._proc}/[0-9]*/fdinfo/*"):
            try:
                with open(f, encoding="ascii", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if f"drm-pdev:\t{self.pci}" not in text:
                continue
            kv = {}
            for line in text.splitlines():
                k, _, v = line.partition(":")
                kv.setdefault(k, v.strip())
            cid, drv = kv.get("drm-client-id"), kv.get("drm-driver")
            try:
                if drv == "xe":
                    busy = 0.0
                    for e in ("rcs", "ccs"):
                        cap = int(kv.get(f"drm-engine-capacity-{e}") or 1)
                        if cap > 0:
                            busy += int(kv.get(f"drm-cycles-{e}", "0")) / cap
                    total = int(kv.get("drm-total-cycles-ccs") or kv.get("drm-total-cycles-rcs") or 0)
                elif drv == "i915":
                    busy = sum(int((kv.get(f"drm-engine-{e}") or "0 ns").split()[0]) for e in ("render", "compute"))
                    total = wall
                else:
                    continue
            except (ValueError, IndexError):
                continue
            if not cid:
                continue
            # one client per (process, client id), however many descriptors hold it: the process is the folder above fdinfo, read
            # with os.path (glob joins with the platform's separator, so a split on "/fdinfo/" kept the descriptor on Windows)
            key = (os.path.basename(os.path.dirname(os.path.dirname(f))), cid)
            drivers[key] = drv
            if total:
                now[key] = (busy, total)
            for k, v in kv.items():   # the client's memory on this card (i915: drm-resident-local0, xe: drm-resident-vram0), "N KiB"
                if re.fullmatch(r"drm-resident-(local|vram)\d+", k):
                    parts = v.split()
                    if parts and parts[0].isdigit():
                        unit = {"KiB": 2**10, "MiB": 2**20, "GiB": 2**30}.get(parts[1] if len(parts) > 1 else "", 1)
                        mem[key + (k,)] = int(parts[0]) * unit
        # one driver per card; a reader without its card (a test) takes xe's clients when there are any
        want = self.driver or ("xe" if "xe" in drivers.values() else "i915")
        now = {c: v for c, v in now.items() if drivers.get(c) == want}
        mem = {c: v for c, v in mem.items() if drivers.get(c[:2]) == want}
        self._fd_mem = sum(mem.values()) if mem else None
        prev, self._busy_prev = self._busy_prev, now
        moved = [c for c in now if c in prev and now[c][1] > prev[c][1] and now[c][0] >= prev[c][0]]
        if not moved:
            return None
        used = sum(now[c][0] - prev[c][0] for c in moved)
        return max(0.0, min(100.0, 100.0 * used / max(now[c][1] - prev[c][1] for c in moved)))

    def read(self):
        out = {"util": None, "mem_used": None, "mem_total": None, "temp": None, "power": None, "power_limit": None}
        st = self._stat()
        if st:
            out["util"] = st.get("busy_pct")
            if st.get("vram_used_mb") is not None:
                out["mem_used"] = st["vram_used_mb"] * 2**20
                out["mem_total"] = st.get("vram_total_mb", 0) * 2**20 or None
            out["temp"] = st.get("temp_pkg")
            out["power"] = st.get("power_w")
            out["power_limit"] = st.get("power_cap_w")
        busy = self._busy()                     # read every time, so the next read has a span to compare
        if out["util"] is None:
            out["util"] = busy
        if out["mem_used"] is None:
            used, total = self.zes.read() if self.zes is not None else (None, None)
            if self.driver == "i915":
                # sysman on i915 (Arc A-series) reports the card's size but no use: free == size, so "used 0" while an engine holds
                # gigabytes (measured on an A750, p1006 11-16). The engine's own DRM clients' resident memory is the reading there.
                used = self._fd_mem
            out["mem_used"], out["mem_total"] = used, total
        if self.hwmon:
            if out["temp"] is None:
                files = os.listdir(self.hwmon)
                # temp*_label only: an xe card labels its energy counter "pkg" as well (energy2_label, microjoules; #6243)
                src = next((lab[:-6] for lab in files if lab.startswith("temp") and lab.endswith("_label") and self._rd(f"{self.hwmon}/{lab}") == "pkg"), None)
                v = self._rd(f"{self.hwmon}/{src or 'temp1'}_input")
                out["temp"] = int(v) / 1000 if v and v.lstrip("-").isdigit() else None
            if out["power"] is None:
                e, t = self._rd(f"{self.hwmon}/energy1_input"), time.time()
                if e and e.isdigit():
                    if self._e and t > self._e[0] and int(e) >= self._e[1]:
                        out["power"] = (int(e) - self._e[1]) / 1e6 / (t - self._e[0])
                    self._e = (t, int(e))
            if out["power_limit"] is None:
                for cap in ("power1_cap", "power1_max"):   # xe / i915
                    v = self._rd(f"{self.hwmon}/{cap}")
                    if v and v.isdigit() and int(v) > 0:
                        out["power_limit"] = int(v) / 1e6
                        break
        out["pcie_gen"], out["pcie_gen_max"], out["pcie_width"] = self._link()
        out["pcie_rx_mb"] = out["pcie_tx_mb"] = None    # sysfs has no PCIe traffic counters; sysman has them when the host allows
        st = self.zes.pci_stats() if self.zes is not None else None
        prev, self._pci = self._pci, st
        if st and prev and st[2] > prev[2] and st[0] >= prev[0] and st[1] >= prev[1]:
            dt = (st[2] - prev[2]) / 1e6
            out["pcie_rx_mb"] = (st[0] - prev[0]) / dt / 2**20
            out["pcie_tx_mb"] = (st[1] - prev[1]) / dt / 2**20
        return out


def gpu_reader(index=0, amd=False):
    """The card's readings: NVML (NVIDIA), the amdgpu sysfs files with the AMD backend (#301), or IOKit's
    IOAccelerator registry on macOS (Metal backend).  `amd` comes first: no Mac runs the HIP backend, and the
    tests' AMD trees are read on a Mac too.  On Linux without NVML, an Intel Arc card's sysfs readings (#5353)."""
    if amd:
        return _Amd(index)
    if sys.platform == "darwin":
        return _Metal(index)
    nv = _Nvml(index)
    if not nv.ok() and sys.platform.startswith("linux"):
        intel = _Intel(index)
        if intel.ok():
            return intel
    return nv


def monitor_reader(index=0, amd=False):
    """The Monitor's reader (Telemetry): gpu_reader's, except that with the AMD backend on Windows, where the amdgpu sysfs reader
    has nothing to read, the driver's ADL2 (#4884), and on a Mac the Apple Silicon temperature and power when STRATA_MACOS_SENSORS=1
    (AppleSensors: private calls, so only here, never in the server's VRAM checks or the drafter's sizing).  Every other caller keeps
    gpu_reader, so their decisions are unchanged."""
    g = gpu_reader(index, amd)
    if sys.platform == "darwin" and not amd and hasattr(g, "attach_sensors"):
        sensors = apple_sensors()                    # None unless STRATA_MACOS_SENSORS=1
        if sensors is not None:
            g.attach_sensors(sensors)
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


_DX_ADAPTERS = r"SOFTWARE\Microsoft\DirectX"   # the adapter list Windows and Task Manager use: Description, VendorId, AdapterLuid


def _norm_card(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _dx_amd_absent() -> set:
    """Normalized Descriptions of AMD adapters the DirectX adapter list knows but reports as not plugged in now
    (AdapterLuid 0; the app's own host reads the same key).  The display class keeps the driver key of a card that
    was taken out (#3892): measured on a 7900 XTX PC, a removed RX 9070 XT's key sat between the iGPU and the card
    and took a Monitor row.  A name counts as absent only when DirectX lists it with no live twin, so a name the
    list does not know at all is never filtered."""
    if os.name != "nt":
        return set()
    try:
        import winreg
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DX_ADAPTERS)
    except OSError:
        return set()
    live, gone = set(), set()
    i = 0
    while True:
        try:
            sub = winreg.EnumKey(root, i)
        except OSError:
            break
        i += 1
        try:
            k = winreg.OpenKey(root, sub)
            desc = str(winreg.QueryValueEx(k, "Description")[0] or "")
            vid = int(winreg.QueryValueEx(k, "VendorId")[0])
            luid = winreg.QueryValueEx(k, "AdapterLuid")[0]
        except (OSError, TypeError, ValueError):
            continue
        if isinstance(luid, (bytes, bytearray)):        # REG_QWORD or REG_BINARY; a Luid of 0 means the card is not there
            luid = int.from_bytes(bytes(luid[:8]), "little")
        if vid == 0x1002:
            (live if luid else gone).add(_norm_card(desc))
    return gone - live


def windows_amd_cards(adapters=None) -> list:
    """The AMD cards HIP runs on, as windows_display_adapters lists them.  A Ryzen's integrated Radeon is listed too,
    and size cannot tell it apart: 512 MiB on a Ryzen 5 7600, but exactly 2 GiB on a board carrying a 7900 XTX
    (gfx1036, measured) and 43 GiB on a Strix Halo - so like _adl_cards the driver name decides, an integrated Radeon
    being the one called "... Graphics"; an APU alone keeps it.  Driver keys of cards no longer plugged in (#3892)
    are left out when the DirectX adapter list reports that exact name as absent."""
    cards = windows_display_adapters("amd") if adapters is None else adapters
    gone = _dx_amd_absent()
    if gone:
        cards = [c for c in cards if _norm_card(c["name"]) not in gone] or cards
    discrete = [c for c in cards if "graphics" not in c["name"].lower()]
    return discrete or cards


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
            out = subprocess.run(["sysctl", "-n", "hw.packages"], capture_output=True, stdin=subprocess.DEVNULL, text=True, timeout=5).stdout.strip()
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
