"""Intel Arc (xe driver) readings for the server's Monitor tab, in the shape of serve/telemetry.py's readers.

Kept out of serve/telemetry.py so the SYCL port touches no shared file: sycl/serve/server_intel.py installs it as
telemetry.gpu_reader when there is no NVIDIA card.  Every reading is taken inside this process, with no root and no
helper service running outside it:

- VRAM: Level Zero's management API (sysman, libze_loader.so.1, which the GPU runtime installs) through ctypes.  sysfs
  gives the xe driver no memory counters at all, and this needs no privilege: `xpu-smi stats` reads the same API as a
  normal user.
- Load: the DRM clients this process can see in /proc/<pid>/fdinfo, counted as engine cycles over total cycles.  xe
  keeps one fdinfo entry per client, so a container sees the engine it runs and nothing else.  The card-wide load
  across other users' processes would need root; this reading says so in the Monitor instead of guessing.
- Temperature, power and the PCIe link: sysfs (hwmon and the PCI path), which everyone can read.
- PCIe traffic: sysman's PCIe counters, which Intel PMT backs with /sys/class/intel_pmt/telem*/telem. That node
  ships 0440, so the reading needs a udev rule granting it to a group the caller is in (`render`), and a container
  needs --group-add=keep-groups to carry that group across the user namespace. Without it sysman answers
  INSUFFICIENT_PERMISSIONS and this reports None, never a zero.

A reading that no source can give is None, never a zero: the Monitor shows a dash rather than a made-up number.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time


# the cards sycl/setup_intel.py names by PCI id (INTEL_ARC)
ARC_NAMES = {"e223": "Arc Pro B70", "e221": "Arc Pro B60", "e211": "Arc Pro B60", "e20b": "Arc B580", "e20c": "Arc B570",
             "e212": "Arc B50", "56a0": "Arc A770", "56a1": "Arc A750", "56a2": "Arc A580", "56a5": "Arc A380",
             "56a6": "Arc A310", "5690": "Arc A770M"}


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


class _ZesSpeed(ctypes.Structure):       # zes_pci_speed_t
    _fields_ = [("gen", ctypes.c_int32), ("width", ctypes.c_int32), ("maxBandwidth", ctypes.c_int64)]


class _ZesPciStats(ctypes.Structure):    # zes_pci_stats_t: monotonic counters, timestamp in microseconds
    _fields_ = [("timestamp", ctypes.c_uint64), ("replayCounter", ctypes.c_uint64), ("packetCounter", ctypes.c_uint64),
                ("rxCounter", ctypes.c_uint64), ("txCounter", ctypes.c_uint64), ("speed", _ZesSpeed)]


class _Zes:
    """The card's VRAM through Level Zero's management API (sysman), matched to the card by its PCI address, so a
    machine with two Intel cards reports the one this reader was asked for.  Runs as the calling user."""

    STYPE_PCI_PROPERTIES, STYPE_MEM_STATE = 0x2, 0x1E

    def __init__(self, pci):
        self.lib, self.mems, self.dev = None, [], None
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
                    self.dev = ctypes.c_void_p(dev)
                    k = ctypes.c_uint32(0)
                    if self.lib.zesDeviceEnumMemoryModules(ctypes.c_void_p(dev), ctypes.byref(k), None) != 0:
                        continue
                    mods = (ctypes.c_void_p * k.value)()
                    self.lib.zesDeviceEnumMemoryModules(ctypes.c_void_p(dev), ctypes.byref(k), mods)
                    self.mems = list(mods)
        except (OSError, AttributeError):
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
            except (OSError, AttributeError):
                return None, None
            free += st.free
            total += st.size
        return (total - free, total) if total else (None, None)

    def pci_stats(self):
        """(rx bytes, tx bytes, timestamp us) of the card's PCIe link, or None.

        The counters come from Intel PMT telemetry, which ships /sys/class/intel_pmt/telem*/telem as 0440; when that
        file is group-readable by the caller (a udev rule granting it to `render`, plus --group-add=keep-groups in
        the container) sysman returns them, and otherwise the call fails with INSUFFICIENT_PERMISSIONS and this
        returns None, so the Monitor shows a dash instead of a flat zero."""
        if self.lib is None or self.dev is None:
            return None
        st = _ZesPciStats()
        try:
            if self.lib.zesDevicePciGetStats(self.dev, ctypes.byref(st)) != 0:
                return None
        except (OSError, AttributeError):
            return None
        return st.rxCounter, st.txCounter, st.timestamp


class _XeGpu:
    """The Intel Arc readings in NVML's shape: VRAM from sysman, load from the DRM clients this process can see,
    temperature, power and the PCIe link from sysfs.  No root, no sampler outside the process."""

    STAT = "/run/gpustat.json"               # the old root sampler (a fallback only, #5994 review)

    def __init__(self, index=0):
        import glob
        self.dev = self.hwmon = self.zes = None
        cards = []                              # #2001: the Intel cards in sysfs order, as setup numbers them
        for d in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
            try:
                if open(f"{d}/vendor").read().strip() == "0x8086" and os.path.isdir(f"{d}/tile0"):
                    cards.append(os.path.realpath(d))
            except OSError:
                continue
        if 0 <= index < len(cards):
            self.dev = cards[index]
        for n in glob.glob(f"{self.dev}/hwmon/hwmon*/name") if self.dev else []:
            self.hwmon = os.path.dirname(n)
        self.pci = os.path.basename(self.dev) if self.dev else ""   # "0000:03:00.0", as fdinfo names it
        if self.dev:
            self.zes = _Zes(self.pci)
        self._cycles = {}                       # client id -> (busy cycles, total cycles) of the last read
        self._proc = "/proc"                    # where the DRM clients' fdinfo is read (tests point it at a fixture)
        self._e = None                          # (t, card energy uJ) for power
        self._pci = None                        # (rx bytes, tx bytes, ts us) of the last read, for the traffic rate

    def ok(self):
        return self.dev is not None and sys.platform.startswith("linux")

    def name(self):
        try:                                    # the PCI id's name (setup_intel.INTEL_ARC), not a generic one
            with open(f"{self.dev}/device") as f:
                return "Intel " + ARC_NAMES.get(f.read().strip().lower().replace("0x", ""), "Arc GPU")
        except (OSError, TypeError):
            return "Intel Arc GPU"

    def _rd(self, p):
        try:
            return open(p).read().strip()
        except OSError:
            return None

    def _link(self):
        """The link the card trained at: its own functions sit behind an internal x1 switch, so the first port up
        the path wider than x1; the max is what card AND slot allow (the root port caps it)."""
        gen = {"2.5": 1, "5.0": 2, "8.0": 3, "16.0": 4, "32.0": 5, "64.0": 6}
        d, chain = self.dev, []
        while d and d.startswith("/sys/devices/pci") and os.path.exists(f"{d}/current_link_speed"):
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
        """The render and compute load of the DRM clients this process can read in <proc>/<pid>/fdinfo, as a percentage of
        the cycles the card ran.  Scope: the clients this user can see (in a container that is the engine; on a host it
        can include other programs of the same user), never the whole card, which needs other users' fdinfo.
        Unknown (None) until a client has two readings whose counters moved forward: a client seen for the first time,
        or one whose counters went back (a reset, a reused id), says nothing about the load yet and is never taken as
        idle (#5994 review).  A client open in several file descriptors is counted once."""
        import glob
        now = {}
        for f in glob.glob(os.path.join(self._proc, "[0-9]*", "fdinfo", "*")):
            try:
                with open(f, encoding="ascii", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            if "drm-driver:\txe\n" not in text or f"drm-pdev:\t{self.pci}" not in text:
                continue
            kv = {}
            for line in text.splitlines():
                k, _, v = line.partition(":")
                kv.setdefault(k, v.strip())
            try:    # a class's cycles count all its engines (drm-engine-capacity-*, 1 when not given)
                busy = sum(int(kv.get(f"drm-cycles-{e}", "0")) / int(kv.get(f"drm-engine-capacity-{e}", "1") or 1)
                           for e in ("rcs", "ccs"))
                total = int(kv.get("drm-total-cycles-ccs") or kv.get("drm-total-cycles-rcs") or 0)
            except (ValueError, IndexError, ZeroDivisionError):
                continue
            cid = kv.get("drm-client-id")
            if cid and total:
                now[cid] = (busy, total)
        prev, self._cycles = self._cycles, now
        # only clients with an earlier reading whose counters both moved forward give a delta
        moved = [c for c in now if c in prev and now[c][1] > prev[c][1] and now[c][0] >= prev[c][0]]
        if not moved:
            return None
        span = max(now[c][1] - prev[c][1] for c in moved)
        used = sum(now[c][0] - prev[c][0] for c in moved)
        return max(0.0, min(100.0, 100.0 * used / span))

    def _stat(self):
        """The old root sampler's file (sycl/tools/gpustat.py, /run/gpustat.json) when it is fresh: a PC that still runs it
        keeps its readings for what this process cannot read itself (VRAM without sysman, the load before a delta)."""
        try:
            import json
            with open(self.STAT) as f:
                st = json.load(f)
            return st if time.time() - float(st.get("ts", 0)) < 15 else None
        except (OSError, ValueError, TypeError, AttributeError):
            return None

    def _temps(self):
        """The GPU's own sensor (label "pkg") as the GPU temperature, as NVIDIA's core and AMD's edge readings are;
        the VRAM's sensors (labels vram, vram_ch_N) apart.  A hwmon without labels: the hottest, as before."""
        temps = {}
        if not self.hwmon:
            return None, None
        try:
            names = os.listdir(self.hwmon)
        except OSError:
            return None, None
        for f in names:
            if not (f.startswith("temp") and f.endswith("_input")):
                continue
            v = self._rd(f"{self.hwmon}/{f}")
            if v and v.isdigit():
                temps[self._rd(f"{self.hwmon}/{f[:-6]}_label") or ""] = int(v) / 1000.0
        if not temps:
            return None, None
        pkg = temps.get("pkg")
        vram = [v for k, v in temps.items() if k.startswith("vram")]
        return pkg if pkg is not None else max(temps.values()), max(vram) if vram else None

    def read(self):
        out = {}
        out["util"] = self._busy()
        used, total = self.zes.read() if self.zes else (None, None)
        out["mem_used"], out["mem_total"] = used, total
        if out["util"] is None or out["mem_used"] is None:
            st = self._stat()
            if st:
                if out["util"] is None and st.get("busy_pct") is not None:
                    out["util"] = st["busy_pct"]
                if out["mem_used"] is None and st.get("vram_used_mb") is not None:
                    out["mem_used"] = st["vram_used_mb"] * 2**20
                    out["mem_total"] = (st.get("vram_total_mb") or 0) * 2**20 or out["mem_total"]
        out["temp"], out["vram_temp"] = self._temps()
        out["power"] = out["power_limit"] = None
        if self.hwmon:
            e, t = self._rd(f"{self.hwmon}/energy1_input"), time.time()
            if e and e.isdigit():
                if self._e and t > self._e[0] and int(e) >= self._e[1]:
                    out["power"] = (int(e) - self._e[1]) / 1e6 / (t - self._e[0])
                self._e = (t, int(e))
            cap = self._rd(f"{self.hwmon}/power1_cap")
            out["power_limit"] = int(cap) / 1e6 if cap and cap.isdigit() and int(cap) > 0 else None
        out["pcie_gen"], out["pcie_gen_max"], out["pcie_width"] = self._link()
        out["pcie_rx_mb"] = out["pcie_tx_mb"] = None
        st = self.zes.pci_stats() if self.zes else None
        prev, self._pci = self._pci, st
        if st and prev and st[2] > prev[2] and st[0] >= prev[0] and st[1] >= prev[1]:
            dt = (st[2] - prev[2]) / 1e6
            out["pcie_rx_mb"] = (st[0] - prev[0]) / dt / 2**20
            out["pcie_tx_mb"] = (st[1] - prev[1]) / dt / 2**20
        return out
