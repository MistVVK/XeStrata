"""#5353: the Monitor reads an Intel Arc card from sysfs on every serve/server.py route (no GPU, no root: a fake sysfs tree).

Before, serve/telemetry.py knew NVML, amdgpu and Metal only; the Intel reader lived in sycl/serve/xe_telemetry.py and only the source
route through sycl/serve/server_intel.py installed it, so the container, the packaged Terminal and the app showed no GPU readings on an
Arc. The fake trees: an xe Arc Pro B70 beside a UHD iGPU (card0, no hwmon), and an i915 Arc A750 (no tile0, hwmon "i915").
#5988 (from PR #5994): the engine's load from its DRM clients' fdinfo (xe cycles, i915 nanoseconds) and VRAM / PCIe traffic from Level
Zero sysman (a fake library object here), without root or the sampler."""
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import telemetry  # noqa: E402


class NoZes:
    """sysman with nothing to say (no Level Zero, or the card not found)."""
    def read(self):
        return None, None

    def pci_stats(self):
        return None


def write(p, text):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def pci(root, bus, vendor, device, width=16, speed="16.0 GT/s PCIe"):
    d = root / "devices" / "pci0000:00" / bus
    write(d / "vendor", vendor + "\n")
    write(d / "device", device + "\n")
    for k, v in (("current_link_speed", speed), ("max_link_speed", speed), ("current_link_width", str(width))):
        write(d / k, v + "\n")
    return d


def link_card(root, n, dev):
    c = root / "class" / "drm" / f"card{n}"
    c.mkdir(parents=True, exist_ok=True)
    os.symlink(dev, c / "device")


def b70_tree(root):
    igpu = pci(root, "0000:00:02.0", "0x8086", "0x3e92", width=0)              # UHD 630: no hwmon, no tile0
    link_card(root, 0, igpu)
    b70 = pci(root, "0000:03:00.0", "0x8086", "0xe223")
    (b70 / "tile0").mkdir()
    h = b70 / "hwmon" / "hwmon3"
    write(h / "name", "xe\n")
    write(h / "temp1_label", "vram\n"); write(h / "temp1_input", "60000\n")
    write(h / "temp2_label", "pkg\n"); write(h / "temp2_input", "52000\n")
    write(h / "energy1_input", "1000000\n")
    write(h / "power1_cap", "230000000\n")
    link_card(root, 1, b70)
    return h


def a750_tree(root):
    a750 = pci(root, "0000:2d:00.0", "0x8086", "0x56a1", speed="8.0 GT/s PCIe")
    h = a750 / "hwmon" / "hwmon2"
    write(h / "name", "i915\n")
    write(h / "temp1_input", "47000\n")
    write(h / "energy1_input", "5000000\n")
    write(h / "power1_max", "190000000\n")
    link_card(root, 0, a750)
    return h


@unittest.skipUnless(sys.platform.startswith("linux") and hasattr(os, "symlink"), "sysfs trees with symlinks: Linux")
class IntelSysfsReader(unittest.TestCase):
    def test_b70_beside_an_igpu_is_found_and_read(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            h = b70_tree(root)
            r = telemetry._Intel(0, sysfs=str(root), stat=str(root / "no-gpustat.json"), proc=str(root / "proc"), zes=NoZes())
            self.assertTrue(r.ok(), "the B70 (xe, tile0), not the UHD iGPU at card0")
            self.assertEqual(r.name(), "Intel Arc Pro B70")
            first = r.read()
            self.assertEqual(first["temp"], 52.0, "xe's pkg sensor, not temp1 (vram)")
            self.assertEqual(first["power_limit"], 230.0)
            self.assertIsNone(first["power"], "power needs two energy readings")
            for k in ("util", "mem_used", "mem_total"):
                self.assertIsNone(first[k], f"{k}: needs the root sampler, absent without it")
            self.assertEqual((first["pcie_gen"], first["pcie_gen_max"], first["pcie_width"]), (4, 4, 16))
            r._e = (r._e[0] - 1.0, r._e[1])                          # one second earlier
            write(h / "energy1_input", "51000000\n")                  # +50 J
            second = r.read()
            self.assertAlmostEqual(second["power"], 50.0, delta=5.0)
            self.assertFalse(telemetry._Intel(1, sysfs=str(root), zes=NoZes()).ok(), "only one discrete Intel card here")

    def test_a750_on_i915_is_found_and_read(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a750_tree(root)
            r = telemetry._Intel(0, sysfs=str(root), stat=str(root / "no-gpustat.json"), proc=str(root / "proc"), zes=NoZes())
            self.assertTrue(r.ok(), "an i915 Arc has no tile0; its hwmon marks it as a discrete card")
            self.assertEqual(r.name(), "Intel Arc A750")
            out = r.read()
            self.assertEqual(out["temp"], 47.0)
            self.assertEqual(out["power_limit"], 190.0)
            self.assertEqual(out["pcie_gen"], 3)

    def test_the_root_sampler_adds_load_and_vram(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            b70_tree(root)
            stat = root / "gpustat.json"
            stat.write_text(json.dumps({"ts": time.time(), "busy_pct": 37, "vram_used_mb": 1024, "vram_total_mb": 32768}))
            out = telemetry._Intel(0, sysfs=str(root), stat=str(stat), proc=str(root / "proc"), zes=NoZes()).read()
            self.assertEqual((out["util"], out["mem_used"], out["mem_total"]), (37, 1024 * 2**20, 32768 * 2**20))
            stat.write_text(json.dumps({"ts": time.time() - 60, "busy_pct": 99}))
            self.assertIsNone(telemetry._Intel(0, sysfs=str(root), stat=str(stat), proc=str(root / "proc"), zes=NoZes()).read()["util"],
                              "a stale sampler file is not read")


def fdinfo(root, pid, fd, text):
    write(root / "proc" / str(pid) / "fdinfo" / str(fd), text)


class FakeZes:
    """sysman with a card: 30 of 32 GiB free memory, PCIe counters that move 2 GiB in and 1 GiB out per second."""
    def __init__(self):
        self.t = 1_000_000

    def read(self):
        return 2 * 2**30, 32 * 2**30

    def pci_stats(self):
        self.t += 1_000_000
        k = self.t // 1_000_000
        return k * 2 * 2**30, k * 2**30, self.t


@unittest.skipUnless(sys.platform.startswith("linux") and hasattr(os, "symlink"), "sysfs trees with symlinks: Linux")
class RootlessLoadAndVram(unittest.TestCase):
    def test_xe_load_from_the_engines_fdinfo_cycles(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            b70_tree(root)
            other = "drm-driver:\txe\ndrm-pdev:\t0000:09:00.0\ndrm-client-id:\t7\ndrm-cycles-ccs:\t999\ndrm-total-cycles-ccs:\t1000\n"
            fdinfo(root, 222, 4, other)                              # another card's client: not counted
            def mine(busy, total):
                fdinfo(root, 111, 5, f"drm-driver:\txe\ndrm-pdev:\t0000:03:00.0\ndrm-client-id:\t42\n"
                                     f"drm-cycles-rcs:\t0\ndrm-cycles-ccs:\t{busy}\ndrm-total-cycles-ccs:\t{total}\n")
            mine(1000, 10000)
            r = telemetry._Intel(0, sysfs=str(root), stat=str(root / "none.json"), proc=str(root / "proc"), zes=NoZes())
            self.assertIsNone(r.read()["util"], "one read: nothing to compare yet")
            mine(1000 + 4000, 10000 + 10000)                         # 4000 busy of 10000 cycles
            self.assertAlmostEqual(r.read()["util"], 40.0, places=3)
            mine(5000, 30000)                                        # idle
            self.assertEqual(r.read()["util"], 0.0)

    def test_i915_load_from_busy_nanoseconds_over_wall_time(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a750_tree(root)
            def mine(ns):
                fdinfo(root, 333, 6, f"drm-driver:\ti915\ndrm-pdev:\t0000:2d:00.0\ndrm-client-id:\t9\n"
                                     f"drm-engine-render:\t0 ns\ndrm-engine-compute:\t{ns} ns\n")
            mine(0)
            clock = iter([10_000_000_000, 11_000_000_000])           # one second of wall time
            r = telemetry._Intel(0, sysfs=str(root), stat=str(root / "none.json"), proc=str(root / "proc"), zes=NoZes())
            with mock.patch.object(telemetry.time, "monotonic_ns", side_effect=lambda: next(clock)):
                self.assertIsNone(r.read()["util"])
                mine(250_000_000)                                    # 0.25 s busy
                self.assertAlmostEqual(r.read()["util"], 25.0, places=3)

    def test_vram_and_pcie_traffic_from_sysman(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            b70_tree(root)
            r = telemetry._Intel(0, sysfs=str(root), stat=str(root / "none.json"), proc=str(root / "proc"), zes=FakeZes())
            first = r.read()
            self.assertEqual((first["mem_used"], first["mem_total"]), (2 * 2**30, 32 * 2**30))
            self.assertIsNone(first["pcie_rx_mb"], "traffic needs two counter readings")
            second = r.read()
            self.assertAlmostEqual(second["pcie_rx_mb"], 2048.0, places=3)
            self.assertAlmostEqual(second["pcie_tx_mb"], 1024.0, places=3)

    def test_i915_vram_is_the_engines_fdinfo_memory_not_sysmans_zero(self):
        """sysman on i915 reports the size but free == size (used 0 while an engine holds gigabytes: an A750, p1006 11-16)."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a750_tree(root)
            fdinfo(root, 444, 7, "drm-driver:\ti915\ndrm-pdev:\t0000:2d:00.0\ndrm-client-id:\t3\ndrm-engine-render:\t0 ns\n"
                                 "drm-engine-compute:\t0 ns\ndrm-total-local0:\t6291456 KiB\ndrm-resident-local0:\t6291456 KiB\n"
                                 "drm-resident-system0:\t1024 KiB\n")
            fdinfo(root, 444, 8, "drm-driver:\ti915\ndrm-pdev:\t0000:2d:00.0\ndrm-client-id:\t3\ndrm-resident-local0:\t6291456 KiB\n")
            out = telemetry._Intel(0, sysfs=str(root), stat=str(root / "none.json"), proc=str(root / "proc"), zes=FakeZes()).read()
            self.assertEqual(out["mem_used"], 6 * 2**30, "one client (two fds of it) holding 6 GiB of local memory; system memory not counted")
            self.assertEqual(out["mem_total"], 32 * 2**30, "the size still comes from sysman")
            empty = Path(d) / "empty"
            a750_tree(empty)
            out = telemetry._Intel(0, sysfs=str(empty), stat=str(empty / "none.json"), proc=str(empty / "proc"), zes=FakeZes()).read()
            self.assertIsNone(out["mem_used"], "no engine client: no reading, never sysman's 0")

    def test_the_sampler_still_wins_for_load_and_vram(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            b70_tree(root)
            stat = root / "gpustat.json"
            stat.write_text(json.dumps({"ts": time.time(), "busy_pct": 37, "vram_used_mb": 1024, "vram_total_mb": 32768}))
            out = telemetry._Intel(0, sysfs=str(root), stat=str(stat), proc=str(root / "proc"), zes=FakeZes()).read()
            self.assertEqual((out["util"], out["mem_used"]), (37, 1024 * 2**20))

    def test_a_missing_level_zero_is_no_reading_not_an_error(self):
        z = telemetry._Zes("0000:03:00.0", lib=mock.Mock(zesInit=mock.Mock(return_value=0x78000001)))
        self.assertEqual(z.read(), (None, None))
        self.assertIsNone(z.pci_stats())


@unittest.skipUnless(sys.platform.startswith("linux"), "the Intel fallback is Linux only")
class GpuReaderFallback(unittest.TestCase):
    def test_no_nvml_and_an_arc_card_gives_the_intel_reader(self):
        fake = mock.Mock(ok=mock.Mock(return_value=True))
        with mock.patch.object(telemetry._Nvml, "ok", return_value=False), mock.patch.object(telemetry, "_Intel", return_value=fake):
            self.assertIs(telemetry.gpu_reader(0), fake)

    def test_nvml_wins_and_amd_is_unchanged(self):
        with mock.patch.object(telemetry._Nvml, "ok", return_value=True), mock.patch.object(telemetry, "_Intel") as intel:
            self.assertIsInstance(telemetry.gpu_reader(0), telemetry._Nvml)
            intel.assert_not_called()
        with mock.patch.object(telemetry, "_Intel") as intel:
            self.assertIsInstance(telemetry.gpu_reader(0, amd=True), telemetry._Amd)
            intel.assert_not_called()

    def test_no_nvml_and_no_arc_keeps_the_not_ok_nvml_reader(self):
        with mock.patch.object(telemetry._Nvml, "ok", return_value=False), \
                mock.patch.object(telemetry, "_Intel", return_value=mock.Mock(ok=mock.Mock(return_value=False))):
            self.assertIsInstance(telemetry.gpu_reader(0), telemetry._Nvml)


if __name__ == "__main__":
    unittest.main()
