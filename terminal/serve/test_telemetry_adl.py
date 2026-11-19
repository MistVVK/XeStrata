"""#4884: the Monitor's AMD-on-Windows readings (ADL2 PMLog + Device Manager's PCIe link) degrade safely, map to the keys the
Monitor uses, and stay inside the Monitor (the server's own VRAM checks keep their reader).  No GPU or driver is needed: on a box
without atiadlxx.dll every reader reports "unavailable" (ok() False, read() {}) and never raises.  Tests adapted from the PR's
serve/test_telemetry_adl.py (Lip Wee Yeo)."""
import os
import subprocess
import time
import unittest
from unittest import mock

import serve.telemetry as t


class Degrades(unittest.TestCase):
    def test_reader_never_raises(self):
        for index in (0, 1, 5):
            r = t.monitor_reader(index, amd=True)
            self.assertIsInstance(r.read(), dict)
            self.assertIsInstance(r.name(), str)
            if hasattr(r, "close"):
                r.close()

    def test_unavailable_reader_is_off(self):
        r = t._AmdWin(0)
        if r.lib is None:
            self.assertFalse(r.ok())
            self.assertEqual(r.read(), {})
        r.close()
        r.close()                       # twice: safe
        self.assertFalse(r.ok())
        self.assertEqual(r.read(), {})


class OnlyTheMonitor(unittest.TestCase):
    """Strictly additive: only Telemetry's readers change on Windows AMD; free_vram_mib / gpu_vram_mib / the drafter keep _Amd."""

    def test_reader_choice(self):
        with mock.patch.object(t.os, "name", "nt"), mock.patch.object(t, "_AmdWin") as win:
            self.assertIsInstance(t.gpu_reader(0, amd=True), t._Amd)             # the server's checks: unchanged
            win.assert_not_called()
            t.monitor_reader(1, amd=True)                                        # the Monitor: ADL2 when sysfs reads nothing
            win.assert_called_once_with(1)
        with mock.patch.object(t.os, "name", "posix"), mock.patch.object(t, "_AmdWin") as win:
            self.assertIsInstance(t.monitor_reader(0, amd=True), t._Amd)
            win.assert_not_called()
        with mock.patch.object(t.os, "name", "nt"), mock.patch.object(t, "_AmdWin") as win, mock.patch.object(t, "_Nvml"), mock.patch.object(t, "_Metal"):
            t.monitor_reader(0, amd=False)                                       # NVIDIA / Metal: never ADL2
            win.assert_not_called()


class FakeWin:
    """A Windows AMD reader with fixed readings (the sample() aggregation of several cards)."""

    def __init__(self, name, r):
        self._n, self._r = name, r

    def ok(self):
        return True

    def name(self):
        return self._n

    def read(self):
        return dict(self._r)


class SampleRule(unittest.TestCase):
    """Two Windows AMD cards: top-level gpu_pcie_* stay card 0's, gpu_pcie_bottleneck names the slower card, "gpus" has each link."""

    def test_split(self):
        a = {"util": 40, "mem_used": 4 << 30, "mem_total": 16 << 30, "temp": 60.0, "power": 120.0, "power_limit": 300.0,
             "pcie_gen": 5, "pcie_gen_max": 5, "pcie_width": 8, "pcie_width_max": 16}
        b = {"util": 20, "mem_used": 2 << 30, "mem_total": 32 << 30, "temp": 70.0, "power": 80.0, "power_limit": 300.0,
             "pcie_gen": 5, "pcie_gen_max": 5, "pcie_width": 4, "pcie_width_max": 16}
        tel = t.Telemetry.__new__(t.Telemetry)
        tel.gpus = [(0, FakeWin("AMD Radeon RX 9070 XT", a)), (1, FakeWin("AMD Radeon AI PRO R9700", b))]
        tel.gpu = tel.gpus[0][1]
        tel.ps, tel.extra, tel._disk_prev = None, None, None
        tel.fallback = mock.Mock(cpu=lambda: None, ram=lambda: (None, None))
        s = tel.sample()
        self.assertEqual((s["gpu_pcie_gen"], s["gpu_pcie_width"]), (5, 8))              # card 0's link
        self.assertEqual(s["gpu_power"], 200.0)                                          # additive fields add up
        self.assertEqual(s["gpu_pcie_bottleneck"]["card"], 1)
        self.assertEqual(s["gpu_pcie_bottleneck"]["width"], 4)
        self.assertEqual([g["pcie_width"] for g in s["gpus"]], [8, 4])
        self.assertEqual([g["power_limit"] for g in s["gpus"]], [300.0, 300.0])


class Mapping(unittest.TestCase):
    def test_pick_card_by_name_then_position(self):
        adl = ["AMD Radeon RX 6800 XT", "AMD Radeon AI PRO R9700"]
        self.assertEqual(t._pick_card(adl, 0, ["AMD Radeon AI PRO R9700", "AMD Radeon RX 6800 XT"]), 1)   # HIP order differs
        self.assertEqual(t._pick_card(adl, 1, ["AMD Radeon AI PRO R9700", "AMD Radeon RX 6800 XT"]), 0)
        same = ["AMD Radeon RX 7900 XTX", "AMD Radeon RX 7900 XTX"]
        self.assertEqual([t._pick_card(same, i, same) for i in (0, 1)], [0, 1])                               # two of one name
        self.assertEqual(t._pick_card(adl, 1, []), 1)                                                         # no registry list
        self.assertIsNone(t._pick_card(adl, 2, []))
        self.assertIsNone(t._pick_card(["AMD Radeon(TM) Graphics"], 0, ["AMD Radeon RX 6900 XT"]))           # no ADL row for the HIP card: no reading

    def test_hip_device_beyond_a_one_card_list_reads_the_only_card(self):
        # 0.2.17 (#5946): an iGPU + one RX 9070 XT; the run config says "gpu": 1 (HIP counts the iGPU), the registry list holds
        # the one card, ADL lists the one card at row 0: the reading was blank (None), now it is the card's row.
        card = "AMD Radeon RX 9070 XT"
        self.assertEqual(t._pick_card([card], 1, [card]), 0)
        self.assertIsNone(t._pick_card([card, card], 1, [card]))                  # two rows of that name: no guess
        self.assertIsNone(t._pick_card(["AMD Radeon RX 6800 XT"], 1, [card]))     # no row of that name: no guess
        self.assertIsNone(t._pick_card([card], 2, [card, "AMD Radeon RX 6800 XT"]))   # a longer list keeps the old rule

    def test_cards_skip_the_igpu_beside_a_card_and_keep_an_apu_alone(self):
        A = t._Adl.AdapterInfo

        def rows(*spec):
            buf = (A * len(spec))()
            for k, (name, dev) in enumerate(spec):
                buf[k].AdapterName, buf[k].UDID, buf[k].Present = name, b"PCI_VEN_1002&DEV_" + dev + b"&SUBSYS", 1
            return buf
        buf = rows((b"AMD Radeon(TM) Graphics", b"164e"), (b"AMD Radeon RX 9070 XT", b"7550"), (b"AMD Radeon RX 9070 XT", b"7550"))
        self.assertEqual(t._adl_cards(buf, 3), [(1, "7550", "AMD Radeon RX 9070 XT", False)])   # one row per card, no iGPU
        apu = rows((b"AMD Radeon(TM) 8060S Graphics", b"1586"), (b"AMD Radeon(TM) 8060S Graphics", b"1586"))
        nv = rows((b"NVIDIA GeForce RTX 5070", b"2F04"), (b"AMD Radeon(TM) Graphics", b"164E"))
        nv[0].UDID = b"PCI_VEN_10DE&DEV_2F04"
        self.assertEqual(t._adl_cards(nv, 2), [(1, "164E", "AMD Radeon(TM) Graphics", True)])            # another vendor's adapter is never a card
        self.assertEqual(t._adl_cards(apu, 2), [(0, "1586", "AMD Radeon(TM) 8060S Graphics", True)])

    def test_link_fields(self):
        self.assertEqual(t._adl_link((4, 16, 5, 16), 8), {"pcie_gen": 4, "pcie_gen_max": 5, "pcie_width_max": 16, "pcie_width": 8})
        self.assertEqual(t._adl_link(None, None), {"pcie_width": None})                    # an APU: no link at all
        self.assertEqual(t._adl_link((0, 0, 0, 0), None)["pcie_gen_max"], None)


class WindowsAmdCards(unittest.TestCase):
    """The card list _pick_card and hardware_static use, rows measured on a 7900 XTX desktop (0.2.16): its Ryzen iGPU
    carves out exactly 2 GiB, so the old size rule kept it as card 0, and the display class also kept the driver key
    of a taken-out RX 9070 XT (DirectX AdapterLuid 0).  Neither may take the card's row; an APU alone keeps its own."""

    ROWS = [{"name": "AMD Radeon(TM) Graphics", "vram_bytes": 2 << 30},
            {"name": "AMD Radeon RX 9070 XT", "vram_bytes": 16 << 30},                     # stale key: not plugged in
            {"name": "AMD Radeon RX 7900 XTX", "vram_bytes": 24 << 30}]

    def test_igpu_with_2_gib_and_a_stale_key_leave_the_card(self):
        with mock.patch.object(t, "_dx_amd_absent", return_value={t._norm_card("AMD Radeon RX 9070 XT")}):
            self.assertEqual([c["name"] for c in t.windows_amd_cards(self.ROWS)], ["AMD Radeon RX 7900 XTX"])

    def test_apu_alone_and_a_name_directx_does_not_report_change_nothing(self):
        apu = [{"name": "AMD Radeon(TM) 8060S Graphics", "vram_bytes": 512 << 20}]
        with mock.patch.object(t, "_dx_amd_absent", return_value=set()):
            self.assertEqual([c["name"] for c in t.windows_amd_cards(apu)], ["AMD Radeon(TM) 8060S Graphics"])
            # name rule alone: the iGPU still goes, and a stale key DirectX does not report as absent stays (as before the fix)
            self.assertEqual([c["name"] for c in t.windows_amd_cards(self.ROWS)],
                             ["AMD Radeon RX 9070 XT", "AMD Radeon RX 7900 XTX"])

    def test_hip_card_0_finds_the_adl_row(self):
        with mock.patch.object(t, "_dx_amd_absent", return_value={t._norm_card("AMD Radeon RX 9070 XT")}):
            hip = [c["name"] for c in t.windows_amd_cards(self.ROWS)]
        self.assertEqual(t._pick_card(["AMD Radeon RX 7900 XTX"], 0, hip), 0)              # was None: the iGPU's name, no ADL row


class Sensors(unittest.TestCase):
    """The PMLog sensors a reading takes: the first temperature the card reports (an 8060S has the graphics one, 28, no edge),
    board power before ASIC power, and an integrated Radeon shows no PCIe link."""

    def reader(self, sensors, integrated=False):
        r = t._AmdWin(0)
        r.lib, r.adapter, r.integrated, r.dev_code = object(), 0, integrated, "1586"
        d = t._Adl.PmLogDataOutput()
        for sid, v in sensors.items():
            d.sensors[sid].supported, d.sensors[sid].value = 1, v
        r._read_pmlog = lambda: d
        r._call = lambda *a: 1                       # every other ADL call fails: no VRAM numbers
        self.addCleanup(lambda: setattr(r, "lib", None))
        return r

    def test_temperature_order(self):
        self.assertEqual(self.reader({28: 53, 27: 70}).read()["temp"], 53.0)
        self.assertEqual(self.reader({8: 45, 28: 53}).read()["temp"], 45.0)
        self.assertIsNone(self.reader({19: 5}).read()["temp"])

    def test_power_and_link(self):
        with mock.patch.object(t, "_dm_link_map", return_value={"1586": (4, 16, 4, 16)}):
            out = self.reader({73: 120, 23: 90, 41: 8}).read()
            self.assertEqual((out["power"], out["pcie_width"], out["pcie_gen_max"]), (120.0, 8, 4))
            apu = self.reader({23: 45}, integrated=True).read()
            self.assertEqual(apu["power"], 45.0)
            self.assertFalse(any(k.startswith("pcie") for k in apu))


class Stall(unittest.TestCase):
    def test_stalled_sensors_restart_the_stream_once(self):
        r = t._AmdWin(0)
        restarts = []
        r._restart_pmlog = lambda: restarts.append(1)
        for _ in range(r.STALL_LIMIT - 1):
            r._watch_stall({"util": 0.0, "power": 5.0})
        self.assertEqual(restarts, [])
        r._watch_stall({"util": 0.0, "power": 5.0})
        self.assertEqual(len(restarts), 1)
        r._watch_stall({"util": 0.0, "power": 5.0})
        self.assertEqual(len(restarts), 1)
        for _ in range(r.STALL_LIMIT):
            r._watch_stall({"util": None, "power": None})   # no sensors: never restart on dead keys
        self.assertEqual(len(restarts), 1)
        r.close()


class DmLinkMap(unittest.TestCase):
    def setUp(self):
        for name, value in (("_DM_LINK", None), ("_DM_LINK_RETRY", 0.0), ("_DM_LINK_STARTED", False)):
            setattr(t, name, value)
        self.addCleanup(lambda: [setattr(t, n, v) for n, v in (("_DM_LINK", None), ("_DM_LINK_RETRY", 0.0), ("_DM_LINK_STARTED", False))])

    def test_query_never_shows_a_console(self):
        with mock.patch.object(t.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="")
            t._dm_link_query()
        self.assertEqual(run.call_args.kwargs["creationflags"], getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.assertEqual(run.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_first_call_returns_empty_and_starts_the_query(self):
        with mock.patch.object(t.os, "name", "nt"), mock.patch.object(t.threading, "Thread") as thread:
            self.assertEqual(t._dm_link_map(), {})
        thread.return_value.start.assert_called_once()

    def test_failure_retries_after_60_s(self):
        with mock.patch.object(t.subprocess, "run", side_effect=OSError("no powershell")):
            t._dm_link_query()
        self.assertIsNone(t._DM_LINK)
        self.assertGreater(t._DM_LINK_RETRY, time.monotonic())
        with mock.patch.object(t.os, "name", "nt"), mock.patch.object(t.threading, "Thread") as thread:
            self.assertEqual(t._dm_link_map(), {})
        thread.return_value.start.assert_not_called()

    def test_success_caches_upper_case_codes(self):
        with mock.patch.object(t.subprocess, "run") as run:
            run.return_value = mock.Mock(returncode=0, stdout="DEV_744c 4 16 4 16\nDEV_7550 5 8 5 16\njunk\n")
            t._dm_link_query()
        self.assertEqual(t._DM_LINK, {"744C": (4, 16, 4, 16), "7550": (5, 8, 5, 16)})


@unittest.skipIf(os.name != "nt", "the ADL reader is Windows only")
class OnThisBox(unittest.TestCase):
    """Run on a PC with an AMD driver (Strix Halo's 8060S was checked, #4884): the reader fills the Monitor's keys in sane units."""

    def test_readings_when_driver_present(self):
        r = t._AmdWin(0)
        try:
            if not r.ok():
                self.skipTest("no AMD driver (atiadlxx.dll) or no AMD card here")
            out = r.read()             # an older integrated Radeon may answer VRAM only (no PMLog): every key may be absent
            self.assertIsInstance(out, dict)
            if out.get("mem_used") is not None and out.get("mem_total") is not None:
                self.assertLessEqual(out["mem_used"], out["mem_total"])
            if out.get("temp") is not None:
                self.assertTrue(0 < out["temp"] < 120)
            if out.get("power") is not None:
                self.assertTrue(0 < out["power"] < 1000)
        finally:
            r.close()


if __name__ == "__main__":
    unittest.main()
