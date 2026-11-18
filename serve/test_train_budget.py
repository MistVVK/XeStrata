# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The training budget's reuse of its own process's memory."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import lora_train, train_budget, vram_ledger

PCI = "0000:0c:00.0"


def fd(driver, client, fields, pci=PCI):
    return f"drm-driver:\t{driver}\ndrm-client-id:\t{client}\ndrm-pdev:\t{pci}\n{fields}"


# The same table is also passed to gpu.cpp's DRM reader in the AMD measurement.
DRM_CASES = [
    ([fd("xe", "1", "drm-total-vram0:\t2 MiB\ndrm-total-vram1:\t3 MiB\n"),
      fd("xe", "1", "drm-total-vram0:\t2 MiB\n"),
      fd("xe", "2", "drm-total-vram0:\t7 MiB\n")], 12),
    ([fd("amdgpu", "3", "drm-total-vram0:\t20 MiB\ndrm-memory-vram:\t4 MiB\n"),
      fd("amdgpu", "3", "drm-memory-vram:\t4 MiB\n")], 4),
    ([fd("amdgpu", "4", "drm-memory-vram:\t0 KiB\n")], 0),
    ([fd("xe", "5", "drm-total-vram0:\t1 GiB\n"),
      fd("amdgpu", "6", "drm-memory-vram:\t8 MiB\n", "0000:0d:00.0")], None),
]


def write_fdinfo(root, texts):
    d = Path(root) / "123" / "fdinfo"
    d.mkdir(parents=True, exist_ok=True)
    for n, text in enumerate(texts):
        (d / str(n)).write_text(text)
    return d


class ProcessRoom(unittest.TestCase):
    @staticmethod
    def decider():
        card = {"pci": PCI, "kind": "share", "total_mib": 16384}
        with mock.patch.object(train_budget, "candidates", return_value=[card]):
            return train_budget.Decider({}, "mage-flow", "int8", image_tokens=2237, rank=8)

    def test_process_room_running_and_paused(self):
        for state, free, used, process in (("paused", 6500, 0, 1263), ("running", 2500, 4024, 5290)):
            for source in ("fdinfo", "nvml", "delta", "global", "unknown"):
                with self.subTest(state=state, source=source):
                    st = {"free": {PCI: free}, "device_free": {PCI: free}, "current_pci": PCI,
                          "proc_alive": True, "train": {"state": state, "used_mib": used,
                                                        "process_mib": process, "process_source": source,
                                                        "need_mib": 6782}}
                    act = self.decider().tick(st)
                    self.assertEqual(act["budget_mib"] > 0, source in ("fdinfo", "nvml"))
        st = {"free": {PCI: 6500}, "current_pci": PCI, "proc_alive": True,
              "train": {"state": "running", "used_mib": 1300, "need_mib": 6782}}
        self.assertGreater(self.decider().tick(st)["budget_mib"], 0)
        st["proc_alive"] = False
        self.assertEqual(self.decider().tick(st)["budget_mib"], 0)

    def test_synchronized_read_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            svc = mock.Mock()
            svc.images.default.cfg = {"exe": str(Path(tmp) / "xestrata-image")}
            svc.status = {"busy": False, "queued": 0}
            svc.live_reqs = None
            jobs = lora_train.TrainJobs(Path(tmp) / "jobs", svc)
            jobs.place = {"pci": PCI}
            jobs._proc = mock.Mock(pid=123)
            jobs._proc.poll.return_value = None
            jobs._train_state = {"state": "running", "process_source": "fdinfo", "process_mib": 3170,
                                 "used_mib": 2948, "need_mib": 6782}
            dev = [{"pci": PCI, "free_bytes": 2148 << 20}]
            with mock.patch.object(lora_train, "devices", return_value=dev), \
                 mock.patch.object(lora_train, "text_placement", return_value=(set(), "", {})), \
                 mock.patch.object(vram_ledger, "process_memory_mib", side_effect=[(5457, "fdinfo"), (5458, "fdinfo")]):
                st = jobs._state({})
            self.assertEqual(st["train"]["process_mib"], 5457)  # not the stale report's 3170
            self.assertTrue(st["train"]["process_live"])
            self.assertGreater(self.decider().tick(st)["budget_mib"], 0)
        d = self.decider()
        d._last_action = {"pci": PCI, "budget_mib": 7100, "mode": "run", "image_ready": False}
        st = {"free": {PCI: 2148}, "current_pci": PCI, "proc_alive": True,
              "train": {"state": "running", "used_mib": 2948, "need_mib": 6782, "process_live": False}}
        self.assertEqual(d.tick(st)["budget_mib"], 7100)
        self.assertEqual(d.tick(st)["budget_mib"], 0)  # two low readings when live use is unavailable
        d._last_action = {"pci": PCI, "budget_mib": 7100, "mode": "run", "image_ready": False}
        st["chat_busy"] = True
        self.assertEqual(d.tick(st)["budget_mib"], 0)  # higher priority does not wait for hysteresis
        d._last_action = {"pci": PCI, "budget_mib": 7100, "mode": "run", "image_ready": False}
        st["chat_busy"] = False
        st["train"] = {}  # open has not emitted its first report
        self.assertEqual(d.tick(st)["budget_mib"], 7100)
        self.assertEqual(d.tick(st)["budget_mib"], 7100)

    def test_drm_client_table(self):
        for texts, expected in DRM_CASES:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as tmp:
                write_fdinfo(tmp, texts)
                self.assertEqual(vram_ledger.drm_process_memory_mib(123, PCI, Path(tmp)), expected)


if __name__ == "__main__":
    unittest.main()
