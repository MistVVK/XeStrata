# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""The training budget's reuse of its own process's memory."""
import unittest
from unittest import mock

from serve import train_budget


class ProcessRoom(unittest.TestCase):
    def test_process_room_running_and_paused(self):
        pci = "0000:0c:00.0"
        card = {"pci": pci, "kind": "share", "total_mib": 16384}
        with mock.patch.object(train_budget, "candidates", return_value=[card]):
            decider = train_budget.Decider({}, "mage-flow", "int8", image_tokens=2237, rank=8)
        for state, free, used, process in (("paused", 6500, 0, 1263), ("running", 2500, 4024, 5290)):
            for source in ("fdinfo", "nvml", "delta", "global", "unknown"):
                with self.subTest(state=state, source=source):
                    st = {"free": {pci: free}, "device_free": {pci: free}, "current_pci": pci,
                          "proc_alive": True, "train": {"state": state, "used_mib": used,
                                                        "process_mib": process, "process_source": source,
                                                        "need_mib": 6782}}
                    act = decider.tick(st)
                    self.assertEqual(act["budget_mib"] > 0, source in ("fdinfo", "nvml"))
        st = {"free": {pci: 6500}, "current_pci": pci, "proc_alive": True,
              "train": {"used_mib": 1300, "need_mib": 6782}}
        self.assertGreater(decider.tick(st)["budget_mib"], 0)  # no process report: use the held allocation count
        st["proc_alive"] = False
        self.assertEqual(decider.tick(st)["budget_mib"], 0)     # no allocation from a dead process to reuse


if __name__ == "__main__":
    unittest.main()
