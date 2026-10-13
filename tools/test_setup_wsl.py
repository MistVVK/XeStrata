# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""WSL2 NVIDIA discovery without hardware: python -m unittest tools.test_setup_wsl."""
import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup as S  # noqa: E402


class WslDiscovery(unittest.TestCase):
    def discover(self, version=2, dxg=True, devices=None):
        if devices is None:
            devices = {"0000:01:00.0": {"name": "Test GPU", "mib": 4096, "cc": "8.6"}}
        with patch.object(S, "wsl_version", return_value=version), \
                patch.object(S.Path, "glob", return_value=[]), \
                patch.object(S.Path, "exists", return_value=dxg), \
                patch.object(S, "nvidia_smi", return_value=devices) as smi:
            result = S.gpus()
            return result, smi.call_count

    def test_dxg_gpu(self):
        found, calls = self.discover()
        self.assertEqual(calls, 1)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["index"], 0)
        self.assertEqual(found[0]["vram_gb"], 4)
        self.assertEqual(found[0]["render"], "/dev/dxg")
        self.assertEqual(found[0]["vendor"], "nvidia")
        self.assertEqual(found[0]["driver"], "dxg")
        self.assertIsNone(found[0]["bar_gb"])
        self.assertIsNone(found[0]["link"])

    def test_native_linux_does_not_probe_dxg(self):
        self.assertEqual(self.discover(version=0), ([], 0))

    def test_wsl1_does_not_probe_dxg(self):
        self.assertEqual(self.discover(version=1), ([], 0))

    def test_no_dxg(self):
        self.assertEqual(self.discover(dxg=False), ([], 0))

    def test_no_smi_devices(self):
        self.assertEqual(self.discover(devices={}), ([], 1))

    def test_incomplete_device_is_skipped(self):
        for info in ({"mib": 0, "cc": "8.6"}, {"mib": 4096}, {"mib": 4096, "cc": "N/A"}):
            with self.subTest(info=info):
                self.assertEqual(self.discover(devices={"0000:01:00.0": info})[0], [])

    def test_multiple_devices_have_pci_order(self):
        info = {"name": "Test GPU", "mib": 4096, "cc": "8.6"}
        found, _ = self.discover(devices={"0000:02:00.0": info, "0000:01:00.0": info})
        self.assertEqual([g["pci"] for g in found], ["0000:01:00.0", "0000:02:00.0"])
        self.assertEqual([g["index"] for g in found], [0, 1])

    def test_free_mode_still_rejects_nvidia(self):
        found, _ = self.discover()
        with patch.object(S, "PACKAGED", False), patch.object(S, "ARGS", argparse.Namespace()), \
                patch.object(S, "license_mode", return_value="free"):
            self.assertIsNotNone(S.gpu_problem(found[0]))


if __name__ == "__main__":
    unittest.main()
