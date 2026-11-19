"""The Monitor samples the helper-cache and peer-tier cards too (serve.server.helper_gpus)."""
import unittest

from serve.server import helper_gpus

ENV = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0,1"}


class HelperGpus(unittest.TestCase):
    def test_helper_cache(self):
        cfg = {"gpu": 0, "args": ["--expert-cache", "auto", "--expert-cache-device1", "4800", "--remote-expert-opt"]}
        self.assertEqual(helper_gpus(cfg, ENV), [(1, "helper cache")])

    def test_auto_and_peer(self):
        self.assertEqual(helper_gpus({"args": ["--expert-cache-device1", "auto"]}, ENV), [(1, "helper cache")])
        self.assertEqual(helper_gpus({"args": ["--peer-device", "1"]}, ENV), [(1, "peer tier")])

    def test_mapped_through_visible_devices(self):
        env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "2,0"}
        self.assertEqual(helper_gpus({"args": ["--expert-cache-device1", "4800"]}, env), [(0, "helper cache")])

    def test_nothing_without_an_unambiguous_list(self):
        cfg = {"args": ["--expert-cache-device1", "4800"]}
        self.assertEqual(helper_gpus(cfg, {"CUDA_VISIBLE_DEVICES": "0,1"}), [])    # CUDA's own order: ambiguous
        self.assertEqual(helper_gpus(cfg, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"}), [])
        self.assertEqual(helper_gpus({"args": ["--expert-cache-device1", "0"]}, ENV), [])
        self.assertEqual(helper_gpus({"args": ["--expert-cache-device2", "100"]}, ENV), [])   # no CUDA2 visible


if __name__ == "__main__":
    unittest.main()
