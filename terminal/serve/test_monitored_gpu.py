"""#4640: on a PC with two AMD cards, an "Automatic" start (or the person's own HIP_VISIBLE_DEVICES) ran the engine on card 1 while the
Monitor and the benchmark export read card 0, an idle card beside it. The Monitor's card now follows the engine's visible devices when the
config names no card; a config that names cards keeps them; an environment that does not say clearly leaves card 0.

    python -m pytest serve/test_monitored_gpu.py
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from serve import server, telemetry


def amd_tree(d):
    """Two AMD cards as KFD numbers them (the CPU node skipped): HIP 0 is idle (renderD129), HIP 1 runs the engine (renderD128)."""
    nodes = Path(d) / "class/kfd/kfd/topology/nodes"
    for n, props in ((0, "cpu_cores_count 16\nsimd_count 0\ngfx_target_version 0\ndrm_render_minor 0\n"),
                     (1, "simd_count 60\ngfx_target_version 90006\ndrm_render_minor 129\n"),
                     (2, "simd_count 128\ngfx_target_version 120001\ndrm_render_minor 128\n")):
        (nodes / str(n)).mkdir(parents=True)
        (nodes / str(n) / "properties").write_text(props)
    for minor, name, busy, used in ((129, "AMD Radeon Instinct MI50", 0, 1 << 30), (128, "AMD Radeon AI PRO R9700", 97, 20 << 30)):
        dev = Path(d) / f"class/drm/renderD{minor}/device"
        (dev / "hwmon" / "hwmon1").mkdir(parents=True)
        (dev / "gpu_busy_percent").write_text(f"{busy}\n")
        (dev / "mem_info_vram_used").write_text(f"{used}\n")
        (dev / "mem_info_vram_total").write_text(f"{32 << 30}\n")
        (dev / "product_name").write_text(name + "\n")


class MonitoredGpu(unittest.TestCase):
    def test_the_engine_s_visible_card_when_the_config_names_none(self):
        self.assertEqual(server.monitored_gpu([], {"HIP_VISIBLE_DEVICES": "1"}, "hip"), 1)
        self.assertEqual(server.monitored_gpu([], {"HIP_VISIBLE_DEVICES": "2,0"}, "hip"), 2)

    def test_the_config_s_cards_win(self):
        self.assertEqual(server.monitored_gpu([3, 1], {"HIP_VISIBLE_DEVICES": "1"}, "hip"), 3)

    def test_an_unclear_environment_leaves_card_0(self):
        for env, backend in (({}, "hip"), (None, "hip"), ({"HIP_VISIBLE_DEVICES": "GPU-abc"}, "hip"), ({"HIP_VISIBLE_DEVICES": "1,1"}, "hip"),
                             ({"CUDA_VISIBLE_DEVICES": "1"}, "cuda"),          # CUDA's own order is not the bus order the readers use
                             ({"HIP_VISIBLE_DEVICES": "1"}, "vulkan")):
            self.assertEqual(server.monitored_gpu([], env, backend), 0, (env, backend))
        self.assertEqual(server.monitored_gpu([], {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}, "cuda"), 1)

    def test_the_monitor_reads_the_card_the_engine_runs_on(self):
        with tempfile.TemporaryDirectory() as d:
            amd_tree(d)
            with mock.patch.object(telemetry, "SYSFS", d):
                index = server.monitored_gpu([], {"HIP_VISIBLE_DEVICES": "1"}, "hip")
                reader = telemetry.gpu_reader(index, amd=True)
                self.assertIn("R9700", reader.name())
                self.assertEqual(reader.read()["util"], 97)
                self.assertIn("MI50", telemetry.gpu_reader(0, amd=True).name(), "card 0 is the idle one the Monitor used to read")

    def test_the_server_sets_its_card_through_monitored_gpu(self):
        src = Path(server.__file__).read_text(encoding="utf-8")
        self.assertIn('svc.gpu_index = monitored_gpu(gpu_list(cfg), env if a.engine == "strata" else None, cfg.get("backend"))', src)


if __name__ == "__main__":
    unittest.main()
