"""#4654: the sampler's and the HIP probe's files are closed (no ResourceWarning lines on every run).

    python -m unittest serve.test_resource_warnings
"""
import gc
import os
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import server, telemetry  # noqa: E402


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc and /sys")
class Closed(unittest.TestCase):
    def quiet(self, call):
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always", ResourceWarning)
            call()
            gc.collect()
        return [str(w.message) for w in seen if issubclass(w.category, ResourceWarning)]

    def test_the_cpu_and_ram_sampler_closes_its_files(self):
        def run():
            fallback = telemetry._CpuRamFallback()
            fallback.cpu()
            fallback.ram()
            telemetry._cpu_name()
        self.assertEqual(self.quiet(run), [])

    def test_the_cpu_topology_closes_its_files(self):
        self.assertEqual(self.quiet(lambda: telemetry._cpu_topology()), [])

    def test_the_hip_probes_close_their_files(self):
        with tempfile.TemporaryDirectory() as root:
            node = Path(root) / "0"
            node.mkdir()
            (node / "properties").write_text("simd_count 256\nmax_engine_clk_fcompute 2000\ngfx_target_version 110501\n")
            self.assertEqual(self.quiet(lambda: server.hip_speed_scores([0], root)), [])
            self.assertEqual(self.quiet(lambda: server.hip_gpu_archs(root)), [])
            self.assertEqual(server.hip_gpu_archs(root), {0: "gfx1151"})


if __name__ == "__main__":
    unittest.main()
