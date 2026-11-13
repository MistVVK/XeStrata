# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""The sampler's files are closed after reading, so the ResourceWarning lines of every run are gone (upstream #4654).

    python -m unittest serve.test_resource_warnings
"""
import gc
import sys
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve import telemetry  # noqa: E402


@unittest.skipUnless(sys.platform.startswith("linux"), "reads /proc")
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


if __name__ == "__main__":
    unittest.main()
