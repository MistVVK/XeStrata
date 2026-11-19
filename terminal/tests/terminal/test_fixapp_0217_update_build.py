"""#5214: `./setup.sh --update --build` finished in a second with "Strata is updated" and compiled nothing: update_install skipped the engine step when
--build was given. With --build the installed engine is compiled again from this source; without it nothing changes (a ready-made engine that is new
enough is kept, as before).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import setup  # noqa: E402

GPU = {"index": 0, "name": "RTX 4090", "arch": "89", "vram_gb": 24}


class UpdateWithBuild(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.eng = Path(tmp.name) / "engine"
        self.eng.mkdir()
        (self.eng / setup.EXE).write_bytes(b"x")
        version = ".".join(str(x) for x in setup.MIN_ENGINE)
        (self.eng / "BUILD.json").write_text(json.dumps({"version": version, "backend": "cuda", "source": "prebuilt", "archs": ["89"]}), encoding="utf-8")
        for name, value in (("engine_dir", self.eng), ("update_engine_gpu", dict(GPU))):
            patch = mock.patch.object(setup, name, return_value=value)
            patch.start()
            self.addCleanup(patch.stop)
        self.build = mock.patch.object(setup, "build_engine").start()
        self.fetch = mock.patch.object(setup, "get_prebuilt", return_value=self.eng).start()
        for name in ("llama_override", "pip_cuda_libs", "say", "warn"):
            mock.patch.object(setup, name).start()
        self.addCleanup(mock.patch.stopall)

    def test_a_current_ready_made_engine_is_left_alone_without_build(self):
        setup.update_installed_engine("https://example.invalid/", 13)
        self.build.assert_not_called()
        self.fetch.assert_not_called()

    def test_update_build_compiles_the_installed_engine(self):
        setup.update_installed_engine("https://example.invalid/", 13, rebuild=True)
        self.build.assert_called_once()
        self.fetch.assert_not_called()

    def test_update_install_passes_build_on(self):
        cfg = Path(self.eng.parent / "strata-x.json")
        cfg.write_text("{}", encoding="utf-8")
        seen = []
        with mock.patch.object(setup, "model_config", return_value=True), mock.patch.object(setup, "pip_install"), \
                mock.patch.object(setup, "upgrade_config", side_effect=lambda p, c: {"model_name": "x"}), \
                mock.patch.object(setup, "own_path_config", return_value=True), mock.patch.object(setup, "ok"), \
                mock.patch.object(setup, "config_engine_version", return_value=setup.MIN_ENGINE), \
                mock.patch.object(setup, "runtime_packages", return_value=[]), \
                mock.patch.object(setup, "update_installed_engine", side_effect=lambda *a, **k: seen.append((a, k))):
            for build in (False, True):
                setup.update_install([cfg], argparse.Namespace(build=build, prebuilt="u"))
        self.assertEqual([k.get("rebuild") for _, k in seen], [False, True])


if __name__ == "__main__":
    unittest.main()
