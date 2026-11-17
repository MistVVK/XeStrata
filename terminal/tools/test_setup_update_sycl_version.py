"""#4544 (PR #4645): `setup.py --update` on an Intel (SYCL) install reads the engine version from the engine, not from the launcher
script sycl/serve/strata-sycl.sh that the run config names as its exe. The script carries no version, so every update ended with
"the engine was NOT updated: it is still 0.1.12" although the engine had just been built."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

LAUNCHER = "#!/usr/bin/env bash\n# strata-sycl.sh: the SYCL-built engine as a drop-in exe\nexec docker run strata-sycl-dev\n"


class SyclVersion(unittest.TestCase):
    def tree(self, d: Path, *, stamp=None, binary=None, bin_path="build-sycl-aot/strata") -> dict:
        (d / "sycl" / "serve").mkdir(parents=True)
        (d / "sycl" / "serve" / "strata-sycl.sh").write_text(LAUNCHER, encoding="utf-8")
        if stamp is not None:
            (d / "engine-sycl").mkdir()
            (d / "engine-sycl" / "BUILD.json").write_text(json.dumps({"version": stamp, "source": "local"}), encoding="utf-8")
        if binary is not None:
            b = d / bin_path
            b.parent.mkdir(parents=True, exist_ok=True)
            b.write_bytes(b"\x7fELF...strata engine=" + binary.encode() + b"\n...")
        return {"model_name": "Qwen IQ4_XS", "backend": "sycl", "exe": str(d / "sycl" / "serve" / "strata-sycl.sh"),
                "args": ["--native", "/work/models/x"]}

    def version(self, d: Path, cfg: dict, env=None) -> tuple:
        with mock.patch.object(setup, "ROOT", d), mock.patch.dict("os.environ", env or {}, clear=False):
            return setup.config_engine_version(cfg)

    def test_the_stamped_engine_folder_says_the_version(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self.assertEqual(self.version(d, self.tree(d, stamp="0.1.60", binary="0.1.59")), (0, 1, 60))

    def test_without_a_stamp_the_binary_the_launcher_runs(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cfg = self.tree(d, binary="0.1.60")
            self.assertEqual(self.version(d, cfg), (0, 1, 60))
            d2 = d / "x"
            d2.mkdir()
            cfg2 = self.tree(d2, binary="0.1.58", bin_path="build-sycl/strata")
            cfg2["env"] = {"STRATA_SYCL_BIN": "build-sycl/strata"}
            self.assertEqual(self.version(d2, cfg2), (0, 1, 58))

    def test_nothing_readable_is_unknown_never_the_old_default(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            self.assertEqual(self.version(d, self.tree(d)), ())

    def test_a_cuda_config_is_unchanged(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            (d / "engine").mkdir()
            (d / "engine" / "BUILD.json").write_text(json.dumps({"version": "0.1.60"}), encoding="utf-8")
            cfg = {"exe": str(d / "engine" / setup.EXE)}
            with mock.patch.object(setup, "engine_version", wraps=setup.engine_version) as ev:
                self.assertEqual(self.version(d, cfg), (0, 1, 60))
            ev.assert_called_once_with(Path(cfg["exe"]))

    def test_the_update_says_updated_on_a_sycl_install(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            cfg = self.tree(d, stamp="0.1.60")
            p = d / "strata-iq4_xs.json"
            p.write_text(json.dumps(cfg), encoding="utf-8")
            a = mock.Mock(**{"build": False, "prebuilt": "URL"})
            out = io.StringIO()
            with mock.patch.object(setup, "ROOT", d), mock.patch.object(setup, "pip_install"), \
                    mock.patch.object(setup, "update_installed_engine"), mock.patch.object(setup, "refresh_draft_vocab"), \
                    mock.patch.object(setup, "start"), mock.patch("subprocess.call"), contextlib.redirect_stdout(out):
                rc = setup.update_install([p], a)
            self.assertEqual(rc, 0)
            self.assertIn("Strata is updated (engine 0.1.60)", out.getvalue())
            self.assertNotIn("the engine was NOT", out.getvalue())


if __name__ == "__main__":
    unittest.main()
