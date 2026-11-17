"""#5369 (port part): `strata bench run` that finds no server names the address it asked, and for a profile the way to the one that serves: a model
started with strata launch has its own profile (launch-<preset>, with the launch's port), or the profile's port can be set. (The profile made from
setup's installed run config keeps setup's port 8080 on purpose: a launch's --port lives in its own launch config.)"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from strata_cli import bench_standard as B
from strata_control.profiles import ProfileError


class Silent:
    root = "http://127.0.0.1:8080"

    def get(self, path, timeout=10):
        return None


class NoServer(unittest.TestCase):
    def test_the_words(self):
        self.assertEqual(B.no_server_text("http://127.0.0.1:8080"), "No Strata server answers at http://127.0.0.1:8080.")
        self.assertEqual(B.no_server_text("http://127.0.0.1:8080", "bench1"),
                         "No Strata server answers at http://127.0.0.1:8080 (this profile's address). If you started the model with strata launch, "
                         "use its profile launch-<preset> (see strata profile list), or set the port: strata profile set bench1 port=N.")

    def test_the_run_refuses_with_the_address(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ProfileError) as cm:
                B.run(Silent(), Path(tmp) / "out", profile="bench1")
        self.assertEqual(cm.exception.exit_code, 5)
        self.assertIn("http://127.0.0.1:8080 (this profile's address)", str(cm.exception))
        self.assertIn("strata profile set bench1 port=N", str(cm.exception))

    def test_the_command_passes_the_profile(self):
        src = (Path(__file__).resolve().parents[2] / "strata_cli" / "cli.py").read_text(encoding="utf-8")
        self.assertIn("hash_model=not a.skip_model_hash, profile=a.profile)", src)


if __name__ == "__main__":
    unittest.main()
