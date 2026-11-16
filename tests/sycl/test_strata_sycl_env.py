"""#4546 / #4639: sycl/serve/strata-sycl.sh passes the STRATA_* settings of its environment into the container, a value of 0 included.

The launcher dropped every STRATA_* value of 0 ("the engine tests the switches by presence"), so a setting whose 0 means off while
unset means on (STRATA_PREFILL_MMQ=0, the documented FP16 prompt path) never reached the engine: it ran with the switch on. Now the
container gets what a native start would get. The launcher runs here with a fake `docker` on PATH that records its arguments, so
no container, image or GPU is needed. bash 4 (declare -A, compgen): Linux."""
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sycl" / "serve" / "strata-sycl.sh"
FAKE_DOCKER = """#!/usr/bin/env bash
# records `docker run ...` arguments, one per line; `docker rm` does nothing
if [ "$1" = run ]; then printf '%s\\n' "$@" > "$FAKE_DOCKER_OUT"; fi
exit 0
"""


@unittest.skipUnless(sys.platform.startswith("linux") and shutil.which("bash"), "the launcher is a bash script for Linux")
class LauncherPassesZero(unittest.TestCase):
    def run_launcher(self, extra_env):
        with tempfile.TemporaryDirectory() as d:
            bindir = Path(d) / "bin"
            bindir.mkdir()
            fake = bindir / "docker"
            fake.write_text(FAKE_DOCKER)
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            out = Path(d) / "args.txt"
            env = {k: v for k, v in os.environ.items() if not k.startswith(("STRATA_", "ONEAPI_", "SYCL_", "UR_", "IGC_", "ZES_"))}
            env.update({"PATH": f"{bindir}:{env.get('PATH', '/usr/bin:/bin')}", "FAKE_DOCKER_OUT": str(out),
                        "STRATA_SYCL_ROOT": d, "STRATA_SYCL_SHM": "0"})
            env.update(extra_env)
            r = subprocess.run(["bash", str(SCRIPT), "--serve", "--prefill", "1024"], env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            return out.read_text().splitlines()

    @staticmethod
    def passed(args):
        return {args[i + 1] for i, a in enumerate(args[:-1]) if a == "-e"}

    def test_a_setting_of_zero_reaches_the_container(self):
        got = self.passed(self.run_launcher({"STRATA_PREFILL_MMQ": "0", "STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_SPEC": "4"}))
        self.assertIn("STRATA_PREFILL_MMQ=0", got, "the documented FP16 prompt path switch must reach the engine")
        self.assertIn("STRATA_ROUTE_TAIL_SKIP=0", got)
        self.assertIn("STRATA_SPEC=4", got)

    def test_the_launchers_own_defaults_can_be_set_to_zero(self):
        got = self.passed(self.run_launcher({"STRATA_VERIFY_NO_HOST": "0"}))
        self.assertIn("STRATA_VERIFY_NO_HOST=0", got, "the config's override of the port default (the engine reads its value)")
        self.assertNotIn("STRATA_VERIFY_NO_HOST=1", got)
        self.assertIn("STRATA_VERIFY_DEVICE_PLAN=1", got, "the other defaults stay")

    def test_an_empty_value_is_still_not_set_and_the_launchers_own_settings_stay_out(self):
        got = self.passed(self.run_launcher({"STRATA_EMPTY_SETTING": "", "STRATA_SYCL_IMAGE": "strata-sycl-dev"}))
        self.assertFalse([g for g in got if g.startswith("STRATA_EMPTY_SETTING")], "an empty value means not set")
        self.assertFalse([g for g in got if g.startswith("STRATA_SYCL_")], "STRATA_SYCL_* are the launcher's own")


if __name__ == "__main__":
    unittest.main()
