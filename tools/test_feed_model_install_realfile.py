"""The release gate for a feed model's install, as a test in `python tools/check` (py-tools): a REAL synthetic GGUF installed through setup.py from a signed
test feed (throw-away key) and a local Hugging Face stand-in. See tools/feed_model_install_gate.py for what it does and checks; it catches the Q4-P8 rehearsal's
F2 (KeyError 'mmproj'), F3 (the Flash-Next draft layer fetched for `drafter: none`), F4 (the shipped expert profile of another shape) and F5 (no RAM budget, a
second copy of the experts).

    python -m pytest tools/test_feed_model_install_realfile.py

Needs numpy (no llama.cpp checkout and no STRATA_TEST_LLAMA_CPP any more: gguf-py is vendored in third_party/strata-gguf-py); without it the test is SKIPPED
with that reason, which `tools/check` prints. The engine step (one request) is the gate script's own --engine option and needs a box with an engine built for the file's shape.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import feed_model_install_gate as gate  # noqa: E402


class FeedModelInstallFromRealFile(unittest.TestCase):
    def test_a_small_synthetic_model_installs_through_setup(self):
        try:
            gate.check_vendored_gguf_py()
            import numpy  # noqa: F401
        except gate.NotRun as e:
            self.skipTest(str(e))
        except ImportError:
            self.skipTest("numpy is not installed (tools/q4geo_synth.py needs it)")
        work = Path(tempfile.mkdtemp(prefix="strata-feed-install-"))
        self.addCleanup(shutil.rmtree, work, True)
        try:
            res = gate.check(work, None, gate.TINY, lambda s: None)
        except gate.NotRun as e:
            self.skipTest(str(e))
        self.assertEqual(res["problems"], [], "\n".join(res["out"].splitlines()[-30:]))


if __name__ == "__main__":
    unittest.main()
