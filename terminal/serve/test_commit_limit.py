"""The engine's Windows commit-limit warning (#1607) becomes engine.info["commit_warning"]. The pattern is pinned to the engine's own text: the first line below is
the one a Strix Halo start wrote (par/soak1117/evidence/engine-log-1304.txt, line 58), the second is what include/strata/platform/host_limits.hpp prints for other numbers."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

from serve import commit_limit
from serve.server import StrataEngine

SOAK_LINE = ("strata serve: WARNING the Windows commit limit is 92.8 GiB (RAM + page file) and 0.8 GiB of it is left, but this run plans 2.0 GiB of host allocations "
             "and keeps 4.0 GiB free: it can end at the limit with no message (exit code 0xC0000409, #1607) while RAM is still free. Raise the page file (System managed, "
             "or RAM-sized) or lower --conversation-cache-mib / --max-context / --prefill")
OTHER_LINE = SOAK_LINE.replace("92.8", "31.9").replace("0.8 GiB", "1.5 GiB").replace("plans 2.0", "plans 18.0")
INFO_LINE = "strata serve: Windows commit: 0.8 GiB left of a 92.8 GiB limit (RAM + page file)"


class CommitWarning(unittest.TestCase):
    def test_the_engines_line_gives_its_numbers(self):
        log = "strata generate: PCIe probe\n" + INFO_LINE + "\n" + SOAK_LINE + "\nstrata serve: every expert is in the GPU cache\n"
        self.assertEqual(commit_limit.commit_warning(log), {"limit_gib": 92.8, "left_gib": 0.8, "planned_gib": 2.0, "floor_gib": 4.0})

    def test_the_last_warning_of_the_log_wins_and_other_numbers_read(self):
        got = commit_limit.commit_warning(SOAK_LINE + "\n" + OTHER_LINE + "\n")
        self.assertEqual(got, {"limit_gib": 31.9, "left_gib": 1.5, "planned_gib": 18.0, "floor_gib": 4.0})

    def test_no_warning_no_fact(self):
        for text in ("", None, INFO_LINE + "\n", "strata serve: ready\n", "WARNING the Windows commit limit is soon\n"):
            self.assertIsNone(commit_limit.commit_warning(text), repr(text))

    def test_the_pattern_follows_the_engines_format_string(self):
        # host_limits.hpp builds the text from these pieces: if the engine's words change, this fails here instead of the notice silently never showing
        header = Path(__file__).resolve().parents[1] / "include" / "strata" / "platform" / "host_limits.hpp"
        if not header.exists():
            self.skipTest("the engine headers are not in this checkout")
        source = header.read_text(encoding="utf-8")
        for piece in ("WARNING the Windows commit limit is %.1f GiB (RAM + page file) and %.1f GiB of it is left, but this run ",
                      "plans %.1f GiB of host allocations and keeps %.1f GiB free: it can end at the limit with no message "):
            self.assertIn(piece, source)

    def test_the_server_applies_it_once_per_start_and_clears_it(self):
        source = (Path(__file__).resolve().parent / "server.py").read_text(encoding="utf-8")
        self.assertIn('self.info.pop("commit_warning", None)', source)
        self.assertIn("commit_limit.commit_warning(start_log_text(log, log_start))", source)
        self.assertIn('self.info["commit_warning"] = warned', source)


FAKE = r"""
import sys
args = sys.argv[1:]
sys.stderr.write('strata serve: Windows commit: 0.8 GiB left of a 92.8 GiB limit (RAM + page file)' + chr(10))
if '--warn' in args:
    sys.stderr.write(WARNING + chr(10))
sys.stderr.flush()
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
"""


class RealEngineStart(unittest.TestCase):
    """A real process and a real pipe (a fake engine program that writes the engine's own line to its stderr, which is the log), like serve/test_drafting_retry.py."""

    def make(self, *extra):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        py = root / "native.py"
        source = "WARNING = " + repr(SOAK_LINE.split("strata serve: ", 1)[1]) + chr(10) + FAKE
        if os.name == "nt":
            py.write_text(source)
            script = root / "native.cmd"
            script.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            script = py
            py.write_text("#!" + sys.executable + chr(10) + source)
            py.chmod(0o700)
        engine = StrataEngine(str(script), ["--pack", "/d/packs/g", *extra], log=str(root / "engine.log"))
        self.addCleanup(engine.close)
        return engine

    def test_a_start_whose_log_has_the_warning_reports_it_in_the_engine_facts(self):
        engine = self.make("--warn")
        self.assertEqual(engine.info["commit_warning"], {"limit_gib": 92.8, "left_gib": 0.8, "planned_gib": 2.0, "floor_gib": 4.0})

    def test_a_start_without_the_warning_reports_none(self):
        engine = self.make()
        self.assertNotIn("commit_warning", engine.info)


if __name__ == "__main__":
    unittest.main()
