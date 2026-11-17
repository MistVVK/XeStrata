"""`strata status` and the plain monitor text run on the standard library alone (requirements-monitor.txt: "The core server and `strata status` use stdlib").

A Terminal install made without textual (so without rich) used to end `strata status` in "ModuleNotFoundError: No module named 'rich'": the plain renderer's
per-card line took the card-name shortener from strata_cli/overview.py, whose top imports rich. These tests run the command in a fresh interpreter in which
rich and textual cannot be imported, so they do not depend on what the box has installed."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

BLOCK = '''
import sys
class NoRichOrTextual:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("rich", "textual"):
            raise ModuleNotFoundError("No module named '" + name.split(".")[0] + "'", name=name.split(".")[0])
sys.meta_path.insert(0, NoRichOrTextual())
sys.path.insert(0, %(root)r)
'''

SNAPSHOT = {
    "sampled_at": None, "model": {"id": "m", "loaded": True}, "instance": {"state": "ready"},
    "work": {"active": 1, "occupied_slots": 1, "slots_effective": 2, "waiting": 0, "queued": 0},
    "rates": {"answer_tok_s": 61.2, "answer_window_s": 5.0, "prefill_tok_s": 1500.0},
    "topology": {"mode": "single"}, "totals": {"requests": 3},
    "devices": [{"index": 0, "name": "NVIDIA GeForce RTX 5070", "mem_used": 6 * 2 ** 30, "mem_total": 12 * 2 ** 30, "util": 40, "temp": 61, "power": 150}],
    "recent": [{"finish": "stop", "prompt_tokens": 100, "output_tokens": 50, "duration_s": 1.2}],
}


def run_without_rich(code: str) -> subprocess.CompletedProcess:
    # -I would drop PYTHONPATH and the cwd; the code puts the tree on sys.path itself
    return subprocess.run([sys.executable, "-c", BLOCK % {"root": str(ROOT)} + code], capture_output=True, text=True, cwd=str(ROOT), timeout=120,
                          env={**os.environ, "PYTHONPATH": ""})


class StatusOnTheStdlib(unittest.TestCase):
    def test_the_blocker_works(self):
        r = run_without_rich("try:\n    import rich\nexcept ModuleNotFoundError:\n    print('blocked')\n")
        self.assertEqual(r.stdout.strip(), "blocked", r.stderr)

    def test_every_plain_view_renders_wide_and_narrow_with_a_card_listed(self):
        code = ("import json\nfrom strata_cli.render import VIEWS, render_view\n"
                "snap = json.loads(%r)\n"
                "for view in VIEWS:\n    for w, h in ((120, 40), (100, 30), (80, 24), (50, 12)):\n        render_view(snap, view, w, h)\n"
                "print(render_view(snap, 'overview', 80, 24))\n"
                "import sys\nassert 'rich' not in sys.modules and 'textual' not in sys.modules\n") % json.dumps(SNAPSHOT)
        r = run_without_rich(code)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("RTX 5070", r.stdout)                              # the shortened name of the card: the line that used to need rich
        self.assertNotIn("NVIDIA GeForce", r.stdout)

    def test_the_status_command_prints_the_overview(self):
        code = ("import json, sys\nfrom strata_cli import client, monitor_commands\n"
                "snap = json.loads(%r)\n"
                "client.Endpoint.snapshot = lambda self: snap\n"
                "code = monitor_commands.main(['status', '--url', 'http://127.0.0.1:9'])\n"
                "assert 'rich' not in sys.modules and 'textual' not in sys.modules\n"
                "sys.exit(code)\n") % json.dumps(SNAPSHOT)
        r = run_without_rich(code)
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])
        self.assertIn("RTX 5070", r.stdout)

    def test_the_interactive_monitor_says_what_is_missing_instead_of_a_traceback(self):
        code = ("import io, sys\nfrom strata_cli import client, monitor_commands\n"
                "client.Endpoint.snapshot = lambda self: {}\n"
                "class Tty(io.StringIO):\n    def isatty(self):\n        return True\n"
                "real = sys.stdout\nsys.stdout = Tty()\n"
                "code = monitor_commands.main(['monitor', '--url', 'http://127.0.0.1:9'])\n"
                "sys.stdout = real\nsys.exit(code)\n")
        r = run_without_rich(code)
        self.assertEqual(r.returncode, 5, r.stderr[-1500:])
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("monitor component missing", r.stderr + r.stdout)


class TheNameStaysWhereItWas(unittest.TestCase):
    def test_overview_still_exports_short_gpu(self):
        from strata_cli import overview, render
        self.assertIs(overview.short_gpu, render.short_gpu)
        self.assertEqual(render.short_gpu("NVIDIA GeForce RTX 5070"), "RTX 5070")
        self.assertEqual(render.short_gpu("AMD Radeon RX 9070 XT"), "RX 9070 XT")
        self.assertEqual(render.short_gpu(None), "graphics card")


if __name__ == "__main__":
    unittest.main()
