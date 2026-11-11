"""A launch, a setup and a model download never update an engine by themselves (owner, 11-01; the same shape as #4823: "downloading a model changed my
engine without asking").  An engine update is the person's choice, and never beside a loaded model.

* Qwen: `strata launch` with an engine older than the model needs warns and goes on; it updates only on a yes (--yes, or the answer on the screens),
  and never with a model loaded.  No engine installed at all is still fetched (nothing runs on it yet).
* The app's model download for Qwen never fetches an engine (control_engine_step), as before.
"""
from __future__ import annotations

import ast
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_cli.launch import settings as S  # noqa: E402
from strata_cli.launch.runner import Runner  # noqa: E402
from strata_control import engine_component as ec, engine_update, updater  # noqa: E402
from tests.terminal.test_launch_engine_first import FakeProc, qwen_catalog, update_events  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OLD = {"usable": True, "engine": {"version": "0.1.45", "backend": "cuda"}, "notice": None}
NEW = {"usable": True, "engine": {"version": "0.1.50", "backend": "cuda"}, "notice": None}
NONE = {"usable": False, "engine": None, "notice": "No engine for this PC was found"}
RUNNING = [{"profile": "launch-demo", "state": "ready"}]


class QwenStaleEngine(unittest.TestCase):
    def ensure(self, statuses, *, serving=(), confirm=None, assume_yes=False):
        spawns, seen = [], []
        r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=qwen_catalog(), emit=seen.append,
                   confirm=confirm, assume_yes=assume_yes)
        r.check_cancel = lambda: None
        with mock.patch.object(ec, "status", side_effect=list(statuses)), \
                mock.patch.object(Runner, "needed_engine", return_value=(0, 1, 50)), \
                mock.patch.object(updater, "serving_now", return_value=list(serving)), \
                mock.patch.object(Runner, "_spawn", side_effect=lambda argv, **kw: spawns.append(argv) or FakeProc(update_events("updated"), 0)):
            r.ensure_engine()
        return spawns, r, seen

    def test_a_loaded_model_blocks_the_update_even_with_yes(self):
        for kwargs in ({"assume_yes": True}, {"confirm": lambda *a: True}, {}):
            spawns, r, seen = self.ensure([OLD], serving=RUNNING, **kwargs)
            self.assertEqual(spawns, [], kwargs)
            self.assertTrue(any("running, so Strata did not update the engine" in n for n in r.notices), r.notices)
            self.assertTrue(any("strata launch --stop" in n for n in r.notices))

    def test_no_yes_no_update(self):
        spawns, r, _ = self.ensure([OLD])
        self.assertEqual(spawns, [])
        self.assertIn("Strata did not update it. Run `strata update` (or start with --yes)", r.notices[0])

    def test_no_to_the_question_no_update(self):
        asked = []
        spawns, r, _ = self.ensure([OLD], confirm=lambda t, b: asked.append((t, b)) or False)
        self.assertEqual(spawns, [])
        self.assertEqual(len(asked), 1)
        self.assertIn("older than this model needs", asked[0][1])

    def test_yes_to_the_question_updates(self):
        spawns, r, _ = self.ensure([OLD, NEW], confirm=lambda t, b: True)
        self.assertEqual(len(spawns), 1)
        self.assertEqual(r.notices, [])

    def test_yes_flag_updates_when_nothing_is_loaded(self):
        spawns, r, _ = self.ensure([OLD, NEW], assume_yes=True)
        self.assertEqual(len(spawns), 1)

    def test_no_engine_at_all_is_still_fetched_because_nothing_runs_on_it(self):
        spawns, r, _ = self.ensure([NONE, NEW])
        self.assertEqual(len(spawns), 1)

    def test_an_engine_that_is_new_enough_asks_nothing(self):
        asked = []
        spawns, r, _ = self.ensure([NEW], confirm=lambda t, b: asked.append(t))
        self.assertEqual((spawns, asked), ([], []))

    def test_plan_says_it_will_ask(self):
        r = Runner(S.LaunchSettings(family="qwen", model="Q2_0", context=32768), catalog=qwen_catalog(), emit=lambda e: None)
        with mock.patch.object(ec, "status", return_value=OLD), mock.patch.object(Runner, "needed_engine", return_value=(0, 1, 50)), \
                mock.patch("strata_cli.launch.runner.install_isolation_words", return_value=[]):
            words = r.preflight_words()
        self.assertTrue(any("will ask before updating it" in w for w in words), words)


class SourceRules(unittest.TestCase):
    def calls(self, path, function):
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function:
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Call):
                        f = inner.func
                        names.add(f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else "")
        return names

    def test_the_apps_qwen_model_download_never_fetches_an_engine(self):
        names = self.calls("setup.py", "control_engine_step")
        for forbidden in ("engine_from_feed", "get_prebuilt", "get_prebuilt_hip", "update_installed_engine", "run", "urlopen", "check_call", "Popen"):
            self.assertNotIn(forbidden, names)


if __name__ == "__main__":
    unittest.main()
