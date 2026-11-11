"""The same rules for DeepSeek's own engine package (engine-dsv41/): a loaded DeepSeek model is named, asked about, stopped after the download, the
package is installed NEXT TO the old one, every saved run config is pointed at it, and the model is started again (or not) with the same settings.
Nothing replaces or removes the package folder a loaded model runs from, and a refusal happens before any download."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_control import dsv41_paths, engine_update, signing, updater  # noqa: E402
from tests.terminal.test_dsv41_engine import Dsv41Engine  # noqa: E402

LINUX = sys.platform.startswith("linux")
RUNNING = [{"profile": "launch-deepseek", "state": "ready"}]
EXE = dsv41_paths.exe_name()


class Script:
    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def __call__(self, question, default):
        self.asked.append((question, default))
        return self.answers.pop(0) if self.answers else default


class DeepSeekWithAModel(Dsv41Engine):
    test_downloads_verifies_installs_beside_the_engine = None
    test_a_cut_download_resumes = None
    test_no_package_for_this_pc_is_a_plain_line_not_a_crash = None
    test_a_feed_signed_with_another_key_is_refused = None

    def setUp(self):
        super().setUp()
        self.dest = self.root / "data" / "engine-dsv41"
        self.url = self.server.url + "/strata-feed.json"
        self.publish("0.1.49")
        self.old = self.install()
        self.publish("0.1.49.1")
        self.install_dir = self.root / "data"
        self.cfg = self.root / "run" / "launch-deepseek.json"
        self.cfg.parent.mkdir()
        self.cfg.write_text(json.dumps({"exe": str(self.old), "args": ["--pack", "p"], "tokenizer": "t", "lib_dirs": [str(self.old.parent)]}))
        for p in (mock.patch.object(dsv41_paths, "saved_configs", return_value=[self.cfg]),
                  mock.patch.object(dsv41_paths, "test_start", return_value=(True, "strata-dsv41 0.1.49.1 build x"))):
            p.start()
            self.addCleanup(p.stop)
        self.lines, self.events, self.stops, self.starts = [], [], [], []

    def exe_in_config(self):
        return Path(json.loads(self.cfg.read_text())["exe"])

    def folders(self):
        return sorted(p.name.split("-")[0] for p in self.dest.iterdir() if p.is_dir())

    def ctx(self, **over):
        values = dict(components=self.root / "c", source=self.url, trusted=signing.load_trusted_keys(), hardware=self.hw, serving=lambda: list(RUNNING),
                      emit=self.events.append, environ={**os.environ, "STRATA_INSTALL_DIR": str(self.install_dir)})
        values.update(over)
        return updater.Context(**values)

    def gate(self, *answers, nobody=False, **flags):
        self.script = Script(*answers)
        calls = {"stopped": False}

        def stop(model):
            self.stops.append((model.profile, len(self.server.requests), self.exe_in_config()))
            calls["stopped"] = True

        def start(model):
            self.starts.append((model.profile, self.exe_in_config()))
            return True, f"{model.label} is serving again with the same settings."
        gate = engine_update.ModelGate(ask=None if nobody else self.script, say=self.lines.append, emit=self.events.append, stop=stop, start=start, **flags)
        self.calls = calls
        return gate

    def serving_until_stopped(self):
        return lambda: [] if self.calls["stopped"] else list(RUNNING)

    def update(self, gate, **over):
        ctx = self.ctx(serving=self.serving_until_stopped(), **over)
        return engine_update.update_with_models(ctx, gate, control=False)

    def test_yes_downloads_first_then_stops_then_points_the_config_at_the_new_package_and_starts_again(self):
        out = self.update(self.gate(True, True))
        r = out["results"][-1] if out["results"][-1]["component"] == "dsv41" else next(x for x in out["results"] if x["component"] == "dsv41")
        self.assertEqual((out["exit_code"], r["status"], r["installed_after"]), (0, "updated", "0.1.49.1"), out)
        profile, requests_at_stop, config_at_stop = self.stops[0]
        self.assertGreaterEqual(requests_at_stop, 3, "the package was already downloaded when the model was stopped")
        self.assertEqual(config_at_stop, self.old, "the run config named the old package until the model was stopped")
        self.assertEqual(self.exe_in_config().parent.name.split("-")[0], "0.1.49.1")
        self.assertEqual(self.starts, [("launch-deepseek", self.exe_in_config())], "started again from the same profile, now on the new package")
        self.assertEqual(self.folders(), ["0.1.49", "0.1.49.1"], "the old package stays")
        self.assertTrue(self.old.is_file())
        self.assertEqual(len(self.script.asked), 2)

    def test_no_changes_nothing(self):
        before = len(self.server.requests)
        with self.assertRaises(engine_update.Declined):
            self.update(self.gate(False))
        self.assertEqual(self.stops, [])
        self.assertEqual(self.folders(), ["0.1.49"])
        self.assertEqual(self.exe_in_config(), self.old)
        self.assertEqual([r for r in self.server.requests[before:] if r[1].endswith(".tar.gz")], [])

    def test_nothing_without_a_gate_not_even_a_download(self):
        before = len(self.server.requests)
        with self.assertRaises(updater.Busy):
            updater.run_update(self.ctx(), control=False)
        self.assertEqual([r for r in self.server.requests[before:] if r[1].endswith(".tar.gz")], [])
        self.assertEqual((self.folders(), self.exe_in_config()), (["0.1.49"], self.old))

    def test_a_script_without_stop_model_is_refused_with_the_model_named(self):
        with self.assertRaises(updater.UpdateError) as caught:
            self.update(self.gate(nobody=True))
        self.assertEqual(caught.exception.exit_code, 7)
        self.assertIn("deepseek", str(caught.exception))
        self.assertIn("--stop-model", str(caught.exception))
        self.assertEqual(self.folders(), ["0.1.49"])

    def test_a_package_that_will_not_start_changes_no_config_and_the_model_can_start_on_the_old_one(self):
        with mock.patch.object(dsv41_paths, "test_start", return_value=(False, "it did not start correctly (libcudart.so.13: not found)")):
            out = self.update(self.gate(True, True))
        self.assertEqual(out["exit_code"], 6, out)
        self.assertEqual(self.exe_in_config(), self.old)
        self.assertEqual(self.folders(), ["0.1.49"])
        self.assertEqual(self.starts, [("launch-deepseek", self.old)], "started again, on the engine it had")
        self.assertIn("old engine is unchanged", self.script.asked[1][0])

    @unittest.skipUnless(LINUX, "the stand-in model process runs from the engine's own file (a copy of this Python)")
    def test_the_file_a_loaded_model_runs_from_is_never_touched(self):
        # The stand-in is a copy of the Python running the tests, started under the engine's file name: the process's program
        # is that file, as a loaded model's engine is (dsv41_paths.in_use looks at the program's path).  Not a copy of
        # /bin/sleep: on Ubuntu 26.04 that is coreutils' multi-call program, which refuses any other program name
        # ("coreutils: unknown program 'strata-dsv41'") and exits at once.
        import shutil
        shutil.copy(os.path.realpath(sys.executable), self.old)
        self.old.chmod(0o755)
        proc = subprocess.Popen([str(self.old), "-I", "-S", "-c", "import time; time.sleep(300)"])
        self.addCleanup(lambda: (proc.kill(), proc.wait()))
        time.sleep(0.3)
        before = self.old.read_bytes()
        out = self.update(self.gate(nobody=True, allow_stop=True, start_again=True))         # (the stand-in is not stopped by the fake stop)
        self.assertEqual(out["exit_code"], 0, out)
        self.assertTrue(self.old.is_file())
        self.assertEqual(self.old.read_bytes(), before)
        self.assertEqual(proc.poll(), None, "the running program was left running")
        self.assertEqual(self.folders(), ["0.1.49", "0.1.49.1"])


if __name__ == "__main__":
    unittest.main()
