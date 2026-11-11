"""An engine update beside a loaded model (owner, 11-01: "it updated the engine by itself mid-session and he had a model loaded").

The rule under test: an update never starts by itself while a model is loaded; nothing replaces or removes the engine a running model uses; the
update is the person's choice: warn naming the model, download while it keeps serving, stop it cleanly, update, ask whether to start it again with the
same settings; a failed update leaves the old engine in use and the model can be started again; a script is never asked anything.

Real pieces: the owner runtime starts and stops a real (stub) engine from the installed package, so "serving", "stopped" and "serving again on the new
engine" are read from the runtime and from the file the stub engine writes when it starts.  Only the questions are scripted.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import socket
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_cli import update_cmd  # noqa: E402
from strata_control import engine_component as ec, engine_update, profiles, runtime, updater  # noqa: E402
from tests.terminal import update_fixture as fx  # noqa: E402
from tests.terminal.test_updater import PLATFORM, UpdaterCase  # noqa: E402

PROFILE = "launch-demo"
NAME = "Demo model"


class Script:
    """The person at the keyboard: answers in order, remembers every question."""

    def __init__(self, *answers):
        self.answers, self.asked = list(answers), []

    def __call__(self, question, default):
        self.asked.append((question, default))
        return self.answers.pop(0) if self.answers else default


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ModelCase(UpdaterCase):
    def setUp(self):
        super().setUp()
        self.install_current("0.1.45")
        self.lines: list[str] = []
        self.events: list[dict] = []

    # ------------------------------------------------------------------ a real loaded model
    def start_model(self, profile=PROFILE, name=NAME, port=None):
        cfg = json.loads(self.config.read_text())
        cfg.update(exe=ec.REFERENCE_PREFIX + ec.COMPONENT_ID, backend="cuda", host="127.0.0.1", port=port or free_port(), model_name=name,
                   log=str(self.root / f"{profile}.engine.log"))
        cfg.pop("lib_dirs", None)
        path = self.root / f"run-{profile}.json"
        path.write_text(json.dumps(cfg))
        profiles.create(profile, str(path))
        got = runtime.start(profile, wait=60)
        self.assertTrue(got.get("ok"), got)
        self.addCleanup(self._stop, profile)
        return profile

    def _stop(self, profile):
        try:
            runtime.stop(profile, timeout=20)
        except Exception:                                  # noqa: BLE001
            pass

    def state(self, profile=PROFILE):
        return runtime.status(profile).get("state")

    def engine_of_last_start(self):
        """Which engine folder the stub engine ran from the last time it started."""
        return Path((self.root / "ran.txt").read_text().splitlines()[0]).parent.name

    # ------------------------------------------------------------------ the update
    def gate(self, *answers, **flags):
        self.script = Script(*answers)
        ask = None if flags.pop("nobody", False) else self.script
        return engine_update.ModelGate(ask=ask, say=self.lines.append, emit=self.events.append, **flags)

    def update(self, gate, **over):
        over.setdefault("serving", updater.serving_now)
        ctx = self.ctx(events=self.events, **over)
        return engine_update.update_with_models(ctx, gate, control=False)

    def publish_new(self, version="0.1.46", source=None):
        self.publish((version, source or fx.GOOD_ENGINE))

    def said(self):
        return " ".join(self.lines)


class WithAModel(ModelCase):
    def setUp(self):
        super().setUp()
        self.start_model()
        self.publish_new()

    # ---- yes, then start it again
    def test_yes_and_start_again_stops_after_the_download_updates_and_starts_the_same_model_on_the_new_engine(self):
        archives_when_stopped = []
        real_stop = engine_update.stop_model

        def stop(model):
            archives_when_stopped.append(len(self.archive_requests()))
            real_stop(model)
        gate = self.gate(True, True)
        gate._stop = stop
        out = self.update(gate)
        self.assertEqual((out["exit_code"], out["results"][0]["status"]), (0, "updated"), out)
        self.assertEqual(archives_when_stopped, [1], "the new engine was downloaded while the model kept serving; it was stopped after")
        self.assertEqual(self.state(), "ready", "the model is serving again")
        self.assertEqual(self.resolved(), "0.1.46")
        self.assertEqual(self.engine_of_last_start(), f"0.1.46-{PLATFORM}-cuda-cuda13", "the restarted model runs on the NEW engine")
        # the warning came first, named the model, and said what would happen
        first, second = self.script.asked
        self.assertIn(NAME, first[0])
        self.assertFalse(first[1], "stopping a model is never the default answer")
        self.assertIn(NAME, second[0])
        self.assertIn("same settings", second[0])
        self.assertTrue(second[1])
        self.assertIn(NAME, self.said())
        self.assertIn("keeps serving", self.said())
        phases = [e.get("phase") for e in self.events if e.get("event") == "models"]
        self.assertEqual(phases, ["consent", "stopping", "stopped", "restarting", "restarted"])
        self.assertEqual(out["models"]["restarts"][0]["restarted"], True)
        self.assert_user_data_untouched()

    def test_the_old_engine_stays_installed_and_the_restart_uses_the_saved_settings_untouched(self):
        before = (profiles.load(PROFILE)[0], Path(profiles.load(PROFILE)[0]["source_config"]).read_bytes())
        self.update(self.gate(True, True))
        after = (profiles.load(PROFILE)[0], Path(profiles.load(PROFILE)[0]["source_config"]).read_bytes())
        self.assertEqual(before, after, "the model's profile and run config are exactly what they were")
        self.assertEqual(self.engine_folders(), [f"0.1.45-{PLATFORM}-cuda-cuda13", f"0.1.46-{PLATFORM}-cuda-cuda13"])

    # ---- yes, but do not start it again
    def test_yes_and_no_to_start_again_leaves_it_stopped_and_says_how_to_start_it(self):
        out = self.update(self.gate(True, False))
        self.assertEqual(out["exit_code"], 0)
        self.assertEqual(self.state(), "stopped")
        self.assertEqual(self.resolved(), "0.1.46")
        self.assertIn("is stopped. Start it when you want with", self.said())
        self.assertEqual(out["models"]["restarts"][0]["restarted"], False)
        self.assertEqual([e["phase"] for e in self.events if e.get("event") == "models"][-1], "left-stopped")

    # ---- no
    def test_no_changes_nothing_downloads_nothing_and_the_model_keeps_serving(self):
        with self.assertRaises(engine_update.Declined) as caught:
            self.update(self.gate(False))
        self.assertEqual(caught.exception.exit_code, 0)
        self.assertEqual(self.state(), "ready")
        self.assertEqual(self.resolved(), "0.1.45")
        self.assertEqual(self.archive_requests(), [], "nothing was downloaded")
        self.assertEqual(len(self.script.asked), 1)

    # ---- the update fails
    def test_a_new_engine_that_will_not_start_leaves_the_old_one_in_use_and_the_model_can_be_started_again(self):
        self.publish_new(source=fx.BAD_ENGINE)
        out = self.update(self.gate(True, True))
        self.assertEqual(out["exit_code"], 6, out)
        self.assertEqual(out["results"][0]["status"], "failed")
        self.assertEqual(self.resolved(), "0.1.45", "the old engine is still the one in use")
        self.assertEqual(self.engine_folders(), [f"0.1.45-{PLATFORM}-cuda-cuda13"], "the engine that failed is not installed")
        self.assertEqual(self.state(), "ready", "the model was started again on the old engine")
        self.assertEqual(self.engine_of_last_start(), f"0.1.45-{PLATFORM}-cuda-cuda13")
        question = self.script.asked[1][0]
        self.assertIn("old engine is unchanged", question)
        self.assertIn(NAME, question)

    def test_a_failed_update_and_no_to_start_again_leaves_the_model_stopped_with_the_old_engine(self):
        self.publish_new(source=fx.BAD_ENGINE)
        out = self.update(self.gate(True, False))
        self.assertEqual(out["exit_code"], 6)
        self.assertEqual(self.state(), "stopped")
        self.assertEqual(self.resolved(), "0.1.45")

    def test_a_download_that_fails_never_stops_the_model(self):
        (next((self.release / "engine").glob("*.tar.gz"))).unlink()     # the server answers 404 for the archive (a cut connection is resumed in the same run now)
        out = self.update(self.gate(True, True))
        self.assertEqual(out["exit_code"], 6, out)
        self.assertEqual(self.state(), "ready", "the model was not stopped for a download that failed")
        self.assertEqual(self.resolved(), "0.1.45")
        self.assertEqual(out["models"]["stopped"], [])
        self.assertEqual(len(self.script.asked), 1, "no 'start it again' question: it was never stopped")

    # ---- the model does not start again
    def test_a_model_that_does_not_start_again_is_said_plainly_and_the_exit_code_is_8(self):
        gate = self.gate(True, True)
        real_stop = engine_update.stop_model

        def stop_and_break(model):
            real_stop(model)
            Path(profiles.load(PROFILE)[0]["source_config"]).write_text("{ not a run config")        # whatever would stop it starting
        gate._stop = stop_and_break
        out = self.update(gate)
        self.assertEqual(out["exit_code"], engine_update.EXIT_NOT_RESTARTED, out)
        self.assertEqual(out["results"][0]["status"], "updated", "the update itself is done and stays")
        self.assertEqual(self.resolved(), "0.1.46")
        self.assertIn("did not start again", self.said())
        self.assertEqual(self.state(), "stopped", "nothing is left half alive")
        self.assertEqual([r["restarted"] for r in out["models"]["restarts"]], [False])

    # ---- nobody can be asked
    def test_a_script_is_never_asked_and_nothing_is_downloaded_or_changed(self):
        with self.assertRaises(updater.UpdateError) as caught:
            self.update(self.gate(nobody=True))
        self.assertEqual(caught.exception.exit_code, 7)
        text = str(caught.exception)
        self.assertIn(NAME, text)
        self.assertIn("--stop-model", text)
        self.assertIn("strata launch --stop", text)
        self.assertIn("nothing was changed", text)
        self.assertEqual(self.archive_requests(), [])
        self.assertEqual((self.state(), self.resolved()), ("ready", "0.1.45"))

    def test_stop_model_stops_updates_and_leaves_it_stopped_when_start_again_is_not_given(self):
        out = self.update(self.gate(nobody=True, allow_stop=True))
        self.assertEqual(out["exit_code"], 0)
        self.assertEqual((self.state(), self.resolved()), ("stopped", "0.1.46"))
        self.assertIn("Start it when you want with", self.said())

    def test_stop_model_and_start_again_in_a_script_do_it_all_without_a_question(self):
        gate = self.gate(nobody=True, allow_stop=True, start_again=True)
        out = self.update(gate)
        self.assertEqual(out["exit_code"], 0)
        self.assertEqual(self.state(), "ready")
        self.assertEqual(self.engine_of_last_start(), f"0.1.46-{PLATFORM}-cuda-cuda13")
        self.assertEqual(self.script.asked, [])

    def test_start_again_after_a_failed_update_puts_the_model_back_on_the_old_engine(self):
        self.publish_new(source=fx.BAD_ENGINE)
        out = self.update(self.gate(nobody=True, allow_stop=True, start_again=True))
        self.assertEqual(out["exit_code"], 6)
        self.assertEqual((self.state(), self.resolved()), ("ready", "0.1.45"))

    # ---- a server nobody here started (a service)
    def test_a_server_that_strata_did_not_start_is_never_stopped_even_with_stop_model(self):
        with self.assertRaises(updater.UpdateError) as caught:
            self.update(self.gate(nobody=True, allow_stop=True), serving=lambda: [{"profile": "a running Strata server", "state": "serving"}])
        self.assertEqual(caught.exception.exit_code, 7)
        self.assertIn("did not start it from here", str(caught.exception))
        self.assertIn("Stop it yourself", str(caught.exception))
        self.assertEqual(self.archive_requests(), [])
        self.assertEqual(self.state(), "ready")

    # ---- offline and a feed that does not answer
    def test_an_unreachable_feed_is_said_before_anything_is_stopped(self):
        for dead in ("http://127.0.0.1:9/strata-feed.json", str(self.root / "no-such-feed")):
            with self.assertRaises(Exception) as caught:
                self.update(self.gate(True, True), source=dead)
            self.assertTrue("could not be read" in str(caught.exception) or "so the update check could not run" in str(caught.exception),
                            str(caught.exception))            # a folder that is gone: the old sentence; a host nobody answers for: the network rule's (W7)
            self.assertEqual(self.state(), "ready", dead)
        self.assertEqual(self.resolved(), "0.1.45")

    def test_a_feed_that_never_answers_gives_up_and_leaves_the_model_alone(self):
        hang = socket.socket()
        hang.bind(("127.0.0.1", 0))
        hang.listen(5)
        self.addCleanup(hang.close)
        started = time.monotonic()
        with self.assertRaises(Exception) as caught:
            self.update(self.gate(True, True), source=f"http://127.0.0.1:{hang.getsockname()[1]}/strata-feed.json", feed_timeout=1.0)
        self.assertLess(time.monotonic() - started, 20)
        self.assertIn("did not answer within", str(caught.exception))       # the network rule (W7): the host, and that nothing was changed
        self.assertIn("so the update check could not run", str(caught.exception))
        self.assertEqual((self.state(), self.resolved()), ("ready", "0.1.45"))

    def test_nothing_to_update_asks_nothing_and_leaves_the_model_alone(self):
        self.publish((("0.1.45", fx.GOOD_ENGINE)))
        out = self.update(self.gate(True, True))
        self.assertEqual((out["exit_code"], out["results"][0]["status"]), (0, "current"))
        self.assertEqual(self.script.asked, [])
        self.assertEqual(self.state(), "ready")


class StartedDuringTheUpdate(ModelCase):
    """A model started after the person said yes is a different model: nobody agreed to stop it."""

    def setUp(self):
        super().setUp()
        self.start_model()
        self.publish_new()

    def run_update(self, gate):
        """Runs the update; a second model is started as soon as the download is done (before the engine is put in place)."""
        real = updater._download

        def download(*args, **kwargs):
            got = real(*args, **kwargs)
            self.start_model("launch-second", "Second model")
            return got
        ctx = self.ctx(serving=updater.serving_now, events=self.events)
        with mock.patch.object(updater, "_download", download):
            return engine_update.update_with_models(ctx, gate, control=False)

    def test_in_a_script_the_new_model_is_refused_and_nothing_more_is_changed(self):
        out = self.run_update(self.gate(nobody=True, allow_stop=True, start_again=True))
        self.assertEqual(out["exit_code"], 7)
        self.assertEqual(out["results"][0]["status"], "failed")
        self.assertIn("Second model", out["results"][0]["error"])
        self.assertIn("started while the update was running", out["results"][0]["error"])
        self.assertIn("the download is kept", out["results"][0]["error"])
        self.assertEqual(self.resolved(), "0.1.45")
        self.assertEqual(self.state("launch-second"), "ready", "the model nobody agreed to stop was not stopped")
        self.assertEqual(self.state(PROFILE), "ready", "nothing was stopped at all")

    def test_at_the_keyboard_the_person_is_asked_again_and_both_come_back(self):
        out = self.run_update(self.gate(True, True, True, True))
        self.assertEqual(out["exit_code"], 0, out)
        self.assertEqual(self.resolved(), "0.1.46")
        self.assertEqual((self.state(PROFILE), self.state("launch-second")), ("ready", "ready"))
        self.assertTrue(any("Second model" in q for q, _ in self.script.asked))

    def test_at_the_keyboard_no_to_the_new_model_changes_nothing_more(self):
        out = self.run_update(self.gate(True, False))
        self.assertEqual(out["exit_code"], 7)
        self.assertEqual(self.resolved(), "0.1.45")
        self.assertEqual((self.state("launch-second"), self.state(PROFILE)), ("ready", "ready"), "nothing was stopped")


class TwoTerminals(ModelCase):
    def setUp(self):
        super().setUp()
        self.start_model()
        self.publish_new()

    def test_a_second_update_while_the_first_runs_is_refused_and_stops_nothing(self):
        with updater.update_lock(self.components):
            with self.assertRaises(updater.UpdateError) as caught:
                self.update(self.gate(True, True))
        self.assertEqual(caught.exception.exit_code, 7)
        self.assertIn("another update is already running", str(caught.exception))
        self.assertEqual((self.state(), self.resolved()), ("ready", "0.1.45"))
        self.assertEqual(self.script.asked, [], "the model was not even asked about: the lock comes first")

    def test_two_updates_at_once_stop_the_model_once_and_start_it_once(self):
        results = []

        def go():
            try:
                results.append(("ok", self.update(self.gate(True, True))["exit_code"]))
            except updater.UpdateError as exc:
                results.append(("refused", exc.exit_code))
        threads = [threading.Thread(target=go) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(180)
        self.assertEqual(sorted(results), [("ok", 0), ("refused", 7)], results)
        self.assertEqual((self.state(), self.resolved()), ("ready", "0.1.46"))


class WithoutAGate(ModelCase):
    """Anything that runs the update without the questions (the app's button today, a library caller) is refused BEFORE a download, whole."""

    def setUp(self):
        super().setUp()
        self.start_model()

    def bundle(self):
        folder = self.root / "control-bundle"
        (folder / "strata_control").mkdir(parents=True)
        (folder / "strata.py").write_text("print('strata 0.1.46 (python stub)')\n")
        (folder / "strata_control" / "__init__.py").write_text("")
        from feed_tool import build_feed, pack_control
        (self.release / "control").mkdir()
        pack_control(folder, self.release / "control", version="0.1.46", platform=PLATFORM)
        build_feed(self.release, key=self.key, sequence=self.sequence + 1, engine_dir=self.release / "engine", control_dir=self.release / "control")

    def test_engine_and_control_are_both_left_alone_and_nothing_is_downloaded(self):
        self.publish_new()
        self.bundle()
        with self.assertRaises(updater.Busy) as caught:
            updater.run_update(self.ctx(serving=updater.serving_now))
        self.assertEqual(caught.exception.exit_code, 7)
        self.assertEqual(self.archive_requests(), [], "no download beside a loaded model")
        self.assertIsNone(updater.active_control(self.components), "Control did not move on while the engine was refused")
        self.assertEqual((self.state(), self.resolved()), ("ready", "0.1.45"))

    def test_nothing_to_install_is_not_a_busy_error(self):
        self.publish((("0.1.45", fx.GOOD_ENGINE)))
        out = updater.run_update(self.ctx(serving=updater.serving_now), control=False)
        self.assertEqual(out["exit_code"], 0)

    def test_wait_lets_the_model_finish_first(self):
        self.publish_new()
        answers = [[{"profile": PROFILE}], [{"profile": PROFILE}], [], [], []]
        out = updater.run_update(self.ctx(serving=lambda: answers.pop(0) if answers else [], wait=30), control=False)
        self.assertEqual(out["results"][0]["status"], "updated")


class TheCommand(ModelCase):
    """`strata update` as a person or a script runs it."""

    def setUp(self):
        super().setUp()
        self.start_model()
        self.publish_new()
        sp = argparse.ArgumentParser().add_subparsers(dest="command")
        comp = sp.add_parser("components").add_subparsers(dest="components_command")
        update_cmd.add_parsers(sp, comp)
        self.parser = sp

    def run_cmd(self, *flags, asks=None):
        argv = ["update", "--feed", self.server.url + "/strata-feed.json", "--components-dir", str(self.components), "--no-control",
                "--model-config", str(self.config), *flags]
        a = self.parser.choices["update"].parse_args(argv[1:])
        a.command, a.components_command = "update", None
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(update_cmd, "_ask_factory", return_value=asks), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = update_cmd.run(a)
        return code, out.getvalue(), err.getvalue()

    def test_yes_alone_never_stops_a_model(self):
        code, out, err = self.run_cmd("--yes")
        self.assertEqual(code, 7)
        self.assertIn(NAME, err)
        self.assertIn("--stop-model", err)
        self.assertEqual((self.state(), self.resolved()), ("ready", "0.1.45"))
        self.assertEqual(self.archive_requests(), [])

    def test_stop_model_and_start_again(self):
        code, out, err = self.run_cmd("--yes", "--stop-model", "--start-again")
        self.assertEqual(code, 0, (out, err))
        self.assertIn("Engine", out)
        self.assertIn("serving again with the same settings", out)
        self.assertEqual((self.state(), self.resolved()), ("ready", "0.1.46"))

    def test_stop_model_alone_leaves_it_stopped_and_prints_the_start_line(self):
        code, out, err = self.run_cmd("--yes", "--stop-model")
        self.assertEqual(code, 0, (out, err))
        self.assertIn("is stopped. Start it when you want with", out)
        self.assertEqual(self.state(), "stopped")

    def test_the_questions_in_a_terminal(self):
        asks = Script(True, True)
        code, out, err = self.run_cmd(asks=asks)
        self.assertEqual(code, 0, (out, err))
        self.assertEqual(len(asks.asked), 2)
        self.assertEqual((self.state(), self.resolved()), ("ready", "0.1.46"))

    def test_no_in_a_terminal_exits_0_and_says_nothing_was_changed(self):
        code, out, err = self.run_cmd(asks=Script(False))
        self.assertEqual(code, 0)
        self.assertIn("Nothing was changed", out)
        self.assertEqual((self.state(), self.resolved()), ("ready", "0.1.45"))

    def test_the_exit_code_when_a_model_does_not_start_again_is_8(self):
        real = engine_update.start_model
        with mock.patch.object(engine_update, "start_model", return_value=(False, "Demo model did not start again: out of memory.")):
            code, out, err = self.run_cmd("--yes", "--stop-model", "--start-again")
        self.assertEqual(code, 8, (out, err))
        self.assertIn("did not start again", out)
        self.assertIn("the new engine is in use", out)
        self.assertEqual(self.resolved(), "0.1.46")
        del real

    def test_machine_output_has_the_models_and_never_a_question(self):
        code, out, err = self.run_cmd("--output", "json", "--stop-model", "--start-again")
        self.assertEqual(code, 0, (out, err))
        data = json.loads(out)
        self.assertTrue(data["ok"])
        self.assertEqual(data["models"]["stopped"][0]["name"], NAME)
        self.assertTrue(data["models"]["restarts"][0]["restarted"])

    def test_machine_output_with_a_model_and_no_flag_is_error_7_as_json(self):
        code, out, err = self.run_cmd("--output", "json")
        self.assertEqual(code, 7)
        data = json.loads(out)
        self.assertFalse(data["ok"])
        self.assertEqual(data["error"]["code"], 7)
        self.assertIn(NAME, data["error"]["message"])

    def test_jsonl_carries_the_model_events_for_the_app(self):
        code, out, err = self.run_cmd("--output", "jsonl", "--stop-model", "--start-again")
        self.assertEqual(code, 0, (out, err))
        events = [json.loads(line) for line in out.splitlines() if line.startswith("{")]
        phases = [e["phase"] for e in events if e.get("event") == "models"]
        self.assertEqual(phases, ["consent", "stopping", "stopped", "restarting", "restarted"])
        self.assertEqual(events[[e.get("phase") for e in events].index("consent")]["models"][0]["profile"], PROFILE)

    def test_check_names_the_model_and_the_one_next_step(self):
        code, out, err = self.run_cmd("--check")
        self.assertEqual(code, 0)
        self.assertIn(NAME, out)
        self.assertIn("asks before it stops it", out)
        self.assertEqual((self.state(), self.resolved()), ("ready", "0.1.45"), "a check changes nothing")


class AfterTheStop(unittest.TestCase):
    """Found by the real DeepSeek flow: right after a big model was stopped, the next look still saw 'a running Strata server' (its engine had not let
    go of the shared lease yet) and the update was refused half way."""

    def gate(self, **kw):
        stopped = []
        asked = []
        gate = engine_update.ModelGate(ask=lambda q, d: asked.append(q) or True, stop=stopped.append, start=lambda m: (True, "ok"), **kw)
        model = engine_update.RunningModel("launch-x", "X", port=8080)
        gate.consented["launch-x"] = model
        gate.asked, gate.stops = asked, stopped
        return gate, model

    def test_the_update_waits_until_the_stopped_engine_has_let_go(self):
        gate, _ = self.gate(settle_s=30)
        answers = [[{"profile": "launch-x"}], [{"profile": "a running Strata server"}], [{"profile": "a running Strata server"}], []]
        with mock.patch("time.sleep"):
            gate.release(lambda: answers.pop(0) if answers else [])
        self.assertEqual(len(gate.stops), 1)
        self.assertEqual(answers, [], "it looked until nothing was serving")

    def test_an_engine_that_never_lets_go_is_said_plainly_and_nothing_more_is_changed(self):
        gate, _ = self.gate(settle_s=0.3)
        calls = {"n": 0}

        def serving():
            calls["n"] += 1
            return [{"profile": "launch-x"}] if calls["n"] == 1 else [{"profile": "a running Strata server"}]
        with self.assertRaises(updater.UpdateError) as caught:
            gate.release(serving)
        self.assertEqual(caught.exception.exit_code, 7)
        self.assertIn("has not let go of the graphics card", str(caught.exception))

    def test_the_start_again_question_says_same_settings_when_part_of_the_update_was_made(self):
        for changed, text in ((True, "with the same settings"), (False, "The old engine is unchanged")):
            gate, model = self.gate()
            gate.stopped.append(model)
            gate.finish(False, changed)
            self.assertIn(text, gate.asked[0], changed)


class StartWait(unittest.TestCase):
    """Found by the real DeepSeek flow: a DeepSeek run config asks for a long engine wait; runtime.start takes 0..1800 only, so the model did not start again."""

    def test_the_wait_never_exceeds_what_the_runtime_accepts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "run.json"
            for ready, want in ((None, 960.0), (300, 360.0), (1800, 1800.0), (5000, 1800.0)):
                cfg.write_text(json.dumps({"engine_ready_s": ready} if ready else {}))
                model = engine_update.RunningModel("launch-x", "X")
                with mock.patch.object(profiles, "load", return_value=({"source_config": str(cfg)}, "r")):
                    self.assertEqual(engine_update._start_wait(model), want, ready)


class TheQuestion(unittest.TestCase):
    def test_a_person_at_a_keyboard_gets_the_question_and_default_on_enter(self):
        a = argparse.Namespace(yes=False)
        with mock.patch.object(sys.stdin, "isatty", return_value=True), mock.patch.object(sys.stdout, "isatty", return_value=True):
            ask = update_cmd._ask_factory(a, False)
            self.assertIsNotNone(ask)
            for typed, default, want in (("", False, False), ("", True, True), ("y", False, True), ("Yes", False, True), ("n", True, False), ("what", True, False)):
                with mock.patch("builtins.input", return_value=typed):
                    self.assertEqual(ask("q?", default), want, (typed, default))
            with mock.patch("builtins.input", side_effect=EOFError):
                self.assertFalse(ask("q?", False))

    def test_nobody_is_asked_for_a_program_a_pipe_or_yes(self):
        with mock.patch.object(sys.stdin, "isatty", return_value=True), mock.patch.object(sys.stdout, "isatty", return_value=True):
            self.assertIsNone(update_cmd._ask_factory(argparse.Namespace(yes=True), False))
            self.assertIsNone(update_cmd._ask_factory(argparse.Namespace(yes=False), True))
        with mock.patch.object(sys.stdin, "isatty", return_value=False):
            self.assertIsNone(update_cmd._ask_factory(argparse.Namespace(yes=False), False))


if __name__ == "__main__":
    unittest.main()
