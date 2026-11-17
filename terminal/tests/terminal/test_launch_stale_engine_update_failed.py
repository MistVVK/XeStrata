"""#5038: `strata launch --model-path` with an installed engine older than the model wants, and an engine update whose test start
fails. The reporter's 0.2.6 counted the old engine as absent and gave up the whole install. Now the old engine is "stale", not
absent: the launch says so, says the update did not finish, keeps going with the engine that is there (never refuses), and the
final message tells the person to run `strata update`. Without --yes nothing is updated and the launch goes on the same way."""
from __future__ import annotations

import unittest
from unittest import mock

from strata_cli.launch import settings as S
from strata_cli.launch.runner import Runner
from tests.terminal.launch_support import HW_12, Home, catalog

OLD = {"usable": True, "engine": {"version": "0.1.45", "backend": "cuda"}}


def runner(yes: bool) -> tuple[Runner, list]:
    r = Runner(S.LaunchSettings(family="qwen", model="IQ3_XXS", context=32768, vision="off"), catalog=catalog(HW_12))
    r.assume_yes = yes                                       # the reporter ran with --yes: the update is tried
    events: list = []
    r.emit = lambda stage, text, pct=None, level="info", **kw: events.append((level, text))
    return r, events


class StaleEngineUpdateFailed(unittest.TestCase):
    def run_engine_step(self, after: dict, *, update_lines=(), code=1):
        r, events = runner(True)
        proc = mock.Mock()
        proc.stdout = iter(update_lines)
        proc.wait.return_value = code
        from strata_control import engine_component as ec, engine_update
        with mock.patch.object(ec, "status", side_effect=[OLD, after]), \
                mock.patch.object(engine_update, "running_models", return_value=[]), \
                mock.patch.object(Runner, "_spawn", return_value=proc), \
                mock.patch.object(Runner, "no_engine") as no_engine:
            r.ensure_engine()                                   # must not raise: the install goes on
        return r, events, no_engine

    def test_a_failed_update_keeps_the_old_engine_and_says_why(self):
        lines = ['{"event": "phase", "phase": "warn", "message": "The new engine 0.1.55 did not start correctly on this computer."}\n']
        with Home():
            r, events, no_engine = self.run_engine_step(OLD, update_lines=lines, code=1)
        texts = [t for _level, t in events]
        self.assertNotIn("No engine is installed yet. Getting the one for your graphics card…", texts)
        self.assertTrue(any("older than this model needs" in t and "Looking for a newer one" in t for t in texts), texts)
        self.assertTrue(any("did not start correctly" in t for t in texts), "the update's own reason reaches the person")
        self.assertTrue(any(level == "warn" and "The update did not finish" in t for level, t in events), events)
        self.assertTrue(any("strata update" in n for n in r.notices), r.notices)
        no_engine.assert_not_called()

    def test_an_update_that_brings_the_new_engine_says_so(self):
        new = {"usable": True, "engine": {"version": "9.9.9", "backend": "cuda"}}
        with Home():
            r, events, _ = self.run_engine_step(new, code=0)
        self.assertIn(("info", "Engine 9.9.9 is installed."), events)
        self.assertFalse(r.notices)

    def test_without_yes_nothing_is_updated_and_the_launch_goes_on(self):
        from strata_control import engine_component as ec, engine_update
        with Home():
            r, events = runner(False)
            with mock.patch.object(ec, "status", return_value=OLD), \
                    mock.patch.object(engine_update, "running_models", return_value=[]), \
                    mock.patch.object(Runner, "_spawn") as spawn:
                r.ensure_engine()
        spawn.assert_not_called()
        self.assertTrue(any(level == "warn" and "did not update it" in t for level, t in events), events)


if __name__ == "__main__":
    unittest.main()
