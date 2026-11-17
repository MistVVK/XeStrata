"""Round 0213 QATUI: a profile or run config of the launch itself that cannot be read stops the launch BEFORE anything is written.

Found on l620 (job qatui-s03c / s03d): with `chmod 000` on the launch's own profile, `strata launch --access lan` printed "A new API key was made", rewrote the
launch's run config with that new key, and only then failed with `profile 'launch-x' already exists` (exit 2): the key every app was set up with was gone.
With `chmod 000` on the run config the launch went on and replaced it silently, again with a new key. Real files, an isolated home; only the Runner's
step is called (the rest of a launch needs an engine).

    python -m pytest tests/terminal/test_launch_unreadable_state_0213_qatui.py
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.terminal.launch_support import Home  # noqa: E402

POSIX_NOT_ROOT = os.name != "nt" and hasattr(os, "geteuid") and os.geteuid() != 0


def _runner():
    from strata_cli.launch import settings as S
    from strata_cli.launch.runner import Runner
    r = Runner(S.LaunchSettings(model="iq3_s", context=32768))
    assert r.name == "iq3_s-32k" and r.profile == "launch-iq3_s-32k"
    return r


def _write_state(name: str = "iq3_s-32k", key: str = "sk-strata-OLD"):
    from strata_cli.launch.runner import launch_dir
    cfg = launch_dir() / f"{name}.json"
    cfg.write_text(json.dumps({"exe": "x", "args": [], "api_key": key}), encoding="utf-8")
    pdir = Path(os.environ["STRATA_PROFILE_DIR"])
    pdir.mkdir(parents=True, exist_ok=True)
    prof = pdir / f"launch-{name}.json"
    prof.write_text(json.dumps({"schema_version": 1, "id": f"launch-{name}", "source_config": str(cfg), "overrides": {}}), encoding="utf-8")
    return cfg, prof


class CheckSavedState(unittest.TestCase):
    def test_nothing_saved_yet_is_fine(self):
        with Home():
            _runner().check_saved_state_readable()

    def test_readable_state_is_fine(self):
        with Home():
            _write_state()
            _runner().check_saved_state_readable()

    def test_a_damaged_profile_of_this_launch_stops_it_before_any_write(self):
        from strata_control import profiles
        with Home():
            cfg, prof = _write_state()
            prof.write_text("{not json", encoding="utf-8")
            before = cfg.read_bytes()
            with self.assertRaises(profiles.ProfileError) as ctx:
                _runner().check_saved_state_readable()
            text = str(ctx.exception)
            self.assertTrue(text.startswith(f"The profile of this launch ({prof}) cannot be read ("), text)
            self.assertTrue(text.endswith("). Fix its permissions or move it away, then run the launch again."), text)
            self.assertEqual(ctx.exception.exit_code, 2)
            self.assertEqual(cfg.read_bytes(), before)

    @unittest.skipUnless(POSIX_NOT_ROOT, "needs permissions that deny the owner")
    def test_an_unreadable_profile_stops_the_launch_and_the_key_stays(self):
        from strata_control import profiles
        with Home():
            cfg, prof = _write_state()
            os.chmod(prof, 0)
            try:
                with self.assertRaises(profiles.ProfileError) as ctx:
                    _runner().check_saved_state_readable()
            finally:
                os.chmod(prof, 0o600)
            self.assertIn("(Permission denied)", str(ctx.exception))
            self.assertIn("sk-strata-OLD", cfg.read_text(encoding="utf-8"))

    @unittest.skipUnless(POSIX_NOT_ROOT, "needs permissions that deny the owner")
    def test_an_unreadable_run_config_stops_the_launch_instead_of_minting_a_key(self):
        from strata_cli.launch.runner import LaunchError
        with Home():
            cfg, prof = _write_state()
            os.chmod(cfg, 0)
            try:
                with self.assertRaises(LaunchError) as ctx:
                    _runner().check_saved_state_readable()
            finally:
                os.chmod(cfg, 0o600)
            self.assertEqual(str(ctx.exception), f"The saved settings of this launch ({cfg}) cannot be read (Permission denied). Fix its permissions or move it away, "
                                                 "then run the launch again: writing a new one would drop its API key and your changes.")
            self.assertEqual(ctx.exception.code, 6)
            self.assertIn("sk-strata-OLD", cfg.read_text(encoding="utf-8"))

    def test_an_unreadable_run_config_is_refused_on_every_platform(self):
        """The same, with the read failing the way a locked or permission-denied file fails (also testable on Windows)."""
        from strata_cli.launch.runner import LaunchError
        with Home():
            cfg, prof = _write_state()
            real = Path.read_bytes

            def denied(self_):
                if self_ == cfg:
                    raise PermissionError(13, "Permission denied")
                return real(self_)
            with mock.patch.object(Path, "read_bytes", denied):
                with self.assertRaises(LaunchError) as ctx:
                    _runner().check_saved_state_readable()
            self.assertIn("cannot be read (Permission denied)", str(ctx.exception))


NEW_KEY = "A new API key was made, because the server is shared beyond this computer. It is saved with the preset; show it with: strata launch --show-key"


class NewKeySentenceIsTrue(unittest.TestCase):
    """The sentence "A new API key was made" is said only after the run config that holds the key has been written: not in the plan, not before a refusal."""

    def _run(self, *, refuse=None, fail_after_write=None):
        from strata_cli.launch import settings as S
        from strata_cli.launch.runner import Runner
        events, order = [], []
        r = Runner(S.LaunchSettings(model="iq3_s", context=32768), emit=lambda e: events.append(e))
        r.key_written_notice = NEW_KEY
        base = Path(os.environ["STRATA_INSTALL_DIR"]) / "base.json"

        def build(self_, base_path):
            order.append("written")
            return base
        patches = [mock.patch.object(Runner, "check_own_files"), mock.patch.object(Runner, "check_serving", return_value=None),
                   mock.patch.object(Runner, "ensure_engine"), mock.patch.object(Runner, "ensure_model", return_value=base),
                   mock.patch.object(Runner, "ensure_drafter"), mock.patch.object(Runner, "build_config", build),
                   mock.patch.object(Runner, "ensure_profile", side_effect=fail_after_write or (lambda *a, **k: None)),
                   mock.patch.object(Runner, "tune", side_effect=RuntimeError("stop here")),
                   mock.patch.object(Runner, "check_saved_state_readable", side_effect=refuse or (lambda *a, **k: None))]
        for p in patches:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patches])
        try:
            r.run()
        except Exception as exc:  # noqa: BLE001
            return r, events, order, exc
        return r, events, order, None

    def test_a_refusal_before_the_write_says_nothing_about_a_new_key(self):
        from strata_cli.launch.runner import LaunchError
        with Home():
            r, events, order, exc = self._run(refuse=LaunchError("cannot be read", "fix it"))
        self.assertIsInstance(exc, LaunchError)
        self.assertEqual(order, [])                                    # nothing was written
        self.assertFalse(r.key_written)
        self.assertNotIn(NEW_KEY, [e.text for e in events])

    def test_the_sentence_comes_once_right_after_the_write(self):
        with Home():
            r, events, order, exc = self._run()
        self.assertEqual(order, ["written"])
        self.assertTrue(r.key_written)
        said = [e for e in events if e.text == NEW_KEY]
        self.assertEqual(len(said), 1)
        self.assertEqual(said[0].level, "warn")
        texts = [e.text for e in events]
        self.assertLess(texts.index(NEW_KEY), texts.index("Settings are saved.") if "Settings are saved." in texts else len(texts))

    def test_it_stays_true_when_something_fails_after_the_write(self):
        from strata_control import profiles
        with Home():
            r, events, order, exc = self._run(fail_after_write=profiles.ProfileError("later problem"))
        self.assertIsInstance(exc, profiles.ProfileError)
        self.assertTrue(r.key_written)                                 # the key IS on disk then
        self.assertIn(NEW_KEY, [e.text for e in events])

    def test_the_plan_never_says_it(self):
        from tests.terminal.launch_support import run_cli
        with Home():
            code, out, err = run_cli(["launch", "--plan", "--family", "qwen", "--model", "q2_0", "--access", "lan", "--yes"])
        self.assertIn("Plan for", out, out + err)                       # the plan really ran
        self.assertIn("API key on", out)                                 # and it knows a key is wanted
        self.assertNotIn("A new API key was made", out + err)


if __name__ == "__main__":
    unittest.main()
