"""Round 0213 TUICFG, step 1: profile identity and keys (#4395, #4396).

Real temporary profile files (isolated home); only the runtime status is a double. A damaged profile must neither throw nor lend its
neighbour's endpoint/config/key, in either order."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.terminal.launch_support import Home  # noqa: E402


def _cfg(home: Home, name: str, key: str) -> Path:
    path = home.base / f"{name}.run.json"
    path.write_text(json.dumps({"exe": str(home.base / "engine.exe"), "tokenizer": str(home.base / "tok.json"), "args": [],
                                "api_key": key, "model_name": name}), encoding="utf-8")
    return path


def _profile(home: Home, pid: str, cfg: Path | None, port: int | None = None, host: str | None = None) -> None:
    d = Path(home.env["STRATA_PROFILE_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    ov = {}
    if port is not None:
        ov["port"] = port
    if host is not None:
        ov["host"] = host
    (d / f"{pid}.json").write_text(json.dumps({"schema_version": 1, "id": pid, "source_config": str(cfg), "overrides": ov}), encoding="utf-8")


def _broken(home: Home, pid: str, how: str = "garbage") -> None:
    d = Path(home.env["STRATA_PROFILE_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{pid}.json").write_text("{not json" if how == "garbage" else json.dumps({"schema_version": 1, "id": pid}), encoding="utf-8")


class RunningLaunchesBrokenProfile(unittest.TestCase):
    """#4395: prof was read outside the try that binds it."""

    def _rows(self, first_broken: bool):
        from strata_cli.launch import cmd
        with Home() as home:
            good_cfg = _cfg(home, "good", "sk-strata-GOOD")
            if first_broken:
                _broken(home, "launch-a-broken")
                _profile(home, "launch-b-good", good_cfg, port=18123, host="0.0.0.0")
            else:
                _profile(home, "launch-a-good", good_cfg, port=18123, host="0.0.0.0")
                _broken(home, "launch-b-broken")
            with mock.patch("strata_control.runtime.status", return_value={"state": "ready"}):
                return cmd.running_launches()

    def test_broken_first_does_not_throw_and_good_row_is_intact(self):
        rows = self._rows(first_broken=True)
        self.assertEqual(len(rows), 2)
        bad = next(r for r in rows if r["name"].endswith("broken"))
        good = next(r for r in rows if r["name"].endswith("good"))
        self.assertIsNone(bad["config"])
        self.assertEqual(good["port"], 18123)
        self.assertTrue(good["config"].endswith("good.run.json"))

    def test_broken_after_valid_never_shows_the_previous_rows_endpoint_or_config(self):
        rows = self._rows(first_broken=False)
        self.assertEqual(len(rows), 2)
        bad = next(r for r in rows if r["name"].endswith("broken"))
        self.assertIsNone(bad["config"])
        self.assertNotEqual(bad["port"], 18123)
        self.assertEqual(bad["host"], "127.0.0.1")

    def test_connect_info_never_returns_another_profiles_key(self):
        from strata_cli.launch import cmd
        for first_broken in (True, False):
            with self.subTest(first_broken=first_broken), Home() as home:
                good_cfg = _cfg(home, "good", "sk-strata-GOOD")
                bad_id, good_id = ("launch-a-broken", "launch-b-good") if first_broken else ("launch-b-broken", "launch-a-good")
                _broken(home, bad_id)
                _profile(home, good_id, good_cfg, port=18123)
                with mock.patch("strata_control.runtime.status", return_value={"state": "ready"}):
                    bad = cmd.launch_info(bad_id[7:])
                    good = cmd.launch_info(good_id[7:])
                self.assertIsNone(bad["key"])
                self.assertNotEqual(bad["port"], 18123)
                self.assertEqual(good["key"], "sk-strata-GOOD")
                self.assertEqual(good["port"], 18123)


class KeptKeyBrokenProfile(unittest.TestCase):
    """#4396: one unreadable profile must not make kept_key() mint a new key."""

    def test_valid_key_survives_a_damaged_profile_in_either_order(self):
        from strata_cli.launch import access
        for order in ("broken-first", "broken-last"):
            for how in ("garbage", "missing-fields"):
                with self.subTest(order=order, how=how), Home() as home:
                    cfg = _cfg(home, "ok", "sk-strata-KEEP")
                    if order == "broken-first":
                        _broken(home, "launch-a-bad", how)
                        _profile(home, "launch-b-ok", cfg)
                    else:
                        _profile(home, "launch-a-ok", cfg)
                        _broken(home, "launch-b-bad", how)
                    self.assertEqual(access.kept_key(), "sk-strata-KEEP")

    def test_the_named_launch_with_a_damaged_profile_falls_back_to_the_other_valid_key(self):
        from strata_cli.launch import access
        with Home() as home:
            cfg = _cfg(home, "ok", "sk-strata-KEEP")
            _broken(home, "launch-mine")
            _profile(home, "launch-other", cfg)
            self.assertEqual(access.kept_key("mine"), "sk-strata-KEEP")

    def test_a_config_row_without_a_source_is_skipped_not_fatal(self):
        from strata_cli.launch import access
        with Home() as home:
            cfg = _cfg(home, "ok", "sk-strata-KEEP")
            _profile(home, "launch-ok", cfg)
            rows = [{"id": "launch-err", "error": "cannot read profile (PermissionError)"}, {"id": "launch-ok", "source_config": str(cfg)}]
            with mock.patch("strata_control.profiles.list_profiles", return_value=rows):
                self.assertEqual(access.kept_key(), "sk-strata-KEEP")

    def test_no_valid_profile_still_returns_none(self):
        from strata_cli.launch import access
        with Home() as home:
            _broken(home, "launch-only")
            self.assertIsNone(access.kept_key())


if __name__ == "__main__":
    unittest.main()
