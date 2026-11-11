"""The gate of CLEAN-1A: in every machine class x model set, the app's plain Load and the app's Models-screen Load start the engine as `strata launch` does,
apart from the allow-list (tests/snapshot/allowlist.py).  Reads the golden files, so it needs no Node and no machine; test_config_snapshot.py keeps the golden files true."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.snapshot import allowlist  # noqa: E402
from tests.snapshot.generate import GOLDEN  # noqa: E402


def flat(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            out.update(flat(v, f"{prefix}/{k}"))
        return out
    return {prefix: value}


def comparable(rec: dict, allowed_cfg: set[str]) -> dict:
    cfg = {k: v for k, v in rec["config_file"].items() if k not in allowed_cfg}
    env = {k: v for k, v in rec["engine"]["env_set"].items() if k not in allowed_cfg and k not in allowlist.ENV_BY_DESIGN}
    return {"config": flat(cfg), "argv": rec["engine"]["argv"], "env": env, "fetches": rec.get("fetches")}


class EntryPointsAgree(unittest.TestCase):
    maxDiff = None

    def cells(self):
        for path in sorted(GOLDEN.glob("*/*.json")):
            yield path.parent.name + "/" + path.stem, json.loads(path.read_text(encoding="utf-8"))["entries"]

    def check(self, other: str, allowed: set[str]):
        seen = 0
        for cell, entries in self.cells():
            with self.subTest(cell=cell, entry=other):
                want, got = comparable(entries["launch"], allowed), comparable(entries[other], allowed)
                self.assertEqual(want["argv"], got["argv"], f"{cell}: engine arguments")
                self.assertEqual(want["env"], got["env"], f"{cell}: engine environment")
                self.assertEqual(want["config"], got["config"], f"{cell}: run config")
                self.assertEqual(want["fetches"], got["fetches"], f"{cell}: the drafter downloads asked for")
            seen += 1
        self.assertGreaterEqual(seen, 24)

    def test_the_plain_load_starts_the_engine_as_launch_does(self):
        self.check("app_plain", set(allowlist.BY_DESIGN))

    def test_the_models_screen_load_differs_only_by_the_choice_it_saves(self):
        self.check("app_models", set(allowlist.BY_DESIGN) | set(allowlist.MODELS_SCREEN))

    def test_no_app_entry_writes_request_slots(self):
        for cell, entries in self.cells():
            for entry in ("app_plain", "app_models"):
                self.assertNotIn("parallel", entries[entry]["config_file"], f"{cell}/{entry}")
                self.assertNotIn("--batch-mtp", entries[entry]["engine"]["argv"], f"{cell}/{entry}")


if __name__ == "__main__":
    unittest.main()
