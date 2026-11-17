"""QAGUI-STRIX 11-09 (APP-WIN item 3): on Windows Strix Halo the app's Models-screen Load rewrote the reserve setup wrote (700) to 1024, 2048 with pictures,
while `strata launch` started the same model with 700.  The app's Load ("Leave room for my work" and "Full performance", what the Models screen passes)
now starts the engine as `strata launch` does, from the run config setup writes; the only difference allowed is the work profile's --pool-workers (the
profile's meaning) and the variables it sets (allowlist.ENV_BY_DESIGN, STRATA_RAM_KEEP_GIB).

  python -m pytest tests/snapshot/test_strix_app_reserve.py -q        (needs Node, as the app entries of the snapshot do)
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.snapshot.driver import drive, drive_app, drive_launch  # noqa: E402
from tests.snapshot.matrix import machine, model_set  # noqa: E402
from tests.snapshot.world import World  # noqa: E402

NODE = os.environ.get("STRATA_SNAPSHOT_NODE") or shutil.which("node")


def flag(argv: list[str], name: str):
    return argv[argv.index(name) + 1] if name in argv else None


def without(argv: list[str], name: str) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == name:
            skip = True
            continue
        out.append(a)
    return out


def setup_reserve(model: str) -> str:
    """The reserve setup itself writes on this class (strata_control/setup_config.py, driven by the snapshot's setup entry)."""
    args = drive("strix_halo_win", model, "setup")["config_file"]["args"]
    return flag(args, "--vram-reserve-mib")


def world_with_setup_reserve(model: str, reserve: str) -> World:
    world = World(machine("strix_halo_win"), model_set(model)).build()
    cfg = json.loads(world.config_path.read_text(encoding="utf-8"))
    if "--vram-reserve-mib" not in cfg["args"]:
        cfg["args"] += ["--vram-reserve-mib", reserve]
    world.config_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return world


@unittest.skipIf(not NODE, "no Node: the app's Load runs in Node")
class StrixAppReserve(unittest.TestCase):
    maxDiff = None

    def engine(self, model: str, reserve: str, entry: str, app_args: dict | None = None) -> dict:
        world = world_with_setup_reserve(model, reserve)
        try:
            if entry == "launch":
                with world.patched():
                    rec = drive_launch(world)
            else:
                with mock.patch.dict(os.environ, {"STRATA_SNAPSHOT_APP_ARGS": json.dumps(app_args or {})}):
                    rec = drive_app(world, "models")
            self.assertNotIn("failed", rec, json.dumps(rec.get("failed")))
            # each entry runs in its own scratch world: its folder is written as <ROOT> so two engine lines can be compared
            root = str(world.root)
            return {**rec["engine"], "argv": [a.replace(root, "<ROOT>") for a in rec["engine"]["argv"]]}
        finally:
            world.close()

    def test_the_models_screen_load_keeps_setups_reserve_and_starts_as_launch_does(self):
        for model in ("ud_q4_k_xl", "iq3_s"):
            reserve = setup_reserve(model)
            self.assertEqual(reserve, "700", "setup's own reserve on Windows Strix Halo")
            launch = self.engine(model, reserve, "launch")
            self.assertEqual(flag(launch["argv"], "--vram-reserve-mib"), "700")
            for profile in ("work", "performance"):
                with self.subTest(model=model, profile=profile):
                    app = self.engine(model, reserve, "app", {"resourceProfile": profile, "vision": "no", "values": {"drafting": "auto"}})
                    self.assertEqual(flag(app["argv"], "--vram-reserve-mib"), "700", "the app keeps what setup wrote (it wrote 1024)")
                    self.assertEqual(without(app["argv"], "--pool-workers"), without(launch["argv"], "--pool-workers"), "the same engine line")
                    if profile == "performance":
                        self.assertIsNone(flag(app["argv"], "--pool-workers"))


if __name__ == "__main__":
    unittest.main()
