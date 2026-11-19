"""#6278 (and #5940, #5819): `strata launch` took seconds per key on some PCs. Every screen asks which engine is installed on every paint
(the step list asks whether the Drafting step exists: Catalog.installed_backend -> engine_component.status), and status() read the engine's whole
entrypoint (171 MB on Windows CUDA) to check its SHA-256 each time: 120 ms per call here with the file in the cache, seconds on a slow disk or with a
virus checker, and several calls per key. The check is made once per file as it is on disk."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from strata_control import engine_component as ec
from tests.terminal.test_engine_component import make_package, speak_about_pc
from tests.terminal.test_launch_tui import Base, make_ctx, press


class QuickCheckReadsTheEntrypointOnce(unittest.TestCase):
    def setUp(self):
        speak_about_pc(self)
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        self.folder = make_package(self.root, backend="cuda")
        p = mock.patch.dict(os.environ, {ec.ENV_BACKEND: "cuda"})
        p.start()
        self.addCleanup(p.stop)
        ec._HASHED_OK.clear() if hasattr(ec, "_HASHED_OK") else None

    def test_twenty_listings_read_the_entrypoint_once(self):
        real = ec._sha256
        with mock.patch.object(ec, "_sha256", side_effect=real) as sha:
            for _ in range(20):
                self.assertTrue(ec.status(self.root)["usable"])
        self.assertEqual(sha.call_count, 1)

    def test_a_changed_entrypoint_is_read_again_and_refused(self):
        exe = self.folder / ec.read_manifest(self.folder).entrypoint
        self.assertTrue(ec.status(self.root)["usable"])
        data = exe.read_bytes()
        stat = exe.stat()
        exe.write_bytes(bytes([data[0] ^ 1]) + data[1:])             # the same size, other content
        os.utime(exe, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
        self.assertFalse(ec.status(self.root)["usable"])

    def test_a_start_still_reads_every_file(self):
        real = ec._sha256
        with mock.patch.object(ec, "_sha256", side_effect=real) as sha:
            for _ in range(3):
                ec.resolve(self.root, full=True)
        self.assertEqual(sha.call_count, 3 * 3)                      # entrypoint, runtime library and BUILD.json, each time


class WizardKeysDoNotReadTheEngine(Base):
    def test_twenty_down_keys_on_the_screens_read_the_entrypoint_at_most_once(self):
        root = Path(os.environ["STRATA_COMPONENTS_DIR"])
        make_package(root, backend="cuda")
        p = mock.patch.dict(os.environ, {ec.ENV_BACKEND: "cuda"})
        p.start()
        self.addCleanup(p.stop)
        speak_about_pc(self)
        real = ec._sha256
        ctx = make_ctx()
        ctx.settings.family, ctx.settings.model = "qwen", "IQ3_XXS"

        async def case(app, pilot, ctx):
            with mock.patch.object(ec, "_sha256", side_effect=real) as sha:
                for step in ("model", "context", "pictures", "parallel", "tuning", "advanced", "access"):
                    app.goto(step)
                    await pilot.pause(0.1)
                    await press(pilot, *(["down"] * 20), pause=0.0)
                self.assertLessEqual(sha.call_count, 1)
        self.run_app(case, ctx=ctx)


class PresetListIsReadOncePerChange(Base):
    """The start screen and the Advanced step's default name list every saved preset on every key; 300 presets cost 60 ms per key."""

    def test_twenty_lists_parse_each_file_once_and_a_change_is_seen(self):
        from strata_cli.launch import settings as S, store
        for i in range(5):
            store.save(S.LaunchSettings(name=f"P{i}", family="qwen", model="IQ3_XXS", context=65536))
        real = S.LaunchSettings.from_dict
        with mock.patch.object(S.LaunchSettings, "from_dict", side_effect=real) as parse:
            for _ in range(20):
                names = sorted(p.name for p in store.list_presets())
            self.assertEqual(names, [f"P{i}" for i in range(5)])
            self.assertEqual(parse.call_count, 5)
            first = store.list_presets()
            first[0].name = "changed by a caller"                   # a caller's edit never reaches the next list
            self.assertNotIn("changed by a caller", [p.name for p in store.list_presets()])
            store.save(S.LaunchSettings(name="P9", family="qwen", model="IQ3_XXS", context=131072))
            self.assertIn("P9", [p.name for p in store.list_presets()])
            store.delete("P0")
            self.assertNotIn("P0", [p.name for p in store.list_presets()])


if __name__ == "__main__":
    unittest.main()
