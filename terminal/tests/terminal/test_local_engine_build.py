"""#5114: `strata components add-build FOLDER` turns an engine built on this computer into a local, unsigned engine package. It is labelled as
such (the engine list, doctor), it is used only because the person chose it (never picked by itself over a released engine), it can be removed
again, and a build that does not run here (an illegal instruction on a CPU without AVX), is for another kind of card, or names no card family is
refused with nothing written. Stub engines are shell scripts: POSIX only."""

from __future__ import annotations

import json
import os
import stat
import sys
import unittest
from pathlib import Path
from unittest import mock

from strata_control import engine_component as ec
from strata_control import engine_library as lib
from strata_control import local_engine, updater
from tests.terminal.test_updater import UpdaterCase

POSIX = os.name != "nt"


def make_build(folder: Path, *, version="0.1.61", backend="cuda", archs=("86",), script=None, record=True) -> Path:
    folder.mkdir(parents=True)
    exe = folder / "strata"
    exe.write_text(script or f"#!/bin/sh\necho 'strata {version} (built from source)'\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (folder / "libstrata_kq.so").write_bytes(b"\x7fELF" + os.urandom(64))
    (folder / "CMakeCache.txt").write_text("not part of the engine")
    if record:
        (folder / "BUILD.json").write_text(json.dumps({"version": version, "source": "abc1234", "backend": backend, "archs": list(archs)}))
    return folder


@unittest.skipUnless(POSIX, "the stub engines are shell scripts")
class AddBuild(UpdaterCase):
    def build(self, **kw):
        kw.setdefault("backend", self.family["backend"])
        kw.setdefault("archs", [a.removeprefix("sm_") for a in self.family["archs"] if not a.startswith("compute_")] or ["x"])
        return make_build(self.root / "build", **kw)

    def add(self, folder, **kw):
        return local_engine.add_build(self.components, folder, detected=self.family["backend"], **kw)

    def test_the_build_becomes_a_local_unsigned_package_with_its_own_files(self):
        self.install_current()
        out = self.add(self.build())
        self.assertEqual(out["folder"], f"0.1.61-{self.platform}-{self.family['backend']}-local")
        pkg = ec.read_manifest(self.components / "engine" / out["folder"])
        self.assertTrue(pkg.local)
        self.assertIsNone(pkg.signature)
        self.assertEqual(sorted(f[0] for f in pkg.files), ["BUILD.json", "libstrata_kq.so", "strata"], "the engine's files, not the build folder's")
        ec.verify(pkg, full=True)
        self.assertEqual(ec.verify_signature(pkg), "unsigned")
        with mock.patch.dict(os.environ, {ec.ENV_REQUIRE_SIGNED: "1"}):
            with self.assertRaises(ec.IntegrityProblem):
                ec.verify_signature(pkg)

    def test_never_picked_by_itself_and_used_when_chosen(self):
        self.install_current()                                           # a released engine, older than the build
        out = self.add(self.build())
        self.assertEqual(self.resolved(), "0.1.45", "a newer local build is not picked over the released engine")
        lib.use_engine(self.ctx(), lib.QWEN, out["folder"])
        self.assertEqual(ec.resolve(self.components, backend=self.family["backend"]).root.name, out["folder"])
        lib.use_engine(self.ctx(), lib.QWEN, "0.1.45-" + self.platform + "-" + self.tag)   # `components use newest`: the released one
        self.assertEqual(self.resolved(), "0.1.45")

    def test_a_local_build_alone_is_not_started_until_chosen(self):
        out = self.add(self.build())
        with self.assertRaises(ec.EngineMissing) as cm:
            ec.resolve(self.components, backend=self.family["backend"])
        self.assertIn(f"`strata components use {out['folder']}`", str(cm.exception))

    def test_it_is_labelled_and_can_be_removed(self):
        self.install_current()
        out = self.add(self.build())
        rows = {r["folder"]: r for r in lib.list_engines(self.ctx())["families"][0]["engines"]}
        self.assertTrue(rows[out["folder"]]["local"])
        self.assertFalse(rows[out["folder"]]["signed"])
        self.assertFalse(rows[f"0.1.45-{self.platform}-{self.tag}"]["local"])
        lib.delete_engine(self.ctx(), lib.QWEN, out["folder"])
        self.assertFalse((self.components / "engine" / out["folder"]).exists())

    def test_refusals_write_nothing(self):
        ill = self.build(script="#!/bin/sh\nkill -ILL $$\n")
        with self.assertRaises(local_engine.AddBuildError) as cm:
            self.add(ill)
        self.assertIn("an instruction this processor does not have", str(cm.exception))
        self.assertEqual(cm.exception.exit_code, 5)
        self.assertFalse((self.components / "engine").exists() and any((self.components / "engine").glob("*-local")))
        other = "hip" if self.family["backend"] != "hip" else "cuda"
        with self.assertRaises(local_engine.AddBuildError) as cm:
            local_engine.add_build(self.components, make_build(self.root / "b2", backend=other), detected=self.family["backend"])
        self.assertIn("engine", str(cm.exception))
        with self.assertRaises(local_engine.AddBuildError) as cm:
            self.add(make_build(self.root / "b3", record=False))
        self.assertIn("add --backend", str(cm.exception))
        empty = self.root / "b4"
        empty.mkdir()
        with self.assertRaises(local_engine.AddBuildError):
            self.add(empty)

    def test_the_command_and_doctor_say_it(self):
        from tests.terminal.launch_support import run_cli
        self.install_current()
        build = self.build()
        code, out, err = run_cli(["components", "add-build", str(build), "--components-dir", str(self.components), "--output", "json"])
        self.assertEqual(code, 0, err + out)
        doc = json.loads(out)
        folder = doc["added"]["folder"]
        self.assertTrue(folder.endswith("-local"))
        code, out, err = run_cli(["components", "engines", "--components-dir", str(self.components)])
        self.assertIn("local build, not signed", out)
        from strata_control import doctor
        cfg = self.root / "local-run.json"
        cfg.write_text(json.dumps({**json.loads(self.config.read_text()), "exe": str(self.components / "engine" / folder / "strata")}))
        report = doctor.inspect(cfg)
        self.assertIn("LOCAL_ENGINE_BUILD", [i["code"] for i in report["issues"]])
        self.assertTrue(any(c.get("local_build") for c in report["components"]))


class BuildRecord(unittest.TestCase):
    def test_the_family_and_cards_from_build_json(self):
        self.assertEqual(local_engine.backend_of({"cuda": "13.0", "archs": [75, 86]}), "cuda")
        self.assertEqual(local_engine.backend_of({"backend": "rocm"}), "hip")
        self.assertIsNone(local_engine.backend_of({"version": "0.1.60"}))
        self.assertEqual(local_engine.archs_of({"archs": [75, 86]}, "cuda"), ["sm_75", "sm_86"])
        self.assertEqual(local_engine.archs_of({"archs": ["gfx1100"]}, "hip"), ["gfx1100"])
        # the Linux CUDA packages write one string (0.1.62 cuda12 / cuda13 BUILD.json)
        self.assertEqual(local_engine.archs_of({"archs": "75;86;89;120"}, "cuda"), ["sm_75", "sm_86", "sm_89", "sm_120"])
        self.assertEqual(local_engine.archs_of({"archs": "60;61;70;75;80;86;120;89"}, "cuda")[:3], ["sm_60", "sm_61", "sm_70"])


if __name__ == "__main__":
    unittest.main()
