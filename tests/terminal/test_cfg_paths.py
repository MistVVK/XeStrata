"""#4891 ("Qwen breaks each time Strata is updated"): a saved run config names folders of one version of Strata's own components (the Control bundle that
ran setup: cwd, --expert-profile; an engine package: the picture reader, library folders; the DeepSeek package).  The update keeps the new Control and
the one before it, so the SECOND update after setup removes the folder the config names; the model then showed as "not installed" and a start failed.

The updates here are real (`updater.run_update(control=True)` unpacks and prunes Control folders; `prune_engines` / `dsv41_paths.prune` remove the
older engine packages); the starts are real `runtime.start` calls with a fake engine, the way the app (with --die-with-pid) and the Terminal start.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strata_control import cfg_paths, doctor, dsv41_paths, engine_component as ec, profiles, runtime, updater  # noqa: E402
from tests.terminal.test_updater import PLATFORM, UpdaterCase  # noqa: E402

WINDOWS = os.name == "nt"
FAKE_ENGINE = ("#!" + sys.executable + "\nimport sys\n"
               "print('READY 4096 stop', flush=True)\n"
               "for line in sys.stdin:\n"
               "    if line.strip() == 'QUIT': break\n"
               "    if line.startswith(('GEN ', 'BGEN ')):\n"
               "        print('T 97', flush=True)\n"
               "        print('DONE 1 1 1.0 1.0 length 0 0 0', flush=True)\n")


class StaleConfigPaths(UpdaterCase):
    # (UpdaterCase's own tests are not repeated)
    def setUp(self):
        super().setUp()
        self.control = self.components / "control"
        self.engine_root = self.components / "engine"
        self.ds_root = self.root / "data" / "engine-dsv41"
        self.sequence = 0
        # Control 0.1.44 is what ran setup: its data/ holds the shipped expert profile
        self.install_control("0.1.44")
        (self.control / "active.json").write_text(json.dumps({"version": "0.1.44", "folder": "0.1.44", "sha256": "0" * 64}))
        # the engine package that setup (or an earlier update) wrote into the config, and the DeepSeek package
        self.engine_kind = f"{PLATFORM}-cuda-cuda13"
        (self.root / "packs" / "iq3_s").mkdir(parents=True)
        self.old_engine = self.make_engine_folder(self.engine_root, "0.1.44")
        self.old_ds = self.make_ds_folder("0.1.49")
        self.tokenizer = str(self.root / "tokenizer")
        self.mine = self.root / "mine" / "my-profile.bin"   # a path the person typed themselves
        self.configs = {}

    # ------------------------------------------------------------------ the world

    def install_control(self, version):
        folder = self.control / version
        (folder / "data").mkdir(parents=True)
        (folder / "strata.py").write_text(f"print('strata {version} (python stub)')\n")
        (folder / "data" / "expert-profile.bin").write_bytes(b"profile-" + version.encode())
        return folder

    def make_engine_folder(self, root, version, script_name=None):
        folder = root / f"{version}-{self.engine_kind}"
        folder.mkdir(parents=True)
        name = script_name or ("engine.cmd" if WINDOWS else "engine.py")
        script = folder / "engine.py"
        script.write_text(FAKE_ENGINE, encoding="utf-8")
        script.chmod(0o700)
        if WINDOWS:
            (folder / "engine.cmd").write_bytes(f'@echo off\r\n"{sys.executable}" "%~dp0engine.py" %*\r\n'.encode())
        (folder / "strata-vision").write_bytes(b"reader " + version.encode())
        return folder

    def make_ds_folder(self, version):
        folder = self.ds_root / f"{version}-{self.engine_kind}"
        folder.mkdir(parents=True)
        (folder / dsv41_paths.exe_name()).write_bytes(b"deepseek " + version.encode())
        (folder / dsv41_paths.exe_name()).chmod(0o755)
        return folder

    def engine_exe(self, folder):
        return folder / ("engine.cmd" if WINDOWS else "engine.py")

    def publish_control(self, version):
        bundle = self.root / f"bundle-{version}"
        (bundle / "strata_control").mkdir(parents=True)
        (bundle / "data").mkdir()
        (bundle / "data" / "expert-profile.bin").write_bytes(b"profile-" + version.encode())
        (bundle / "strata.py").write_text(f"print('strata {version} (python stub)')\n")
        (bundle / "strata_control" / "__init__.py").write_text("")
        from feed_tool import build_feed, pack_control
        shutil.rmtree(self.release, ignore_errors=True)
        (self.release / "control").mkdir(parents=True)
        pack_control(bundle, self.release / "control", version=version, platform=PLATFORM)
        self.sequence += 1
        build_feed(self.release, key=self.key, sequence=self.sequence, control_dir=self.release / "control")

    def update_to(self, version):
        self.publish_control(version)
        out = updater.run_update(self.ctx(), engine=False, control=True)
        self.assertEqual(out["results"][0]["status"], "updated", out)

    def two_updates(self):
        """Setup ran on Control 0.1.44.  Two updates later (0.1.45, then 0.1.46) Control keeps 0.1.46 and 0.1.45, and the engine and DeepSeek packages
        are replaced the way their updates do it."""
        self.update_to("0.1.45")
        self.assertTrue((self.control / "0.1.44").is_dir(), "one update keeps the folder before it")
        self.update_to("0.1.46")
        self.assertFalse((self.control / "0.1.44").exists(), "the second update removes the folder setup wrote into the config")
        new_engine = self.make_engine_folder(self.engine_root, "0.1.46")
        updater.prune_engines(self.components, {new_engine.name})
        self.assertFalse(self.old_engine.exists())
        new_ds = self.make_ds_folder("0.1.50")
        with mock.patch.object(dsv41_paths, "in_use", return_value=False):
            dsv41_paths.prune(self.ds_root, new_ds / dsv41_paths.exe_name())
        self.assertFalse(self.old_ds.exists())
        return new_engine, new_ds

    def qwen_config(self, name, vision=False, **extra):
        cfg = {"exe": str(self.engine_exe(self.old_engine)), "backend": "cuda", "tokenizer": self.tokenizer,
               "cwd": str(self.control / "0.1.44"), "model_name": "fake", "host": "127.0.0.1", "port": 0,
               "lib_dirs": [str(self.old_engine)],
               "args": ["--pack", str(self.root / "packs" / "iq3_s"),
                        "--expert-profile", str(self.control / "0.1.44" / "data" / "expert-profile.bin"),
                        "--expert-profile-save", str(self.control / "0.1.44" / "data" / "learned.bin"),
                        "--mtp", "2", "--lookup-file", str(self.mine)]}
        if vision:                                            # (a real start would run the picture reader: only the read-side tests ask for it)
            cfg["vision"] = {"exe": str(self.old_engine / "strata-vision")}
        cfg.update(extra)
        path = self.root / "run" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        profiles.create(name, str(path))
        self.configs[name] = path
        return path

    def deepseek_config(self, name):
        exe = self.old_ds / dsv41_paths.exe_name()
        cfg = {"exe": str(exe), "arch": "deepseek41", "backend": "cuda", "tokenizer": self.tokenizer, "cwd": str(self.control / "0.1.44"),
               "model_name": "deepseek", "host": "127.0.0.1", "port": 0, "lib_dirs": [str(self.old_ds)],
               "args": ["--pack", str(self.root / "packs" / "ds"), "--max-context", "8192"]}
        path = self.root / "run" / f"{name}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        profiles.create(name, str(path))
        self.configs[name] = path
        return path

    def cfg_of(self, path):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    def without_output_path(self, path):
        """The config's text without the one path the engine WRITES (--expert-profile-save), which is deliberately not looked for."""
        return Path(path).read_text(encoding="utf-8").replace(json.dumps(str(self.control / "0.1.44" / "data" / "learned.bin"))[1:-1], "")

    def host_log_lines(self, name, word="config:"):
        _, _, log_path = runtime._paths(name)
        return [l for l in log_path.read_text(encoding="utf-8").splitlines() if word in l] if log_path.exists() else []

    def stop(self, name):
        try:
            if runtime._read_state(name) is not None:
                runtime.stop(name, timeout=12)
        except Exception:                                     # noqa: BLE001
            pass

    # ------------------------------------------------------------------ what an update does

    def test_the_folders_the_config_names_are_gone_after_two_updates_and_the_model_looks_missing(self):
        path = self.qwen_config("qwen", vision=True)
        self.two_updates()
        cfg = self.cfg_of(path)
        self.assertFalse(Path(cfg["cwd"]).exists())
        self.assertFalse(Path(cfg["args"][cfg["args"].index("--expert-profile") + 1]).exists())
        report = doctor.inspect(path)
        stale = [c for c in report["components"] if c["status"] != "ok" and c["name"] in ("expert-profile", "vision/exe", "engine")]
        self.assertEqual(stale, [], "the doctor (and so the model list) must follow the current folders: " + json.dumps(stale))

    def test_reading_the_config_follows_the_current_version_in_memory_and_writes_nothing(self):
        path = self.qwen_config("qwen", vision=True)
        new_engine, _ = self.two_updates()
        before = path.read_bytes()
        cfg = profiles.read_run_config(path)
        active = self.control / "0.1.46"
        self.assertEqual(Path(cfg["cwd"]), active)
        self.assertEqual(Path(cfg["args"][cfg["args"].index("--expert-profile") + 1]), active / "data" / "expert-profile.bin")
        self.assertEqual(Path(cfg["vision"]["exe"]), new_engine / "strata-vision")
        self.assertEqual([Path(d) for d in cfg["lib_dirs"]], [new_engine])
        self.assertEqual(Path(cfg["exe"]), self.engine_exe(new_engine))
        self.assertEqual(path.read_bytes(), before)

    def test_the_paths_the_person_typed_and_the_data_are_left_alone(self):
        path = self.qwen_config("qwen")
        self.two_updates()
        cfg = profiles.read_run_config(path)
        args = cfg["args"]
        self.assertEqual(args[args.index("--pack") + 1], str(self.root / "packs" / "iq3_s"))
        self.assertEqual(args[args.index("--lookup-file") + 1], str(self.mine))      # (typed, outside Strata's folders, does not exist: not touched)
        self.assertEqual(args[args.index("--mtp") + 1], "2")
        self.assertEqual(args[args.index("--expert-profile-save") + 1], str(self.control / "0.1.44" / "data" / "learned.bin"),
                         "a path the engine writes is not a file to look for")
        self.assertEqual(cfg["tokenizer"], self.tokenizer)
        self.assertEqual(cfg["model_name"], "fake")

    def test_a_folder_that_still_exists_is_not_replaced_by_a_newer_one(self):
        path = self.qwen_config("qwen")
        self.update_to("0.1.45")                              # 0.1.44 is still there (the version before the newest)
        before = path.read_bytes()
        cfg = profiles.read_run_config(path)
        self.assertEqual(cfg["cwd"], str(self.control / "0.1.44"))
        self.assertEqual(cfg_paths.repair([path]), ([], []))
        self.assertEqual(path.read_bytes(), before)

    # ------------------------------------------------------------------ starts

    def test_a_terminal_start_rewrites_the_config_once_and_says_so_in_one_line(self):
        path = self.qwen_config("term")
        new_engine, _ = self.two_updates()
        self.addCleanup(self.stop, "term")
        started = runtime.start("term", wait=60)
        self.assertTrue(started["ok"], started)
        cfg = self.cfg_of(path)
        self.assertEqual(cfg["cwd"], str(self.control / "0.1.46"))
        self.assertEqual(cfg["exe"], str(self.engine_exe(new_engine)))
        self.assertNotIn("0.1.44", self.without_output_path(path))
        self.assertEqual(len(self.host_log_lines("term")), 1, self.host_log_lines("term"))
        self.assertIn("earlier Strata version", self.host_log_lines("term")[0])
        runtime.stop("term", timeout=12)
        again = runtime.start("term", wait=60)
        self.assertTrue(again["ok"], again)
        self.assertEqual(len(self.host_log_lines("term")), 1, "nothing is rewritten the second time")

    def test_an_app_start_with_die_with_pid_does_the_same(self):
        path = self.qwen_config("app-1")
        self.two_updates()
        self.addCleanup(self.stop, "app-1")
        started = runtime.start("app-1", wait=60, die_with_pid=os.getpid())
        self.assertTrue(started["ok"], started)
        self.assertEqual(self.cfg_of(path)["cwd"], str(self.control / "0.1.46"))

    def test_a_start_planned_on_the_old_file_is_not_refused_because_strata_rewrote_it(self):
        import hashlib
        path = self.qwen_config("app-2")
        self.two_updates()
        self.addCleanup(self.stop, "app-2")
        planned = hashlib.sha256(path.read_bytes()).hexdigest()
        started = runtime.start("app-2", wait=60, expected_source_sha256=planned)
        self.assertTrue(started["ok"], started)

    def test_every_saved_profile_is_repaired_by_one_start(self):
        a, b = self.qwen_config("one"), self.qwen_config("two")
        self.two_updates()
        self.addCleanup(self.stop, "one")
        self.assertTrue(runtime.start("one", wait=60)["ok"])
        self.assertNotIn("0.1.44", self.without_output_path(b), "the Models page reads the other profiles' configs too")

    def test_deepseek_cwd_exe_and_libraries_follow_the_update_too(self):
        path = self.deepseek_config("ds")
        _, new_ds = self.two_updates()
        report = doctor.inspect(path)
        self.assertEqual([c for c in report["components"] if c["name"] == "engine" and c["status"] != "ok"], [])
        notes, problems = cfg_paths.repair([path])
        self.assertEqual(problems, [])
        self.assertEqual(len(notes), 1)
        cfg = self.cfg_of(path)
        self.assertEqual(cfg["exe"], str(new_ds / dsv41_paths.exe_name()))
        self.assertEqual(cfg["lib_dirs"], [str(new_ds)])
        self.assertEqual(cfg["cwd"], str(self.control / "0.1.46"))
        self.assertEqual(cfg["args"], ["--pack", str(self.root / "packs" / "ds"), "--max-context", "8192"])

    def test_deepseek_start_goes_through_the_same_repair(self):
        path = self.deepseek_config("ds-app")
        self.two_updates()
        with mock.patch.object(runtime, "_check_start_candidate", side_effect=RuntimeError("stop after the repair")), \
                mock.patch.object(runtime, "_engine_preflight"):
            with self.assertRaises(RuntimeError):
                runtime.start("ds-app", wait=1, die_with_pid=os.getpid())
        cfg = self.cfg_of(path)
        self.assertIn("0.1.50", cfg["exe"])
        self.assertEqual(cfg["cwd"], str(self.control / "0.1.46"))

    # ------------------------------------------------------------------ a file that is nowhere

    def test_a_file_the_new_version_no_longer_has_stops_the_start_with_its_name(self):
        path = self.qwen_config("lost")
        self.two_updates()
        (self.control / "0.1.46" / "data" / "expert-profile.bin").unlink()
        (self.control / "0.1.45" / "data" / "expert-profile.bin").unlink()
        with self.assertRaises(profiles.ProfileError) as caught:
            runtime.start("lost", wait=5)
        text = str(caught.exception)
        self.assertIn("expert-profile.bin", text)
        self.assertIn("removed by an update", text)
        self.assertIn("run setup again", text)
        self.assertIn("0.1.44", path.read_text(encoding="utf-8"), "nothing is guessed: the file is left as it was")


if __name__ == "__main__":
    unittest.main()
