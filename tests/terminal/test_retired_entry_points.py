"""Every place a DeepSeek model can still come in (DeepSeek removed in release B; strata_control/retired.py) refuses it in plain words, at
once, and leaves the person's config and files exactly as they were.  No model is needed: fake configs, folders and GGUF headers."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from strata_control import cfg_paths, models, retired, runtime  # noqa: E402
from strata_control.profiles import ProfileError  # noqa: E402

D1_HEAD = "is a DeepSeek model. This version of Strata no longer runs DeepSeek models, so it was not started."


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ds_install(d: Path, *, model_bytes: int = 3_000_000) -> tuple[Path, Path, Path]:
    """(config, model folder, pack) of a fake DeepSeek install as setup 0.2.13 wrote it."""
    models_dir = d / "data" / "models" / "deepseek-v4.1-flash-ist-das-2.00bpw"
    models_dir.mkdir(parents=True)
    (models_dir / "DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf").write_bytes(b"\0" * model_bytes)
    pack = d / "data" / "packs" / "deepseek"
    pack.mkdir(parents=True)
    (pack / "ds4.txt").write_text(f"format gguf\nmodel_dir {models_dir}\n", encoding="utf-8")
    eng = d / "engine-dsv41" / "0.1.59-linux-x86_64-cuda-cuda13"
    eng.mkdir(parents=True)
    exe = eng / "strata-dsv41"
    exe.write_text("#!/bin/sh\necho started > \"$0.ran\"\n", encoding="utf-8")
    cfg = {"exe": str(exe), "args": ["--pack", str(pack), "--max-context", "131072", "--eos", "1"], "cwd": str(d),
           "tokenizer": str(pack / "tokenizer"), "model_name": "deepseek-v4.1-flash", "arch": "deepseek41", "port": 8080,
           "lib_dirs": [str(d / "components" / "engine" / "0.1.50-linux-x86_64-cuda" / "lib")]}
    path = d / "strata-deepseek-v4.1-flash-ist-das-2.00bpw.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path, models_dir, pack


class StartRefused(unittest.TestCase):
    """The app and `strata start` (and a saved `strata launch` profile) start through runtime.start."""

    def test_runtime_start_refuses_before_anything_is_written(self):
        with tempfile.TemporaryDirectory() as d:
            path, models_dir, pack = ds_install(Path(d))
            before = sha(path)
            profile = {"schema_version": 1, "id": "launch-ds", "source_config": str(path), "overrides": {}}
            with mock.patch.object(runtime.profiles, "load", return_value=(profile, "0" * 16)), \
                    mock.patch.object(runtime, "_repair_config_paths", side_effect=AssertionError("wrote configs")), \
                    mock.patch.object(runtime.subprocess, "Popen", side_effect=AssertionError("started a process")):
                with self.assertRaises(ProfileError) as caught:
                    runtime.start("launch-ds", wait=0)
            text = str(caught.exception)
            self.assertTrue(text.startswith("DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw " + D1_HEAD), text)
            self.assertIn(f"Its files are still on this computer: 0.0 GB in {models_dir} and {pack}. Nothing was changed or deleted.", text)
            self.assertTrue(text.endswith("you can delete that folder to free the space."), text)
            self.assertEqual(sha(path), before)
            self.assertFalse(Path(json.loads(path.read_text())["exe"] + ".ran").exists())

    def test_a_qwen_profile_is_not_stopped_here(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strata-iq3_s.json"
            path.write_text(json.dumps({"exe": "/x/strata", "args": [], "arch": "qwen3next", "model_name": "q"}), encoding="utf-8")
            profile = {"schema_version": 1, "id": "app-1", "source_config": str(path), "overrides": {}}
            with mock.patch.object(runtime.profiles, "load", return_value=(profile, "0" * 16)):
                runtime._refuse_deepseek("app-1")        # no error


class ConfigsNeverWritten(unittest.TestCase):
    def test_path_repair_leaves_a_deepseek_config_alone(self):
        with tempfile.TemporaryDirectory() as d:
            path, _m, _p = ds_install(Path(d))
            before = sha(path)
            notes, problems = cfg_paths.repair([path])
            self.assertEqual((notes, problems), ([], []))
            self.assertEqual(sha(path), before)


class ModelList(unittest.TestCase):
    """The app's model list and `strata models` read strata_control/models.py."""

    def test_listed_as_retired_without_inspection(self):
        with tempfile.TemporaryDirectory() as d:
            path, models_dir, _p = ds_install(Path(d))
            before = sha(path)
            with mock.patch.object(models, "inspect_components", side_effect=AssertionError("inspected")):
                out = models.inspect("deepseek-v4.1-flash-ist-das-2.00bpw", override=d)["model"]
            self.assertEqual(out["component_status"], "retired")
            self.assertEqual(out["retired"], "deepseek")
            self.assertEqual(out["name"], "DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw")
            self.assertTrue(out["refusal"].startswith("DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw " + D1_HEAD))
            self.assertIn(str(models_dir), out["retired_line"])
            self.assertEqual(sha(path), before)

    def test_import_of_deepseek_files_is_named(self):
        with tempfile.TemporaryDirectory() as d:
            _c, models_dir, _p = ds_install(Path(d))
            with self.assertRaises(ProfileError) as caught:
                models.plan_import(str(models_dir))
            self.assertIn("These are DeepSeek model files (0.0 GB in", str(caught.exception))
            self.assertIn("nothing in the folder was changed", str(caught.exception))


class Server(unittest.TestCase):
    """serve/server.py started directly (a script, Docker, a gateway) with a DeepSeek config: stops before the engine starts."""

    def run_server(self, cfg: dict) -> tuple[int, str]:
        from serve import server
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps(cfg), encoding="utf-8")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = server.main(["--config", str(p), "--port", "0"])
            return code, out.getvalue()

    def test_arch(self):
        code, out = self.run_server({"exe": "/nowhere/strata-dsv41", "args": [], "arch": "deepseek41"})
        self.assertEqual(code, 2)
        self.assertIn("this config runs a DeepSeek model (arch deepseek41)", out)
        self.assertIn("so the server stops here. The model files were not changed.", out)

    def test_tokenizer_without_arch(self):
        with tempfile.TemporaryDirectory() as d:
            tok = Path(d) / "tok"
            tok.mkdir()
            (tok / "tokenizer.json").write_text(json.dumps({"pre": "deepseek3"}), encoding="utf-8")
            code, out = self.run_server({"exe": "/nowhere/strata", "args": [], "tokenizer": str(tok)})
        self.assertEqual(code, 2)
        self.assertIn("(DeepSeek tokenizer)", out)


class Setup(unittest.TestCase):
    def run_setup(self, *argv: str) -> tuple[int, str]:
        import setup
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["setup.py", *argv]), contextlib.redirect_stdout(out):
            code = setup.main()
        return code, out.getvalue()

    def test_family_and_model_names(self):
        for argv in (("--family", "deepseek41"), ("--family", "deepseek41", "--model", "IST-DAS-2.00bpw"), ("--model", "IST-DAS-2.25bpw"),
                     ("--model=ISTA-DAS-Coder",)):
            code, out = self.run_setup(*argv, "--yes", "--no-start")
            self.assertEqual(code, 2, argv)
            self.assertIn("Strata no longer offers DeepSeek models. Run `strata launch` to choose one of the Qwen models.", out)

    def test_config_lists_leave_deepseek_out(self):
        import setup
        with tempfile.TemporaryDirectory() as d:
            path, _m, _p = ds_install(Path(d))
            qwen = Path(d) / "strata-iq3_s.json"
            qwen.write_text(json.dumps({"exe": "/x/strata", "args": ["--model", "/m/q.gguf"], "model_name": "q"}), encoding="utf-8")
            before = sha(path)
            with mock.patch.object(setup, "home", return_value=Path(d)):
                self.assertEqual(setup.installed_configs(), [qwen])
                self.assertEqual(setup.retired_configs(), [path])
                qwen.unlink()
                self.assertIsNone(setup.previous_config([Path(d)], {"installs": []}))     # never set up "like" the DeepSeek model
            self.assertEqual(sha(path), before)
            self.assertEqual(retired.left_as_is(retired.title(None, path)),
                             "DeepSeek-V4.1-Flash ISTA-DAS 2.00 bpw: a DeepSeek model, left as it is (this version of Strata no longer "
                             "runs DeepSeek models).")


class Launch(unittest.TestCase):
    def parse(self, *argv: str) -> argparse.Namespace:
        from strata_cli.launch import cmd
        ap = argparse.ArgumentParser()
        cmd.add_parsers(ap.add_subparsers(dest="command"))
        return ap.parse_args(["launch", *argv])

    def test_typed_names(self):
        from strata_cli.launch import cmd
        for argv in (("--model", "IST-DAS-2.00bpw"), ("--family", "deepseek41", "--model", "UD-Q8_K_XL"), ("--model", "ISTA-DAS-Coder")):
            with self.assertRaises(ValueError) as caught:
                cmd.settings_from_args(self.parse(*argv), catalog=object())
            self.assertEqual(str(caught.exception), retired.NOT_OFFERED, argv)

    def test_saved_preset(self):
        from strata_cli.launch import cmd, store
        from strata_cli.launch.settings import LaunchSettings
        saved = LaunchSettings()
        saved.family, saved.model = "deepseek41", "IST-DAS-2.00bpw"
        with mock.patch.object(store, "load", return_value=saved):
            with self.assertRaises(ValueError) as caught:
                cmd.settings_from_args(self.parse("--preset", "big"), catalog=object())
        self.assertEqual(str(caught.exception), retired.NOT_OFFERED)

    def test_own_files(self):
        from strata_cli.launch import runner
        with tempfile.TemporaryDirectory() as d:
            _c, models_dir, _p = ds_install(Path(d))
            fake = mock.Mock()
            fake.s = mock.Mock(model_path=str(models_dir))
            with self.assertRaises(runner.LaunchError) as caught:
                runner.Runner.resolve_own_files(fake)
            self.assertIn("These are DeepSeek model files", caught.exception.text)
            self.assertEqual(caught.exception.code, 2)

    def test_qwen_names_pass(self):
        from strata_cli.launch import cmd

        class Cat:
            def resolve(self, model, family):
                return ("qwen", model)
        s, _ = cmd.settings_from_args(self.parse("--model", "IQ3_S"), catalog=Cat())
        self.assertEqual((s.family, s.model), ("qwen", "IQ3_S"))


if __name__ == "__main__":
    unittest.main()
