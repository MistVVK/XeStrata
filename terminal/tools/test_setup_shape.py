"""setup.py and a feed model of its own shape (strata_control/shape_engine.py, strata_control/feed_models.py).

    python -m pytest tools/test_setup_shape.py

A Qwen4 model whose geometry is not Flash-Next's runs on its own engine component. The model list puts it in the same tables as every feed model, with the shape; setup's engine
step fetches that engine (not the Strata engine's) and the run config names it ("component:strata-q4-lite"); the Terminal's launch checks it in its first step. Every other model
is exactly what it was. The engine fetch, downloads and processes are mocked; nothing is fetched from the network.
"""
from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from strata_control import engine_component as ec, feed_models, shape_engine  # noqa: E402
import test_setup_feed_models as base  # noqa: E402


def shaped(**kw):
    kw = {"id": "q4lite-test", "family": "qwen4exp_q4_lite", "drafter": "none", "pictures": False, "mmproj": None, "min_engine": "", **kw}
    return base.flash_next(**kw)


def clean(entry: dict) -> dict:
    entry.pop("mmproj", None)
    return entry


class Registered(base.Base):
    def setUp(self):
        super().setUp()
        self.result = feed_models.apply(self.setup, base.parsed(clean(shaped())), {"qwen4exp"})        # the Strata engine runs qwen4exp only: not what decides
        self.fam = self.setup.FAMILIES["q4lite-test"]

    def test_it_is_listed_and_installable_though_the_strata_engine_does_not_list_its_family(self):
        self.assertEqual((self.result.registered, self.result.unavailable), (["q4lite-test"], {}))
        self.assertEqual(self.fam["shape"], "q4_lite")
        self.assertTrue(self.fam["feed"])
        self.assertNotIn("q4lite-test", self.setup.FEED_UNAVAILABLE)

    def test_its_min_engine_is_not_compared_with_the_strata_engine(self):
        """The app marks a size "needs a newer engine" when the Strata engine is older than the size's `engine`; a shape model runs on its own engine."""
        feed_models.apply(self.setup, base.parsed(clean(shaped(id="q4lite-min", min_engine="0.1.59"))), None)
        sizes = [m for m in self.setup.MODELS.values() if "q4lite-min" in m.get("families", ())]
        self.assertTrue(sizes)
        self.assertTrue(all("engine" not in m for m in sizes))
        feed_models.apply(self.setup, base.parsed(base.flash_next(id="plain-min", min_engine="0.1.59")), None)
        self.assertTrue(all(m.get("engine") == (0, 1, 59) for m in self.setup.MODELS.values() if "plain-min" in m.get("families", ())))

    def test_a_plain_qwen4_model_has_no_shape(self):
        feed_models.apply(self.setup, base.parsed(base.flash_next(id="plain-one", title="Plain One")), {"qwen4exp"})
        self.assertNotIn("shape", self.setup.FAMILIES["plain-one"])

    def test_it_is_offered_on_a_mac_like_the_two_file_sizes(self):
        from strata_control import mac_fit
        self.assertIn("q4lite-test", mac_fit.FEED_FAMILIES)
        rows = {k: v for k, v in mac_fit.MAC_MODELS.items() if v.get("family") == "q4lite-test"}
        self.assertEqual(len(rows), 1)
        (key, row), = rows.items()
        self.assertTrue(row["no_mtp"], "its entry says drafter none")
        self.assertEqual(mac_fit.app_sizes("q4lite-test"), [key])

    def test_the_mac_arguments_for_a_one_file_model_without_a_draft_layer(self):
        from strata_control import mac_fit
        (key,) = [k for k, v in mac_fit.MAC_MODELS.items() if v.get("family") == "q4lite-test"]
        shard = Path("model.gguf")
        args = mac_fit.engine_args(Path("pack"), [shard], Path("rt"), 8192, False, 128.0, "profile.bin", key)
        self.assertEqual(args[args.index("--ple-gguf") + 1], "model.gguf", "one file: its own PLE table")
        self.assertNotIn("--mtp", args)
        plain = mac_fit.engine_args(Path("pack"), [Path("a-1.gguf"), Path("a-2.gguf")], Path("rt"), 8192, False, 128.0, "profile.bin", "IQ3_S")
        self.assertEqual(plain[plain.index("--ple-gguf") + 1], "a-2.gguf")
        self.assertIn("--mtp", plain, "a built-in model is exactly what it was")

    def test_the_same_list_without_an_engine_to_ask_gives_the_same_registration(self):
        other = base.fresh_setup("setup_other")
        feed_models.apply(other, base.parsed(clean(shaped())), None)
        self.assertEqual(other.FAMILIES["q4lite-test"]["shape"], "q4_lite")

    def test_a_family_older_controls_do_not_know_is_unavailable_for_them(self):
        """What the Control before this does with the same entry: its install table lacks the family, so the model is listed as unavailable and never offered."""
        from strata_control import feed as feedmod
        self.assertNotIn("qwen4exp_q4_lite", ("qwen4exp", "qwen35moe"))
        self.assertIsNone(feedmod.FeedModel.shape.fget(base.parsed(base.flash_next())[0]))


class EngineStep(base.Base):
    def setUp(self):
        super().setUp()
        feed_models.apply(self.setup, base.parsed(clean(shaped())), None)
        self.addCleanup(self.setup.SHAPE_ENGINE.clear)

    def test_the_run_config_names_the_shape_engine_only_for_a_model_of_that_shape(self):
        self.assertEqual(self.setup.engine_exe(Path("engine")), "component:strata-engine" if self.setup.CONTROL else str(Path("engine") / self.setup.EXE))
        self.setup.SHAPE_ENGINE["shape"] = "q4_lite"
        self.assertEqual(self.setup.engine_exe(Path("engine")), "component:strata-q4-lite")
        self.setup.SHAPE_ENGINE.clear()
        self.assertNotEqual(self.setup.engine_exe(Path("engine")), "component:strata-q4-lite")

    def test_engine_step_fetches_the_shape_engine_and_not_the_strata_engine(self):
        self.setup.SHAPE_ENGINE["shape"] = "q4_lite"
        folder = self.root / "components" / "shapes" / "q4_lite" / "engine" / "0.1.59-linux-x86_64-cuda"
        folder.mkdir(parents=True)
        found = mock.Mock(root=str(folder), version="0.1.59", backend="cuda", platform="linux-x86_64")
        with mock.patch.object(shape_engine, "ensure", return_value=folder / "strata") as ensure, \
                mock.patch.object(shape_engine, "find", return_value=found), \
                mock.patch.object(self.setup, "llama_cpp_if_here", return_value=Path("llama")), \
                mock.patch.object(self.setup, "get_llama_cpp", return_value=Path("llama")), \
                mock.patch.object(self.setup, "control_engine_step", side_effect=AssertionError("the Strata engine step must not run")), \
                mock.patch.object(ec, "resolve", side_effect=AssertionError("the Strata engine must not be resolved")):
            llama, eng, meta, lib_dirs, gpu, vision = self.setup.engine_step(argparse.Namespace(feed="/some/feed"), {"index": 0}, False, False, 13, "none")
        self.assertEqual(ensure.call_args.args[:1], ("q4_lite",))
        self.assertEqual(ensure.call_args.args[1], "/some/feed")
        self.assertEqual((eng, meta["version"], meta["shape"], lib_dirs, vision), (folder, "0.1.59", "q4_lite", [], "none"))

    def test_the_lines_and_the_run_config_name_the_models_title(self):
        """A person reads the model's title, not the geometry's name; the run config keeps the title for the refusal at a start (strata_control/shape_engine.py USE_TITLE)."""
        self.setup.SHAPE_ENGINE.update(shape="q4_lite", title="Q4 Lite Shape")
        folder = self.root / "components" / "shapes" / "q4_lite" / "engine" / "0.1.59-linux-x86_64-cuda"
        folder.mkdir(parents=True)
        found = mock.Mock(root=str(folder), version="0.1.59", backend="cuda", platform="linux-x86_64")
        said = []
        with mock.patch.object(shape_engine, "ensure", return_value=folder / "strata") as ensure, mock.patch.object(shape_engine, "find", return_value=found),                 mock.patch.object(self.setup, "llama_cpp_if_here", return_value=Path("llama")), mock.patch.object(self.setup, "get_llama_cpp", return_value=Path("llama")),                 mock.patch.object(self.setup, "ok", side_effect=said.append):
            self.setup.engine_step(argparse.Namespace(feed=None), {"index": 0}, False, False, 13, "none")
        self.assertEqual(ensure.call_args.kwargs["title"], "Q4 Lite Shape")
        self.assertTrue(any(line.startswith("engine for Q4 Lite Shape: 0.1.59") for line in said), said)
        self.assertFalse(any("q4_lite" in line.split(" in ")[0] for line in said), said)
        cfg = {"exe": "component:strata-q4-lite", "args": [], "model_name": "x"}
        target = self.root / "strata-q4lite.json"
        self.setup.write_setup_config(target, cfg)
        self.assertEqual(json.loads(target.read_text(encoding="utf-8"))["shape_title"], "Q4 Lite Shape")
        plain = {"exe": "component:strata-engine", "args": [], "model_name": "y"}      # another model's config is exactly what it was
        self.setup.write_setup_config(self.root / "strata-other.json", plain)
        self.assertNotIn("shape_title", json.loads((self.root / "strata-other.json").read_text(encoding="utf-8")))

    def test_without_a_shape_the_strata_engine_step_runs_as_before(self):
        self.assertEqual(self.setup.SHAPE_ENGINE, {})
        with mock.patch.object(self.setup, "control_engine_step", return_value=("L", "E", {}, [], {}, "none")) as step, mock.patch.object(self.setup, "CONTROL", True):
            self.assertEqual(self.setup.engine_step(argparse.Namespace(), {}, False, False, 13, "none")[1], "E")
        step.assert_called_once()

    def test_no_engine_for_the_shape_stops_before_the_model_is_downloaded(self):
        self.setup.SHAPE_ENGINE["shape"] = "q4_lite"
        with mock.patch.object(shape_engine, "ensure", return_value=None), \
                mock.patch.object(self.setup, "llama_cpp_if_here", return_value=Path("llama")), mock.patch.object(self.setup, "get_llama_cpp", return_value=Path("llama")), \
                mock.patch.dict("os.environ", {"STRATA_ALLOW_NO_ENGINE": ""}):
            with self.assertRaises(SystemExit):
                self.setup.engine_step(argparse.Namespace(feed=None), {"index": 0}, False, False, 13, "none")

    def test_the_person_may_go_on_without_the_engine_and_set_it_up_later(self):
        self.setup.SHAPE_ENGINE["shape"] = "q4_lite"
        with mock.patch.object(shape_engine, "ensure", return_value=None), \
                mock.patch.object(self.setup, "llama_cpp_if_here", return_value=Path("llama")), mock.patch.object(self.setup, "get_llama_cpp", return_value=Path("llama")), \
                mock.patch.dict("os.environ", {"STRATA_ALLOW_NO_ENGINE": "1"}):
            out = self.setup.engine_step(argparse.Namespace(feed=None), {"index": 0}, False, False, 13, "none")
        self.assertEqual(out[2]["component"], "missing")


class Launch(unittest.TestCase):
    """The Terminal's first step reads the shape from setup's table and checks that engine, not the Strata engine's."""

    def runner(self, families):
        from strata_cli.launch import runner as R
        fake = mock.Mock()
        fake.s.family = "q4lite-test"
        fake.catalog.strata.setup_module.return_value = mock.Mock(FAMILIES=families)
        fake.multi_file.return_value = False
        return R, fake

    def test_model_shape_reads_the_family_entry(self):
        R, fake = self.runner({"q4lite-test": {"shape": "q4_lite"}})
        self.assertEqual(R.Runner.model_shape(fake), "q4_lite")
        R, fake = self.runner({"q4lite-test": {}})
        self.assertIsNone(R.Runner.model_shape(fake))
        R, fake = self.runner({})
        self.assertIsNone(R.Runner.model_shape(fake))
        fake.catalog.strata.setup_module.side_effect = RuntimeError("no table")
        self.assertIsNone(R.Runner.model_shape(fake))

    def test_step_one_checks_the_shape_engine_and_never_the_strata_engine(self):
        R, fake = self.runner({"q4lite-test": {"shape": "q4_lite"}})
        fake.model_shape.return_value = "q4_lite"
        with mock.patch.object(ec, "status", side_effect=AssertionError("the Strata engine is not looked at")):
            R.Runner.ensure_engine(fake)
        fake.ensure_shape_engine.assert_called_once_with("q4_lite")

    def test_a_missing_engine_is_the_launchs_no_engine_question(self):
        R, fake = self.runner({})
        fake.components_dir = None
        fake.feed = "/feed"
        with mock.patch.object(shape_engine, "ensure", side_effect=lambda shape, source, **kw: (kw["emit"]("There is no engine for this model (q4_lite)."), None)[1]):
            R.Runner.ensure_shape_engine(fake, "q4_lite")
        fake.no_engine.assert_called_once_with("There is no engine for this model (q4_lite).")

    def test_the_launch_lines_name_the_title(self):
        R, fake = self.runner({"q4lite-test": {"shape": "q4_lite", "title": "Q4 Lite Shape"}})
        self.assertEqual(R.Runner.model_title(fake), "Q4 Lite Shape")
        fake.model_title.return_value = "Q4 Lite Shape"
        fake.components_dir = None
        fake.feed = "/feed"
        events = []
        fake.emit.side_effect = lambda stage, text, *a, **k: events.append(text)
        found = mock.Mock(version="0.1.59", backend="cuda")
        with mock.patch.object(shape_engine, "ensure", return_value=Path("/x/strata")) as ensure, mock.patch.object(shape_engine, "find", return_value=found):
            R.Runner.ensure_shape_engine(fake, "q4_lite")
        self.assertEqual(ensure.call_args.kwargs["title"], "Q4 Lite Shape")
        self.assertIn("Checking the engine for Q4 Lite Shape…", events)
        self.assertIn("Engine for Q4 Lite Shape 0.1.59 (cuda) is installed.", events)
        self.assertFalse([e for e in events if "q4_lite" in e], events)


if __name__ == "__main__":
    unittest.main()
