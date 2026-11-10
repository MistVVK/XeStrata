"""Qwen4 day on an Intel Arc card: a model of the signed feed (strata_control/feed_models.py) gets an Arc run config, in the Terminal (strata_cli/launch/adopt.py,
runner.adopt_prepared_files) and in the app (setup.py --backend sycl in CONTROL mode, tools/strata_mcp.py's install plan).

Found by the Q4-READY rehearsal on the A750 (11-09): `strata launch` of a feed model stopped with "Strata can write the settings itself only for the Qwen3.8-Flash-Next,
Swift, Coder and Gyro files, not for q4rehearsal-a", for a model on the Strata engine and for one on its own shape's engine alike. What these tests pin:
  * a feed model is laid out like Unsloth's sets: --native the first file, no --ple-gguf (the engine finds the n-gram table by its tensor name), its one GGUF when it is one file;
  * no Flash-Next draft layer for it; an entry that says drafter: none writes "mtp": false and "drafting": "off" (as setup_config writes it);
  * a model of its own shape names its engine component (component:strata-q4-lite) and the model's title for the engine's refusal;
  * the expert profile comes from setup_config.neutral_profile_if_other_shape (the shipped one, or a neutral one for another shape);
  * a family that is neither built in nor of the feed is refused with the same words as before, and the built-in families' configs are what they were.
Offline: nothing runs but Python."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
import strata_mcp as M  # noqa: E402
from strata_cli.launch import adopt  # noqa: E402
from strata_control import intel_app, setup_config  # noqa: E402

A750 = {"vendor": "intel", "name": "Intel Arc A750", "vram_gb": 8.0, "usable": True, "index": 0}


def feed_family(model_id: str, *, shape: str | None = None, drafter: str = "none") -> dict:
    """A family as strata_control/feed_models.py._flash_next makes it (the keys the Arc path reads)."""
    fam = {"title": "Qwen4 Rehearsal", "by": "Qwen", "about": "x", "tag": model_id + "-", "name": model_id, "shards": 2, "vision": False, "feed": True,
           "drafter": drafter, "pack_args": ["--compat-bf16"], "file": {}, "hf": {}, "sha256": {}, "license": "Apache 2.0"}
    if shape:
        fam["shape"] = shape
    return fam


class Tree:
    """A data folder with a prepared pack, the model files, the Flash-Next draft layer folder and a Control folder with the shipped profile."""

    def __init__(self, root: Path, pack_name: str, files: list[str]):
        self.root = root
        self.data = root / "Strata-data"
        self.pack = self.data / "packs" / pack_name
        (self.pack / "tokenizer").mkdir(parents=True)
        (self.pack / "tokenizer" / "vocab.json").write_text("{}")
        (self.pack / "index.txt").write_text("x")
        (self.data / "mtp" / "rt").mkdir(parents=True)                     # the Flash-Next draft layer is on this PC: a feed model must still not get it
        self.models = root / "models"
        self.models.mkdir()
        self.files = []
        for name in files:
            (self.models / name).write_bytes(b"x")
            self.files.append(self.models / name)
        self.src = root / "control"
        (self.src / "data").mkdir(parents=True)
        (self.src / "data" / "expert-profile.bin").write_bytes(b"STRP")

    def plan(self, family: str, model: str, fam: dict, notes=None):
        return adopt.plan(family=family, model=model, model_dir=self.models, data_dir=self.data, src=self.src, context=32768, kv="int8",
                          model_name=f"{fam.get('name', family)}-{model.lower()}", family_def=fam, log=self.root / "x.log", backend="sycl", notes=notes,
                          pack_name=self.pack.name)


class AdoptFeedModels(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        p = mock.patch("strata_cli.launch.adopt.intel_driver", lambda: "i915")
        p.start()
        self.addCleanup(p.stop)
        self.profiles = []

        def neutral(shipped, first, pack, GGUFFile, say=print):                # what setup_config does, recorded (the real one reads the GGUF's header)
            self.profiles.append((Path(shipped), Path(first), Path(pack)))
            say("  This model's shape differs from Flash-Next, so a neutral expert profile is used.")
            return Path(pack) / "expert-profile-4x256.bin"
        p = mock.patch.object(setup_config, "neutral_profile_if_other_shape", neutral)
        p.start()
        self.addCleanup(p.stop)

    def test_a_feed_model_on_the_strata_engine_is_laid_out_like_unsloths_sets(self):
        t = Tree(self.root, "iq3_xxs-q4rehearsal-a", ["Q4R-IQ3_XXS-00001-of-00002.gguf", "Q4R-IQ3_XXS-00002-of-00002.gguf"])
        notes = []
        cfg, problems = t.plan("q4rehearsal-a", "IQ3_XXS-q4rehearsal-a", feed_family("q4rehearsal-a"), notes)
        self.assertEqual(problems, [])
        a = cfg["args"]
        self.assertEqual(cfg["exe"], "component:strata-engine")
        self.assertEqual(a[a.index("--native") + 1], str(t.files[0]))
        self.assertNotIn("--ple-gguf", a)
        self.assertNotIn("--mtp", a, "the Flash-Next draft layer is not a feed model's own")
        self.assertEqual((cfg["mtp"], cfg["drafting"]), (False, "off"), "drafter: none -> no drafter at any start")
        self.assertEqual(a[a.index("--expert-profile") + 1], str(t.pack / "expert-profile-4x256.bin"))
        self.assertEqual(self.profiles, [(t.src / "data" / "expert-profile.bin", t.files[0], t.pack)])
        self.assertIn("This model's shape differs from Flash-Next, so a neutral expert profile is used.", notes)
        self.assertEqual((a[a.index("--ple-io") + 1], cfg["backend"]), ("ram", "sycl"))
        self.assertNotIn("shape_title", cfg)

    def test_a_model_of_its_own_shape_names_its_engine_and_may_be_one_file(self):
        t = Tree(self.root, "ud-q4_k_xl-q4rehearsal-b", ["q4lite4.gguf"])
        fam = feed_family("q4rehearsal-b", shape="q4_lite")
        cfg, problems = t.plan("q4rehearsal-b", "UD-Q4_K_XL-q4rehearsal-b", fam)
        self.assertEqual(problems, [])
        self.assertEqual(cfg["exe"], "component:strata-q4-lite")
        self.assertEqual(cfg["shape_title"], "Qwen4 Rehearsal")
        self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1], str(t.files[0]))
        self.assertEqual(adopt.model_files(t.models, "q4rehearsal-b", fam), t.files)
        self.assertEqual(adopt.model_files(t.models, "qwen", {}), [], "a built-in GSQ-RCO family still needs its numbered shards")

    def test_a_feed_model_with_a_drafter_still_gets_no_flash_next_draft_layer(self):
        t = Tree(self.root, "iq3_xxs-q4rehearsal-a", ["Q4R-IQ3_XXS-00001-of-00002.gguf", "Q4R-IQ3_XXS-00002-of-00002.gguf"])
        cfg, _ = t.plan("q4rehearsal-a", "IQ3_XXS-q4rehearsal-a", feed_family("q4rehearsal-a", drafter="mtp"))
        self.assertNotIn("--mtp", cfg["args"])
        self.assertNotIn("drafting", cfg)

    def test_a_family_neither_built_in_nor_of_the_feed_is_refused_as_before(self):
        t = Tree(self.root, "x", ["Q4R-00001-of-00002.gguf", "Q4R-00002-of-00002.gguf"])
        cfg, problems = t.plan("q4rehearsal-a", "X", {"title": "x", "tag": "x-"})
        self.assertIsNone(cfg)
        self.assertEqual(problems, ["Strata can write the settings itself only for the Qwen3.8-Flash-Next, Swift, Coder and Gyro files, not for q4rehearsal-a"])

    def test_the_built_in_families_are_what_they_were(self):
        t = Tree(self.root, "q2_0", ["Qwen3.8-Flash-Next-Q2_0-00001-of-00002.gguf", "Qwen3.8-Flash-Next-Q2_0-00002-of-00002.gguf"])
        cfg, problems = t.plan("qwen", "Q2_0", setup.FAMILIES["qwen"])
        self.assertEqual(problems, [])
        self.assertEqual(cfg["args"], ["--pack", str(t.pack), "--native", str(t.files[0]), "--ple-gguf", str(t.files[-1]),
                                       "--expert-profile", str(t.src / "data" / "expert-profile.bin"), "--expert-cache", "auto", "--prefill", "auto",
                                       "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(t.data / "mtp" / "rt"), "--max-context", "32768", "--kv", "int8",
                                       "--ple-io", "ram", "--vram-reserve-mib", "1024"])
        self.assertEqual(sorted(cfg), ["args", "backend", "cwd", "exe", "lib_dirs", "log", "model_name", "port", "tokenizer"])
        self.assertEqual(self.profiles, [], "a built-in family never reads its GGUF for the profile")


class AppArcPath(unittest.TestCase):
    """The app: tools/strata_mcp.py plans `--backend sycl` for a feed model on an Arc, and setup.py's CONTROL mode fetches a shape engine, packs and writes the config."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fams = {"q4rehearsal-a": feed_family("q4rehearsal-a"), "q4rehearsal-b": feed_family("q4rehearsal-b", shape="q4_lite")}
        models = {"IQ3_XXS-q4rehearsal-a": {"about": "x", "download_gb": 75.8, "ram_gb": 40, "arena_gb": 40, "families": ("q4rehearsal-a",), "shards": 2,
                                             "file": "Q4R-IQ3_XXS-{i:05d}-of-00002.gguf", "quant": "IQ3_XXS", "vision": False, "budget": True},
                  "UD-Q4_K_XL-q4rehearsal-b": {"about": "x", "download_gb": 3.5, "ram_gb": 8, "arena_gb": 3, "families": ("q4rehearsal-b",), "shards": 1,
                                                "file": "q4lite4.gguf", "quant": "UD-Q4_K_XL", "vision": False, "budget": True}}
        for p in (mock.patch.dict(setup.FAMILIES, fams), mock.patch.dict(setup.MODELS, models)):
            p.start()
            self.addCleanup(p.stop)

    def test_the_arc_install_plan_takes_a_feed_model(self):
        folder = self.root / "q4"
        folder.mkdir()
        for i in (1, 2):
            (folder / f"Q4R-IQ3_XXS-{i:05d}-of-00002.gguf").write_bytes(b"x")
        self.assertTrue(intel_app.can_set_up("q4rehearsal-a", setup.FAMILIES["q4rehearsal-a"]))
        self.assertFalse(intel_app.can_set_up("deepseek41", {"title": "DeepSeek"}))
        s = M.Strata(ROOT)
        s._setup, s._setup_tried = setup, True                              # the tables with the feed models in them (as feed_models_setup adds them)
        s.hardware = lambda fresh=False: {"ram_gb": 64.0, "gpus": [dict(A750)], "cpu": {"avx2": True}, "engine_vendor": "intel"}
        s.disk_free = lambda p: 1000.0
        with mock.patch.object(M.sys, "platform", "linux"), mock.patch.object(M, "WIN", False):
            _plan, args = M.Tools(s).install_plan("q4rehearsal-a", "IQ3_XXS-q4rehearsal-a", 32768, "no", "int8", None, None, None, None, None, None, str(folder), "dflash2")
        self.assertEqual(args[args.index("--family") + 1], "q4rehearsal-a")
        self.assertEqual(args[-2:], ["--backend", "sycl"])
        self.assertNotIn("--drafter", args)

    def test_setup_control_mode_fetches_the_shape_engine_and_writes_its_config(self):
        folder, data, home = self.root / "b", self.root / "Strata-data", self.root / "install"
        for d in (folder, data, home):
            d.mkdir()
        (folder / "q4lite4.gguf").write_bytes(b"x")
        fetched = []

        def fake_pack(args):
            pack = Path(args.prepare_pack)
            (pack / "tokenizer").mkdir(parents=True, exist_ok=True)
            (pack / "tokenizer" / "vocab.json").write_text("{}")
            (pack / "index.txt").write_text("x")
            return 0
        a = mock.Mock(family="q4rehearsal-b", model="UD-Q4_K_XL-q4rehearsal-b", gguf_dir=str(folder), models_dir=None, data_dir=None, context=32768, kv="int8",
                      port=None, host=None, api_key=None, browser=None, check=False, yes=True, move_data=False, set_default_data_dir=False, prepare_pack=None, feed=None)
        with mock.patch.object(setup, "CONTROL", True), mock.patch.object(setup, "_CONTROL_HOME", home), \
                mock.patch.object(setup, "data_folder_for", lambda a: (data, [])), mock.patch.object(setup, "prepare_pack_only", fake_pack), \
                mock.patch.object(setup, "shape_engine_fetch", lambda a: fetched.append(dict(setup.SHAPE_ENGINE))), \
                mock.patch.object(setup_config, "neutral_profile_if_other_shape", lambda shipped, first, pack, G, say=print: Path(pack) / "expert-profile-4x256.bin"), \
                mock.patch.object(intel_app, "arc_cards", lambda root="": [dict(A750)]), \
                mock.patch("strata_cli.launch.adopt.intel_driver", lambda: "i915"), \
                mock.patch.object(intel_app, "supported_here", lambda platform=None: True):
            self.assertEqual(setup.intel_control_setup(a), 0)
        self.assertEqual(fetched, [{"shape": "q4_lite", "title": "Qwen4 Rehearsal"}])
        cfg = json.loads((home / "strata-q4rehearsal-b-ud-q4_k_xl-q4rehearsal-b.json").read_text(encoding="utf-8"))
        self.assertEqual((cfg["exe"], cfg["backend"], cfg["drafting"], cfg["mtp"]), ("component:strata-q4-lite", "sycl", "off", False))
        self.assertEqual(cfg["shape_title"], "Qwen4 Rehearsal")


if __name__ == "__main__":
    unittest.main()
