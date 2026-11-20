"""Tencent Hy4-Flash in setup.py and the catalog (tools/strata_mcp.py): the family table, the menu lines, when the family is offered, the download from a configured source (one file,
parts joined, a source that fails and the next one), the sizes and hashes, the model rows with the fit words, the install plan and the expected files.

    python -m pytest tools/test_setup_hy4.py
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
import setup  # noqa: E402
import strata_mcp as M  # noqa: E402
from strata_control import hy4_family, hy4_fit  # noqa: E402


def tiny_files(names=("Hy4-Flash-SQ-64.strata", hy4_family.ENGRAM_FILE)) -> tuple:
    """Small stand-ins for the real files: the table with their sizes and hashes, and the bytes."""
    table, data = dict(hy4_family.FILES), {}
    for i, name in enumerate(names):
        blob = (name.encode() * 50)[: 1000 + i * 77]
        table[name] = (len(blob), hashlib.sha256(blob).hexdigest())
        data[name] = blob
    return table, data


class Table(unittest.TestCase):
    def test_the_family_matches_the_rule_and_the_files(self):
        fam = setup.HY4_FAMILIES["hy4-flash"]
        self.assertEqual(set(fam["models"]), set(hy4_fit.BUILDS))
        self.assertEqual(fam["arch"], "hy4flash")
        self.assertEqual(fam["tag"], "hy4-flash-")
        for b in fam["models"]:
            name = hy4_fit.BUILDS[b]["file"]
            size, sha = hy4_family.FILES[name]
            self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", sha), name)
            self.assertAlmostEqual(size / 1e9, hy4_fit.BUILDS[b]["file_gb"], delta=0.006)   # the card's GB and the exact bytes agree
        self.assertAlmostEqual(hy4_family.FILES[hy4_family.ENGRAM_FILE][0] / 1e9, hy4_fit.ENGRAM_GB, delta=0.006)
        self.assertAlmostEqual(hy4_family.FILES[hy4_family.VISION_FILE][0] / 1e9, hy4_fit.VISION_GB, delta=0.001)

    def test_sq_32_is_removed_everywhere(self):
        # owner, 2026-11-20: "we will REMOVE the 32GB variant ... it is too weak": SQ-64, SQ-96 and SQ-128 stay
        fam = setup.HY4_FAMILIES["hy4-flash"]
        self.assertEqual(list(fam["models"]), ["SQ-64", "SQ-96", "SQ-128"])
        self.assertNotIn("SQ-32", hy4_family.BUILD_ABOUT)
        self.assertNotIn("Hy4-Flash-SQ-32.strata", hy4_family.FILES)
        self.assertEqual(sorted(n for n in hy4_family.FILES if n.startswith("Hy4-Flash-SQ-")), ["Hy4-Flash-SQ-128.strata", "Hy4-Flash-SQ-64.strata", "Hy4-Flash-SQ-96.strata"])
        with self.assertRaises(KeyError):
            hy4_family.files_of("SQ-32")
        self.assertIsNone(hy4_fit.build_of_model_name("hy4-flash-sq-32"))
        self.assertNotIn("SQ-32", M.Strata(HERE.parent).own_size_names())
        import subprocess
        r = subprocess.run([sys.executable, str(HERE.parent / "setup.py"), "--family", "hy4-flash", "--model", "SQ-32", "--yes", "--no-start"], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 2)
        self.assertIn("invalid choice: 'SQ-32'", r.stderr)

    def test_every_file_has_a_whole_file_hash_and_the_engram_section_hash_is_kept_apart(self):
        for name, (size, sha) in hy4_family.FILES.items():
            self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", sha or ""), name)
        whole = hy4_family.FILES[hy4_family.ENGRAM_FILE][1]
        self.assertEqual(whole, "ab0cf9b3f9c64bd25f6212d4508e7443ab8b9bc133fbccc7bdfb83510a20c2c1")
        self.assertEqual(hy4_family.ENGRAM_SECTION4_SHA256, "176944953bf2b601645d206171964f7fa51b77522ef5dc14ae61b0eb1c803d74")
        self.assertNotEqual(whole, hy4_family.ENGRAM_SECTION4_SHA256)
        self.assertEqual(hy4_family.FILES[hy4_family.VISION_FILE][1], "ef09bceff8f01b8d226d2dc56aef09cec719b21b45783ee9b5d64e7241b72a09")

    def test_it_is_not_in_the_tables_other_tests_pin(self):
        self.assertNotIn("hy4-flash", setup.FAMILIES)
        self.assertNotIn("hy4-flash", setup.OWN_PATH_FAMILIES)

    def test_an_8_gib_card_with_64_gib_is_offered_sq_64_with_ssd_reads_at_128k(self):
        # HY4-CUDA: SQ-64 (56 GB of experts) + the 5.7 GB of host KV do not fit the 58 GB usable of a 64 GiB PC: about 3.0 GB are read from the SSD.  There is no SQ-32 any more
        # (owner, 2026-11-20): SQ-64 stays the recommendation, still at 128K, with the "may run slower" words; Hy4-Flash is not hidden
        lines = dict(setup.hy4_menu_lines(8, 64, False, 131072, True))
        self.assertEqual([b for b, t in setup.hy4_menu_lines(8, 64, False, 131072, True) if "(recommended)" in t], ["SQ-64"])
        self.assertIn("a few experts are read from the SSD", lines["SQ-64"])
        self.assertNotIn("SQ-32", lines)
        self.assertEqual([b for b, t in setup.hy4_menu_lines(12, 64, False, 131072, True) if "(recommended)" in t], ["SQ-64"])

    def test_menu_lines_recommend_by_the_authors_table(self):
        for vram, ram, want in ((12, 64, "SQ-64"), (12, 32, "SQ-64"), (12, 128, "SQ-128")):
            lines = setup.hy4_menu_lines(vram, ram, False)
            self.assertEqual([b for b, _ in lines], ["SQ-128", "SQ-96", "SQ-64"])
            self.assertEqual([b for b, t in lines if "(recommended)" in t], [want])
        text = dict(setup.hy4_menu_lines(12, 64, False))
        self.assertIn("fits in memory", text["SQ-64"])
        self.assertIn("streams experts from the SSD", text["SQ-128"])

    def test_pc_figures(self):
        self.assertEqual(setup.hy4_pc({"vram_gb": 12.0}, 64.0), (12.0, 64.0, False))
        vram, pool, unified = setup.hy4_pc({"uma": True, "vram_gb": 115.8}, 128.0)
        self.assertIsNone(vram)
        self.assertTrue(unified)
        self.assertAlmostEqual(pool, 107.85, delta=0.1)


class Offered(unittest.TestCase):
    def test_hidden_until_there_is_a_source_or_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            ns = mock.Mock(gguf_dir=None, models_dir=d)
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("STRATA_HY4_SOURCE", None)
                with mock.patch.object(hy4_family, "SOURCES_FILE", Path(d) / "none.json"):
                    self.assertFalse(setup.hy4_offered(ns))
                    os.environ["STRATA_HY4_SOURCE"] = "https://example.invalid/hy4"
                    self.assertTrue(setup.hy4_offered(ns))
                    os.environ.pop("STRATA_HY4_SOURCE")
                    table, data = tiny_files()
                    folder = Path(d) / "hy4-flash"
                    folder.mkdir()
                    (folder / "Hy4-Flash-SQ-64.strata").write_bytes(data["Hy4-Flash-SQ-64.strata"])
                    with mock.patch.object(hy4_family, "FILES", table):
                        self.assertTrue(setup.hy4_offered(ns))

    def test_the_shipped_sources_file_is_empty_on_purpose(self):
        self.assertEqual(hy4_family.sources(env={}, shipped=hy4_family.SOURCES_FILE), [])

    def test_source_forms(self):
        env = {"STRATA_HY4_SOURCE": "https://models.example/hy4/"}
        s = hy4_family.sources(env=env, cfg={"hy4_source": {"kind": "http-parts", "base": "https://hf.example/x"}}, shipped=Path("nope.json"))
        self.assertEqual([x["kind"] for x in s], ["http-parts", "http"])
        self.assertEqual(hy4_family.urls_for("Hy4-Flash-SQ-64.strata", s[1]), ["https://models.example/hy4/Hy4-Flash-SQ-64.strata"])
        parts = hy4_family.urls_for("Hy4-Flash-SQ-64.strata", s[0])
        self.assertEqual(len(parts), 2)                                            # 62.03 GB in parts of 42 GiB
        self.assertTrue(parts[0].endswith("Hy4-Flash-SQ-64.strata.part-aa") and parts[1].endswith(".part-ab"))
        self.assertEqual(len(hy4_family.urls_for(hy4_family.ENGRAM_FILE, s[0])), 2)
        self.assertEqual(len(hy4_family.urls_for("Hy4-Flash-SQ-128.strata", s[0])), 3)
        listed = {"kind": "http", "base": "", "files": {"a": "https://x/a", "b": ["https://x/b1", "https://x/b2"]}}
        self.assertEqual(hy4_family.urls_for("a", listed), ["https://x/a"])
        self.assertEqual(hy4_family.urls_for("b", listed), ["https://x/b1", "https://x/b2"])
        self.assertEqual(hy4_family.sources(env={"STRATA_HY4_SOURCE": "ftp://nope"}, shipped=Path("nope.json")), [])


class Fetch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.table, self.data = tiny_files()
        self.a = mock.Mock(gguf_dir=None, models_dir=str(self.root / "models"))
        p = mock.patch.object(hy4_family, "FILES", self.table)
        p.start()
        self.addCleanup(p.stop)
        self.src = self.root / "src"
        self.src.mkdir()

    def source(self, **kw):
        return [{"kind": "http", "base": "", "files": kw}]

    def run_fetch(self, name, sources):
        folder = self.root / "models" / "hy4-flash"
        with mock.patch.object(hy4_family, "sources", return_value=sources), mock.patch.object(setup, "say"), mock.patch.object(setup, "ok"), \
                mock.patch.object(setup, "warn") as warn:
            got = setup.hy4_fetch(self.a, name, folder, False)
        return got, warn

    def test_one_file_from_a_source(self):
        name = "Hy4-Flash-SQ-64.strata"
        (self.src / name).write_bytes(self.data[name])
        got, _ = self.run_fetch(name, self.source(**{name: (self.src / name).as_posix()}))
        self.assertEqual(got.read_bytes(), self.data[name])

    def test_parts_are_joined_in_order_and_removed(self):
        name = hy4_family.ENGRAM_FILE
        blob = self.data[name]
        cuts = [0, 400, 800, len(blob)]
        urls = []
        for i in range(3):
            p = self.src / f"{name}.part-{i}"
            p.write_bytes(blob[cuts[i]:cuts[i + 1]])
            urls.append(p.as_posix())
        got, _ = self.run_fetch(name, self.source(**{name: urls}))
        self.assertEqual(got.read_bytes(), blob)
        self.assertEqual([p.name for p in got.parent.iterdir() if ".part" in p.name or p.name.endswith(".joining")], [])

    def test_a_wrong_file_is_refused(self):
        name = "Hy4-Flash-SQ-64.strata"
        bad = bytearray(self.data[name])
        bad[5] ^= 1
        (self.src / name).write_bytes(bytes(bad))
        with self.assertRaises(SystemExit):
            self.run_fetch(name, self.source(**{name: (self.src / name).as_posix()}))

    def test_the_next_source_gets_its_turn(self):
        name = "Hy4-Flash-SQ-64.strata"
        (self.src / name).write_bytes(self.data[name])
        sources = [{"kind": "http", "base": "", "files": {name: (self.src / "missing").as_posix()}},
                   {"kind": "http", "base": "", "files": {name: (self.src / name).as_posix()}}]
        got, warn = self.run_fetch(name, sources)
        self.assertEqual(got.read_bytes(), self.data[name])
        self.assertTrue(warn.called)

    def test_no_source_says_where_to_put_the_file(self):
        with self.assertRaises(SystemExit):
            self.run_fetch("Hy4-Flash-SQ-64.strata", [])

    def test_a_file_that_is_there_is_not_downloaded_again(self):
        name = "Hy4-Flash-SQ-64.strata"
        folder = self.root / "models" / "hy4-flash"
        folder.mkdir(parents=True)
        (folder / name).write_bytes(self.data[name])
        setup.mark(folder / name)
        got, _ = self.run_fetch(name, [])
        self.assertEqual(got, folder / name)

    def test_the_engram_file_without_a_hash_is_checked_by_size(self):
        name = hy4_family.ENGRAM_FILE
        self.table[name] = (len(self.data[name]), None)
        (self.src / name).write_bytes(self.data[name])
        got, warn = self.run_fetch(name, self.source(**{name: (self.src / name).as_posix()}))
        self.assertEqual(got.read_bytes(), self.data[name])
        self.assertTrue(any("size only" in str(c) for c in warn.call_args_list))


class Defaults0218(unittest.TestCase):
    """The owner's 0.2.18 defaults (2026-11-19): 128K context, the drafter on, images on (the Vision shard is in the default download set)."""

    def test_the_constants(self):
        self.assertEqual(hy4_family.DEFAULT_CONTEXT, 131072)
        self.assertEqual(hy4_family.DEFAULT_DRAFTER, "dfly2")
        self.assertTrue(hy4_family.DEFAULT_VISION)
        self.assertEqual(hy4_family.files_of("SQ-64"), ["Hy4-Flash-SQ-64.strata", hy4_family.ENGRAM_FILE, hy4_family.VISION_FILE])
        self.assertEqual(hy4_family.files_of("SQ-64", vision=False), ["Hy4-Flash-SQ-64.strata", hy4_family.ENGRAM_FILE])

    def install(self, vram=12.0, **over):
        """install_hy4_flash with the heavy steps stubbed, small stand-in files in the model folder: returns (the config setup wrote, the lines it said)."""
        import argparse
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        names = ("Hy4-Flash-SQ-64.strata", hy4_family.ENGRAM_FILE, hy4_family.VISION_FILE)
        table, data = tiny_files(names)
        folder = root / "models" / "hy4-flash"
        folder.mkdir(parents=True)
        for n in names:
            (folder / n).write_bytes(data[n])
            setup.mark(folder / n)
        a = argparse.Namespace(model="SQ-64", context=None, vision=None, rope_scaling=None, rope_scale=None, kv=None, resident_budget_gib=None, low_ram=None, gguf_dir=None,
                               models_dir=str(root / "models"), gpu=None, host=None, api_key=None, browser=None, no_start=True, yes=True, feed=None, data_dir=None)
        for k, v in over.items():
            setattr(a, k, v)
        written, lines = {}, []
        gpu = {"vram_gb": vram, "count": 1, "index": 0, "name": "RTX"}
        with mock.patch.object(hy4_family, "FILES", table), mock.patch.object(setup, "home", return_value=root), mock.patch.object(setup, "pip_install"), \
                mock.patch.object(setup, "runtime_packages", return_value=[]), mock.patch.object(setup, "step"), mock.patch.object(setup, "ok", lambda m: lines.append(str(m))), \
                mock.patch.object(setup, "say", lambda m="": lines.append(str(m))), mock.patch.object(setup, "warn", lambda m: lines.append("WARN " + str(m))), \
                mock.patch.object(setup, "engine_step", return_value=(None, root / "engine", {"component": "missing"}, [], gpu, "none")), \
                mock.patch.object(setup, "data_folder_for", return_value=(root / "data", None)), mock.patch.object(setup, "hy4_tokenizer_dir", return_value=root / "tok"), \
                mock.patch.object(setup, "write_setup_config", lambda path, cfg, *x, **k: written.update(cfg)), mock.patch.object(setup, "write_run_script", return_value=Path("run.cmd")), \
                mock.patch.object(setup, "explicit_args", return_value=[]), mock.patch.object(setup, "pip_cuda_libs", create=True):
            setup.install_hy4_flash(a, "hy4-flash", gpu, False, False, None, False, 8095, 64.0, None)
        return written, lines, root

    def test_the_default_install_is_128k_with_the_drafter_and_the_pictures(self):
        cfg, lines, root = self.install()
        args = cfg["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "131072")
        self.assertEqual(Path(args[args.index("--vision") + 1]).name, hy4_family.VISION_FILE)
        self.assertEqual(cfg["drafting"], "dfly2")
        self.assertEqual(cfg["arch"], "hy4flash")
        self.assertEqual(cfg["exe"], "component:strata-hy4")
        self.assertEqual(args[args.index("--vram-experts-gb") + 1], "5.08")               # 12 GiB: the KV state (5.5 GB at 128K) is pinned host memory, so the card keeps the author's 5.08 GB of experts
        self.assertEqual(args[args.index("--ram-experts-gb") + 1], "50.96")
        self.assertTrue(any(l.startswith("images: on") for l in lines), lines)

    def test_vision_no_leaves_the_shard_and_the_option_out(self):
        cfg, lines, _ = self.install(vision="no")
        self.assertNotIn("--vision", cfg["args"])
        self.assertTrue(any(l.startswith("images: off") for l in lines), lines)

    def test_128k_is_the_default_on_every_card_with_no_smaller_context_and_no_warning(self):
        # owner, 2026-11-19: "Keep 128K as default, people can change whenever they want themselves": an 8 GiB card gets 128K like a 24 GiB one
        for vram in (8.0, 12.0, 24.0):
            cfg, lines, _ = self.install(vram=vram)
            self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "131072", vram)
            self.assertFalse(any("does not fit this card" in l or "choose a smaller context" in l for l in lines), (vram, lines))
        cfg, lines, _ = self.install(vram=8.0, context=32768)                             # what the person chooses is what runs
        self.assertEqual(cfg["args"][cfg["args"].index("--max-context") + 1], "32768")

    def test_the_planned_placement_on_an_8_gib_card_is_written_at_128k(self):
        cfg, _lines, _ = self.install(vram=8.0)                                          # the engine lanes deal with a cache that does not fit the card
        args = cfg["args"]
        self.assertEqual(args[args.index("--vram-experts-gb") + 1], "0.79")
        self.assertEqual(args[args.index("--ram-experts-gb") + 1], "52.29")        # 8 GiB + 64 GiB, SQ-64: 3.0 GB of its experts are read from the SSD (the planner recommends SQ-64 there: there is no SQ-32)


class Catalog(unittest.TestCase):
    def setUp(self):
        self.show = mock.patch.dict(os.environ, {"STRATA_HY4_SHOW": "1"})
        self.show.start()
        self.addCleanup(self.show.stop)
        self.s = M.Strata(HERE.parent)

    def rows(self, hw):
        fam = next(f for f in self.s.model_table(hw) if f["family"] == "hy4-flash")
        return {x["model"]: x for x in fam["sizes"]}

    def test_listed_only_when_offered(self):
        os.environ.pop("STRATA_HY4_SHOW")
        with mock.patch.object(hy4_family, "sources", return_value=[]), mock.patch.object(hy4_family, "local_files", return_value={}):
            self.assertNotIn("hy4-flash", [f["family"] for f in self.s.model_table({"ram_gb": 64.0, "gpus": []})])

    def test_rows_and_fit_words(self):
        by = self.rows({"ram_gb": 64.0, "gpus": [{"index": 0, "name": "x", "vram_gb": 12.0, "usable": True, "vendor": "nvidia"}]})
        self.assertEqual(list(by), ["SQ-64", "SQ-96", "SQ-128"])                  # SQ-32 is not offered (owner, 2026-11-20)
        self.assertEqual(by["SQ-64"]["id"], "hy4-flash-sq-64")                      # the tag the install writes its config under
        self.assertEqual(by["SQ-64"]["download_gb"], round(62.03 + 67.58 + 0.934, 1))                      # the build, the Engram shard and the Vision shard (images are on by default)
        self.assertTrue(by["SQ-64"]["images"])
        self.assertEqual({k: by[k]["on_this_pc"] for k in by}, {"SQ-64": "fits", "SQ-96": "may run slower", "SQ-128": "may run slower"})
        self.assertEqual([k for k in by if by[k]["hy4"]["recommended"]], ["SQ-64"])
        self.assertEqual(by["SQ-64"]["hy4"]["placement"], {"vram_gb": 5.08, "ram_gb": 50.96, "ssd_gb": 0.0})            # at 128K the engine keeps the KV state in pinned host memory on a 12 GiB card (HY4-CUDA)
        self.assertIsNone(by["SQ-64"]["gpu_options"])

    def test_unified_memory(self):
        gpu = {"index": 0, "name": "Radeon 8060S", "vram_gb": 115.8, "dedicated_gb": 0.0, "usable": True, "uma": True, "vendor": "amd"}
        by = self.rows({"ram_gb": 128.0, "gpus": [gpu]})
        self.assertEqual([k for k in by if by[k]["hy4"]["recommended"]], ["SQ-96"])
        self.assertEqual(by["SQ-128"]["hy4"]["mode"], "cross_tier_ssd")          # 8 GB of 113 from the SSD with the 128K cache and the image reader budgeted
        self.assertEqual(by["SQ-128"]["on_this_pc"], "may run slower")

    def test_no_ram_figure_gives_no_verdict(self):
        by = self.rows({"gpus": []})
        self.assertIsNone(by["SQ-64"]["on_this_pc"])

    def test_the_install_tool_knows_the_sizes_and_the_expected_files(self):
        self.assertIn("SQ-96", self.s.own_size_names())
        S = self.s.setup_module()
        files = M.expected_files(S, "hy4-flash", "SQ-64", Path("/m"), False)
        self.assertEqual([Path(f["path"]).name for f in files], ["Hy4-Flash-SQ-64.strata", hy4_family.ENGRAM_FILE])
        self.assertEqual([f["bytes"] for f in files], [62_030_467_072, 67_584_905_216])
        self.assertTrue(all(Path(f["path"]).parent.name == "hy4-flash" for f in files))
        files = M.expected_files(S, "hy4-flash", "SQ-64", Path("/m"), True)                       # images on (the default): the Vision shard too
        self.assertEqual([Path(f["path"]).name for f in files], ["Hy4-Flash-SQ-64.strata", hy4_family.ENGRAM_FILE, hy4_family.VISION_FILE])
        self.assertEqual(files[2]["bytes"], 934_092_800)


if __name__ == "__main__":
    unittest.main()
