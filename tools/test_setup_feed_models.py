"""setup.py and the model tables with models from the signed feed (strata_control/feed_models.py).

    python -m pytest tools/test_setup_feed_models.py

A feed model goes into the same tables a built-in model is in, so setup, the Terminal and the app list it after the built-in ones and it is installed, pinned,
checked and (when wrong) kept as `.bad` by the same code. Downloads, the engine and processes are mocked; a folder signed with a throwaway key is the feed.
With no `models` member nothing changes: the tables are compared before and after.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from strata_control import feed as feedmod  # noqa: E402
from strata_control import feed_models, signing  # noqa: E402

REV = "0123456789abcdef0123456789abcdef01234567"
SHIPPED_POLICY = feed_models.UNAVAILABLE_POLICY
CONTENT = {"a-00001-of-00002.gguf": b"first part of a model", "a-00002-of-00002.gguf": b"second part", "mmproj-q4.gguf": b"the picture reader",
           "Q4-model.gguf": b"one whole model file"}


def pin(name):
    return len(CONTENT[name]), hashlib.sha256(CONTENT[name]).hexdigest()


def file_entry(name, folder=""):
    size, sha = pin(name)
    return {"name": name, "size": size, "sha256": sha, **({"path": f"{folder}/{name}"} if folder else {})}


def flash_next(**kw):
    m = {"id": "qwen4-test", "title": "Qwen4 Test", "by": "Qwen", "about": "a test model", "licence": "Apache 2.0",
         "licence_url": "https://huggingface.co/test/qwen4-test", "family": "qwen4exp", "repo": "test/Qwen4-Test-GGUF", "revision": REV,
         "pictures": True, "drafter": "mtp", "context": 131072, "min_engine": "0.2.12", "mmproj": file_entry("mmproj-q4.gguf"),
         "quants": [{"name": "Q4_K_M", "about": "4-bit", "ram_gb": 48, "arena_gb": 30.5,
                     "files": [file_entry("a-00001-of-00002.gguf", "Q4_K_M"), file_entry("a-00002-of-00002.gguf", "Q4_K_M")]}]}
    m.update(kw)
    return m


def own_path(**kw):
    m = flash_next(id="qwen35-test", title="Qwen35 Test", family="qwen35moe", repo="test/Qwen35-Test-GGUF", kv_bytes_per_token=90112,
                   quants=[{"name": "UD-Q4", "about": "4-bit", "ram_gb": 30, "files": [file_entry("Q4-model.gguf")]}], mmproj=file_entry("mmproj-q4.gguf"))
    m.update(kw)
    return m


# setup.main() on a Mac runs the Metal installer (metal/setup_mac.py): the command-line tests below are about the PC flow, so on macOS
# they run it as on Linux (on Linux and Windows this is the platform they already have)
LINUX_PC = "linux" if sys.platform == "darwin" else sys.platform


def fresh_setup(name="setup_under_test"):
    """setup.py loaded again under its own name, so a test's registrations never reach another test."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "setup.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tables(mod) -> str:
    return json.dumps({k: getattr(mod, k) for k in ("MODELS", "FAMILIES", "OWN_PATH_FAMILIES", "MULTI_FILE_FAMILIES", "REPO_PINS", "HF_REVISIONS")},
                      sort_keys=True, default=str) + repr(sorted(mod.REV_PINS)) + repr(getattr(mod, "FEED_MODELS", None))


def parsed(*models) -> tuple:
    doc = {"schema_version": 1, "feed": "strata-updates", "channel": "stable", "sequence": 1, "generated_at": "2026-11-06T00:00:00Z", "expires_at": None,
           "models": {"packages": list(models)}}
    f = feedmod.parse(json.dumps(doc).encode(), source="x", key_id="k")
    assert not f.models_dropped, f.models_dropped
    return f.models


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = signing.generate_key()
        keys = self.root / "keys.json"
        keys.write_text(json.dumps({"keys": [signing.public_part(self.key)]}), encoding="utf-8")
        patch = mock.patch.dict(os.environ, {"STRATA_COMPONENTS_DIR": str(self.root / "components"), "STRATA_UPDATE_TRUSTED_KEYS": str(keys)})
        patch.start()
        self.addCleanup(patch.stop)
        os.environ.pop("STRATA_FEED_MODELS", None)
        policy = mock.patch.object(feed_models, "UNAVAILABLE_POLICY", "not_offered")     # the tests below were written for "not_offered"; OwnerRule sets its own
        policy.start()
        self.addCleanup(policy.stop)
        from strata_control import mac_fit             # the Mac tables are module globals: each test gets them back as they were
        saved = (dict(mac_fit.MAC_MODELS), dict(mac_fit.MAC_ABOUT), list(mac_fit.FEED_FAMILIES))

        def restore_mac():
            mac_fit.MAC_MODELS.clear()
            mac_fit.MAC_MODELS.update(saved[0])
            mac_fit.MAC_ABOUT.clear()
            mac_fit.MAC_ABOUT.update(saved[1])
            mac_fit.FEED_FAMILIES[:] = saved[2]
        self.addCleanup(restore_mac)
        self.setup = fresh_setup()

    def signed_feed(self, name="feed", sequence=1, models=None) -> str:
        out = self.root / name
        out.mkdir(exist_ok=True)
        members = {"schema_version": 1, "feed": "strata-updates", "channel": "stable", "sequence": sequence, "generated_at": "2026-11-06T00:00:00Z",
                   "expires_at": None}
        if models is not None:
            members["models"] = {"packages": models}
        payload = (json.dumps(members, indent=1, sort_keys=True) + "\n").encode()
        (out / feedmod.FEED_NAME).write_bytes(payload)
        (out / (feedmod.FEED_NAME + ".sig")).write_text(json.dumps(signing.sign_bytes(self.key, payload)), encoding="utf-8")
        return str(out)


class NothingChangesWithoutModels(Base):
    def test_the_tables_are_the_same_without_a_models_member(self):
        before = tables(self.setup)
        reference = tables(fresh_setup("setup_reference"))
        self.assertEqual(before, reference)
        self.setup.feed_models_setup(fetch=False)                                   # no cache
        self.assertEqual(tables(self.setup), before)
        self.setup.feed_models_setup(self.signed_feed("plain"))                     # a signed feed with no models member
        self.assertEqual(tables(self.setup), before)
        self.assertIsNone(self.setup.FEED_MODELS)
        self.assertEqual(self.setup.FEED_UNAVAILABLE, {})
        self.assertFalse((self.root / "components").exists())                       # nothing written either

    def test_apply_of_an_empty_list_touches_nothing(self):
        before = tables(self.setup)
        result = feed_models.apply(self.setup, (), {"qwen4exp"})
        self.assertEqual((result.registered, result.unavailable, result.skipped), ([], {}, []))
        self.assertEqual(tables(self.setup), before)

    def test_the_menu_helpers_see_the_same_families(self):
        self.setup.feed_models_setup(self.signed_feed("plain"))
        self.assertEqual(self.setup.menu_families(False), list(fresh_setup("setup_reference").FAMILIES))


class FlashNextModel(Base):
    def setUp(self):
        super().setUp()
        self.result = feed_models.apply(self.setup, parsed(flash_next()), {"qwen4exp", "qwen35moe"})
        self.fam = self.setup.FAMILIES["qwen4-test"]

    def test_it_is_listed_after_the_built_in_ones(self):
        self.assertEqual(self.result.registered, ["qwen4-test"])
        built_in = list(fresh_setup("setup_reference").FAMILIES)
        self.assertEqual(list(self.setup.FAMILIES), built_in + ["qwen4-test"])
        self.assertEqual(self.setup.menu_families(False)[-1], "qwen4-test")
        self.assertEqual(self.fam["title"], "Qwen4 Test")
        self.assertIn("Apache 2.0: https://huggingface.co/test/qwen4-test", self.fam["license"])

    def test_files_folders_and_shards_come_from_the_feed(self):
        m = "Q4_K_M"
        self.assertEqual(self.setup.model_shards(self.fam, m), 2)
        self.assertEqual([self.setup.model_file(self.fam, m, i) for i in (1, 2)], ["a-00001-of-00002.gguf", "a-00002-of-00002.gguf"])
        base = f"{self.setup.hf_endpoint()}/test/Qwen4-Test-GGUF/resolve/{REV}/Q4_K_M/"
        self.assertEqual(self.setup.model_hf(self.fam, m), base)
        self.assertEqual(self.fam["mmproj_hf"], f"{self.setup.hf_endpoint()}/test/Qwen4-Test-GGUF/resolve/{REV}/")
        d = self.setup.MODELS[m]
        self.assertEqual((d["download_gb"], d["ram_gb"], d["arena_gb"], d["engine"], d["families"]), (0.0, 48.0, 30.5, (0, 2, 12), ("qwen4-test",)))
        self.assertTrue(d["vision"])

    def test_every_file_is_pinned_where_downloads_look_it_up(self):
        base = self.setup.model_hf(self.fam, "Q4_K_M")
        for name in ("a-00001-of-00002.gguf", "a-00002-of-00002.gguf"):
            self.assertEqual(self.setup.file_pin(base + name), pin(name))
        self.assertEqual(self.setup.file_pin(self.fam["mmproj_hf"] + "mmproj-q4.gguf"), pin("mmproj-q4.gguf"))
        self.assertEqual(self.fam["mmproj_sha256"], pin("mmproj-q4.gguf"))
        self.assertIsNone(self.setup.file_pin(base + "other.gguf"))

    def test_its_config_is_found_again_by_choices_from_config(self):
        cfg = self.root / "strata-qwen4-test-q4_k_m.json"
        cfg.write_text(json.dumps({"args": ["--max-context", "65536"], "arch": "qwen4exp"}), encoding="utf-8")
        got = self.setup.choices_from_config(cfg)
        self.assertEqual((got["family"], got["model"], got["context"]), ("qwen4-test", "Q4_K_M", 65536))

    def test_a_size_name_that_a_built_in_model_has_gets_the_models_id(self):
        feed_models.apply(self.setup, parsed(flash_next(id="qwen4-two", quants=[{"name": "IQ3_S", "about": "x", "ram_gb": 40, "files": [file_entry("Q4-model.gguf")]}])), None)
        self.assertIn("IQ3_S", self.setup.MODELS)                                   # the built-in one is untouched
        self.assertEqual(self.setup.MODELS["IQ3_S"], fresh_setup("setup_reference").MODELS["IQ3_S"])
        self.assertEqual(self.setup.MODELS["IQ3_S-qwen4-two"]["quant"], "IQ3_S")

    def test_a_built_in_id_or_tag_is_never_replaced(self):
        before = copy.deepcopy(self.setup.FAMILIES["swift"])
        for clash in ("swift", "qwen", "unsloth", "qwen36", "deepseek41", "coder-x", "unsloth-2"):
            result = feed_models.apply(self.setup, parsed(flash_next(id=clash)), None)
            self.assertEqual(result.registered, [], clash)
            self.assertEqual(len(result.skipped), 1, clash)
        self.assertEqual(self.setup.FAMILIES["swift"], before)

    def test_a_model_without_a_picture_reader_has_no_mmproj_and_install_does_not_look_one_up(self):
        # found by the Q4-P8 rehearsal on a real install: setup's step 5 read fam['mmproj'] for a feed model that lists none (KeyError after the checks of the files)
        entry = flash_next(id="no-reader")
        entry.pop("mmproj", None)
        entry["pictures"] = False
        result = feed_models.apply(self.setup, parsed(entry), None)
        self.assertEqual(result.registered, ["no-reader"])
        fam = self.setup.FAMILIES["no-reader"]
        self.assertNotIn("mmproj", fam)
        self.assertIs(fam["vision"], False)
        src = (Path(self.setup.__file__)).read_text(encoding="utf-8")
        self.assertNotIn("find_in(roots, f\"models/{fam['mmproj']}\") or mmproj\n    if vision != \"none\":", src)

    def test_the_family_carries_the_feeds_drafter_so_setup_can_leave_the_draft_layer_out(self):
        # found by the Q4-P8 rehearsal: setup fetched the Flash-Next MTP draft layer (5 GB) for a feed model whose entry says drafter none
        none = feed_models.apply(self.setup, parsed(flash_next(id="no-drafter", drafter="none")), None)
        self.assertEqual(self.setup.FAMILIES["no-drafter"]["drafter"], "none")
        src = Path(self.setup.__file__).read_text(encoding="utf-8")
        self.assertIn('fam.get("feed") and fam.get("drafter") == "none" and a.mtp is None', src)

    def test_a_feed_flash_next_size_is_read_in_place_with_a_ram_budget_like_the_built_in_unsloth_sizes(self):
        # found by the Q4-P8 rehearsal on Strix: without "budget" setup chose the low-RAM mode and began writing a 77 GB second copy of the experts
        result = feed_models.apply(self.setup, parsed(flash_next(id="budgeted")), None)
        for key in [k for k, v in self.setup.MODELS.items() if "budgeted" in v.get("families", ())]:
            self.assertIs(self.setup.MODELS[key]["budget"], True)
            self.assertFalse(self.setup.low_ram_wanted(key, 64.0, "auto"))

    def test_parts_that_are_not_numbered_are_left_out_with_a_line(self):
        odd = flash_next(id="odd-parts")
        odd["quants"][0]["files"][1] = file_entry("Q4-model.gguf", "Q4_K_M")
        result = feed_models.apply(self.setup, parsed(odd), None)
        self.assertEqual((result.registered, len(result.skipped)), ([], 1))
        self.assertNotIn("odd-parts", self.setup.FAMILIES)


class OwnPathModel(Base):
    def setUp(self):
        super().setUp()
        self.result = feed_models.apply(self.setup, parsed(own_path()), {"qwen4exp", "qwen35moe"})
        self.fam = self.setup.OWN_PATH_FAMILIES["qwen35-test"]

    def test_it_joins_the_own_path_families(self):
        self.assertEqual(self.result.registered, ["qwen35-test"])
        self.assertIs(self.setup.PREVIEW_FAMILIES, self.setup.OWN_PATH_FAMILIES)
        self.assertEqual(list(self.setup.OWN_PATH_FAMILIES), ["qwen36", "qwen35-test"])
        self.assertEqual((self.fam["arch"], self.fam["mtp"], self.fam["kv_bytes_per_token"], list(self.fam["models"])), ("qwen35moe", 2, 90112, ["UD-Q4"]))
        self.assertEqual(self.fam["file"].format(q="UD-Q4"), "Q4-model.gguf")
        url = self.fam["hf"] + "Q4-model.gguf"
        self.assertEqual(self.setup.file_pin(url), pin("Q4-model.gguf"))
        self.assertEqual(self.setup.own_path_size(self.fam, 64.0), "UD-Q4")

    def test_the_built_in_qwen36_is_unchanged(self):
        self.assertEqual(self.setup.OWN_PATH_FAMILIES["qwen36"], fresh_setup("setup_reference").OWN_PATH_FAMILIES["qwen36"])

    def test_install_uses_the_feeds_pin_and_the_same_steps(self):
        calls, cfg = [], {}
        eng = self.root / "engine"
        eng.mkdir()
        (eng / self.setup.VEXE).write_text("x")

        def engine_step(a, gpu, hip, amd, cuda_tk, vision):
            return self.root / "llama", eng, {"version": "0.2.12"}, ["/libs"], gpu, vision

        def write_cfg(path, c, source=None, explicit=None):
            cfg.update(c)
            calls.append(("config", path.name))

        s = self.setup
        patches = [mock.patch.object(s, "download", lambda url, dst, what=None: calls.append(("get", url, dst.name))),
                   mock.patch.object(s, "verify_sha256", lambda p, size, sha, own=False: calls.append(("sha", p.name, size, sha))),
                   mock.patch.object(s, "free_gb", lambda p: 500.0), mock.patch.object(s, "pip_install", lambda *a, **k: None),
                   mock.patch.object(s, "engine_step", engine_step), mock.patch.object(s, "engine_families", lambda e: {"qwen4exp", "qwen35moe"}),
                   mock.patch.object(s, "run", lambda cmd, **k: calls.append(("run", cmd))),
                   mock.patch.object(s, "data_folder", lambda d, *a, **k: (self.root / "data", [])),
                   mock.patch.object(s, "write_setup_config", write_cfg),
                   mock.patch.object(s, "write_run_script", lambda tag, p, port, o=True: self.root / f"run-{tag}.sh"),
                   mock.patch.object(s, "recommend_pool_workers", lambda a: a), mock.patch.object(s, "ROOT", self.root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        a = argparse.Namespace(family="qwen35-test", model="UD-Q4", check=False, yes=True, vision="cpu", models_dir=str(self.root / "models"), gguf_dir=None,
                               context=None, rope_scaling=None, rope_scale=None, kv=None, resident_budget_gib=None, low_ram="auto", vram_reserve_mib=None,
                               host=None, api_key=None, browser=None, no_start=True, data_dir=None, gpu=None)
        gpu = {"index": 0, "count": 1, "vram_gb": 16.0, "name": "Test GPU"}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = s.install_own_path(a, "qwen35-test", gpu, False, [], 13, [], 8080, 64.0, None)
        self.assertEqual(rc, 0, out.getvalue())
        base = f"{s.hf_endpoint()}/test/Qwen35-Test-GGUF/resolve/{REV}/"
        self.assertIn(("get", base + "Q4-model.gguf", "Q4-model.gguf"), calls)
        self.assertIn(("get", base + "mmproj-q4.gguf", "mmproj-q4.gguf"), calls)
        self.assertIn(("sha", "Q4-model.gguf", *pin("Q4-model.gguf")), calls)         # the feed's size and SHA-256 reach the check
        self.assertIn(("sha", "mmproj-q4.gguf", *pin("mmproj-q4.gguf")), calls)
        self.assertEqual((cfg["arch"], cfg["model_name"]), ("qwen35moe", "qwen35-test-ud-q4"))
        self.assertIn(("config", "strata-qwen35-test-ud-q4.json"), calls)
        self.assertEqual(cfg["args"][cfg["args"].index("--native") + 1], str(self.root / "models" / "qwen35-test-UD-Q4" / "Q4-model.gguf"))
        self.assertEqual(cfg["args"][cfg["args"].index("--mtp") + 1], "2")


class PinVerifyAndBad(Base):
    """The same functions a built-in model's download goes through, on a feed model's tiny files."""

    def setUp(self):
        super().setUp()
        feed_models.apply(self.setup, parsed(own_path()), None)
        self.fam = self.setup.OWN_PATH_FAMILIES["qwen35-test"]
        self.dst = self.root / "models" / "Q4-model.gguf"
        self.dst.parent.mkdir()
        self.url = self.fam["hf"] + "Q4-model.gguf"

    def test_a_wrong_file_is_kept_as_bad_and_stops_setup(self):
        self.dst.write_bytes(b"x" * len(CONTENT["Q4-model.gguf"]))                  # the pinned size, other bytes
        with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
            self.setup.verify_sha256(self.dst, *self.fam["sha256"]["Q4-model.gguf"])
        self.assertTrue(self.dst.with_name("Q4-model.gguf.bad").is_file())
        self.assertFalse(self.dst.exists())
        self.assertIn("does not match the pinned SHA-256", out.getvalue())

    def test_a_right_file_passes_and_gets_its_finish_mark(self):
        self.dst.write_bytes(CONTENT["Q4-model.gguf"])
        with contextlib.redirect_stdout(io.StringIO()):
            self.setup.verify_sha256(self.dst, *self.fam["sha256"]["Q4-model.gguf"])
        self.assertTrue(self.setup.done(self.dst))

    def test_download_takes_back_a_bad_file_that_matches_the_pin_without_the_network(self):
        bad = self.dst.with_name("Q4-model.gguf.bad")
        bad.write_bytes(CONTENT["Q4-model.gguf"])
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(self.setup, "_urlopen", side_effect=AssertionError("network")):
            self.setup.download(self.url, self.dst)
        self.assertEqual(self.dst.read_bytes(), CONTENT["Q4-model.gguf"])
        self.assertFalse(bad.exists())

    def test_download_does_not_take_back_a_bad_file_of_the_wrong_size_or_bytes(self):
        bad = self.dst.with_name("Q4-model.gguf.bad")
        bad.write_bytes(b"x" * len(CONTENT["Q4-model.gguf"]))                      # the pinned size, other bytes
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(self.setup, "_urlopen", side_effect=RuntimeError("network")), \
                mock.patch.object(self.setup, "acquisition") as acquisition:
            acquisition.require.side_effect = RuntimeError("stopped before the network")
            with self.assertRaises(RuntimeError):
                self.setup.download(self.url, self.dst)
        self.assertFalse(self.dst.exists())


class RefusalsAndOffline(Base):
    def test_a_family_the_engine_does_not_run_is_listed_as_not_available_and_never_added(self):
        result = feed_models.apply(self.setup, parsed(own_path(), flash_next(id="later-one", family="qwen5new")), {"qwen4exp"})
        self.assertEqual(result.registered, [])
        self.assertEqual(sorted(result.unavailable), ["later-one", "qwen35-test"])
        self.assertNotIn("qwen35-test", self.setup.OWN_PATH_FAMILIES)
        self.assertNotIn("later-one", self.setup.FAMILIES)
        self.assertEqual(self.setup.FEED_UNAVAILABLE["qwen35-test"]["reason"], feed_models.UNAVAILABLE_ENGINE)
        self.assertEqual(self.setup.FEED_UNAVAILABLE["later-one"]["reason"], feed_models.UNAVAILABLE_UNKNOWN)
        self.assertNotIn("test/Qwen35-Test-GGUF", self.setup.HF_REVISIONS)               # no pin was added for a model nobody can install
        self.assertFalse([k for k in self.setup.REV_PINS if k[0] == "test/Qwen35-Test-GGUF"])

    def test_without_an_engine_to_ask_an_installable_family_is_offered_and_setup_checks_at_install(self):
        result = feed_models.apply(self.setup, parsed(own_path()), None)
        self.assertEqual(result.registered, ["qwen35-test"])

    def test_naming_an_unavailable_model_on_the_command_line_says_why_and_stops(self):
        feed = self.signed_feed("f", 1, [own_path()])
        argv = ["setup.py", "--family", "qwen35-test", "--feed", feed, "--yes"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(self.setup, "feed_engine_families", lambda: {"qwen4exp"}), \
                mock.patch.object(self.setup.sys, "platform", LINUX_PC), \
                contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as stop:
            self.setup.main()
        self.assertNotEqual(stop.exception.code, 0)
        self.assertIn(feed_models.UNAVAILABLE_ENGINE, out.getvalue())

    def test_the_command_line_accepts_a_feed_model_and_its_size(self):
        feed = self.signed_feed("f", 1, [own_path()])
        argv = ["setup.py", "--family", "qwen35-test", "--model", "UD-Q4", "--feed", feed, "--check"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(self.setup, "feed_engine_families", lambda: {"qwen4exp", "qwen35moe"}), \
                mock.patch.object(self.setup.sys, "platform", LINUX_PC), \
                mock.patch.object(self.setup, "gather_hardware", side_effect=SystemExit(77), create=True), contextlib.redirect_stdout(io.StringIO()):
            try:
                self.setup.main()
            except SystemExit:
                pass
            except Exception:                                                            # noqa: BLE001 - the hardware step is not under test
                pass
        self.assertIn("qwen35-test", self.setup.OWN_PATH_FAMILIES)                       # argparse got past --family / --model

    def test_offline_the_last_verified_list_is_what_the_tables_get(self):
        feed = self.signed_feed("f", 7, [flash_next()])
        self.setup.feed_models_setup(feed)
        self.assertIn("qwen4-test", self.setup.FAMILIES)
        other = fresh_setup("setup_offline")
        other.feed_models_setup(str(self.root / "unreachable"))
        self.assertEqual(list(other.FAMILIES)[-1], "qwen4-test")
        none = fresh_setup("setup_none")
        none.feed_models_setup(fetch=False)
        self.assertIn("qwen4-test", none.FAMILIES)                                       # the Terminal and app read the cache only


class TerminalAndApp(Base):
    """strata_mcp is the app's and the Terminal's one catalog."""

    def table(self, ram=64.0):
        import strata_mcp as M
        s = M.Strata(ROOT)
        return s, s.model_table({"ram_gb": ram, "gpus": [{"index": 0, "name": "x", "vram_gb": 12.0, "usable": True}]})

    def test_without_a_list_the_table_is_the_same(self):
        s, rows = self.table()
        self.assertEqual([r["family"] for r in rows], [f for f in fresh_setup("setup_reference").FAMILIES] + ["qwen36", "deepseek41"])
        self.assertTrue(all("available" not in r for r in rows))

    def test_a_cached_list_adds_its_sizes_after_the_built_in_ones_and_hides_what_cannot_run(self):
        feed_models.load(self.signed_feed("f", 7, [flash_next(), flash_next(id="later-one", family="qwen5new")]), trusted={
            self.key["key_id"]: base64.b64decode(self.key["public"])})
        s, rows = self.table()
        by = {r["family"]: r for r in rows}
        self.assertEqual([r["family"] for r in rows][-2:], ["qwen4-test", "later-one"])
        row = by["qwen4-test"]
        self.assertEqual([x["model"] for x in row["sizes"]], ["Q4_K_M"])
        size = row["sizes"][0]
        self.assertEqual((size["id"], size["images"], size["engine_min"], size["ram_needed_gb"]), ("qwen4-test-q4_k_m", True, "0.2.12", 48.0))
        self.assertEqual(by["later-one"]["sizes"], [])                                   # the app hides a family without sizes
        self.assertFalse(by["later-one"]["available"])
        self.assertEqual(by["later-one"]["unavailable_reason"], feed_models.UNAVAILABLE_UNKNOWN)
        import strata_mcp as M
        plan, args = M.Tools(s).install_plan("qwen4-test", "Q4_K_M", 65536, "no", None, None, None, None, "auto", None, None)
        self.assertEqual(args[args.index("--family") + 1:args.index("--family") + 4:2], ["qwen4-test", "Q4_K_M"])        # the install runs setup with the feed model's names
        self.assertEqual(plan["download_gb"], 0.0)


class OwnerRule(Base):
    """feed_models.UNAVAILABLE_POLICY: one line decides whether a model the engine does not run is only said ("not_offered") or warned about and allowed ("warn_allow")."""

    def test_warn_allow_registers_a_known_family_and_keeps_the_sentence(self):
        with mock.patch.object(feed_models, "UNAVAILABLE_POLICY", "warn_allow"):
            result = feed_models.apply(self.setup, parsed(own_path(), flash_next(id="later-one", family="qwen5new")), {"qwen4exp"})
        self.assertEqual(result.registered, ["qwen35-test"])
        self.assertIn("qwen35-test", self.setup.OWN_PATH_FAMILIES)
        self.assertTrue(self.setup.FEED_UNAVAILABLE["qwen35-test"]["allowed"])
        self.assertNotIn("later-one", self.setup.FAMILIES)                              # a family this Control does not know is never installable
        self.assertFalse(self.setup.FEED_UNAVAILABLE["later-one"]["allowed"])

    def test_warn_allow_install_warns_and_goes_on(self):
        with mock.patch.object(feed_models, "UNAVAILABLE_POLICY", "warn_allow"):
            feed_models.apply(self.setup, parsed(own_path()), {"qwen4exp"})
        s = self.setup
        eng = self.root / "engine"
        eng.mkdir()
        calls = []
        patches = [mock.patch.object(s, "download", lambda url, dst, what=None: calls.append(("get", dst.name))),
                   mock.patch.object(s, "verify_sha256", lambda p, size, sha, own=False: None), mock.patch.object(s, "free_gb", lambda p: 500.0),
                   mock.patch.object(s, "pip_install", lambda *a, **k: None),
                   mock.patch.object(s, "engine_step", lambda a, gpu, hip, amd, tk, v: (self.root / "llama", eng, {"version": "0.2.12"}, [], gpu, "none")),
                   mock.patch.object(s, "engine_families", lambda e: {"qwen4exp"}), mock.patch.object(s, "run", lambda cmd, **k: None),
                   mock.patch.object(s, "data_folder", lambda d, *a, **k: (self.root / "data", [])),
                   mock.patch.object(s, "write_setup_config", lambda *a, **k: calls.append(("config",))),
                   mock.patch.object(s, "write_run_script", lambda tag, p, port, o=True: self.root / "run.sh"),
                   mock.patch.object(s, "recommend_pool_workers", lambda a: a), mock.patch.object(s, "ROOT", self.root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        a = argparse.Namespace(family="qwen35-test", model="UD-Q4", check=False, yes=True, vision="no", models_dir=str(self.root / "models"), gguf_dir=None,
                               context=None, rope_scaling=None, rope_scale=None, kv=None, resident_budget_gib=None, low_ram="auto", vram_reserve_mib=None,
                               host=None, api_key=None, browser=None, no_start=True, data_dir=None, gpu=None)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = s.install_own_path(a, "qwen35-test", {"index": 0, "count": 1, "vram_gb": 16.0, "name": "T"}, False, [], 13, [], 8080, 64.0, None)
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn(feed_models.UNAVAILABLE_ENGINE, out.getvalue())
        self.assertIn(("get", "Q4-model.gguf"), calls)

    def test_the_shipped_policy_and_words(self):
        self.assertEqual(SHIPPED_POLICY, "warn_allow")                                  # the owner's standing rule: warn plainly, still allow, never refuse
        self.assertEqual(feed_models.UNAVAILABLE_ENGINE, "needs a newer Strata engine: update Strata to use it")
        self.assertEqual(feed_models.UNAVAILABLE_UNKNOWN, "this Strata version doesn't know this model yet: update Strata to use it")
        for family in ("qwen35moe", "qwen5new"):
            self.assertNotIn(family, feed_models.unavailable_reason(family))


class MacTables(Base):
    """The Mac (Metal) tables take the feed's models too: strata_control/mac_fit.py, metal/setup_mac.py and the app's Mac table."""

    def setUp(self):
        super().setUp()
        from strata_control import mac_fit
        self.mf = mac_fit
        saved = (dict(mac_fit.MAC_MODELS), dict(mac_fit.MAC_ABOUT), list(mac_fit.FEED_FAMILIES))

        def restore():
            mac_fit.MAC_MODELS.clear()
            mac_fit.MAC_MODELS.update(saved[0])
            mac_fit.MAC_ABOUT.clear()
            mac_fit.MAC_ABOUT.update(saved[1])
            mac_fit.FEED_FAMILIES[:] = saved[2]
        self.addCleanup(restore)

    def test_nothing_changes_without_feed_models(self):
        self.assertEqual(self.mf.all_families(), self.mf.APP_FAMILIES)
        feed_models.apply(self.setup, (), None)
        self.assertEqual(self.mf.all_families(), ("qwen", "unsloth", "gyro"))

    def test_a_feed_model_of_one_or_two_files_is_a_mac_size(self):
        feed_models.apply(self.setup, parsed(flash_next()), None)
        self.assertEqual(self.mf.all_families(), ("qwen", "unsloth", "gyro", "qwen4-test"))
        self.assertEqual(self.mf.app_sizes("qwen4-test"), ["Q4_K_M"])
        d = self.mf.MAC_MODELS["Q4_K_M"]
        self.assertEqual((d["family"], d["shards"], d["tested"]), ("qwen4-test", 2, False))
        self.assertIn("has not been measured on a Mac", " ".join(self.mf.model_note("Q4_K_M", 64.0)))
        self.assertEqual(self.mf.repacked_gb("Q4_K_M"), 0.0)

    def test_a_model_of_more_than_two_files_is_not_offered_on_a_mac(self):
        big = flash_next(id="big-one")
        big["quants"][0]["files"] = [{**file_entry("a-00001-of-00002.gguf"), "name": f"b-0000{i}-of-00003.gguf", "path": f"Q4_K_M/b-0000{i}-of-00003.gguf"} for i in (1, 2, 3)]
        result = feed_models.apply(self.setup, parsed(big), None)
        self.assertEqual(result.registered, ["big-one"])
        self.assertEqual(self.mf.app_sizes("big-one"), [])
        self.assertNotIn("big-one", self.mf.FEED_FAMILIES)

    def test_the_mac_table_lists_it_after_the_built_in_sizes(self):
        feed_models.load(self.signed_feed("f", 7, [flash_next()]), trusted={self.key["key_id"]: base64.b64decode(self.key["public"])})
        import strata_mcp as M
        rows = M.Strata(ROOT).model_table({"ram_gb": 64.0, "gpus": [], "mac": {"chip": "M4", "unified_gb": 64}})
        by = {r["family"]: r for r in rows}
        self.assertTrue(by["qwen4-test"]["feed"])
        self.assertEqual([x["model"] for x in by["qwen4-test"]["sizes"]], ["Q4_K_M"])
        self.assertNotIn("feed", by["qwen"])

    def test_the_terminal_setup_reads_the_feed_for_a_name_it_does_not_know(self):
        sys.path.insert(0, str(ROOT / "metal"))
        import setup as global_setup
        import setup_mac
        saved = {k: copy.deepcopy(getattr(global_setup, k)) for k in ("MODELS", "FAMILIES", "HF_REVISIONS", "REV_PINS", "FEED_MODELS", "FEED_UNAVAILABLE")}

        def restore():
            for k, v in saved.items():
                if isinstance(v, dict) and k in ("MODELS", "FAMILIES", "HF_REVISIONS", "REV_PINS"):
                    getattr(global_setup, k).clear()
                    getattr(global_setup, k).update(v)
                else:
                    setattr(global_setup, k, v)
        self.addCleanup(restore)
        feed = self.signed_feed("f", 7, [flash_next()])
        setup_mac._feed_models(["--family", "qwen4-test", "--feed", feed], False)
        self.assertIn("qwen4-test", global_setup.FAMILIES)
        self.assertEqual(setup_mac.model_shard_names("Q4_K_M"), ["a-00001-of-00002.gguf", "a-00002-of-00002.gguf"])
        self.assertEqual(setup_mac.model_shard_names("Q2_0"), [global_setup.model_file(global_setup.FAMILIES["qwen"], "Q2_0", i) for i in (1, 2)])


class TerminalRow(Base):
    def test_the_terminal_names_what_this_engine_does_not_run(self):
        from strata_cli.launch.catalog import Catalog
        feed_models.load(self.signed_feed("f", 7, [flash_next(id="later-one", family="qwen5new")]), trusted={self.key["key_id"]: base64.b64decode(self.key["public"])})
        hw = {"ram_gb": 64.0, "gpus": [], "disk_free_gb": 500.0}
        cat = Catalog(src=ROOT, root=self.root / "inst", hardware=hw)
        self.assertEqual(cat.unavailable(hw), [("Qwen4 Test", feed_models.UNAVAILABLE_UNKNOWN)])
        self.assertNotIn("later-one", [r.family for r in cat.rows(hw)])


if __name__ == "__main__":
    unittest.main()
