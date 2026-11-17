"""The signed feed's optional ``models`` member (strata_control/feed.py FeedModel, strata_control/feed_models.py, packaging/feed_tool.py --models).

What these tests pin:
  * a feed without the member parses to exactly what it did before (no models, no dropped lines, the other members equal), and writes nothing;
  * a malformed model entry is dropped with one plain line while the rest of the feed and the other models work; unknown keys are ignored;
  * Control 0.2.11 (the released feed.py, read from git) ignores the member;
  * the list survives offline: the last verified feed's list comes back from the cache, its signature checked again;
  * feed_tool build-feed --models / verify-feed.

Only a throwaway key made here is used (strata_control.signing.generate_key); no release key, no network (a folder is the feed).
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "tools" / "packaging"))

import feed_tool  # noqa: E402
from strata_control import feed as feedmod  # noqa: E402
from strata_control import feed_models, signing  # noqa: E402

import base64  # noqa: E402

SHA_A, SHA_B, SHA_C = "a" * 64, "b" * 64, "c" * 64
REV = "0123456789abcdef0123456789abcdef01234567"


def model(**kw) -> dict:
    m = {"id": "qwen4-test", "title": "Qwen4 Test", "by": "Qwen", "about": "a test model", "licence": "Apache 2.0",
         "licence_url": "https://huggingface.co/test/qwen4-test", "family": "qwen4exp", "repo": "test/Qwen4-Test-GGUF", "revision": REV,
         "pictures": True, "drafter": "mtp", "context": 131072, "min_engine": "0.2.12",
         "mmproj": {"name": "mmproj-F16.gguf", "size": 900, "sha256": SHA_C},
         "quants": [{"name": "Q4_K_M", "about": "4-bit", "ram_gb": 48, "arena_gb": 30.5,
                     "files": [{"name": "Qwen4-Test-Q4_K_M-00001-of-00002.gguf", "path": "Q4_K_M/Qwen4-Test-Q4_K_M-00001-of-00002.gguf", "size": 1000, "sha256": SHA_A},
                               {"name": "Qwen4-Test-Q4_K_M-00002-of-00002.gguf", "path": "Q4_K_M/Qwen4-Test-Q4_K_M-00002-of-00002.gguf", "size": 500, "sha256": SHA_B}]}]}
    m.update(kw)
    return m


def engine(**kw):
    e = {"version": "0.2.12", "platform": "linux-x86_64", "backend": "cuda", "variant": "cuda13", "archs": ["sm_86"], "min_driver": {},
         "min_runtime": None, "build_id": "b", "format": "tar.gz", "url": "e.tar.gz", "size": 10, "sha256": "a" * 64,
         "manifest_sha256": "b" * 64, "installed_bytes": 20, "notes": ""}
    e.update(kw)
    return e


def doc(**members) -> dict:
    d = {"schema_version": 1, "feed": "strata-updates", "channel": "stable", "sequence": 1, "generated_at": "2026-11-06T00:00:00Z", "expires_at": None}
    d.update(members)
    return d


def parse(d) -> feedmod.Feed:
    return feedmod.parse(json.dumps(d).encode(), source="x/strata-feed.json", key_id="k")


def pubkeys(key: dict) -> dict:
    return {key["key_id"]: base64.b64decode(key["public"])}


class Parsing(unittest.TestCase):
    def test_a_complete_entry_is_read_field_by_field(self):
        f = parse(doc(models={"packages": [model()]}))
        self.assertEqual(f.models_dropped, ())
        (m,) = f.models
        self.assertEqual((m.id, m.title, m.by, m.family, m.repo, m.revision), ("qwen4-test", "Qwen4 Test", "Qwen", "qwen4exp", "test/Qwen4-Test-GGUF", REV))
        self.assertEqual((m.licence, m.licence_url), ("Apache 2.0", "https://huggingface.co/test/qwen4-test"))
        self.assertEqual((m.drafter, m.context, m.min_engine, m.pictures, m.mtp_drafts), ("mtp", 131072, "0.2.12", True, 2))
        self.assertEqual(m.mmproj.name, "mmproj-F16.gguf")
        (q,) = m.quants
        self.assertEqual((q.name, q.ram_gb, q.arena_gb, q.download_bytes), ("Q4_K_M", 48.0, 30.5, 1500))
        self.assertEqual(q.files[0].repo_path, "Q4_K_M/Qwen4-Test-Q4_K_M-00001-of-00002.gguf")
        self.assertEqual(m.to_dict()["quants"][0]["files"][1]["sha256"], SHA_B)

    def test_a_feed_without_the_member_is_what_it_was(self):
        base = doc(engine={"packages": [engine()]})
        plain = parse(base)
        self.assertEqual((plain.models, plain.models_dropped, plain.signature), ((), (), b""))
        more = parse({**base, "models": {"packages": [model()]}})
        for field in dataclasses.fields(plain):
            if field.name in ("models", "models_dropped", "raw"):
                continue
            self.assertEqual(getattr(plain, field.name), getattr(more, field.name), field.name)

    def test_unknown_keys_and_members_are_ignored(self):
        e = model(future_field={"x": 1})
        e["quants"][0]["note"] = "later"
        f = parse(doc(models={"packages": [e], "next_release": 2}, something_else=[1]))
        self.assertEqual(len(f.models), 1)
        self.assertEqual(f.models_dropped, ())

    def test_a_malformed_entry_is_dropped_with_a_line_and_the_rest_works(self):
        bad = {
            "no id": model(id=""),
            "an id with capitals": model(id="Qwen4"),
            "no sha256": None,
            "a short revision": model(revision="main"),
            "an http licence url": model(licence_url="http://example.com/l"),
            "a size of zero": None,
            "a file name with a slash": None,
            "pictures without an mmproj": model(mmproj=None),
            "an unknown drafter": model(drafter="medusa"),
            "a family that needs kv": model(family="qwen35moe"),
            "no quants": model(quants=[]),
            "a version that is not one": model(min_engine="newest"),
            "not an object": "qwen4",
        }
        no_sha = model()
        del no_sha["quants"][0]["files"][0]["sha256"]
        bad["no sha256"] = no_sha
        zero = model()
        zero["quants"][0]["files"][0]["size"] = 0
        bad["a size of zero"] = zero
        slash = model()
        slash["quants"][0]["files"][0]["name"] = "../x.gguf"
        bad["a file name with a slash"] = slash
        good = model(id="qwen4-fine")
        with self.assertLogs("strata.feed", level="WARNING") as logs:
            f = parse(doc(models={"packages": [*bad.values(), good]}, engine={"packages": [engine()]}))
        self.assertEqual([m.id for m in f.models], ["qwen4-fine"])
        self.assertEqual(len(f.models_dropped), len(bad), f.models_dropped)
        self.assertEqual(len(logs.records), len(bad))
        self.assertEqual(len(f.engine), 1)                       # the rest of the feed is untouched
        for line in f.models_dropped:
            self.assertNotIn("\n", line)

    def test_a_repeated_id_or_file_or_quant_is_dropped(self):
        twice = parse(doc(models={"packages": [model(), model(title="again")]}))
        self.assertEqual(len(twice.models), 1)
        self.assertEqual(twice.models[0].title, "Qwen4 Test")
        self.assertEqual(len(twice.models_dropped), 1)
        dup = model()
        dup["quants"].append(copy.deepcopy(dup["quants"][0]))
        self.assertEqual(len(parse(doc(models={"packages": [dup]})).models), 0)
        same_file = model()
        same_file["quants"].append({**copy.deepcopy(same_file["quants"][0]), "name": "Q5"})
        self.assertEqual(len(parse(doc(models={"packages": [same_file]})).models), 0)

    def test_a_member_of_the_wrong_shape_is_ignored_whole(self):
        for member in ("none", ["x"], {"packages": "no"}, {"nothing": 1}):
            f = parse(doc(models=member, engine={"packages": [engine()]}))
            self.assertEqual(f.models, ())
            self.assertEqual(len(f.models_dropped), 1)
            self.assertEqual(len(f.engine), 1)

    def test_the_qwen35moe_entry_needs_kv_and_the_other_families_are_just_names(self):
        ok = model(family="qwen35moe", kv_bytes_per_token=90112)
        f = parse(doc(models={"packages": [ok, model(id="later-one", family="qwen5new")]}))
        self.assertEqual([(m.id, m.family) for m in f.models], [("qwen4-test", "qwen35moe"), ("later-one", "qwen5new")])
        self.assertEqual(f.models[0].kv_bytes_per_token, 90112)


class OlderControls(unittest.TestCase):
    def test_control_0211_ignores_the_member(self):
        """The feed.py of the released Control 0.2.11 (git tag v0.2.11) reads the document by member name: a feed with models parses to what the same feed
        without them does. Skipped where the tag is not in the clone."""
        shown = subprocess.run(["git", "-C", str(ROOT), "show", "v0.2.11:strata_control/feed.py"], capture_output=True)
        if shown.returncode != 0:
            self.skipTest("the v0.2.11 tag is not in this clone")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "feed_0211.py"
            path.write_bytes(shown.stdout)
            spec = importlib.util.spec_from_file_location("feed_0211", path)
            old = importlib.util.module_from_spec(spec)
            sys.modules["feed_0211"] = old
            try:
                spec.loader.exec_module(old)
            finally:
                sys.modules.pop("feed_0211", None)
        base = doc(engine={"packages": [engine()]})
        with_models = {**base, "models": {"packages": [model(), model(id="bad-one", revision="x")]}, "drafters": {"packages": []}}
        a = old.parse(json.dumps(base).encode(), source="s", key_id="k")
        b = old.parse(json.dumps(with_models).encode(), source="s", key_id="k")
        self.assertEqual(a.engine, b.engine)
        self.assertEqual((a.sequence, a.channel, a.app, a.control), (b.sequence, b.channel, b.app, b.control))
        self.assertFalse(hasattr(b, "models"))


class Signed(unittest.TestCase):
    """A folder signed with a throwaway key stands for the release feed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = signing.generate_key()
        self.trusted = pubkeys(self.key)
        self.components = self.root / "components"
        self.old_env = os.environ.get("STRATA_COMPONENTS_DIR")
        os.environ["STRATA_COMPONENTS_DIR"] = str(self.components)
        self.addCleanup(self._restore)

    def _restore(self):
        if self.old_env is None:
            os.environ.pop("STRATA_COMPONENTS_DIR", None)
        else:
            os.environ["STRATA_COMPONENTS_DIR"] = self.old_env

    def feed(self, name: str, sequence: int, models=None, **extra) -> str:
        """A signed feed folder; returns its path."""
        out = self.root / name
        out.mkdir()
        members = dict(extra)
        if models is not None:
            members["models"] = {"packages": models}
        payload = (json.dumps(doc(sequence=sequence, **members), indent=1, sort_keys=True) + "\n").encode()
        (out / feedmod.FEED_NAME).write_bytes(payload)
        (out / (feedmod.FEED_NAME + ".sig")).write_text(json.dumps(signing.sign_bytes(self.key, payload)), encoding="utf-8")
        return str(out)

    def test_the_fetched_list_is_kept_and_comes_back_offline(self):
        online = feed_models.load(self.feed("f1", 5, [model()]), trusted=self.trusted)
        self.assertEqual((online.via, [m.id for m in online.models], online.sequence), ("feed", ["qwen4-test"], 5))
        self.assertTrue(feed_models.cache_file().is_file())
        gone = feed_models.load(str(self.root / "no-such-feed"), trusted=self.trusted)
        self.assertEqual((gone.via, [m.id for m in gone.models]), ("cache", ["qwen4-test"]))
        self.assertEqual(feed_models.load(None, fetch=False, trusted=self.trusted).via, "cache")

    def test_a_cache_that_was_edited_or_signed_by_another_key_is_worth_nothing(self):
        feed_models.load(self.feed("f1", 5, [model()]), trusted=self.trusted)
        path = feed_models.cache_file()
        saved = path.read_text(encoding="utf-8")
        edited = json.loads(saved)
        payload = json.loads(base64.b64decode(edited["payload"]))
        payload["models"]["packages"][0]["quants"][0]["files"][0]["sha256"] = "d" * 64
        edited["payload"] = base64.b64encode(json.dumps(payload).encode()).decode()
        path.write_text(json.dumps(edited), encoding="utf-8")
        self.assertEqual(feed_models.load(None, fetch=False, trusted=self.trusted).via, "none")
        path.write_text(saved, encoding="utf-8")
        other = pubkeys(signing.generate_key())
        self.assertEqual(feed_models.load(None, fetch=False, trusted=other).via, "none")
        self.assertEqual(feed_models.load(None, fetch=False, trusted=self.trusted).via, "cache")

    def test_an_older_feed_does_not_replace_a_newer_list(self):
        feed_models.load(self.feed("f5", 5, [model()]), trusted=self.trusted)
        old = feed_models.load(self.feed("f3", 3, [model(id="replayed-one")]), trusted=self.trusted)
        self.assertEqual((old.via, [m.id for m in old.models]), ("cache", ["qwen4-test"]))

    def test_a_newer_feed_without_models_withdraws_the_list(self):
        feed_models.load(self.feed("f1", 5, [model()]), trusted=self.trusted)
        none = feed_models.load(self.feed("f2", 6), trusted=self.trusted)
        self.assertEqual((none.via, none.models), ("feed", ()))
        self.assertFalse(feed_models.cache_file().exists())
        self.assertEqual(feed_models.load(str(self.root / "gone"), trusted=self.trusted).via, "none")

    def test_a_feed_without_models_writes_nothing(self):
        got = feed_models.load(self.feed("plain", 1, engine={"packages": [engine()]}), trusted=self.trusted)
        self.assertEqual((got.via, got.models), ("feed", ()))
        self.assertFalse(self.components.exists())

    def test_the_switch_turns_it_off(self):
        os.environ["STRATA_FEED_MODELS"] = "0"
        self.addCleanup(os.environ.pop, "STRATA_FEED_MODELS", None)
        self.assertEqual(feed_models.load(self.feed("f1", 5, [model()]), trusted=self.trusted).via, "none")
        self.assertFalse(self.components.exists())

    def test_an_expired_feed_is_not_taken_but_its_cached_list_still_serves(self):
        feed_models.load(self.feed("f1", 5, [model()]), trusted=self.trusted)
        stale = self.feed("f2", 6, [model(id="newer-one")], expires_at="2020-01-01T00:00:00Z")
        got = feed_models.load(stale, trusted=self.trusted)
        self.assertEqual((got.via, [m.id for m in got.models]), ("cache", ["qwen4-test"]))


class UpdaterKeepsTheList(Signed):
    """`strata update` / the app's check reads the feed through updater.open_feed; the model list is kept there for the offline PC."""

    def ctx(self, source: str):
        from strata_control import updater
        hardware = {"platform": "linux-x86_64", "backend": "cpu", "cuda_archs": [], "gfx": []}
        return updater, updater.Context(components=self.components, source=source, trusted=self.trusted, hardware=hardware, environ={})

    def test_an_update_check_keeps_the_list_and_a_feed_without_models_removes_it(self):
        updater, ctx = self.ctx(self.feed("f1", 5, [model()]))
        with mock_acquire():
            loaded = updater.open_feed(ctx, updater.State(self.components))
        self.assertEqual([m.id for m in loaded.models], ["qwen4-test"])
        self.assertEqual(feed_models.load(None, fetch=False, trusted=self.trusted).via, "cache")
        updater, ctx = self.ctx(self.feed("f2", 6))
        with mock_acquire():
            updater.open_feed(ctx, updater.State(self.components))
        self.assertFalse(feed_models.cache_file().exists())

    def test_an_update_check_of_a_feed_without_models_writes_no_cache(self):
        updater, ctx = self.ctx(self.feed("f1", 5, engine={"packages": [engine()]}))
        with mock_acquire():
            updater.open_feed(ctx, updater.State(self.components))
        self.assertFalse(feed_models.cache_file().exists())


def mock_acquire():
    from unittest import mock
    return mock.patch("strata_control.updater.require_acquisition", lambda what: None)


class FeedTool(unittest.TestCase):
    def test_model_files_prints_the_pins_of_real_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "x-00001-of-00002.gguf"
            f.write_bytes(b"abc")
            got = feed_tool.model_files([f], "Q4_K_M")
            self.assertEqual(got, [{"name": "x-00001-of-00002.gguf", "size": 3, "sha256": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                                    "path": "Q4_K_M/x-00001-of-00002.gguf"}])
            self.assertEqual(feed_tool.model_files([f]), [{k: v for k, v in got[0].items() if k != "path"}])

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = signing.generate_key()
        self.models = self.root / "models"
        self.models.mkdir()

    def entry(self, name: str, **kw):
        (self.models / f"{name}.entry.json").write_text(json.dumps(model(id=name, **kw)), encoding="utf-8")

    def test_build_feed_models_makes_a_member_control_reads(self):
        self.entry("qwen4-test")
        self.entry("qwen4-test-b", family="qwen35moe", kv_bytes_per_token=1000)
        out = self.root / "rel"
        document = feed_tool.build_feed(out, key=self.key, sequence=3, models_dir=self.models)
        self.assertEqual(sorted(m["id"] for m in document["models"]["packages"]), ["qwen4-test", "qwen4-test-b"])
        loaded = feedmod.fetch(str(out), pubkeys(self.key))
        self.assertEqual(sorted(m.id for m in loaded.models), ["qwen4-test", "qwen4-test-b"])
        self.assertEqual(loaded.models_dropped, ())

    def test_build_feed_without_models_has_no_member(self):
        out = self.root / "rel"
        document = feed_tool.build_feed(out, key=self.key, sequence=3)
        self.assertNotIn("models", document)
        self.assertNotIn(b"models", (out / feedmod.FEED_NAME).read_bytes())

    def test_a_malformed_entry_stops_the_build(self):
        self.entry("qwen4-test", revision="main")
        with self.assertRaises(SystemExit) as caught:
            feed_tool.build_feed(self.root / "rel", key=self.key, sequence=3, models_dir=self.models)
        self.assertIn("revision", str(caught.exception))

    def test_verify_feed_lists_the_models_and_refuses_a_feed_with_a_dropped_one(self):
        self.entry("qwen4-test")
        out = self.root / "rel"
        feed_tool.build_feed(out, key=self.key, sequence=3, models_dir=self.models)
        pub = self.root / "pub.json"
        pub.write_text(json.dumps({"keys": [signing.public_part(self.key)]}), encoding="utf-8")
        self.assertEqual(feed_tool.main(["verify-feed", str(out), "--pub", str(pub)]), 0)
        broken = self.root / "broken"
        broken.mkdir()
        payload = json.loads((out / feedmod.FEED_NAME).read_text(encoding="utf-8"))
        payload["models"]["packages"][0]["revision"] = "main"
        raw = (json.dumps(payload, indent=1, sort_keys=True) + "\n").encode()
        (broken / feedmod.FEED_NAME).write_bytes(raw)
        (broken / (feedmod.FEED_NAME + ".sig")).write_text(json.dumps(signing.sign_bytes(self.key, raw)), encoding="utf-8")
        self.assertEqual(feed_tool.main(["verify-feed", str(broken), "--pub", str(pub)]), 3)


if __name__ == "__main__":
    logging.basicConfig()
    unittest.main()
