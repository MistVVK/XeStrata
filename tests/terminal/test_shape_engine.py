"""A Qwen4 model of its own shape runs on its own engine component (strata_control/shape_engine.py), picked by the model, beside the Strata engine.

What these tests pin:
  * the names: shape q4_lite, component strata-q4-lite, feed model family qwen4exp_q4_lite, run config reference component:strata-q4-lite;
  * packaging: package_engine --shape makes a package whose manifest id is the shape's component; read as the Strata engine it is refused, so an older Control (which reads only
    strata-engine) ignores one put into its engine folder by hand; a normal package never claims a shape;
  * the feed: member shape_engines, parsed, never mixed into ``engine``; feed_tool packs and lists it and refuses a model of a shape that has no engine;
  * the install (a local signed feed over loopback, a stub engine, a throwaway key): the package is fetched, checked (signature, hashes, shape), placed in
    <components>/shapes/<shape>/engine/, started once, and the Strata engine's files are byte for byte what they were; a bad package or a failed start changes nothing;
  * use: component:strata-q4-lite resolves to that engine, component:strata-engine still to the Strata engine; a missing shape engine is a plain refusal, not a fetch;
  * update: `strata update` brings an installed shape engine up to date beside the older one and prunes beyond two; a PC with none has no such step;
  * older Controls: the feed.py of the Control before this (git) reads the same feed as one without the new member and does not offer the model.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packaging"))

import feed_tool  # noqa: E402
import package_engine  # noqa: E402
from strata_control import components, engine_component as ec, feed as feedmod, shape_engine, signing, updater  # noqa: E402
from strata_control.targets import host_platform  # noqa: E402
from tests.terminal import update_fixture as fx  # noqa: E402

REV = "0123456789abcdef0123456789abcdef01234567"
SHAPE = "q4_lite"
COMP = "strata-q4-lite"
EXE = shape_engine.exe_name()


def stub_source(parent: Path, version: str, *, text: str = "stub strata", padding: int = 0) -> Path:
    src = parent / f"src-{version}"
    src.mkdir(parents=True)
    (src / EXE).write_bytes(f"{text} {version} ".encode() + (os.urandom(padding) if padding else b"x" * 100))
    (src / ("cudart64_13.dll" if os.name == "nt" else "libcudart.so.13")).write_bytes(b"a runtime library " * 50)
    return src


def build(parent: Path, version: str, key: dict | None, *, shape: str = SHAPE, padding: int = 0, min_runtime: str | None = None) -> Path:
    out = parent / "built"
    out.mkdir(parents=True, exist_ok=True)
    folder = package_engine.build(stub_source(parent, version, padding=padding), "cuda", version, components.host_platform(), out, entry=EXE, archs=("sm_86",),
                                  min_driver={"nvidia": "580.65"}, build_id=f"{shape or 'plain'}-{version}", variant="cuda13", shape=shape, min_runtime=min_runtime)
    if key is not None:
        feed_tool.sign_package(folder, key)
    return folder


def tree(d: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(Path(d).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(d)).encode() + p.read_bytes())
    return h.hexdigest()


def model_entry(**kw) -> dict:
    m = {"id": "q4lite-test", "title": "Qwen4 Lite Test", "by": "Qwen", "about": "a test model", "licence": "Apache 2.0",
         "licence_url": "https://huggingface.co/test/q4lite", "family": shape_engine.family_of(SHAPE), "repo": "test/Q4Lite-GGUF", "revision": REV,
         "drafter": "none", "context": 131072,
         "quants": [{"name": "Q4_K_M", "about": "4-bit", "ram_gb": 8, "arena_gb": 2.5,
                     "files": [{"name": "Q4Lite-Q4_K_M.gguf", "size": 1000, "sha256": "a" * 64}]}]}
    m.update(kw)
    return m


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="strata-shape-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = fx.make_key()
        trust = self.root / "trust.json"
        trust.write_text(json.dumps({"keys": [signing.public_part(self.key)]}))
        self.comp = self.root / "c"
        for p in (mock.patch.dict(os.environ, {"STRATA_UPDATE_TRUSTED_KEYS": str(trust), "STRATA_COMPONENTS_DIR": str(self.comp)}),
                  mock.patch.object(signing, "BUILTIN_KEYS", self.root / "none.json"),
                  mock.patch.object(ec, "cuda_archs_present", return_value=["sm_86"]),
                  mock.patch.object(ec, "nvidia_driver_version", return_value="610.47"),
                  mock.patch.object(ec, "detect_backend", return_value=("cuda", "test"))):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(ec.register_signature_verifier, None)
        self.hw = {"platform": host_platform(), "backend": "cuda", "runtime": None, "nvidia_driver": "610.47", "cuda_archs": ["sm_86"], "gfx": []}
        self.release = self.root / "release"
        self.release.mkdir()
        self.server = fx.FeedServer(self.release)
        self.server.__enter__()
        self.addCleanup(self.server.__exit__)
        self.url = self.server.url + "/strata-feed.json"
        # the Strata engine of this PC: a signed stub package in <components>/engine, whose bytes must never change
        plain = build(self.root / "plain", "0.1.58", self.key, shape="")
        (self.comp / "engine").mkdir(parents=True)
        shutil.copytree(plain, self.comp / "engine" / plain.name)
        self.strata_before = tree(self.comp / "engine")

    def publish(self, *versions, shape: str = SHAPE, models=None, sequence=1, key=None, extra_members=None):
        shutil.rmtree(self.release, ignore_errors=True)
        (self.release / "shapes").mkdir(parents=True)
        self.published = getattr(self, "published", 0) + 1
        for v in versions:
            folder = build(self.root / f"pub-{shape}-{v}-{self.published}", v, key or self.key, shape=shape)
            feed_tool.pack_engine(folder, self.release / "shapes", url_prefix="shapes/")
        if models:
            (self.release / "models").mkdir()
            for m in models:
                (self.release / "models" / f"{m['id']}.entry.json").write_text(json.dumps(m))
        return feed_tool.build_feed(self.release, key=self.key, sequence=sequence, shapes_dir=self.release / "shapes" if versions else None,
                                    models_dir=self.release / "models" if models else None)

    def ensure(self, lines=None, **kw):
        out = [] if lines is None else lines
        return shape_engine.ensure(SHAPE, self.url, components=self.comp, hardware=self.hw, emit=out.append, **kw)

    def folders(self):
        return sorted(p.name.split("-")[0] for p in (self.comp / "shapes" / SHAPE / "engine").iterdir() if p.is_dir())


class Names(unittest.TestCase):
    def test_a_shape_has_one_component_one_family_one_reference(self):
        self.assertEqual(shape_engine.component_id("q4_lite"), "strata-q4-lite")
        self.assertEqual(shape_engine.shape_of_component("strata-q4-lite"), "q4_lite")
        self.assertEqual(shape_engine.family_of("q4_lite"), "qwen4exp_q4_lite")
        self.assertEqual(shape_engine.shape_of_family("qwen4exp_q4_lite"), "q4_lite")
        self.assertEqual(shape_engine.reference("q4_lite"), "component:strata-q4-lite")
        self.assertEqual(ec.REFERENCE_PREFIX + COMP, shape_engine.reference(SHAPE))

    def test_what_is_not_a_shape_is_none(self):
        for family in ("qwen4exp", "qwen35moe", "qwen4exp_", "qwen4exp_Q4", "qwen4exp_q4-lite", "qwen4exp_" + "x" * 30, None, 7):
            self.assertIsNone(shape_engine.shape_of_family(family), family)
        for comp in ("strata-engine", "strata-dsv41", "strata-control", "engine", "strata-", "strata-q4_lite", "strata-Q4-lite", None):
            self.assertIsNone(shape_engine.shape_of_component(comp), comp)

    def test_a_shape_never_becomes_a_component_that_means_something_else(self):
        for shape in ("engine", "dsv41", "control", "vision", "device"):
            self.assertFalse(shape_engine.valid(shape), shape)
        for shape in ("", "q", "Q4_lite", "q4-lite", "q4__lite", "q4_lite_", "1q4", "q4 lite", "q4/lite", "x" * 24):
            self.assertFalse(shape_engine.valid(shape), repr(shape))
        for shape in ("q4_lite", "q4_mid", "q4_deep", "q4_big", "qwen4_a", "x" * 23):
            self.assertTrue(shape_engine.valid(shape), shape)
        self.assertLessEqual(len(shape_engine.family_of("x" * 23)), 32, "a family name must fit the feed's family pattern")


class Packaging(Base):
    def test_a_shape_package_is_its_own_component(self):
        folder = build(self.root / "p", "0.1.59", self.key)
        raw = json.loads((folder / ec.MANIFEST).read_text())
        self.assertEqual((raw["id"], raw["name"], raw["shape"]), (COMP, COMP, SHAPE))
        comp = ec.read_manifest(folder, COMP)
        self.assertEqual((comp.shape, comp.id), (SHAPE, COMP))
        self.assertEqual(ec.read_manifest_any(folder).shape, SHAPE)
        self.assertEqual(comp.to_dict()["shape"], SHAPE)

    def test_it_is_not_read_as_the_strata_engine_so_an_older_control_ignores_it(self):
        folder = build(self.root / "p", "0.1.59", self.key)
        with self.assertRaises(ec.ComponentProblem) as ctx:
            ec.read_manifest(folder)                                   # what a Control from before this reads, and what the Strata engine's scan reads
        self.assertIn("not an engine component", str(ctx.exception))
        wrong = shape_engine.component_id("q4_mid")
        with self.assertRaises(ec.ComponentProblem):
            ec.read_manifest(folder, wrong)                            # the id and the shape must agree
        (self.comp / "engine" / folder.name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(folder, self.comp / "engine" / folder.name)    # copied into the Strata engine's folder by hand
        found, problems = ec.scan(self.comp)
        self.assertEqual([c.version for c in found], ["0.1.58"])
        self.assertEqual(len(problems), 1)
        self.assertEqual(ec.resolve(self.comp, backend="cuda").version, "0.1.58")

    def test_a_strata_engine_never_claims_a_shape(self):
        plain = build(self.root / "p2", "0.1.60", self.key, shape="")
        raw = json.loads((plain / ec.MANIFEST).read_text())
        self.assertEqual(raw["id"], "strata-engine")
        self.assertNotIn("shape", raw)
        raw["shape"] = SHAPE
        (plain / ec.MANIFEST).write_text(json.dumps(raw))
        with self.assertRaises(ec.ComponentProblem):
            ec.read_manifest(plain)

    def test_a_bad_shape_name_stops_the_packager(self):
        with self.assertRaises(SystemExit):
            build(self.root / "p3", "0.1.59", None, shape="Not A Shape")
        with self.assertRaises(SystemExit):
            build(self.root / "p4", "0.1.59", None, shape="engine")

    def test_the_command_line_makes_and_verifies_one(self):
        src = stub_source(self.root / "cli", "0.1.59")
        out = self.root / "cli-out"
        done = subprocess.run([sys.executable, str(ROOT / "packaging" / "package_engine.py"), "--src", str(src), "--backend", "cuda", "--version", "0.1.59",
                               "--out", str(out), "--arch", "sm_86", "--entrypoint", EXE, "--shape", SHAPE, "--build-id", "x"], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        folder = next(out.iterdir())
        verified = subprocess.run([sys.executable, str(ROOT / "packaging" / "package_engine.py"), "--verify", str(folder)], capture_output=True, text=True)
        self.assertEqual(verified.returncode, 0, verified.stdout)
        self.assertEqual(json.loads((folder / ec.MANIFEST).read_text())["id"], COMP)


class Feed(Base):
    def test_the_member_is_parsed_and_kept_apart_from_engine(self):
        doc = self.publish("0.1.59")
        self.assertEqual(doc["shape_engines"]["packages"][0]["shape"], SHAPE)
        loaded = feedmod.fetch(self.url, signing.load_trusted_keys())
        self.assertEqual([(p.shape, p.version) for p in loaded.shape_engines], [(SHAPE, "0.1.59")])
        self.assertEqual(loaded.engine, ())
        self.assertEqual(loaded.dsv41, ())
        self.assertTrue(loaded.shape_engines[0].url.startswith("shapes/strata-q4-lite-0.1.59-"))
        self.assertEqual(loaded.shape_engines[0].to_dict()["shape"], SHAPE)

    def test_a_feed_without_the_member_is_what_it_was(self):
        shutil.rmtree(self.release, ignore_errors=True)
        feed_tool.build_feed(self.release, key=self.key, sequence=1)
        loaded = feedmod.fetch(self.url, signing.load_trusted_keys())
        self.assertEqual(loaded.shape_engines, ())
        self.assertNotIn("shape_engines", json.loads((self.release / feedmod.FEED_NAME).read_text()))

    def test_an_entry_with_a_shape_this_control_does_not_accept_is_left_out_with_a_line(self):
        doc = self.publish("0.1.59")
        doc["shape_engines"]["packages"].append({**doc["shape_engines"]["packages"][0], "shape": "Not Valid"})
        parsed = feedmod.parse(json.dumps(doc).encode(), source="x", key_id="k")
        self.assertEqual(len(parsed.shape_engines), 1)
        self.assertTrue(any("shape" in line for line in parsed.skipped), parsed.skipped)

    def test_the_same_shape_and_version_twice_is_refused(self):
        doc = self.publish("0.1.59")
        doc["shape_engines"]["packages"].append(dict(doc["shape_engines"]["packages"][0]))
        with self.assertRaises(feedmod.FeedError):
            feedmod.parse(json.dumps(doc).encode(), source="x", key_id="k")

    def test_a_model_of_a_shape_is_parsed_with_its_shape(self):
        self.publish("0.1.59", models=[model_entry()])
        loaded = feedmod.fetch(self.url, signing.load_trusted_keys())
        (m,) = loaded.models
        self.assertEqual((m.family, m.shape), (shape_engine.family_of(SHAPE), SHAPE))
        self.assertIsNone(feedmod.parse(json.dumps({"schema_version": 1, "channel": "stable", "generated_at": "2026-11-08T00:00:00Z", "sequence": 1,
                                                    "models": {"packages": [model_entry(id="plain-one", family="qwen4exp")]}}).encode(), source="x", key_id="k").models[0].shape)

    def test_a_model_with_a_broken_shape_name_is_dropped_with_a_line(self):
        doc = {"schema_version": 1, "channel": "stable", "generated_at": "2026-11-08T00:00:00Z", "sequence": 1,
               "models": {"packages": [model_entry(family="qwen4exp_Q4-bad")]}}
        parsed = feedmod.parse(json.dumps(doc).encode(), source="x", key_id="k")
        self.assertEqual((parsed.models, len(parsed.models_dropped)), ((), 1))

    def test_build_feed_refuses_a_model_of_a_shape_with_no_engine(self):
        (self.release / "models").mkdir(parents=True)
        (self.release / "models" / "q4lite-test.entry.json").write_text(json.dumps(model_entry()))
        with self.assertRaises(SystemExit) as ctx:
            feed_tool.build_feed(self.release, key=self.key, sequence=1, models_dir=self.release / "models")
        self.assertIn("q4_lite", str(ctx.exception))
        with self.assertRaises(SystemExit):
            self.publish("0.1.59", shape="q4_mid", models=[model_entry()])       # an engine for another shape does not do

    def test_pack_engine_names_the_archive_by_the_component_and_the_entry_by_the_shape(self):
        folder = build(self.root / "pk", "0.1.59", self.key)
        out = self.root / "pk-out"
        out.mkdir()
        entry = feed_tool.pack_engine(folder, out, url_prefix="")
        self.assertEqual(entry["shape"], SHAPE)
        self.assertTrue(entry["url"].startswith("strata-q4-lite-0.1.59-"), entry["url"])
        plain = build(self.root / "pk2", "0.1.60", self.key, shape="")
        self.assertNotIn("shape", feed_tool.pack_engine(plain, out, url_prefix=""))
        self.assertTrue(next(p for p in out.glob("*.entry.json") if "strata-engine" in p.name))

    def test_verify_feed_lists_the_shape_engines(self):
        self.publish("0.1.59", models=[model_entry()])
        pub = self.root / "pub.json"
        pub.write_text(json.dumps({"keys": [signing.public_part(self.key)]}))
        done = subprocess.run([sys.executable, str(ROOT / "packaging" / "feed_tool.py"), "verify-feed", str(self.release), "--pub", str(pub)], capture_output=True, text=True)
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("1 shape engines (q4_lite)", done.stdout)


class Install(Base):
    def test_the_engine_of_the_shape_is_fetched_beside_the_strata_engine(self):
        self.publish("0.1.59")
        lines = []
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "strata 0.1.59")):
            exe = self.ensure(lines)
        self.assertIsNotNone(exe, lines)
        self.assertEqual(exe.parent.parent.parent, self.comp / "shapes" / SHAPE)
        self.assertTrue(exe.is_file())
        self.assertEqual(self.strata_before, tree(self.comp / "engine"), "the Strata engine's files must not change")
        self.assertEqual(ec.resolve(self.comp, backend="cuda").version, "0.1.58")
        self.assertIn("0.1.59", " ".join(lines))
        self.assertEqual(shape_engine.newest_version(SHAPE, self.comp), "0.1.59")
        self.assertEqual(shape_engine.installed_shapes(self.comp), [SHAPE])
        self.assertEqual(json.loads((exe.parent / ec.MANIFEST).read_text())["shape"], SHAPE)

    def test_the_reference_resolves_to_the_shape_engine_and_the_other_to_the_strata_engine(self):
        self.publish("0.1.59")
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            exe = self.ensure()
        self.assertEqual(Path(ec.expand("component:strata-q4-lite", {"backend": "cuda"})), exe)
        strata = Path(ec.expand("component:strata-engine", {"backend": "cuda"}))
        self.assertEqual(strata.parent.parent.name, "engine")
        self.assertNotEqual(strata, exe)
        self.assertEqual(ec.expand("/some/path/strata"), "/some/path/strata")

    def test_a_shape_with_no_engine_installed_is_a_plain_refusal_and_never_a_fetch(self):
        with self.assertRaises(shape_engine.ShapeEngineMissing) as ctx:
            ec.expand("component:strata-q4-lite", {"backend": "cuda"})
        self.assertIsInstance(ctx.exception, ec.EngineMissing)
        self.assertEqual(ctx.exception.exit_code, 5)
        self.assertIn("q4_lite", str(ctx.exception))
        self.assertIn("never downloads", str(ctx.exception))
        with self.assertRaises(ec.EngineMissing):
            ec.expand("component:strata-nonsense", {})
        self.assertFalse((self.comp / "shapes").exists())

    def test_a_changed_file_stops_the_start_of_that_engine_only(self):
        self.publish("0.1.59")
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            exe = self.ensure()
        exe.write_bytes(exe.read_bytes() + b"changed")
        with self.assertRaises(ec.EngineMissing):
            ec.expand("component:strata-q4-lite", {"backend": "cuda"})
        self.assertTrue(Path(ec.expand("component:strata-engine", {"backend": "cuda"})).is_file())

    def test_an_installed_engine_that_is_current_is_not_fetched_again_and_says_nothing(self):
        self.publish("0.1.59")
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            self.assertIsNotNone(self.ensure())
            self.server.requests.clear()
            lines = []
            self.assertIsNotNone(self.ensure(lines))
        self.assertEqual(lines, [])
        self.assertFalse([r for r in self.server.requests if r[1].endswith(".tar.gz")], self.server.requests)

    def test_a_cut_download_resumes(self):
        self.publish("0.1.59")
        shutil.rmtree(self.release)
        (self.release / "shapes").mkdir(parents=True)
        folder = build(self.root / "big", "0.1.59", self.key, padding=3_000_000)
        feed_tool.pack_engine(folder, self.release / "shapes", url_prefix="shapes/")
        feed_tool.build_feed(self.release, key=self.key, sequence=1, shapes_dir=self.release / "shapes")
        self.server.cut[next((self.release / "shapes").glob("*.tar.gz")).name] = 800_000
        lines = []
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            self.assertIsNone(self.ensure(lines))
            self.assertIsNotNone(self.ensure(lines), lines)
        self.assertEqual(self.strata_before, tree(self.comp / "engine"))

    def test_the_feed_has_no_engine_for_the_shape(self):
        self.publish("0.1.59", shape="q4_mid")
        lines = []
        self.assertIsNone(self.ensure(lines))
        self.assertIn("The update feed has no engine for q4_lite for this computer.", " ".join(lines))
        self.assertFalse((self.comp / "shapes" / SHAPE).exists())
        self.assertEqual(self.strata_before, tree(self.comp / "engine"))

    def test_no_engine_for_this_pc_is_a_plain_line(self):
        self.publish("0.1.59")
        self.hw = {**self.hw, "cuda_archs": ["sm_61"]}
        lines = []
        with mock.patch.object(ec, "cuda_archs_present", return_value=["sm_61"]):
            self.assertIsNone(self.ensure(lines))
        self.assertTrue(lines)
        self.assertFalse((self.comp / "shapes" / SHAPE).exists())

    def test_a_package_signed_with_another_key_is_refused(self):
        self.publish("0.1.59", key=fx.make_key())
        lines = []
        self.assertIsNone(self.ensure(lines))
        self.assertTrue(lines)
        self.assertFalse((self.comp / "shapes" / SHAPE / "engine").exists() and list((self.comp / "shapes" / SHAPE / "engine").iterdir()))
        self.assertEqual(self.strata_before, tree(self.comp / "engine"))

    def test_a_feed_that_names_another_shape_than_the_package_says_is_refused(self):
        doc = self.publish("0.1.59")
        entry_file = next((self.release / "shapes").glob("*.entry.json"))
        item = json.loads(entry_file.read_text())
        item["shape"] = "q4_mid"                                       # the feed says q4_mid, the signed manifest says q4_lite
        entry_file.write_text(json.dumps(item))
        feed_tool.build_feed(self.release, key=self.key, sequence=2, shapes_dir=self.release / "shapes")
        lines = []
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            self.assertIsNone(shape_engine.ensure("q4_mid", self.url, components=self.comp, hardware=self.hw, emit=lines.append))
        self.assertIn("not an engine component", " ".join(lines))
        self.assertFalse((self.comp / "shapes" / "q4_mid" / "engine").exists() and list((self.comp / "shapes" / "q4_mid" / "engine").iterdir()))

    def test_a_package_that_does_not_start_is_removed(self):
        self.publish("0.1.59")
        lines = []
        with mock.patch.object(shape_engine, "_test_start", return_value=(False, "libcudart.so.13: not found")):
            self.assertIsNone(self.ensure(lines))
        self.assertIn("was not installed because it did not start correctly", " ".join(lines))
        self.assertFalse([l for l in lines if "could not be downloaded" in l], "the approved line is said as it is, not inside another one")
        self.assertIn("The Strata engine is unchanged", " ".join(lines))
        self.assertEqual(shape_engine.installed(SHAPE, self.comp), [])
        self.assertEqual(self.strata_before, tree(self.comp / "engine"))

    def test_the_real_start_test_runs_the_program_and_wants_its_name(self):
        src = self.root / "prog"
        src.mkdir()
        script = src / "strata"
        script.write_text(f'#!{sys.executable}\nimport sys\nprint("strata 0.1.59" if "--version" in sys.argv else "no")\n')
        script.chmod(0o755)
        if os.name == "nt":
            self.skipTest("a script is not a Windows program")
        self.assertEqual(shape_engine._test_start(script), (True, "strata 0.1.59"))
        script.write_text(f'#!{sys.executable}\nprint("hello")\n')
        self.assertFalse(shape_engine._test_start(script)[0])

    def _program(self, geometry):
        """A stand-in engine program: `--version` names it, `--geometry` prints `geometry` (None: an engine from before the flag, which exits 2)."""
        src = self.root / f"prog-{geometry}"
        src.mkdir(exist_ok=True)
        script = src / "strata"
        answer = f'print({geometry!r})' if geometry else 'sys.exit(2)'
        script.write_text(f'#!{sys.executable}\nimport sys\nif "--version" in sys.argv:\n    print("strata 0.2.13")\nelif "--geometry" in sys.argv:\n    {answer}\nelse:\n    sys.exit(3)\n')
        script.chmod(0o755)
        return script

    def test_the_engine_must_be_built_for_the_shape_the_signed_manifest_names(self):
        if os.name == "nt":
            self.skipTest("a script is not a Windows program")
        self.assertEqual(shape_engine._test_start(self._program("q4_lite"), shape="q4_lite"), (True, "strata 0.2.13"))
        self.assertEqual(shape_engine._test_start(self._program(None), shape="q4_lite"), (False, "it does not say which model shape it was built for"),
                         "a shape engine that cannot answer --geometry is refused (owner, 11-09)")
        ok, why = shape_engine._test_start(self._program("flash_next"), shape="q4_lite")
        self.assertEqual((ok, why), (False, "it was built for a different model shape (flash_next)"))
        self.assertEqual(shape_engine._test_start(self._program("flash_next")), (True, "strata 0.2.13"), "no shape asked for: the flag is not read")

    def test_the_plain_strata_engine_is_never_asked_for_its_shape(self):
        if os.name == "nt":
            self.skipTest("a script is not a Windows program")
        self.assertEqual(shape_engine._test_start(self._program(None)), (True, "strata 0.2.13"), "an engine with no --geometry still installs as the Strata engine")
        with mock.patch.object(shape_engine, "check_geometry", side_effect=AssertionError("asked")):
            self.assertEqual(shape_engine._test_start(self._program("flash_next")), (True, "strata 0.2.13"))

    def test_a_shape_engine_with_no_answer_is_refused_at_install_with_the_owners_reason(self):
        if os.name == "nt":
            self.skipTest("a script is not a Windows program")
        self.publish("0.1.59")
        with mock.patch.object(shape_engine, "_test_start", return_value=(False, shape_engine.NO_SHAPE_ANSWER)):
            lines = []
            self.assertIsNone(self.ensure(lines, title="Q4 Lite Shape"))
        self.assertIn("The engine for Q4 Lite Shape 0.1.59 was not installed because it did not start correctly (it does not say which model shape it was built for). "
                      "The Strata engine is unchanged.", " ".join(lines))
        self.assertEqual(shape_engine.installed(SHAPE, self.comp), [])
        self.assertEqual(self.strata_before, tree(self.comp / "engine"))

    def test_the_owners_reason_for_no_answer(self):
        self.assertEqual(shape_engine.NO_SHAPE_ANSWER, "it does not say which model shape it was built for")      # owner, 11-09

    def test_a_package_built_for_another_shape_is_refused_at_install_with_the_approved_line(self):
        if os.name == "nt":
            self.skipTest("a script is not a Windows program")
        self.publish("0.1.59")
        with mock.patch.object(shape_engine, "_test_start", return_value=(False, "it was built for a different model shape (flash_next)")):
            lines = []
            self.assertIsNone(self.ensure(lines, title="Q4 Lite Shape"))
        self.assertIn("The engine for Q4 Lite Shape 0.1.59 was not installed because it did not start correctly (it was built for a different model shape (flash_next)). "
                      "The Strata engine is unchanged.", " ".join(lines))
        self.assertEqual(shape_engine.installed(SHAPE, self.comp), [])
        self.assertEqual(self.strata_before, tree(self.comp / "engine"))

    def test_min_runtime_newer_than_this_control_keeps_it_from_starting(self):
        shutil.rmtree(self.release, ignore_errors=True)
        (self.release / "shapes").mkdir(parents=True)
        folder = build(self.root / "mr", "0.1.59", self.key, min_runtime="9.9.9")
        feed_tool.pack_engine(folder, self.release / "shapes", url_prefix="shapes/")
        feed_tool.build_feed(self.release, key=self.key, sequence=1, shapes_dir=self.release / "shapes")
        self.hw = {**self.hw, "runtime": "0.1.58"}
        lines = []
        self.assertIsNone(self.ensure(lines))
        self.assertIn("The update feed has no engine for q4_lite for this computer.", " ".join(lines))     # the reason (needs control 9.9.9) is in the row, not in the line
        feed = feedmod.fetch(self.url, signing.load_trusted_keys())
        ctx = updater.Context(components=self.comp, source=self.url, trusted=signing.load_trusted_keys(), hardware=self.hw)
        self.assertIn("9.9.9", " ".join(shape_engine.update_one(ctx, feed, SHAPE)["reasons"]))


class Update(Base):
    def installed(self, *versions):
        for i, v in enumerate(versions):
            self.publish(*versions[: i + 1], sequence=i + 1)
            with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
                self.assertIsNotNone(self.ensure(), v)

    def test_a_newer_package_goes_next_to_the_old_and_the_oldest_beyond_two_goes(self):
        self.publish("0.1.59")
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            self.assertIsNotNone(self.ensure())
        self.publish("0.1.59", "0.1.60", sequence=2)
        lines = []
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            exe = self.ensure(lines)
        self.assertEqual(self.folders(), ["0.1.59", "0.1.60"], "the older engine stays")
        self.assertIn("updated from 0.1.59 to 0.1.60", " ".join(lines))
        self.assertEqual(Path(ec.expand("component:strata-q4-lite", {"backend": "cuda"})), exe)
        self.publish("0.1.59", "0.1.60", "0.1.61", sequence=3)
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")), mock.patch("strata_control.dsv41_paths.in_use", return_value=False):
            self.ensure()
        self.assertEqual(self.folders(), ["0.1.60", "0.1.61"])
        self.assertEqual(self.strata_before, tree(self.comp / "engine"))

    def test_an_older_package_in_the_feed_changes_nothing(self):
        self.installed("0.1.60")
        self.publish("0.1.59", sequence=5)
        self.assertIsNotNone(self.ensure())
        self.assertEqual(self.folders(), ["0.1.60"])

    def test_a_failed_start_of_the_newer_one_keeps_the_old(self):
        self.installed("0.1.59")
        self.publish("0.1.59", "0.1.60", sequence=9)
        lines = []
        with mock.patch.object(shape_engine, "_test_start", return_value=(False, "boom")):
            exe = self.ensure(lines)
        self.assertEqual(self.folders(), ["0.1.59"])
        self.assertTrue(exe.is_file(), "the installed engine still runs the model")
        self.assertEqual(shape_engine.newest_version(SHAPE, self.comp), "0.1.59")
        self.assertIn("A newer engine for q4_lite is available but could not be installed", " ".join(lines))
        self.assertIn("Still using 0.1.59.", " ".join(lines))
        self.assertIn("(boom). Still using 0.1.59.", " ".join(lines), "the plain reason, not the whole refusal line, is in the newer-engine line")

    def test_a_refused_acquisition_with_an_engine_installed_never_stops_a_launch(self):
        self.installed("0.1.59")
        from strata_control import acquisition
        with mock.patch.object(updater, "open_feed", side_effect=acquisition.AcquisitionRefused("no")):
            lines = []
            self.assertTrue(self.ensure(lines).is_file())
        self.assertEqual(lines, [])

    def test_an_unreachable_feed_leaves_an_installed_engine_alone_and_says_nothing(self):
        self.installed("0.1.59")
        lines = []
        exe = shape_engine.ensure(SHAPE, "http://127.0.0.1:9/strata-feed.json", components=self.comp, hardware=self.hw, emit=lines.append)
        self.assertTrue(exe.is_file())
        self.assertEqual(lines, [])
        lines = []
        self.assertIsNone(shape_engine.ensure("q4_mid", "http://127.0.0.1:9/strata-feed.json", components=self.comp, hardware=self.hw, emit=lines.append))
        self.assertTrue(any("could not be downloaded" in line for line in lines), lines)

    def test_the_explicit_update_has_a_step_only_for_an_installed_shape(self):
        self.publish("0.1.59")
        ctx = updater.Context(components=self.comp, source=self.url, trusted=signing.load_trusted_keys(), hardware=self.hw)
        feed = feedmod.fetch(self.url, ctx.trusted)
        self.assertEqual(updater._shape_steps(ctx, feed, True), [], "nothing installed: no step, an update never installs a shape engine by itself")
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            self.assertIsNotNone(self.ensure())
            self.publish("0.1.59", "0.1.60", sequence=2)
            feed = feedmod.fetch(self.url, ctx.trusted)
            steps = updater._shape_steps(ctx, feed, True)
            self.assertEqual([name for name, _ in steps], [COMP])
            self.assertEqual(updater._shape_steps(ctx, feed, False), [])
            result = steps[0][1]()
        self.assertEqual((result["status"], result["installed_before"], result["installed_after"], result["shape"]), ("updated", "0.1.59", "0.1.60", SHAPE))
        self.assertEqual(self.strata_before, tree(self.comp / "engine"))

    def test_run_update_runs_that_step_and_reports_it(self):
        self.publish("0.1.59")
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            self.assertIsNotNone(self.ensure())
        self.publish("0.1.59", "0.1.60", sequence=2)
        ctx = updater.Context(components=self.comp, source=self.url, trusted=signing.load_trusted_keys(), hardware=self.hw)
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")), \
                mock.patch.object(updater, "update_engine", return_value={"component": "engine", "status": "current", "changed": False}), \
                mock.patch.object(updater, "update_dsv41", return_value={"component": "dsv41", "status": "none-installed", "changed": False}):
            out = updater.run_update(ctx, control=False)
        row = next(r for r in out["results"] if r.get("component") == COMP)
        self.assertEqual((row["status"], row["installed_after"]), ("updated", "0.1.60"))
        self.assertEqual(out["exit_code"], 0)
        from strata_cli import update_cmd
        self.assertEqual(update_cmd._line("Strata-q4-lite", row), "Model engine  Engine for q4_lite updated from 0.1.59 to 0.1.60.")

    def test_the_components_report_lists_a_shape_engine_only_when_there_is_one(self):
        self.assertNotIn("shapes", ec.status(self.comp))
        self.publish("0.1.59")
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            self.assertIsNotNone(self.ensure())
        rows = ec.status(self.comp)["shapes"]
        self.assertEqual([(r["shape"], r["usable"], r["engine"]["shape"]) for r in rows], [(SHAPE, True, SHAPE)])
        self.assertNotIn("shape", ec.status(self.comp)["engine"], "the Strata engine's own report is what it was")


class TitleInLines(Base):
    """A person reads the model's title ("Qwen4 Lite Test"), not the geometry's name (q4_lite), in every line; the shape stays in the keys, the JSON and `strata components engine`.
    shape_engine.USE_TITLE is the one switch back."""
    TITLE = "Qwen4 Lite Test"

    def test_the_fetch_lines_name_the_title(self):
        self.publish("0.1.59")
        lines = []
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            exe = self.ensure(lines, title=self.TITLE)
        text = " ".join(lines)
        self.assertIsNotNone(exe, lines)
        self.assertIn(f"{self.TITLE} needs its own engine.", text)
        self.assertIn(f"Engine for {self.TITLE} 0.1.59", text)
        self.assertNotIn("q4_lite needs", text)
        self.assertNotIn("for q4_lite", text)

    def test_the_refusals_name_the_title(self):
        self.publish("0.1.59", shape="q4_mid")
        lines = []
        self.assertIsNone(self.ensure(lines, title=self.TITLE))
        self.assertIn(f"The update feed has no engine for {self.TITLE} for this computer.", " ".join(lines))
        with mock.patch.object(shape_engine, "_test_start", return_value=(False, "boom")):
            self.publish("0.1.59")
            lines = []
            self.assertIsNone(self.ensure(lines, title=self.TITLE))
        self.assertIn(f"The engine for {self.TITLE} 0.1.59 was not installed because it did not start correctly (boom).", " ".join(lines))

    def test_a_start_without_the_engine_names_the_title_from_the_run_config(self):
        with self.assertRaises(shape_engine.ShapeEngineMissing) as ctx:
            ec.expand("component:strata-q4-lite", {"backend": "cuda", "shape_title": self.TITLE})
        self.assertTrue(str(ctx.exception).startswith(f"{self.TITLE} needs its own engine"), str(ctx.exception))
        self.assertNotIn("q4_lite", str(ctx.exception))
        self.assertEqual(ctx.exception.shape, "q4_lite")
        with self.assertRaises(shape_engine.ShapeEngineMissing) as ctx:            # a config without a title (an older one): the shape's name
            ec.expand("component:strata-q4-lite", {"backend": "cuda"})
        self.assertTrue(str(ctx.exception).startswith("q4_lite needs its own engine"), str(ctx.exception))

    def test_the_explicit_update_finds_the_title_in_the_feeds_model(self):
        self.publish("0.1.59", models=[model_entry(title=self.TITLE)])
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            self.assertIsNotNone(self.ensure())
            self.publish("0.1.59", "0.1.60", sequence=2, models=[model_entry(title=self.TITLE)])
            ctx = updater.Context(components=self.comp, source=self.url, trusted=signing.load_trusted_keys(), hardware=self.hw)
            feed = feedmod.fetch(self.url, ctx.trusted)
            row = updater._shape_steps(ctx, feed, True)[0][1]()
        self.assertEqual(row["note"], f"Engine for {self.TITLE} updated from 0.1.59 to 0.1.60.")
        self.assertEqual((row["shape"], row["component"]), ("q4_lite", COMP), "the keys keep the geometry's name")
        self.assertEqual(shape_engine.title_from_feed(feed, "q4_lite"), self.TITLE)
        self.assertIsNone(shape_engine.title_from_feed(feed, "q4_mid"))

    def test_one_switch_puts_the_geometry_name_back(self):
        self.publish("0.1.59")
        lines = []
        with mock.patch.object(shape_engine, "USE_TITLE", False), mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            self.assertIsNotNone(self.ensure(lines, title=self.TITLE))
        self.assertIn("q4_lite needs its own engine.", " ".join(lines))
        self.assertNotIn(self.TITLE, " ".join(lines))
        self.assertEqual(shape_engine.shown("q4_lite", self.TITLE), self.TITLE)
        self.assertEqual(shape_engine.shown("q4_lite", None), "q4_lite")
        self.assertEqual(shape_engine.shown("q4_lite", "  "), "q4_lite")

    def test_the_components_report_keeps_the_geometry_name(self):
        self.publish("0.1.59")
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            self.assertIsNotNone(self.ensure(title=self.TITLE))
        self.assertEqual([r["shape"] for r in ec.status(self.comp)["shapes"]], ["q4_lite"])


class ApprovedWords(unittest.TestCase):
    """The owner approved these exact lines (11-08, "A ok"). {model} is the model's title. A change of a word is a change of this test, which the owner approves first."""

    def test_the_lines_are_the_approved_ones(self):
        W, M = shape_engine.WORDS, "Q4 Lite Shape"
        self.assertEqual(W["checking"].format(model=M), "Checking the engine for Q4 Lite Shape…")
        self.assertEqual(W["fetching"].format(model=M), "Q4 Lite Shape needs its own engine. It downloads separately, and the Strata engine stays as it is.")
        self.assertEqual(W["installed"].format(model=M, version="0.1.59", backend="cuda"), "Engine for Q4 Lite Shape 0.1.59 (cuda) is installed.")
        self.assertEqual(W["ready"], W["installed"])
        self.assertEqual(W["updated"].format(model=M, old="0.1.59", new="0.1.60"), "Engine for Q4 Lite Shape updated from 0.1.59 to 0.1.60.")
        self.assertEqual(W["not_in_feed"].format(model=M), "The update feed has no engine for Q4 Lite Shape for this computer.")
        self.assertEqual(W["start_failed"].format(model=M, version="0.1.59", why="no library"),
                         "The engine for Q4 Lite Shape 0.1.59 was not installed because it did not start correctly (no library). The Strata engine is unchanged.")
        self.assertEqual(W["fetch_failed"].format(model=M, why="no network"), "The engine for Q4 Lite Shape could not be downloaded: no network")
        self.assertEqual(W["newer_failed"].format(model=M, why="bad hash", have="0.1.59"),
                         "A newer engine for Q4 Lite Shape is available but could not be installed (bad hash). Still using 0.1.59.")
        self.assertEqual(W["missing"].format(model=M, reason="nothing matched"),
                         "Q4 Lite Shape needs its own engine, which is not installed or does not fit this PC: nothing matched. Run `strata launch` again, or Download / repair in "
                         "the app, to fetch it. Starting a model never downloads an engine.")
        self.assertEqual(W["setup_failed"].format(model=M), "the engine for Q4 Lite Shape could not be installed")
        self.assertEqual(W["setup_failed_hint"], "See the line above. Nothing was downloaded for the model. Run this again when the update feed has it.")
        self.assertEqual(W["setup_ok"].format(model=M, version="0.1.59", folder="/x/y"), "engine for Q4 Lite Shape: 0.1.59 in /x/y")
        self.assertEqual(W["update_label"], "Model engine")

    def test_no_reasons_or_platform_names_reach_the_lines(self):
        """The reasons list and the platform-backend stay in the result rows (and the JSON), not in the line a person reads."""
        pkgs = [p for p in feedmod.parse(json.dumps({"schema_version": 1, "channel": "stable", "generated_at": "2026-11-08T00:00:00Z", "sequence": 1,
                                                     "shape_engines": {"packages": [dict(shape="q4_mid", version="0.1.59", platform="linux-x86_64", backend="cuda", variant="", archs=["sm_86"],
                                                                                         min_driver={}, min_runtime=None, build_id="b", format="tar.gz", url="x", size=1, sha256="a" * 64,
                                                                                         manifest_sha256="b" * 64, installed_bytes=1)]}}).encode(), source="s", key_id="k").shape_engines]
        feed = mock.Mock(shape_engines=pkgs, models=())
        ctx = mock.Mock(components=Path(tempfile.mkdtemp()), hardware={"platform": host_platform(), "backend": "cuda"})
        row = shape_engine.update_one(ctx, feed, "q4_lite", title="Q4 Lite Shape")
        self.assertEqual(row["note"], "The update feed has no engine for Q4 Lite Shape for this computer.")
        self.assertTrue(row["reasons"], "the reasons are in the row")
        self.assertEqual((row["shape"], row["component"]), ("q4_lite", "strata-q4-lite"))


class NoAcquisitionOnStart(unittest.TestCase):
    def test_only_setup_launch_and_the_explicit_update_fetch_a_shape_engine(self):
        """Starting and serving never download: the code that fetches (shape_engine.ensure / update_one) is called from setup.py, the Terminal's launch and the updater only."""
        allowed = {"setup.py", "strata_cli/launch/runner.py", "strata_control/updater.py", "strata_control/shape_engine.py"}
        found = set()
        for folder in ("serve", "strata_cli", "strata_control"):
            for p in (ROOT / folder).rglob("*.py"):
                if p.name.startswith("test_"):
                    continue
                text = p.read_text(encoding="utf-8")
                if "shape_engine.ensure(" in text or "shape_engine.update_one(" in text or "update_one(ctx" in text:
                    found.add(p.relative_to(ROOT).as_posix())
        for p in (ROOT / "strata.py", ROOT / "setup.py"):
            if "shape_engine.ensure(" in p.read_text(encoding="utf-8"):
                found.add(p.relative_to(ROOT).as_posix())
        self.assertEqual(sorted(found - allowed), [])
        self.assertTrue(found, "the scan found nothing: the check is broken")

    def test_resolving_a_shape_engine_for_a_start_reaches_no_network_code(self):
        code = ("import sys\nsys.path.insert(0, %r)\nfrom strata_control import engine_component as ec\n"
                "try:\n    ec.expand('component:strata-q4-lite', {})\nexcept ec.EngineMissing:\n    pass\n"
                "bad = [m for m in ('strata_control.updater', 'strata_control.transfer', 'urllib.request') if m in sys.modules]\nprint(bad)\n" % str(ROOT))
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env={**os.environ, "STRATA_COMPONENTS_DIR": tempfile.mkdtemp()})
        self.assertEqual(out.stdout.strip(), "[]", out.stdout + out.stderr)


class OlderControls(Base):
    """A Control from before the member reads the same feed as one without it, and never offers the model."""

    def _old_feed_module(self, ref: str):
        shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{ref}:strata_control/feed.py"], capture_output=True)
        if shown.returncode != 0:
            self.skipTest(f"{ref} is not in this clone")
        path = self.root / f"feed_{ref.replace('.', '_')}.py"
        path.write_bytes(shown.stdout)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = mod
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.modules.pop(path.stem, None)
        return mod

    def _check(self, ref: str, has_models: bool):
        old = self._old_feed_module(ref)
        self.publish("0.1.59", models=[model_entry()])
        payload = (self.release / feedmod.FEED_NAME).read_bytes()
        new = feedmod.parse(payload, source="s", key_id="k")
        before = old.parse(json.dumps({k: v for k, v in json.loads(payload).items() if k not in ("shape_engines", "models")}).encode(), source="s", key_id="k")
        got = old.parse(payload, source="s", key_id="k")
        self.assertEqual(got.engine, before.engine, "no shape engine is ever an engine for an older Control")
        self.assertEqual(got.dsv41, before.dsv41)
        self.assertEqual((got.sequence, got.channel, got.app, got.control), (before.sequence, before.channel, before.app, before.control))
        self.assertEqual(len(new.shape_engines), 1)
        if has_models:
            (m,) = got.models
            self.assertEqual(m.family, "qwen4exp_q4_lite")
            self.assertNotIn(m.family, old.MODEL_FAMILIES_INSTALLABLE, "a family an older Control does not know is never offered to install")
        else:
            self.assertFalse(hasattr(got, "models"))

    def test_the_control_before_this_one_reads_the_feed_and_does_not_offer_the_model(self):
        self._check("3289d8280", True)

    def test_control_0211_ignores_the_new_members(self):
        self._check("v0.2.11", False)

    def test_an_older_scan_does_not_see_the_shape_engine_even_in_its_own_folder_layout(self):
        self.publish("0.1.59")
        with mock.patch.object(shape_engine, "_test_start", return_value=(True, "x")):
            self.assertIsNotNone(self.ensure())
        # an older Control scans <components>/engine only, and reads strata-engine manifests only
        found, problems = ec.scan(self.comp)
        self.assertEqual([c.version for c in found], ["0.1.58"])
        self.assertEqual(problems, [])
        shape_folder = next((self.comp / "shapes" / SHAPE / "engine").iterdir())
        with self.assertRaises(ec.ComponentProblem):
            ec.read_manifest(shape_folder)


class EveryEngineProgramAnswersTheShapeFlags(unittest.TestCase):
    """`strata --geometry` and `strata --families` are answered by every engine program before any device is touched: the CUDA/HIP/Metal
    program (src/program/generate.cpp) and the Intel one (sycl/src/program/generate.cpp, its own program). Without them a SYCL shape engine
    was installed with no geometry check (an engine that does not know the flag is accepted as one from before it)."""

    PROGRAMS = ("src/program/generate.cpp", "sycl/src/program/generate.cpp")

    def _early_block(self, rel):
        src = (ROOT / rel).read_text(encoding="utf-8")
        start = src.index('std::strcmp(argv[i], "--version") == 0')
        end = src.index("\n    }\n", start)            # the end of the early `for (int i = 1; i < argc && i < 3; ++i)` loop
        return src[start:end]

    def test_both_flags_are_in_the_early_loop_of_every_program(self):
        for rel in self.PROGRAMS:
            with self.subTest(program=rel):
                block = self._early_block(rel)
                self.assertIn('std::strcmp(argv[i], "--geometry") == 0', block)
                self.assertIn('std::printf("%s\\n", strata::core::kGeometryName);', block)
                self.assertIn('std::strcmp(argv[i], "--families") == 0', block)
                self.assertIn('std::printf("qwen4exp\\n");', block)

    def test_the_intel_program_names_no_qwen35_family(self):
        block = self._early_block("sycl/src/program/generate.cpp")
        printed = [ln for ln in block.splitlines() if "std::printf(" in ln]
        self.assertFalse([ln for ln in printed if "qwen35moe" in ln], "the Intel engine has no qwen35moe path")


if __name__ == "__main__":
    unittest.main()
