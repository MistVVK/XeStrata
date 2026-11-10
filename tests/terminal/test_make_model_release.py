"""packaging/make_model_release.py: the model-release plan is a dry run, case A leaves the engines alone, case B ships the engine of the new shape as its own
component (the Strata engine, the Controls and the app are not rebuilt), and the steps that need the release key only print a placeholder.

    python -m unittest tests.terminal.test_make_model_release
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "packaging"), str(ROOT / "tools"), str(ROOT)]
import make_model_release as M  # noqa: E402
import test_qwen4_intake as T  # noqa: E402
import qwen4_intake as Q  # noqa: E402

PREV = {"strata-engine-0.1.57-linux-x86_64-cuda-cuda13.tar.gz": 720_000_000, "strata-engine-0.1.57-linux-x86_64-sycl.tar.gz": 187_000_000,
        "strata-engine-0.1.57-windows-x86_64-cuda-cuda13.zip": 900_000_000, "strata-engine-0.1.57-macos-arm64-metal.tar.gz": 150_000_000,
        "strata-dsv41-0.1.57-linux-x86_64-cuda-cuda13.tar.gz": 445_000_000, "strata-control-0.1.57-linux-x86_64.tar.gz": 99_000_000,
        "Strata-0.2.11-windows-x64-setup.exe": 80_000_000, "Strata-0.2.11-windows-x64-setup.exe.sig": 400,
        "Strata-0.2.11-linux-x86_64.AppImage": 90_000_000, "Strata-0.2.11-macos-arm64.dmg": 160_000_000, "Strata.app.tar.gz": 70_000_000,
        "strata-feed.json": 9000, "strata-feed.json.sig": 200, "SHA256SUMS.txt": 5000, "latest.json": 1500, "install.sh": 9000}


def make_prev(d: Path) -> Path:
    d.mkdir()
    for n, size in PREV.items():
        with (d / n).open("wb") as f:
            f.truncate(size)
    return d


def report(name: str) -> dict:
    return Q.run([Q.read_local(p) for p in T.files_for(name)], build_tokenizer=False)


def run(argv) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = M.main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            if isinstance(e.code, str):
                err.write(e.code)
    return code, out.getvalue(), err.getvalue()


class Plan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.prev = make_prev(Path(cls.tmp.name) / "prev")
        cls.a = Path(cls.tmp.name) / "a.json"
        cls.a.write_text(json.dumps(report("q36_ud_q4_k_xl")))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def base(self, rep, *extra):
        return ["--intake", str(rep), "--id", "m1", "--app-version", "0.2.12", "--sequence", "122", "--prev-upload", str(self.prev), *extra]

    def test_case_a_is_feed_and_app_only(self):
        code, out, err = run(self.base(self.a, "--json", "--revision", "abc", "--hf-repo", "x/y"))
        self.assertEqual(code, 0, err)
        p = json.loads(out)
        self.assertEqual(p["case"], "A")
        names = {c["name"] for c in p["changed"]}
        self.assertFalse(any(n.startswith(("strata-engine-", "strata-dsv41-", "strata-control-")) for n in names))
        self.assertTrue({"strata-feed.json", "strata-feed.json.sig", "SHA256SUMS.txt"} <= names)
        # G12 of the Q4-P8 rehearsal: a feed model needs no app rebuild (the app lists it from the feed), so case A plans no app and no Mac step
        self.assertFalse(names & {"latest.json", "Strata-0.2.12-windows-x64-setup.exe"})
        self.assertFalse(any("App text" in s["what"] or "Mac" in s["what"] for s in p["steps"]))
        kept = {k["name"] for k in p["kept"]}
        self.assertTrue({n for n in PREV if n.startswith(("strata-engine-", "strata-dsv41-", "strata-control-", "Strata-"))} <= kept)
        self.assertFalse(any("Build the Qwen engine" in s["what"] for s in p["steps"]))
        self.assertEqual(p["entry"]["family"], "qwen35moe")
        self.assertEqual(p["entry"]["hf"], {"repo": "x/y", "revision": "abc"})
        self.assertIs(p["entry"]["pictures"], False)               # no picture reader next to this set

    def test_case_a_with_app_plans_the_app_steps_when_asked(self):
        code, out, err = run(self.base(self.a, "--json", "--with-app"))
        self.assertEqual(code, 0, err)
        p = json.loads(out)
        self.assertTrue({"latest.json", "Strata-0.2.12-windows-x64-setup.exe"} <= {c["name"] for c in p["changed"]})
        self.assertTrue(any("App text" in s["what"] for s in p["steps"]))

    def test_the_plan_names_the_boxes_of_the_rehearsal(self):
        code, out, err = run(self.base(self.a, "--case", "B", "--engine-version", "0.1.58", "--shape", "q4_lite"))
        self.assertEqual(code, 0, err)
        for good in ("l620", "p1006", "owner PC", "m1002"):
            self.assertIn(good, out)
        for stale in ("z440", "p1004", "m1001"):
            self.assertNotIn(stale, out)
        self.assertIn("STRATA_GEOMETRY_FILE", out)
        self.assertIn("--from-gguf", out)

    def test_case_b_ships_the_shape_engine_as_its_own_component_and_rebuilds_nothing_else(self):
        code, out, err = run(self.base(self.a, "--json", "--case", "B", "--engine-version", "0.1.58", "--shape", "q4_lite"))
        self.assertEqual(code, 0, err)
        p = json.loads(out)
        names = {c["name"] for c in p["changed"]}
        self.assertIn("strata-q4-lite-0.1.58-linux-x86_64-sycl.tar.gz", names)
        self.assertIn("strata-q4-lite-0.1.58-windows-x86_64-cuda-cuda13.zip", names)
        self.assertFalse(any(n.startswith(("strata-engine-", "strata-control-", "strata-dsv41-")) for n in names), "the Strata engine and the Controls are not rebuilt")
        self.assertFalse(any("0.1.57" in n for n in names))
        kept = {k["name"] for k in p["kept"]}
        self.assertTrue({n for n in PREV if n.startswith(("strata-engine-", "strata-dsv41-", "strata-control-", "Strata-"))} <= kept)
        self.assertFalse(any("Controls: pack" in s["what"] or "App text" in s["what"] for s in p["steps"]))
        self.assertIn("strata-q4-lite-0.1.58-macos-arm64-metal.tar.gz", names)
        self.assertTrue(any("Mac" in s["what"] and s["owner_only"] for s in p["steps"]), "the owner signs the Mac shape engine")
        self.assertTrue(any("--shape q4_lite" in c for s in p["steps"] for c in s["commands"]))
        self.assertTrue(any("--shapes <shape-entries>" in c for s in p["steps"] for c in s["commands"]))
        self.assertEqual(p["entry"]["family"], "qwen35moe", "the family changes only for a qwen4exp model")
        for missing in ({"--shape": None}, {"--engine-version": None}):
            args = ["--case", "B", "--engine-version", "0.1.58", "--shape", "q4_lite"]
            for k in missing:
                i = args.index(k)
                del args[i:i + 2]
            code, _, err = run(self.base(self.a, *args))
            self.assertNotEqual(code, 0, missing)                    # case B needs an engine version and a shape
        code, _, err = run(self.base(self.a, "--case", "B", "--engine-version", "0.1.58", "--shape", "Not A Shape"))
        self.assertNotEqual(code, 0)
        code, _, err = run(self.base(self.a, "--shape", "q4_lite"))
        self.assertNotEqual(code, 0)                                # --shape is for case B

    def test_a_case_b_entry_names_the_family_of_its_shape(self):
        from strata_control import feed as feedmod
        rep = json.loads(self.a.read_text())
        rep["sets"][0]["architecture"] = "qwen4exp"
        b = Path(self.tmp.name) / "b.json"
        b.write_text(json.dumps(rep))
        out = Path(self.tmp.name) / "outb"
        code, _, err = run(self.base(b, "--case", "B", "--engine-version", "0.1.59", "--shape", "q4_lite", "--write", str(out), "--revision", "a" * 40, "--hf-repo", "x/y"))
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads((out / "models" / "m1.entry.json").read_text())["family"], "qwen4exp_q4_lite")
        self.assertEqual(json.loads((out / "models" / "m1.model.json").read_text())["family"], "qwen4exp_q4_lite")
        self.assertEqual(feedmod.shape_engine.shape_of_family("qwen4exp_q4_lite"), "q4_lite")

    def test_case_c_is_refused(self):
        p = Path(self.tmp.name) / "c.json"
        p.write_text(json.dumps(report("ds41_ud_q8_k_xl")))
        code, out, err = run(self.base(p))
        self.assertNotEqual(code, 0)
        self.assertIn("case C", err)
        self.assertEqual(out, "")

    def test_it_is_a_dry_run_and_never_touches_a_key(self):
        code, out, err = run(self.base(self.a, "--execute"))
        self.assertEqual(code, 2)
        self.assertIn("REFUSED", out)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            M.main(self.base(self.a, "--key", "k.json"))             # there is no such option
        code, out, err = run(self.base(self.a))
        self.assertEqual(code, 0, err)
        key_lines = [l for l in out.splitlines() if "--key" in l or "sign-package" in l]
        self.assertTrue(key_lines)
        for l in key_lines:
            self.assertIn("<FEED-KEY>", l)
        src = Path(M.__file__).read_text(encoding="utf-8")
        for bad in ("subprocess", "os.system", "urllib", "requests", "socket", "shutil.copy"):
            self.assertNotIn(bad, src)

    def test_write_makes_the_entry_and_fills_the_templates(self):
        t = Path(self.tmp.name) / "tpl"
        t.mkdir()
        (t / "RELEASE-NOTES.md").write_text("# {{model_title}} {{app_version}}\n{{file_rows}}\n[FILL: a number] {{unknown}}\n", encoding="utf-8")
        out = Path(self.tmp.name) / "out"
        code, _, err = run(self.base(self.a, "--write", str(out), "--templates", str(t), "--revision", "abc"))
        self.assertNotEqual(code, 0, "case A with --write needs --min-engine (Q4-READY: it left a [FILL] in the entry)")
        self.assertIn("--min-engine", err)
        self.assertFalse(out.exists())
        code, _, err = run(self.base(self.a, "--write", str(out), "--templates", str(t), "--revision", "abc", "--min-engine", "0.1.58"))
        self.assertEqual(code, 0, err)
        entry = json.loads((out / "models" / "m1.model.json").read_text())
        self.assertIn("DRAFT", entry["_status"])
        self.assertEqual(entry["files"][0]["name"], "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")
        self.assertEqual(entry["drafter"], "mtp")
        self.assertEqual(entry["min_engine"], "0.1.58")
        notes = (out / "RELEASE-NOTES.md").read_text()
        self.assertIn("# Qwen3.6-35B-A3B 0.2.12", notes)
        self.assertIn("[FILL: a number]", notes)
        self.assertIn("{{unknown}}", notes)                              # a key it does not know stays visible
        self.assertTrue((out / "PLAN.md").is_file() and (out / "plan.json").is_file())
        code, _, err = run(self.base(self.a, "--write", str(out)))      # not an empty folder
        self.assertNotEqual(code, 0)

    def test_write_makes_an_entry_in_the_feeds_schema_and_build_feed_refuses_what_is_unfilled(self):
        import feed_tool
        from strata_control import feed as feedmod
        out = Path(self.tmp.name) / "outfeed"
        rev = "a" * 40
        code, _, err = run(self.base(self.a, "--write", str(out), "--revision", rev, "--hf-repo", "x/y", "--default-context", "131072", "--min-engine", "0.1.58"))
        self.assertEqual(code, 0, err)
        path = out / "models" / "m1.entry.json"
        entry = json.loads(path.read_text())
        self.assertIn("[FILL", json.dumps(entry))                         # the values the report cannot know are left to a person
        with self.assertRaises(SystemExit) as cm:
            feed_tool.build_feed(Path(self.tmp.name) / "rel0", key={}, sequence=1, models_dir=out / "models")
        self.assertIn("FILL", str(cm.exception))
        out2 = Path(self.tmp.name) / "outfeed2"
        code, _, err = run(self.base(self.a, "--write", str(out2), "--revision", rev, "--hf-repo", "x/y", "--default-context", "131072", "--about", "one sentence",
                                     "--licence", "Apache 2.0", "--licence-url", "https://example.org/licence", "--quant", "Q4", "--quant-about", "4-bit", "--ram-gb", "24", "--min-engine", "0.1.58",
                                     "--title", "Qwen4 Test"))
        self.assertEqual(code, 0, err)
        filled = json.loads((out2 / "models" / "m1.entry.json").read_text())
        self.assertEqual(filled["title"], "Qwen4 Test")                   # --title (Q4-READY); without it the file's general.name, as before
        self.assertEqual(json.loads((out / "models" / "m1.entry.json").read_text())["title"], "Qwen3.6-35B-A3B")
        self.assertNotIn("[FILL", json.dumps(filled).replace("[FILL: sha256]", ""))
        for f in filled["quants"][0]["files"]:
            f["sha256"] = f["sha256"] if len(f["sha256"]) == 64 else "0" * 64
        feedmod._model_entry(filled)                                     # strata_control/feed.py keeps it

    def test_the_plan_prints_the_launch_line_with_the_size_name_a_person_types(self):
        """Q4-READY: a feed size whose quant name is a built-in size is <quant>-<id> (strata_control/feed_models.py); the rehearsal typed the plain name twice."""
        code, out, err = run(self.base(self.a, "--json", "--quant", "UD-Q4_K_XL"))
        self.assertEqual(code, 0, err)
        launch = json.loads(out)["launch"]
        self.assertIn("strata launch --family m1 --model ud-q4_k_xl-m1", launch)
        self.assertTrue(any(l.startswith("strata launch --family m1 --model ud-q4_k_xl-m1 --model-path ") for l in launch))
        code, out, err = run(self.base(self.a, "--json", "--quant", "Q4REH"))
        self.assertEqual(code, 0, err)
        self.assertIn("strata launch --family m1 --model q4reh", json.loads(out)["launch"])
        code, out, err = run(self.base(self.a, "--quant", "Q4REH"))
        self.assertIn("## How a person starts it", out)
        self.assertIn("    strata launch --family m1 --model q4reh", out)
        with mock.patch.object(M, "builtin_sizes", return_value=None):
            code, out, err = run(self.base(self.a, "--json", "--quant", "Q4REH"))
        launch = json.loads(out)["launch"]
        self.assertIn("strata launch --family m1 --model q4reh", launch)
        self.assertIn("strata launch --family m1 --model q4reh-m1", launch)

    def test_the_repo_templates_have_what_the_script_fills(self):
        """The templates live in the repo (packaging/templates/model-release, the default of --templates), so this runs on every box."""
        d = M.TEMPLATES
        self.assertEqual(d, Path(M.__file__).resolve().parents[1] / "packaging" / "templates" / "model-release")
        self.assertTrue((d / "RELEASE-NOTES.md").is_file() and (d / "MODEL-CARD.md").is_file())
        self.assertFalse((d / "CHECKLIST.md").exists(), "the release checklist is an internal work list and is not in the repo")
        out = Path(self.tmp.name) / "out2"
        code, _, err = run(self.base(self.a, "--write", str(out), "--revision", "abc", "--hf-repo", "x/y"))
        self.assertNotEqual(code, 0, "case A with --write needs --min-engine (078b52563)")
        self.assertIn("case A: give --min-engine", err)
        self.assertFalse(out.exists())
        code, _, err = run(self.base(self.a, "--write", str(out), "--revision", "abc", "--hf-repo", "x/y", "--min-engine", "0.1.58"))
        self.assertEqual(code, 0, err)
        self.assertIn("no template CHECKLIST.md", err)
        for name in ("RELEASE-NOTES.md", "MODEL-CARD.md"):
            text = (out / name).read_text(encoding="utf-8")
            self.assertRegex(text.splitlines()[0], "DRAFT")                 # drafts for the owner's approval
            self.assertNotIn("{{", text, name)                          # every key the templates use is one the script fills
        notes = (out / "RELEASE-NOTES.md").read_text(encoding="utf-8")
        self.assertTrue(notes.rstrip().endswith('height="50"></a>'))      # the coffee button is last
        self.assertIn("[FILL", notes)
        self.assertFalse((out / "CHECKLIST.md").exists())

    def test_the_private_checklist_by_path_beside_the_repo_templates(self):
        """--checklist names the internal checklist (outside the repo) and the notes and the model card still come from the repo's default folder."""
        priv = Path(self.tmp.name) / "private"
        priv.mkdir()
        (priv / "my-list.md").write_text("<!-- WORKING DOCUMENT -->\n# Runbook {{model_title}} case {{case}}\n", encoding="utf-8")
        out = Path(self.tmp.name) / "out-checklist"
        code, _, err = run(self.base(self.a, "--write", str(out), "--checklist", str(priv / "my-list.md"), "--revision", "abc", "--min-engine", "0.1.58"))
        self.assertEqual(code, 0, err)
        self.assertEqual((out / "CHECKLIST.md").read_text(encoding="utf-8"), "<!-- WORKING DOCUMENT -->\n# Runbook Qwen3.6-35B-A3B case A\n")
        self.assertEqual((out / "RELEASE-NOTES.md").read_text(encoding="utf-8").splitlines()[0],
                         (M.TEMPLATES / "RELEASE-NOTES.md").read_text(encoding="utf-8").splitlines()[0])      # the repo's notes
        self.assertTrue((out / "MODEL-CARD.md").is_file())
        self.assertNotIn("no template CHECKLIST.md", err)
        code, _, err = run(self.base(self.a, "--write", str(Path(self.tmp.name) / "out-missing"), "--checklist", str(priv / "nope.md"), "--min-engine", "0.1.58"))
        self.assertNotEqual(code, 0)
        self.assertIn("--checklist: no file", err)

    def test_another_templates_folder_is_still_read(self):
        t = Path(self.tmp.name) / "tpl-other"
        t.mkdir()
        for name in M.TEMPLATE_NAMES:
            (t / name).write_text("# %s {{model_title}}\n" % name, encoding="utf-8")
        out = Path(self.tmp.name) / "out-other"
        code, _, err = run(self.base(self.a, "--write", str(out), "--templates", str(t), "--revision", "abc", "--min-engine", "0.1.58"))
        self.assertEqual(code, 0, err)
        for name in M.TEMPLATE_NAMES:
            self.assertEqual((out / name).read_text(encoding="utf-8"), "# %s Qwen3.6-35B-A3B\n" % name)


if __name__ == "__main__":
    unittest.main()
