# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
"""Tests for setup.py's reproducible installs (upstream #214): Hugging Face files at pinned revisions (a gone
revision stops the download), the pinned requirements file, and an existing install left as it is.  Mocked
network - nothing is downloaded.  (XeStrata compiles its engine, so upstream's release-engine tests are left out.)

    python -m unittest tools.test_setup_pins
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from serve import model_terms  # noqa: E402

SHA = re.compile(r"/resolve/[0-9a-f]{40}/")


class Response(io.BytesIO):
    def __init__(self, body=b"", status=200):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def not_found(url):
    return urllib.error.HTTPError(url, 404, "Not Found", {}, None)


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class HuggingFacePins(unittest.TestCase):
    def test_every_model_url_is_pinned(self):
        for name, fam in setup.FAMILIES.items():
            if fam.get("plain"):   # Ornith-1.5, llm-jp-4.1: each version names its repository, pinned in HF_REVISIONS
                for version, m in fam["models"].items():
                    with self.subTest(family=name, version=version):
                        self.assertRegex(setup.hf(m["repo"]), SHA)
                continue
            if fam.get("edge0"):   # Edge0: a pack a size, each from its own repository, pinned in HF_REVISIONS
                for version, m in fam["models"].items():
                    with self.subTest(family=name, version=version):
                        self.assertRegex(setup.hf(m["repo"]), SHA)
                        # the same commit setup_edge0 fetches the files at (EDGE0_REVISIONS, EDGE0_FILES' pins)
                        self.assertIn("/resolve/" + setup.EDGE0_REVISIONS[version] + "/", setup.hf(m["repo"]))
            for key in ("hf", "mmproj_hf"):
                with self.subTest(family=name, key=key):
                    # a family with no picture reader says so: "vision" False and both mmproj keys None
                    if key == "mmproj_hf" and fam[key] is None:
                        self.assertIs(fam.get("vision"), False)
                        self.assertIsNone(fam["mmproj"])
                        continue
                    self.assertRegex(fam[key], SHA)
                    self.assertNotIn("/resolve/main/", fam[key])
        self.assertRegex(setup.HF, SHA)

    def test_unpinned(self):
        url = setup.FAMILIES["swift"]["hf"] + "x.gguf"
        self.assertEqual(setup.hf_unpinned(url),
                         "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/x.gguf")
        self.assertEqual(setup.hf_unpinned("https://example.com/a/b"), "https://example.com/a/b")

    def test_a_gone_revision_stops_the_download(self):
        """The pinned commit answers 404: setup stops with a plain message and fetches no current file."""
        seen = []

        def urlopen(req, timeout=None):
            seen.append((req.get_method(), req.full_url))
            raise not_found(req.full_url)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", urlopen), \
                mock.patch.dict(setup.os.environ, {"XESTRATA_SOURCE": "huggingface"}):
            setup.os.environ.pop("XESTRATA_ALLOW_UNPINNED", None)
            dst = Path(d) / "m.gguf"
            with self.assertRaises(SystemExit):
                _, out = quiet(setup.download, setup.FAMILIES["qwen"]["mmproj_hf"] + "m.gguf", dst)
            self.assertFalse(dst.exists())
        self.assertEqual([m for m, _ in seen], ["HEAD"])
        self.assertTrue(all("/resolve/main/" not in u for _, u in seen))

    def test_the_opt_out_downloads_the_current_file(self):
        """XESTRATA_ALLOW_UNPINNED=1 (a tester's choice): the repository's current file, announced, off by default."""
        seen = []

        def urlopen(req, timeout=None):
            seen.append((req.get_method(), req.full_url))
            if "/resolve/main/" not in req.full_url:
                raise not_found(req.full_url)
            return Response(b"model bytes", status=200)

        # the pinned Hugging Face path itself (setup's --source huggingface; auto may pick ModelScope); the terms'
        # cache is forced empty, so a user who fetched them before does not hide the fetch this test asks for
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", urlopen), \
                mock.patch.dict(setup.os.environ, {"XESTRATA_SOURCE": "huggingface", "XESTRATA_ALLOW_UNPINNED": "1"}), \
                mock.patch.object(setup, "_unpinned_said", []), mock.patch.object(setup, "_TERMS_DONE", set()), \
                mock.patch.object(model_terms, "_read_ok", lambda *a: None):
            dst = Path(d) / "m.gguf"
            _, out = quiet(setup.download, setup.FAMILIES["qwen"]["mmproj_hf"] + "m.gguf", dst)
            self.assertEqual(dst.read_bytes(), b"model bytes")
        self.assertIn("XESTRATA_ALLOW_UNPINNED=1", out)
        self.assertIn("not at the pinned revision any more", out)
        # the model's file: the pinned path, then the repository's current file
        self.assertEqual([m for m, _ in seen[:3]], ["HEAD", "HEAD", "GET"])
        self.assertTrue(all("/resolve/main/" in u for _, u in seen[1:3]))
        # then its terms' text (About > Model terms), always at its pinned revision (MODEL_TERMS)
        repo = setup.hf_repo(setup.FAMILIES["qwen"]["mmproj_hf"])
        src, rev, path, _size, _sha = setup.MODEL_TERMS[repo]["texts"][0]
        self.assertEqual(seen[3:], [("GET", setup.model_terms.file_url(src, rev, path))])

    def test_mtp_fetch_is_pinned_and_stops_when_the_revision_is_gone(self):
        import mtp_fetch
        self.assertRegex(mtp_fetch.REPO, SHA)
        self.assertIn(mtp_fetch.REVISION, mtp_fetch.REPO)
        pinned = mtp_fetch.REPO

        def urlopen(req, timeout=None):
            raise not_found(req.full_url)

        try:
            with mock.patch.object(mtp_fetch.urllib.request, "urlopen", urlopen), \
                    mock.patch.dict(mtp_fetch.os.environ, {"XESTRATA_ALLOW_UNPINNED": ""}):
                with self.assertRaises(SystemExit) as stop:
                    mtp_fetch.resolve_repo()
                self.assertIn("no longer on Hugging Face", str(stop.exception))
                self.assertEqual(mtp_fetch.REPO, pinned)
            with mock.patch.object(mtp_fetch.urllib.request, "urlopen", urlopen), \
                    mock.patch.dict(mtp_fetch.os.environ, {"XESTRATA_ALLOW_UNPINNED": "1"}), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(mtp_fetch.resolve_repo(),
                                 "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/")
            self.assertIn("XESTRATA_ALLOW_UNPINNED=1", err.getvalue())
        finally:
            mtp_fetch.REPO = pinned

    def test_hf_endpoint(self):
        # #495: HF_ENDPOINT (a mirror) serves the same pinned revision; a trailing slash and blanks are dropped
        repo = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
        with mock.patch.dict(setup.os.environ, {}, clear=False):
            setup.os.environ.pop("HF_ENDPOINT", None)
            self.assertEqual(setup.hf(repo), f"https://huggingface.co/{repo}/resolve/{setup.HF_REVISIONS[repo]}/")
            for value in ("https://hf-mirror.com", "https://hf-mirror.com/", " https://hf-mirror.com/ "):
                setup.os.environ["HF_ENDPOINT"] = value
                url = setup.hf(repo)
                self.assertEqual(url, f"https://hf-mirror.com/{repo}/resolve/{setup.HF_REVISIONS[repo]}/")
                self.assertRegex(url, SHA)
                self.assertEqual(setup.hf_unpinned(url + "x.gguf"), f"https://hf-mirror.com/{repo}/resolve/main/x.gguf")
            setup.os.environ["HF_ENDPOINT"] = ""
            self.assertEqual(setup.hf_endpoint(), "https://huggingface.co")

    def test_mtp_fetch_honours_hf_endpoint(self):
        import importlib
        import mtp_fetch
        try:
            with mock.patch.dict(mtp_fetch.os.environ, {"HF_ENDPOINT": "https://hf-mirror.com/"}):
                importlib.reload(mtp_fetch)
                self.assertTrue(mtp_fetch.REPO.startswith("https://hf-mirror.com/Qwen/Qwen3.8-Flash-Next/resolve/"))
                self.assertRegex(mtp_fetch.REPO, SHA)
        finally:
            importlib.reload(mtp_fetch)
        self.assertTrue(mtp_fetch.REPO.startswith("https://huggingface.co/"))


class Requirements(unittest.TestCase):
    def test_every_package_is_pinned(self):
        lines = setup.requirement_lines()
        names = {setup.req_name(x) for x in lines}
        for p in setup.PY_PACKAGES:
            self.assertIn(p, names)
        for x in lines:
            self.assertIn("==", x, x)
        self.assertEqual(setup.req_name(next(x for x in lines if x.startswith("numpy=="))), "numpy")
        self.assertEqual(setup.req_name("charset_normalizer==3"), "charset-normalizer")

    def pip(self, stamp, packages, legacy=None, stamp_for=None):
        ran = []
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / setup.pip_stamp_path(stamp_for or packages).name
            if stamp is not None:
                path.write_text(json.dumps(stamp))
                # written by this system (#3634's marker): without it the stamp is believed only when
                # numpy/jinja2/regex/yaml import in the Python running the tests, which made these depend on the box
                path.with_name(path.name + ".os").write_text(setup.sys.platform)
            old = Path(d) / ".strata-pip.json"
            if legacy is not None:
                old.write_text(json.dumps(legacy))
                old.with_name(old.name + ".os").write_text(setup.sys.platform)
            with mock.patch.object(setup.sys, "prefix", d), \
                    mock.patch.object(setup, "run", lambda cmd, **kw: ran.append(cmd)):
                quiet(setup.pip_install, packages, "the packages")
            after = json.loads(path.read_text()) if path.exists() else stamp
            old_left = old.exists() or old.with_name(old.name + ".os").exists()
        return [c for cmd in ran for c in cmd if "==" in c or c in setup.PY_PACKAGES], after, old_left

    def test_fresh_install_gets_every_pin(self):
        lines = setup.requirement_lines()
        ran, stamp, _ = self.pip(None, lines)
        self.assertEqual(ran, lines)
        self.assertEqual(sorted(stamp), sorted(lines))
        self.assertEqual(self.pip(stamp, lines)[0], [])                       # the second run: nothing

    def test_an_install_from_before_the_pins_gets_the_current_pins(self):
        lines = setup.requirement_lines()
        legacy = sorted(setup.PY_PACKAGES) + ["nvidia-cublas==13.0.2.14"]
        ran, stamp, old_left = self.pip(None, lines, legacy=legacy)
        self.assertEqual(ran, lines)
        self.assertEqual(sorted(stamp), sorted(lines))
        self.assertFalse(old_left)                                            # the pin-less record is dropped

    def test_a_changed_pin_is_installed(self):
        lines = setup.requirement_lines()
        old = [x.replace("tqdm==4.70.1", "tqdm==4.60.0") for x in lines]
        ran, _, _ = self.pip(old, lines, stamp_for=old)
        self.assertEqual(ran, lines)                                          # the old record names another pin set
        self.assertIn("tqdm==4.70.1", ran)
        self.assertEqual(self.pip(lines, lines)[0], [])                       # the new record: nothing


if __name__ == "__main__":
    unittest.main()
