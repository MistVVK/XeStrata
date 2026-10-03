# SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for setup.py's reproducible installs (upstream #214): Hugging Face files at pinned revisions (the current
files when a revision is gone), the pinned requirements file, and an existing install left as it is.  Mocked
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
            for key in ("hf", "mmproj_hf"):
                with self.subTest(family=name, key=key):
                    self.assertRegex(fam[key], SHA)
                    self.assertNotIn("/resolve/main/", fam[key])
        self.assertRegex(setup.HF, SHA)

    def test_unpinned(self):
        url = setup.FAMILIES["swift"]["hf"] + "x.gguf"
        self.assertEqual(setup.hf_unpinned(url),
                         "https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/main/x.gguf")
        self.assertEqual(setup.hf_unpinned("https://example.com/a/b"), "https://example.com/a/b")

    def test_a_gone_revision_downloads_the_current_file(self):
        seen = []

        def urlopen(req, timeout=None):
            seen.append((req.get_method(), req.full_url))
            if "/resolve/main/" not in req.full_url:
                raise not_found(req.full_url)
            return Response(b"model bytes", status=200)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", urlopen):
            dst = Path(d) / "m.gguf"
            _, out = quiet(setup.download, setup.FAMILIES["qwen"]["mmproj_hf"] + "m.gguf", dst)
            self.assertEqual(dst.read_bytes(), b"model bytes")
        self.assertIn("not at the pinned revision any more", out)
        self.assertEqual([m for m, _ in seen], ["HEAD", "HEAD", "GET"])
        self.assertTrue(all("/resolve/main/" in u for _, u in seen[1:]))

    def test_mtp_fetch_is_pinned_and_falls_back(self):
        import mtp_fetch
        self.assertRegex(mtp_fetch.REPO, SHA)
        self.assertIn(mtp_fetch.REVISION, mtp_fetch.REPO)
        pinned = mtp_fetch.REPO

        def urlopen(req, timeout=None):
            raise not_found(req.full_url)

        try:
            with mock.patch.object(mtp_fetch.urllib.request, "urlopen", urlopen), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(mtp_fetch.resolve_repo(), "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/")
            self.assertIn("pinned revision", err.getvalue())
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
        self.assertEqual(setup.req_name('numpy==2.5.3; python_version >= "3.12"'), "numpy")
        self.assertEqual(setup.req_name("charset_normalizer==3"), "charset-normalizer")

    def pip(self, stamp, packages, installed=()):
        ran = []
        with tempfile.TemporaryDirectory() as d:
            if stamp is not None:
                (Path(d) / ".strata-pip.json").write_text(json.dumps(stamp))
            with mock.patch.object(setup.sys, "prefix", d), \
                    mock.patch.object(setup, "run", lambda cmd, **kw: ran.append(cmd)), \
                    mock.patch.object(setup, "_installed", lambda name: name in installed):
                quiet(setup.pip_install, packages, "the packages")
            after = json.loads((Path(d) / ".strata-pip.json").read_text()) if ran else stamp
        return [c for cmd in ran for c in cmd if "==" in c or c in setup.PY_PACKAGES], after

    def test_fresh_install_gets_every_pin(self):
        lines = setup.requirement_lines()
        ran, stamp = self.pip(None, lines)
        self.assertEqual(ran, lines)
        self.assertEqual(sorted(stamp), sorted(lines))
        self.assertEqual(self.pip(stamp, lines)[0], [])                       # the second run: nothing

    def test_an_install_from_before_the_pins_is_left_alone(self):
        lines = setup.requirement_lines()
        legacy = sorted(setup.PY_PACKAGES) + ["nvidia-cublas==13.0.2.14"]
        deps = {"markupsafe", "certifi", "charset-normalizer", "idna", "urllib3", "colorama"}
        self.assertEqual(self.pip(legacy, lines, installed=deps)[0], [])
        missing = [p for p in legacy if p != "psutil"]                         # an older list without psutil
        ran, _ = self.pip(missing, lines, installed=deps)
        self.assertEqual(ran, ["psutil==7.2.2"])

    def test_a_changed_pin_is_installed(self):
        lines = setup.requirement_lines()
        old = [x.replace("tqdm==4.70.1", "tqdm==4.60.0") for x in lines]
        ran, _ = self.pip(old, lines)
        self.assertEqual(ran, ["tqdm==4.70.1"])


if __name__ == "__main__":
    unittest.main()
