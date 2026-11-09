# SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for setup.py's translation model (--translate, --translate-gpu): auto's choice by the RAM the text and image
models leave (the 2B, else the 0.8B, else none), the sizes asked for, no, the earlier choice kept when none is given,
and the GPU's forms (auto, same, cpu, a PCI address).  Nothing is downloaded and no GPU is used.

    python -m unittest tools.test_translate_setup
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

PCI = "0000:03:00.0"


class TranslateSetup(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.cfg_path = self.dir / "xestrata-x.json"
        self.fetched: list = []
        for p in (mock.patch.object(setup, "download", lambda url, dst, what=None: self.fetched.append(url)),
                  mock.patch.object(setup, "verify_sha256", lambda *a, **k: None),
                  mock.patch.object(setup, "usable_ram_gb", lambda a=None: 32.0),
                  mock.patch.object(setup, "gpus", lambda: [{"pci": PCI, "vram_gb": 16}]),
                  mock.patch.object(setup, "gpu_problem", lambda x: None)):
            p.start()
            self.addCleanup(p.stop)

    def run_setup(self, translate, text_gb=10.0, gpu=None, cfg=None, on_cpu=False) -> dict:
        a = types.SimpleNamespace(translate=translate, translate_gpu=gpu, yes=True, models_dir=str(self.dir))
        cfg = {} if cfg is None else cfg
        with contextlib.redirect_stdout(io.StringIO()):
            setup.setup_translate(a, cfg, self.cfg_path, text_gb, on_cpu)
        return cfg

    def test_auto_lowers_by_the_ram(self):
        self.assertEqual(self.run_setup("auto", 20.0)["translate"]["model"], "plamo-3-translate-2b")
        got = self.run_setup("auto", 28.0)["translate"]
        self.assertEqual((got["model"], got["lowered"]), ("plamo-3-translate-0.8b", "2b"))
        self.assertIsNone(self.run_setup("auto", 31.0)["translate"])
        # the image models' RAM (one at a time: the largest; ~20 GB here) counts as the text model's does
        images = {"images": {"model_name": "qwen-image-2.1-turbo", "quant": "Q8_0"}}
        self.assertEqual(self.run_setup("auto", 8.0)["translate"]["model"], "plamo-3-translate-2b")
        self.assertEqual(self.run_setup("auto", 8.0, cfg=dict(images))["translate"]["model"],
                         "plamo-3-translate-0.8b")
        self.cfg_path.write_text(json.dumps(images))                           # the earlier ones, carried over
        self.assertEqual(self.run_setup("auto", 8.0)["translate"]["model"], "plamo-3-translate-0.8b")

    def test_sizes_and_no(self):
        got = self.run_setup("8b", 31.0)["translate"]
        self.assertEqual((got["model"], got["choice"]), ("plamo-3-translate-8b", "8b"))
        self.assertIn("/pfnet/plamo-3-translate-8b-GGUF/resolve/", self.fetched[-1])
        self.assertIsNone(self.run_setup("no")["translate"])

    def test_not_given(self):
        self.assertEqual(self.run_setup(None)["translate"]["choice"], "auto")    # set up for the first time
        self.cfg_path.write_text(json.dumps({"translate": None}))
        self.assertNotIn("translate", self.run_setup(None))                   # carried over (carry_over)
        old = {"model": "plamo-3-translate-2b", "path": "/m.gguf", "gpu": "auto", "choice": "2b"}
        self.cfg_path.write_text(json.dumps({"translate": old}))
        self.assertNotIn("translate", self.run_setup(None))
        self.assertEqual(self.run_setup(None, gpu="cpu")["translate"], {**old, "gpu": "cpu"})

    def test_gpu_forms(self):
        for want in ("auto", "same", "cpu"):
            self.assertEqual(self.run_setup("2b", gpu=want)["translate"]["gpu"], want)
        self.assertEqual(self.run_setup("2b", gpu=PCI.upper())["translate"]["gpu"], PCI)
        self.assertEqual(self.run_setup("2b", gpu=PCI, on_cpu=True)["translate"]["gpu"], "cpu")
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.run_setup("2b", gpu="0000:09:00.0")


if __name__ == "__main__":
    unittest.main()
